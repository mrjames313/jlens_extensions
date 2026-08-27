"""What made a lens -- recorded beside it, because our runs produce no ``config.yaml``.

Neuronpedia's ``run-all-fit-lens.py`` writes the ``config.yaml`` that sits next to
every published artifact. We deliberately did not vendor that script (see
``PROVENANCE.md``), so nothing in our fit path records what produced a tensor. This
module is the replacement, and it is deliberately *more* than theirs carries: the
published ``config.yaml`` records no library versions at all, which is precisely why
T3 could only bound Neuronpedia's environment rather than know it.

The sidecar answers, for a lens on disk:

* **Which code.** ``jlens_extensions`` commit, whether the tree was dirty, and the
  hash of ``uv.lock``.
* **Which libraries.** torch, transformers, datasets, numpy, and the interpreter.
* **Which corpus.** The dataset coordinates, the number of prompts actually loaded,
  and hashes of the first and last prompt plus the whole corpus. The comparison risk
  this addresses is a dataset revision shifting the stream under a deterministic
  loader -- which would be invisible in every other field.
* **Which box, and which numbers came off it.** ``JLENS_MACHINE`` plus the machine
  profile entry the fit was parameterised from, copied in rather than referenced, so
  the record survives a later profile edit.
* **What came out.** Path, size, storage dtype and hash of the lens and its
  convergence trace.

**A dirty tree is recorded, not rejected.** Refusing to run would be the wrong
trade for a measurement driver -- but an unrecorded dirty tree makes a later
disagreement undiagnosable, which is the failure this file exists to prevent.

Re-derivability has a known limit, stated here so the sidecar is not over-read: the
fit is **not bitwise reproducible** even with every field below pinned. See
``d-2026-08-24-fit-nondeterminism-envelope``. These fields pin the *inputs*; the
output tensor still carries a run-to-run envelope.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "SIDECAR_SCHEMA_VERSION",
    "ArtifactFacts",
    "sha256_file",
    "sha256_text",
    "corpus_fingerprint",
    "git_facts",
    "library_versions",
    "artifact_facts",
    "build_sidecar",
    "write_sidecar",
]

SIDECAR_SCHEMA_VERSION = 1

_HASH_CHUNK = 1 << 20


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_fingerprint(prompts: list[str]) -> dict[str, Any]:
    """Identify a prompt list without storing it.

    First and last are hashed individually because they localise a shift: a stream
    that changed at the head and one that changed at the tail are different
    problems. The joined hash catches a change anywhere in between, which the two
    endpoints alone would miss.
    """
    if not prompts:
        return {"n_prompts": 0, "first_sha256": None, "last_sha256": None, "corpus_sha256": None}
    return {
        "n_prompts": len(prompts),
        "first_sha256": sha256_text(prompts[0]),
        "last_sha256": sha256_text(prompts[-1]),
        "corpus_sha256": sha256_text("\x00".join(prompts)),
        "first_chars": len(prompts[0]),
        "last_chars": len(prompts[-1]),
    }


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_facts(repo: Path | str) -> dict[str, Any]:
    """Commit, branch and cleanliness of the code that ran."""
    repo = Path(repo)
    status = _git(repo, "status", "--porcelain")
    lock = repo / "uv.lock"
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "dirty_paths": sorted(line[3:] for line in status.splitlines())[:20] if status else [],
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
    }


def library_versions() -> dict[str, str | None]:
    """Versions of everything that can move a tensor.

    ``importlib.metadata`` rather than ``__version__`` for consistency, except
    ``torch``, where the two genuinely differ: metadata reports ``2.13.0`` while
    ``torch.__version__`` carries the local segment ``+cu130``. The CUDA build is
    the part that matters here, so torch is read from the module and the rest from
    metadata.
    """
    import platform
    from importlib import metadata

    versions: dict[str, str | None] = {"python": platform.python_version()}
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001 - a missing torch is worth recording, not raising
        versions["torch"] = None
        versions["cuda"] = None
    for name in ("transformers", "datasets", "numpy", "accelerate", "tokenizers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@dataclass(frozen=True)
class ArtifactFacts:
    """One output file: where it is, how big, and what it hashes to."""

    path: str
    bytes: int
    sha256: str

    @classmethod
    def of(cls, path: Path | str) -> ArtifactFacts:
        path = Path(path)
        return cls(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path))


def artifact_facts(path: Path | str) -> dict[str, Any]:
    facts = ArtifactFacts.of(path)
    return {"path": facts.path, "bytes": facts.bytes, "sha256": facts.sha256}


def build_sidecar(
    *,
    task: str,
    run: str,
    machine: str,
    model_id: str,
    command: list[str],
    fit_config: dict[str, Any],
    profile_path: Path | str,
    profile_entry: dict[str, Any],
    corpus: dict[str, Any],
    repo: Path | str,
    artifacts: dict[str, Any],
    results: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    """Assemble the record. Pure -- every input is passed in, nothing is discovered."""
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "task": task,
        "run": run,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": machine,
        "model": model_id,
        # The command as one line, so it can be copied out of the record and re-run.
        "command": " ".join(command),
        "command_argv": command,
        "fit_config": fit_config,
        "profile": {"path": str(profile_path), "entry": profile_entry},
        "corpus": corpus,
        "code": git_facts(repo),
        "libraries": library_versions(),
        "artifacts": artifacts,
        "results": results,
        "reproducibility": (
            "Pins inputs, not the output tensor. Two runs of this exact configuration "
            "differ by a per-layer envelope -- see d-2026-08-24-fit-nondeterminism-envelope."
        ),
        "notes": notes,
    }


def write_sidecar(path: Path | str, sidecar: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar, indent=2, sort_keys=False) + "\n")
    return path
