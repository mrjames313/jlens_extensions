"""Fetching published reference lenses from the HuggingFace Hub.

We deliberately do not port upstream's ``JacobianLens.from_pretrained`` /
``snapshot_download``. The vendored ``load()`` already takes a path, so fetching
belongs in our own driver where we own it, rather than as added surface on the fork.

**Plain ``urllib``, not ``huggingface_hub``.** The artifact repo is public so no auth
is needed, the largest file is 48 MB so chunked resume buys nothing, and it keeps our
dependency list identical to Neuronpedia's -- which ``PROVENANCE.md`` asserts is
verbatim, and which an unrecorded addition would quietly falsify. Integrity comes
from checking sha256 against a value recorded *independently* of the download (the
LFS oid the Hub publishes, captured in T1), which is a stronger check than trusting
a downloader to have got it right.

**This module must not import ``jlens``.** ``harness/`` is deliberately not packaged,
so the library is importable only by a script sitting next to it. Confirming that a
downloaded tensor opens with ``JacobianLens.load`` is therefore a separate step run
with ``harness/`` on the path -- see the module docstring of ``tests/test_fetch.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from jlens_extensions.config import Config, ConfigError, load

HUB = "https://huggingface.co"

#: The published lenses live in an HF **model** repo, not a dataset repo, so the
#: download form is ``/resolve/<revision>/<path>`` with no ``datasets/`` segment.
#: Established in T1 by reading the repo tree; getting this wrong 404s.
ARTIFACT_REPO = "neuronpedia/jacobian-lens"

_CHUNK = 1 << 20


class FetchError(RuntimeError):
    """A download failed, or arrived with contents we did not expect."""


@dataclass(frozen=True)
class RemoteFile:
    """One published file, with whatever we independently know about it.

    ``sha256`` is populated only where the Hub publishes an LFS oid (the oid *is*
    the sha256 of the contents). Small non-LFS files are stored inline and have no
    oid, so only their size is known ahead of time and their hash is recorded on
    arrival rather than verified.
    """

    name: str
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class PublishedLens:
    model: str  # Neuronpedia model id
    hf_model: str  # the HuggingFace id of the model it was fitted on
    path: str  # repo-relative directory
    files: tuple[RemoteFile, ...]


QWEN35_08B = PublishedLens(
    model="qwen3.5-0.8b",
    hf_model="Qwen/Qwen3.5-0.8B",
    path="qwen3.5-0.8b/jlens/Salesforce-wikitext",
    files=(
        RemoteFile(
            "Qwen3.5-0.8B_jacobian_lens.pt",
            size=48_242_373,
            sha256="aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48",
        ),
        RemoteFile("Qwen3.5-0.8B_convergence.csv", size=9_884),
        RemoteFile("config.yaml", size=2_526),
    ),
)

#: Sizes and LFS oids read from the Hub tree API on 2026-08-30, independently of any
#: download, as the module docstring requires.
#:
#: **This repo publishes two 4B lenses and only one 0.8B one.** Alongside the file
#: below there is a ``Qwen3.5-4B_jacobian_lens_n1000.pt`` (406,332,644 bytes, sha256
#: ``1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e``). Both fits
#: were launched with ``--n_prompts 1000 --stop_at_delta 0.002``; early stopping fired
#: at 417, and the ``_n1000`` file is the un-early-stopped full run. The **default**
#: file is the one ``config.yaml``'s ``results`` block describes, so it is the one
#: ``prompts_fitted`` refers to and the direct analogue of what we validated against at
#: 0.8B. It is therefore the Regime A reference here; the ``_n1000`` variant is
#: deliberately not fetched. Recorded rather than silently omitted, because "there is a
#: second lens" is exactly the thing a later reader would otherwise rediscover the hard
#: way.
QWEN35_4B = PublishedLens(
    model="qwen3.5-4b",
    hf_model="Qwen/Qwen3.5-4B",
    path="qwen3.5-4b/jlens/Salesforce-wikitext",
    files=(
        RemoteFile(
            "Qwen3.5-4B_jacobian_lens.pt",
            size=406_333_179,
            sha256="c2e20eb414caf67da4c271fb9be52c5b84cb0af5c3c42ff93b88b55f7e67b154",
        ),
        RemoteFile("Qwen3.5-4B_convergence.csv", size=17_796),
        RemoteFile("config.yaml", size=2_511),
    ),
)

REGISTRY: dict[str, PublishedLens] = {
    QWEN35_08B.model: QWEN35_08B,
    QWEN35_4B.model: QWEN35_4B,
}


@dataclass(frozen=True)
class FetchedFile:
    name: str
    path: Path
    size: int
    sha256: str
    verified: bool  # checked against an independently recorded sha256
    cached: bool  # already on disk and intact, so not re-downloaded


def resolve_url(lens: PublishedLens, filename: str, revision: str = "main") -> str:
    return f"{HUB}/{ARTIFACT_REPO}/resolve/{revision}/{lens.path}/{filename}"


def destination(lens: PublishedLens, cfg: Config) -> Path:
    """Mirror the remote layout under the reference root.

    Keeping the publisher's own path means ``reference/`` is a faithful partial
    mirror of their repo: the model, the lens type and the corpus are all legible
    from the path, and a second corpus or lens type for the same model cannot
    collide with this one.
    """
    return cfg.reference / lens.path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(remote: RemoteFile, size: int, digest: str) -> None:
    if remote.size is not None and size != remote.size:
        raise FetchError(
            f"{remote.name}: expected {remote.size} bytes, got {size}. "
            f"A short read usually means the transfer was truncated; a long one "
            f"means the published artifact changed."
        )
    if remote.sha256 is not None and digest != remote.sha256:
        raise FetchError(
            f"{remote.name}: sha256 mismatch.\n"
            f"  expected {remote.sha256}\n"
            f"  got      {digest}\n"
            f"The expected value is the LFS oid recorded in T1, so this means the "
            f"bytes differ from what the Hub published -- do not use this file."
        )


def _download(url: str, dest: Path, remote: RemoteFile, timeout: float) -> tuple[int, str]:
    """Stream ``url`` to ``dest``, verifying before it appears at that name.

    Written to a ``.part`` sibling and moved into place only once size and hash
    check out, so an interrupted or corrupted transfer can never leave a file that
    looks complete. Same discipline as the harness's ``_atomic_save``.
    """
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    size = 0

    # Identity encoding so Content-Length and the bytes we hash describe the same
    # thing; urllib does not transparently decompress a gzipped response.
    request = urllib.request.Request(
        url, headers={"Accept-Encoding": "identity", "User-Agent": "jlens-extensions/0.1"}
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(
            partial, "wb"
        ) as handle:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise FetchError(
            f"{remote.name}: HTTP {exc.code} from {url}. "
            f"A 404 usually means the repo path or revision is wrong -- note the "
            f"artifacts are in a model repo, so the form is /resolve/<rev>/<path>."
        ) from exc
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{remote.name}: transfer failed: {exc}") from exc

    try:
        _check(remote, size, digest.hexdigest())
    except FetchError:
        partial.unlink(missing_ok=True)
        raise

    os.replace(partial, dest)
    return size, digest.hexdigest()


def fetch(
    lens: PublishedLens,
    cfg: Config,
    *,
    revision: str = "main",
    force: bool = False,
    timeout: float = 60.0,
) -> list[FetchedFile]:
    """Download ``lens``'s files into the reference area, skipping intact ones.

    Idempotent: a file already on disk whose size and (where known) hash match is
    left alone and reported as cached. Pass ``force`` to re-download regardless.
    """
    target = destination(lens, cfg)
    target.mkdir(parents=True, exist_ok=True)

    results: list[FetchedFile] = []
    for remote in lens.files:
        dest = target / remote.name
        cached = False

        if dest.exists() and not force:
            size, digest = dest.stat().st_size, sha256_of(dest)
            try:
                _check(remote, size, digest)
                cached = True
            except FetchError:
                # On disk but wrong -- a truncated earlier attempt, most likely.
                # Re-fetch rather than fail; the new copy is verified before it
                # replaces this one.
                cached = False

        if not cached:
            size, digest = _download(resolve_url(lens, remote.name, revision), dest, remote, timeout)

        results.append(
            FetchedFile(
                name=remote.name,
                path=dest,
                size=size,
                sha256=digest,
                verified=remote.sha256 is not None,
                cached=cached,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a published Neuronpedia lens for comparison.",
    )
    parser.add_argument("--model", default=QWEN35_08B.model, choices=sorted(REGISTRY))
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the file is present and intact"
    )
    args = parser.parse_args(argv)

    try:
        cfg = load()
    except ConfigError as exc:
        print(f"jlens_extensions fetch: FAILED\n\n{exc}", file=sys.stderr)
        return 1

    lens = REGISTRY[args.model]
    print(f"{lens.model}  (fitted on {lens.hf_model})")
    print(f"from  {HUB}/{ARTIFACT_REPO} @ {args.revision}/{lens.path}")
    print(f"into  {destination(lens, cfg)}\n")

    try:
        results = fetch(lens, cfg, revision=args.revision, force=args.force)
    except FetchError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    # Printed in a shape worth pasting into T17's manifest, which cites these as
    # the comparison baseline.
    for item in results:
        state = "cached" if item.cached else "downloaded"
        check = "verified against T1" if item.verified else "recorded"
        print(f"  {item.name}")
        print(f"    {item.size} bytes, {state}")
        print(f"    sha256 {item.sha256}  ({check})")

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
