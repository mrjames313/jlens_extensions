"""Access to upstream's prompt sets and evaluation items, vendored under ``data/``.

The J-lens experiments need prompt material that ships with
``anthropics/jacobian-lens`` -- ``ignition.json``'s carrier sentences and concept
pairs, ``lens-eval-multihop.json``'s two-hop items, and eleven more. That material
lives in a **gitignored per-clone checkout** under the arrangement in
``d-commons-reference-checkouts``, which is fine for code we only read and wrong for
data a *result* depends on: an experiment citing ``ignition.json`` from there cannot
be re-run from a fresh clone without re-cloning upstream at an unpinned revision.

So ``data/`` is a vendored copy, byte-identical to upstream at the revision recorded
in ``PROVENANCE.md``, and this module is the only supported way to reach it. Nothing
should read the reference checkout at runtime.

**The READMEs are part of the data, not documentation about it.** They carry the
protocol each prompt set is scored under -- what a "hit" is, which field is the target,
how the readout is defined. ``ignition.json`` is 4 KB of word lists and is close to
meaningless without ``data/experiments/README.md`` saying what to do with them, so
both are vendored and :func:`readme` exists to make that reachable rather than
something you have to know to go and look for.

Deliberately not a schema layer. Each file has its own shape and upstream may change
them; this returns parsed JSON and leaves interpretation to the driver, which is where
the protocol knowledge belongs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Repo root. ``src/jlens_extensions/upstream_data.py`` -> ``jlens_extensions/``.
#: Matches how ``experiments/`` drivers resolve the repo; the package is run in
#: place rather than installed, exactly as ``harness/`` is.
REPO = Path(__file__).resolve().parents[2]

DATA_ROOT = REPO / "data"

#: The two subdirectories upstream ships. ``experiments`` holds the prompt sets for
#: the global-workspace experiments; ``evaluations`` holds the lens-eval item sets.
KINDS = ("experiments", "evaluations")


class DataError(RuntimeError):
    """A vendored data file is missing or unreadable.

    Never resolved by falling back to the reference checkout. Falling back would
    reintroduce exactly the reproducibility hole the vendoring closed, and would do
    it silently on the one machine that happens to have the checkout.
    """


def _resolve(kind: str, name: str) -> Path:
    if kind not in KINDS:
        raise DataError(f"unknown data kind {kind!r}; expected one of {KINDS}")
    stem = name[:-5] if name.endswith(".json") else name
    path = DATA_ROOT / kind / f"{stem}.json"
    if not path.exists():
        raise DataError(
            f"no vendored data file at {path}. Available in {kind}: "
            f"{', '.join(sorted(available(kind))) or '<none>'}. "
            f"If data/ is missing entirely the clone is incomplete -- it is committed, "
            f"not fetched, so re-clone rather than re-running the reference checkout."
        )
    return path


def available(kind: str) -> list[str]:
    """Stems of the JSON files vendored under ``kind``."""
    directory = DATA_ROOT / kind
    if not directory.is_dir():
        return []
    return [p.stem for p in directory.glob("*.json")]


def load(kind: str, name: str) -> Any:
    """Parsed JSON for one vendored file.

    Args:
        kind: ``"experiments"`` or ``"evaluations"``.
        name: File stem, with or without the ``.json`` suffix.

    Raises:
        DataError: If the file is absent, or ``kind`` is not one upstream ships.
    """
    path = _resolve(kind, name)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DataError(f"{path} is not valid JSON: {exc}") from exc


def experiment(name: str) -> Any:
    """Parsed JSON for an experiment prompt set, e.g. ``experiment("ignition")``."""
    return load("experiments", name)


def evaluation(name: str) -> Any:
    """Parsed JSON for an evaluation item set, e.g. ``evaluation("lens-eval-multihop")``."""
    return load("evaluations", name)


def readme(kind: str) -> str:
    """The README for ``kind`` -- the protocol definitions, not just prose.

    Read this before writing a driver against one of these files. It is what says
    which field is the target, what counts as a hit, and how the readout is defined.
    """
    if kind not in KINDS:
        raise DataError(f"unknown data kind {kind!r}; expected one of {KINDS}")
    path = DATA_ROOT / kind / "README.md"
    if not path.exists():
        raise DataError(f"no README at {path}; the vendored copy is incomplete")
    return path.read_text()
