"""Scaffold guards for the repository layout.

Deliberately free of ``torch`` / ``transformers`` imports so this suite runs before
the heavy environment exists — it is the only thing that can be checked between the
vendoring commit and the environment being resolved.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "harness"

VENDORED = [
    "fit_lens.py",
    "LICENSE",
    "jlens/__init__.py",
    "jlens/_logging.py",
    "jlens/fitting.py",
    "jlens/hf.py",
    "jlens/hooks.py",
    "jlens/lens.py",
    "jlens/protocol.py",
]


def test_package_imports():
    import jlens_extensions

    assert jlens_extensions.__version__


def test_vendored_slice_present():
    missing = [p for p in VENDORED if not (HARNESS / p).is_file()]
    assert not missing, f"missing vendored files: {missing}"


def test_harness_is_not_installed_as_a_package():
    """``import jlens`` must not resolve to an installed copy.

    ``fit_lens.py`` resolves ``import jlens`` through ``sys.path[0]`` — its own
    directory. If ``harness/jlens/`` were ever packaged and installed, an installed
    ``jlens`` would compete with the script-resolved one and which you got would
    depend on the working directory. ``pyproject.toml`` prevents that by searching
    only ``src/``; this asserts the outcome rather than trusting the config.

    Revisit if the experiment phase ever installs upstream ``anthropics/jacobian-lens``
    rather than reading it from a clone — at that point an importable ``jlens`` becomes
    expected, and this guard needs to name *which* copy it found instead.
    """
    spec = importlib.util.find_spec("jlens")
    origin = getattr(spec, "origin", None)
    assert spec is None, f"`jlens` is importable from {origin}; harness/ must not be installed"
