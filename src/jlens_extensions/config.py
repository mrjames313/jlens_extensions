"""Where artifacts live on this machine.

Three environment variables, set once per box, plus the subpaths derived from them.
Lens tensors, checkpoints and downloaded reference artifacts are large and
machine-specific, so none of them live in the repo and nothing about *where* they
live is hardcoded.

**Unset is an error, not a default.** There is deliberately no fallback to
``./artifacts`` or ``/tmp``. A silent default is how a multi-terabyte checkpoint
stream lands on the wrong filesystem, which at 27B costs a week. Every failure here
names the variable that is missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ARTIFACT_ROOT_VAR = "JLENS_ARTIFACT_ROOT"
SCRATCH_ROOT_VAR = "JLENS_SCRATCH_ROOT"
MACHINE_VAR = "JLENS_MACHINE"

_WHAT_THEY_HOLD = {
    ARTIFACT_ROOT_VAR: "fitted lenses, downloaded reference artifacts, machine profiles",
    SCRATCH_ROOT_VAR: "fit checkpoints and the HuggingFace weight cache",
    MACHINE_VAR: "a short label for this box, e.g. gx10-ace5",
}


class ConfigError(RuntimeError):
    """The environment is not configured. Never resolved by falling back to a default."""


@dataclass(frozen=True)
class Config:
    """Resolved absolute roots plus the subpaths derived from them."""

    artifact_root: Path
    scratch_root: Path
    machine: str

    #: True when ``JLENS_SCRATCH_ROOT`` was unset and scratch followed the artifact
    #: root. Single-disk machines are expected to be in this state; surfacing it
    #: keeps that legible rather than looking like an accident.
    scratch_is_derived: bool

    # --- artifact root: write-once, bulk storage is fine ---------------------

    @property
    def lenses(self) -> Path:
        """Lenses we fit ourselves."""
        return self.artifact_root / "lenses"

    @property
    def reference(self) -> Path:
        """Published artifacts downloaded for comparison."""
        return self.artifact_root / "reference"

    @property
    def profiles(self) -> Path:
        return self.artifact_root / "profiles"

    @property
    def profile_path(self) -> Path:
        """This machine's profile. Written by stage 3, read by stage 4."""
        return self.profiles / f"{self.machine}.toml"

    # --- scratch root: high write volume, wants fast local storage -----------

    @property
    def checkpoints(self) -> Path:
        """Fit checkpoints.

        The vendored fitting loop rewrites the entire running sum every prompt --
        ~100 MB per prompt at 0.8B and ~6.7 GB at 27B -- so this is roughly 23 GB
        of writes for a 233-prompt 0.8B fit and ~3 TB for a 27B one. Capacity is
        not the constraint; write volume is.
        """
        return self.scratch_root / "checkpoints"

    @property
    def hf_cache(self) -> Path:
        return self.scratch_root / "hf-cache"

    # --- derived directories, for the check and for callers that create them --

    @property
    def all_dirs(self) -> tuple[Path, ...]:
        return (self.lenses, self.reference, self.profiles, self.checkpoints, self.hf_cache)

    def hf_env(self) -> dict[str, str]:
        """HuggingFace cache variables confining downloads to our scratch root.

        Mirrors exactly what ``fit_lens.py --hf_cache_dir`` sets, and exists so we
        never have to pass that flag.

        **Do not pass ``--hf_cache_dir``.** The script ``rmtree``s that directory in
        a ``finally`` unless ``--keep_hf_cache`` is also given -- correct for a
        38-model fleet run, and wrong for us, since it re-downloads every weight on
        every run. Exporting these four variables gets the confinement without the
        deletion.
        """
        root = str(self.hf_cache)
        return {
            "HF_HOME": root,
            "HF_HUB_CACHE": os.path.join(root, "hub"),
            "HF_XET_CACHE": os.path.join(root, "xet"),
            "HF_DATASETS_CACHE": os.path.join(root, "datasets"),
        }

    def ensure_dirs(self) -> None:
        """Create the derived directories. Roots themselves must already exist."""
        for d in self.all_dirs:
            d.mkdir(parents=True, exist_ok=True)


def _require(env: Mapping[str, str], var: str) -> str:
    value = env.get(var, "").strip()
    if not value:
        raise ConfigError(
            f"{var} is not set. It holds {_WHAT_THEY_HOLD[var]}.\n"
            f"Set it in your shell profile, e.g.:\n"
            f"    export {var}=...\n"
            f"There is no default on purpose -- see jlens_extensions.config."
        )
    return value


def _as_root(value: str, var: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ConfigError(
            f"{var} must be an absolute path, got {value!r}. A relative root would "
            f"resolve against the working directory, so the same command would write "
            f"to different places depending on where it was run from."
        )
    return path.resolve()


def load(env: Mapping[str, str] | None = None) -> Config:
    """Read the configuration from the environment.

    Raises :class:`ConfigError` naming the offending variable if anything is missing
    or malformed. Performs no filesystem writes and does not check that the roots
    exist -- that is :mod:`jlens_extensions.setup_check`'s job.
    """
    env = os.environ if env is None else env

    artifact_root = _as_root(_require(env, ARTIFACT_ROOT_VAR), ARTIFACT_ROOT_VAR)

    raw_scratch = env.get(SCRATCH_ROOT_VAR, "").strip()
    scratch_is_derived = not raw_scratch
    scratch_root = artifact_root if scratch_is_derived else _as_root(raw_scratch, SCRATCH_ROOT_VAR)

    machine = _require(env, MACHINE_VAR)
    # The label becomes a filename (profiles/<machine>.toml) and is stamped into
    # every measurement, so reject anything that would escape the profiles dir.
    if os.sep in machine or (os.altsep and os.altsep in machine) or machine in (".", ".."):
        raise ConfigError(
            f"{MACHINE_VAR}={machine!r} must be a bare label with no path separators "
            f"-- it is used as a filename in profiles/."
        )

    return Config(
        artifact_root=artifact_root,
        scratch_root=scratch_root,
        machine=machine,
        scratch_is_derived=scratch_is_derived,
    )
