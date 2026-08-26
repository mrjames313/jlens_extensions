"""The machine profile -- facts about a box, written at bring-up and read by fits.

Stage 3 of ``environment-setup-and-first-fit`` measures things that are properties of
*this hardware* rather than of the J-lens: how large ``dim_batch`` can be, what a
prompt costs, how much memory a fit peaks at. Leaving those as prose in a finding
means a later fit depends on someone remembering a number, and means a run on
different hardware silently inherits measurements that no longer apply.

So bring-up writes a profile and fits read it. It lives at
``$JLENS_ARTIFACT_ROOT/profiles/$JLENS_MACHINE.toml`` -- outside the repo, because it
describes hardware rather than knowledge.

Two kinds of fact, kept apart
-----------------------------

``[host]`` is about the box: device, capability, total memory, and **which quantity
the memory numbers hold**. That last one matters more than it looks: on a
unified-memory box ``nvidia-smi`` reports ``Memory-Usage: Not Supported``, so the
figures come from ``torch.cuda.max_memory_allocated`` -- the *allocator*, not the
device. Comparing that against a discrete-GPU box's ``nvidia-smi`` reading would
compare two different things, so the profile records which one it is.

``[model."<hf-id>"]`` is per model. This split is deliberate: the spec plan listed
``force_bos_effective`` among the profile's fields, but whether ``force_bos`` fires
is a property of a *tokenizer*, not of a box. Stored flat it would let a second model
inherit the first's answer -- and Qwen3.5's answer is ``false`` only because that
checkpoint has no BOS token at all. Same for ``dim_batch``, whose memory scaling
depends on ``n_layers`` and ``d_model``.

``dim_batch`` is the optimum, not the ceiling
---------------------------------------------

The plan expected the profile to carry "the measured ceiling, neither the scripts'
default of 8 nor the B200's 128", on the reasoning that host transfers scale as
``1/dim_batch`` so the largest value that fits is the right one. Measured on the
GB10, per-prompt time *rises* with ``dim_batch`` and the default of 8 is optimal. So
``dim_batch`` holds the value to run at and ``dim_batch_basis`` says which kind of
value it is -- a reader that assumes "ceiling" would otherwise pick the slowest
setting that fits.

Why the TOML is hand-written
----------------------------

``tomllib`` reads TOML from Python 3.11, but the standard library has no writer, and
``PROVENANCE.md`` asserts our runtime dependency list is verbatim Neuronpedia's --
adding ``tomli-w`` would falsify that for a serialiser we can write in forty lines.
The subset used here is scalars and one level of nested table, which is exactly what
:func:`_dumps` covers.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "ProfileError",
    "HostFacts",
    "ModelFacts",
    "MachineProfile",
    "probe_host",
]

SCHEMA_VERSION = 1

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


class ProfileError(RuntimeError):
    """A profile is missing, malformed, or does not cover the model asked for."""


@dataclass
class HostFacts:
    """What is true of the box, independent of what is being fitted."""

    device: str
    compute_capability: str
    total_memory_gb: float
    unified_memory: bool
    device_map: str
    #: Which quantity ``peak_*_gb`` fields hold. Not cosmetic -- see module docstring.
    memory_metric: str = "torch.cuda.max_memory_allocated"


@dataclass
class ModelFacts:
    """What is true of one model on this box."""

    dim_batch: int
    #: ``"optimum"`` (fastest measured) or ``"ceiling"`` (largest that fits). These
    #: are not the same value and on this hardware they are at opposite ends.
    dim_batch_basis: str
    compile: bool
    s_per_prompt: float
    peak_alloc_gb: float
    peak_reserved_gb: float
    force_bos_effective: bool
    measured_over_prompts: int
    #: Largest value actually observed to fit, and the smallest that did not.
    dim_batch_ceiling_measured: int | None = None
    dim_batch_oom_at: int | None = None
    #: peak_alloc_gb ~= intercept + slope * dim_batch. Lets a box with different
    #: memory recompute its own ceiling instead of inheriting this one's.
    memory_model_intercept_gb: float | None = None
    memory_model_slope_gb: float | None = None
    #: s_per_prompt comes from FitProgress.elapsed_s, which is taken before
    #: write_checkpoint(), so checkpoint I/O is not in it.
    s_per_prompt_excludes: str = "checkpoint I/O"
    notes: str = ""


@dataclass
class MachineProfile:
    machine: str
    updated: str
    host: HostFacts
    models: dict[str, ModelFacts] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def model(self, model_id: str) -> ModelFacts:
        """Facts for ``model_id``, or a clear error.

        Deliberately does not fall back to another model's entry: the fields here
        are model-specific and a silent substitution is the failure this profile
        exists to prevent.
        """
        try:
            return self.models[model_id]
        except KeyError:
            known = ", ".join(sorted(self.models)) or "<none>"
            raise ProfileError(
                f"{self.machine} has no profile entry for {model_id!r} "
                f"(has: {known}). Run bring-up for this model rather than reusing "
                f"another model's numbers -- dim_batch, timing and force_bos_effective "
                f"are all model-specific."
            ) from None

    # --- serialisation -----------------------------------------------------

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_toml())
        return path

    def to_toml(self) -> str:
        head = {
            "schema_version": self.schema_version,
            "machine": self.machine,
            "updated": self.updated,
        }
        out = [_dumps(head), "", "[host]", _dumps(asdict(self.host))]
        for model_id, facts in self.models.items():
            out += ["", f"[model.{_key(model_id)}]", _dumps(asdict(facts))]
        return "\n".join(out).rstrip() + "\n"

    @classmethod
    def load(cls, path: Path | str) -> MachineProfile:
        path = Path(path)
        if not path.exists():
            raise ProfileError(
                f"no machine profile at {path}. Stage-3 bring-up writes it; a fit "
                f"must not be hand-tuned in its absence."
            )
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        try:
            host = HostFacts(**raw["host"])
            models = {k: ModelFacts(**v) for k, v in raw.get("model", {}).items()}
        except (KeyError, TypeError) as exc:
            raise ProfileError(f"{path} is not a valid machine profile: {exc}") from exc
        return cls(
            machine=raw["machine"],
            updated=raw["updated"],
            host=host,
            models=models,
            schema_version=raw.get("schema_version", 0),
        )


def probe_host(device_map: str = "cuda") -> HostFacts:
    """Read the box's own description from torch rather than transcribing it."""
    import torch

    if not torch.cuda.is_available():
        raise ProfileError("no CUDA device; cannot probe host facts")
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    # Integrated == the GPU shares system memory, i.e. nvidia-smi has no discrete
    # pool to report. getattr because the attribute is not on every torch build.
    unified = bool(getattr(props, "is_integrated", False)) or total_gb > 100
    return HostFacts(
        device=props.name,
        compute_capability=f"{props.major}.{props.minor}",
        total_memory_gb=round(total_gb, 2),
        unified_memory=unified,
        device_map=device_map,
    )


# --- a minimal TOML writer (see module docstring for why it is hand-rolled) ----


def _key(name: str) -> str:
    return name if _BARE_KEY.match(name) else _string(name)


def _string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _value(value: Any) -> str:
    if isinstance(value, bool):  # before int -- bool is a subclass of int
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        return text if ("." in text or "e" in text or "n" in text) else text + ".0"
    if isinstance(value, str):
        return _string(value)
    raise TypeError(f"no TOML encoding for {type(value).__name__}: {value!r}")


def _dumps(mapping: dict[str, Any]) -> str:
    # None is dropped: TOML has no null, and an absent key reads as "not measured".
    return "\n".join(
        f"{_key(k)} = {_value(v)}" for k, v in mapping.items() if v is not None
    )
