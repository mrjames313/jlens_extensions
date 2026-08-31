"""Tests for the machine profile.

Two properties carry most of the weight, and both are *negative*:

* **A missing profile raises**, rather than letting a fit proceed hand-tuned. Same
  principle as the config module's no-fallback rule -- a silent default is how a run
  ends up parameterised by someone's memory.
* **An unknown model raises**, rather than falling back to another model's entry.
  `dim_batch`, timing and `force_bos_effective` are all model-specific, and
  Qwen3.5's `force_bos_effective = false` is true only because that checkpoint has
  no BOS token at all. Inheriting it would be silently wrong.

The serialiser is hand-rolled (see the module docstring for why), so it also gets
round-trip coverage -- particularly for model ids like ``Qwen/Qwen3.5-0.8B``, whose
slash and dots make it an invalid TOML bare key.
"""

import pytest

from jlens_extensions.profile import (
    SCHEMA_VERSION,
    HostFacts,
    MachineProfile,
    ModelFacts,
    ProfileError,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def host() -> HostFacts:
    return HostFacts(
        device="NVIDIA GB10",
        compute_capability="12.1",
        total_memory_gb=121.0,
        unified_memory=True,
        device_map="cuda",
    )


def facts(**overrides) -> ModelFacts:
    base = dict(
        dim_batch=8,
        dim_batch_basis="optimum",
        compile=True,
        s_per_prompt=15.53,
        peak_alloc_gb=9.63,
        peak_reserved_gb=9.68,
        force_bos_effective=False,
        measured_over_prompts=5,
        dim_batch_ceiling_measured=64,
        dim_batch_oom_at=128,
        memory_model_intercept_gb=1.60,
        memory_model_slope_gb=1.004,
    )
    base.update(overrides)
    return ModelFacts(**base)


def profile(**overrides) -> MachineProfile:
    return MachineProfile(
        machine="gx10-ace5",
        updated="2026-08-26",
        host=host(),
        models={MODEL_ID: facts(**overrides)},
    )


# --- the no-fallback guarantees ----------------------------------------------


def test_missing_profile_raises_and_names_the_step(tmp_path):
    with pytest.raises(ProfileError) as excinfo:
        MachineProfile.load(tmp_path / "nonexistent.toml")
    assert "bring-up" in str(excinfo.value)


def test_unknown_model_raises_rather_than_substituting(tmp_path):
    loaded = _round_trip(profile(), tmp_path)
    with pytest.raises(ProfileError) as excinfo:
        loaded.model("Qwen/Qwen3.5-27B")
    message = str(excinfo.value)
    assert "Qwen/Qwen3.5-27B" in message
    assert MODEL_ID in message  # tells the reader what it does have
    assert "model-specific" in message


def test_known_model_is_returned(tmp_path):
    loaded = _round_trip(profile(), tmp_path)
    assert loaded.model(MODEL_ID).dim_batch == 8


def test_malformed_profile_raises(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text('machine = "x"\nupdated = "y"\n[host]\ndevice = "z"\n')
    with pytest.raises(ProfileError, match="not a valid machine profile"):
        MachineProfile.load(path)


# --- round-trip ---------------------------------------------------------------


def _round_trip(original: MachineProfile, tmp_path) -> MachineProfile:
    path = original.write(tmp_path / "gx10-ace5.toml")
    return MachineProfile.load(path)


def test_round_trip_preserves_everything(tmp_path):
    original = profile()
    loaded = _round_trip(original, tmp_path)
    assert loaded.machine == original.machine
    assert loaded.updated == original.updated
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.host == original.host
    assert loaded.models == original.models


def test_model_id_with_slash_and_dots_survives(tmp_path):
    """`Qwen/Qwen3.5-0.8B` is not a valid TOML bare key; it must be quoted."""
    text = profile().to_toml()
    assert f'[model."{MODEL_ID}"]' in text
    assert MODEL_ID in _round_trip(profile(), tmp_path).models


def test_booleans_are_toml_booleans_not_ints(tmp_path):
    """bool is a subclass of int, so encoder order matters."""
    text = profile().to_toml()
    assert "compile = true" in text
    assert "force_bos_effective = false" in text
    assert "compile = 1" not in text


def test_whole_floats_keep_a_decimal_point(tmp_path):
    text = profile(s_per_prompt=16.0).to_toml()
    assert "s_per_prompt = 16.0" in text
    assert _round_trip(profile(s_per_prompt=16.0), tmp_path).model(MODEL_ID).s_per_prompt == 16.0


def test_unmeasured_fields_are_omitted_not_nulled(tmp_path):
    """TOML has no null; an absent key reads as 'not measured'."""
    text = profile(dim_batch_oom_at=None).to_toml()
    assert "dim_batch_oom_at" not in text
    assert _round_trip(profile(dim_batch_oom_at=None), tmp_path).model(MODEL_ID).dim_batch_oom_at is None


def test_strings_with_quotes_are_escaped(tmp_path):
    original = profile(notes='he said "8", not 128')
    assert _round_trip(original, tmp_path).model(MODEL_ID).notes == 'he said "8", not 128'


def test_two_models_stay_separate(tmp_path):
    both = profile()
    both.models["Qwen/Qwen3.5-4B"] = facts(dim_batch=4, s_per_prompt=48.0, force_bos_effective=True)
    loaded = _round_trip(both, tmp_path)
    assert loaded.model(MODEL_ID).force_bos_effective is False
    assert loaded.model("Qwen/Qwen3.5-4B").force_bos_effective is True
    assert loaded.model("Qwen/Qwen3.5-4B").s_per_prompt == 48.0


# --- the ceiling/optimum distinction is recorded, not implied -----------------


def test_dim_batch_basis_distinguishes_optimum_from_ceiling(tmp_path):
    """A reader that assumed 'ceiling' would pick the slowest setting that fits."""
    loaded = _round_trip(profile(), tmp_path)
    entry = loaded.model(MODEL_ID)
    assert entry.dim_batch_basis == "optimum"
    assert entry.dim_batch == 8
    assert entry.dim_batch_ceiling_measured == 64
    assert entry.dim_batch != entry.dim_batch_ceiling_measured


# --- add_model_profile's preservation of measured fields (band-location T15) ---


def test_replacing_an_entry_must_not_silently_drop_gate_identity():
    """ModelFacts defaults gate_identity to None, so a rebuild loses it quietly.

    That is 15-20 minutes of GPU time at 4B, and its absence makes
    t15_validation_fit.py refuse a compiled fit -- but nothing complains until the fit
    is attempted, long after the entry was rewritten. This asserts the field's default
    is the trap it looks like, so the preservation in add_model_profile.py has a
    reason recorded next to it.
    """
    fresh = facts()
    assert fresh.gate_identity is None, (
        "if this ever gains a non-None default, add_model_profile.py's preservation "
        "logic should be revisited"
    )


def test_compile_flag_can_be_flipped_without_touching_the_gate_reference(tmp_path):
    """The soak verdict arrives as one boolean, after gate_identity is measured."""
    profile = MachineProfile(
        machine="test-box", updated="2026-08-31",
        host=host(),
        models={"M": facts(compile=False, gate_identity=0.522718)},
    )
    path = profile.write(tmp_path / "p.toml")

    reloaded = MachineProfile.load(path)
    entry = reloaded.model("M")
    entry.compile = True
    reloaded.write(path)

    after = MachineProfile.load(path).model("M")
    assert after.compile is True
    assert after.gate_identity == 0.522718
    assert after.gate_identity_basis == "uncompiled single prompt"
