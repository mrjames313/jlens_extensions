"""Tests for the environment configuration.

The load-bearing property is *negative*: an unset root must raise rather than
silently default. Most of what follows checks that no fallback exists, because a
fallback is how a multi-terabyte checkpoint stream reaches the wrong filesystem.

Every test passes an explicit ``env`` mapping, so none of them read or mutate the
real process environment.
"""

import os

import pytest

from jlens_extensions.config import (
    ARTIFACT_ROOT_VAR,
    MACHINE_VAR,
    SCRATCH_ROOT_VAR,
    ConfigError,
    load,
)


def env(**overrides):
    base = {ARTIFACT_ROOT_VAR: "/srv/artifacts", MACHINE_VAR: "gx10-ace5"}
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


# --- the no-fallback guarantee ------------------------------------------------


@pytest.mark.parametrize("missing", [ARTIFACT_ROOT_VAR, MACHINE_VAR])
def test_missing_required_var_raises_naming_it(missing):
    with pytest.raises(ConfigError) as excinfo:
        load(env(**{missing: None}))
    assert missing in str(excinfo.value)


@pytest.mark.parametrize("missing", [ARTIFACT_ROOT_VAR, MACHINE_VAR])
def test_blank_required_var_is_treated_as_unset(missing):
    """`export JLENS_MACHINE=` is a mistake, not a value."""
    with pytest.raises(ConfigError):
        load(env(**{missing: "   "}))


def test_empty_environment_does_not_fall_back_to_cwd_or_tmp():
    with pytest.raises(ConfigError):
        load({})


def test_relative_root_is_rejected():
    """A relative root resolves against the working directory, so the same command
    would write somewhere different depending on where it was run from."""
    with pytest.raises(ConfigError) as excinfo:
        load(env(**{ARTIFACT_ROOT_VAR: "artifacts"}))
    assert "absolute" in str(excinfo.value)


# --- scratch defaulting -------------------------------------------------------


def test_scratch_defaults_to_artifact_root_when_unset():
    cfg = load(env(**{SCRATCH_ROOT_VAR: None}))
    assert cfg.scratch_root == cfg.artifact_root
    assert cfg.scratch_is_derived is True


def test_blank_scratch_also_derives():
    cfg = load(env(**{SCRATCH_ROOT_VAR: ""}))
    assert cfg.scratch_root == cfg.artifact_root
    assert cfg.scratch_is_derived is True


def test_explicit_scratch_is_kept_and_flagged_as_not_derived():
    cfg = load(env(**{SCRATCH_ROOT_VAR: "/mnt/nvme/scratch"}))
    assert cfg.scratch_root == os.path.realpath("/mnt/nvme/scratch") or cfg.scratch_root.is_absolute()
    assert cfg.scratch_is_derived is False
    assert cfg.scratch_root != cfg.artifact_root


# --- derived paths ------------------------------------------------------------


def test_derived_paths_hang_off_the_right_root():
    cfg = load(env(**{SCRATCH_ROOT_VAR: "/mnt/nvme/scratch"}))

    # Write-once artifacts live under the artifact root...
    for path in (cfg.lenses, cfg.reference, cfg.profiles):
        assert path.is_relative_to(cfg.artifact_root)

    # ...and the high-write-volume ones under scratch.
    for path in (cfg.checkpoints, cfg.hf_cache):
        assert path.is_relative_to(cfg.scratch_root)


def test_profile_path_is_named_for_the_machine():
    cfg = load(env())
    assert cfg.profile_path == cfg.profiles / "gx10-ace5.toml"


@pytest.mark.parametrize("label", ["a/b", "..", "."])
def test_machine_label_that_would_escape_the_profiles_dir_is_rejected(label):
    with pytest.raises(ConfigError):
        load(env(**{MACHINE_VAR: label}))


# --- the HuggingFace cache trap ----------------------------------------------


def test_hf_env_confines_downloads_to_scratch():
    """We set these ourselves precisely so we never pass --hf_cache_dir, which
    fit_lens.py deletes in a finally unless --keep_hf_cache is also given."""
    cfg = load(env(**{SCRATCH_ROOT_VAR: "/mnt/nvme/scratch"}))
    hf = cfg.hf_env()

    assert set(hf) == {"HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_DATASETS_CACHE"}
    for value in hf.values():
        assert value.startswith(str(cfg.hf_cache))


# --- load() is side-effect free ----------------------------------------------


def test_load_creates_nothing(tmp_path):
    root = tmp_path / "not-created-by-load"
    cfg = load({ARTIFACT_ROOT_VAR: str(root), MACHINE_VAR: "testbox"})
    assert not root.exists()
    assert not cfg.lenses.exists()
