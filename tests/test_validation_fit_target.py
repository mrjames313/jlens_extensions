"""The fit's `--n_prompts` pin comes from the published artifact, never a constant.

This is the highest-consequence silent failure in the driver. A wrong pin does not
crash: the fit runs to completion, saves a plausible lens, and is then compared under
Regime A against an artifact fitted on a *different* number of prompts. The published
counts differ per rung -- 233 at 0.8B, 417 at 4B -- so a constant that is right for one
model is wrong for the next, and nothing downstream would say so.
"""

from __future__ import annotations

import pytest

CONFIG_08B = """
np_model_id: "qwen3.5-0.8b"
fit:
  n_prompts: 1000
  dim_batch: 128
results:
  prompts_fitted: 233
  final_identity_distance: 0.531437
"""

CONFIG_4B = CONFIG_08B.replace("233", "417").replace("0.8b", "4b")


@pytest.fixture
def driver(tmp_path, monkeypatch):
    from conftest import load_driver

    module = load_driver("t15_validation_fit")
    monkeypatch.setattr(module.cfg, "artifact_root", tmp_path, raising=False)
    return module


def _publish(root, rel_path, text):
    path = root / rel_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_pin_is_read_from_the_artifact_for_each_rung(driver, tmp_path, monkeypatch):
    """0.8B must give 233 and 4B must give 417, from the same code path."""
    from jlens_extensions.fetch import REGISTRY

    reference = tmp_path / "reference"
    monkeypatch.setattr(type(driver.cfg), "reference",
                        property(lambda self: reference), raising=False)
    _publish(reference, REGISTRY["qwen3.5-0.8b"].path, CONFIG_08B)
    _publish(reference, REGISTRY["qwen3.5-4b"].path, CONFIG_4B)

    assert driver.published_prompts_fitted("Qwen/Qwen3.5-0.8B") == 233
    assert driver.published_prompts_fitted("Qwen/Qwen3.5-4B") == 417


def test_missing_artifact_refuses_rather_than_defaulting(driver, tmp_path, monkeypatch):
    """Substituting a default here is the failure this function exists to prevent."""
    reference = tmp_path / "empty"
    monkeypatch.setattr(type(driver.cfg), "reference",
                        property(lambda self: reference), raising=False)
    with pytest.raises(SystemExit, match="no published config"):
        driver.published_prompts_fitted("Qwen/Qwen3.5-4B")


def test_unregistered_model_is_refused(driver):
    with pytest.raises(SystemExit, match="no published lens registered"):
        driver.published_prompts_fitted("Qwen/Qwen3.5-27B")


def test_config_without_the_field_refuses(driver, tmp_path, monkeypatch):
    from jlens_extensions.fetch import REGISTRY

    reference = tmp_path / "reference"
    monkeypatch.setattr(type(driver.cfg), "reference",
                        property(lambda self: reference), raising=False)
    _publish(reference, REGISTRY["qwen3.5-4b"].path, "results:\n  other: 1\n")
    with pytest.raises(SystemExit, match="no results.prompts_fitted"):
        driver.published_prompts_fitted("Qwen/Qwen3.5-4B")


def test_only_the_default_rung_keeps_the_unqualified_output_name(driver):
    """The validated 0.8B artifacts live at t15-a / t15-b and must stay addressable,
    while a second rung must not be able to write into that directory."""
    default = driver.FitTarget("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", 233, "T15")
    other = driver.FitTarget("Qwen/Qwen3.5-4B", "Qwen3.5-4B", 417, "T16")
    assert default.is_default_rung is True
    assert other.is_default_rung is False
