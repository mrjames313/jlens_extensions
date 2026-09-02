"""The compare driver's per-rung facts come from the artifact, and axis 0 needs two fits.

Two different failure shapes are covered here, and only one of them is loud.

**The loud one**: the lens directory. ``t15_validation_fit.py`` leaves the default rung
unqualified (``t15-a``) and qualifies every other (``t15-Qwen3.5-4B-a``). A compare
driver that assumes the unqualified form simply cannot find the 4B fit and says so.

**The silent one**: ``T18_PROJECTED``. It is a 0.8B per-layer map keyed 0..22, and 4B
has layers 0..30 -- so at 4B *every one of those keys still resolves*, printing a 0.8B
projection beside a 4B measurement with a meaningless ratio next to it and nothing
marking it wrong. That is the same shape as the hardcoded ``N_PROMPTS = 233`` in the
sibling driver: plausible output, wrong comparison, no error. Hence
``T18_PROJECTED_BY_MODEL`` and the test below that a rung with no entry gets no column.

The third concern is axis 0. It is not a per-rung constant but a measurement requiring
two fits of the same configuration, and plan decision 4 of ``workspace-band-location``
gave 4B one fit deliberately. Axes 1 and 2 are floored by that envelope, so a driver
that quietly proceeded on one run would report a tighter floor than it had earned.
"""

from __future__ import annotations

import json

import pytest

# Imported here, at module scope, on purpose. ``load_driver`` stubs
# ``jlens_extensions.config`` for the duration of the driver's import, and the driver
# imports ``fetch`` at module level -- which does ``from jlens_extensions.config import
# Config``, a name the stub does not carry. Caching the real module first makes the
# driver's import a no-op. Without this the file passes in a full-suite run (some
# earlier test having imported fetch) and fails when run alone, which is the ordering
# trap ``conftest.load_driver`` documents, seen from the other side.
import jlens_extensions.fetch  # noqa: F401

CONFIG_08B = """
np_model_id: "qwen3.5-0.8b"
fit:
  n_prompts: 1000
  stop_at_delta: 0.002
  min_prompts: 100
  stop_window: 10
results:
  prompts_fitted: 233
  final_identity_distance: 0.531437
"""

CONFIG_4B = CONFIG_08B.replace("233", "417").replace("0.8b", "4b")


@pytest.fixture
def driver(tmp_path, monkeypatch):
    from conftest import load_driver

    module = load_driver("t16_compare")
    monkeypatch.setattr(module.cfg, "artifact_root", tmp_path, raising=False)
    monkeypatch.setattr(module.cfg, "lenses", tmp_path / "lenses", raising=False)
    return module


@pytest.fixture
def published(driver, tmp_path, monkeypatch):
    """Both rungs' config.yaml on disk, under a patched reference root."""
    from jlens_extensions.fetch import REGISTRY

    reference = tmp_path / "reference"
    monkeypatch.setattr(type(driver.cfg), "reference",
                        property(lambda self: reference), raising=False)
    for key, text in (("qwen3.5-0.8b", CONFIG_08B), ("qwen3.5-4b", CONFIG_4B)):
        path = reference / REGISTRY[key].path / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return reference


@pytest.fixture(autouse=True)
def _stub_slug(monkeypatch):
    """``_slug`` lives in ``fit_lens``, which imports torch at module scope."""
    import sys
    import types

    stub = types.ModuleType("fit_lens")
    stub._slug = lambda model: model.rstrip("/").split("/")[-1]
    monkeypatch.setitem(sys.modules, "fit_lens", stub)


# --- the pin, per rung ------------------------------------------------------

def test_prompts_fitted_is_read_per_rung(driver, published):
    assert driver.resolve_target("Qwen/Qwen3.5-0.8B").n_prompts == 233
    assert driver.resolve_target("Qwen/Qwen3.5-4B").n_prompts == 417


def test_unregistered_model_is_refused(driver):
    with pytest.raises(SystemExit, match="no published lens registered"):
        driver.resolve_target("Qwen/Qwen3.5-27B")


# --- the lens directory, which is where the loud failure was ----------------

def test_lens_dir_matches_the_fit_driver_stem(driver, published):
    """The default rung stays unqualified; every other rung is qualified by slug."""
    assert driver.resolve_target("Qwen/Qwen3.5-0.8B").lens_dir("a").name == "t15-a"
    assert driver.resolve_target("Qwen/Qwen3.5-4B").lens_dir("a").name == "t15-Qwen3.5-4B-a"


# --- the silent one ---------------------------------------------------------

def test_t18_projection_is_keyed_by_model(driver):
    """A rung T18 never measured gets no projection column, rather than 0.8B's."""
    table = driver.T18_PROJECTED_BY_MODEL
    assert table.get("Qwen/Qwen3.5-0.8B"), "0.8B is the rung T18 was measured on"
    assert table.get("Qwen/Qwen3.5-4B") is None

    # The trap restated as an assertion: these keys are all valid 4B layers too.
    overlapping = [layer for layer in table["Qwen/Qwen3.5-0.8B"] if layer <= 30]
    assert overlapping, "if this is empty the test below proves nothing"


# --- the early-stop rule ----------------------------------------------------

def test_stop_rule_is_read_from_the_published_config(driver, published):
    rule, source = driver.published_stop_rule(driver.resolve_target("Qwen/Qwen3.5-4B"))
    assert rule == {"stop_at_delta": 0.002, "min_prompts": 100, "window": 10}
    assert set(source.values()) == {"read"}


def test_stop_rule_refuses_when_the_threshold_is_absent(driver, tmp_path, monkeypatch):
    """min_prompts and stop_window have harness defaults; stop_at_delta does not."""
    from jlens_extensions.fetch import REGISTRY

    reference = tmp_path / "reference"
    monkeypatch.setattr(type(driver.cfg), "reference",
                        property(lambda self: reference), raising=False)
    path = reference / REGISTRY["qwen3.5-4b"].path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("results:\n  prompts_fitted: 417\n")

    target = driver.resolve_target("Qwen/Qwen3.5-4B")
    with pytest.raises(SystemExit, match="stop_at_delta"):
        driver.published_stop_rule(target)


def test_stop_rule_marks_a_defaulted_key_as_defaulted(driver, tmp_path, monkeypatch):
    from jlens_extensions.fetch import REGISTRY

    reference = tmp_path / "reference"
    monkeypatch.setattr(type(driver.cfg), "reference",
                        property(lambda self: reference), raising=False)
    path = reference / REGISTRY["qwen3.5-4b"].path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fit:\n  stop_at_delta: 0.002\nresults:\n  prompts_fitted: 417\n")

    rule, source = driver.published_stop_rule(driver.resolve_target("Qwen/Qwen3.5-4B"))
    assert rule["min_prompts"] == 100
    assert source["stop_at_delta"] == "read"
    assert "default" in source["min_prompts"]


# --- the injected envelope --------------------------------------------------

def test_envelope_accepts_both_shapes(driver, tmp_path):
    full = tmp_path / "full.json"
    full.write_text(json.dumps({"source": "single-prompt instrument", "alpha": 0.473,
                                "per_layer": {"0": 1e-4, "1": 5e-5}}))
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"0": 1e-4, "1": 5e-5}))

    for path in (full, bare):
        envelope, prov = driver.load_envelope(path)
        assert envelope == {0: 1e-4, 1: 5e-5}
        assert prov["kind"] == "predicted"
    assert driver.load_envelope(full)[1]["alpha"] == 0.473, "provenance is carried, not dropped"


def test_absent_envelope_is_marked_as_such(driver):
    envelope, prov = driver.load_envelope(None)
    assert envelope == {}
    assert prov["kind"] == "none"


def test_malformed_envelope_is_refused(driver, tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"per_layer": {}}))
    with pytest.raises(SystemExit, match="no per-layer map"):
        driver.load_envelope(empty)

    junk = tmp_path / "junk.json"
    junk.write_text(json.dumps({"per_layer": {"0": "not-a-number"}}))
    with pytest.raises(SystemExit, match="not"):
        driver.load_envelope(junk)

    with pytest.raises(SystemExit, match="does not exist"):
        driver.load_envelope(tmp_path / "absent.json")


# --- axis 0 cannot be silently skipped --------------------------------------

def test_single_run_without_an_envelope_is_refused(driver, monkeypatch):
    """The whole point: axes 1 and 2 are floored by axis 0, so it cannot be dropped."""
    monkeypatch.setattr("sys.argv", ["t16_compare.py", "--runs", "a"])
    with pytest.raises(SystemExit, match="axis 0 cannot be measured"):
        driver.main()


def test_more_than_two_runs_is_refused(driver, monkeypatch):
    monkeypatch.setattr("sys.argv", ["t16_compare.py", "--runs", "a,b,c"])
    with pytest.raises(SystemExit, match="one or two labels"):
        driver.main()
