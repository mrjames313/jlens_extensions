"""Steps 8 and 9 of the single-prompt instrument, and the guards around them.

The arithmetic is three lines, so what is worth testing is not the arithmetic. It is
that the driver refuses to emit an envelope it does not have, and that the exponent's
range caveat reaches the output rather than staying in a task note.

Both matter for the same reason. The envelope becomes the FLOOR that
``t16_compare.py`` scores axes 1 and 2 against, so a plausible-but-wrong envelope
silently moves every A1 verdict on the rung. A default substituted for a failed
measurement would be exactly that.
"""

from __future__ import annotations

import json

import pytest

import jlens_extensions.fetch  # noqa: F401  -- see test_compare_target for why


@pytest.fixture(autouse=True)
def _stub_slug(monkeypatch):
    """``_slug`` lives in ``fit_lens``, which imports torch at module scope.

    Imported from there rather than reimplemented because the stem it produces has to
    match the one ``t15_validation_fit.py`` wrote the lens under -- two implementations
    of "filesystem-safe stem" that drift is how a driver stops finding its own output.
    """
    import sys
    import types

    stub = types.ModuleType("fit_lens")
    stub._slug = lambda model: model.rstrip("/").split("/")[-1]
    monkeypatch.setitem(sys.modules, "fit_lens", stub)


@pytest.fixture
def driver(tmp_path, monkeypatch):
    from conftest import load_driver

    module = load_driver("t17_envelope")
    monkeypatch.setattr(module.cfg, "artifact_root", tmp_path, raising=False)
    return module


def _write_prompt_result(driver, slug, prompt, noise):
    path = driver.per_prompt_path(slug, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"noise_rms": {str(k): v for k, v in noise.items()}}))
    return path


# --- step 8: across prompts -------------------------------------------------

def test_prompts_combine_in_quadrature_not_arithmetically(driver):
    """These are noise amplitudes; the arithmetic mean under-states a noisy prompt."""
    combined = driver.average_across_prompts({0: {0: 3.0}, 1: {0: 4.0}})
    assert combined[0] == pytest.approx(12.5 ** 0.5)  # rms, not 3.5


def test_a_layer_missing_from_one_prompt_still_averages(driver):
    combined = driver.average_across_prompts({0: {0: 1.0, 1: 1.0}, 1: {0: 1.0}})
    assert sorted(combined) == [0, 1]
    assert combined[1] == pytest.approx(1.0)


# --- step 9: the division ---------------------------------------------------

def test_envelope_divides_by_n_to_the_alpha(driver):
    predicted = driver.predict({0: 1.0}, n=100, alpha=0.5)
    assert predicted[0] == pytest.approx(0.1)


def test_a_higher_alpha_predicts_a_smaller_envelope(driver):
    """The direction the caveat turns on: too high an alpha under-predicts."""
    low = driver.predict({0: 1.0}, n=417, alpha=0.45)[0]
    high = driver.predict({0: 1.0}, n=417, alpha=0.50)[0]
    assert high < low


# --- the guards -------------------------------------------------------------

def test_nothing_to_aggregate_refuses_rather_than_defaulting(driver, monkeypatch):
    monkeypatch.setattr("sys.argv", ["t17_envelope.py", "--n-prompts", "417",
                                     "--aggregate-only"])
    with pytest.raises(SystemExit, match="no per-prompt noise"):
        driver.main()


def test_a_result_with_no_noise_rms_is_skipped_not_read_as_zero(driver):
    slug = "Qwen3.5-4B"
    path = driver.per_prompt_path(slug, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"noise_rms": {}}))
    assert driver.collect(slug, [0]) == {}


def test_the_out_of_range_caveat_reaches_the_json(driver, monkeypatch, tmp_path):
    slug = "Qwen3.5-4B"
    for prompt in range(4):
        _write_prompt_result(driver, slug, prompt, {0: 1e-3, 1: 5e-4})

    monkeypatch.setattr("sys.argv", [
        "t17_envelope.py", "--model", "Qwen/Qwen3.5-4B", "--n-prompts", "417",
        "--prompts", "0,1,2,3", "--aggregate-only"])
    driver.main()

    blob = json.loads((tmp_path / "measurements" / "t17" /
                       f"envelope_{slug}.json").read_text())
    assert blob["alpha_in_range"] is False, "417 is outside the measured n=5..233"
    assert "UNDER-predicted" in blob["caveat"]
    assert blob["per_layer_bracket"], "a verdict needs the range, not the point value"
    assert blob["prompts_used"] == [0, 1, 2, 3]


def test_the_emitted_envelope_is_what_t16_compare_consumes(driver, monkeypatch, tmp_path):
    """The contract between the two drivers, tested rather than assumed.

    T17 writes the floor that T16's axes 1 and 2 are scored against. They are separate
    scripts that never import each other, so nothing but this test says the file one
    writes is the file the other reads -- and a mismatch would surface on the box, at
    the end of a ~2.5 h measurement, as an argument-parsing error.
    """
    from conftest import load_driver

    slug = "Qwen3.5-4B"
    for prompt in range(4):
        _write_prompt_result(driver, slug, prompt, {0: 1e-3, 1: 5e-4})
    monkeypatch.setattr("sys.argv", [
        "t17_envelope.py", "--model", "Qwen/Qwen3.5-4B", "--n-prompts", "417",
        "--prompts", "0,1,2,3", "--aggregate-only"])
    driver.main()
    emitted = tmp_path / "measurements" / "t17" / f"envelope_{slug}.json"

    compare = load_driver("t16_compare")
    envelope, provenance = compare.load_envelope(emitted)

    assert envelope == {0: pytest.approx(5.7634e-05, rel=1e-3),
                        1: pytest.approx(2.8817e-05, rel=1e-3)}
    assert provenance["kind"] == "predicted"
    # The caveat has to survive the hand-off, or the verdict loses it.
    assert "UNDER-predicted" in provenance["caveat"]
    assert provenance["alpha"] == 0.473
    assert provenance["n"] == 417


def test_in_range_use_says_so(driver, monkeypatch, tmp_path):
    slug = "Qwen3.5-0.8B"
    for prompt in range(4):
        _write_prompt_result(driver, slug, prompt, {0: 1e-3})
    monkeypatch.setattr("sys.argv", [
        "t17_envelope.py", "--n-prompts", "233", "--prompts", "0,1,2,3",
        "--aggregate-only"])
    driver.main()
    blob = json.loads((tmp_path / "measurements" / "t17" /
                       f"envelope_{slug}.json").read_text())
    assert blob["alpha_in_range"] is True
    assert "UNDER-predicted" not in blob["caveat"]


# --- the timeout that broke the first 4B run --------------------------------

def test_child_timeout_covers_the_slowest_draw_at_every_known_model():
    """childproc's 300 s default is a 0.8B number and killed every 4B gate reference.

    The gate reference is uncompiled -- the slowest draw type -- and runs first, so a
    too-short limit takes out the whole measurement before a single gated draw. This
    asserts the derived limit clears the cost table's own estimate with headroom, at
    every model in it, which is what stops the same bug arriving again at 27B.
    """
    from conftest import load_driver

    offset = load_driver("offset_profile")
    for model, costs in offset.COST_S.items():
        for policy, expected_s in costs.items():
            limit = offset.child_timeout_s(model, policy)
            assert limit > expected_s, f"{model}/{policy}: {limit}s <= {expected_s}s"
            assert limit >= offset.TIMEOUT_FACTOR * expected_s or \
                limit == offset.TIMEOUT_FLOOR_S


def test_the_uncompiled_gate_reference_is_the_binding_case():
    """It is the slowest policy, so its limit must be the largest -- and > 300 s."""
    from conftest import load_driver

    offset = load_driver("offset_profile")
    for model in offset.COST_S:
        limit = offset.child_timeout_s(model, "none")
        assert limit == max(offset.child_timeout_s(model, p) for p in offset.COST_S[model])
        assert limit > 300, "the childproc default that broke the first 4B run"


def test_an_unknown_model_still_gets_a_usable_limit():
    from conftest import load_driver

    offset = load_driver("offset_profile")
    assert offset.child_timeout_s("Qwen/Qwen3.5-27B", "none") >= offset.TIMEOUT_FLOOR_S


# --- the prompt-4 shape: a group that is uniformly wrong --------------------

def test_blind_spot_ratio_separates_the_offsets_actually_measured():
    """Sound variant offsets sat at 0.95x and 3.3x the noise; 4B/p4 sat at 65x.

    The threshold has to fall in that gap, and this pins it there so a later tweak
    cannot quietly move it past either the real signal or the real nulls.
    """
    from conftest import load_driver

    offset = load_driver("offset_profile")
    sound_ratios = (0.95, 3.3)          # 4B prompts 0 and 3
    suspect_ratio = 65.0                # 4B prompt 4
    assert max(sound_ratios) < offset.BLIND_SPOT_RATIO < suspect_ratio


def test_variant_readings_print_enough_digits_to_show_a_split(capsys):
    """At 6 dp, two variants split in their last bits print the same number.

    That made the 4B prompt-3 and prompt-4 output look like a clustering bug --
    identical values in two groups -- when grouping is exact and the split is the
    benign one cluster_variants documents.
    """
    from conftest import load_driver

    offset = load_driver("offset_profile")
    draws = [{"identity_distance": 0.5549601, "compile_layers": "linear-attn",
              "dim_batch": 8, "prompt_idx": 4},
             {"identity_distance": 0.5549604, "compile_layers": "linear-attn",
              "dim_batch": 8, "prompt_idx": 4}]
    offset.assign_variants(draws)
    printed = capsys.readouterr().out
    assert "2 variant(s)" in printed
    assert "0.5549601" in printed and "0.5549604" in printed, \
        f"the two readings must be distinguishable on screen; got: {printed.strip()}"
