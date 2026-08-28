"""Tests for the compile policy and the identity gate.

Both exist because torch.compile miscompiles this model in 30-50% of processes and the
resulting lens looks entirely valid -- right shape, right dtype, plausible size. The
policy narrows the risk to a measured 0/20; the gate catches what is left.

The block-kind rule is the load-bearing part: it must pick the linear-attention blocks
on a hybrid model and everything on a homogeneous one, because the second case is what
happens on the next model we fit and a rule that silently compiled nothing there would
cost 1.85x throughput without anyone noticing.
"""

import pytest

from jlens_extensions.compile_policy import (
    DEFAULT_POLICY,
    IDENTITY_TOL,
    CompileGateError,
    check_identity_gate,
    classify_blocks,
    identity_in_band,
    select_block_indices,
)


class Blk:
    """A residual block; hybrid models mark state-space blocks with .linear_attn."""

    def __init__(self, linear: bool):
        if linear:
            self.linear_attn = object()


def hybrid(n=24, interval=4):
    """Qwen3.5's shape: full attention every `interval`-th block, rest linear."""
    return [Blk(linear=(i % interval != interval - 1)) for i in range(n)]


def homogeneous(n=24):
    return [Blk(linear=False) for i in range(n)]


# --- block classification ---------------------------------------------------


def test_classify_splits_a_hybrid_stack():
    kinds = classify_blocks(hybrid())
    assert len(kinds["linear-attn"]) == 18
    assert len(kinds["full-attn"]) == 6
    assert set(kinds["linear-attn"]) & set(kinds["full-attn"]) == set()


def test_classify_a_homogeneous_stack_has_no_linear_blocks():
    kinds = classify_blocks(homogeneous())
    assert kinds["linear-attn"] == []
    assert len(kinds["full-attn"]) == 24


# --- the auto rule ----------------------------------------------------------


def test_auto_compiles_only_linear_attn_on_a_hybrid_model():
    """The measured-safe subset: 0/20 failures, at full-compile speed."""
    chosen = select_block_indices(hybrid(), "auto")
    assert len(chosen) == 18
    assert chosen == classify_blocks(hybrid())["linear-attn"]


def test_auto_compiles_everything_on_a_homogeneous_model():
    """The case that matters for the next model. A rule that compiled nothing here
    would silently cost 1.85x and look like it was working."""
    assert select_block_indices(homogeneous(), "auto") == list(range(24))


def test_auto_is_the_default():
    assert DEFAULT_POLICY == "auto"
    assert select_block_indices(hybrid()) == select_block_indices(hybrid(), "auto")


# --- explicit policies ------------------------------------------------------


@pytest.mark.parametrize("policy,expected", [("all", 24), ("none", 0),
                                             ("linear-attn", 18), ("full-attn", 6)])
def test_explicit_policies_select_the_right_count(policy, expected):
    assert len(select_block_indices(hybrid(), policy)) == expected


def test_unknown_policy_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown compile policy"):
        select_block_indices(hybrid(), "linear")


def test_full_attn_policy_on_a_homogeneous_model_selects_everything():
    """Not a special case in the code, but worth pinning: with no linear blocks,
    'full-attn' and 'all' coincide."""
    assert select_block_indices(homogeneous(), "full-attn") == list(range(24))


# --- the gate ---------------------------------------------------------------


def test_sound_values_pass_the_gate():
    """Every clean value observed across four compile configurations."""
    for observed in (0.531268, 0.531295, 0.531351, 0.531430, 0.531469, 0.531523):
        check_identity_gate(observed, 0.5314)


def test_every_observed_failure_is_caught():
    """Subtle (2.3% off), intermediate, and catastrophic modes."""
    for observed in (0.543330, 0.543808, 0.827450, 0.899078, 5.109395, 8.410543):
        with pytest.raises(CompileGateError):
            check_identity_gate(observed, 0.5314)


def test_the_bands_do_not_touch():
    """The gate is only usable because there is no near-miss: the worst sound value
    is far inside the tolerance and the mildest failure is well outside it."""
    worst_sound = max(abs(v - 0.5314) / 0.5314
                      for v in (0.531268, 0.531295, 0.531351, 0.531430, 0.531469, 0.531523))
    mildest_failure = min(abs(v - 0.5314) / 0.5314 for v in (0.543330, 0.543808))
    assert worst_sound < IDENTITY_TOL < mildest_failure
    assert mildest_failure / worst_sound > 40


def test_gate_error_names_the_numbers_and_the_way_out():
    with pytest.raises(CompileGateError) as exc:
        check_identity_gate(8.4105, 0.5314, context="run t15-a")
    msg = str(exc.value)
    assert "8.410" in msg and "0.531" in msg
    assert "run t15-a" in msg
    assert "--compile_blocks none" in msg
    assert "per-process" in msg


def test_nan_is_a_failure_not_a_pass():
    """NaN comparisons are false, so a naive threshold would let it through."""
    assert not identity_in_band(float("nan"), 0.5314)
    with pytest.raises(CompileGateError):
        check_identity_gate(float("nan"), 0.5314)


def test_tolerance_is_relative_not_absolute():
    assert identity_in_band(1.005, 1.0, tol=0.01)
    assert not identity_in_band(1.02, 1.0, tol=0.01)
    assert identity_in_band(100.5, 100.0, tol=0.01)


def test_zero_expected_is_rejected_rather_than_dividing():
    with pytest.raises(ValueError, match="non-zero"):
        identity_in_band(0.1, 0.0)


# --- the gate as fit_lens.py actually uses it -------------------------------


@pytest.fixture
def tracker_cls():
    """Import the vendored driver the way it imports itself."""
    import importlib
    import sys
    from pathlib import Path

    pytest.importorskip("torch")
    harness = Path(__file__).resolve().parents[1] / "harness"
    sys.path.insert(0, str(harness))
    pre = set(sys.modules)
    try:
        yield importlib.import_module("fit_lens").ConvergenceTracker
    finally:
        sys.path.remove(str(harness))
        for name in set(sys.modules) - pre:
            if name == "fit_lens" or name.startswith("jlens"):
                del sys.modules[name]
        importlib.invalidate_caches()


def _progress(identity, n_done=1):
    from types import SimpleNamespace

    return SimpleNamespace(n_done=n_done, prompt_idx=n_done - 1, seq_len=128,
                           n_valid_positions=111, elapsed_s=15.5,
                           identity_distance=identity, mean_rel_change=float("nan"))


def test_tracker_aborts_the_fit_on_a_miscompiled_first_prompt(tracker_cls, tmp_path):
    """The whole point: an hour is not spent, and no lens reaches disk."""
    t = tracker_cls(str(tmp_path / "c.csv"), (1e-2,), gate_identity=0.5314)
    with pytest.raises(CompileGateError):
        t.record(_progress(8.410543))
    t.close()


def test_tracker_passes_a_sound_first_prompt(tracker_cls, tmp_path):
    t = tracker_cls(str(tmp_path / "c.csv"), (1e-2,), gate_identity=0.5314)
    assert t.record(_progress(0.531268)) is False
    t.close()


def test_the_gate_only_fires_at_prompt_1(tracker_cls, tmp_path):
    """Later prompts move legitimately as the running mean evolves -- the envelope
    run saw 0.5313 rise past 0.542 by prompt 10 -- so gating them would be wrong."""
    t = tracker_cls(str(tmp_path / "c.csv"), (1e-2,), gate_identity=0.5314)
    t.record(_progress(0.531268, n_done=1))
    assert t.record(_progress(0.556, n_done=7)) is False
    t.close()


def test_no_gate_configured_means_no_check(tracker_cls, tmp_path):
    """Default behaviour is unchanged: the vendored driver gated nothing."""
    t = tracker_cls(str(tmp_path / "c.csv"), (1e-2,))
    assert t.record(_progress(8.410543)) is False
    t.close()
