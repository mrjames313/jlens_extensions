"""Tests for the activation-dependent signatures S1, S2 and S3.

Like the corpus-free pair, these have no reference implementation and the paper's evidence
is a set of qualitative shapes, so each statistic gets a case whose answer is known before
the code runs — several of them exactly, from closed-form kurtosis of standard
distributions and from hand-computed arithmetic.

The load-bearing tests are:

* :func:`test_a_position_independent_readout_scores_exactly_zero` — S3's null is
  calibrated. If the lens says the same thing everywhere, there is no persistence to
  detect, and the statistic must return exactly 0 rather than something small.
* :func:`test_the_null_leaves_out_both_ends` — that the two exclusions are real.
* :func:`test_moments_path_matches_the_direct_path` — S2's cache summary against the
  full-vector computation, since T8 will only ever store the summary.
* :func:`test_raw_power_sums_lose_the_answer_in_fp32` — why that summary is central
  moments rather than the power sums that would be cheaper to accumulate.
"""

import pytest

torch = pytest.importorskip("torch")

from jlens_extensions.activation_signatures import (  # noqa: E402
    central_moments,
    excess_kurtosis,
    excess_kurtosis_from_moments,
    kurtosis_curve,
    mark_transition,
    top1_autocorrelation,
    topk_accuracy,
)


# --- S1: top-k next-token accuracy ------------------------------------------------


def test_topk_accuracy_on_a_hand_countable_case():
    # 1 layer, 4 positions, top-3 lens tokens each.
    lens_topk = torch.tensor([[[5, 1, 2], [7, 8, 9], [3, 4, 5], [0, 6, 7]]])
    model_top1 = torch.tensor([5, 9, 9, 6])
    # k=1: position 0 only            -> 1/4
    # k=2: positions 0, 3             -> 2/4
    # k=3: positions 0, 1, 3          -> 3/4
    acc = topk_accuracy(lens_topk, model_top1, ks=(1, 2, 3))
    assert acc.shape == (3, 1)
    assert float(acc[0, 0]) == pytest.approx(0.25)
    assert float(acc[1, 0]) == pytest.approx(0.50)
    assert float(acc[2, 0]) == pytest.approx(0.75)


def test_topk_accuracy_is_one_when_the_lens_leads_with_the_model_top1():
    g = torch.Generator().manual_seed(0)
    model_top1 = torch.randint(0, 500, (32,), generator=g)
    lens_topk = torch.stack([torch.stack([model_top1, model_top1 + 1000], dim=1)] * 3)
    acc = topk_accuracy(lens_topk, model_top1, ks=(1,))
    assert torch.allclose(acc, torch.ones(1, 3, dtype=torch.float64))


def test_topk_accuracy_is_zero_when_the_sets_are_disjoint():
    model_top1 = torch.arange(16)
    lens_topk = torch.full((2, 16, 4), 9999)
    acc = topk_accuracy(lens_topk, model_top1, ks=(1, 4))
    assert torch.allclose(acc, torch.zeros(2, 2, dtype=torch.float64))


def test_rank_order_of_the_topk_axis_is_load_bearing():
    """A hit at rank 5 counts at k=5 and not at k=4; an unsorted axis breaks that."""
    lens_topk = torch.tensor([[[0, 1, 2, 3, 42]]])
    model_top1 = torch.tensor([42])
    acc = topk_accuracy(lens_topk, model_top1, ks=(4, 5))
    assert float(acc[0, 0]) == 0.0
    assert float(acc[1, 0]) == 1.0


def test_topk_accuracy_is_non_decreasing_in_k():
    g = torch.Generator().manual_seed(1)
    lens_topk = torch.randint(0, 50, (4, 200, 10), generator=g)
    model_top1 = torch.randint(0, 50, (200,), generator=g)
    acc = topk_accuracy(lens_topk, model_top1, ks=(1, 2, 5, 10))
    assert bool(((acc[1:] - acc[:-1]) >= 0).all())


def test_topk_accuracy_honours_the_valid_mask():
    lens_topk = torch.tensor([[[1], [2], [3], [4]]])
    model_top1 = torch.tensor([1, 2, 99, 99])
    assert float(topk_accuracy(lens_topk, model_top1, ks=(1,))[0, 0]) == pytest.approx(0.5)
    mask = torch.tensor([True, True, False, False])
    scored = topk_accuracy(lens_topk, model_top1, ks=(1,), valid=mask)
    assert float(scored[0, 0]) == pytest.approx(1.0)


def test_topk_accuracy_rejects_bad_arguments():
    lens_topk = torch.zeros(2, 5, 3, dtype=torch.int64)
    model_top1 = torch.zeros(5, dtype=torch.int64)
    with pytest.raises(ValueError, match="ascending"):
        topk_accuracy(lens_topk, model_top1, ks=(3, 1))
    with pytest.raises(ValueError, match=r"lie in \[1, 3\]"):
        topk_accuracy(lens_topk, model_top1, ks=(4,))
    with pytest.raises(ValueError, match="at least one k"):
        topk_accuracy(lens_topk, model_top1, ks=())
    with pytest.raises(ValueError, match="model_top1 must be"):
        topk_accuracy(lens_topk, torch.zeros(4, dtype=torch.int64), ks=(1,))


# --- S2: excess kurtosis ----------------------------------------------------------


def test_gaussian_has_zero_excess_kurtosis():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(400_000, generator=g, dtype=torch.float64)
    assert float(excess_kurtosis(x)) == pytest.approx(0.0, abs=0.05)


def test_two_point_distribution_is_exactly_minus_two():
    """A symmetric two-point distribution has m4 = m2**2, so excess kurtosis = -2."""
    x = torch.tensor([-1.0, 1.0] * 50, dtype=torch.float64)
    assert float(excess_kurtosis(x)) == pytest.approx(-2.0)


def test_discrete_uniform_matches_its_closed_form():
    """Excess kurtosis of the uniform distribution on {1..n} is -6(n**2+1)/(5(n**2-1))."""
    for n in (5, 12, 101):
        x = torch.arange(1, n + 1, dtype=torch.float64)
        expected = -6.0 * (n**2 + 1) / (5.0 * (n**2 - 1))
        assert float(excess_kurtosis(x)) == pytest.approx(expected)


def test_a_readout_peaked_on_a_few_tokens_has_high_kurtosis():
    """S2's actual subject: a readout concentrated on a handful of the vocabulary."""
    flat = torch.zeros(10_000, dtype=torch.float64)
    peaked = flat.clone()
    peaked[:5] = 50.0
    assert float(excess_kurtosis(peaked)) > 100.0
    diffuse = flat.clone()
    diffuse[:2000] = 50.0
    assert float(excess_kurtosis(diffuse)) < float(excess_kurtosis(peaked))


def test_excess_kurtosis_is_invariant_to_shift_and_scale():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(5000, generator=g, dtype=torch.float64)
    base = float(excess_kurtosis(x))
    assert float(excess_kurtosis(x * 7.0 - 300.0)) == pytest.approx(base)


def test_moments_path_matches_the_direct_path():
    """T8 stores the summary, never the vector; the two must not diverge."""
    g = torch.Generator().manual_seed(4)
    x = torch.randn(6, 40, 2000, generator=g, dtype=torch.float64)
    m2, m4 = central_moments(x)
    assert torch.allclose(excess_kurtosis_from_moments(m2, m4), excess_kurtosis(x))


def test_raw_power_sums_lose_the_answer_in_fp32():
    """Why the cache stores central moments and not the cheaper raw power sums.

    Logits sit far from zero, so the fourth raw moment is enormous next to the central
    one it is differenced down to, and fp32 cannot survive the cancellation.
    """
    g = torch.Generator().manual_seed(5)
    x64 = torch.randn(20_000, generator=g, dtype=torch.float64) + 1000.0
    truth = float(excess_kurtosis(x64))

    x32 = x64.to(torch.float32)
    m2, m4 = central_moments(x32)
    central_fp32 = float(excess_kurtosis_from_moments(m2, m4))

    mean = x32.mean()
    r2, r3, r4 = (x32.pow(p).mean() for p in (2, 3, 4))
    raw_m2 = r2 - mean.pow(2)
    raw_m4 = r4 - 4 * mean * r3 + 6 * mean.pow(2) * r2 - 3 * mean.pow(4)
    raw_fp32 = float(raw_m4 / raw_m2.pow(2) - 3.0)

    assert central_fp32 == pytest.approx(truth, abs=0.1)
    assert abs(raw_fp32 - truth) > 1.0


def test_a_degenerate_readout_reports_nan_rather_than_a_number():
    constant = torch.full((100,), 4.0, dtype=torch.float64)
    assert bool(torch.isnan(excess_kurtosis(constant)))


def test_kurtosis_curve_aggregates_and_counts():
    m2 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float64)
    m4 = torch.tensor([[3.0, 9.0, 27.0], [3.0, 3.0, 5.0]], dtype=torch.float64)
    moments = torch.stack([m2, m4], dim=-1)
    curve = kurtosis_curve(moments, layers=[0, 1])
    # Layer 0: excess kurtosis 0, 6, 24 -> mean 10, median 6, all three scored.
    assert float(curve.mean[0]) == pytest.approx(10.0)
    assert float(curve.median[0]) == pytest.approx(6.0)
    assert int(curve.n_scored[0]) == 3
    # Layer 1: the m2 == 0 readout is degenerate and must not average in.
    assert float(curve.mean[1]) == pytest.approx(0.0)
    assert int(curve.n_scored[1]) == 2
    assert curve.layers == [0, 1]


def test_kurtosis_curve_honours_the_valid_mask():
    m2 = torch.ones(1, 4, dtype=torch.float64)
    m4 = torch.tensor([[3.0, 3.0, 103.0, 103.0]], dtype=torch.float64)
    moments = torch.stack([m2, m4], dim=-1)
    unmasked = kurtosis_curve(moments, [0])
    masked = kurtosis_curve(moments, [0], valid=torch.tensor([True, True, False, False]))
    assert float(unmasked.mean[0]) == pytest.approx(50.0)
    assert float(masked.mean[0]) == pytest.approx(0.0)
    assert int(masked.n_scored[0]) == 2


def test_kurtosis_curve_rejects_a_bad_moment_shape():
    with pytest.raises(ValueError, match=r"\(m2, m4\)"):
        kurtosis_curve(torch.zeros(2, 5, 4), [0, 1])


# --- S3: top-1 autocorrelation ----------------------------------------------------


def test_top1_autocorrelation_on_a_hand_computed_case():
    # 1 layer, 4 positions, 2 candidates; top-1 is candidate 0 everywhere.
    a = [-4.0, -3.0, -2.0, -1.0]
    lp = torch.tensor([[[a[i], 0.0] for i in range(4)]], dtype=torch.float64)
    top1 = torch.zeros(1, 4, dtype=torch.int64)
    # delta=1, leave-two-out null over 4 valid positions:
    #   t=0: a1 - (a2+a3)/2 = -3 - (-1.5) = -1.5
    #   t=1: a2 - (a0+a3)/2 = -2 - (-2.5) =  0.5
    #   t=2: a3 - (a0+a1)/2 = -1 - (-3.5) =  2.5
    #   mean = 0.5
    out = top1_autocorrelation(lp, top1, deltas=(1,))
    assert out.shape == (1, 1)
    assert float(out[0, 0]) == pytest.approx(0.5)


def test_a_position_independent_readout_scores_exactly_zero():
    """The calibration test. If the lens says the same thing at every position there is
    no persistence to find, and the null must cancel the signal exactly — not
    approximately, and not with a sign."""
    g = torch.Generator().manual_seed(6)
    per_candidate = torch.randn(3, 1, 20, generator=g, dtype=torch.float64)
    lp = per_candidate.expand(3, 50, 20).contiguous()
    top1 = torch.randint(0, 20, (3, 50), generator=g)
    out = top1_autocorrelation(lp, top1, deltas=(1, 3, 7))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-12)


def test_persistent_content_scores_positive_and_transient_content_does_not():
    g = torch.Generator().manual_seed(7)
    n_pos, n_cand = 60, 12
    base = torch.randn(1, n_pos, n_cand, generator=g, dtype=torch.float64) * 0.1
    # Positions in the same block of 6 share a boosted candidate: content that persists
    # over short offsets and not long ones.
    persistent = base.clone()
    for t in range(n_pos):
        persistent[0, t, (t // 6) % n_cand] += 5.0
    top1 = persistent[0].argmax(dim=1).unsqueeze(0)
    out = top1_autocorrelation(persistent, top1, deltas=(1, 30))
    assert float(out[0, 0]) > 1.0  # within-block: content persists
    assert float(out[0, 0]) > float(out[1, 0])  # and decays with distance


def test_the_null_leaves_out_both_ends():
    """Both exclusions change the answer, so neither is decoration.

    ``t`` is excluded because the token was chosen as the argmax there; ``t + Δ`` because
    it is the quantity being compared against.
    """
    # Deliberately not an arithmetic progression: on an evenly spaced sequence the
    # leave-two-out and full-mean nulls happen to average to the same number across t,
    # and this test passes while asserting nothing.
    a = [-8.0, -3.0, -2.0, -1.0]
    lp = torch.tensor([[[a[i], 0.0] for i in range(4)]], dtype=torch.float64)
    top1 = torch.zeros(1, 4, dtype=torch.int64)
    ours = float(top1_autocorrelation(lp, top1, deltas=(1,))[0, 0])
    assert ours == pytest.approx(5.5 / 3)  # hand-computed with both ends left out

    col = torch.tensor(a, dtype=torch.float64)
    null_over_everything = float(torch.stack([col[t + 1] - col.mean() for t in range(3)]).mean())
    null_dropping_only_t = float(
        torch.stack(
            [col[t + 1] - (col.sum() - col[t]) / 3 for t in range(3)]
        ).mean()
    )
    assert ours != pytest.approx(null_over_everything)
    assert ours != pytest.approx(null_dropping_only_t)


def test_top1_autocorrelation_handles_deltas_beyond_the_corpus():
    lp = torch.randn(1, 5, 3, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    top1 = torch.zeros(1, 5, dtype=torch.int64)
    out = top1_autocorrelation(lp, top1, deltas=(1, 9))
    assert torch.isfinite(out[0, 0])
    assert bool(torch.isnan(out[1, 0]))


def test_top1_autocorrelation_honours_the_valid_mask():
    g = torch.Generator().manual_seed(9)
    lp = torch.randn(2, 30, 6, generator=g, dtype=torch.float64)
    top1 = torch.randint(0, 6, (2, 30), generator=g)
    mask = torch.ones(30, dtype=torch.bool)
    mask[:10] = False
    assert not torch.allclose(
        top1_autocorrelation(lp, top1, deltas=(2,)),
        top1_autocorrelation(lp, top1, deltas=(2,), valid=mask),
    )


def test_top1_autocorrelation_rejects_bad_arguments():
    lp = torch.zeros(1, 6, 3, dtype=torch.float64)
    top1 = torch.zeros(1, 6, dtype=torch.int64)
    with pytest.raises(ValueError, match="index the candidate axis"):
        top1_autocorrelation(lp, torch.full((1, 6), 3), deltas=(1,))
    with pytest.raises(ValueError, match="ascending"):
        top1_autocorrelation(lp, top1, deltas=(3, 1))
    with pytest.raises(ValueError, match="positive"):
        top1_autocorrelation(lp, top1, deltas=(0,))
    with pytest.raises(ValueError, match="disagreeing about which positions"):
        top1_autocorrelation(lp, torch.zeros(1, 5, dtype=torch.int64), deltas=(1,))
    with pytest.raises(ValueError, match="leave two out"):
        top1_autocorrelation(
            lp, top1, deltas=(1,), valid=torch.tensor([True, True, False, False, False, False])
        )


# --- Marking a transition ---------------------------------------------------------


def test_a_step_curve_puts_both_marks_at_the_step():
    curve = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    layers = [10, 11, 12, 13, 14, 15]
    t = mark_transition(curve, layers)
    assert t.crossing_layer == 13
    assert t.steepest_layer == 13
    assert t.steepest_rise == pytest.approx(1.0)


def test_the_two_marks_disagree_on_a_drift_then_jump_curve():
    """The case the pair exists for: an early crossing and a late steep rise."""
    curve = torch.tensor([0.0, 0.3, 0.45, 0.5, 0.55, 2.0], dtype=torch.float64)
    layers = list(range(6))
    # Range is [0, 2], so fraction 0.2 puts the threshold at 0.4 — first reached at
    # layer 2, while the steep rise is at layer 5. Which of those is "the onset" is the
    # judgement T10 has to make; the pair exists so that it is made explicitly.
    t = mark_transition(curve, layers, fraction=0.2)
    assert t.crossing_layer == 2
    assert t.steepest_layer == 5
    assert t.steepest_rise == pytest.approx(1.45)


def test_the_threshold_fraction_moves_the_crossing():
    curve = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    layers = list(range(5))
    assert mark_transition(curve, layers, fraction=0.2).crossing_layer == 1
    assert mark_transition(curve, layers, fraction=0.9).crossing_layer == 4


def test_a_flat_curve_reports_no_transition():
    curve = torch.full((5,), 2.0, dtype=torch.float64)
    t = mark_transition(curve, [3, 4, 5, 6, 7])
    assert t.crossing_layer == 3
    assert t.steepest_rise == 0.0


def test_mark_transition_rejects_non_finite_and_bad_shapes():
    with pytest.raises(ValueError, match="non-finite"):
        mark_transition(torch.tensor([0.0, float("nan"), 1.0]), [0, 1, 2])
    with pytest.raises(ValueError, match="must be 1-D"):
        mark_transition(torch.zeros(2, 3), [0, 1])
    with pytest.raises(ValueError, match="names"):
        mark_transition(torch.zeros(3), [0, 1])
    with pytest.raises(ValueError, match=r"fraction must lie in \(0, 1\)"):
        mark_transition(torch.tensor([0.0, 1.0]), [0, 1], fraction=1.0)
