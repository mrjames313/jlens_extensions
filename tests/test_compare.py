"""Tests for the lens comparison primitives.

The load-bearing property is the one in the module docstring: **fp16 quantisation is
shared, not an independent noise draw.** Two tensors closer than one quantum round to
the same fp16 value, so an fp16 comparison erases sub-quantum differences rather than
adding noise. That is why T18's envelope (measured on fp16-stored lenses) and T15's
a-vs-b (fp32) are not directly comparable, and why `as_fp16=True` exists.

Numerics are float64/float32 with exact constructions, so the assertions are exact
rather than statistical.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from jlens_extensions.compare import (  # noqa: E402
    FP16_EPS,
    FP16_FLOOR,
    binding_constraint,
    compare_lenses,
    fp16_roundtrip,
    fit_power_law,
    fp16_storage_floor,
    identity_distance,
    identity_fraction,
    layer_correspondence,
    rel_frobenius,
    replay_early_stop,
)

D = 8


def _lens(**layers):
    return {int(k[1:]): v for k, v in layers.items()}


# --- relative Frobenius -----------------------------------------------------


def test_rel_frobenius_is_zero_for_identical_tensors():
    a = torch.randn(D, D)
    assert rel_frobenius(a, a.clone()) == 0.0


def test_rel_frobenius_normalises_by_the_second_argument():
    a, b = torch.zeros(D, D), torch.zeros(D, D)
    b[0, 0] = 2.0
    a[0, 0] = 3.0
    # ||a-b|| = 1, ||b|| = 2
    assert rel_frobenius(a, b) == pytest.approx(0.5)


def test_rel_frobenius_rejects_a_zero_reference():
    with pytest.raises(ValueError, match="zero Frobenius norm"):
        rel_frobenius(torch.ones(D, D), torch.zeros(D, D))


# --- the fp16 trap this module exists for -----------------------------------


def test_fp16_roundtrip_erases_a_subquantum_difference():
    """The central claim: fp16 storage does not add noise between two close
    tensors, it removes their difference entirely."""
    a = torch.full((D, D), 1.0)
    b = torch.full((D, D), 1.0 + FP16_EPS / 8)  # far inside one quantum

    assert rel_frobenius(a, b) > 0.0
    assert rel_frobenius(fp16_roundtrip(a), fp16_roundtrip(b)) == 0.0


def test_fp16_roundtrip_preserves_a_suprquantum_difference():
    a = torch.full((D, D), 1.0)
    b = torch.full((D, D), 1.5)

    assert rel_frobenius(fp16_roundtrip(a), fp16_roundtrip(b)) == pytest.approx(
        rel_frobenius(a, b), rel=1e-3
    )


def test_fp16_max_abs_difference_lands_on_a_power_of_two():
    """T18's table reports max_abs = 1.953e-3 = 2**-9 exactly, which is what
    identified its lenses as fp16-stored. Reproduce that signature."""
    a = torch.full((1, 1), 3.0)
    b = torch.full((1, 1), 3.0 + 2.0**-9)  # one ULP apart in [2, 4)

    diff = compare_lenses(_lens(l0=a), _lens(l0=b), as_fp16=True)[0]

    assert diff.max_abs == pytest.approx(2.0**-9)
    assert math.log2(diff.max_abs) == pytest.approx(-9.0)


def test_as_fp16_inflates_a_subquantum_difference():
    """The corrected directional claim, and the reason `as_fp16` exists.

    Naively, fp16 should erase a sub-quantum difference. Most entries do collapse to
    zero -- but entries straddling a rounding boundary land a *full quantum* apart,
    which for a sub-quantum difference is an amplification, and those dominate the
    Frobenius norm. Net effect: the measured difference is inflated, not truncated.

    This is why reading an fp32 a-vs-b against T18's fp16 numbers would be wrong in
    the dangerous direction -- it would look like compile had reduced the noise.

    The perturbation is *relative*, not a fixed absolute offset: fp16 is floating
    point, so the quantum scales with each element's magnitude.
    """
    torch.manual_seed(0)
    a = torch.randn(64, 64)
    b = a * (1.0 + FP16_EPS / 4)

    fp32 = compare_lenses(_lens(l0=a), _lens(l0=b))[0].rel_frobenius
    fp16 = compare_lenses(_lens(l0=a), _lens(l0=b), as_fp16=True)[0].rel_frobenius

    assert fp32 > 0.0
    assert fp16 > fp32


def test_as_fp16_is_faithful_well_above_the_quantum():
    """The other half: above ~2q the storage format stops mattering, which is why
    T18's L0 figure is trustworthy and its high-layer figures are not."""
    torch.manual_seed(0)
    a = torch.randn(64, 64)
    b = a * (1.0 + FP16_EPS * 8)

    fp32 = compare_lenses(_lens(l0=a), _lens(l0=b))[0].rel_frobenius
    fp16 = compare_lenses(_lens(l0=a), _lens(l0=b), as_fp16=True)[0].rel_frobenius

    assert fp16 == pytest.approx(fp32, rel=0.05)


def test_fp16_distortion_follows_a_geometric_mean_shape():
    """Quantifies the inflation: fp16-measured ~ sqrt(true * quantum) below the
    quantum. Establishes the direction and rough shape, not a correction factor --
    a Jacobian near the identity is not a random matrix.
    """
    torch.manual_seed(0)
    a = torch.randn(128, 128)
    delta = FP16_EPS / 16
    b = a * (1.0 + delta)

    fp32 = compare_lenses(_lens(l0=a), _lens(l0=b))[0].rel_frobenius
    fp16 = compare_lenses(_lens(l0=a), _lens(l0=b), as_fp16=True)[0].rel_frobenius

    assert fp16 == pytest.approx(math.sqrt(fp32 * FP16_EPS), rel=0.5)


# --- compare_lenses ---------------------------------------------------------


def test_compare_lenses_covers_every_layer_in_order():
    a = {0: torch.randn(D, D), 5: torch.randn(D, D), 22: torch.randn(D, D)}
    b = {k: v.clone() for k, v in a.items()}

    diffs = compare_lenses(a, b)

    assert [d.layer for d in diffs] == [0, 5, 22]
    assert all(d.rel_frobenius == 0.0 for d in diffs)


def test_compare_lenses_rejects_mismatched_layer_sets():
    """A silent misalignment would put L0's noise on L1's row."""
    with pytest.raises(ValueError, match="different layers"):
        compare_lenses({0: torch.eye(D)}, {0: torch.eye(D), 1: torch.eye(D)})


def test_compare_lenses_counts_differing_entries():
    a = torch.zeros(D, D)
    b = torch.zeros(D, D)
    b[0, 0] = 1.0
    b[1, 1] = 1.0

    diff = compare_lenses(_lens(l0=a), _lens(l0=b))[0]

    assert diff.n_differing == 2
    assert diff.n_entries == D * D
    assert diff.frac_differing == pytest.approx(2 / (D * D))


# --- identity_distance ------------------------------------------------------


def test_identity_distance_is_zero_on_the_identity():
    assert identity_distance({22: torch.eye(D)}) == 0.0


def test_identity_distance_defaults_to_the_highest_layer():
    """fitting.py uses late_layer = max(source_layers); reading a different layer
    would produce a number that looks valid and compares against nothing."""
    lens = {0: torch.zeros(D, D), 22: torch.eye(D)}

    assert identity_distance(lens) == 0.0
    assert identity_distance(lens, layer=0) == pytest.approx(1.0)


def test_identity_distance_matches_the_harness_formula():
    J = torch.eye(D) * 2.0
    # ||2I - I||_F / sqrt(d) = ||I||_F / sqrt(d) = sqrt(d)/sqrt(d) = 1
    assert identity_distance({0: J}) == pytest.approx(1.0)


# --- early-stop replay ------------------------------------------------------


def test_replay_early_stop_fires_on_the_window_mean_not_a_full_window_of_low_values():
    """The rule tests the *mean* of the last 10, so it fires before the window has
    flushed. Here nine 0.001s and one leftover 0.01 average 0.0019, already under
    the 0.002 threshold -- one prompt earlier than a full-flush reading predicts.

    This is why the published run's stop at 233 is a smoothed crossing and cannot
    be reproduced by looking for ten consecutive low readings.
    """
    deltas = [float("nan")] + [0.01] * 120 + [0.001] * 20

    result = replay_early_stop(deltas)

    assert result["would_stop_at"] == 130
    assert result["smoothed_delta"] == pytest.approx(0.0019)


def test_replay_early_stop_respects_min_prompts():
    """Converged from the start, but the rule must not fire before 100."""
    deltas = [float("nan")] + [0.0001] * 200

    assert replay_early_stop(deltas)["would_stop_at"] == 100


def test_replay_early_stop_ignores_nan_in_the_window():
    """The first prompt's mean_rel_change is NaN; the tracker returns before
    appending it. A NaN inside the window would poison the mean forever."""
    deltas = [float("nan")] + [0.0001] * 120

    result = replay_early_stop(deltas)

    assert result["would_stop_at"] is not None
    assert not math.isnan(result["smoothed_delta"])


def test_replay_early_stop_reports_no_stop_and_the_final_window():
    deltas = [float("nan")] + [0.5] * 200

    result = replay_early_stop(deltas)

    assert result["would_stop_at"] is None
    assert result["smoothed_delta_final"] == pytest.approx(0.5)


def test_replay_early_stop_uses_a_smoothed_window_not_a_single_crossing():
    """One low reading inside a noisy run must not trip it -- the published rule
    averages 10, which is why 233 is a smoothed crossing rather than a spike."""
    deltas = [float("nan")] + [0.01] * 150 + [0.0001] + [0.01] * 20

    assert replay_early_stop(deltas)["would_stop_at"] is None


# --- binding constraint -----------------------------------------------------


def test_binding_constraint_picks_the_larger_floor():
    at_l0 = binding_constraint(1e-3, envelope=7.5e-4)
    assert at_l0["binds"] == "nondeterminism"
    assert at_l0["floor"] == pytest.approx(7.5e-4)

    at_l22 = binding_constraint(1e-3, envelope=2.6e-6)
    assert at_l22["binds"] == "fp16_storage"
    assert at_l22["floor"] == pytest.approx(FP16_FLOOR)


def test_binding_constraint_without_a_measured_envelope():
    result = binding_constraint(1e-3, envelope=None)
    assert result["binds"] == "fp16_storage"


def test_binding_constraint_flags_a_difference_under_the_floor():
    """The verdict this exists to support: at or under the floor is not
    disagreement, it is resolution."""
    assert binding_constraint(1e-4, envelope=None)["above_floor"] is False
    assert binding_constraint(1e-2, envelope=None)["above_floor"] is True


# --- measured storage floor -------------------------------------------------


def test_fp16_storage_floor_is_smaller_than_the_elementwise_bound():
    """The point of measuring it. 2**-11 is the worst-case *element-wise* relative
    error; a Frobenius aggregate over a million entries partially cancels, so the
    applicable floor is smaller -- and using the larger one flatters agreement."""
    torch.manual_seed(0)
    J = {0: torch.eye(64) + torch.randn(64, 64) * 0.5}

    floor = fp16_storage_floor(J)[0]

    assert 0.0 < floor < FP16_EPS


def test_fp16_storage_floor_is_zero_for_an_exactly_representable_tensor():
    """Powers of two survive fp16 exactly, so the floor is a property of the
    distribution rather than a constant."""
    assert fp16_storage_floor({0: torch.full((8, 8), 0.5)})[0] == 0.0


def test_fp16_storage_floor_covers_every_layer():
    torch.manual_seed(0)
    J = {l: torch.randn(16, 16) for l in (0, 5, 22)}
    assert sorted(fp16_storage_floor(J)) == [0, 5, 22]


# --- identity fraction ------------------------------------------------------


def test_identity_fraction_is_zero_for_the_identity_itself():
    assert identity_fraction({0: torch.eye(32)})[0] == pytest.approx(0.0)


def test_identity_fraction_reports_the_deviation_share():
    """A near-identity Jacobian carries most of its Frobenius norm in the diagonal;
    this is the number that says how much."""
    J = torch.eye(32) * 1.0
    J[0, 1] = 1.0  # one off-diagonal unit
    frac = identity_fraction({0: J})[0]
    # ||J - I|| = 1, ||J|| = sqrt(32 + 1)
    assert frac == pytest.approx(1.0 / math.sqrt(33))


# --- the negative control ---------------------------------------------------


def test_layer_correspondence_puts_the_diagonal_first_on_distinct_layers():
    """The control passing: each layer matches itself, by a wide margin."""
    torch.manual_seed(0)
    ours = {l: torch.randn(32, 32) for l in range(5)}
    ref = {l: v + torch.randn(32, 32) * 1e-4 for l, v in ours.items()}

    result = layer_correspondence(ours, ref)

    assert result["all_correct"]
    assert result["n_correct"] == 5
    assert result["min_margin_x"] > 100


def test_layer_correspondence_detects_a_shifted_reference():
    """The control failing, which is what makes it a control: a lens whose layers
    are off by one is flagged rather than passed."""
    torch.manual_seed(0)
    base = {l: torch.randn(32, 32) for l in range(5)}
    ours = {l: base[l] for l in range(4)}
    shifted = {l: base[l + 1] for l in range(4)}

    result = layer_correspondence(ours, shifted)

    assert not result["all_correct"]
    assert result["n_correct"] == 0


def test_layer_correspondence_would_flatten_on_near_identity_tensors():
    """The 'both tensors near-identity' failure the criteria name: if the metric
    were dominated by the identity diagonal, every layer would match every other
    equally well and the margin would collapse toward 1."""
    ours = {l: torch.eye(64) + torch.randn(64, 64) * 1e-6 for l in range(4)}
    ref = {l: torch.eye(64) + torch.randn(64, 64) * 1e-6 for l in range(4)}

    result = layer_correspondence(ours, ref)

    assert result["min_margin_x"] < 2.0, "margin should collapse when signal is absent"


def test_layer_correspondence_skips_mismatched_shapes():
    """A different model's lens has a different d_model, which is why the criterion's
    literal cross-model control is undefined at this shape."""
    ours = {0: torch.randn(32, 32)}
    other = {0: torch.randn(16, 16)}

    assert layer_correspondence(ours, other)["rows"] == []


# --- power-law fitting ------------------------------------------------------


def test_fit_power_law_recovers_a_known_exponent():
    points = [(n, 1e-3 * n ** -0.5) for n in (5, 10, 20, 40, 80)]
    fit = fit_power_law(points)
    assert fit["alpha"] == pytest.approx(0.5, abs=1e-9)
    assert fit["C"] == pytest.approx(1e-3, rel=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_fit_power_law_recovers_systematic_noise_as_alpha_zero():
    """Noise that does not average down at all is the alpha = 0 endpoint."""
    assert fit_power_law([(n, 3.06e-3) for n in (5, 20, 60)])["alpha"] == pytest.approx(0.0)


def test_fit_power_law_reports_a_poor_fit_rather_than_hiding_it():
    """A single alpha must not be quoted for an envelope that is not scaling
    cleanly, so r_squared is returned alongside it."""
    fit = fit_power_law([(5, 1e-3), (10, 9e-4), (20, 5e-3), (40, 1e-4)])
    assert fit["r_squared"] < 0.5


def test_fit_power_law_needs_two_positive_points():
    with pytest.raises(ValueError, match="at least two"):
        fit_power_law([(20, 1e-3)])
    with pytest.raises(ValueError, match="at least two"):
        fit_power_law([(20, 1e-3), (40, 0.0)])


def test_fit_power_law_rejects_a_degenerate_x_range():
    with pytest.raises(ValueError, match="one prompt count"):
        fit_power_law([(20, 1e-3), (20, 2e-3)])
