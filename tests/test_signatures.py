"""Tests for the corpus-free signatures S0 (CKA) and S4 (effective dimensionality).

These readouts have no reference implementation to check against — the paper's evidence
is a pair of qualitative figures — so where a signature's shape disagrees with its figure
we would have no way to tell a real difference from a bug. Every statistic here therefore
gets a synthetic case whose answer is known in advance: planted block structure with known
boundaries, an exactly low-rank dictionary with a known rank, an isotropic one whose
dimensionality curve is forced by arithmetic.

The load-bearing tests are:

* :func:`test_cka_matches_the_gram_definition` — the fast feature-space form against the
  paper's literal ``k x k`` similarity-matrix definition, computed by a separate route.
  Everything else about CKA would pass just as happily if both were wrong together.
* :func:`test_planted_blocks_are_recovered` — the segmentation against known boundaries.
* :func:`test_axes_follow_the_declared_layer_order` — a permuted CKA axis produces a
  matrix with no block structure, which is exactly S0's negative result, so this bug
  would confirm itself rather than announce itself.
* :func:`test_rank_starvation_is_flagged` — a token set narrower than the residual stream
  caps S4's curve, and the plateau reads as a property of the model.

Numerics are float64 throughout; the properties under test are exact, not statistical,
except where a test says otherwise in its name.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from jlens_extensions.signatures import (  # noqa: E402
    cka,
    cka_matrix,
    cka_null_baseline,
    effective_dimension,
    gram_cka,
    three_block_split,
)

K, D = 64, 32


def _rand(k: int = K, d: int = D, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, d, generator=g, dtype=torch.float64)


def _orthonormal(d: int, r: int, seed: int) -> torch.Tensor:
    """A ``[d, r]`` matrix with orthonormal columns."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, r, generator=g, dtype=torch.float64))
    return Q[:, :r]


def planted_stack(
    block_sizes: tuple[int, int, int] = (4, 6, 3),
    *,
    rank: int = 8,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    """A dictionary stack with three planted blocks and known boundaries.

    **The blocks differ in their token coefficients, not in their subspace** — and that
    is not an arbitrary choice. Linear CKA is invariant to orthogonal transforms of the
    features, so a layer whose dictionary is ``Z @ B.T`` has Gram matrix ``Z B.T B Z.T =
    Z Z.T`` for *any* ``B`` with orthonormal columns. Planting a different random basis
    per block therefore plants no signal at all: every block scores CKA 1 against every
    other, and the segmentation has nothing to find. (That mistake produced a
    within-minus-between contrast of 1.4e-4 here before it was caught.)

    What S0 compares is the pattern of pairwise similarities *among tokens*, so a block
    boundary means the token-to-token geometry changed. ``Z_b`` carries that.
    """
    rows = []
    for b, size in enumerate(block_sizes):
        gb = torch.Generator().manual_seed(seed + 500 + b)
        Z = torch.randn(K, rank, generator=gb, dtype=torch.float64)
        basis = _orthonormal(D, rank, seed=100 + b)
        for _ in range(size):
            perturb = torch.randn(K, D, generator=gb, dtype=torch.float64) * noise
            rows.append(Z @ basis.T + perturb)
    stack = torch.stack(rows)
    return stack, list(range(stack.shape[0]))


# --- CKA: the statistic ----------------------------------------------------------


def test_cka_of_a_matrix_with_itself_is_one():
    X = _rand()
    assert cka(X, X) == pytest.approx(1.0)


def test_cka_matches_the_gram_definition():
    """The fast form against the paper's literal ``k x k`` similarity matrices."""
    for seed in range(4):
        X, Y = _rand(seed=seed), _rand(seed=seed + 50)
        assert cka(X, Y) == pytest.approx(gram_cka(X, Y), rel=1e-10)


def test_cka_is_symmetric():
    X, Y = _rand(seed=1), _rand(seed=2)
    assert cka(X, Y) == pytest.approx(cka(Y, X))


def test_cka_lies_in_the_unit_interval():
    for seed in range(6):
        value = cka(_rand(seed=seed), _rand(seed=seed + 20))
        assert 0.0 <= value <= 1.0 + 1e-12


def test_cka_is_invariant_to_orthogonal_transforms():
    """Rotating one layer's dictionary must not change its similarity to another."""
    X, Y = _rand(seed=3), _rand(seed=4)
    Q = _orthonormal(D, D, seed=7)
    assert cka(X, Y @ Q) == pytest.approx(cka(X, Y))


def test_cka_is_invariant_to_isotropic_scaling():
    X, Y = _rand(seed=5), _rand(seed=6)
    assert cka(X, 17.0 * Y) == pytest.approx(cka(X, Y))


def test_cka_is_not_invariant_to_anisotropic_rescaling():
    """A sanity bound on the invariance above: CKA is not blind to *every* linear map.

    If it were, it would be measuring nothing about the geometry.
    """
    X, Y = _rand(seed=8), _rand(seed=9)
    stretch = torch.ones(D, dtype=torch.float64)
    stretch[: D // 2] = 40.0
    assert cka(X, Y * stretch) != pytest.approx(cka(X, Y), rel=1e-3)


def test_cka_centres_its_input():
    """A shared offset is not shared geometry; adding one must not change the answer."""
    X, Y = _rand(seed=10), _rand(seed=11)
    offset = torch.full((1, D), 25.0, dtype=torch.float64)
    assert cka(X + offset, Y + offset) == pytest.approx(cka(X, Y))


def test_an_uncentred_computation_would_differ():
    """Guards the test above from passing vacuously.

    A large shared offset genuinely does swamp the statistic when centring is skipped, so
    `test_cka_centres_its_input` is checking a real behaviour rather than an offset too
    small to matter.
    """
    X, Y = _rand(seed=10), _rand(seed=11)
    offset = torch.full((1, D), 25.0, dtype=torch.float64)
    Xo, Yo = X + offset, Y + offset
    uncentred = float((Yo.T @ Xo).pow(2).sum() / ((Xo.T @ Xo).norm() * (Yo.T @ Yo).norm()))
    assert uncentred > 0.9  # nearly 1: the offset dominates
    assert cka(Xo, Yo) < 0.5  # the centred answer is the honest one


def test_the_independent_baseline_tracks_d_over_k_not_zero():
    """CKA's floor is set by the shape of the measurement, not by 0.

    Unrelated dictionaries score about ``d_model / k``, approaching it from below as the
    ratio shrinks. This is the reason :func:`cka_null_baseline` exists and the reason a
    raw CKA matrix cannot be compared between two rungs of different width.
    """
    for k, d in [(1024, 32), (512, 32), (256, 16)]:
        base = cka_null_baseline(k, d, n_draws=4)
        assert base["mean"] == pytest.approx(d / k, rel=0.35)
        assert base["mean"] > 0.5 * (d / k)


def test_the_baseline_falls_as_the_token_set_grows():
    """The mitigation, stated as a test: widen the token set to lower the floor."""
    narrow = cka_null_baseline(256, 32, n_draws=4)["mean"]
    wide = cka_null_baseline(2048, 32, n_draws=4)["mean"]
    assert wide < narrow / 4


def test_the_baseline_rises_with_model_width_at_a_fixed_token_set():
    """Why a 4096-token set that is comfortable at 0.8B is not at 4B."""
    at_small_width = cka_null_baseline(2048, 64, n_draws=4)["mean"]
    at_large_width = cka_null_baseline(2048, 256, n_draws=4)["mean"]
    assert at_large_width > 3 * at_small_width


def test_baseline_rejects_degenerate_shapes():
    with pytest.raises(ValueError, match="need n_tokens"):
        cka_null_baseline(1, 32)
    with pytest.raises(ValueError, match="need n_tokens"):
        cka_null_baseline(64, 32, n_draws=0)


def test_cka_of_a_constant_dictionary_is_zero():
    """No variance means no correlation, not perfect correlation."""
    constant = torch.ones(K, D, dtype=torch.float64) * 3.0
    assert cka(constant, _rand()) == 0.0
    assert cka(constant, constant) == 0.0


def test_cka_rejects_mismatched_token_counts():
    with pytest.raises(ValueError, match="token counts must match"):
        cka(_rand(k=K), _rand(k=K // 2))


def test_cka_rejects_a_single_token():
    with pytest.raises(ValueError, match="at least 2 tokens"):
        cka(_rand(k=1), _rand(k=1))


# --- CKA: the layer-by-layer matrix ---------------------------------------------


def test_cka_matrix_is_symmetric_with_a_unit_diagonal():
    stack, layers = planted_stack()
    C, out_layers = cka_matrix(stack, layers)
    assert out_layers == layers
    assert torch.allclose(C, C.T)
    assert torch.allclose(C.diagonal(), torch.ones(C.shape[0], dtype=torch.float64))


def test_cka_matrix_entries_equal_the_pairwise_statistic():
    """The matrix is not a second implementation of CKA."""
    stack, layers = planted_stack()
    C, _ = cka_matrix(stack, layers)
    for i, j in [(0, 1), (0, 7), (4, 9), (2, 12)]:
        assert float(C[i, j]) == pytest.approx(cka(stack[i], stack[j]))


def test_axes_follow_the_declared_layer_order():
    """A stack loaded out of order, with its true layer order declared, must give the
    same matrix up to the corresponding permutation — not a different one."""
    stack, layers = planted_stack()
    perm = torch.randperm(stack.shape[0], generator=torch.Generator().manual_seed(3))
    shuffled = stack[perm]
    shuffled_layers = [layers[i] for i in perm.tolist()]

    C_ordered, _ = cka_matrix(stack, layers)
    C_shuffled, out_layers = cka_matrix(shuffled, shuffled_layers)

    assert out_layers == shuffled_layers
    assert torch.allclose(C_shuffled, C_ordered[perm][:, perm])


def test_cka_matrix_rejects_a_layer_list_of_the_wrong_length():
    stack, layers = planted_stack()
    with pytest.raises(ValueError, match="silently permutes"):
        cka_matrix(stack, layers[:-1])


# --- Three-block segmentation ----------------------------------------------------


def test_planted_blocks_are_recovered():
    sizes = (4, 6, 3)
    stack, layers = planted_stack(sizes)
    C, _ = cka_matrix(stack, layers)
    split = three_block_split(C, layers, depth=len(layers) + 1)
    assert split.boundaries == (sizes[0], sizes[0] + sizes[1])
    assert split.layer_spans == ((0, 3), (4, 9), (10, 12))


def test_planted_blocks_separate_and_unstructured_ones_do_not():
    stack, layers = planted_stack()
    structured, _ = cka_matrix(stack, layers)
    planted = three_block_split(structured, layers, depth=len(layers) + 1)

    g = torch.Generator().manual_seed(99)
    noise_stack = torch.randn(len(layers), K, D, generator=g, dtype=torch.float64)
    unstructured, _ = cka_matrix(noise_stack, layers)
    flat = three_block_split(unstructured, layers, depth=len(layers) + 1)

    assert planted.contrast > 0.5
    assert flat.contrast < 0.1
    # And the split is still returned for the unstructured case: the caller judges the
    # contrast, the function does not decide whether a band exists.
    assert len(flat.layer_spans) == 3


def test_middle_block_is_reported_as_a_fraction_of_depth():
    """The portable form. Layer / (depth - 1), so 0.0 is the first layer and 1.0 the
    final one — including the target layer the fit does not cover."""
    C = torch.eye(9, dtype=torch.float64)
    layers = list(range(9))
    split = three_block_split(C, layers, depth=10)
    lo, hi = split.layer_spans[1]
    assert split.middle_fraction == pytest.approx((lo / 9, hi / 9))


def test_depth_defaults_to_one_above_the_last_fitted_layer():
    """A full fitted stack runs 0..L-2, so the model's depth is max(layers) + 2."""
    C = torch.eye(9, dtype=torch.float64)
    layers = list(range(9))
    assert three_block_split(C, layers).middle_fraction == pytest.approx(
        three_block_split(C, layers, depth=10).middle_fraction
    )


def test_the_same_relative_band_reads_alike_at_two_depths():
    """The point of expressing the band as a fraction: a 24-layer and a 32-layer model
    with the band in the same relative place must report nearly the same numbers."""
    small = three_block_split(torch.eye(23, dtype=torch.float64), list(range(23)), depth=24)
    large = three_block_split(torch.eye(31, dtype=torch.float64), list(range(31)), depth=32)
    # An identity matrix has many tied splits, so pin the comparison to the arithmetic
    # rather than to which tie the search happens to take.
    for frac_s, frac_l in zip(
        (8 / 23, 15 / 23),
        (11 / 31, 20 / 31),
    ):
        assert frac_s == pytest.approx(frac_l, abs=0.02)
    assert len(small.layer_spans) == len(large.layer_spans) == 3


def test_min_block_is_respected():
    stack, layers = planted_stack((4, 6, 3))
    C, _ = cka_matrix(stack, layers)
    split = three_block_split(C, layers, min_block=4)
    i, j = split.boundaries
    assert i >= 4 and (j - i) >= 4 and (len(layers) - j) >= 4


def test_rejects_a_stack_too_short_for_three_blocks():
    C = torch.eye(5, dtype=torch.float64)
    with pytest.raises(ValueError, match="need at least 6 layers"):
        three_block_split(C, list(range(5)))


def test_rejects_a_min_block_that_leaves_within_undefined():
    C = torch.eye(9, dtype=torch.float64)
    with pytest.raises(ValueError, match="min_block must be at least 2"):
        three_block_split(C, list(range(9)), min_block=1)


# --- S4: effective linear dimensionality -----------------------------------------


def _stack_from(mats: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    return torch.stack(mats), list(range(len(mats)))


def test_an_exactly_low_rank_dictionary_reports_its_rank():
    """Vectors confined to an r-dimensional subspace need exactly r components."""
    rank = 5
    g = torch.Generator().manual_seed(12)
    basis = _orthonormal(D, rank, seed=13)
    coeffs = torch.randn(K, rank, generator=g, dtype=torch.float64)
    stack, layers = _stack_from([coeffs @ basis.T])

    profile = effective_dimension(stack, layers, shares=(1.0,))
    assert int(profile.components[0, 0]) == rank
    assert float(profile.fractions[0, 0]) == pytest.approx(rank / D)


def test_an_isotropic_dictionary_is_linear_in_the_share():
    """Equal eigenvalues make the cumulative share m/rank, so the count is forced:
    the smallest m with m/rank >= share, i.e. ceil(share * rank)."""
    rank = 16
    basis = _orthonormal(D, rank, seed=21)
    # Orthonormal coefficient columns scaled alike => an exactly flat spectrum.
    coeffs = _orthonormal(K, rank, seed=22)
    stack, layers = _stack_from([coeffs @ basis.T])

    shares = (0.25, 0.5, 0.75, 1.0)
    profile = effective_dimension(stack, layers, shares=shares)
    for row, share in enumerate(shares):
        assert int(profile.components[row, 0]) == math.ceil(share * rank)


def test_the_curve_is_monotone_in_the_share():
    stack, layers = planted_stack()
    profile = effective_dimension(stack, layers, shares=(0.5, 0.9, 0.99))
    diffs = profile.components[1:] - profile.components[:-1]
    assert bool((diffs >= 0).all())


def test_effective_dimension_is_scale_invariant():
    stack, layers = planted_stack()
    base = effective_dimension(stack, layers)
    scaled = effective_dimension(stack * 1e4, layers)
    assert torch.equal(base.components, scaled.components)


def test_effective_dimension_centres_its_input():
    """A shared offset is one direction shared by every vector, and an uncentred
    spectrum would count it as a dimension of the J-space."""
    stack, layers = planted_stack()
    offset = torch.full((1, 1, D), 30.0, dtype=torch.float64)
    assert torch.equal(
        effective_dimension(stack, layers).components,
        effective_dimension(stack + offset, layers).components,
    )


def test_rank_starvation_is_flagged():
    """A token set narrower than the residual stream caps the curve; the plateau is an
    artifact of the token set and must not read as a property of the model."""
    narrow = torch.randn(3, 12, D, generator=torch.Generator().manual_seed(31), dtype=torch.float64)
    profile = effective_dimension(narrow, [0, 1, 2], shares=(1.0,))
    assert profile.rank_limited is True
    assert profile.rank_cap == pytest.approx(11 / D)
    assert float(profile.fractions.max()) <= profile.rank_cap + 1e-12


def test_a_token_set_wider_than_the_residual_stream_is_not_flagged():
    wide = torch.randn(2, D * 2, D, generator=torch.Generator().manual_seed(32), dtype=torch.float64)
    profile = effective_dimension(wide, [0, 1], shares=(1.0,))
    assert profile.rank_limited is False
    assert profile.rank_cap == pytest.approx(1.0)


def test_a_constant_layer_reports_no_components():
    stack = torch.stack([torch.ones(K, D, dtype=torch.float64), _rand()])
    profile = effective_dimension(stack, [0, 1], shares=(0.9,))
    assert int(profile.components[0, 0]) == 0
    assert int(profile.components[0, 1]) > 0


def test_profile_records_what_a_quoted_number_needs():
    stack, layers = planted_stack()
    profile = effective_dimension(stack, layers, shares=(0.9,))
    assert profile.shares == (0.9,)
    assert profile.layers == layers
    assert profile.d_model == D
    assert profile.fractions.shape == (1, len(layers))


def test_rejects_unsorted_or_out_of_range_shares():
    stack, layers = planted_stack()
    with pytest.raises(ValueError, match="ascending"):
        effective_dimension(stack, layers, shares=(0.9, 0.5))
    with pytest.raises(ValueError, match="must lie in"):
        effective_dimension(stack, layers, shares=(0.0,))
    with pytest.raises(ValueError, match="must lie in"):
        effective_dimension(stack, layers, shares=(1.5,))
    with pytest.raises(ValueError, match="at least one variance share"):
        effective_dimension(stack, layers, shares=())


def test_rejects_a_layer_list_of_the_wrong_length():
    stack, layers = planted_stack()
    with pytest.raises(ValueError, match="names"):
        effective_dimension(stack, layers[:-1])


# --- The two signatures agree on the planted band --------------------------------


def test_both_signatures_see_the_planted_middle_block():
    """S0 and S4 are the free internal cross-check the plan relies on. On a stack where
    the middle block genuinely spans a larger subspace, both must say so."""
    g = torch.Generator().manual_seed(41)
    rows = []
    for block, (size, rank) in enumerate([(4, 3), (6, 20), (3, 6)]):
        basis = _orthonormal(D, rank, seed=200 + block)
        Z = torch.randn(K, rank, generator=g, dtype=torch.float64)
        for _ in range(size):
            noise = torch.randn(K, D, generator=g, dtype=torch.float64) * 0.02
            rows.append(Z @ basis.T + noise)
    stack, layers = _stack_from(rows)

    C, _ = cka_matrix(stack, layers)
    split = three_block_split(C, layers, depth=len(layers) + 1)
    assert split.boundaries == (4, 10)

    profile = effective_dimension(stack, layers, shares=(0.99,))
    early = profile.components[0, :4].double().mean()
    middle = profile.components[0, 4:10].double().mean()
    late = profile.components[0, 10:].double().mean()
    assert middle > early
    assert middle > late
