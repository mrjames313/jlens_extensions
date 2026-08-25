"""Tests for the gamma-corrected J-lens dictionary.

The load-bearing test is :func:`test_corrected_logits_rank_like_a_real_rmsnorm`,
which checks our algebra against ``torch.nn.RMSNorm`` rather than against a
re-statement of the same formula. Everything else here would pass just as happily
if the derivation were wrong in the same way twice.

The failure mode this module exists to prevent is silent: an un-corrected
dictionary selects the right tokens (selection comes from proper logits) and then
builds the wrong vectors for them, so a wrong subspace gets ablated or swapped and
nothing raises. Tests therefore assert *ranking* and *vector* agreement separately
-- agreement on one does not imply the other.

Numerics are float64 throughout so that near-ties don't make ordering assertions
flaky; the properties under test are exact, not statistical.
"""

import pytest
import torch

from jlens_extensions.dictionary import (
    corrected_unembedding,
    dictionary_vectors,
    effective_gain,
    gain_spread,
    lens_logits,
)

D_MODEL, VOCAB, N_POS = 16, 64, 5


class OffsetRMSNorm(torch.nn.Module):
    """RMSNorm under the ``x/rms(x) * (1 + w)`` convention.

    Qwen3.5 and Gemma do this. Reading ``.weight`` off such a module and treating
    it as the gain is the bug that invalidated T9's first run, so it gets a
    stand-in here rather than being trusted to a comment.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(d_model, dtype=torch.float64) * 0.5 + 3.0)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 + self.weight) * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def fixture(gain: torch.Tensor | None = None, seed: int = 0):
    """A tiny lens setup, plus a real RMSNorm module carrying ``gain``."""
    torch.manual_seed(seed)
    W_U = torch.randn(VOCAB, D_MODEL, dtype=torch.float64)
    J_bar = torch.randn(D_MODEL, D_MODEL, dtype=torch.float64)
    h = torch.randn(N_POS, D_MODEL, dtype=torch.float64)
    norm = torch.nn.RMSNorm(D_MODEL, dtype=torch.float64)
    with torch.no_grad():
        norm.weight.copy_(
            torch.rand(D_MODEL, dtype=torch.float64) * 3.0 + 0.2 if gain is None else gain
        )
    return W_U, J_bar, h, norm, norm.weight.detach()


def ranks(logits: torch.Tensor) -> torch.Tensor:
    return logits.argsort(dim=-1, descending=True)


# --- the derivation, against an independent implementation of the norm --------


def test_corrected_logits_rank_like_a_real_rmsnorm():
    """``W_U norm(J h)`` and our corrected form must induce the same order."""
    W_U, J_bar, h, norm, gamma = fixture()
    reference = norm(h @ J_bar.T) @ W_U.T  # the real path: rms, eps, gain and all
    ours = lens_logits(h, J_bar, W_U, gamma, correct=True)
    assert torch.equal(ranks(reference), ranks(ours))


def test_the_omitted_factor_is_exactly_a_shared_positive_scalar():
    """Stronger than ordering: reference/ours is constant across tokens per row.

    That constant is ``1/rms(z)``. If anything token-dependent were missing from
    the correction, the ratio would vary within a row.
    """
    W_U, J_bar, h, norm, gamma = fixture()
    reference = norm(h @ J_bar.T) @ W_U.T
    ours = lens_logits(h, J_bar, W_U, gamma, correct=True)
    ratio = reference / ours
    assert torch.all(ratio > 0)
    relative_spread = ratio.std(dim=-1) / ratio.mean(dim=-1).abs()
    assert torch.all(relative_spread < 1e-9)


# --- the iff: the two constructions agree exactly when the gain is uniform ----


@pytest.mark.parametrize("scale", [1.0, 0.25, 7.5])
def test_uniform_gain_makes_the_two_constructions_agree(scale):
    uniform = torch.full((D_MODEL,), scale, dtype=torch.float64)
    W_U, J_bar, h, _, gamma = fixture(gain=uniform)
    corrected = lens_logits(h, J_bar, W_U, gamma, correct=True)
    plain = lens_logits(h, J_bar, W_U, gamma, correct=False)
    assert torch.equal(ranks(corrected), ranks(plain))


def test_non_uniform_gain_makes_them_disagree():
    """Guards against the previous test passing vacuously."""
    W_U, J_bar, h, _, gamma = fixture()
    corrected = lens_logits(h, J_bar, W_U, gamma, correct=True)
    plain = lens_logits(h, J_bar, W_U, gamma, correct=False)
    assert not torch.equal(ranks(corrected), ranks(plain))


# --- vectors and logits must be the same object seen two ways ----------------


@pytest.mark.parametrize("correct", [True, False])
def test_dictionary_vectors_reproduce_the_logits_they_come_from(correct):
    """``<v_t, h>`` must equal the logit for ``t``. Catches a stray transpose."""
    W_U, J_bar, h, _, gamma = fixture()
    token_ids = torch.tensor([0, 3, 17, 63])
    vectors = dictionary_vectors(W_U, gamma, J_bar, token_ids, correct=correct)
    from_vectors = h @ vectors.T
    from_logits = lens_logits(h, J_bar, W_U, gamma, correct=correct)[:, token_ids]
    assert torch.allclose(from_vectors, from_logits)


def test_uncorrected_vectors_are_literally_the_papers_rows():
    W_U, J_bar, _, _, gamma = fixture()
    token_ids = torch.tensor([1, 2, 5])
    plain = dictionary_vectors(W_U, gamma, J_bar, token_ids, correct=False)
    assert torch.allclose(plain, W_U[token_ids] @ J_bar)


def test_corrected_vectors_use_the_folded_unembedding():
    W_U, J_bar, _, _, gamma = fixture()
    token_ids = torch.tensor([1, 2, 5])
    corrected = dictionary_vectors(W_U, gamma, J_bar, token_ids, correct=True)
    folded = corrected_unembedding(W_U, gamma)
    assert torch.allclose(corrected, folded[token_ids] @ J_bar)


def test_the_correction_moves_the_vectors_it_keeps_the_tokens_for():
    """The documented hazard, stated as a test: same tokens, different directions."""
    W_U, J_bar, _, _, gamma = fixture()
    token_ids = torch.tensor([4, 8, 15, 16, 23, 42])
    corrected = dictionary_vectors(W_U, gamma, J_bar, token_ids, correct=True)
    plain = dictionary_vectors(W_U, gamma, J_bar, token_ids, correct=False)
    cosine = torch.nn.functional.cosine_similarity(corrected, plain, dim=-1)
    assert torch.all(cosine < 0.999)


# --- gain_spread -------------------------------------------------------------


def test_gain_spread_of_a_uniform_gain_is_one():
    stats = gain_spread(torch.full((D_MODEL,), 2.5))
    assert stats["spread_max_over_min"] == pytest.approx(1.0)
    assert stats["n_nonpositive"] == 0
    assert stats["d_model"] == D_MODEL


def test_gain_spread_reports_the_ratio():
    gamma = torch.tensor([0.5, 1.0, 2.0, 4.0])
    stats = gain_spread(gamma)
    assert stats["spread_max_over_min"] == pytest.approx(8.0)
    assert stats["percentiles"]["p0"] == pytest.approx(0.5)
    assert stats["percentiles"]["p100"] == pytest.approx(4.0)


def test_gain_spread_declines_to_divide_through_a_non_positive_entry():
    """max/min is meaningless if the gain crosses zero -- say so, don't emit it."""
    stats = gain_spread(torch.tensor([-1.0, 0.5, 2.0]))
    assert stats["spread_max_over_min"] is None
    assert stats["n_nonpositive"] == 1
    assert stats["spread_absmax_over_absmin"] == pytest.approx(4.0)


# --- shape guards ------------------------------------------------------------


def test_transposed_unembedding_is_rejected():
    W_U, J_bar, h, _, gamma = fixture()
    with pytest.raises(ValueError, match="gamma must be"):
        lens_logits(h, J_bar, W_U.T, gamma, correct=True)


def test_wrong_jacobian_shape_is_rejected():
    W_U, _, h, _, gamma = fixture()
    with pytest.raises(ValueError, match="J_bar must be"):
        lens_logits(h, torch.eye(D_MODEL + 1, dtype=torch.float64), W_U, gamma)


# --- recovering the gain from a module, without assuming its convention -------
#
# These exist because the tests above all pass while the *binding* to a real model
# is wrong: they validate the formula, not which tensor is the gain. T9's first run
# read `Qwen3_5RMSNorm.weight` on a `1 + w` module and every number was wrong.


@pytest.mark.parametrize("eps", [None, 1e-6, 1e-5])
def test_effective_gain_of_a_plain_rmsnorm_is_its_weight(eps):
    """``eps=None`` is ``torch.nn.RMSNorm``'s default and means "use finfo eps".

    It is parametrised because scanning for the attribute rather than for a usable
    value found it present and ``None``, which is how this first failed.
    """
    torch.manual_seed(0)
    norm = torch.nn.RMSNorm(D_MODEL, eps=eps, dtype=torch.float64)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(D_MODEL, dtype=torch.float64) * 3.0 + 0.2)
    assert torch.allclose(effective_gain(norm), norm.weight.detach(), atol=1e-9)


def test_effective_gain_of_an_offset_rmsnorm_is_one_plus_its_weight():
    """The regression test for the bug: the gain is NOT the raw weight here."""
    torch.manual_seed(0)
    norm = OffsetRMSNorm(D_MODEL)
    gain = effective_gain(norm)
    assert torch.allclose(gain, 1.0 + norm.weight.detach(), atol=1e-9)
    assert not torch.allclose(gain, norm.weight.detach(), atol=1e-3)


def test_effective_gain_rejects_a_module_that_is_not_a_diagonal_rescaling():
    """LayerNorm subtracts a mean, so no per-dimension gain reproduces it."""
    norm = torch.nn.LayerNorm(D_MODEL, dtype=torch.float64)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(D_MODEL, dtype=torch.float64) + 0.5)
    with pytest.raises(ValueError, match="not a diagonal rescaling"):
        effective_gain(norm)


def test_effective_gain_rejects_a_module_with_no_weight():
    with pytest.raises(ValueError, match="no .weight"):
        effective_gain(torch.nn.ReLU())


def test_recovered_gain_ranks_correctly_where_the_raw_weight_does_not():
    """End-to-end, through a module whose convention we deliberately do not assume.

    This is the test that would have caught T9's first run. It asserts both halves:
    the recovered gain reproduces the module's ranking, *and* the raw weight does
    not -- so the test cannot pass by the two being interchangeable.
    """
    torch.manual_seed(0)
    W_U = torch.randn(VOCAB, D_MODEL, dtype=torch.float64)
    J_bar = torch.randn(D_MODEL, D_MODEL, dtype=torch.float64)
    h = torch.randn(N_POS, D_MODEL, dtype=torch.float64)
    norm = OffsetRMSNorm(D_MODEL)

    reference = norm(h @ J_bar.T) @ W_U.T
    with_recovered = lens_logits(h, J_bar, W_U, effective_gain(norm), correct=True)
    with_raw_weight = lens_logits(h, J_bar, W_U, norm.weight.detach(), correct=True)

    assert torch.equal(ranks(reference), ranks(with_recovered))
    assert not torch.equal(ranks(reference), ranks(with_raw_weight))
