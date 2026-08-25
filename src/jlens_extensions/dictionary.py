"""J-lens dictionary vectors, with the final norm's gain folded in.

The paper defines the J-lens vectors as the rows of ``W_U J_l``, in the sentence
immediately after defining the readout as ``softmax(W_U norm(J_l h_l))``. ``norm``
appears in one and not the other. For an RMSNorm model ``norm(x) = g * x / rms(x)``,
so for token ``t``::

    logit_t = <W_U[t], norm(z)>                       with z = J_l h_l
            = (1 / rms(z)) * sum_i W_U[t,i] * g_i * z_i
            = (1 / rms(z)) * <W_U[t] * g, z>
            = (1 / rms(z)) * <J_l^T (W_U[t] * g), h_l>

Two factors fall out, and they behave differently:

* ``1 / rms(z)`` is a positive scalar, shared by every token. It changes softmax
  temperature but not order, and scaling a direction by a positive scalar is a
  no-op -- so it is irrelevant both to ranking and to the vectors.
* ``g`` is a per-dimension reweighting *inside* the inner product. It does not
  cancel across tokens, so it changes which token wins.

The direction whose inner product with ``h_l`` gives token ``t``'s logit is
therefore row ``t`` of ``W_U diag(g) J_l``. The two constructions agree, up to
harmless per-token positive rescaling, **iff g is uniform** -- which is why
:func:`gain_spread` is the statistic that decides how much this matters, and why
it is reported per model.

Readout is unaffected: ``jlens.hf.HFLensModel.unembed`` calls the model's real
norm module, so shipped logits and ranks already contain ``g``. The hazard is
confined to building the dictionary, and its shape is unpleasant *because* the
readout is right -- you select the top-k tokens correctly from proper logits,
then construct wrong direction vectors for those correctly-chosen tokens, and
ablate or swap a subspace other than the one you identified. Plausible degraded
results, no error raised.

Fold into the unembedding, never into the lens: ``(W_U diag(g)) J_l`` and
``W_U (diag(g) J_l)`` are algebraically identical, but mutating the saved ``J_l``
would shift ``identity_distance`` and break Regime A comparability against the
published artifacts for a reason unrelated to fit fidelity.

See ``f-2026-08-18-jspace-construction-and-norm-gain`` for the full argument and
``ex-2026-08-18-jspace-dictionary-definition`` for research's reading of the paper.
"""

from __future__ import annotations

import torch

__all__ = [
    "corrected_unembedding",
    "dictionary_vectors",
    "lens_logits",
    "gain_spread",
]


def corrected_unembedding(W_U: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """``W~_U = W_U * gamma`` -- the gain folded into the unembedding.

    Args:
        W_U: Unembedding matrix, ``[vocab, d_model]``.
        gamma: Final-norm gain, ``[d_model]`` (an RMSNorm module's ``.weight``).

    Returns:
        ``[vocab, d_model]``. One row-wise scaling, computed once.
    """
    _check_shapes(W_U, gamma)
    return W_U * gamma


def dictionary_vectors(
    W_U: torch.Tensor,
    gamma: torch.Tensor,
    J_bar: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    correct: bool = True,
) -> torch.Tensor:
    """Form the dictionary vectors for ``token_ids`` at one layer.

    Only the requested rows are formed. The full dictionary is ``vocab x d_model``
    per layer -- roughly 2.5 GB in fp16 at Qwen3.5's 248,320 vocab -- and is not
    materialisable for all layers at scale, so this deliberately takes token ids
    rather than returning the whole thing.

    Args:
        W_U: Unembedding, ``[vocab, d_model]``.
        gamma: Final-norm gain, ``[d_model]``.
        J_bar: The fitted Jacobian for one layer, ``[d_model, d_model]``, in the
            harness's convention (``z = J_bar @ h``, i.e. ``h @ J_bar.T``).
        token_ids: Which vocabulary rows to form, ``[k]``.
        correct: Fold ``gamma`` in. ``False`` reproduces the paper's literal
            ``W_U J_l``, which is what you want only when deliberately measuring
            the difference between the two constructions.

    Returns:
        ``[k, d_model]``, one direction per requested token, in layer-``l``
        residual-stream space.
    """
    _check_shapes(W_U, gamma, J_bar)
    rows = W_U[token_ids]
    if correct:
        rows = rows * gamma
    # v_t = J_bar^T (W_U[t] * g); in row form that is (W_U[t] * g) @ J_bar.
    return rows @ J_bar


def lens_logits(
    residual: torch.Tensor,
    J_bar: torch.Tensor,
    W_U: torch.Tensor,
    gamma: torch.Tensor,
    *,
    correct: bool = True,
) -> torch.Tensor:
    """Lens logits at one layer, up to a shared positive scalar.

    The ``1 / rms(z)`` factor is omitted because it is positive and identical
    across tokens: rankings, top-k and argmax are unaffected. Use
    :meth:`jlens.hf.HFLensModel.unembed` when the *values* matter (probabilities,
    calibration); use this when the ordering or the construction is what is under
    test, or when you want the un-corrected construction for comparison.

    Args:
        residual: ``[..., d_model]`` at layer ``l``.
        J_bar: ``[d_model, d_model]`` for that layer.
        W_U: ``[vocab, d_model]``.
        gamma: ``[d_model]``.
        correct: Fold ``gamma`` in; ``False`` gives the paper's construction.

    Returns:
        ``[..., vocab]``, order-equivalent to the true logits when ``correct``.
    """
    _check_shapes(W_U, gamma, J_bar)
    z = residual @ J_bar.T
    if correct:
        z = z * gamma
    return z @ W_U.T


def gain_spread(gamma: torch.Tensor) -> dict:
    """Summarise the gain vector -- the number that decides how much this matters.

    ``max/min`` is only meaningful for a strictly positive gain, so the
    absolute-value form is reported alongside it and non-positive entries are
    counted rather than silently producing a negative or exploded ratio.

    Args:
        gamma: ``[d_model]``.

    Returns:
        A JSON-serialisable dict of the spread, moments and percentiles.
    """
    g = gamma.detach().to(torch.float32).flatten().cpu()
    n_nonpositive = int((g <= 0).sum())
    return {
        "d_model": int(g.numel()),
        "min": g.min().item(),
        "max": g.max().item(),
        "mean": g.mean().item(),
        "std": g.std().item(),
        "n_nonpositive": n_nonpositive,
        "spread_max_over_min": (g.max() / g.min()).item() if n_nonpositive == 0 else None,
        "spread_absmax_over_absmin": (g.abs().max() / g.abs().min()).item(),
        "percentiles": {
            f"p{q:g}": torch.quantile(g, q / 100.0).item()
            for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)
        },
    }


def _check_shapes(
    W_U: torch.Tensor, gamma: torch.Tensor, J_bar: torch.Tensor | None = None
) -> None:
    if W_U.ndim != 2:
        raise ValueError(f"W_U must be [vocab, d_model], got {tuple(W_U.shape)}")
    d_model = W_U.shape[1]
    if gamma.shape != (d_model,):
        raise ValueError(
            f"gamma must be [d_model={d_model}], got {tuple(gamma.shape)} -- a "
            f"transposed W_U or a LayerNorm bias mistaken for the gain both land here"
        )
    if J_bar is not None and J_bar.shape != (d_model, d_model):
        raise ValueError(
            f"J_bar must be [{d_model}, {d_model}], got {tuple(J_bar.shape)}"
        )
