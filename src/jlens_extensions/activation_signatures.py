"""The activation-dependent workspace signatures: S1, S2 and S3.

The other half of stage 2. Where S0 and S4 read the J-lens vectors themselves and need
no corpus (:mod:`jlens_extensions.signatures`), these three read what the lens *says* at
each position of an evaluation set, so they need one pass over that set — T8's cache.
The split is load-bearing rather than organisational: a band located by S0/S4 and a band
located by S1–S3 disagreeing points at the activation set rather than at the model, and
that free cross-check only exists because the two halves share no inputs.

What the paper measures (Figure 28):

* **S1** — "the fraction of positions at which any top-k J-lens token matches the model's
  top-1 prediction." Near zero early, ticks up at the band's start, and jumps steeply in
  the final layers. That late jump marks the band's **end**, and it has to be positively
  identified rather than assumed to be the last layer.
* **S2** — "excess kurtosis of the J-lens readout distribution; high kurtosis indicates a
  readout sharply peaked on a few tokens." Its early rise marks the band's **start**.
* **S3** — "autocorrelation of the top-1 lens token across positions, as Δ log probability
  relative to a position-shuffled null." Whether abstract content persists across the
  token stream. The null is not optional: it is what makes the number interpretable, and
  it doubles as the self-test for S3's half of the T3 capability precondition — a layer
  whose autocorrelation cannot exceed its null reports exactly that.

The cache contract (what T8 must produce)
-----------------------------------------

This module was written before T8, so its inputs *are* T8's output contract. Three
arrays, all indexed ``[layer, position, ...]`` with a shared position axis — shared
because three signatures over three passes is three chances to disagree about which
positions were scored:

* ``lens_topk`` ``[n_layers, n_positions, k_max]`` — the J-lens readout's top tokens per
  position, **rank-ordered**, for S1.
* ``moments`` ``[n_layers, n_positions, 2]`` — the central moments ``(m2, m4)`` of the
  readout's logit vector over the vocabulary, for S2. See below.
* ``candidate_logprobs`` ``[n_layers, n_positions, n_candidates]`` with ``top1_index``
  ``[n_layers, n_positions]``, for S3.

Plus ``model_top1`` ``[n_positions]`` — the model's own prediction, shared across layers —
and an optional ``valid`` mask, since a real corpus has positions (BOS, padding, the
tail of a truncated prompt) that must not be scored.

**S2 settles T8's open question about the logit distribution.** T8 notes that the
per-(position, layer) logit distribution "is the large object and may need summarising
rather than storing whole". It cannot be stored whole and does not need to be. At
Qwen3.5's 248,320-token vocabulary, 23 layers and 8192 positions, the full fp32 tensor is
**174 GiB**, while excess kurtosis needs only the second and fourth *central* moments —
**1.44 MiB** for the same run. The reduction is exactly ``vocab / 2``, ~124,000x here.
Store central moments, computed in the same pass while the logit vector is still in hand;
never raw power sums, which lose the answer to cancellation (see :func:`central_moments`).

S3's candidates are the compromise in the other direction: it genuinely needs
``log p(token | some other position)``, so a summary will not do. Restricting to the
distinct top-1 tokens actually observed keeps it bounded — at 8192 positions the set is at
most 8192 wide and in practice far narrower. At the same 23 x 8192 shape that is 0.36 GiB
for 512 candidates, 1.44 GiB for 2048 and 5.75 GiB for 8192, so this is the term that
sizes the cache and the one to profile. Cap ``n_candidates`` or subsample positions if it
does not fit; both are honest, and both must be recorded.

Reading the numbers
-------------------

``k`` for S1 and ``Δ`` for S3 are choices the paper does not fix, so both entry points
take several at once and return a curve per value. A signature visible at only one ``k``
or one ``Δ`` is a fact about that choice, and T9 requires them stated wherever a number
is quoted.

:func:`mark_transition` reports two views of where a curve turns — a threshold crossing
and the steepest single-layer rise — and deliberately does not pick between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = [
    "central_moments",
    "excess_kurtosis",
    "excess_kurtosis_from_moments",
    "topk_accuracy",
    "kurtosis_curve",
    "top1_autocorrelation",
    "Transition",
    "mark_transition",
]


def _check_layer_position(x: torch.Tensor, name: str, n_layers: int, n_positions: int) -> None:
    if x.shape[0] != n_layers or x.shape[1] != n_positions:
        raise ValueError(
            f"{name} is {tuple(x.shape)}; expected leading dims "
            f"[{n_layers}, {n_positions}]. All three signatures must score the same "
            "positions at the same layers — a mismatch here is how they end up "
            "disagreeing about which positions they scored."
        )


def _valid_mask(valid: torch.Tensor | None, n_positions: int) -> torch.Tensor:
    if valid is None:
        return torch.ones(n_positions, dtype=torch.bool)
    if valid.shape != (n_positions,):
        raise ValueError(f"valid must be [{n_positions}], got {tuple(valid.shape)}")
    mask = valid.to(torch.bool)
    if not bool(mask.any()):
        raise ValueError("valid mask excludes every position")
    return mask


# --- S1: top-k next-token accuracy ------------------------------------------------


def topk_accuracy(
    lens_topk: torch.Tensor,
    model_top1: torch.Tensor,
    *,
    ks: Sequence[int],
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """S1: fraction of positions where any top-``k`` lens token is the model's top-1.

    Args:
        lens_topk: ``[n_layers, n_positions, k_max]`` token ids, **rank-ordered** along
            the last axis. Order matters: accuracy at ``k`` reads the first ``k`` columns,
            so an unsorted axis silently reports accuracy at ``k_max`` for every ``k``.
        model_top1: ``[n_positions]`` token ids — the model's own prediction, which is a
            property of the position and not of the layer.
        ks: The cutoffs to report, ascending, each ``<= k_max``.
        valid: Optional ``[n_positions]`` bool mask of positions to score.

    Returns:
        ``[len(ks), n_layers]``. Rows are non-decreasing in ``k`` by construction.

    The curve's late steep rise is the band's end. It must be read off the curve, not
    assumed to be the last layer — the final layers are where ``J`` approaches the
    identity and the lens inherits the unembedding, so accuracy there is near-tautological
    and says nothing about a workspace.
    """
    if lens_topk.ndim != 3:
        raise ValueError(f"lens_topk must be [n_layers, n_positions, k_max], got {tuple(lens_topk.shape)}")
    n_layers, n_positions, k_max = lens_topk.shape
    if model_top1.shape != (n_positions,):
        raise ValueError(
            f"model_top1 must be [{n_positions}], got {tuple(model_top1.shape)}"
        )
    ks_t = [int(k) for k in ks]
    if not ks_t:
        raise ValueError("need at least one k")
    if list(ks_t) != sorted(ks_t):
        raise ValueError(f"ks must be ascending, got {ks_t}")
    if any(k < 1 or k > k_max for k in ks_t):
        raise ValueError(f"every k must lie in [1, {k_max}], got {ks_t}")

    mask = _valid_mask(valid, n_positions)
    hit = lens_topk == model_top1.view(1, n_positions, 1)
    out = torch.zeros((len(ks_t), n_layers), dtype=torch.float64)
    for row, k in enumerate(ks_t):
        found = hit[:, :, :k].any(dim=2)  # [n_layers, n_positions]
        out[row] = found[:, mask].to(torch.float64).mean(dim=1)
    return out


# --- S2: excess kurtosis of the readout -------------------------------------------


def central_moments(x: torch.Tensor, *, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """The second and fourth central moments along ``dim``, for S2's cache summary.

    Returns ``(m2, m4)`` — population moments, ``mean((x - mean)**p)``.

    **Central, not raw.** Excess kurtosis can also be assembled from raw power sums
    (``sum(x)``, ``sum(x**2)``, ``sum(x**3)``, ``sum(x**4)``), which is tempting because
    those accumulate in one pass with no stored mean. Don't: logits run to tens, so the
    fourth raw moment runs to ~1e6 while the central one it is differenced down to may be
    ~1, and in fp32 that cancellation destroys the answer. T8 has the logit vector in hand
    when it computes this, so the two-pass central form costs nothing there and is stable.
    """
    mean = x.mean(dim=dim, keepdim=True)
    centred = x - mean
    m2 = centred.pow(2).mean(dim=dim)
    m4 = centred.pow(4).mean(dim=dim)
    return m2, m4


def excess_kurtosis_from_moments(m2: torch.Tensor, m4: torch.Tensor) -> torch.Tensor:
    """``m4 / m2**2 - 3`` — the cache path, from :func:`central_moments`.

    Zero for a Gaussian; positive for a distribution peaked on a few tokens, which is what
    S2 is looking for. A degenerate readout (``m2 == 0``, every logit identical) has no
    defined kurtosis and reports ``nan`` rather than a number that would average in.
    """
    out = m4 / m2.pow(2) - 3.0
    return torch.where(m2 > 0, out, torch.full_like(out, float("nan")))


def excess_kurtosis(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Excess kurtosis along ``dim``, computed directly from the values.

    Use this when the logit vector is in hand. Use :func:`central_moments` plus
    :func:`excess_kurtosis_from_moments` when it is not — they agree exactly, which is
    asserted in the tests.
    """
    m2, m4 = central_moments(x, dim=dim)
    return excess_kurtosis_from_moments(m2, m4)


@dataclass(frozen=True)
class KurtosisCurve:
    """S2 per layer, aggregated over positions.

    ``mean`` and ``median`` are both reported because they answer different questions and
    can disagree: kurtosis is heavy-tailed across positions, so a handful of very peaked
    readouts move the mean without moving the typical position. Quote which one you used.
    ``n_scored`` is the count of positions that contributed after ``nan`` readouts were
    dropped, so a layer that was mostly degenerate cannot pass as a measured one.
    """

    mean: torch.Tensor
    median: torch.Tensor
    n_scored: torch.Tensor
    layers: list[int]


def kurtosis_curve(
    moments: torch.Tensor,
    layers: Sequence[int],
    *,
    valid: torch.Tensor | None = None,
) -> KurtosisCurve:
    """S2: the per-layer excess-kurtosis curve from T8's cached moments.

    Args:
        moments: ``[n_layers, n_positions, 2]`` — ``(m2, m4)`` per readout, from
            :func:`central_moments` over the vocabulary axis.
        layers: Layer index per row, handed back on the result.
        valid: Optional ``[n_positions]`` bool mask.

    The early rise in this curve marks the band's start.
    """
    if moments.ndim != 3 or moments.shape[2] != 2:
        raise ValueError(
            f"moments must be [n_layers, n_positions, 2] carrying (m2, m4), got {tuple(moments.shape)}"
        )
    n_layers, n_positions, _ = moments.shape
    if len(layers) != n_layers:
        raise ValueError(f"moments has {n_layers} layers but `layers` names {len(layers)}")
    mask = _valid_mask(valid, n_positions)

    values = excess_kurtosis_from_moments(moments[..., 0], moments[..., 1])[:, mask]
    finite = torch.isfinite(values)
    means = torch.zeros(n_layers, dtype=torch.float64)
    medians = torch.zeros(n_layers, dtype=torch.float64)
    counts = torch.zeros(n_layers, dtype=torch.int64)
    for i in range(n_layers):
        row = values[i][finite[i]].to(torch.float64)
        counts[i] = row.numel()
        if row.numel() == 0:
            means[i] = float("nan")
            medians[i] = float("nan")
        else:
            means[i] = row.mean()
            medians[i] = row.median()
    return KurtosisCurve(mean=means, median=medians, n_scored=counts, layers=list(layers))


# --- S3: top-1 autocorrelation against a position-shuffled null --------------------


def top1_autocorrelation(
    candidate_logprobs: torch.Tensor,
    top1_index: torch.Tensor,
    *,
    deltas: Sequence[int],
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """S3: does the top-1 lens token at ``t`` still score well at ``t + Δ``?

    For each position ``t``, take the lens's top-1 token there and read its log
    probability under the readout at ``t + Δ``. Subtract what that token would have
    scored at a *randomly chosen other* position — the position-shuffled null. A positive
    result means content persists across the token stream beyond what the token's general
    frequency explains; zero or negative means it does not.

    Args:
        candidate_logprobs: ``[n_layers, n_positions, n_candidates]`` — log probabilities
            restricted to the candidate token set (see the module docstring).
        top1_index: ``[n_layers, n_positions]`` — index **into the candidate axis** of the
            lens's top-1 token at that layer and position.
        deltas: Position offsets to report, ascending and positive.
        valid: Optional ``[n_positions]`` bool mask.

    Returns:
        ``[len(deltas), n_layers]`` of mean Δ log probability.

    **The null is computed exactly, not sampled.** Shuffling positions estimates an
    expectation — the mean of ``log p(token | t')`` over positions ``t'`` — and that
    expectation is a column mean we already have, so taking it directly removes the
    shuffle's variance for free and makes the result reproducible without a seed.

    Two positions are excluded from each null: ``t`` itself, because the token was chosen
    as the argmax there and including it biases the null upward; and ``t + Δ``, because
    that is the quantity being compared against and including it would contaminate the
    baseline with the signal.

    This is also the self-test for S3's half of the T3 capability precondition. There is no
    threshold to apply — a layer whose value does not exceed zero has failed to beat its
    own null, and reports that.
    """
    if candidate_logprobs.ndim != 3:
        raise ValueError(
            f"candidate_logprobs must be [n_layers, n_positions, n_candidates], "
            f"got {tuple(candidate_logprobs.shape)}"
        )
    n_layers, n_positions, n_candidates = candidate_logprobs.shape
    _check_layer_position(top1_index, "top1_index", n_layers, n_positions)
    if top1_index.ndim != 2:
        raise ValueError(f"top1_index must be 2-D, got {tuple(top1_index.shape)}")
    if int(top1_index.min()) < 0 or int(top1_index.max()) >= n_candidates:
        raise ValueError(
            f"top1_index values must index the candidate axis [0, {n_candidates}); "
            f"got range [{int(top1_index.min())}, {int(top1_index.max())}]"
        )
    deltas_t = [int(d) for d in deltas]
    if not deltas_t:
        raise ValueError("need at least one delta")
    if list(deltas_t) != sorted(deltas_t):
        raise ValueError(f"deltas must be ascending, got {deltas_t}")
    if any(d < 1 for d in deltas_t):
        raise ValueError(f"deltas must be positive, got {deltas_t}")

    mask = _valid_mask(valid, n_positions)
    n_valid = int(mask.sum())
    if n_valid < 3:
        raise ValueError(
            f"need at least 3 valid positions to leave two out of the null, got {n_valid}"
        )

    lp = candidate_logprobs.to(torch.float64)
    # Column sums over valid positions only: the null's numerator before leave-two-out.
    col_sum = (lp * mask.view(1, n_positions, 1)).sum(dim=1)  # [n_layers, n_candidates]

    out = torch.zeros((len(deltas_t), n_layers), dtype=torch.float64)
    for row, delta in enumerate(deltas_t):
        if delta >= n_positions:
            out[row] = float("nan")
            continue
        # Pairs (t, t+delta) where both ends are valid.
        pair = mask[: n_positions - delta] & mask[delta:]
        if not bool(pair.any()):
            out[row] = float("nan")
            continue
        t_idx = torch.nonzero(pair, as_tuple=False).squeeze(1)
        for layer in range(n_layers):
            cand = top1_index[layer, t_idx]  # [n_pairs]
            at_t = lp[layer, t_idx, cand]
            at_t_plus = lp[layer, t_idx + delta, cand]
            total = col_sum[layer, cand]
            null = (total - at_t - at_t_plus) / (n_valid - 2)
            out[row, layer] = (at_t_plus - null).mean()
    return out


# --- Marking where a curve turns ---------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """Two views of where a signature curve turns, neither of them authoritative.

    Attributes:
        crossing_layer: First layer at which the curve reaches
            ``min + fraction * (max - min)``.
        steepest_layer: Layer with the largest single-step rise; the reported index is the
            layer *arrived at*.
        steepest_rise: That rise.
        fraction: The threshold used for ``crossing_layer``.

    They are reported together because they disagree in the informative cases — a curve
    that drifts up early and jumps late gives an early crossing and a late steepest rise,
    and which one is "the onset" is exactly the judgement T10 has to make and record.
    """

    crossing_layer: int
    steepest_layer: int
    steepest_rise: float
    fraction: float


def mark_transition(
    curve: torch.Tensor,
    layers: Sequence[int],
    *,
    fraction: float = 0.5,
) -> Transition:
    """Locate a curve's turn, as a threshold crossing and as the steepest rise.

    A heuristic over a noisy curve, not a measurement: use it to *propose* an onset or an
    end, and record which view a reported band came from. T9 requires onset and end marked
    per signature; T10 takes the intersection where signatures disagree at the margins.
    """
    if curve.ndim != 1:
        raise ValueError(f"curve must be 1-D over layers, got {tuple(curve.shape)}")
    if len(layers) != curve.shape[0]:
        raise ValueError(f"curve has {curve.shape[0]} points but `layers` names {len(layers)}")
    if curve.shape[0] < 2:
        raise ValueError("need at least 2 layers to mark a transition")
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must lie in (0, 1), got {fraction}")

    c = curve.to(torch.float64)
    if not bool(torch.isfinite(c).all()):
        raise ValueError("curve contains non-finite values; drop or impute them first")

    lo, hi = float(c.min()), float(c.max())
    if hi == lo:
        # A flat curve has no transition. Report the first layer rather than inventing one.
        return Transition(
            crossing_layer=int(layers[0]),
            steepest_layer=int(layers[0]),
            steepest_rise=0.0,
            fraction=fraction,
        )
    threshold = lo + fraction * (hi - lo)
    crossing = int(torch.nonzero(c >= threshold, as_tuple=False)[0])
    steps = c[1:] - c[:-1]
    steepest = int(torch.argmax(steps))
    return Transition(
        crossing_layer=int(layers[crossing]),
        steepest_layer=int(layers[steepest + 1]),
        steepest_rise=float(steps[steepest]),
        fraction=fraction,
    )
