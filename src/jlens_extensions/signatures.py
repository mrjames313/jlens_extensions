"""The corpus-free workspace signatures: S0 (CKA block structure) and S4 (effective
linear dimensionality).

Both read the J-lens vectors themselves — the rows of ``W_U diag(gamma) J_l``, as
produced by :func:`jlens_extensions.dictionary.dictionary_stack` — and neither needs
a corpus. That is what makes them the cheap half of stage 2, and it is also what makes
them a cross-check: a band located by S0/S4 and a band located by the
activation-dependent signatures S1–S3 disagreeing points at the activation set rather
than at the model.

What the paper says
-------------------

S0: "we compute the similarity of the J-space's geometry [...] using centered kernel
alignment (CKA), which compares, for each pair of layers, the matrices of pairwise
similarities among J-lens vectors. The resulting matrix has a clear block structure: an
early block encompassing roughly the first third of the model, a long middle block, and
a small late block." It adds the warning this module's callers have to respect: "the
observed sharpness is exaggerated by layer subsampling."

S4: "the fraction of residual-stream dimensions needed to capture a given share of
variance across the J-lens vectors ``W_U J_l``. Through the early layers this fraction is
small [...] The effective dimensionality rises sharply around the same layer as the other
metrics [...] It rises again, less dramatically, at the transition to 'motor' layers, as
``J_l`` approaches the identity."

Note "a given share" — the paper never fixes it. Ours is therefore a choice, not a
reading, and :func:`effective_dimension` takes several shares at once so the driver can
report whether the *shape* survives the choice. A curve that only shows an onset at one
share is a result about our threshold.

Four traps
----------

**CKA's floor is not zero, and it rises with model width.** Two *completely unrelated*
dictionaries over ``k`` tokens in ``d_model`` dimensions score roughly ``d_model / k``,
not 0 — measured, and approaching that ratio from below as it gets small. At the planned
4096-token set that is ``~0.20`` at Qwen3.5-0.8B (``d_model=1024``), ``~0.39`` at 4B
(2560) and ``~0.56`` at 27B (5120). So an off-block CKA of 0.2 at 0.8B is the null rather
than weak similarity, the same matrix read at 4B looks far more uniform for a reason that
has nothing to do with the model, and comparing a *raw* CKA matrix between two rungs
compares two different instruments. :func:`cka_null_baseline` measures the floor for the
configuration actually used; report it alongside the matrix, and scale the token set with
``d_model`` if the band is to be compared across rungs.

**Axis order.** The CKA matrix's axes are layers, and a permuted axis order produces a
matrix with no block structure — which is precisely the negative result S0 exists to be
able to report, so the bug would confirm itself rather than announce itself. Both entry
points therefore take ``layers`` explicitly and hand it back, exactly as
``dictionary_stack`` does, rather than trusting a dict's iteration order.

**Rank starvation in S4.** The centered dictionary has rank at most ``min(k - 1,
d_model)``, so a token set smaller than the residual stream *caps* the measurable
fraction at ``(k - 1) / d_model`` regardless of the model. The curve then flattens near
the top of the stack — which reads exactly like "the J-space saturates below full
dimensionality", a substantive-sounding claim that would be an artifact of the token set.
:class:`DimensionProfile` carries ``rank_limited`` and ``rank_cap`` so this cannot be
reported by accident. At Qwen3.5-0.8B (``d_model=1024``) the planned 4096-token set is
comfortable; at 27B (``d_model=5120``) it would not be.

**Centering.** Both statistics are about *variance across the token set*, so both center
the dictionary over the token axis. Skipping it makes S0's CKA dominated by the shared
mean direction — every layer looks similar to every other and the block structure washes
out — and makes S4 count the mean offset as a dimension. Neither raises.

Precision
---------

Stacks arrive as float32 (``dictionary_stack``'s default, chosen because stored lenses
are fp16 and Gram matrices throw away bits accumulated in the storage dtype — see
``f-2026-08-27-fp16-comparison-distortion``). Both reductions here upcast to float64:
they are ``d_model x d_model`` products and an eigendecomposition, cheap at these sizes,
and CKA is a ratio of Frobenius norms where the numerator and denominator are each sums
of ``d_model**2`` positive-ish terms.

See ``f-2026-08-18-jspace-construction-and-norm-gain`` for why the dictionary is
gamma-corrected before it reaches any of this, and the ``workspace-band-location`` spec
(T6, T7) for what the outputs are used for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = [
    "cka",
    "cka_matrix",
    "cka_null_baseline",
    "gram_cka",
    "BlockSplit",
    "three_block_split",
    "DimensionProfile",
    "effective_dimension",
]


def _centered(X: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Column-centre ``X`` [n, p] over the sample axis, in ``dtype``."""
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D [n_tokens, d_model] matrix, got {tuple(X.shape)}")
    if X.shape[0] < 2:
        raise ValueError(f"need at least 2 tokens to centre, got {X.shape[0]}")
    Xd = X.to(dtype)
    return Xd - Xd.mean(dim=0, keepdim=True)


def cka(X: torch.Tensor, Y: torch.Tensor, *, dtype: torch.dtype = torch.float64) -> float:
    """Linear CKA between two token-by-dimension matrices, in ``[0, 1]``.

    ``X`` and ``Y`` are ``[k, d]`` sets of J-lens vectors for the *same* token set at two
    layers. Both are centred over the token axis here; do not pre-centre.

    Computed in feature space as ``||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)``, which is
    algebraically identical to the Gram-matrix definition in the paper — "the matrices of
    pairwise similarities among J-lens vectors" — but costs ``d**2`` rather than ``k**2``.
    :func:`gram_cka` implements the definition literally and the two are asserted equal in
    the tests; this is the one to call.

    Returns 0.0 when either input has no variance at all (a layer of identical vectors),
    since the correlation is undefined rather than perfect there.
    """
    Xc, Yc = _centered(X, dtype), _centered(Y, dtype)
    if Xc.shape[0] != Yc.shape[0]:
        raise ValueError(
            f"token counts must match: {Xc.shape[0]} vs {Yc.shape[0]}. CKA compares two "
            "layers over one token set, not two token sets."
        )
    num = (Yc.T @ Xc).pow(2).sum()
    den_x = (Xc.T @ Xc).norm()
    den_y = (Yc.T @ Yc).norm()
    if den_x == 0 or den_y == 0:
        return 0.0
    return float(num / (den_x * den_y))


def gram_cka(X: torch.Tensor, Y: torch.Tensor, *, dtype: torch.dtype = torch.float64) -> float:
    """The paper's definition, literally: CKA over the ``[k, k]`` similarity matrices.

    ``tr(K H L H) / sqrt(tr(K H K H) tr(L H L H))`` with ``K = X X^T``, ``L = Y Y^T`` and
    ``H`` the centring matrix. Costs ``k**2`` and exists as the reference implementation
    :func:`cka` is checked against — an independent route to the same number, rather than
    a restatement of the same algebra. Use :func:`cka` in anything that runs at scale.
    """
    Xd, Yd = X.to(dtype), Y.to(dtype)
    if Xd.ndim != 2 or Yd.ndim != 2:
        raise ValueError("expected 2-D [n_tokens, d_model] matrices")
    n = Xd.shape[0]
    if Yd.shape[0] != n:
        raise ValueError(f"token counts must match: {n} vs {Yd.shape[0]}")
    K, L = Xd @ Xd.T, Yd @ Yd.T
    H = torch.eye(n, dtype=dtype) - torch.full((n, n), 1.0 / n, dtype=dtype)
    KH, LH = H @ K @ H, H @ L @ H
    num = (KH * LH).sum()
    den = torch.sqrt((KH * KH).sum() * (LH * LH).sum())
    if den == 0:
        return 0.0
    return float(num / den)


def cka_matrix(
    stack: torch.Tensor,
    layers: Sequence[int],
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, list[int]]:
    """S0: the layer-by-layer CKA matrix over one token set.

    Args:
        stack: ``[n_layers, k, d_model]`` from
            :func:`jlens_extensions.dictionary.dictionary_stack`.
        layers: The layer index of each row of ``stack``, as returned alongside it.
            Required rather than inferred — see the module docstring on axis order.

    Returns:
        ``([n_layers, n_layers], layers)``. Symmetric, unit diagonal.

    Run it on **all** fitted layers. The paper warns that the block structure's sharpness
    "is exaggerated by layer subsampling", so a subsampled matrix overstates exactly the
    thing S0 is evidence for.
    """
    if stack.ndim != 3:
        raise ValueError(f"expected [n_layers, k, d_model], got {tuple(stack.shape)}")
    if len(layers) != stack.shape[0]:
        raise ValueError(
            f"stack has {stack.shape[0]} layers but `layers` names {len(layers)}. These "
            "must correspond row-for-row; a mismatch silently permutes the CKA axes."
        )
    n = stack.shape[0]
    centred = [_centered(stack[i], dtype) for i in range(n)]
    # Precompute the self terms: each is used in n-1 off-diagonal entries.
    self_norm = [(C.T @ C).norm() for C in centred]

    C_out = torch.eye(n, dtype=dtype)
    for i in range(n):
        for j in range(i + 1, n):
            den = self_norm[i] * self_norm[j]
            if den == 0:
                value = 0.0
            else:
                value = float((centred[j].T @ centred[i]).pow(2).sum() / den)
            C_out[i, j] = value
            C_out[j, i] = value
    return C_out, list(layers)


def cka_null_baseline(
    n_tokens: int,
    d_model: int,
    *,
    n_draws: int = 8,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """The CKA two unrelated dictionaries score at this token count and width.

    Monte Carlo over independent Gaussian pairs. Report it alongside any CKA matrix: the
    statistic's floor is ``~d_model / n_tokens`` rather than zero, so "the off-block
    entries are around 0.2" is a statement about the token set until this number is in
    hand. See the module docstring.

    Returns ``{"mean", "max", "min", "n_draws", "n_tokens", "d_model", "ratio_to_d_over_k"}``.

    Gaussian vectors are not J-lens vectors, so treat this as the floor imposed by the
    *shape* of the measurement — the point at which the matrix stops discriminating —
    rather than as a null the real dictionary would produce if the model had no band. The
    random-token-set arm is what tests the token set; this tests the statistic.
    """
    if n_tokens < 2 or d_model < 1 or n_draws < 1:
        raise ValueError(
            f"need n_tokens >= 2, d_model >= 1, n_draws >= 1; got {n_tokens}, {d_model}, {n_draws}"
        )
    values = []
    for draw in range(n_draws):
        g1 = torch.Generator().manual_seed(seed + 2 * draw)
        g2 = torch.Generator().manual_seed(seed + 2 * draw + 1)
        X = torch.randn(n_tokens, d_model, generator=g1, dtype=dtype)
        Y = torch.randn(n_tokens, d_model, generator=g2, dtype=dtype)
        values.append(cka(X, Y, dtype=dtype))
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "max": max(values),
        "min": min(values),
        "n_draws": n_draws,
        "n_tokens": n_tokens,
        "d_model": d_model,
        "ratio_to_d_over_k": mean / (d_model / n_tokens),
    }


@dataclass(frozen=True)
class BlockSplit:
    """A three-way contiguous partition of the layer axis, and how well it separates.

    Attributes:
        boundaries: ``(i, j)`` — index positions into ``layers``, with the early block
            ``[0, i)``, the middle ``[i, j)`` and the late ``[j, n)``.
        layer_spans: The same three blocks as inclusive ``(first_layer, last_layer)``
            pairs of actual layer indices.
        middle_fraction: The middle block as ``(start, end)`` fractions of model depth,
            each ``layer / (depth - 1)`` so that 0.0 is the first layer and 1.0 the final
            one. This is the portable form — absolute indices do not carry between a
            24-layer and a 32-layer model.
        within: Mean CKA over distinct same-block pairs.
        between: Mean CKA over cross-block pairs.
        contrast: ``within - between``. The separation statistic.
    """

    boundaries: tuple[int, int]
    layer_spans: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    middle_fraction: tuple[float, float]
    within: float
    between: float
    contrast: float


def three_block_split(
    C: torch.Tensor,
    layers: Sequence[int],
    *,
    depth: int | None = None,
    min_block: int = 2,
) -> BlockSplit:
    """Find the contiguous three-block partition of a CKA matrix with the best contrast.

    Exhaustive over both boundaries — ``O(n**2)`` candidates at ``n <= 32``, so there is
    no reason to approximate — maximising mean within-block CKA minus mean between-block
    CKA.

    **This always returns a split.** Every matrix has a best three-way partition,
    including one with no block structure at all, so the return value is not evidence that
    a band exists. ``contrast`` is the statistic that bears on that question, and it is
    deliberately left for the caller to judge against the random-token-set arm rather than
    thresholded here: "the block structure appears on the frequent-token set and not the
    random one" is a result about the instrument, and T6 requires reporting it as one
    rather than resolving it by picking the token set that worked.

    Args:
        C: ``[n, n]`` CKA matrix from :func:`cka_matrix`.
        layers: The layer index of each axis position.
        depth: The model's total layer count, for the depth fractions. Defaults to
            ``max(layers) + 2`` — the fitted layers stop one below the target, so this
            recovers the model's depth from a full fitted stack. Pass it explicitly if the
            stack is subsampled or restricted.
        min_block: Smallest admissible block. At least 2, so ``within`` is defined.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"expected a square CKA matrix, got {tuple(C.shape)}")
    n = C.shape[0]
    if len(layers) != n:
        raise ValueError(f"CKA matrix is {n}x{n} but `layers` names {len(layers)}")
    if min_block < 2:
        raise ValueError("min_block must be at least 2 for `within` to be defined")
    if n < 3 * min_block:
        raise ValueError(
            f"need at least {3 * min_block} layers for three blocks of {min_block}; got {n}"
        )
    if depth is None:
        depth = int(max(layers)) + 2
    if depth < 2:
        raise ValueError(f"depth must be at least 2, got {depth}")

    Cd = C.to(torch.float64)
    best: BlockSplit | None = None
    for i in range(min_block, n - 2 * min_block + 1):
        for j in range(i + min_block, n - min_block + 1):
            blocks = [(0, i), (i, j), (j, n)]
            within_sum = within_count = 0.0
            for lo, hi in blocks:
                sub = Cd[lo:hi, lo:hi]
                # Off-diagonal only: the unit diagonal is an artifact of the statistic,
                # not evidence of within-block coherence.
                within_sum += float(sub.sum() - sub.diagonal().sum())
                within_count += (hi - lo) * (hi - lo - 1)
            total_sum = float(Cd.sum() - Cd.diagonal().sum())
            total_count = n * (n - 1)
            between_sum = total_sum - within_sum
            between_count = total_count - within_count
            within = within_sum / within_count
            between = between_sum / between_count if between_count else 0.0
            contrast = within - between
            if best is None or contrast > best.contrast:
                spans = tuple(
                    (int(layers[lo]), int(layers[hi - 1])) for lo, hi in blocks
                )
                mid_lo, mid_hi = spans[1]
                best = BlockSplit(
                    boundaries=(i, j),
                    layer_spans=spans,  # type: ignore[arg-type]
                    middle_fraction=(mid_lo / (depth - 1), mid_hi / (depth - 1)),
                    within=within,
                    between=between,
                    contrast=contrast,
                )
    assert best is not None  # n >= 3 * min_block guarantees one candidate
    return best


@dataclass(frozen=True)
class DimensionProfile:
    """S4's curve, plus what bounds it.

    Attributes:
        shares: The variance shares the curve was computed at, ascending.
        fractions: ``[n_shares, n_layers]`` — the fraction of ``d_model`` needed to reach
            each share, per layer.
        components: ``[n_shares, n_layers]`` — the same as raw component counts.
        layers: Layer index per column.
        d_model: Residual stream width, the denominator of ``fractions``.
        rank_cap: ``min(k - 1, d_model) / d_model`` — the largest fraction this token set
            can express.
        rank_limited: True when ``k - 1 < d_model``, i.e. the token set, not the model,
            bounds the curve.
    """

    shares: tuple[float, ...]
    fractions: torch.Tensor
    components: torch.Tensor
    layers: list[int]
    d_model: int
    rank_cap: float
    rank_limited: bool


def effective_dimension(
    stack: torch.Tensor,
    layers: Sequence[int],
    *,
    shares: Sequence[float] = (0.5, 0.9, 0.99),
    dtype: torch.dtype = torch.float64,
) -> DimensionProfile:
    """S4: the fraction of residual-stream dimensions carrying a given share of variance.

    Per layer, centre the dictionary over the token axis, take the eigenspectrum of its
    covariance, and count the smallest number of components whose cumulative share of
    total variance reaches each of ``shares``. Divide by ``d_model``.

    Args:
        stack: ``[n_layers, k, d_model]`` from ``dictionary_stack``.
        layers: Layer index per row of ``stack``.
        shares: Variance shares to report at. Several by default, because the paper fixes
            none and a shape visible at only one share is a fact about the threshold. The
            share used **must** be stated wherever a number from this is quoted.

    The token set bounds the answer: with ``k`` tokens the centred dictionary has rank at
    most ``k - 1``, so when ``k - 1 < d_model`` the curve saturates at ``(k - 1) /
    d_model`` and its plateau is an artifact. ``rank_limited`` flags it; check it before
    reading a flat top as a property of the model.
    """
    if stack.ndim != 3:
        raise ValueError(f"expected [n_layers, k, d_model], got {tuple(stack.shape)}")
    if len(layers) != stack.shape[0]:
        raise ValueError(
            f"stack has {stack.shape[0]} layers but `layers` names {len(layers)}"
        )
    shares_t = tuple(float(s) for s in shares)
    if not shares_t:
        raise ValueError("need at least one variance share")
    if any(not (0.0 < s <= 1.0) for s in shares_t):
        raise ValueError(f"shares must lie in (0, 1], got {shares_t}")
    if list(shares_t) != sorted(shares_t):
        raise ValueError(f"shares must be ascending, got {shares_t}")

    n_layers, k, d_model = stack.shape
    components = torch.zeros((len(shares_t), n_layers), dtype=torch.int64)
    for col in range(n_layers):
        Xc = _centered(stack[col], dtype)
        # Singular values of the centred matrix; eigenvalues of the covariance are
        # s**2 / (k - 1), and the constant cancels in a cumulative *share*.
        s = torch.linalg.svdvals(Xc)
        if s.numel() == 0 or float(s[0]) == 0.0:
            continue  # a layer of identical vectors: 0 components for every share
        # Discard numerically-zero directions before counting, on the same tolerance
        # torch.linalg.matrix_rank uses. Without this, a dictionary of exact rank r
        # reports the full d_model at share 1.0: the tail singular values are ~1e-16
        # rather than 0, so the cumulative share reaches 1 only at the last index. The
        # tail is rounding noise, not dimensions of the J-space.
        tol = float(s[0]) * max(Xc.shape) * torch.finfo(dtype).eps
        s = s[s > tol]
        var = s.pow(2)
        total = var.sum()
        cumulative = torch.cumsum(var, dim=0) / total
        for row, share in enumerate(shares_t):
            # First index whose cumulative share reaches the target; +1 turns a 0-based
            # index into a component count. The empty case is share=1.0 against a
            # cumulative sum that lands a rounding error below it — take the full rank.
            reached = (cumulative >= share).nonzero()
            count = int(s.shape[0]) if reached.numel() == 0 else int(reached[0]) + 1
            components[row, col] = count

    rank_cap_n = min(k - 1, d_model)
    return DimensionProfile(
        shares=shares_t,
        fractions=components.to(torch.float64) / d_model,
        components=components,
        layers=list(layers),
        d_model=d_model,
        rank_cap=rank_cap_n / d_model,
        rank_limited=(k - 1) < d_model,
    )
