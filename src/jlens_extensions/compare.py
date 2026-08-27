"""Comparing two lenses, and the floors that bound what a difference can mean.

T16 asks whether our fit reproduces Neuronpedia's published artifact. That question
only has an answer relative to the resolution of the instrument, and there are two
different limits in play:

* **fp16 storage.** Published lenses are fp16 snapshots of an fp32 computation, so
  element-wise relative error against one is floored at fp16 machine epsilon,
  ``2**-11 = 4.9e-4``, however perfectly the fit is reproduced.
* **Run-to-run nondeterminism.** Two runs of identical code on identical inputs
  differ, worst at the earliest layer. It is per-layer and spans ~800x across the
  stack -- see ``d-2026-08-24-fit-nondeterminism-envelope``.

Whichever is larger binds, and which one that is *changes by layer*.

A trap this module exists to avoid
----------------------------------

**An fp16 comparison does not measure a sub-quantum difference; it inflates it.**

fp16 quantisation is deterministic and shared, so it is tempting to reason that two
fp32 tensors closer than one quantum round to the same value and their difference is
simply erased. That is what happens to most entries -- but not all. An entry whose two
values straddle a rounding boundary rounds to values a *full quantum* apart, which for
a sub-quantum difference is an amplification. Erasure and amplification both occur, and
the amplified entries dominate the Frobenius norm.

Measured on uniform relative perturbations of a random matrix, the fp16-measured
difference tracks roughly ``sqrt(delta * q)`` -- the geometric mean of the true
difference and the quantum -- for ``delta < q``:

===============  ===============  =====
true rel. diff   fp16-measured    ratio
===============  ===============  =====
7.6e-6           7.4e-5           9.7x
1.2e-4           2.9e-4           2.4x
4.9e-4 (= q)     5.9e-4           1.2x
2.0e-3           2.0e-3           1.0x
===============  ===============  =====

So an fp16 comparison is faithful only above roughly ``2q`` (~1e-3 relative), and
below that it reports a number governed by the storage format rather than by the fits.

Why this matters for reading T18. Its envelope was measured on **fp16-stored** lenses
(``--save_dtype`` did not exist yet), which is visible in its own numbers: the max
absolute difference is reported as exactly ``1.953e-3 = 2**-9``, a power of two, which
is the fp16 ULP for values in ``[2, 4)`` rather than anything a float comparison would
produce. Its L0 figure (1.8e-3 at n=20) sits above ``2q`` and is trustworthy. Its
high-layer figures do not, so how much of those is fit spread and how much is storage
is not recoverable from that measurement.

Our T15 lenses are fp32, so :func:`compare_lenses` on them measures the real thing. To
compare *like with like* against T18, round-trip both sides through fp16 first --
:func:`fp16_roundtrip` -- and read the two results side by side. Reporting only the
fp32 number against T18's fp16 one would attribute a storage artifact to
``torch.compile``, and the artifact points the wrong way: it would look like compile
had *reduced* the noise.

The synthetic table above is a uniform perturbation of a random matrix, which a
Jacobian near the identity is not. Treat it as establishing the direction and rough
shape of the distortion, not as a correction factor to divide T18's numbers by.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FP16_EPS",
    "FP16_FLOOR",
    "LayerDiff",
    "fp16_roundtrip",
    "rel_frobenius",
    "compare_lenses",
    "identity_distance",
    "replay_early_stop",
    "binding_constraint",
]

#: fp16 machine epsilon: 10 explicit mantissa bits, so 2**-11.
FP16_EPS = 2.0**-11

#: The element-wise relative-error floor against any fp16-stored lens. One
#: quantisation draw -- comparing two fp16 tensors to each other is ~sqrt(2) worse.
FP16_FLOOR = FP16_EPS


@dataclass
class LayerDiff:
    """One layer's difference between two lenses."""

    layer: int
    rel_frobenius: float
    abs_frobenius: float
    max_abs: float
    norm_a: float
    norm_b: float
    n_differing: int
    n_entries: int

    @property
    def frac_differing(self) -> float:
        return self.n_differing / self.n_entries if self.n_entries else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "rel_frobenius": self.rel_frobenius,
            "abs_frobenius": self.abs_frobenius,
            "max_abs": self.max_abs,
            "norm_a": self.norm_a,
            "norm_b": self.norm_b,
            "n_differing": self.n_differing,
            "n_entries": self.n_entries,
            "frac_differing": self.frac_differing,
        }


def fp16_roundtrip(tensor):
    """Cast to fp16 and back, in fp32. Models what storing a lens costs it.

    Used to put an fp32 pair on the same footing as a comparison made between two
    fp16-stored lenses -- see the module docstring.
    """
    import torch

    return tensor.to(torch.float16).to(torch.float32)


def rel_frobenius(a, b) -> float:
    """``||a - b||_F / ||b||_F``.

    ``b`` is the denominator, so pass the reference second -- the published lens, or
    for a symmetric pair either one, since the norms agree to many digits. Both norms
    are recorded on :class:`LayerDiff` so a caller can renormalise without a re-run.
    """
    denom = b.norm().item()
    if denom == 0.0:
        raise ValueError("reference tensor has zero Frobenius norm")
    return (a - b).norm().item() / denom


def compare_lenses(
    a: Mapping[int, Any],
    b: Mapping[int, Any],
    *,
    as_fp16: bool = False,
) -> list[LayerDiff]:
    """Per-layer comparison of two Jacobian dicts.

    Args:
        a: Layer -> tensor. Compared against ``b``.
        b: Layer -> tensor. The denominator of the relative figures.
        as_fp16: Round-trip both sides through fp16 first, to compare like with like
            against a measurement made on fp16-stored lenses.

    Raises:
        ValueError: if the two lenses do not cover the same layers, which would make
            a per-layer table silently misaligned.
    """
    import torch

    if set(a) != set(b):
        raise ValueError(
            f"lenses cover different layers: {sorted(set(a) ^ set(b))} differ. "
            f"A per-layer comparison across mismatched stacks would misalign silently."
        )

    diffs: list[LayerDiff] = []
    for layer in sorted(a):
        x, y = a[layer].to(torch.float32), b[layer].to(torch.float32)
        if as_fp16:
            x, y = fp16_roundtrip(x), fp16_roundtrip(y)
        delta = x - y
        diffs.append(
            LayerDiff(
                layer=layer,
                rel_frobenius=delta.norm().item() / y.norm().item(),
                abs_frobenius=delta.norm().item(),
                max_abs=delta.abs().max().item(),
                norm_a=x.norm().item(),
                norm_b=y.norm().item(),
                n_differing=int((delta != 0).sum().item()),
                n_entries=delta.numel(),
            )
        )
    return diffs


def identity_distance(jacobians: Mapping[int, Any], layer: int | None = None) -> float:
    """``||J_bar_late - I||_F / sqrt(d)``, recomputed from the tensor.

    T16 requires this be recomputed rather than read from ``config.yaml``: the
    published value is transcribed from the final row of the convergence CSV, so
    reading it back would compare a number against a copy of itself rather than
    against the artifact.

    ``layer`` defaults to ``max(jacobians)``, which is ``fitting.py``'s ``late_layer``.
    """
    import torch

    layer = max(jacobians) if layer is None else layer
    J = jacobians[layer].to(torch.float32)
    d = J.shape[0]
    return (J - torch.eye(d)).norm().item() / math.sqrt(d)


def replay_early_stop(
    deltas: Sequence[float],
    *,
    stop_at_delta: float = 0.002,
    min_prompts: int = 100,
    window: int = 10,
) -> dict[str, Any]:
    """Where the published early-stopping rule *would* have fired on a trace.

    T15 pins ``--n_prompts`` and disables early stopping, which removes
    ``prompts_fitted`` as a comparison axis by construction. This puts it back as a
    *derived* quantity: apply the published run's own stop rule to our completed
    trace and see where it would have triggered. No brittleness, because nothing
    about the run depended on it.

    Mirrors ``fit_lens.ConvergenceTracker.record`` exactly, including that NaN
    entries (the first prompt) never enter the window.

    Args:
        deltas: ``mean_rel_change`` per prompt, in order, NaNs included.
        stop_at_delta: Published value, 0.002.
        min_prompts: Published value, 100.
        window: Published value, 10.
    """
    recent: list[float] = []
    n_done = 0
    for delta in deltas:
        n_done += 1
        if delta != delta:  # NaN, as the tracker tests it
            continue
        recent.append(delta)
        if len(recent) > window:
            recent.pop(0)
        if n_done >= min_prompts and len(recent) == window:
            smoothed = sum(recent) / window
            if smoothed < stop_at_delta:
                return {
                    "would_stop_at": n_done,
                    "smoothed_delta": smoothed,
                    "rule": {
                        "stop_at_delta": stop_at_delta,
                        "min_prompts": min_prompts,
                        "window": window,
                    },
                }
    final = sum(recent) / len(recent) if recent else float("nan")
    return {
        "would_stop_at": None,
        "smoothed_delta_final": final,
        "rule": {
            "stop_at_delta": stop_at_delta,
            "min_prompts": min_prompts,
            "window": window,
        },
    }


def binding_constraint(
    rel_diff: float,
    *,
    envelope: float | None,
    fp16_floor: float = FP16_FLOOR,
) -> dict[str, Any]:
    """Which floor bounds a difference, and whether it clears it.

    ``envelope`` is the measured run-to-run spread for this layer, or ``None`` where
    none was measured. The larger of the two floors binds; a difference at or under
    it is not evidence of disagreement, only of instrument resolution.
    """
    floors = {"fp16_storage": fp16_floor}
    if envelope is not None:
        floors["nondeterminism"] = envelope
    which = max(floors, key=lambda k: floors[k])
    floor = floors[which]
    return {
        "floor": floor,
        "binds": which,
        "ratio_to_floor": rel_diff / floor if floor else float("inf"),
        "above_floor": rel_diff > floor,
    }


def fp16_storage_floor(jacobians: Mapping[int, Any]) -> dict[int, float]:
    """Per-layer fp16 storage floor, **measured** rather than derived.

    The ~4.9e-4 figure quoted throughout is ``2**-11``, fp16's machine epsilon: the
    worst-case *element-wise* relative error. A per-layer comparison reports a
    *Frobenius-aggregate* relative difference, which is a different and smaller
    quantity -- quantisation errors are spread over a million entries and partially
    cancel, and how much they cancel depends on the distribution of ``J``'s entries.

    Using the element-wise bound as if it were the aggregate floor is conservative in
    the direction that matters: it makes agreement look better than it is, because a
    difference is scored against a floor larger than the one that actually applies.

    This measures it instead. Round-trip our own fp32 lens through fp16 and take
    ``||J - fp16(J)||_F / ||J||_F`` per layer -- exactly the operation the published
    artifact underwent, on a tensor with the right distribution.

    One draw, not two. The published lens is quantised and ours (at fp32) is not, so
    a ours-vs-published comparison carries exactly one of these. Two fp16 lenses
    compared to each other carry ~sqrt(2) of it.
    """
    return {
        layer: (J - fp16_roundtrip(J)).norm().item() / J.norm().item()
        for layer, J in sorted(jacobians.items())
    }


def identity_fraction(jacobians: Mapping[int, Any]) -> dict[int, float]:
    """Per-layer ``||J - I||_F / ||J||_F`` -- how much of the norm is *not* the identity.

    A Jacobian through a residual stream is near-identity, so a difference normalised
    by ``||J||`` is normalised largely by the identity diagonal. This reports how much
    of the norm carries actual signal, so a reader can rescale to the deviation if they
    want the more demanding view.

    It does **not** change any verdict on its own: the storage floor and the run-to-run
    envelope are both normalised the same way, so a difference and its floor rescale
    together and their ratio is invariant. It is reported because "is this metric being
    washed out by the identity?" is a question the criteria explicitly raise, and the
    answer should be a number rather than an argument.
    """
    import torch

    out = {}
    for layer, J in sorted(jacobians.items()):
        d = J.shape[0]
        out[layer] = (J - torch.eye(d, dtype=J.dtype)).norm().item() / J.norm().item()
    return out


def layer_correspondence(
    ours: Mapping[int, Any],
    reference: Mapping[int, Any],
) -> dict[str, Any]:
    """Negative control: does the metric put the *right* layer pairing first?

    A1 requires a control -- "a criterion that only ever sees agreement has not been
    shown to be able to see disagreement". The criterion names a different model's
    published lens, which is not available at this shape: no other published lens has
    ``d_model=1024``, so a per-layer Frobenius comparison against one is undefined.

    This is the shape-safe equivalent, and a stronger test of the same thing. Compare
    every ``ours[i]`` against every ``reference[j]`` and check that ``j == i`` wins each
    row. It rules out the three failure modes the criterion names -- a wrong file loaded,
    two near-identity tensors matching trivially, a normalisation that washes the
    difference out -- because all three would flatten this grid, and it additionally
    verifies the layer indexing, which a cross-model comparison cannot.

    Returns the per-row winner, the margin over the runner-up, and whether the diagonal
    won everywhere.
    """
    layers = sorted(ours)
    rows = []
    for i in layers:
        scores = {j: rel_frobenius(ours[i], reference[j]) for j in sorted(reference)
                  if reference[j].shape == ours[i].shape}
        if not scores:
            continue
        ranked = sorted(scores, key=lambda j: scores[j])
        best, runner = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        rows.append({
            "layer": i,
            "best_match": best,
            "correct": best == i,
            "self_score": scores.get(i),
            "best_score": scores[best],
            "runner_up": runner,
            "runner_up_score": scores[runner] if runner is not None else None,
            "margin_x": (scores[runner] / scores[best]) if runner is not None and scores[best] else None,
        })
    correct = [r for r in rows if r["correct"]]
    margins = [r["margin_x"] for r in rows if r["margin_x"] is not None]
    return {
        "rows": rows,
        "n_layers": len(rows),
        "n_correct": len(correct),
        "all_correct": len(correct) == len(rows) and bool(rows),
        "min_margin_x": min(margins) if margins else None,
        "median_margin_x": sorted(margins)[len(margins) // 2] if margins else None,
    }
