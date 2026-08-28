"""Which residual blocks to compile, and the gate that catches it when we get it wrong.

Measured on `gx10-ace5` 2026-08-28: ``torch.compile`` miscompiles Qwen3.5-0.8B in
30-50% of processes, silently, and the fault is localised to the **full-attention**
blocks. See ``f-2026-08-28-compile-miscompilation``.

    compiled                 blocks   failed   median s/child
    all                          24    10/20             42.5
    full-attention only           6     6/20             68.0
    linear-attention only        18     0/20             43.0
    none                          0     0/12             79.5

Two things make this actionable rather than merely alarming. The safe subset is the
*large* one -- 18 of 24 blocks -- so skipping the six risky blocks costs 0.5 s/child,
because those six were contributing almost none of the speedup. And the failure is
per-process rather than per-prompt: a process either compiles correctly, and every
prompt in it is right, or it does not, and every prompt is wrong. That is what makes a
single check at prompt 1 sufficient.

**0/20 is not zero.** The 95% upper bound on that rate is 14%. The policy narrows the
risk; the gate is what catches what is left.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = [
    "POLICIES",
    "ensure_cuda_context",
    "DEFAULT_POLICY",
    "IDENTITY_TOL",
    "CompileGateError",
    "classify_blocks",
    "select_block_indices",
    "apply_compile_policy",
    "identity_in_band",
    "check_identity_gate",
]

#: ``auto`` is the default and means: on a hybrid model compile only the
#: linear-attention blocks, on a homogeneous one compile everything. Stated as a rule
#: rather than a per-model table so it needs no maintenance for the next model.
POLICIES = ("auto", "all", "linear-attn", "full-attn", "none")
DEFAULT_POLICY = "auto"

#: A sound run's prompt-1 identity_distance sits within 0.05% of the reference; every
#: observed failure is at least 2.3% away. 1% separates them with room on both sides.
IDENTITY_TOL = 0.01


class CompileGateError(RuntimeError):
    """A fit produced an identity_distance outside its expected band.

    Raised rather than warned. The failure mode this guards against yields a lens that
    looks like a lens -- right shape, right dtype, plausible file size -- so anything
    short of refusing to continue leaves a corrupt artifact on disk.
    """


def classify_blocks(layers: Sequence[Any]) -> dict[str, list[int]]:
    """Split residual blocks by kind.

    A block carrying ``.linear_attn`` is a Gated DeltaNet / state-space block; one
    without it is full attention. Qwen3.5 alternates them roughly 3:1.
    """
    linear = [i for i, b in enumerate(layers) if hasattr(b, "linear_attn")]
    full = [i for i in range(len(layers)) if i not in set(linear)]
    return {"linear-attn": linear, "full-attn": full}


def select_block_indices(layers: Sequence[Any], policy: str = DEFAULT_POLICY) -> list[int]:
    """Block indices to compile under ``policy``."""
    if policy not in POLICIES:
        raise ValueError(f"unknown compile policy {policy!r}; expected one of {POLICIES}")
    kinds = classify_blocks(layers)
    if policy == "none":
        return []
    if policy == "all":
        return list(range(len(layers)))
    if policy in ("linear-attn", "full-attn"):
        return list(kinds[policy])
    # auto: hybrid -> the safe majority; homogeneous -> everything.
    hybrid = bool(kinds["linear-attn"]) and bool(kinds["full-attn"])
    return list(kinds["linear-attn"]) if hybrid else list(range(len(layers)))


def apply_compile_policy(model: Any, policy: str = DEFAULT_POLICY) -> dict[str, Any]:
    """Compile the blocks ``policy`` selects, in place, and describe what was done.

    Deliberately applied *after* ``from_hf(compile=False)`` rather than by changing
    ``hf.py``: the vendored library stays as Neuronpedia wrote it, and the policy lives
    in our own package where it belongs.
    """
    import torch

    chosen = select_block_indices(model.layers, policy)
    for i in chosen:
        model.layers[i] = torch.compile(model.layers[i], mode="default", dynamic=False)
    kinds = classify_blocks(model.layers)
    return {
        "policy": policy,
        "compiled_indices": chosen,
        "n_compiled": len(chosen),
        "n_blocks": len(model.layers),
        "n_linear_attn": len(kinds["linear-attn"]),
        "n_full_attn": len(kinds["full-attn"]),
        "hybrid": bool(kinds["linear-attn"]) and bool(kinds["full-attn"]),
    }


def identity_in_band(value: float, expected: float, tol: float = IDENTITY_TOL) -> bool:
    """Is ``value`` within ``tol`` relative of ``expected``?"""
    if expected == 0:
        raise ValueError("expected identity_distance must be non-zero")
    if value != value:  # NaN
        return False
    return abs(value - expected) / abs(expected) <= tol


def check_identity_gate(
    value: float,
    expected: float,
    *,
    tol: float = IDENTITY_TOL,
    n_done: int = 1,
    context: str = "",
) -> None:
    """Raise :class:`CompileGateError` if ``value`` is out of band.

    Call at prompt 1. Because the failure is per-process, a sound first prompt means a
    sound run, and a bad one means an hour wasted if it is not caught here.
    """
    if identity_in_band(value, expected, tol):
        return
    off = abs(value - expected) / abs(expected)
    raise CompileGateError(
        f"identity_distance={value:.6f} at prompt {n_done} is {off:.1%} from the "
        f"expected {expected:.6f} (tolerance {tol:.1%}).\n"
        f"{context}\n"
        f"This is the torch.compile miscompilation described in "
        f"f-2026-08-28-compile-miscompilation: it is per-process, so re-running is "
        f"usually enough. If it repeats, fit with --compile_blocks none.\n"
        f"Refusing to continue -- the lens this run would save looks valid and is not."
    )


def ensure_cuda_context() -> dict[str, Any]:
    """Initialise the CUDA context and the autograd worker threads before real work.

    Every run of this fitting path emits, on its first backward::

        UserWarning: Attempting to run cuBLAS, but there was no current CUDA context!
        Attempting to set the primary context...
        (from Variable._execution_engine.run_backward)

    The autograd engine runs backwards on **worker threads**. Moving the model to the
    GPU establishes the primary context on the *main* thread, so the first backward on a
    fresh worker finds none current on its own thread and cuBLAS attaches one lazily. It
    succeeds, and the value it produces is correct -- the uncompiled probe returns
    0.531523, matching three prior runs to six decimal places.

    So this is not a fix for a known bug. It is a **hypothesis worth testing**: lazy
    context attachment on a worker thread is per-process state that varies between
    otherwise identical runs, which is the shape of the miscompilation in
    ``f-2026-08-28-compile-miscompilation``. Inductor-generated kernels may be less
    tolerant of it than eager cuBLAS, which would fit uncompiled never failing while
    compiled fails 30-50% of the time.

    It does not obviously explain the block-kind localisation, so treat it as a lead
    rather than an explanation. ``dim_batch_diagnosis.py --warmup-context`` measures
    whether it changes the failure rate; until that says otherwise, the gate is what
    protects a fit.

    The warm-up runs a matmul **and its backward**, because a forward alone initialises
    cuBLAS on the main thread and leaves the worker threads -- the ones that emit the
    warning -- untouched.
    """
    import torch

    if not torch.cuda.is_available():
        return {"warmed": False, "reason": "no CUDA device"}
    torch.cuda.init()
    a = torch.zeros(8, 8, device="cuda", requires_grad=True)
    (a @ a).sum().backward()
    torch.cuda.synchronize()
    return {"warmed": True, "device": torch.cuda.get_device_name(0)}
