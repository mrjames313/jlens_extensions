"""The one place a `fit_lens.py` command is built, so the gate cannot be forgotten.

``torch.compile`` miscompiles this model in 30-50% of processes and the lens it then
saves looks entirely valid -- right shape, right dtype, plausible size. The defence is a
check on prompt 1's ``identity_distance``
(``f-2026-08-28-compile-miscompilation``), and a defence that each caller has to
remember to pass is one that will eventually be missed. T15 and the dim_batch probe
already built their command lists independently and neither carried it.

So the command is built here, the gate argument is **required rather than optional**, and
a caller with no reference value has to say so explicitly by passing
``gate_identity=UNGATED`` -- which is greppable, unlike an omission.

The reference is per model, per corpus and per box, so it lives in the machine profile
next to ``dim_batch`` and ``s_per_prompt``. :func:`probe_reference_identity` measures it
in about 90 seconds by fitting one prompt uncompiled, which is the configuration that has
never failed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "UNGATED",
    "MissingGateError",
    "build_fit_command",
    "probe_reference_identity",
]

#: Explicit opt-out. Passing this instead of a number records a deliberate choice in the
#: caller's source, where omission would have recorded nothing.
UNGATED = "ungated"


class MissingGateError(RuntimeError):
    """A fit command was built with neither a reference value nor an explicit opt-out."""


def build_fit_command(
    *,
    fit_lens_path: Path | str,
    model_id: str,
    out_dir: Path | str,
    n_prompts: int,
    dim_batch: int,
    gate_identity: float | str,
    max_seq_len: int = 128,
    dtype: str = "bfloat16",
    save_dtype: str = "float32",
    device_map: str = "cuda",
    compile_blocks: str = "auto",
    no_compile: bool = False,
    gate_tol: float = 0.01,
    dataset: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    dataset_split: str = "train",
    text_field: str = "text",
    max_chars: int = 2000,
    python: str | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """Assemble a `fit_lens.py` argv.

    ``gate_identity`` is keyword-only and has no default: a caller must supply a
    reference value or :data:`UNGATED`. That is the whole point of this function.

    ``--hf_cache_dir`` is deliberately never passed -- ``fit_lens.py`` ``rmtree``s it in a
    ``finally`` unless ``--keep_hf_cache`` is also given, which would re-download every
    weight per run. Confine downloads with ``Config.hf_env()`` instead.
    """
    if gate_identity is None:
        raise MissingGateError(
            "build_fit_command needs gate_identity. Pass the model's reference "
            "prompt-1 identity_distance (from the machine profile, or measure it with "
            "probe_reference_identity), or pass UNGATED to opt out deliberately.\n"
            "An ungated compiled fit has a 30-50% chance of silently saving a corrupt "
            "lens -- see f-2026-08-28-compile-miscompilation."
        )
    if isinstance(gate_identity, str) and gate_identity != UNGATED:
        raise MissingGateError(
            f"gate_identity must be a number or UNGATED, got {gate_identity!r}"
        )

    cmd = [
        python or sys.executable, str(fit_lens_path), model_id,
        "--out_dir", str(out_dir),
        "--n_prompts", str(n_prompts),
        "--dim_batch", str(dim_batch),
        "--max_seq_len", str(max_seq_len),
        "--dtype", dtype,
        "--device_map", device_map,
        "--save_dtype", save_dtype,
        "--dataset", dataset,
        "--dataset_config", dataset_config,
        "--dataset_split", dataset_split,
        "--text_field", text_field,
        "--max_chars", str(max_chars),
    ]
    if no_compile:
        cmd.append("--no_compile")
    else:
        cmd += ["--compile_blocks", compile_blocks]
        if gate_identity != UNGATED:
            cmd += ["--gate_identity", repr(float(gate_identity)),
                    "--gate_tol", repr(float(gate_tol))]
    cmd += list(extra)
    return cmd


def probe_reference_identity(
    model_id: str,
    *,
    prompt: str,
    dim_batch: int = 8,
    max_seq_len: int = 128,
) -> dict[str, Any]:
    """Measure prompt 1's ``identity_distance`` **uncompiled**, as the gate's reference.

    Uncompiled is the only configuration that has never failed: 0/12 draws, reproducing
    to six decimal places. About 90 seconds, and it makes the gate self-contained --
    no per-model constant to look up, and nothing to go stale when a model or corpus
    changes.

    Assumes ``jlens`` is importable, i.e. the caller has ``harness/`` on ``sys.path``
    the way ``fit_lens.py`` does.
    """
    import torch
    import transformers

    import jlens
    from jlens.fitting import jacobian_for_prompt

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok, compile=False)

    source_layers = list(range(model.n_layers - 1))
    J, seq_len, n_valid = jacobian_for_prompt(
        model, prompt, source_layers, target_layer=None,
        dim_batch=dim_batch, max_seq_len=max_seq_len,
    )
    late = max(source_layers)
    d_model = J[late].shape[0]
    identity = (J[late].float() - torch.eye(d_model)).norm().item() / d_model**0.5
    return {
        "identity_distance": identity,
        "basis": "uncompiled single prompt",
        "model": model_id,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "seq_len": seq_len,
        "n_valid_positions": n_valid,
    }
