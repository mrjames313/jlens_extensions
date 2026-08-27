"""Where does `dim_batch` change the answer? One prompt, no fit.

Diagnostic, not a measurement. Written because `dim_batch_neutrality.py` returned a
result that cannot be taken at face value.

The contradiction
-----------------

Fitting at ``dim_batch=64`` and comparing against our ``dim_batch=8`` pair gave per-layer
relative Frobenius differences of **1.0 to 4.1** -- not a small numerical effect, but
essentially unrelated tensors. Yet T16 found our ``dim_batch=8`` lens agreed with
Neuronpedia's, fitted at ``dim_batch=128``, to **1.4e-3**. Both cannot be true of a
well-behaved estimator: if slicing width mattered at order 1, an 8-wide fit could not
match a 128-wide reference at 1e-3.

So the ``dim_batch=64`` result is a symptom of something, and what it is a symptom of has
to be established before any of it is written up.

What is already known, from the harness's own diagnostics rather than from any comparison
code of ours:

* ``identity_distance`` on the **first prompt** is 0.531268 at ``dim_batch=8`` and 0.543
  at 64. One prompt, one Jacobian -- so this is inside ``jacobian_for_prompt``, not in
  accumulation, and needs no 233-prompt fit to reproduce.
* The whole convergence trajectory differs (Δmean < 0.01 at 45 prompts against 68).

What this separates
-------------------

Three candidates, and the three checks that tell them apart.

1. **The replicated forward differs by batch size.** ``jacobian_for_prompt`` runs the
   prompt replicated ``dim_batch`` times. Qwen3.5 is a **hybrid**: `linear_attn`
   state-space blocks with full attention every fourth layer
   (``full_attention_interval: 4``). SSM kernels commonly switch between chunked and
   sequential scans on batch size, and the observed error had a **period-4 sawtooth**
   across layers, which is suspicious in exactly that direction. Check A compares the
   forward's residual stream across batch sizes, and checks that replicas within one
   batch agree with each other.

2. **`torch.compile` specialises on batch size and one specialisation is wrong.**
   ``from_hf(compile=True)`` compiles per shape, and T11 already noted a new
   ``dim_batch`` recompiles. Check B runs the same comparison with compile off.

3. **The estimator itself mishandles a wide slice.** If the forward agrees across batch
   sizes and the effect survives with compile off, it is in the backward or the row
   assembly. Check C maps the effect across ``dim_batch`` to locate a threshold.

Cost: seconds to a couple of minutes. No fitting.

Run::

    uv run python experiments/dim_batch_diagnosis.py
    uv run python experiments/dim_batch_diagnosis.py --dim-batches 8,16,32,64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import config as jx_config  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LEN = 128
D_MODEL = 1024


def rel(a, b) -> float:
    n = b.norm().item()
    return (a - b).norm().item() / n if n else float("nan")


def build(compile_model: bool):
    import torch
    import transformers

    import jlens

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    return jlens.from_hf(hf, tok, compile=compile_model)


def one_prompt() -> str:
    from fit_lens import load_prompts

    return load_prompts(
        dataset="Salesforce/wikitext", config="wikitext-103-raw-v1", split="train",
        text_field="text", n_prompts=1, max_chars=2000,
    )[0]


def check_a_forward(model, prompt: str, batches: list[int]) -> dict:
    """Does the replicated forward depend on how many replicas there are?"""
    import torch

    from jlens.hooks import ActivationRecorder

    print("\n=== Check A: the replicated forward, across batch sizes ===")
    print("jacobian_for_prompt replicates the prompt dim_batch times. Every replica is")
    print("identical, so every batch size should give the same residual stream.")

    ids = model.encode(prompt, max_length=MAX_SEQ_LEN)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    layers = list(range(model.n_layers))
    out: dict[int, dict] = {}

    with torch.no_grad():
        for bs in batches:
            batched = ids.expand(bs, -1).contiguous()
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(batched)
                acts = {l: rec.activations[l].detach().float() for l in layers}
            # Within one batch: do the replicas agree with each other?
            spread = max(rel(acts[l][i], acts[l][0]) for l in layers for i in (1, bs - 1)) \
                if bs > 1 else 0.0
            out[bs] = {"row0": {l: acts[l][0].clone() for l in layers},
                       "within_batch_spread": spread}
            print(f"  batch={bs:>4}: replicas within the batch agree to {spread:.3e}")
            del acts

    base = batches[0]
    print(f"\n  residual stream vs batch={base}, per layer (max over layers shown):")
    cross = {}
    for bs in batches[1:]:
        per_layer = {l: rel(out[bs]["row0"][l], out[base]["row0"][l]) for l in layers}
        worst = max(per_layer, key=lambda l: per_layer[l])
        cross[bs] = per_layer
        print(f"  batch={bs:>4}: max {per_layer[worst]:.3e} at L{worst}, "
              f"L0 {per_layer[0]:.3e}, L{layers[-1]} {per_layer[layers[-1]]:.3e}")

    verdict = max((max(v.values()) for v in cross.values()), default=0.0)
    print(f"\n  VERDICT A: forward {'DIFFERS' if verdict > 1e-3 else 'agrees'} across "
          f"batch sizes (worst {verdict:.3e})")
    if verdict > 1e-3:
        print("  -> the model, not the estimator. A hybrid SSM kernel switching on batch")
        print("     size would do this, and it would make dim_batch unsafe on this family.")
    return {"within_batch": {str(k): v["within_batch_spread"] for k, v in out.items()},
            "cross_batch": {str(k): {str(l): x for l, x in v.items()} for k, v in cross.items()},
            "worst": verdict}


def check_bc_jacobian(prompt: str, batches: list[int], compile_model: bool) -> dict:
    """Does one prompt's Jacobian depend on dim_batch?"""
    import torch

    from jlens.fitting import jacobian_for_prompt

    tag = "compiled" if compile_model else "uncompiled"
    print(f"\n=== Check {'B' if not compile_model else 'C'}: "
          f"jacobian_for_prompt, {tag} ===")
    model = build(compile_model)
    source_layers = list(range(model.n_layers - 1))
    results = {}
    for db in batches:
        J, seq_len, n_valid = jacobian_for_prompt(
            model, prompt, source_layers, target_layer=None,
            dim_batch=db, max_seq_len=MAX_SEQ_LEN,
        )
        late = max(source_layers)
        ident = (J[late].float() - torch.eye(D_MODEL)).norm().item() / D_MODEL**0.5
        results[db] = {"J": {l: J[l].float() for l in source_layers}, "identity": ident}
        print(f"  dim_batch={db:>4}: identity_distance={ident:.6f}  "
              f"seq_len={seq_len} n_valid={n_valid}")

    base = batches[0]
    print(f"\n  per-layer difference vs dim_batch={base}:")
    hdr = f"{'dim_batch':>10}" + "".join(f"{('L' + str(l)):>11}" for l in (0, 1, 2, 3, 11, 22))
    print(hdr)
    print("-" * len(hdr))
    cross = {}
    for db in batches[1:]:
        per = {l: rel(results[db]["J"][l], results[base]["J"][l]) for l in source_layers}
        cross[db] = per
        print(f"{db:>10}" + "".join(f"{per[l]:>11.3e}" for l in (0, 1, 2, 3, 11, 22)))

    worst = max((max(v.values()) for v in cross.values()), default=0.0)
    print(f"\n  VERDICT: Jacobian {'DIFFERS' if worst > 1e-3 else 'agrees'} across "
          f"dim_batch when {tag} (worst {worst:.3e})")

    identities = {str(db): results[db]["identity"] for db in batches}
    del results, model
    torch.cuda.empty_cache()
    return {"identity_distance": identities,
            "cross": {str(k): {str(l): x for l, x in v.items()} for k, v in cross.items()},
            "worst": worst}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim-batches", default="8,16,32,64")
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()
    batches = [int(x) for x in args.dim_batches.split(",") if x.strip()]

    print(f"machine={cfg.machine}  model={MODEL_ID}  dim_batches={batches}")
    print("One prompt. No fit. Reproduces what the 233-prompt run showed on its first prompt.")

    prompt = one_prompt()
    print(f"prompt: {len(prompt)} chars, starts {prompt[:60]!r}")

    results: dict = {"task": "dim-batch-diagnosis", "machine": cfg.machine,
                     "model": MODEL_ID, "dim_batches": batches}

    if not args.skip_forward:
        model = build(compile_model=False)
        results["forward"] = check_a_forward(model, prompt, batches)
        del model
        import torch
        torch.cuda.empty_cache()

    results["jacobian_uncompiled"] = check_bc_jacobian(prompt, batches, compile_model=False)
    results["jacobian_compiled"] = check_bc_jacobian(prompt, batches, compile_model=True)

    print("\n=== reading the result ===")
    fwd = results.get("forward", {}).get("worst", 0.0)
    unc = results["jacobian_uncompiled"]["worst"]
    com = results["jacobian_compiled"]["worst"]
    print(f"  forward across batch sizes : {fwd:.3e}")
    print(f"  Jacobian, uncompiled       : {unc:.3e}")
    print(f"  Jacobian, compiled         : {com:.3e}")
    if fwd > 1e-3:
        print("\n  -> THE MODEL. The replicated forward is batch-size dependent, so every")
        print("     dim_batch gives a different Jacobian and the criteria's permission to")
        print("     vary it is unsafe on this architecture.")
    elif unc < 1e-3 <= com:
        print("\n  -> TORCH.COMPILE. The estimator is fine uncompiled and wrong compiled at")
        print("     some batch size. That is a miscompilation, and it would also mean our")
        print("     own compiled fits need re-examining, T15 included.")
    elif unc > 1e-3:
        print("\n  -> THE ESTIMATOR. The forward agrees and compile is not implicated, so")
        print("     the fault is in the backward or in row assembly.")
    else:
        print("\n  -> NOT REPRODUCED at one prompt. The 233-prompt result then needs a")
        print("     different explanation; suspect the fit driver or the artifacts.")

    out = cfg.artifact_root / "measurements" / "dim-batch-diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dim_batch_diagnosis.json"
    path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
