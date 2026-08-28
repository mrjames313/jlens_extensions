"""Where does `dim_batch` change the answer? One prompt, no fit.

Diagnostic, not a measurement. Written because `dim_batch_neutrality.py` returned a
result that cannot be taken at face value.

Run 1 result, 2026-08-27, and why the question changed
------------------------------------------------------

**Uncompiled, everything is fine.** ``identity_distance`` is 0.5314-0.5315 at every
``dim_batch``, matching T15's 0.531422 and the published tensor's 0.531418, and the
Jacobian spread across ``dim_batch`` is 1e-2 to 2e-2 -- the same order as the forward's
own batch-dependence, so roughly 1.6x amplification and nothing more.

**Compiled, one configuration returned garbage**: ``identity_distance`` **8.41** at
``dim_batch=8`` against an expected 0.53, with a uniform ~1.0 relative difference at every
layer. Uniform-at-every-layer is not a ``dim_batch`` signature; it is one tensor being
wrong and the others right. And the cause was printed alongside it::

    torch._dynamo hit config.recompile_limit (8)
    last reason: 0/7: KeyError on self._modules['linear_attn']

Qwen3.5 is hybrid -- ``linear_attn`` state-space blocks with full attention every fourth
layer -- so dynamo guards on the module dict and treats the two layer kinds as separate
compile variants. Four ``dim_batch`` values on top of that exhausts a budget of 8.

Run 2 repeated it with each *check* in its own process and reproduced it: compiled
``dim_batch=8`` gave 7.83 again. So it is not cross-check contamination. But that run
still compiled all four ``dim_batch`` values inside the Check C process, and the pattern
across them is monotonic in the number of **retained-graph backward passes** rather than
in slicing width::

    dim_batch   backward passes   identity_distance
        8            128              7.826707     garbage
       16             64              0.545104     off by 2.6%
       32             32              0.531534     correct
       64             16              0.531402     correct

Which is where it stops adding up. T15 ran **compiled at dim_batch=8** -- 128 passes, the
worst cell -- over 233 prompts, twice, and produced 0.531422 both times, matching the
published tensor to 1.4e-3. A deterministically broken configuration cannot do that.

The one thing neither run isolated: **a real fit compiles exactly one `dim_batch`.** Both
diagnostics compiled four in the same process, which is precisely the condition that
exhausts dynamo's recompile budget on a hybrid model. So the remaining hypothesis is that
the breakage is caused by the diagnostic's own multiplicity, not by compile per se -- and
that decides whether T15 stands.

**The question is therefore no longer about `dim_batch`.** It is: *does `torch.compile`
change the answer at a single configuration in a clean process?* Our fits run compiled, so
if it does, T15, T16 and the envelope work are all suspect. Every configuration below now
runs in its own process, and compiled-vs-uncompiled at the same `dim_batch` is the first
comparison reported.

The original contradiction
--------------------------

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

Each check runs in a **fresh subprocess**, which is T11's established pattern here and
is load-bearing rather than tidy. Tearing a CUDA model down and building another in one
process left the first run emitting ``cuBLAS ... there was no current CUDA context`` from
inside the backward -- recoverable, but it means the process is no longer in the state the
measurement assumes, and a diagnostic that cannot be trusted is worse than none. Separate
processes also mean one check crashing does not lose the others.

Cost: seconds to a couple of minutes. No fitting.

Run::

    uv run python experiments/dim_batch_diagnosis.py
    uv run python experiments/dim_batch_diagnosis.py --dim-batches 8,16,32,64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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


def compute_jacobian(dim_batch: int, compile_model: bool, out_path: Path) -> dict:
    """One prompt's Jacobian at one configuration, in this process, saved to disk.

    Saved rather than returned because each configuration runs in its own process --
    see the module docstring for why that is not optional here.
    """
    import torch

    from jlens.fitting import jacobian_for_prompt

    model = build(compile_model)
    prompt = one_prompt()
    source_layers = list(range(model.n_layers - 1))
    J, seq_len, n_valid = jacobian_for_prompt(
        model, prompt, source_layers, target_layer=None,
        dim_batch=dim_batch, max_seq_len=MAX_SEQ_LEN,
    )
    late = max(source_layers)
    ident = (J[late].float() - torch.eye(D_MODEL)).norm().item() / D_MODEL**0.5

    recompiles = None
    try:  # how close this configuration ran to the dynamo recompile ceiling
        import torch._dynamo as dynamo

        recompiles = {"limit": dynamo.config.recompile_limit}
    except Exception:  # noqa: BLE001 - diagnostics must not fail the diagnostic
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({l: J[l].float() for l in source_layers}, str(out_path))
    return {"dim_batch": dim_batch, "compile": compile_model,
            "identity_distance": ident, "seq_len": seq_len, "n_valid": n_valid,
            "path": str(out_path), "dynamo": recompiles}


RESULT_PREFIX = "@@DIAG_RESULT@@ "


def child(args) -> None:
    if args.child == "forward":
        payload = check_a_forward(build(compile_model=False), one_prompt(),
                                  [int(x) for x in args.dim_batches.split(",")])
    elif args.child == "jacobian":
        payload = compute_jacobian(args.dim_batch, args.compiled, Path(args.out))
    else:
        raise SystemExit(f"unknown check {args.child!r}")
    print(RESULT_PREFIX + json.dumps(payload, default=str), flush=True)


#: Per-child ceiling. A child that exceeds this is killed and reported rather than
#: hanging the run: the longest legitimate child is a cold compile plus 128 backward
#: passes, which T11 measured at well under two minutes.
CHILD_TIMEOUT_S = 900


def run_child(label: str, extra: list[str], quiet: bool = False) -> dict | None:
    """Run one child to completion, announcing it first.

    ``capture_output=True`` means nothing the child prints is visible until it exits,
    so without the announcement below a two-minute compile looks like a hang. The
    announcement is flushed explicitly because this output is usually piped to a log,
    where Python block-buffers and an unflushed line would not appear either.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), *extra]
    print(f"  -> {label} ... ", end="", flush=True)
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"TIMED OUT after {CHILD_TIMEOUT_S}s", flush=True)
        print(f"     killed. The other configurations still run; this one is missing.",
              flush=True)
        return None
    elapsed = time.time() - started

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX):])
        elif line.strip() and not quiet:
            print(f"\n{line}", end="", flush=True)

    if payload is None:
        print(f"NO RESULT after {elapsed:.0f}s (exit {proc.returncode})", flush=True)
        for t in (proc.stderr or "<no stderr>").strip().splitlines()[-4:]:
            print(f"     {t}", flush=True)
    else:
        note = ""
        if "recompile_limit" in (proc.stderr or ""):
            note = "  !! dynamo hit its recompile limit -- output suspect"
            payload["dynamo_limit_hit"] = True
        ident = payload.get("identity_distance")
        detail = f"identity_distance={ident:.6f}  " if ident is not None else ""
        print(f"{detail}({elapsed:.0f}s){note}", flush=True)
    return payload


def rel_tensors(pa: str, pb: str) -> dict[int, float]:
    import torch

    A, B = torch.load(pa, map_location="cpu"), torch.load(pb, map_location="cpu")
    return {l: rel(A[l], B[l]) for l in sorted(A)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim-batches", default="8,16,32,64")
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the saved Jacobians")
    parser.add_argument("--repeat", type=int, default=2,
                        help="runs per configuration, for the reproducibility null (default 2)")
    parser.add_argument("--child")
    parser.add_argument("--dim-batch", type=int)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    batches = [int(x) for x in args.dim_batches.split(",") if x.strip()]

    if args.child:
        child(args)
        return

    print(f"machine={cfg.machine}  model={MODEL_ID}  dim_batches={batches}")
    print("One prompt, no fit. Every configuration in its own process.")
    print("\nThe question that matters: does torch.compile change the answer? Our fits")
    print("run compiled, so if it does, T15 and everything downstream is suspect.")

    scratch = cfg.scratch_root / "diag-jacobians"
    results: dict = {"task": "dim-batch-diagnosis", "machine": cfg.machine,
                     "model": MODEL_ID, "dim_batches": batches}

    if not args.skip_forward:
        results["forward"] = run_child("forward", ["--child", "forward",
                                                   "--dim-batches", args.dim_batches])

    n_children = 2 * len(batches) * args.repeat
    print(f"\n=== computing Jacobians, ONE configuration per process ===")
    print(f"{n_children} child processes. Each loads the model and compiles from scratch,")
    print(f"so expect roughly 60-120s per child -- the first is slowest on a cold inductor")
    print(f"cache (T11 measured 81.8s there). Total on the order of {n_children * 90 // 60} minutes.")
    print(f"A child is killed and reported if it exceeds {CHILD_TIMEOUT_S}s rather than hanging.")
    print("This is what the previous run did not isolate: it compiled four dim_batch")
    print("values in a single process. A real fit compiles exactly one.")
    print(f"Each configuration runs {args.repeat}x, so within-configuration reproducibility")
    print("is measured rather than assumed -- without it the compile number has no null.")
    meta: dict[tuple[int, bool, int], dict] = {}
    for compiled in (False, True):
        for db in batches:
            for rep in range(args.repeat):
                tag = f"db{db}-{'c' if compiled else 'nc'}-r{rep}"
                path = scratch / f"{tag}.pt"
                extra = ["--child", "jacobian", "--dim-batch", str(db), "--out", str(path)]
                if compiled:
                    extra.append("--compiled")
                r = run_child(tag, extra, quiet=True)
                if r:
                    meta[(db, compiled, rep)] = r

    results["configs"] = {f"db{db}-{'c' if c else 'nc'}-r{r_}": v
                          for (db, c, r_), v in meta.items()}

    # --- The null: does one configuration reproduce itself? -------------------
    if args.repeat > 1:
        print("\n=== within-configuration reproducibility (the null) ===")
        hdr = f"{'config':>14} {'identity r0':>13} {'identity r1':>13} {'max rel diff':>14}"
        print(hdr); print("-" * len(hdr))
        selfdiff = {}
        for compiled in (False, True):
            for db in batches:
                if (db, compiled, 0) not in meta or (db, compiled, 1) not in meta:
                    continue
                tag = f"db{db}-{'c' if compiled else 'nc'}"
                d = rel_tensors(meta[(db, compiled, 0)]["path"],
                                meta[(db, compiled, 1)]["path"])
                selfdiff[tag] = max(d.values())
                print(f"{tag:>14} {meta[(db, compiled, 0)]['identity_distance']:>13.6f} "
                      f"{meta[(db, compiled, 1)]['identity_distance']:>13.6f} "
                      f"{selfdiff[tag]:>14.4e}")
        results["self_reproducibility"] = selfdiff
        flaky = {k: v for k, v in selfdiff.items() if v > 1e-2}
        if flaky:
            print(f"\n  !! these do not reproduce themselves: {sorted(flaky)}")
            print("     A configuration that differs from ITSELF across two identical runs")
            print("     is nondeterministic, not merely different -- and no comparison")
            print("     against it means anything until that is understood.")

    # Collapse to r0 for the remaining comparisons.
    meta = {(db, c): v for (db, c, r_), v in meta.items() if r_ == 0}

    # --- The critical axis: does compile change the answer? -------------------
    print("\n=== compiled vs uncompiled, SAME dim_batch ===")
    print("Both sides identical except for torch.compile. Any difference here is compile.")
    hdr = f"{'dim_batch':>10} {'identity nc':>13} {'identity c':>13} {'max rel diff':>14}"
    print(hdr); print("-" * len(hdr))
    compile_effect = {}
    for db in batches:
        if (db, False) not in meta or (db, True) not in meta:
            continue
        d = rel_tensors(meta[(db, False)]["path"], meta[(db, True)]["path"])
        compile_effect[db] = d
        print(f"{db:>10} {meta[(db, False)]['identity_distance']:>13.6f} "
              f"{meta[(db, True)]['identity_distance']:>13.6f} {max(d.values()):>14.4e}")
    results["compile_vs_uncompiled"] = {str(k): {str(l): x for l, x in v.items()}
                                        for k, v in compile_effect.items()}

    # --- dim_batch, within each compile setting -------------------------------
    for compiled in (False, True):
        tag = "compiled" if compiled else "uncompiled"
        avail = [db for db in batches if (db, compiled) in meta]
        if len(avail) < 2:
            continue
        base = avail[0]
        print(f"\n=== dim_batch effect, {tag} (vs dim_batch={base}) ===")
        hdr = f"{'dim_batch':>10}" + "".join(f"{('L' + str(l)):>11}" for l in (0, 3, 11, 22))
        print(hdr); print("-" * len(hdr))
        store = {}
        for db in avail[1:]:
            d = rel_tensors(meta[(base, compiled)]["path"], meta[(db, compiled)]["path"])
            store[db] = d
            print(f"{db:>10}" + "".join(f"{d[l]:>11.3e}" for l in (0, 3, 11, 22)))
        results[f"dim_batch_{tag}"] = {str(k): {str(l): x for l, x in v.items()}
                                       for k, v in store.items()}

    # --- verdict --------------------------------------------------------------
    print("\n=== reading the result ===")
    worst_compile = max((max(v.values()) for v in compile_effect.values()), default=None)
    fwd = (results.get("forward") or {}).get("worst")
    if worst_compile is None:
        print("  compile axis did not complete; cannot conclude.")
    elif worst_compile > 1e-2:
        print(f"  compile changes the Jacobian by up to {worst_compile:.3e}.")
        print("\n  -> TORCH.COMPILE IS NOT SAFE on this model, and our fits run compiled.")
        print("     T15, T16 and the envelope work all used it. Before anything else is")
        print("     believed, refit uncompiled and compare -- and note the dynamo warning")
        print("     about recompile_limit and self._modules['linear_attn']: Qwen3.5 is")
        print("     hybrid, so the two layer kinds are separate compile variants.")
    else:
        print(f"  compile changes the Jacobian by at most {worst_compile:.3e} -- within the")
        print(f"  bf16 batch-dependence the forward already shows"
              f"{f' ({fwd:.3e})' if fwd else ''}.")
        print("\n  -> COMPILE IS SAFE when one configuration is compiled per process, which")
        print("     is what a real fit does. The earlier compiled anomaly came from")
        print("     compiling several dim_batch values in one process and exhausting")
        print("     dynamo's recompile budget. T15 is not implicated; the neutrality")
        print("     probe's 233-prompt run still is, and needs re-running.")

    out = cfg.artifact_root / "measurements" / "dim-batch-diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dim_batch_diagnosis.json"
    path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nwrote {path}")

    if not args.keep:
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)
        print(f"removed {scratch} (pass --keep to retain the tensors)")


if __name__ == "__main__":
    main()
