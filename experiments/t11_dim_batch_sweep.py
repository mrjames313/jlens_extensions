"""T11 -- the dim_batch ceiling, per-prompt wall-clock and peak VRAM on this box.

Spec: ``environment-setup-and-first-fit``, stage 3. **These are the numbers T14's
gate decision turns on.**

What dim_batch actually trades
------------------------------

``jacobian_for_prompt`` runs one forward on the prompt replicated ``dim_batch``
times, retains the graph, and then does ``ceil(d_model / dim_batch)`` backward
passes over it. So:

* **Memory** scales with ``dim_batch`` -- the replicated forward and its retained
  graph, plus a ``[dim_batch, seq_len, d_model]`` cotangent buffer.
* **Backward FLOPs do not.** Upstream's docstring says per-prompt time is
  "nearly invariant since the total backward FLOPs are the same however they're
  sliced", and for FLOPs that is true.
* **Synchronising host transfers scale as 1/dim_batch.** ``fitting.py:218`` does
  ``rows.cpu()`` once per layer per pass. That is
  ``ceil(d_model/dim_batch) x n_source_layers`` blocking round-trips per prompt --
  **2944 at dim_batch=8**, 184 at 128, 23 at 1024, for this 23-layer 1024-wide
  model. The bytes moved are identical (~96 MB/prompt); the number of stalls is
  not.

So the docstring and the plan are both right about different things, and which
dominates wall-clock is exactly what this measures. The plan's expectation is that
the largest value that fits is the right one.

Method
------

Each configuration runs in a **fresh subprocess**: CUDA allocator state and the
dynamo compile cache both persist within a process, and ``dynamic=False`` means a
new ``dim_batch`` recompiles anyway. Fresh processes also mean an OOM kills only
that configuration.

Each configuration gets its **own checkpoint path**. ``fit(resume=True)`` validates
``source_layers`` / ``target_layer`` / ``skip_first`` but *not* ``dim_batch``, so a
shared path would let one configuration silently resume another's running sum.

A ``metrics_callback`` is passed because ``fit_lens.py`` always passes one, and it
is not free -- it materialises two ``d_model**2`` CPU temporaries per layer per
prompt. Measuring without it would not describe what we run. Its share is T12's
subject.

Peak memory comes from ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved``
and **not** ``nvidia-smi``: the GB10 has unified memory and reports
``Memory-Usage: Not Supported``. Both are recorded, labelled, because they measure
the allocator rather than the device and a later comparison against a discrete-GPU
box would otherwise compare two different things.

Run::

    uv run python experiments/t11_dim_batch_sweep.py
    uv run python experiments/t11_dim_batch_sweep.py --n-prompts 8
    uv run python experiments/t11_dim_batch_sweep.py --also-uncompiled
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
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
# d_model is 1024, so ceil(1024/dim_batch) == 1 at 1024 and larger values only pad
# the final pass. 8 is both scripts' default; 128 is what the published run used.
SWEEP = (8, 16, 32, 64, 128, 256, 512, 1024)
TARGET_PROMPTS = 233  # T1: the published results.prompts_fitted, what T15 pins to
GATE_HOURS = 2.0

RESULT_PREFIX = "@@T11_RESULT@@ "


def run_one(dim_batch: int, n_prompts: int, compile_model: bool) -> dict:
    """Fit `n_prompts` at one dim_batch and report timing and peak memory."""
    import torch
    import transformers

    import jlens
    from jlens.fitting import FitProgress, fit

    from fit_lens import load_prompts

    prompts = load_prompts(
        dataset="Salesforce/wikitext", config="wikitext-103-raw-v1", split="train",
        text_field="text", n_prompts=n_prompts, max_chars=2000,
    )

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf_model, tokenizer, compile=compile_model)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    per_prompt: list[float] = []

    def callback(progress: FitProgress) -> None:
        per_prompt.append(progress.elapsed_s)
        print(
            f"    prompt {progress.prompt_idx}: {progress.elapsed_s:7.2f}s "
            f"seq_len={progress.seq_len} valid={progress.n_valid_positions}",
            flush=True,
        )

    # Distinct per configuration: resume does not validate dim_batch.
    tag = f"t11-db{dim_batch}-{'c' if compile_model else 'nc'}"
    checkpoint_path = str(cfg.checkpoints / f"{tag}.pt")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).unlink(missing_ok=True)

    started = time.time()
    fit(
        model,
        prompts,
        dim_batch=dim_batch,
        max_seq_len=MAX_SEQ_LEN,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,   # the published cadence
        resume=False,
        metrics_callback=callback,
    )
    total_s = time.time() - started
    torch.cuda.synchronize()

    Path(checkpoint_path).unlink(missing_ok=True)

    # The first prompt carries torch.compile; the median of the rest is the
    # steady-state cost T14 projects from.
    steady = per_prompt[1:] or per_prompt
    n_passes = -(-1024 // dim_batch)
    return {
        "dim_batch": dim_batch,
        "compile": compile_model,
        "n_prompts": len(per_prompt),
        "n_passes_per_prompt": n_passes,
        "first_prompt_s": per_prompt[0] if per_prompt else None,
        "median_s_per_prompt": statistics.median(steady),
        "min_s_per_prompt": min(steady),
        "max_s_per_prompt": max(steady),
        "total_s": total_s,
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "device_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "oom": False,
    }


def child(args: argparse.Namespace) -> None:
    try:
        result = run_one(args.single, args.n_prompts, not args.no_compile)
    except Exception as exc:  # noqa: BLE001 - the OOM is the measurement
        message = f"{type(exc).__name__}: {exc}"
        is_oom = "out of memory" in message.lower() or "OutOfMemory" in type(exc).__name__
        result = {
            "dim_batch": args.single,
            "compile": not args.no_compile,
            "oom": is_oom,
            "error": message[:400],
        }
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def parent(args: argparse.Namespace) -> None:
    print(f"machine={cfg.machine}  model={MODEL_ID}  max_seq_len={MAX_SEQ_LEN}")
    print(f"sweep={list(SWEEP)}  n_prompts={args.n_prompts} per configuration")
    print("each configuration runs in a fresh subprocess\n")

    plans = [(d, True) for d in SWEEP]
    if args.also_uncompiled:
        plans += [(d, False) for d in SWEEP]

    results: list[dict] = []
    oom_at: dict[bool, int | None] = {True: None, False: None}
    for dim_batch, compiled in plans:
        if oom_at[compiled] is not None and dim_batch >= oom_at[compiled]:
            print(f"-- dim_batch={dim_batch} compile={compiled}: skipped (OOM at "
                  f"{oom_at[compiled]})", flush=True)
            continue
        print(f"-- dim_batch={dim_batch} compile={compiled} ...", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--single", str(dim_batch), "--n-prompts", str(args.n_prompts)]
        if not compiled:
            cmd.append("--no-compile")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith(RESULT_PREFIX):
                payload = json.loads(line[len(RESULT_PREFIX):])
            elif line.strip():
                print("   " + line, flush=True)
        if payload is None:
            tail = (proc.stderr or "<no stderr>").strip().splitlines()[-3:]
            payload = {"dim_batch": dim_batch, "compile": compiled, "oom": False,
                       "error": "child produced no result; stderr tail: " + " | ".join(tail)}
        results.append(payload)
        if payload.get("oom"):
            oom_at[compiled] = dim_batch
            print(f"   OOM at dim_batch={dim_batch}; higher values skipped", flush=True)
        elif "error" in payload:
            print(f"   FAILED: {payload['error'][:200]}", flush=True)

    print("\n--- dim_batch sweep ---")
    header = (f"{'dim_batch':>9} {'cmp':>4} {'passes':>7} {'s/prompt':>9} {'first_s':>8} "
              f"{'alloc_GB':>9} {'resvd_GB':>9}  {'233-prompt projection':>21}")
    print(header)
    print("-" * len(header))
    best = None
    for row in results:
        if row.get("oom"):
            print(f"{row['dim_batch']:>9} {str(row['compile'])[0]:>4} {'':>7} {'OOM':>9}")
            continue
        if "error" in row:
            print(f"{row['dim_batch']:>9} {str(row['compile'])[0]:>4} {'':>7} {'ERROR':>9}")
            continue
        hours = row["median_s_per_prompt"] * TARGET_PROMPTS / 3600.0
        verdict = "under gate" if hours < GATE_HOURS else "OVER GATE"
        print(f"{row['dim_batch']:>9} {str(row['compile'])[0]:>4} "
              f"{row['n_passes_per_prompt']:>7} {row['median_s_per_prompt']:>9.2f} "
              f"{row['first_prompt_s']:>8.1f} {row['peak_alloc_gb']:>9.2f} "
              f"{row['peak_reserved_gb']:>9.2f}  {hours:>7.2f}h {verdict:>12}")
        if row["compile"] and (best is None or row["median_s_per_prompt"] < best["median_s_per_prompt"]):
            best = row

    print("\nmemory is torch.cuda.max_memory_{allocated,reserved} -- the ALLOCATOR, not the")
    print("device. nvidia-smi reports 'Not Supported' on this unified-memory box.")

    summary = {
        "task": "T11",
        "machine": cfg.machine,
        "model": MODEL_ID,
        "max_seq_len": MAX_SEQ_LEN,
        "n_prompts_per_config": args.n_prompts,
        "target_prompts": TARGET_PROMPTS,
        "gate_hours": GATE_HOURS,
        "results": results,
        "best_compiled": best,
    }
    if best:
        hours = best["median_s_per_prompt"] * TARGET_PROMPTS / 3600.0
        print(f"\n=== fastest compiled: dim_batch={best['dim_batch']}, "
              f"{best['median_s_per_prompt']:.2f} s/prompt ===")
        print(f"=== {TARGET_PROMPTS} prompts projects to {hours:.2f}h "
              f"({'UNDER' if hours < GATE_HOURS else 'OVER'} the ~{GATE_HOURS}h gate) ===")
        summary["projected_hours_at_best"] = hours

    out_dir = cfg.artifact_root / "measurements" / "t11"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t11_dim_batch_sweep.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=int, help="internal: run one dim_batch and emit JSON")
    parser.add_argument("--n-prompts", type=int, default=5)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--also-uncompiled", action="store_true",
                        help="sweep uncompiled as well, for the compile speedup")
    args = parser.parse_args()
    if args.single is not None:
        child(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
