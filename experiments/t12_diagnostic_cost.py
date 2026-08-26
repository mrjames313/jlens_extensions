"""T12 -- what share of per-prompt wall-clock do the convergence diagnostics cost?

Spec: ``environment-setup-and-first-fit``, stage 3.

The measurement needs a build of ``fit()`` with the diagnostics removed, and the
task rightly insists that edit must never be committed -- a patched ``harness/``
would break the published-diagnostic comparability T16 depends on.

**So nothing in the repo is edited.** The driver reads ``harness/jlens/fitting.py``,
applies string replacements to the *text*, writes the result to a temp directory
outside the repo, and imports that. "Reverted and never committed" holds by
construction rather than by remembering. The repo file's sha256 is checked before
and after, and printed.

One thing worth knowing before reading the numbers
--------------------------------------------------

**Both diagnostics are computed unconditionally.** ``mean_rel_change`` and
``identity_distance`` are calculated in the prompt loop *outside* the
``if metrics_callback is not None`` guard -- only their *delivery* is conditional.
So passing ``metrics_callback=None`` does not skip the work, and the cost cannot be
avoided at 27B without a code change. That also means T11's timings already included
it.

Variants
--------

``full``     unmodified -- the baseline, and what we actually run
``no_mrc``   the per-layer ``mean_rel_change`` work removed, accumulation kept
``no_diag``  also ``identity_distance`` removed

so ``full - no_mrc`` is the convergence metric's cost and ``no_mrc - no_diag`` is
the identity distance's. The first is per-layer, the second once per prompt, so
they should differ by roughly the layer count.

A CPU micro-benchmark then isolates the arithmetic itself, and settles a discrepancy:
[[findings/f-2026-08-16-jlens-estimator-mechanics]] says the metric materialises
*two* ``d_model**2`` temporaries per layer, but the three lines it quotes allocate
*four* in eager mode -- ``jsum/n``, ``X - prev``, ``jsum + X``, and ``(...)/m``. The
benchmark times a two-temporary rewrite that is algebraically identical (``norm`` is
positively homogeneous, and ``torch.sub(..., alpha=)`` fuses the scaled subtract) to
see how much of the cost is avoidable without changing the reported number.

Run::

    uv run python experiments/t12_diagnostic_cost.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
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
DIM_BATCH = 8          # T11's measured optimum -- measure the share against what we run
N_PROMPTS = 5
TARGET_PROMPTS = 233
FITTING_PY = REPO / "harness" / "jlens" / "fitting.py"
RESULT_PREFIX = "@@T12_RESULT@@ "

# Exact text from harness/jlens/fitting.py. Each is asserted to appear exactly once,
# so a future edit upstream fails loudly instead of silently measuring nothing.
MRC_BLOCK = """            if n_done >= 1:
                prev_mean = jacobian_sum[layer] / n_done
                step = (X - prev_mean).norm().item() / m
                new_mean_norm = ((jacobian_sum[layer] + X) / m).norm().item()
                if new_mean_norm > 0.0:
                    rel_changes.append(step / new_mean_norm)
            jacobian_sum[layer] += X
"""
MRC_REPLACEMENT = """            jacobian_sum[layer] += X
"""

IDENT_BLOCK = """        identity_distance = (
            (jacobian_sum[late_layer] / n_done) - torch.eye(d_model)
        ).norm().item() / math.sqrt(d_model)
"""
IDENT_REPLACEMENT = """        identity_distance = float("nan")
"""


def patched_fit(variant: str):
    """Import a text-patched copy of fitting.py from a temp dir. Repo untouched."""
    import importlib.util

    source = FITTING_PY.read_text()
    if variant in ("no_mrc", "no_diag"):
        if source.count(MRC_BLOCK) != 1:
            raise SystemExit("mean_rel_change block not found exactly once -- fitting.py changed")
        source = source.replace(MRC_BLOCK, MRC_REPLACEMENT)
    if variant == "no_diag":
        if source.count(IDENT_BLOCK) != 1:
            raise SystemExit("identity_distance block not found exactly once -- fitting.py changed")
        source = source.replace(IDENT_BLOCK, IDENT_REPLACEMENT)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"t12-{variant}-"))
    tmp_file = tmp_dir / "fitting_patched.py"
    tmp_file.write_text(source)
    spec = importlib.util.spec_from_file_location(f"jlens_fitting_{variant}", tmp_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit


def run_one(variant: str) -> dict:
    import torch
    import transformers

    import jlens

    from fit_lens import load_prompts

    fit = patched_fit(variant)
    prompts = load_prompts(
        dataset="Salesforce/wikitext", config="wikitext-103-raw-v1", split="train",
        text_field="text", n_prompts=N_PROMPTS, max_chars=2000,
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf_model, tokenizer, compile=True)

    per_prompt: list[float] = []

    def callback(progress) -> None:
        per_prompt.append(progress.elapsed_s)
        print(f"    prompt {progress.prompt_idx}: {progress.elapsed_s:7.2f}s", flush=True)

    checkpoint_path = str(cfg.checkpoints / f"t12-{variant}.pt")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).unlink(missing_ok=True)
    fit(
        model, prompts,
        dim_batch=DIM_BATCH, max_seq_len=MAX_SEQ_LEN,
        checkpoint_path=checkpoint_path, checkpoint_every=1, resume=False,
        metrics_callback=callback,
    )
    Path(checkpoint_path).unlink(missing_ok=True)

    # Prompt 0 carries torch.compile AND skips the n_done >= 1 diagnostic branch,
    # so it is doubly unrepresentative. Median over the rest.
    steady = per_prompt[1:] or per_prompt
    return {
        "variant": variant,
        "n_prompts": len(per_prompt),
        "first_prompt_s": per_prompt[0] if per_prompt else None,
        "median_s_per_prompt": statistics.median(steady),
        "all_s": per_prompt,
    }


def micro_benchmark() -> dict:
    """Isolate the arithmetic, and count the temporaries by timing a rewrite."""
    import torch

    d_model, n_layers, repeats = 1024, 23, 5
    torch.manual_seed(0)
    jacobian_sum = [torch.randn(d_model, d_model) for _ in range(n_layers)]
    per_prompt_J = [torch.randn(d_model, d_model) for _ in range(n_layers)]
    n_done, m = 40, 41

    def as_shipped() -> None:
        for jsum, X in zip(jacobian_sum, per_prompt_J):
            prev_mean = jsum / n_done                             # temp 1
            step = (X - prev_mean).norm().item() / m              # temp 2
            _ = ((jsum + X) / m).norm().item()                    # temps 3 and 4
            _ = step

    def two_temporaries() -> None:
        # Algebraically identical: ||A/m|| == ||A||/m for m > 0, and
        # torch.sub(X, jsum, alpha=1/n) fuses X - jsum/n into one allocation.
        for jsum, X in zip(jacobian_sum, per_prompt_J):
            step = torch.sub(X, jsum, alpha=1.0 / n_done).norm().item() / m
            _ = (jsum + X).norm().item() / m
            _ = step

    def timed(fn) -> float:
        fn()  # warm
        best = min(_time_once(fn) for _ in range(repeats))
        return best

    def _time_once(fn) -> float:
        started = time.perf_counter()
        fn()
        return time.perf_counter() - started

    shipped_s = timed(as_shipped)
    rewrite_s = timed(two_temporaries)
    return {
        "d_model": d_model,
        "n_layers": n_layers,
        "as_shipped_s": shipped_s,
        "two_temporaries_s": rewrite_s,
        "ratio": shipped_s / rewrite_s if rewrite_s else None,
        "bytes_per_prompt_shipped_gb": 4 * n_layers * d_model * d_model * 4 / 1024**3,
    }


def child(args: argparse.Namespace) -> None:
    try:
        result = run_one(args.single)
    except Exception as exc:  # noqa: BLE001
        result = {"variant": args.single, "error": f"{type(exc).__name__}: {exc}"[:400]}
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def parent(args: argparse.Namespace) -> None:
    before_hash = hashlib.sha256(FITTING_PY.read_bytes()).hexdigest()
    print(f"machine={cfg.machine}  dim_batch={DIM_BATCH} (T11's optimum)  compile=True")
    print(f"harness/jlens/fitting.py sha256 before: {before_hash}")
    print("patched copies are written to a temp dir; the repo file is never touched\n")

    results = []
    for variant in ("full", "no_mrc", "no_diag"):
        print(f"-- {variant} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--single", variant],
            capture_output=True, text=True,
        )
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith(RESULT_PREFIX):
                payload = json.loads(line[len(RESULT_PREFIX):])
            elif line.strip():
                print("   " + line, flush=True)
        if payload is None:
            tail = (proc.stderr or "<none>").strip().splitlines()[-4:]
            payload = {"variant": variant, "error": "no result; stderr: " + " | ".join(tail)}
        if "error" in payload:
            print(f"   FAILED: {payload['error'][:300]}", flush=True)
        results.append(payload)

    after_hash = hashlib.sha256(FITTING_PY.read_bytes()).hexdigest()
    print(f"\nharness/jlens/fitting.py sha256 after:  {after_hash}")
    print(f"repo file unchanged: {before_hash == after_hash}")

    by = {r["variant"]: r for r in results if "error" not in r}
    print("\n--- per-prompt wall-clock ---")
    for r in results:
        if "error" in r:
            print(f"  {r['variant']:<9} ERROR")
        else:
            print(f"  {r['variant']:<9} {r['median_s_per_prompt']:7.2f}s"
                  f"   (first {r['first_prompt_s']:.1f}s)")

    summary = {"task": "T12", "machine": cfg.machine, "dim_batch": DIM_BATCH,
               "n_prompts_per_variant": N_PROMPTS, "results": results,
               "fitting_py_sha256": before_hash,
               "repo_file_unchanged": before_hash == after_hash}

    if {"full", "no_mrc", "no_diag"} <= by.keys():
        full = by["full"]["median_s_per_prompt"]
        no_mrc = by["no_mrc"]["median_s_per_prompt"]
        no_diag = by["no_diag"]["median_s_per_prompt"]
        mrc_cost, ident_cost = full - no_mrc, no_mrc - no_diag
        total = full - no_diag
        print(f"\n  mean_rel_change   {mrc_cost:6.2f}s  ({100*mrc_cost/full:5.2f}% of a prompt)")
        print(f"  identity_distance {ident_cost:6.2f}s  ({100*ident_cost/full:5.2f}%)")
        print(f"  both              {total:6.2f}s  ({100*total/full:5.2f}%)")
        print(f"  over {TARGET_PROMPTS} prompts: {total*TARGET_PROMPTS/60:.1f} min of "
              f"{full*TARGET_PROMPTS/3600:.2f}h")
        summary |= {"mean_rel_change_s": mrc_cost, "identity_distance_s": ident_cost,
                    "both_s": total, "share_of_prompt": total / full,
                    "minutes_over_target": total * TARGET_PROMPTS / 60}

    print("\n--- CPU micro-benchmark: the arithmetic alone ---")
    micro = micro_benchmark()
    summary["micro_benchmark"] = micro
    print(f"  as shipped (4 temporaries/layer)  {micro['as_shipped_s']*1000:7.1f} ms")
    print(f"  2-temporary rewrite, identical    {micro['two_temporaries_s']*1000:7.1f} ms")
    print(f"  ratio                             {micro['ratio']:7.2f}x")
    print(f"  allocated per prompt as shipped   {micro['bytes_per_prompt_shipped_gb']:7.2f} GB")

    out_dir = cfg.artifact_root / "measurements" / "t12"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t12_diagnostic_cost.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", choices=("full", "no_mrc", "no_diag"))
    args = parser.parse_args()
    child(args) if args.single else parent(args)


if __name__ == "__main__":
    main()
