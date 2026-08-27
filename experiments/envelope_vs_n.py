"""How the run-to-run envelope scales with prompt count, measured cleanly.

**Not a spec task.** `environment-setup-and-first-fit` closed at T17 with all eighteen
tasks done; this is follow-on work that spec's results prompted, and it is named for what
it measures rather than given a task number, so a later spec's numbering cannot collide
with it.

Why this exists
---------------

T18 answered this question and its answer is contaminated twice over.

1. **It was measured through fp16.** ``--save_dtype`` did not exist, so every lens it
   compared was an fp16 snapshot -- and an fp16 comparison *inflates* a sub-quantum
   difference rather than erasing it, by up to 51x at the quiet end of this stack. See
   ``f-2026-08-27-fp16-comparison-distortion``. T18's L0 figure survives that; its
   high-layer figures are measuring the storage format.
2. **Its exponents were fitted across a contaminated range.** ``alpha`` came from an
   n=5 to n=20 comparison where the distortion varies with the difference being
   measured, so the slope is not the slope of the underlying quantity.

And its result cannot be repaired from T16's data, because T18 ran **uncompiled** while
T15/T16 ran **compiled**. Comparing those two endpoints mixes the prompt count with the
execution configuration, which is exactly the confound T18 itself warned about.

So this measures it again at **one fixed configuration** -- the production one, compiled
at ``dim_batch=8``, stored fp32 -- across six prompt counts, with T15's n=233 pair as a
seventh point on the same corpus prefix and the same settings.

What alpha means
----------------

A lens is a running mean over prompts, so ``envelope ~ C * n**-alpha`` where:

* ``alpha = 0.5`` -- per-prompt noise is independent and averages down as 1/sqrt(n).
* ``alpha = 0`` -- the noise is systematic and never averages down.

T18 measured 0.37-0.42 and concluded "it averages down, just more slowly than
1/sqrt(n)". That conclusion rests on the contaminated fit and is what this re-measures.

The cheap trick: one pass gives every prompt count
--------------------------------------------------

A lens at n prompts is the running mean of the first n, and ``fit(resume=True)`` restores
that running sum from a checkpoint and continues. So a single 60-prompt pass yields a
lens at *every* target count along the way, at no extra compute: fit the first 5, save;
resume and fit to 10, save; and so on. Prompts 0-4 are computed exactly once.

Total cost is therefore 60 prompts per run rather than the 165 that six independent fits
would need -- ~31 minutes for the pair at the measured 15.53 s/prompt, against ~41 for
the two-point design it replaces, while yielding six points instead of two.

This also exercises the resume path that T6 patched and T7 could not reach, on real
data rather than the stub in ``tests/test_harness_backports.py``.

Run::

    uv run python experiments/envelope_vs_n.py
    uv run python experiments/envelope_vs_n.py --max-n 20        # a quick rehearsal
    uv run python experiments/envelope_vs_n.py --analyse-only    # re-tabulate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import compare as jx_cmp  # noqa: E402
from jlens_extensions import config as jx_config  # noqa: E402
from jlens_extensions.profile import MachineProfile  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

MODEL_ID = "Qwen/Qwen3.5-0.8B"
SLUG = "Qwen3.5-0.8B"
MAX_SEQ_LEN = 128
TARGETS = (5, 10, 20, 30, 40, 60)
T15_N = 233

DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"
TEXT_FIELD = "text"
MAX_CHARS = 2000

# T18's fp16-measured projections to n=233, carried for comparison only.
T18_ALPHA = {0: 0.366, 1: 0.411, 2: 0.412, 3: 0.420, 4: 0.423, 6: 0.410, 8: 0.403,
             10: 0.386, 12: 0.363, 15: 0.331, 18: 0.241, 20: 0.190, 21: 0.200, 22: -0.055}


def out_root() -> Path:
    return cfg.artifact_root / "measurements" / "envelope-vs-n"


def snapshot_path(label: str, n: int) -> Path:
    return out_root() / f"run-{label}" / f"{SLUG}_n{n:04d}_jacobian_lens.pt"


def run_one(label: str, targets: tuple[int, ...], dim_batch: int, compile_model: bool) -> dict:
    """Fit to the largest target, saving a lens at each target along the way."""
    import torch
    import transformers

    import jlens
    from jlens.fitting import fit

    from fit_lens import load_prompts

    prompts = load_prompts(
        dataset=DATASET, config=DATASET_CONFIG, split=DATASET_SPLIT,
        text_field=TEXT_FIELD, n_prompts=max(targets), max_chars=MAX_CHARS,
    )
    if len(prompts) != max(targets):
        raise SystemExit(f"corpus gave {len(prompts)} prompts, need {max(targets)}")

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok, compile=compile_model)

    # Distinct per run, and cleared: a stale checkpoint would silently resume the
    # other run's running sum, which is the one failure that would look like a result.
    checkpoint = cfg.checkpoints / f"envn-{label}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.unlink(missing_ok=True)
    snapshot_path(label, targets[0]).parent.mkdir(parents=True, exist_ok=True)

    per_prompt: list[float] = []
    saved: dict[int, str] = {}
    started = time.time()

    for i, n in enumerate(targets):
        def callback(p, _store=per_prompt, _n=n):
            _store.append(p.elapsed_s)
            print(f"    [{label}] prompt {p.n_done}/{_n}  {p.elapsed_s:6.2f}s  "
                  f"id={p.identity_distance:.6f}", flush=True)

        lens = fit(
            model, prompts[:n],
            dim_batch=dim_batch,
            max_seq_len=MAX_SEQ_LEN,
            checkpoint_path=str(checkpoint),
            checkpoint_every=1,
            resume=(i > 0),          # continue the running sum rather than refit
            metrics_callback=callback,
        )
        if lens.n_prompts != n:
            raise SystemExit(
                f"run {label}: lens reports n_prompts={lens.n_prompts}, expected {n}. "
                f"The resume path did not continue the running sum as assumed."
            )
        path = snapshot_path(label, n)
        lens.save(str(path), dtype=torch.float32)
        saved[n] = str(path)
        print(f"  [{label}] n={n}: saved {path.name}", flush=True)

    checkpoint.unlink(missing_ok=True)
    return {"label": label, "snapshots": saved, "wall_s": time.time() - started,
            "n_prompts_computed": len(per_prompt)}


def analyse(targets: tuple[int, ...]) -> dict:
    """Compare the two runs at each n, then fit an exponent per layer."""
    from jlens.lens import JacobianLens
    import torch

    envelopes: dict[int, dict[int, float]] = {}   # n -> layer -> rel_frobenius
    for n in targets:
        pa, pb = snapshot_path("a", n), snapshot_path("b", n)
        if not (pa.exists() and pb.exists()):
            print(f"  n={n}: snapshots missing, skipping")
            continue
        a = {k: v.to(torch.float32) for k, v in JacobianLens.load(str(pa)).jacobians.items()}
        b = {k: v.to(torch.float32) for k, v in JacobianLens.load(str(pb)).jacobians.items()}
        envelopes[n] = {d.layer: d.rel_frobenius for d in jx_cmp.compare_lenses(a, b)}
        del a, b

    # T15's pair is the same configuration on the same corpus prefix -- the anchor.
    t15 = {lbl: cfg.lenses / f"t15-{lbl}" / f"{SLUG}_jacobian_lens.pt" for lbl in ("a", "b")}
    if all(p.exists() for p in t15.values()):
        a = {k: v.to(torch.float32) for k, v in JacobianLens.load(str(t15["a"])).jacobians.items()}
        b = {k: v.to(torch.float32) for k, v in JacobianLens.load(str(t15["b"])).jacobians.items()}
        envelopes[T15_N] = {d.layer: d.rel_frobenius for d in jx_cmp.compare_lenses(a, b)}
        del a, b
        print(f"  included T15's n={T15_N} pair as the far anchor")
    else:
        print(f"  T15 lenses not found; fitting without the n={T15_N} anchor")

    ns = sorted(envelopes)
    if len(ns) < 2:
        raise SystemExit("need at least two prompt counts to fit an exponent")
    layers = sorted(envelopes[ns[0]])

    print(f"\n--- envelope per layer, fp32, compiled, dim_batch=8 ---")
    hdr = f"{'layer':>6}" + "".join(f"{('n=' + str(n)):>12}" for n in ns) + \
          f"{'alpha':>8}{'r2':>7}{'T18 a':>8}"
    print(hdr)
    print("-" * len(hdr))
    fits: dict[int, dict] = {}
    for layer in layers:
        pts = [(n, envelopes[n][layer]) for n in ns if envelopes[n].get(layer, 0) > 0]
        row = f"{layer:>6}" + "".join(f"{envelopes[n][layer]:>12.3e}" for n in ns)
        try:
            f = jx_cmp.fit_power_law(pts)
            fits[layer] = f
            t18 = T18_ALPHA.get(layer)
            row += f"{f['alpha']:>8.3f}{f['r_squared']:>7.3f}"
            row += f"{t18:>8.3f}" if t18 is not None else f"{'—':>8}"
        except ValueError as exc:
            row += f"  {exc}"
        print(row)

    print("\nalpha = 0.5 is independent per-prompt noise averaging down as 1/sqrt(n).")
    print("alpha = 0   is systematic noise that never averages down.")
    print("A low r2 means the envelope is not scaling as a clean power law and a single")
    print("alpha should not be quoted for that layer.")
    return {"envelopes": {str(n): envelopes[n] for n in ns},
            "fits": {str(k): {kk: vv for kk, vv in v.items() if kk != "points"}
                     for k, v in fits.items()},
            "prompt_counts": ns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-n", type=int, default=max(TARGETS),
                        help="largest prompt count to fit (default 60)")
    parser.add_argument("--analyse-only", action="store_true",
                        help="skip fitting; re-tabulate from saved snapshots")
    args = parser.parse_args()

    targets = tuple(n for n in TARGETS if n <= args.max_n)
    if not targets:
        raise SystemExit(f"--max-n {args.max_n} is below the smallest target {min(TARGETS)}")

    facts = MachineProfile.load(cfg.profile_path).model(MODEL_ID)
    print(f"machine={cfg.machine}  model={MODEL_ID}")
    print(f"targets={list(targets)}  dim_batch={facts.dim_batch}  compile={facts.compile}")
    print(f"storing fp32; T18's contamination is the reason this is being re-measured")

    runs = []
    if not args.analyse_only:
        est = 2 * max(targets) * facts.s_per_prompt / 60
        print(f"two runs of {max(targets)} prompts, ~{est:.0f} min plus compile\n")
        for label in ("a", "b"):
            print(f"=== run {label} ===", flush=True)
            r = run_one(label, targets, facts.dim_batch, facts.compile)
            print(f"  [{label}] done in {r['wall_s'] / 60:.1f} min, "
                  f"{r['n_prompts_computed']} prompts computed "
                  f"(vs {sum(targets)} if refitted each time)")
            runs.append(r)

    print("\n=== analysis ===")
    result = analyse(targets)
    result.update(task="envelope-vs-n", machine=cfg.machine, model=MODEL_ID,
                  dim_batch=facts.dim_batch, compile=facts.compile,
                  save_dtype="float32", runs=runs)
    out_root().mkdir(parents=True, exist_ok=True)
    path = out_root() / "envelope_vs_n.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
