"""Is `dim_batch` numerically neutral? A direct test, on one box.

Follow-on from T16. Not a spec task.

.. warning::

   **First run 2026-08-27 returned per-layer differences of 1.0 to 4.1 relative --
   essentially unrelated tensors, not a numerical effect.** That is not a `dim_batch`
   result, it is a symptom: T16 found our ``dim_batch=8`` lens matching a ``dim_batch=128``
   reference to 1.4e-3, and both cannot be true. The harness's own ``identity_distance``
   differs from the **first prompt** (0.531268 at 8, 0.543 at 64), so it reproduces
   without a fit. Run ``dim_batch_diagnosis.py`` first; until that resolves, treat this
   driver's verdict as uninterpretable rather than as evidence about `dim_batch`.

The question
------------

T16 found that ours-vs-published leaves a residual after removing the measured storage
floor and our own measured run-to-run envelope -- and that the residual does **not**
follow the depth profile nondeterminism has, so it cannot all be their noise. Something
depth-independent, of order 1e-4, is unaccounted for.

The leading suspect is `dim_batch`: we fit at 8, Neuronpedia at 128. The reproduction
criteria permit that deviation, on the argument that each backward pass writes disjoint
rows of ``J_l``, so the fitted tensor is identical however the output dimensions are
sliced. That argument is about *which entries* get written. It says nothing about the
arithmetic *within* a pass -- and a 128-wide cotangent block reduces differently from an
8-wide one, in bf16, where reduction order is exactly what makes fits non-reproducible.

Why one fit is enough
---------------------

The obvious experiment -- refit at their 128 and compare against the published lens --
is both impossible and weaker than it looks. Impossible because T11 measured
``dim_batch=128`` as an **OOM** on this box (the memory model puts it near 130 GB against
121.63 available); 64 is the largest value observed to fit. Weaker because a comparison
against *their* lens carries their environment, their noise draw and their storage
quantisation on top of the variable under test.

So the test is run against **our own** lenses instead. T15 produced two independent fits
at ``dim_batch=8``, and their difference is a measured pure-noise null at this exact
prompt count and configuration. One new fit at ``dim_batch=64`` then gives:

* ``d(64, a)`` and ``d(64, b)`` -- the dim_batch effect, each carrying one noise draw.
* ``d(a, b)`` -- pure noise, already measured, no dim_batch difference at all.

Everything else is held: same box, same torch, same commit, same corpus, same prompts,
same compile setting, same fp32 storage. `dim_batch` is the only thing that moves, which
is what makes this decisive in a way the vs-published comparison cannot be.

**If `dim_batch` is neutral**, ``d(64, ·)`` sits inside the ``d(a, b)`` envelope at every
layer and the criteria's permission is safe.

**If it is not**, the excess is quantified per layer, and the shape matters: an excess
that is *depth-independent* at ~1e-4 would match the residual T16 could not explain and
would identify its cause.

64 is not their 128, so a null result here does not fully clear `dim_batch` -- it bounds
the effect over an 8x change in slice width and leaves the last 2x unmeasured. A positive
result, on the other hand, is conclusive.

Validated against synthetic fixtures in both directions before first use, which for a
verdict-producing driver is the part worth stating: with a 3e-4 depth-independent excess
planted into the dim_batch lens it reports NOT neutral and recovers the excess as
depth-independent (spanning 1.1x across a stack whose noise spans ~9000x); with no excess
planted it reports NEUTRAL, ratios 0.88-1.16. An instrument that only ever fires one way
would not be evidence -- the same argument A1 makes for its control.

Run::

    uv run python experiments/dim_batch_neutrality.py
    uv run python experiments/dim_batch_neutrality.py --dim-batch 32
    uv run python experiments/dim_batch_neutrality.py --analyse-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import compare as jx_cmp  # noqa: E402
from jlens_extensions import config as jx_config  # noqa: E402
from jlens_extensions import fetch as jx_fetch  # noqa: E402
from jlens_extensions import provenance as jx_prov  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

MODEL_ID = "Qwen/Qwen3.5-0.8B"
SLUG = "Qwen3.5-0.8B"
N_PROMPTS = 233
MAX_SEQ_LEN = 128
DTYPE = "bfloat16"
DEVICE_MAP = "cuda"
SAVE_DTYPE = "float32"
PUBLISHED_DIM_BATCH = 128

DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"
TEXT_FIELD = "text"
MAX_CHARS = 2000

FIT_LENS = REPO / "harness" / "fit_lens.py"


def lens_dir(dim_batch: int) -> Path:
    return cfg.lenses / f"dimbatch-{dim_batch}"


def load(path: Path) -> dict:
    import torch

    from jlens.lens import JacobianLens

    return {k: v.to(torch.float32)
            for k, v in JacobianLens.load(str(path)).jacobians.items()}


def run_fit(dim_batch: int, fresh: bool) -> dict:
    """One fit at `dim_batch`, otherwise identical to T15's command."""
    out_dir = cfg.scratch_root / "fits" / f"dimbatch-{dim_batch}"
    lens_path = out_dir / f"{SLUG}_jacobian_lens.pt"
    checkpoint = out_dir / f"{SLUG}_checkpoint.pt"

    existing = [p for p in (lens_path, checkpoint) if p.exists()]
    if existing and not fresh:
        raise SystemExit(
            f"{out_dir} already holds {[p.name for p in existing]}. fit_lens.py resumes "
            f"by default, so continuing would silently pick that up. Pass --fresh to refit."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        p.unlink()

    from jlens_extensions.fitcmd import UNGATED, build_fit_command
    from jlens_extensions.profile import MachineProfile

    facts = MachineProfile.load(cfg.profile_path).model(MODEL_ID)
    gate = facts.gate_identity if facts.gate_identity is not None else UNGATED
    if gate is UNGATED:
        print("  WARNING: no gate_identity in the profile; this fit is ungated and a "
              "compiled run has a 30-50% chance of saving a corrupt lens", flush=True)
    cmd = build_fit_command(
        fit_lens_path=FIT_LENS, model_id=MODEL_ID, out_dir=out_dir,
        n_prompts=N_PROMPTS, dim_batch=dim_batch, max_seq_len=MAX_SEQ_LEN,
        dtype=DTYPE, device_map=DEVICE_MAP, save_dtype=SAVE_DTYPE,
        dataset=DATASET, dataset_config=DATASET_CONFIG, dataset_split=DATASET_SPLIT,
        text_field=TEXT_FIELD, max_chars=MAX_CHARS, gate_identity=gate,
    )
    print(f"  {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO / "harness"))
    wall = time.time() - started
    if proc.returncode != 0:
        raise SystemExit(f"fit_lens.py exited {proc.returncode} after {wall / 60:.1f} min")

    dest = lens_dir(dim_batch)
    dest.mkdir(parents=True, exist_ok=True)
    import shutil

    for name in (f"{SLUG}_jacobian_lens.pt", f"{SLUG}_convergence.csv"):
        shutil.copy2(out_dir / name, dest / name)
    print(f"  done in {wall / 3600:.2f}h -> {dest}", flush=True)
    return {"dim_batch": dim_batch, "wall_s": wall, "dest": str(dest)}


def table(title: str, diffs, reference: dict[int, float] | None, ref_name: str) -> list[dict]:
    print(f"\n{title}")
    hdr = f"{'layer':>6} {'rel_frobenius':>14}"
    if reference is not None:
        hdr += f" {ref_name:>14} {'ratio':>8} {'excess':>12}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for d in diffs:
        line = f"{d.layer:>6} {d.rel_frobenius:>14.4e}"
        row = d.as_dict()
        if reference is not None:
            ref = reference[d.layer]
            # Quadrature excess: what is left after removing one noise draw's worth.
            ex2 = d.rel_frobenius**2 - ref**2
            excess = math.sqrt(ex2) if ex2 > 0 else 0.0
            row.update(reference=ref, ratio=d.rel_frobenius / ref, excess=excess)
            line += f" {ref:>14.4e} {d.rel_frobenius / ref:>8.2f} {excess:>12.4e}"
        rows.append(row)
        print(line)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim-batch", type=int, default=64,
                        help="dim_batch to test (default 64, the largest T11 saw fit)")
    parser.add_argument("--fresh", action="store_true", help="discard an existing fit and refit")
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    db = args.dim_batch

    t15 = {lbl: cfg.lenses / f"t15-{lbl}" / f"{SLUG}_jacobian_lens.pt" for lbl in ("a", "b")}
    for lbl, p in t15.items():
        if not p.exists():
            raise SystemExit(f"T15 run {lbl} not found at {p}; it is the null this compares to")

    print(f"machine={cfg.machine}  model={MODEL_ID}")
    print(f"testing dim_batch={db} against T15's pair at dim_batch=8")
    print(f"published ran at dim_batch={PUBLISHED_DIM_BATCH}, which OOMs here (T11)")

    run = None
    if not args.analyse_only:
        print(f"\n=== fit at dim_batch={db} ===")
        run = run_fit(db, args.fresh)

    new_path = lens_dir(db) / f"{SLUG}_jacobian_lens.pt"
    if not new_path.exists():
        raise SystemExit(f"no lens at {new_path}; run without --analyse-only first")

    print("\n=== analysis ===")
    new = load(new_path)
    a, b = load(t15["a"]), load(t15["b"])

    # The null: two fits that differ ONLY by nondeterminism.
    null = {d.layer: d.rel_frobenius for d in jx_cmp.compare_lenses(a, b)}
    results: dict = {"task": "dim-batch-neutrality", "machine": cfg.machine,
                     "model": MODEL_ID, "dim_batch_tested": db, "dim_batch_baseline": 8,
                     "n_prompts": N_PROMPTS, "run": run,
                     "null_a_vs_b": null}

    rows_a = table(f"dim_batch={db} vs T15-a (dim_batch=8), against the a-vs-b null",
                   jx_cmp.compare_lenses(new, a), null, "a-vs-b null")
    rows_b = table(f"dim_batch={db} vs T15-b (dim_batch=8), against the a-vs-b null",
                   jx_cmp.compare_lenses(new, b), null, "a-vs-b null")
    results["vs_t15_a"], results["vs_t15_b"] = rows_a, rows_b

    # Verdict. If dim_batch is neutral both comparisons are pure noise, so the ratio
    # to the null hovers near sqrt(2)/1 -- one noise draw against another.
    ratios = [r["ratio"] for r in rows_a + rows_b]
    excesses = [r["excess"] for r in rows_a]
    print(f"\nratio to the null: min {min(ratios):.2f}, median "
          f"{sorted(ratios)[len(ratios) // 2]:.2f}, max {max(ratios):.2f}")
    neutral = max(ratios) < 1.5
    print(f"\nVERDICT: dim_batch {db} vs 8 is "
          f"{'NEUTRAL — inside the noise null at every layer' if neutral else 'NOT neutral'}")
    if not neutral:
        deep = [e for r, e in zip(rows_a, excesses) if r["layer"] >= 18]
        shallow = [e for r, e in zip(rows_a, excesses) if r["layer"] <= 4]
        print(f"  excess at L0-L4:   {min(shallow):.3e} to {max(shallow):.3e}")
        print(f"  excess at L18-L22: {min(deep):.3e} to {max(deep):.3e}")
        span = (max(shallow) / max(deep)) if max(deep) > 0 else float("inf")
        print(f"  excess spans {span:.1f}x across the stack")
        print(f"  (our nondeterminism spans ~9000x, so an excess spanning far less than")
        print(f"   that is depth-independent — which is the shape T16's residual has)")
    results["verdict_neutral"] = neutral

    # Does moving toward their dim_batch move the vs-published difference?
    pub = jx_fetch.destination(jx_fetch.QWEN35_08B, cfg) / f"{SLUG}_jacobian_lens.pt"
    if pub.exists():
        p_J = load(pub)
        d_new = {d.layer: d.rel_frobenius for d in jx_cmp.compare_lenses(new, p_J)}
        d_old = {d.layer: d.rel_frobenius for d in jx_cmp.compare_lenses(a, p_J)}
        print(f"\n=== vs the published lens: does dim_batch={db} close the gap? ===")
        print(f"{'layer':>6} {'db=8 vs pub':>13} {'db=' + str(db) + ' vs pub':>15} {'change':>9}")
        print("-" * 46)
        for layer in sorted(d_new):
            chg = (d_new[layer] - d_old[layer]) / d_old[layer]
            print(f"{layer:>6} {d_old[layer]:>13.4e} {d_new[layer]:>15.4e} {chg:>+8.1%}")
        results["vs_published"] = {"db8": d_old, f"db{db}": d_new}
        moved = sum(1 for l in d_new if d_new[l] < d_old[l])
        print(f"\n  {moved}/{len(d_new)} layers moved closer to the published lens.")
        print(f"  Closer at most layers implicates dim_batch; unchanged argues their")
        print(f"  environment instead. Neither is conclusive on its own — {db} is not 128.")
    else:
        print(f"\n(published lens not found at {pub}; skipping the vs-published axis)")

    out = cfg.artifact_root / "measurements" / "dim-batch-neutrality"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"dim_batch_{db}_vs_8.json"
    results["published_artifact"] = jx_prov.artifact_facts(pub) if pub.exists() else None
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
