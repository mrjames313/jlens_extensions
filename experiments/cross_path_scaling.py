"""Does a cross-path difference average down at the same rate as run-to-run noise?

The load-bearing gap. Everything the single-prompt instrument claims rests on taking
a difference measured at n=1 and dividing by ``n**alpha`` to predict what two
233-prompt fits will differ by. ``alpha = 0.473 +- 0.010`` is measured
(`f-2026-08-27-envelope-scaling`) -- but it was measured for **within-variant
run-to-run noise**, two runs of the *same* computation. Applying it to a difference
between two *different* execution paths is an inference, and this driver is what
turns it into a measurement.

The prediction, registered before the run
-----------------------------------------

Cross-path differences should scale at **alpha ~ 0.50**, distinguishably above the
within-variant 0.473.

The reasoning is not decorative -- it is what the measurement can falsify. A lens is
a running mean, so a difference between two runs is ``(1/n) sum_i delta_i`` and the
exponent reports how independent the ``delta_i`` are across prompts: 0.5 for fully
independent, 0 for perfectly systematic. `f-2026-08-27-envelope-scaling` attributed
the within-variant deficit below 0.5 to the ``delta_i`` being *slightly positively
correlated* -- "the same kernels making similar reduction-order choices on different
inputs". Two different configurations do not share kernels, so they should not share
that correlation, and should sit at the independent limit.

There is already indirect evidence. Predicting the ours-vs-published residual from
the measured cross-configuration offset overshoots by a **consistent** 1.19-1.27x
(sd 0.031) across five layers spanning a 17x range in magnitude. A constant
overshoot across that range is an exponent error rather than a model error, and the
exponent that closes it is 0.511.

So this run has three outcomes and they are not equally interesting:

* **alpha ~ 0.50** -- the residual is explained to within ~10%, the instrument's
  cross-path use is justified, and the last open thread from the reproduction
  verdict closes.
* **alpha ~ 0.473** -- cross-path differences scale like noise after all, and the
  consistent 1.23x overshoot is something else that needs its own account.
* **alpha materially below 0.4** -- a systematic component that does not average
  down. That would matter most: it would mean cross-path differences do *not*
  vanish with prompt count, and A1's same-variant qualifier is load-bearing at
  every prompt count rather than fading.

Design
------

One fit per process, several per configuration, each fit to ``max(targets)`` prompts
saving a lens at every target on the way -- ``fit(resume=True)`` continues the running
sum, so a single pass yields every prompt count at no extra compute
(`envelope_vs_n.py` established this trick and this driver reuses it).

**One process per run is not tidiness.** Which compile variant a process draws is
per-process (`f-2026-08-28-compile-miscompilation`), so two runs sharing a process
share a variant and the cross-variant comparison measures nothing. Runs are labelled
by their prompt-1 ``identity_distance``, which is the variant fingerprint, and grouped
after the fact -- you cannot ask for a variant, only observe which one you got.

At each target n the runs are grouped into (configuration, variant) and compared:

* **within-variant** pairs give the noise null at that n -- and independently
  re-measure the 0.473 exponent as a control.
* **cross-variant, same configuration** and **cross-configuration** pairs give the
  quantities under test, each with the noise subtracted in quadrature
  (``compare.group_offset``) before the exponent is fitted.

Run::

    uv run python experiments/cross_path_scaling.py
    uv run python experiments/cross_path_scaling.py --runs 4 --max-n 15   # rehearsal
    uv run python experiments/cross_path_scaling.py --analyse-only
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import config as jx_config  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

from jlens_extensions.childproc import emit_result, run_child  # noqa: E402
from jlens_extensions.compare import (  # noqa: E402
    fit_power_law, group_offset, rel_frobenius,
)
from jlens_extensions.compile_policy import (  # noqa: E402
    apply_compile_policy, cluster_variants, identity_in_band,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
SLUG = "Qwen3.5-0.8B"
MAX_SEQ_LEN = 128
DIM_BATCH = 8
DATASET, DATASET_CONFIG, DATASET_SPLIT, TEXT_FIELD = (
    "Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text")
MAX_CHARS = 2000

#: Log-spaced so the exponent is fitted over the widest lever arm the budget allows,
#: rather than over a cluster of large n where the differences are all similar.
DEFAULT_TARGETS = (1, 2, 4, 8, 15, 30)

#: Two configurations, both compiled and therefore both fast. `none` would be the
#: cleanest second configuration -- single-variant, no compile risk -- but runs 2.75x
#: slower, and the cross-configuration signal is what matters rather than which
#: configurations supply it.
DEFAULT_POLICIES = ("linear-attn", "all")

CHILD_TIMEOUT_S = 3600


def out_root() -> Path:
    return cfg.artifact_root / "measurements" / "cross-path-scaling"


def snapshot_path(label: str, n: int) -> Path:
    return out_root() / f"run-{label}" / f"{SLUG}_n{n:04d}.pt"


# --------------------------------------------------------------------------- child


def child_run(label: str, policy: str, targets: tuple[int, ...]) -> dict:
    """One fit, in this process, saving a lens at each target prompt count."""
    import torch
    import transformers

    import jlens
    from jlens.fitting import fit

    from fit_lens import load_prompts

    prompts = load_prompts(dataset=DATASET, config=DATASET_CONFIG, split=DATASET_SPLIT,
                           text_field=TEXT_FIELD, n_prompts=max(targets),
                           max_chars=MAX_CHARS)
    if len(prompts) != max(targets):
        raise SystemExit(f"corpus gave {len(prompts)} prompts, need {max(targets)}")

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok, compile=False)
    applied = apply_compile_policy(model, policy)

    checkpoint = cfg.checkpoints / f"xps-{label}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # A stale checkpoint would silently resume another run's running sum, which is
    # the one failure here that would look like a result rather than an error.
    checkpoint.unlink(missing_ok=True)
    snapshot_path(label, targets[0]).parent.mkdir(parents=True, exist_ok=True)

    first_identity = {}
    saved = {}
    started = time.time()
    for i, n in enumerate(targets):
        def callback(p, _store=first_identity):
            if p.n_done == 1:
                _store["v"] = p.identity_distance

        lens = fit(model, prompts[:n], dim_batch=DIM_BATCH, max_seq_len=MAX_SEQ_LEN,
                   checkpoint_path=str(checkpoint), checkpoint_every=1,
                   resume=(i > 0), metrics_callback=callback)
        if lens.n_prompts != n:
            raise SystemExit(f"run {label}: lens reports n_prompts={lens.n_prompts}, "
                             f"expected {n}; the resume path did not continue the sum")
        path = snapshot_path(label, n)
        lens.save(str(path), dtype=torch.float32)
        saved[n] = str(path)
    checkpoint.unlink(missing_ok=True)
    return {"label": label, "policy": policy, "compiled": applied["n_compiled"],
            "identity_distance": first_identity.get("v"), "snapshots": saved,
            "wall_s": round(time.time() - started, 1)}


# ------------------------------------------------------------------------- analysis


def load_at(runs: list[dict], n: int) -> dict[str, dict]:
    import torch

    out = {}
    for r in runs:
        path = r["snapshots"].get(str(n)) or r["snapshots"].get(n)
        if path and Path(path).exists():
            out[r["label"]] = torch.load(path, map_location="cpu")
    return out


def classify(a: dict, b: dict) -> str:
    if a["policy"] != b["policy"]:
        return "cross-config"
    return "within-variant" if a["variant"] == b["variant"] else "cross-variant"


def offsets_at(runs: list[dict], n: int) -> dict[str, dict[int, dict]]:
    """Per-category, per-layer offset at one prompt count."""
    tensors = load_at(runs, n)
    have = [r for r in runs if r["label"] in tensors]
    buckets: dict[str, dict[int, list[float]]] = {}
    for a, b in itertools.combinations(have, 2):
        kind = classify(a, b)
        A, B = tensors[a["label"]], tensors[b["label"]]
        for layer in sorted(A):
            buckets.setdefault(kind, {}).setdefault(layer, []).append(
                rel_frobenius(A[layer], B[layer]))
    null = buckets.get("within-variant", {})
    out = {}
    for kind, per_layer in buckets.items():
        if kind == "within-variant":
            # The null measures itself: no subtraction, the raw pair rms IS the noise.
            out[kind] = {l: {"offset": (sum(v * v for v in vals) / len(vals)) ** 0.5,
                             "n_pairs": len(vals)}
                         for l, vals in per_layer.items()}
        else:
            out[kind] = {l: group_offset(null.get(l, []), vals)
                         for l, vals in per_layer.items()}
    return out


def analyse(runs: list[dict], targets: tuple[int, ...]) -> dict:
    per_n = {n: offsets_at(runs, n) for n in targets}
    kinds = sorted({k for v in per_n.values() for k in v})
    layers = sorted({l for v in per_n.values() for k in v.values() for l in k})
    fits: dict[str, dict[int, dict]] = {}
    for kind in kinds:
        for layer in layers:
            points = []
            for n in targets:
                row = per_n.get(n, {}).get(kind, {}).get(layer)
                if row and row.get("offset"):
                    points.append((n, row["offset"]))
            if len(points) >= 3:
                try:
                    fits.setdefault(kind, {})[layer] = fit_power_law(points)
                except ValueError:
                    pass
    return {"per_n": per_n, "fits": fits, "targets": list(targets)}


def report(result: dict, runs: list[dict], layers_shown=(0, 4, 8, 15, 22)) -> None:
    print("\n=== runs, grouped by the variant their process drew ===")
    for r in sorted(runs, key=lambda r: (r["policy"], r["identity_distance"] or 0)):
        print(f"  {r['label']:>6}  {r['policy']:>12}  id={r['identity_distance']:.6f}"
              f"  -> variant v{r['variant']}   ({r['wall_s']/60:.1f} min)")

    print("\n=== offset vs prompt count, median layer L8 ===")
    hdr = f"{'kind':>16} " + " ".join(f"{('n=' + str(n)):>11}" for n in result["targets"])
    print(hdr); print("-" * len(hdr))
    for kind in sorted(result["per_n"].get(result["targets"][0], {})):
        cells = []
        for n in result["targets"]:
            row = result["per_n"].get(n, {}).get(kind, {}).get(8)
            cells.append(f"{row['offset']:>11.3e}" if row and row.get("offset")
                         else f"{'-':>11}")
        print(f"{kind:>16} " + " ".join(cells))

    print("\n=== fitted exponent, per layer ===")
    print("  alpha 0.5 = per-prompt differences independent; 0 = systematic, never averages down")
    hdr = f"{'kind':>16} " + " ".join(f"{('L' + str(l)):>18}" for l in layers_shown)
    print(hdr); print("-" * len(hdr))
    for kind, per_layer in sorted(result["fits"].items()):
        cells = []
        for l in layers_shown:
            f = per_layer.get(l)
            cells.append(f"{f['alpha']:>9.3f} (r2 {f['r_squared']:.2f})" if f
                         else f"{'-':>18}")
        print(f"{kind:>16} " + " ".join(cells))

    for kind, per_layer in sorted(result["fits"].items()):
        alphas = [f["alpha"] for f in per_layer.values()]
        r2s = [f["r_squared"] for f in per_layer.values()]
        if not alphas:
            continue
        mean = sum(alphas) / len(alphas)
        sd = (sum((a - mean) ** 2 for a in alphas) / max(len(alphas) - 1, 1)) ** 0.5
        print(f"\n  {kind}: alpha = {mean:.3f} +- {sd:.3f} over {len(alphas)} layers, "
              f"min r2 {min(r2s):.3f}")

    wv = result["fits"].get("within-variant", {})
    if wv:
        a = sum(f["alpha"] for f in wv.values()) / len(wv)
        print(f"\n  Control: within-variant should reproduce 0.473 +- 0.010 "
              f"(f-2026-08-27-envelope-scaling). Measured {a:.3f}.")
        print(f"  If the control misses, the cross-path numbers are not trustworthy either.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=6,
                        help="fits per configuration (default 6; the variant split is "
                             "~50/50 at linear-attn:8, so 6 gives both with ~97%% odds)")
    parser.add_argument("--max-n", type=int, default=max(DEFAULT_TARGETS))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--analyse-only", action="store_true",
                        help="skip fitting; re-tabulate from saved snapshots")
    parser.add_argument("--child")
    parser.add_argument("--label")
    parser.add_argument("--policy")
    args = parser.parse_args()

    targets = tuple(n for n in DEFAULT_TARGETS if n <= args.max_n) or (args.max_n,)
    if targets[-1] != args.max_n:
        targets = targets + (args.max_n,)

    if args.child:
        emit_result(child_run(args.label, args.policy, targets))
        return

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    manifest = out_root() / "runs.json"
    if args.analyse_only:
        runs = json.loads(manifest.read_text())["runs"]
    else:
        out_root().mkdir(parents=True, exist_ok=True)
        per_run_s = args.max_n * (15.5 if "none" not in policies else 43)
        print(f"machine={cfg.machine}  model={MODEL_ID}  dim_batch={DIM_BATCH}")
        print(f"{len(policies)} configurations x {args.runs} runs, targets {targets}")
        print(f"Estimated {len(policies) * args.runs * per_run_s / 60:.0f} minutes.\n")
        runs = []
        for policy in policies:
            for i in range(args.runs):
                label = f"{policy}-{i}"
                r = run_child(__file__, label,
                              ["--child", "fit", "--label", label, "--policy", policy,
                               "--max-n", str(args.max_n)],
                              timeout_s=CHILD_TIMEOUT_S, quiet=True)
                if r:
                    runs.append(r)
                    manifest.write_text(json.dumps({"runs": runs}, indent=2) + "\n")

    # A miscompiled run is not a variant; drop it before the fingerprints are clustered.
    reference = next((r["identity_distance"] for r in runs if r["policy"] == "none"), None)
    sound = [r for r in runs if r["identity_distance"] and
             identity_in_band(r["identity_distance"], reference or 0.5314, 0.01)]
    if len(sound) < len(runs):
        print(f"\ndropped {len(runs) - len(sound)} run(s) failing the identity gate")

    for policy in {r["policy"] for r in sound}:
        members = [r for r in sound if r["policy"] == policy]
        for vi, cluster in enumerate(
                cluster_variants([r["identity_distance"] for r in members])):
            for i in cluster:
                members[i]["variant"] = vi

    result = analyse(sound, targets)
    report(result, sound)
    dest = out_root() / "cross_path_scaling.json"
    dest.write_text(json.dumps(
        {"task": "cross-path-scaling", "machine": cfg.machine, "model": MODEL_ID,
         "dim_batch": DIM_BATCH, "targets": list(targets),
         "runs": [{k: v for k, v in r.items() if k != "snapshots"} for r in sound],
         "fits": result["fits"]}, indent=2, default=str) + "\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
