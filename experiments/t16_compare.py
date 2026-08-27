"""T16 -- compare our lens against the published artifact, on four axes.

Spec: ``environment-setup-and-first-fit``, stage 4. Consumes T15's two fits and T8's
downloaded artifact; produces the numbers T17 turns into a verdict.

The axes, and the floor that binds each
---------------------------------------

**Axis 0, ours-a versus ours-b.** The production-configuration noise envelope,
measured rather than projected. T18 measured the envelope uncompiled at
``dim_batch=8`` and said explicitly that it does not transfer here. Both our lenses
are fp32, so this axis carries no storage quantisation at all -- it is the true
run-to-run spread.

It is computed **first** because axes 1 and 2 are floored by it.

**Axis 1, ours versus published, per layer.** Relative Frobenius, floored per layer by
whichever is larger of the ~4.9e-4 fp16 storage floor and axis 0's measured envelope.
The published lens is fp16 whatever we do, so this axis keeps that floor.

**Axis 2, identity_distance.** Recomputed from the tensor at ``max(source_layers)``,
never read from ``config.yaml`` -- the published value there is transcribed from the
last row of the convergence CSV, so reading it back would compare a number against a
copy of itself rather than against the artifact.

**Axis 3, the convergence trace**, prompt-for-prompt over the same pinned count. Still
version-qualified: ``mean_rel_change`` is Neuronpedia's statistic and upstream's
same-named one computes something different.

``prompts_fitted`` is deliberately **not** an axis -- pinning fixes it by construction,
so it carries no information. But see the early-stop replay below, which recovers it as
a derived quantity.

Two things this driver is careful about
---------------------------------------

**Comparing fp32 against T18 needs a like-for-like translation.** T18's envelope was
measured on fp16-stored lenses, and an fp16 comparison *inflates* a sub-quantum
difference rather than erasing it -- see ``jlens_extensions.compare``. So axis 0 is
reported twice: in fp32 (the real envelope, what T16 should use) and with both sides
round-tripped through fp16 (what is comparable with T18's numbers). Reading the fp32
figure against T18's fp16 one would look like ``torch.compile`` had reduced the noise,
which is an artifact of the storage format rather than a result.

**The early-stopping rule is replayed, not run.** T15 pinned ``--n_prompts`` and
disabled early stopping, which removes ``prompts_fitted`` as an axis by construction.
Applying the published run's own rule (``stop_at_delta 0.002``, ``min_prompts 100``,
``stop_window 10``) to our completed trace puts it back as a derived quantity: if it
would have fired at or near 233, that is independent evidence the two runs converge
alike, obtained at zero extra compute and with none of the brittleness that made
pinning necessary in the first place.

Run::

    uv run python experiments/t16_compare.py
    uv run python experiments/t16_compare.py --runs a,b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import config as jx_config  # noqa: E402
from jlens_extensions import compare as jx_cmp  # noqa: E402
from jlens_extensions import fetch as jx_fetch  # noqa: E402
from jlens_extensions import provenance as jx_prov  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

MODEL_ID = "Qwen/Qwen3.5-0.8B"
SLUG = "Qwen3.5-0.8B"
N_PROMPTS = 233

# The published run's early-stopping rule, from its config.yaml (T1).
PUBLISHED_STOP = {"stop_at_delta": 0.002, "min_prompts": 100, "window": 10}

# T18's projection to n=233, measured uncompiled and on fp16-stored lenses. Carried
# as a cross-check against axis 0, not as the floor -- axis 0 supersedes it.
T18_PROJECTED = {
    0: 7.513e-04, 1: 4.965e-04, 2: 4.147e-04, 3: 3.448e-04, 4: 3.086e-04,
    6: 2.510e-04, 8: 2.253e-04, 10: 2.014e-04, 12: 1.819e-04, 15: 1.255e-04,
    18: 9.693e-05, 20: 5.853e-05, 21: 2.240e-05, 22: 2.644e-06,
}


def load_lens(path: Path):
    """Load a lens's Jacobian dict, upcast to fp32."""
    import torch

    from jlens.lens import JacobianLens

    lens = JacobianLens.load(str(path))
    return {k: v.to(torch.float32) for k, v in lens.jacobians.items()}, lens


def read_deltas(csv_path: Path) -> tuple[list[float], list[dict]]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    deltas = [float(r["mean_rel_change"]) for r in rows]
    return deltas, rows


def table(diffs, envelope=None, fp16_floor=None) -> None:
    """Print a per-layer table. With floors supplied, add the binding-floor verdict.

    ``fp16_floor`` is a per-layer *measured* map, not the 2**-11 constant -- see
    ``compare.fp16_storage_floor`` for why the constant is the wrong quantity here.
    """
    show = envelope is not None
    header = (f"{'layer':>6} {'rel_frobenius':>14} {'max_abs':>11} {'%differ':>8}"
              + (f" {'floor':>11} {'binds':>15} {'x floor':>8}" if show else ""))
    print(header)
    print("-" * len(header))
    for d in diffs:
        line = (f"{d.layer:>6} {d.rel_frobenius:>14.4e} {d.max_abs:>11.3e} "
                f"{d.frac_differing * 100:>7.1f}%")
        if show:
            b = jx_cmp.binding_constraint(
                d.rel_frobenius,
                envelope=envelope.get(d.layer),
                fp16_floor=(fp16_floor or {}).get(d.layer, jx_cmp.FP16_FLOOR),
            )
            line += (f" {b['floor']:>11.3e} {b['binds']:>15} {b['ratio_to_floor']:>8.2f}"
                     + ("  ABOVE" if b["above_floor"] else ""))
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="a,b", help="T15 run labels to compare (default: a,b)")
    args = parser.parse_args()
    labels = [x.strip() for x in args.runs.split(",") if x.strip()]
    if len(labels) != 2:
        raise SystemExit("--runs needs exactly two labels; the envelope is a pairwise measurement")

    published_dir = jx_fetch.destination(jx_fetch.QWEN35_08B, cfg)
    pub_lens_path = published_dir / f"{SLUG}_jacobian_lens.pt"
    pub_csv_path = published_dir / f"{SLUG}_convergence.csv"
    if not pub_lens_path.exists():
        raise SystemExit(
            f"published artifact not found at {pub_lens_path}. Run T8's fetcher first:\n"
            f"    uv run python -m jlens_extensions.fetch qwen3.5-0.8b"
        )

    ours = {}
    for label in labels:
        d = cfg.lenses / f"t15-{label}"
        path = d / f"{SLUG}_jacobian_lens.pt"
        if not path.exists():
            raise SystemExit(f"T15 run {label} not found at {path}. Run t15_validation_fit.py first.")
        ours[label] = {"dir": d, "lens_path": path}

    print(f"machine={cfg.machine}  model={MODEL_ID}  runs={labels}")
    print(f"published: {pub_lens_path}")

    a_label, b_label = labels
    a_J, a_obj = load_lens(ours[a_label]["lens_path"])
    b_J, b_obj = load_lens(ours[b_label]["lens_path"])
    p_J, p_obj = load_lens(pub_lens_path)
    print(f"loaded: ours {sorted(a_J)[0]}..{sorted(a_J)[-1]} ({len(a_J)} layers), "
          f"published {len(p_J)} layers")
    for name, obj in ((a_label, a_obj), (b_label, b_obj), ("published", p_obj)):
        print(f"  {name}: n_prompts={obj.n_prompts}")

    results: dict = {"task": "T16", "machine": cfg.machine, "model": MODEL_ID,
                     "runs": labels, "n_prompts": N_PROMPTS}

    # --- Axis 0: the production-config envelope ------------------------------
    print(f"\n=== Axis 0: {a_label} vs {b_label} — the production-config envelope (fp32) ===")
    print("Both lenses fp32, so no storage quantisation. This is the real run-to-run spread.")
    env_fp32 = jx_cmp.compare_lenses(a_J, b_J)
    table(env_fp32)
    envelope = {d.layer: d.rel_frobenius for d in env_fp32}
    results["axis0_envelope_fp32"] = [d.as_dict() for d in env_fp32]

    print(f"\n--- the same pair through fp16, for comparison with T18 ---")
    print("T18 measured fp16-stored lenses, and an fp16 comparison inflates a sub-quantum")
    print("difference. Only this column is comparable with T18's numbers.")
    env_fp16 = jx_cmp.compare_lenses(a_J, b_J, as_fp16=True)
    results["axis0_envelope_fp16"] = [d.as_dict() for d in env_fp16]
    hdr = f"{'layer':>6} {'fp32 (real)':>13} {'fp16 (vs T18)':>15} {'T18 proj':>11} {'fp16/T18':>9}"
    print(hdr)
    print("-" * len(hdr))
    for d32, d16 in zip(env_fp32, env_fp16):
        proj = T18_PROJECTED.get(d32.layer)
        ratio = f"{d16.rel_frobenius / proj:>9.2f}" if proj else f"{'—':>9}"
        projs = f"{proj:>11.3e}" if proj else f"{'—':>11}"
        print(f"{d32.layer:>6} {d32.rel_frobenius:>13.4e} {d16.rel_frobenius:>15.4e} {projs} {ratio}")

    # --- The floor, measured rather than assumed ------------------------------
    print("\n=== The fp16 storage floor, measured on our own fp32 lens ===")
    print("2**-11 = 4.883e-04 is the worst-case ELEMENT-WISE relative error. A per-layer")
    print("comparison reports a Frobenius aggregate, where quantisation partly cancels.")
    print("Scoring a difference against the element-wise bound flatters agreement.")
    measured_floor = jx_cmp.fp16_storage_floor(a_J)
    idfrac = jx_cmp.identity_fraction(p_J)
    hdr = (f"{'layer':>6} {'measured fp16':>14} {'2**-11':>11} {'ratio':>7} "
           f"{'||J-I||/||J||':>14}")
    print(hdr)
    print("-" * len(hdr))
    for layer in sorted(measured_floor):
        m = measured_floor[layer]
        print(f"{layer:>6} {m:>14.4e} {jx_cmp.FP16_FLOOR:>11.3e} "
              f"{m / jx_cmp.FP16_FLOOR:>7.2f} {idfrac[layer]:>14.4f}")
    results["measured_fp16_floor"] = measured_floor
    results["identity_fraction_published"] = idfrac
    print("  The last column is how much of the norm is NOT the identity diagonal.")
    print("  It rescales differences and floors alike, so it changes no ratio — it is")
    print("  reported because the criteria ask whether the metric is washed out.")

    # --- Axis 1: ours vs published -------------------------------------------
    for label, J in ((a_label, a_J), (b_label, b_J)):
        print(f"\n=== Axis 1: run {label} vs published, per layer ===")
        print("  floor = max(measured fp16 storage, measured a-vs-b envelope)")
        diffs = jx_cmp.compare_lenses(J, p_J)
        table(diffs, envelope=envelope, fp16_floor=measured_floor)
        above = [d.layer for d in diffs
                 if jx_cmp.binding_constraint(
                     d.rel_frobenius, envelope=envelope.get(d.layer),
                     fp16_floor=measured_floor.get(d.layer, jx_cmp.FP16_FLOOR))["above_floor"]]
        print(f"  layers above their binding floor: {above or 'none'}")
        worst = max(diffs, key=lambda d: d.rel_frobenius)
        print(f"  A1 threshold is 1e-2 relative; worst layer is L{worst.layer} at "
              f"{worst.rel_frobenius:.3e} = {worst.rel_frobenius / 1e-2:.1%} of it")
        results[f"axis1_vs_published_{label}"] = [d.as_dict() for d in diffs]
        results[f"axis1_above_floor_{label}"] = above

    # --- Axis 2: identity_distance, recomputed -------------------------------
    print("\n=== Axis 2: identity_distance, recomputed from the tensor ===")
    late = max(p_J)
    ids = {label: jx_cmp.identity_distance(J, late) for label, J in
           ((a_label, a_J), (b_label, b_J), ("published", p_J))}
    for label, value in ids.items():
        print(f"  {label:>10}: {value:.6f}  (layer {late})")
    spread = max(ids.values()) - min(ids.values())
    ours_mean = (ids[a_label] + ids[b_label]) / 2
    print(f"  ours mean {ours_mean:.6f} vs published {ids['published']:.6f}  "
          f"delta {ours_mean - ids['published']:+.2e}  "
          f"({abs(ours_mean - ids['published']) / ids['published']:.2e} relative)")
    print(f"  our own two runs differ by {abs(ids[a_label] - ids[b_label]):.2e} — "
          f"the scale at which this statistic is meaningful")
    results["axis2_identity_distance"] = {"layer": late, "values": ids, "spread": spread}

    # --- Axis 3: the convergence trace ---------------------------------------
    print("\n=== Axis 3: convergence trace, prompt-for-prompt ===")
    pub_deltas, pub_rows = read_deltas(pub_csv_path)
    trace_stats = {}
    for label in labels:
        our_deltas, our_rows = read_deltas(ours[label]["dir"] / f"{SLUG}_convergence.csv")
        n = min(len(our_deltas), len(pub_deltas))
        pairs = [(o, p) for o, p in zip(our_deltas[:n], pub_deltas[:n])
                 if o == o and p == p]
        abs_diff = [abs(o - p) for o, p in pairs]
        rel_diff = [abs(o - p) / p for o, p in pairs if p != 0]
        trace_stats[label] = {
            "n_compared": len(pairs),
            "max_abs_diff": max(abs_diff) if abs_diff else None,
            "mean_abs_diff": sum(abs_diff) / len(abs_diff) if abs_diff else None,
            "max_rel_diff": max(rel_diff) if rel_diff else None,
            "final_ours": our_deltas[-1],
            "final_published": pub_deltas[-1],
        }
        s = trace_stats[label]
        print(f"  run {label}: {s['n_compared']} prompts compared, "
              f"max |Δ| {s['max_abs_diff']:.3e}, mean |Δ| {s['mean_abs_diff']:.3e}, "
              f"max relative {s['max_rel_diff']:.2%}")
        print(f"    final mean_rel_change: ours {s['final_ours']:.8f} vs "
              f"published {s['final_published']:.8f}")
    results["axis3_convergence"] = trace_stats

    # --- The recovered axis: replay the published stop rule ------------------
    print("\n=== Recovered: where the published stop rule would have fired on our trace ===")
    print("prompts_fitted is not an axis (pinning fixes it), but the rule can be replayed.")
    replay = {}
    for label in labels:
        our_deltas, _ = read_deltas(ours[label]["dir"] / f"{SLUG}_convergence.csv")
        r = jx_cmp.replay_early_stop(our_deltas, **PUBLISHED_STOP)
        replay[label] = r
        if r["would_stop_at"] is not None:
            print(f"  run {label}: would stop at {r['would_stop_at']} prompts "
                  f"(smoothed Δmean {r['smoothed_delta']:.3e} < {PUBLISHED_STOP['stop_at_delta']}), "
                  f"published stopped at {N_PROMPTS} — delta {r['would_stop_at'] - N_PROMPTS:+d}")
        else:
            print(f"  run {label}: would NOT have stopped within {len(our_deltas)} prompts "
                  f"(final smoothed Δmean {r['smoothed_delta_final']:.3e})")
    pub_replay = jx_cmp.replay_early_stop(pub_deltas, **PUBLISHED_STOP)
    replay["published_selfcheck"] = pub_replay
    print(f"  self-check, published trace: {pub_replay['would_stop_at']} "
          f"(should be {N_PROMPTS}; validates the replay against the run that used it)")
    results["recovered_early_stop"] = replay

    # --- A1's required negative control --------------------------------------
    print("\n=== A1 control: can the metric see disagreement? ===")
    print("A1 names a different model's published lens. That is undefined at this shape —")
    print("no other published lens has d_model=1024 (nearest: gpt2-small 768, qwen3-1.7b")
    print("2048), so a per-layer Frobenius against one cannot be computed. This is the")
    print("shape-safe equivalent: match every one of our layers against every published")
    print("layer and check the right pairing wins.")
    control = jx_cmp.layer_correspondence(a_J, p_J)
    hdr = (f"{'layer':>6} {'best match':>11} {'ok':>4} {'self':>12} {'runner-up':>10} "
           f"{'its score':>12} {'margin':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in control["rows"]:
        print(f"{r['layer']:>6} {r['best_match']:>11} {'yes' if r['correct'] else 'NO':>4} "
              f"{r['self_score']:>12.4e} {str(r['runner_up']):>10} "
              f"{r['runner_up_score']:>12.4e} {r['margin_x']:>8.1f}x")
    verdict = "PASSES" if control["all_correct"] else "FAILS"
    print(f"  control {verdict}: {control['n_correct']}/{control['n_layers']} layers "
          f"matched themselves, min margin {control['min_margin_x']:.1f}x, "
          f"median {control['median_margin_x']:.1f}x")
    if not control["all_correct"]:
        print("  A FAILING CONTROL VOIDS AXIS 1. The metric has not been shown able to")
        print("  distinguish the right lens from a wrong one, so its agreement is not")
        print("  evidence. Investigate before reading any verdict above.")
    results["a1_control_layer_correspondence"] = control

    out_dir = cfg.artifact_root / "measurements" / "t16"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t16_compare.json"
    results["published_artifact"] = jx_prov.artifact_facts(pub_lens_path)
    results["fp16_floor"] = jx_cmp.FP16_FLOOR
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    print("\nT17 turns this into a manifest and a verdict. The verdict must state, per axis,")
    print("the floor that binds it and where that floor came from — and that a comparison")
    print("against a single published run carries their noise draw as well as ours.")


if __name__ == "__main__":
    main()
