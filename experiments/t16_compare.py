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

Per-rung, and the one thing that does not generalise
---------------------------------------------------

Everything that differs between rungs is read from the published artifact rather than
typed: ``prompts_fitted``, the early-stopping rule, the artifact stem, the lens
directory. The exception is **axis 0**, which is not a per-rung constant but a
*measurement requiring two fits of the same configuration*. Where a rung was fitted
once -- 4B, by plan decision 4 of ``workspace-band-location``, which bought the
envelope from the single-prompt instrument instead of a second ~17 h fit -- axis 0
cannot be computed and the envelope must be supplied with ``--envelope``. It is then a
prediction, and axes 1 and 2 inherit its uncertainty; the driver labels it as such in
both the output and the JSON so a verdict cannot quietly treat it as measured.

Run::

    uv run python experiments/t16_compare.py                       # 0.8B, both fits
    uv run python experiments/t16_compare.py --runs a,b
    uv run python experiments/t16_compare.py --model Qwen/Qwen3.5-4B \\
        --runs a --envelope $JLENS_ARTIFACT_ROOT/measurements/t17/envelope_4b.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
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

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"

# The published run's early-stopping rule. Read from the artifact's config.yaml rather
# than typed -- see ``published_stop_rule``. These are the harness defaults, used only
# for keys the config does not state, and every key is reported with its source.
STOP_RULE_DEFAULTS = {"stop_at_delta": None, "min_prompts": 100, "window": 10}

#: T18's projection to n=233, measured uncompiled and on fp16-stored lenses. Carried
#: as a cross-check against axis 0, not as the floor -- axis 0 supersedes it.
#:
#: **Keyed by model, and it must stay that way.** These are 0.8B layer indices 0..22.
#: 4B has layers 0..30, so at 4B every one of these keys still *resolves* -- printing a
#: 0.8B projection beside a 4B measurement, and a meaningless ratio next to it, with
#: nothing in the output marking it wrong. That is the same failure shape as the
#: hardcoded ``N_PROMPTS = 233`` this driver's sibling carried: plausible output, wrong
#: comparison. A rung with no entry gets no column, which is the honest answer.
T18_PROJECTED_BY_MODEL: dict[str, dict[int, float]] = {}
T18_PROJECTED_BY_MODEL[DEFAULT_MODEL] = {
    0: 7.513e-04, 1: 4.965e-04, 2: 4.147e-04, 3: 3.448e-04, 4: 3.086e-04,
    6: 2.510e-04, 8: 2.253e-04, 10: 2.014e-04, 12: 1.819e-04, 15: 1.255e-04,
    18: 9.693e-05, 20: 5.853e-05, 21: 2.240e-05, 22: 2.644e-06,
}


@dataclass(frozen=True)
class CompareTarget:
    """What differs between rungs. Resolved once from the published artifact."""

    model_id: str
    slug: str
    published: "jx_fetch.PublishedLens"
    n_prompts: int          # results.prompts_fitted, READ not typed

    @property
    def is_default_rung(self) -> bool:
        return self.model_id == DEFAULT_MODEL

    def lens_dir(self, label: str) -> Path:
        """Where t15_validation_fit.py put run ``label`` for this rung.

        Must mirror that driver's own stem exactly: it leaves the default rung
        unqualified so ``t15-a``/``t15-b`` keep working, and qualifies every other.
        """
        stem = f"t15-{label}" if self.is_default_rung else f"t15-{self.slug}-{label}"
        return cfg.lenses / stem


def _published_config(target_model: str) -> tuple["jx_fetch.PublishedLens", Path]:
    matches = [lens for lens in jx_fetch.REGISTRY.values() if lens.hf_model == target_model]
    if not matches:
        raise SystemExit(
            f"no published lens registered for {target_model}; known: "
            f"{sorted(l.hf_model for l in jx_fetch.REGISTRY.values())}. Add it to "
            f"fetch.py's REGISTRY -- every number here is scored against a published "
            f"artifact, so there is nothing to compare without one."
        )
    lens = matches[0]
    config_path = jx_fetch.destination(lens, cfg) / "config.yaml"
    if not config_path.exists():
        raise SystemExit(
            f"no published config at {config_path}. Fetch the artifact first:\n"
            f"    uv run python -m jlens_extensions.fetch --model {lens.model}"
        )
    return lens, config_path


def _scalar(text: str, key: str):
    found = re.search(rf"^\s*{key}:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
    return found.group(1) if found else None


def resolve_target(model_id: str) -> CompareTarget:
    """Resolve the rung from the published artifact, never from a typed constant.

    ``prompts_fitted`` differs per rung -- 233 at 0.8B, 417 at 4B -- and is recorded
    in the output and used to frame the early-stop replay. Reading it from the very
    artifact axis 1 compares against makes the two impossible to disagree.
    """
    from fit_lens import _slug

    lens, config_path = _published_config(model_id)
    raw = _scalar(config_path.read_text(), "prompts_fitted")
    if raw is None:
        raise SystemExit(
            f"{config_path} has no results.prompts_fitted line. Do not substitute a "
            f"default -- the pin is the comparison."
        )
    return CompareTarget(model_id=model_id, slug=_slug(model_id),
                         published=lens, n_prompts=int(raw))


def published_stop_rule(target: CompareTarget) -> tuple[dict, dict]:
    """The published run's early-stopping rule, read from its own config.yaml.

    Returns the rule and a per-key record of where each value came from. The rule is
    replayed over our trace, so a value silently carried over from another rung would
    produce a wrong ``would_stop_at`` and no error. ``stop_at_delta`` is refused rather
    than defaulted: it is the threshold the whole replay turns on, and the harness has
    no default for it -- a fit without it does not early-stop at all.
    """
    _, config_path = _published_config(target.model_id)
    text = config_path.read_text()
    rule, source = {}, {}
    for key, cfg_key in (("stop_at_delta", "stop_at_delta"),
                         ("min_prompts", "min_prompts"),
                         ("window", "stop_window")):
        raw = _scalar(text, cfg_key)
        if raw is not None:
            rule[key] = float(raw) if key == "stop_at_delta" else int(float(raw))
            source[key] = "read"
        elif STOP_RULE_DEFAULTS[key] is None:
            raise SystemExit(
                f"{config_path} states no {cfg_key}, which is the threshold the "
                f"early-stop replay turns on. It has no harness default -- a fit "
                f"without it does not early-stop -- so there is nothing to replay."
            )
        else:
            rule[key] = STOP_RULE_DEFAULTS[key]
            source[key] = "harness default (config states none)"
    return rule, source


def load_envelope(path: Path | None) -> tuple[dict[int, float], dict]:
    """A per-layer run-to-run envelope supplied from outside, for a single-fit rung.

    Axis 0 measures the envelope by comparing two fits of the same configuration. A
    rung fitted once has no such pair, and the alternative is not "no floor" -- it is
    a *predicted* envelope from the single-prompt instrument
    (``f-2026-08-30-single-prompt-envelope-instrument``), which is what plan decision 4
    of ``workspace-band-location`` chose over spending a second fit.

    Accepts either ``{"per_layer": {...}, ...}`` with free-form provenance keys
    alongside, or a bare ``{layer: value}`` map. The provenance is carried into the
    output rather than dropped: a predicted floor and a measured one license different
    verdicts, and the JSON is what a later reader has.
    """
    if path is None:
        return {}, {"kind": "none", "note": "--no-envelope; fp16 storage floor alone"}
    if not path.exists():
        raise SystemExit(f"--envelope {path} does not exist")
    blob = json.loads(path.read_text())
    per_layer = blob.get("per_layer", blob) if isinstance(blob, dict) else None
    if not isinstance(per_layer, dict) or not per_layer:
        raise SystemExit(
            f"--envelope {path} carries no per-layer map. Expected "
            f'{{"per_layer": {{"0": 1.2e-4, ...}}}} or a bare {{layer: value}} map.'
        )
    try:
        envelope = {int(k): float(v) for k, v in per_layer.items()}
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"--envelope {path}: per-layer map is not {{int: float}} ({exc})")
    prov = {k: v for k, v in blob.items() if k != "per_layer"} if isinstance(blob, dict) else {}
    prov.update({"kind": "predicted", "path": str(path), "n_layers": len(envelope)})
    return envelope, prov


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
    show = envelope is not None or fp16_floor is not None
    envelope = envelope or {}
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
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HF model id of the rung to compare (default: {DEFAULT_MODEL})")
    parser.add_argument("--runs", default="a,b",
                        help="T15 run labels (default: a,b). One label is allowed where "
                             "the rung was fitted once; see --envelope.")
    parser.add_argument("--envelope", type=Path, default=None,
                        help="JSON carrying a per-layer run-to-run envelope, for a rung "
                             "with a single fit. Either {\"per_layer\": {layer: value}} "
                             "with optional source/alpha/n metadata, or a bare "
                             "{layer: value} map.")
    parser.add_argument("--no-envelope", action="store_true",
                        help="Single-fit rung, deliberately scored against the fp16 "
                             "storage floor alone. Weaker, and labelled as such.")
    args = parser.parse_args()
    labels = [x.strip() for x in args.runs.split(",") if x.strip()]
    if len(labels) not in (1, 2):
        raise SystemExit("--runs takes one or two labels")
    if len(labels) == 1 and not (args.envelope or args.no_envelope):
        raise SystemExit(
            "one run label, so axis 0 cannot be measured -- it is a pairwise comparison "
            "of two fits of the same configuration.\n"
            "Axes 1 and 2 are floored by that envelope, so scoring them without one "
            "silently substitutes the fp16 storage floor and reports a tighter floor "
            "than the measurement supports.\n"
            "Supply the envelope the single-prompt instrument predicts:\n"
            "    --envelope <path to per-layer json>\n"
            "or state the weaker comparison explicitly with --no-envelope."
        )

    target = resolve_target(args.model)
    stop_rule, stop_source = published_stop_rule(target)
    slug = target.slug

    published_dir = jx_fetch.destination(target.published, cfg)
    pub_lens_path = published_dir / f"{slug}_jacobian_lens.pt"
    pub_csv_path = published_dir / f"{slug}_convergence.csv"
    if not pub_lens_path.exists():
        raise SystemExit(
            f"published artifact not found at {pub_lens_path}. Run T8's fetcher first:\n"
            f"    uv run python -m jlens_extensions.fetch --model {target.published.model}"
        )

    ours = {}
    for label in labels:
        d = target.lens_dir(label)
        path = d / f"{slug}_jacobian_lens.pt"
        if not path.exists():
            raise SystemExit(f"T15 run {label} not found at {path}. Run t15_validation_fit.py first.")
        ours[label] = {"dir": d, "lens_path": path}

    print(f"machine={cfg.machine}  model={target.model_id}  runs={labels}")
    print(f"published: {pub_lens_path}  (prompts_fitted={target.n_prompts}, read from config.yaml)")
    print(f"published stop rule: " + ", ".join(
        f"{k}={v} [{stop_source[k]}]" for k, v in stop_rule.items()))

    a_label = labels[0]
    b_label = labels[1] if len(labels) > 1 else None
    a_J, a_obj = load_lens(ours[a_label]["lens_path"])
    b_J, b_obj = (load_lens(ours[b_label]["lens_path"]) if b_label else (None, None))
    p_J, p_obj = load_lens(pub_lens_path)
    print(f"loaded: ours {sorted(a_J)[0]}..{sorted(a_J)[-1]} ({len(a_J)} layers), "
          f"published {len(p_J)} layers")
    loaded = [(a_label, a_obj)] + ([(b_label, b_obj)] if b_label else []) + [("published", p_obj)]
    for name, obj in loaded:
        print(f"  {name}: n_prompts={obj.n_prompts}")
    for name, obj in loaded[:-1]:
        if obj.n_prompts != target.n_prompts:
            print(f"  WARNING: run {name} was fitted on {obj.n_prompts} prompts but the "
                  f"published artifact reports {target.n_prompts}. Axis 1 is not "
                  f"like-for-like; the pin is the comparison.")

    results: dict = {"task": "T16", "machine": cfg.machine, "model": target.model_id,
                     "runs": labels, "n_prompts": target.n_prompts}

    # --- Axis 0: the production-config envelope ------------------------------
    if b_label:
        print(f"\n=== Axis 0: {a_label} vs {b_label} — the production-config envelope (fp32) ===")
        print("Both lenses fp32, so no storage quantisation. This is the real run-to-run spread.")
        env_fp32 = jx_cmp.compare_lenses(a_J, b_J)
        table(env_fp32)
        envelope = {d.layer: d.rel_frobenius for d in env_fp32}
        results["axis0_envelope_fp32"] = [d.as_dict() for d in env_fp32]
        results["envelope_provenance"] = {"kind": "measured", "runs": labels}

        print(f"\n--- the same pair through fp16, for comparison with T18 ---")
        print("T18 measured fp16-stored lenses, and an fp16 comparison inflates a sub-quantum")
        print("difference. Only this column is comparable with T18's numbers.")
        env_fp16 = jx_cmp.compare_lenses(a_J, b_J, as_fp16=True)
        results["axis0_envelope_fp16"] = [d.as_dict() for d in env_fp16]
        projected = T18_PROJECTED_BY_MODEL.get(target.model_id)
        if projected is None:
            print(f"  (no T18 projection for {target.model_id} — column omitted. T18 was "
                  f"measured at 0.8B and its layer indices do not name the same layers here.)")
        hdr = (f"{'layer':>6} {'fp32 (real)':>13} {'fp16 (vs T18)':>15}"
               + (f" {'T18 proj':>11} {'fp16/T18':>9}" if projected else ""))
        print(hdr)
        print("-" * len(hdr))
        for d32, d16 in zip(env_fp32, env_fp16):
            line = f"{d32.layer:>6} {d32.rel_frobenius:>13.4e} {d16.rel_frobenius:>15.4e}"
            if projected:
                proj = projected.get(d32.layer)
                ratio = f"{d16.rel_frobenius / proj:>9.2f}" if proj else f"{'—':>9}"
                line += (f"{proj:>11.3e}" if proj else f"{'—':>11}") + f" {ratio}"
            print(line)
    else:
        print(f"\n=== Axis 0: not measured — this rung was fitted once ===")
        envelope, prov = load_envelope(args.envelope)
        results["envelope_provenance"] = prov
        if envelope:
            print(f"Envelope supplied from {args.envelope}: {prov}")
            print("It is a PREDICTION, not a measurement of these two tensors. Axes 1 and 2")
            print("inherit its uncertainty, and the verdict must say so.")
            hdr = f"{'layer':>6} {'predicted envelope':>20}"
            print(hdr)
            print("-" * len(hdr))
            for layer in sorted(envelope):
                print(f"{layer:>6} {envelope[layer]:>20.4e}")
        else:
            print("--no-envelope: axes 1 and 2 are floored by fp16 storage alone.")
            print("binding_constraint takes the LARGER of the two floors, so dropping the")
            print("envelope can only lower the floor, never raise it — the comparison gets")
            print("stricter, not laxer. Where the envelope would have bound (the earliest")
            print("layers, where it is largest), run-to-run noise will read as above-floor.")
            print("Under-reporting agreement, not over-reporting it; state it either way.")

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
    our_lenses = [(a_label, a_J)] + ([(b_label, b_J)] if b_label else [])
    envelope_kind = {"measured": f"measured {a_label}-vs-{b_label} envelope",
                     "predicted": "PREDICTED envelope (not measured on these tensors)",
                     "none": "no envelope — fp16 storage alone"}[
                         results["envelope_provenance"]["kind"]]
    for label, J in our_lenses:
        print(f"\n=== Axis 1: run {label} vs published, per layer ===")
        print(f"  floor = max(measured fp16 storage, {envelope_kind})")
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
    ids = {label: jx_cmp.identity_distance(J, late)
           for label, J in our_lenses + [("published", p_J)]}
    for label, value in ids.items():
        print(f"  {label:>10}: {value:.6f}  (layer {late})")
    spread = max(ids.values()) - min(ids.values())
    ours_mean = sum(ids[label] for label, _ in our_lenses) / len(our_lenses)
    print(f"  ours mean {ours_mean:.6f} vs published {ids['published']:.6f}  "
          f"delta {ours_mean - ids['published']:+.2e}  "
          f"({abs(ours_mean - ids['published']) / ids['published']:.2e} relative)")
    if b_label:
        print(f"  our own two runs differ by {abs(ids[a_label] - ids[b_label]):.2e} — "
              f"the scale at which this statistic is meaningful")
    else:
        print("  one run, so this rung has no same-configuration spread to read the")
        print("  delta against. It is a point estimate; the envelope above is what")
        print("  supplies its scale, and that envelope is predicted rather than measured.")
    results["axis2_identity_distance"] = {"layer": late, "values": ids, "spread": spread}

    # --- Axis 3: the convergence trace ---------------------------------------
    print("\n=== Axis 3: convergence trace, prompt-for-prompt ===")
    pub_deltas, pub_rows = read_deltas(pub_csv_path)
    trace_stats = {}
    for label in labels:
        our_deltas, our_rows = read_deltas(ours[label]["dir"] / f"{slug}_convergence.csv")
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
        our_deltas, _ = read_deltas(ours[label]["dir"] / f"{slug}_convergence.csv")
        r = jx_cmp.replay_early_stop(our_deltas, **stop_rule)
        replay[label] = r
        if r["would_stop_at"] is not None:
            print(f"  run {label}: would stop at {r['would_stop_at']} prompts "
                  f"(smoothed Δmean {r['smoothed_delta']:.3e} < {stop_rule['stop_at_delta']}), "
                  f"published stopped at {target.n_prompts} — "
                  f"delta {r['would_stop_at'] - target.n_prompts:+d}")
        else:
            print(f"  run {label}: would NOT have stopped within {len(our_deltas)} prompts "
                  f"(final smoothed Δmean {r['smoothed_delta_final']:.3e})")
            print(f"    Read this against where the threshold falls, not as divergence. "
                  f"The published run stopped at {target.n_prompts} because its smoothed "
                  f"Δmean crossed {stop_rule['stop_at_delta']} there; a trace sitting just "
                  f"above that line at the same prompt count agrees with it to within the "
                  f"per-prompt scatter and still reports 'would not have stopped'.")
    pub_replay = jx_cmp.replay_early_stop(pub_deltas, **stop_rule)
    replay["published_selfcheck"] = pub_replay
    print(f"  self-check, published trace: {pub_replay['would_stop_at']} "
          f"(should be {target.n_prompts}; validates the replay against the run that used it)")
    replay["rule"] = dict(stop_rule)
    replay["rule_source"] = stop_source
    results["recovered_early_stop"] = replay

    # --- A1's required negative control --------------------------------------
    print("\n=== A1 control: can the metric see disagreement? ===")
    print("A control is a deliberate mismatch, fed to the metric to prove it returns a")
    print("large number when it should. Without one, the agreement above could equally")
    print("be one file loaded twice, or a metric swamped by the near-identity diagonal.")
    d_model = next(iter(p_J.values())).shape[-1]
    print(f"A1 nominates a different MODEL's lens as the mismatch. This model's own")
    print(f"published lens is d_model={d_model} and is what axis 1 uses; every other")
    print("published model is a different width, and ||A-B||_F between different shapes")
    print("is undefined. So: same principle, shape-safe — match every one of our layers")
    print("against every published layer, right pair wins.")
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
    stem = "t16_compare" if target.is_default_rung else f"t16_compare_{slug}"
    out_path = out_dir / f"{stem}.json"
    results["published_artifact"] = jx_prov.artifact_facts(pub_lens_path)
    results["fp16_floor"] = jx_cmp.FP16_FLOOR
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    print("\nT17 turns this into a manifest and a verdict. The verdict must state, per axis,")
    print("the floor that binds it and where that floor came from — and that a comparison")
    print("against a single published run carries their noise draw as well as ours.")
    if results["envelope_provenance"]["kind"] != "measured":
        print("On this rung the envelope was not measured from a pair of our own fits, so")
        print("the verdict must additionally name its provenance and carry its uncertainty:")
        print(f"  {results['envelope_provenance']}")


if __name__ == "__main__":
    main()
