"""T17 -- predict a rung's run-to-run envelope from single-prompt draws.

Spec: ``workspace-band-location``, stage 4. Produces the per-layer envelope that
``t16_compare.py --envelope`` needs in order to score axes 1 and 2 on a rung that was
fitted once.

Why this exists rather than a second fit
----------------------------------------

Every claim about a lens needs a null: two lenses differ by *X* -- is *X* noise, or
disagreement? The direct null is to fit twice and compare, which at 4B is a second
~17 h run. Plan decision 4 of the spec bought the cheaper one instead:
``f-2026-08-30-single-prompt-envelope-instrument`` establishes that a lens is a running
mean, so the difference between two fits is ``(1/n) * sum_i delta_i`` and falls as
``n**-alpha``. Measure ``delta`` on a single prompt and divide.

The procedure, and which steps live where
-----------------------------------------

Steps 1-7 of the instrument are exactly what ``offset_profile.py`` already does per
prompt -- measured gate reference, draws in their own processes, gating, clustering by
exact ``identity_distance``, the depth-profile screen, and the within-group pairwise
rms that *is* the single-prompt noise. This driver runs that once **per prompt** and
then does steps 8 and 9: average across prompts, divide by ``n**alpha``.

**One prompt per invocation, not one invocation for all of them.** The analysis is
eager and a 4B draw is ~812 MB, so a single run holding 5 prompts x 8 draws would need
~32 GB resident. Per prompt it is ~6.5 GB. The subprocess boundary is what keeps that
bounded, which is also why this shells out rather than importing -- drivers in
``experiments/`` are scripts, never import targets.

The exponent is used outside its measured range, and that direction matters
------------------------------------------------------------------------

``alpha = 0.473`` is measured over **n=5..233** (``f-2026-08-27-envelope-scaling``).
4B's pin is **417**, outside it. ``f-2026-08-30-alpha-depends-on-the-fit-range`` shows
alpha *falls* as the fit range moves up -- 0.502 over n=1..30, 0.467 over n=8..30 --
so 0.473 extrapolated to 417 is likely **too high**, and dividing by ``n**alpha`` with
too high an alpha **under-predicts** the envelope.

An under-predicted envelope is a floor that is too tight, and a floor that is too tight
makes layers read as *above* floor. The error therefore points at **falsely reporting
disagreement** with the published lens, not at falsely reporting agreement. That is the
safer direction, but it is not a free pass: the output brackets the prediction at
alpha = 0.45 and 0.50 so a verdict can quote a range, and the emitted JSON carries the
caveat into ``t16_compare.py``'s provenance rather than leaving it in a task note.

Run::

    # 5 prompts, 8 draws each, at the production configuration. ~2.5 h at 4B.
    uv run python experiments/t17_envelope.py --model Qwen/Qwen3.5-4B --n-prompts 417

    # re-aggregate from the per-prompt runs already on disk, no GPU
    uv run python experiments/t17_envelope.py --model Qwen/Qwen3.5-4B --n-prompts 417 \
        --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import config as jx_config  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
OFFSET_PROFILE = REPO / "experiments" / "offset_profile.py"

#: The production configuration. The envelope must be measured on the path the fit
#: actually took, not on a convenient one -- `f-2026-08-28-compile-miscompilation`
#: makes compile policy part of the execution path, and a null measured on a different
#: path is not this fit's null.
ARM_POLICY = "linear-attn"
ARM_DIM_BATCH = 8

#: alpha and the range it was measured over. The range is carried, not dropped:
#: `f-2026-08-30-alpha-depends-on-the-fit-range` exists because a bare 0.473 was
#: quoted outside its range once already.
ALPHA = 0.473
ALPHA_RANGE = (5, 233)
ALPHA_BRACKET = (0.45, 0.50)


def per_prompt_path(model_slug: str, prompt: int) -> Path:
    return (cfg.artifact_root / "measurements" / "offset-profile"
            / f"offset_profile_t17-{model_slug}-p{prompt}.json")


def run_prompt(model_id: str, model_slug: str, prompt: int, draws: int) -> Path:
    """One prompt's worth of the instrument, in its own process tree."""
    dest = per_prompt_path(model_slug, prompt)
    cmd = [sys.executable, str(OFFSET_PROFILE),
           "--model", model_id,
           "--arms", f"{ARM_POLICY}:{ARM_DIM_BATCH}:{draws}:{prompt}",
           "--screen",
           "--out-tag", f"t17-{model_slug}-p{prompt}"]
    print(f"\n=== prompt {prompt}: {draws} draws at {ARM_POLICY}:{ARM_DIM_BATCH} ===",
          flush=True)
    print("  " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  prompt {prompt} FAILED (exit {result.returncode}); continuing. "
              f"Steps 8-9 average over whatever prompts succeeded, and the count is "
              f"recorded -- but see the minimum below.")
        return dest if dest.exists() else None
    return dest


def collect(model_slug: str, prompts: list[int]) -> dict[int, dict[int, float]]:
    """Each prompt's per-layer single-prompt noise, from its offset_profile run."""
    noise: dict[int, dict[int, float]] = {}
    for prompt in prompts:
        path = per_prompt_path(model_slug, prompt)
        if not path.exists():
            print(f"  prompt {prompt}: no result at {path}, skipping")
            continue
        blob = json.loads(path.read_text())
        rms = blob.get("noise_rms") or {}
        if not rms:
            print(f"  prompt {prompt}: result carries no noise_rms -- every group was "
                  f"screened out or single-draw. Skipping; see that run's output.")
            continue
        noise[prompt] = {int(k): float(v) for k, v in rms.items()}
        print(f"  prompt {prompt}: {len(noise[prompt])} layers")
    return noise


def average_across_prompts(noise: dict[int, dict[int, float]]) -> dict[int, float]:
    """Step 8. Per layer, the rms across prompts.

    rms rather than the arithmetic mean: these are noise amplitudes, they combine in
    quadrature, and the mean would under-state a stack where one prompt is noisier.
    """
    layers = sorted(set().union(*(set(v) for v in noise.values())))
    out = {}
    for layer in layers:
        vals = [v[layer] for v in noise.values() if layer in v]
        if vals:
            out[layer] = (sum(x * x for x in vals) / len(vals)) ** 0.5
    return out


def predict(single_prompt: dict[int, float], n: int, alpha: float) -> dict[int, float]:
    """Step 9. envelope(n) = single-prompt noise / n**alpha."""
    return {layer: value / (n ** alpha) for layer, value in single_prompt.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-prompts", type=int, required=True,
                        help="the prompt count to predict the envelope AT -- the "
                             "rung's published prompts_fitted (233 at 0.8B, 417 at 4B)")
    parser.add_argument("--prompts", default="0,1,2,3,4",
                        help="corpus indices to draw on (default: 0,1,2,3,4). The "
                             "instrument wants 4-6: identity_distance varies by prompt, "
                             "so one prompt measures one prompt's noise, not the model's")
    parser.add_argument("--draws", type=int, default=8,
                        help="draws per prompt (default 8). Below ~6 the variant "
                             "grouping has too few members to screen")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip the GPU work and re-derive from per-prompt results "
                             "already on disk")
    args = parser.parse_args()

    from fit_lens import _slug

    model_slug = _slug(args.model)
    prompts = [int(x) for x in args.prompts.split(",") if x.strip()]
    if len(prompts) < 4 and not args.aggregate_only:
        print(f"WARNING: {len(prompts)} prompts. The instrument specifies 4-6 because "
              f"single-prompt noise varies by prompt; fewer measures one prompt rather "
              f"than the model. Proceeding, and recording the count.")

    if not args.aggregate_only:
        per_draw_min = {"Qwen/Qwen3.5-4B": 207}.get(args.model, 47) / 60
        est_h = len(prompts) * (args.draws + 1) * per_draw_min / 60
        print(f"machine={cfg.machine}  model={args.model}")
        print(f"{len(prompts)} prompts x ({args.draws} draws + 1 uncompiled gate "
              f"reference), each in its own process.")
        print(f"Estimated ~{est_h:.1f} h. One offset_profile run per prompt, so the "
              f"analysis never holds more than one prompt's draws at once.")
        for prompt in prompts:
            run_prompt(args.model, model_slug, prompt, args.draws)

    print("\n=== steps 8-9: average across prompts, then divide by n**alpha ===")
    noise = collect(model_slug, prompts)
    if not noise:
        raise SystemExit(
            "no per-prompt noise to aggregate. Every prompt either failed or had all "
            "its groups screened out -- there is no envelope to emit, and emitting a "
            "default here would be the whole hazard this measurement exists to avoid."
        )
    if len(noise) < 4:
        print(f"  NOTE: aggregating over {len(noise)} prompts, fewer than the 4-6 the "
              f"instrument specifies. Recorded in the output; treat the result as "
              f"provisional.")

    single = average_across_prompts(noise)
    envelope = predict(single, args.n_prompts, ALPHA)
    bracket = {str(a): predict(single, args.n_prompts, a) for a in ALPHA_BRACKET}

    lo_a, hi_a = ALPHA_BRACKET
    hdr = (f"{'layer':>6} {'single-prompt':>15} {'envelope':>13} "
           f"{'a=' + str(lo_a):>13} {'a=' + str(hi_a):>13}")
    print(hdr)
    print("-" * len(hdr))
    for layer in sorted(envelope):
        print(f"{layer:>6} {single[layer]:>15.4e} {envelope[layer]:>13.4e} "
              f"{bracket[str(lo_a)][layer]:>13.4e} {bracket[str(hi_a)][layer]:>13.4e}")

    in_range = ALPHA_RANGE[0] <= args.n_prompts <= ALPHA_RANGE[1]
    caveat = (
        f"alpha={ALPHA} is measured over n={ALPHA_RANGE[0]}..{ALPHA_RANGE[1]} and is "
        f"applied here at n={args.n_prompts}, OUTSIDE that range. "
        f"f-2026-08-30-alpha-depends-on-the-fit-range shows alpha falls as the range "
        f"rises, so this alpha is likely too high and the envelope is likely "
        f"UNDER-predicted -- a floor too tight, erring toward falsely reporting "
        f"disagreement rather than agreement. Quote the bracket, not the point value."
    ) if not in_range else (
        f"alpha={ALPHA} applied at n={args.n_prompts}, inside its measured range "
        f"n={ALPHA_RANGE[0]}..{ALPHA_RANGE[1]}."
    )
    print(f"\n{caveat}")

    out_dir = cfg.artifact_root / "measurements" / "t17"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"envelope_{model_slug}.json"
    out_path.write_text(json.dumps({
        "source": "single-prompt envelope instrument (T17)",
        "model": args.model,
        "machine": cfg.machine,
        "n": args.n_prompts,
        "alpha": ALPHA,
        "alpha_measured_over": list(ALPHA_RANGE),
        "alpha_in_range": in_range,
        "alpha_bracket": ALPHA_BRACKET,
        "caveat": caveat,
        "prompts_requested": prompts,
        "prompts_used": sorted(noise),
        "draws_per_prompt": args.draws,
        "arm": {"policy": ARM_POLICY, "dim_batch": ARM_DIM_BATCH},
        "single_prompt_noise": single,
        "per_layer": envelope,
        "per_layer_bracket": bracket,
    }, indent=2) + "\n")
    print(f"wrote {out_path}")
    print("\nFeed it to the comparison:")
    print(f"  uv run python experiments/t16_compare.py --model {args.model} \\")
    print(f"      --runs a --envelope {out_path}")


if __name__ == "__main__":
    main()
