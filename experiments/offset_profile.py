"""Per-layer offsets between compile variants, compile configurations, and `dim_batch`.

Three questions that turned out to be one experiment. Each asks how far apart two
*groups* of Jacobians sit once the noise both groups carry has been subtracted, so
each needs the same thing: several gated single-prompt draws per arm, tensors kept,
and a within-group/between-group comparison per layer.

1. **Between compile variants.** A compiled process draws one of two numerically
   distinct sound compilations (`f-2026-08-28-compile-miscompilation`). Two fits that
   drew different ones differ by a *systematic* term, which does not average down and
   which the run-to-run envelope does not describe. A1 of
   `d-commons-reproduction-criteria` now requires "two fits **that drew the same
   compile variant**" for exactly this reason, but the offset's per-layer profile is
   written in as unmeasured. It matters most at **L0**, where the envelope binds
   tightest (4.787e-4) -- an offset comparable to that makes the qualifier
   load-bearing; one well below it relaxes the qualifier to a note and simplifies the
   27B instruction.

2. **Between compile configurations.** We do not know how Neuronpedia compiled their
   fits, so A1 carries the transfer of our envelope to the published artifacts as a
   *stated, unbounded* assumption. Bounding it needs a Frobenius-side number -- the
   per-layer tensor difference between all-blocks, linear-attention-only and
   uncompiled. The `identity_distance` separation is **not** that number: it is a
   scalar at one layer, and substituting it for a bound on a Frobenius aggregate is
   the cross-quantity error this project has now made four times.

3. **Is `dim_batch` numerically neutral?** Still unperformed. `dim_batch_neutrality.py`
   ran once, at `dim_batch=64` compiled all-blocks, and drew a miscompilation -- so its
   1.0-4.1 relative differences were one bad lens against two good ones and were never
   a `dim_batch` result. Re-run here, gated, at the safe configuration.

Why one prompt is enough
------------------------

All three quantities are **systematic**: a variant, a configuration and a slicing
width are fixed properties of a run, not draws from a distribution. They do not
average down with prompt count, so they are as visible at n=1 as at n=233. The noise
is not -- it is ~13x its 233-prompt amplitude at one prompt (alpha = 0.473,
`f-2026-08-27-envelope-scaling`) -- which is why the estimator subtracts a measured
within-group null rather than reading a cross-group difference directly. See
``jlens_extensions.compare.group_offset``.

The honest limit: if an offset sits well under the single-prompt noise, this returns
an upper bound rather than a value. That is still an answer for A1, and it is the
cheap answer. Resolving further means more draws, not more prompts.

Design notes
------------

* **One configuration per process.** The miscompilation is per-process, so two arms
  sharing a process share a draw. ``jlens_extensions.childproc`` holds that machinery.
* **Miscompiled draws are dropped, not averaged.** Each draw is gated on prompt-1
  ``identity_distance`` before it enters any group.
* **Variants are discovered, not tabulated** --
  ``jlens_extensions.compile_policy.cluster_variants`` splits sound readings on a gap,
  so this driver does not carry a per-model table that would go stale.
* The `all-blocks` arm loses ~37% of its draws to miscompilation
  (`f-2026-08-28-compile-miscompilation`, pooled), so ask for more of them than the
  others if that arm's variant split matters to you.

Run::

    uv run python experiments/offset_profile.py                 # all four arms
    uv run python experiments/offset_profile.py --draws 10
    uv run python experiments/offset_profile.py --arms linear-attn:8,linear-attn:64
"""

from __future__ import annotations

import argparse
import itertools
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

from jlens_extensions.childproc import emit_result, run_child  # noqa: E402
from jlens_extensions.compare import group_offset, rel_frobenius  # noqa: E402
from jlens_extensions.compile_policy import cluster_variants, identity_in_band  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LEN = 128
D_MODEL = 1024
#: Prompt 0's sound identity_distance on this model. Used only as a fallback and
#: as a sanity check -- the gate reference is MEASURED per prompt, see
#: reference_identity(). Different prompts have different Jacobians and therefore
#: different identity_distance: prompt 1 sits at ~0.5644, 6.2% from this value, so
#: gating prompt 1 against this constant rejects every sound draw including the
#: uncompiled ones. That is exactly what the first run of the 18-arm grid did.
EXPECTED_IDENTITY = 0.5314
IDENTITY_TOL = 0.01

#: (compile-layer policy, dim_batch). `none` is uncompiled and single-variant, so it
#: anchors the configuration comparison; `linear-attn` is production and carries both
#: the variant question and the dim_batch one.
DEFAULT_ARMS = (("all", 8), ("linear-attn", 8), ("none", 8), ("linear-attn", 64))


def parse_arms(spec: str | None,
               default_draws: int) -> list[tuple[str, int, int, int]]:
    """``policy:dim_batch[:draws[:prompt]]``. Returns (policy, dim_batch, draws, prompt).

    Draws are per-arm because the arms do not need the same number, and giving them
    the same number spends the budget in the wrong places. Two effects set the
    requirement:

    * **Miscompilation losses.** An `all`-blocks arm loses ~37% of its draws to the
      gate (`f-2026-08-28-compile-miscompilation`, pooled over three soaks), so 8
      draws leave ~5 sound and ~2.5 per variant. `none` loses none.
    * **How lopsided the variant split is.** At a balanced arm 8 draws miss a variant
      0.4% of the time; at a skewed one -- `linear-attn:64` drew 7:1 -- the same 8
      draws miss it 34% of the time, and a variant seen once contributes no averaging
      at all to its own offset estimate.

    `none` is uncompiled and therefore single-variant, so it needs draws only for its
    noise null and can be the cheapest row despite being the slowest per draw.

    **The prompt field is the fourth axis and the cheapest insurance here.** Every
    offset this driver measures is measured *on one input*, and scaling an n=1 number
    to n=233 assumes the per-prompt magnitude is representative rather than a lucky
    draw. Repeating a subset of arms at a second prompt tests that directly. Jacobians
    for different prompts are different quantities, so the prompt is part of the arm
    and no comparison ever crosses it.
    """
    if not spec:
        return [(p, d, default_draws, 0) for p, d in DEFAULT_ARMS]
    arms = []
    for item in spec.split(","):
        parts = [x for x in item.strip().split(":") if x != ""]
        if not parts:
            continue
        policy = parts[0]
        if policy not in ("all", "linear-attn", "full-attn", "none"):
            raise SystemExit(f"unknown compile policy {policy!r} in --arms")
        db = int(parts[1]) if len(parts) > 1 else 8
        draws = int(parts[2]) if len(parts) > 2 else default_draws
        prompt = int(parts[3]) if len(parts) > 3 else 0
        arms.append((policy, db, draws, prompt))
    return arms


# --------------------------------------------------------------------------- child


def child_draw(dim_batch: int, which: str, out_path: Path,
               prompt_idx: int = 0) -> dict:
    """One prompt's Jacobian at one configuration, in this process, saved to disk."""
    import torch
    import transformers

    import jlens
    from jlens.fitting import jacobian_for_prompt

    from fit_lens import load_prompts

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    compiled = which != "none"
    model = jlens.from_hf(hf, tok, compile=(which == "all"))
    if compiled and which != "all":
        wanted_linear = which == "linear-attn"
        for i, block in enumerate(model.layers):
            if hasattr(block, "linear_attn") == wanted_linear:
                model.layers[i] = torch.compile(block, mode="default", dynamic=False)

    # The corpus is a deterministic prefix, so asking for idx+1 and taking the last
    # selects prompt idx reproducibly -- the same prompt every process, every run.
    prompt = load_prompts(dataset="Salesforce/wikitext", config="wikitext-103-raw-v1",
                          split="train", text_field="text", n_prompts=prompt_idx + 1,
                          max_chars=2000)[-1]
    source_layers = list(range(model.n_layers - 1))
    J, seq_len, n_valid = jacobian_for_prompt(
        model, prompt, source_layers, target_layer=None,
        dim_batch=dim_batch, max_seq_len=MAX_SEQ_LEN,
    )
    late = max(source_layers)
    ident = (J[late].float() - torch.eye(D_MODEL)).norm().item() / D_MODEL**0.5
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({l: J[l].float() for l in source_layers}, str(out_path))
    return {"dim_batch": dim_batch, "compile_layers": which, "compiled": compiled,
            "prompt_idx": prompt_idx, "identity_distance": ident, "seq_len": seq_len,
            "n_valid": n_valid, "path": str(out_path)}


def reference_identity(prompt_idx: int, scratch: Path) -> float:
    """One uncompiled draw, to establish what a sound run looks like on THIS prompt.

    `f-2026-08-28-compile-miscompilation` already prescribes this for a model with no
    reference value: compute prompt 1 uncompiled first and gate the compiled runs
    against it. The prompt is the same kind of axis -- a different prompt is a
    different Jacobian and a different identity_distance -- so the reference is
    measured per prompt rather than carried from prompt 0.

    Uncompiled cannot miscompile, so this draw needs no gate of its own.
    """
    dest = scratch / f"gate-ref-p{prompt_idx}.pt"
    r = run_child(__file__, f"gate-reference-p{prompt_idx}",
                  ["--child", "draw", "--dim-batch", "8", "--compile-layers", "none",
                   "--prompt-index", str(prompt_idx), "--out", str(dest)])
    dest.unlink(missing_ok=True)
    if not r:
        raise SystemExit(f"could not establish a gate reference for prompt {prompt_idx}")
    return r["identity_distance"]


# ------------------------------------------------------------------------- analysis


def load_tensors(draws: list[dict]) -> None:
    """Load every draw's Jacobians into memory once, keyed onto the draw.

    Deliberately eager. The comparisons are pairwise over every draw in an arm and
    then across arms, so a load-per-pair would re-read each 96 MB file dozens of
    times -- around 95 GB of reads at the default arm count, for 3 GB of distinct
    data. Peak memory is ``n_draws x 96 MB``: ~3 GB for the 32-draw default, which
    is nothing against this box's 121 GB but is worth knowing before someone passes
    ``--draws 40``.
    """
    import torch

    for d in draws:
        d["tensors"] = torch.load(d["path"], map_location="cpu")


def group_of(d: dict) -> tuple:
    """The unit of comparison: one execution path, on one input.

    Policy, ``dim_batch``, prompt and compile variant together. Two draws in the
    same group are two runs of the *same* computation and differ only by run-to-run
    noise; two draws in different groups differ by whatever separates the paths.
    """
    return (d["compile_layers"], d["dim_batch"], d["prompt_idx"], d["variant"])


def group_label(g: tuple) -> str:
    return f"{g[0]}:{g[1]}@p{g[2]}/v{g[3]}"


def classify_pair(a: tuple, b: tuple) -> str:
    """What separates two groups. Only one thing differing is the interesting case."""
    if a[2] != b[2]:
        return "cross-prompt"          # never compared; different quantities
    diffs = [a[0] != b[0], a[1] != b[1], a[3] != b[3]]
    if sum(diffs) > 1:
        return "multiple"
    if diffs[0]:
        return "compile config"
    if diffs[1]:
        return "dim_batch"
    return "variant"


def all_pair_rels(draws: list[dict]) -> dict[tuple, dict[int, list[float]]]:
    """Every pair's per-layer relative difference, bucketed by the two groups.

    One pass. Previously each comparison recomputed its own pairs, so a pair could
    be evaluated several times; here every pair is computed once and looked up.
    """
    out: dict[tuple, dict[int, list[float]]] = {}
    for a, b in itertools.combinations(draws, 2):
        ga, gb = group_of(a), group_of(b)
        if ga[2] != gb[2]:             # never compare across prompts
            continue
        key = tuple(sorted((ga, gb)))
        A, B = a["tensors"], b["tensors"]
        bucket = out.setdefault(key, {})
        for layer in sorted(A):
            bucket.setdefault(layer, []).append(rel_frobenius(A[layer], B[layer]))
    return out


def rms(vals) -> float:
    return (sum(v * v for v in vals) / len(vals)) ** 0.5 if vals else float("nan")


SUSPECT_FALLOFF = 100.0


def screen_groups(draws: list[dict], min_falloff: float = SUSPECT_FALLOFF,
                  top_layer: int | None = None) -> list[dict]:
    """Drop groups whose within-group spread cannot be run-to-run noise.

    Genuine bf16 noise falls ~4 orders of magnitude from L0 to the top of the stack,
    and the 121-draw grid measured that falloff at 3646-8691x across sixteen groups
    spanning four compile configurations and four ``dim_batch`` values, with their
    L0 floors agreeing to 17%. A group far off that profile is not a noisier version
    of the same thing; it holds draws that are not the same computation.

    Also drops single-draw groups. They contribute no within-group pairs, so nothing
    can be said about whether they are sound, and in this grid every one of them sat
    in a row that had contamination elsewhere.

    This is a *screen*, not a gate: it runs on the assembled draws rather than on one
    process, and it catches what the prompt-1 identity gate structurally cannot --
    a fault that is large at early layers and invisible at L22, where
    ``identity_distance`` is computed.
    """
    by_group: dict[tuple, list[dict]] = {}
    for d in draws:
        by_group.setdefault(group_of(d), []).append(d)
    keep, notes = [], []
    for g, members in sorted(by_group.items()):
        if len(members) < 2:
            notes.append(f"  dropped {group_label(g)}: single draw, unassessable")
            continue
        rels = {}
        for a, b in itertools.combinations(members, 2):
            for l in a["tensors"]:
                rels.setdefault(l, []).append(rel_frobenius(a["tensors"][l],
                                                            b["tensors"][l]))
        top = top_layer if top_layer is not None else max(rels)
        lo, hi = rms(rels.get(0, [])), rms(rels.get(top, []))
        fall = lo / hi if hi > 0 else float("inf")
        if fall < min_falloff:
            notes.append(f"  dropped {group_label(g)}: n={len(members)}, "
                         f"falloff {fall:.0f}x, L0 {lo:.3e} -- not noise")
            continue
        keep.extend(members)
    if notes:
        print(f"\n=== screen: {len(draws) - len(keep)} of {len(draws)} draws dropped ===")
        for n in notes:
            print(n)
    return keep


def analyse(draws: list[dict]) -> dict:
    """Pairwise offsets between every pair of groups, against a pooled noise null.

    **Why a matrix and not a decomposition.** The first version of this modelled a
    cross-arm difference as ``dim_batch offset`` plus ``variant offset`` plus noise,
    added in quadrature, and subtracted the variant term measured in the same run.
    The data says that model is wrong: at L0 two draws from *different* ``dim_batch``
    values sit 1.554e-2 apart while two variants at ``dim_batch=64`` sit 1.845e-2
    apart, so the "components" are not separable and the subtraction went negative
    and clamped to zero across half the stack.

    What the numbers look like instead is a cloud: every distinct execution path
    lands somewhere, and the paths are all roughly the same distance from each other.
    So the honest report is the distance between each pair of paths, with the
    run-to-run noise removed, and a breakdown by what separates the pair -- which is
    what says whether a ``dim_batch`` change costs more, less, or the same as drawing
    a different compile variant.
    """
    pairs = all_pair_rels(draws)
    groups = sorted({g for k in pairs for g in k})
    layers = sorted({l for b in pairs.values() for l in b})

    # Pooled noise null: pairs whose two draws are in the same group.
    null = {l: [] for l in layers}
    for (ga, gb), bucket in pairs.items():
        if ga == gb:
            for l, vals in bucket.items():
                null[l].extend(vals)

    per_group = {}
    for (ga, gb), bucket in pairs.items():
        if ga == gb and bucket:
            per_group[group_label(ga)] = {l: rms(v) for l, v in bucket.items()}

    matrix = {}
    for (ga, gb), bucket in pairs.items():
        if ga == gb:
            continue
        kind = classify_pair(ga, gb)
        rows = {l: group_offset(null.get(l, []), bucket.get(l, [])) for l in layers}
        matrix[f"{group_label(ga)} vs {group_label(gb)}"] = {
            "kind": kind, "a": list(ga), "b": list(gb),
            "n_pairs": len(next(iter(bucket.values()), [])), "layers": rows,
        }
    return {"noise_rms": {l: rms(null[l]) for l in layers},
            "noise_per_group": per_group,
            "groups": [group_label(g) for g in groups], "pairs": matrix}


def assign_variants(draws: list[dict]) -> None:
    """Cluster each arm's sound readings into compile variants, in place.

    Variants are per (configuration, dim_batch, prompt): the sound values differ
    between arms, so clustering across arms would split on the arm instead.
    """
    for d in draws:
        d["arm"] = (d["compile_layers"], d["dim_batch"], d["prompt_idx"])
    for arm in sorted({d["arm"] for d in draws}):
        members = [d for d in draws if d["arm"] == arm]
        clusters = cluster_variants([d["identity_distance"] for d in members])
        for vi, cluster in enumerate(clusters):
            for i in cluster:
                members[i]["variant"] = vi
        vals = ", ".join(f"v{i}: {members[c[0]]['identity_distance']:.6f} x{len(c)}"
                         for i, c in enumerate(clusters))
        print(f"  {arm[0]}:{arm[1]}@p{arm[2]} -> {len(clusters)} variant(s)  [{vals}]")


def reanalyse(manifest: Path, screen: bool = False) -> None:
    """Redo the analysis from a finished run's saved Jacobians.

    The reason tensors are kept. The analysis has been rebuilt twice now while the
    draws underneath it stayed valid, and each rebuild would otherwise have cost a
    fresh grid -- hours -- instead of the couple of minutes this takes.
    """
    payload = json.loads(manifest.read_text())
    draws = [d for d in payload["draws"] if Path(d["path"]).exists()]
    missing = len(payload["draws"]) - len(draws)
    print(f"re-analysing {manifest}")
    print(f"{len(draws)} draws with tensors on disk"
          + (f", {missing} whose tensors are gone" if missing else ""))
    if not draws:
        raise SystemExit("no saved tensors to re-analyse; the run must keep them")
    load_tensors(draws)
    assign_variants(draws)
    if screen:
        draws = screen_groups(draws)
        assign_variants(draws)
    results = analyse(draws)
    report(results)
    dest = manifest.with_name(manifest.stem + "_reanalysed.json")
    dest.write_text(json.dumps(
        {"task": "offset-profile-reanalysis", "source": str(manifest),
         "n_draws": len(draws), "groups": results["groups"],
         "noise_rms": results["noise_rms"],
         "noise_per_group": results["noise_per_group"],
         "results": results["pairs"]}, indent=2, default=str) + "\n")
    print(f"\nwrote {dest}")


def report(result: dict, layers_shown=(0, 4, 8, 15, 22)) -> None:
    noise = result["noise_rms"]
    print("\n=== run-to-run noise (same path, different process), per layer ===")
    print("   " + "  ".join(f"L{l}: {noise[l]:.3e}" for l in layers_shown if l in noise))

    # Per group, sorted by the worst top-of-stack value. Genuine bf16 noise falls
    # ~4 orders of magnitude from L0 to L22; a group whose L22 sits near its L0 is
    # not noisy, it holds draws that are not the same computation. Pooling hides
    # that -- the 18-arm grid pooled a contaminated null into a figure 1300x too
    # large at L22 with nothing on screen saying so.
    pg = result.get("noise_per_group") or {}
    if pg:
        top = max(l for l in layers_shown if any(l in v for v in pg.values()))
        print("\n=== that noise per group -- a flat depth profile means contamination ===")
        hdr = f"{'group':>26} {'L0':>11} {'L' + str(top):>11} {'falloff':>10}  flag"
        print(hdr); print("-" * len(hdr))
        for label, prof in sorted(pg.items(), key=lambda kv: -(kv[1].get(top) or 0)):
            lo, hi = prof.get(0), prof.get(top)
            if lo is None or hi is None or hi <= 0:
                continue
            fall = lo / hi
            print(f"{label:>26} {lo:>11.3e} {hi:>11.3e} {fall:>9.0f}x"
                  + ("" if fall > 100 else "  <-- SUSPECT"))

    kinds = ["variant", "dim_batch", "compile config", "multiple"]
    print("\n=== offsets between execution paths, grouped by what differs ===")
    hdr = f"{'what differs':>15} {'n pairs':>8} " + " ".join(f"{'L'+str(l):>11}" for l in layers_shown)
    print(hdr); print("-" * len(hdr))
    for kind in kinds:
        rows = [v for v in result["pairs"].values() if v["kind"] == kind]
        if not rows:
            continue
        cells = []
        for l in layers_shown:
            offs = sorted(r["layers"][l]["offset"] for r in rows
                          if r["layers"].get(l, {}).get("offset") is not None)
            cells.append(f"{offs[len(offs)//2]:>11.3e}" if offs else f"{'-':>11}")
        print(f"{kind:>15} {len(rows):>8} " + " ".join(cells))
    print("\n  (median over pairs; the full matrix is in the JSON)")

    print("\n=== spread across ALL path pairs, per layer ===")
    hdr = f"{'layer':>6} {'min':>12} {'median':>12} {'max':>12} {'max/min':>9}"
    print(hdr); print("-" * len(hdr))
    for l in layers_shown:
        offs = sorted(v["layers"][l]["offset"] for v in result["pairs"].values()
                      if v["layers"].get(l, {}).get("offset") is not None)
        if not offs:
            continue
        lo, hi = offs[0], offs[-1]
        med = offs[len(offs) // 2]
        ratio = f"{hi/lo:>9.1f}" if lo > 0 else f"{'inf':>9}"
        print(f"{l:>6} {lo:>12.3e} {med:>12.3e} {hi:>12.3e} {ratio}")
    print("\n  A narrow spread here is the claim that every valid execution path")
    print("  differs from every other by about the same amount. A wide one is not.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=8,
                        help="default draws per arm, when --arms does not give a "
                             "per-arm count (default 8)")
    parser.add_argument("--arms", default=None,
                        help="comma-separated policy:dim_batch[:draws], e.g. "
                             "'all:8:12,none:8:6' (default: all four arms). The "
                             "per-arm count matters: see parse_arms")
    parser.add_argument("--discard-tensors", action="store_true",
                        help="delete the saved Jacobians when done. NOT the default: "
                             "the analysis has already been rebuilt once, and "
                             "re-analysing costs minutes where re-running the grid "
                             "costs hours")
    parser.add_argument("--screen", action="store_true",
                        help="drop groups whose within-group spread has the wrong "
                             "depth profile to be run-to-run noise, and single-draw "
                             "groups; see screen_groups")
    parser.add_argument("--analyse", metavar="MANIFEST",
                        help="skip all draws and redo the analysis from a previous "
                             "run's saved Jacobians (its offset_profile.json)")
    parser.add_argument("--child")
    parser.add_argument("--dim-batch", type=int)
    parser.add_argument("--compile-layers")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.analyse:
        reanalyse(Path(args.analyse), screen=args.screen)
        return

    if args.child:
        emit_result(child_draw(args.dim_batch, args.compile_layers, Path(args.out),
                               args.prompt_index))
        return

    arms = parse_arms(args.arms, args.draws)
    scratch = cfg.scratch_root / "offset-profile"
    out = cfg.artifact_root / "measurements" / "offset-profile"
    out.mkdir(parents=True, exist_ok=True)
    #: Per-draw cost, seconds, by policy -- compiled arms run at roughly half the
    #: uncompiled rate, and the estimate is what tells an unattended run whether it
    #: fits the budget before it starts rather than after.
    COST_S = {"all": 47, "linear-attn": 47, "full-attn": 60, "none": 72}
    total_draws = sum(n for _, _, n, _ in arms)
    est_min = sum(n * COST_S.get(p, 55) for p, _, n, _ in arms) / 60
    prompts = sorted({q for _, _, _, q in arms})
    print(f"machine={cfg.machine}  model={MODEL_ID}")
    print(f"{len(arms)} arms, {total_draws} draws, each in its own process.")
    print(f"Prompts: {prompts}"
          + ("  (comparisons never cross a prompt)" if len(prompts) > 1 else ""))
    print(f"Arms: {', '.join(f'{p}:{d}x{n}@p{q}' for p, d, n, q in arms)}")
    print(f"Estimated {est_min:.0f} minutes of draws, then a few minutes of analysis.")
    print(f"Analysis holds every draw in memory: ~{total_draws * 96 / 1024:.1f} GB.\n")

    # One uncompiled probe per prompt, before any gated draw on that prompt.
    references: dict[int, float] = {}
    for q in prompts:
        references[q] = reference_identity(q, scratch)
        off = abs(references[q] - EXPECTED_IDENTITY) / EXPECTED_IDENTITY
        note = "" if q == 0 else f"   ({off:.1%} from prompt 0 -- as expected, different prompt)"
        print(f"  gate reference, prompt {q}: {references[q]:.6f}{note}", flush=True)

    draws: list[dict] = []
    rejected: list[dict] = []
    dropped = 0
    # Written after every draw. An unattended run that dies partway then leaves both
    # the manifest and the tensors on disk, so the completed arms stay analysable
    # instead of costing the whole sitting.
    progress = out / "offset_profile_progress.json"
    for policy, db, n_draws, prompt_idx in arms:
        for rep in range(n_draws):
            tag = f"{policy}-db{db}-p{prompt_idx}-r{rep}"
            dest = scratch / f"{tag}.pt"
            r = run_child(__file__, tag,
                          ["--child", "draw", "--dim-batch", str(db),
                           "--compile-layers", policy, "--prompt-index", str(prompt_idx),
                           "--out", str(dest)], quiet=True)
            if not r:
                continue
            # Gate before the draw enters any group. A miscompiled Jacobian is not a
            # variant and not a datum -- averaging one in would contaminate every
            # comparison it appears in.
            if not identity_in_band(r["identity_distance"], references[prompt_idx],
                                    IDENTITY_TOL):
                print(f"     {tag} MISCOMPILED (identity={r['identity_distance']:.6f})"
                      f" -- dropped", flush=True)
                # Keep the tensor. A "drop" is a gate verdict, and the gate has
                # already been wrong once: the first 18-arm grid gated prompt 1
                # against prompt 0's reference and discarded 50 sound draws,
                # uncompiled ones included, deleting the evidence as it went.
                r["dropped"] = True
                rejected.append(r)
                dropped += 1
                continue
            draws.append(r)
            progress.write_text(json.dumps(
                {"complete": False, "n_sound": len(draws), "n_dropped": dropped,
                 "arms": [list(a) for a in arms], "draws": draws,
                 "rejected": rejected}, indent=2,
                default=str) + "\n")

    print(f"\n{len(draws)} sound draws, {dropped} dropped as miscompiled "
          f"(their tensors are kept too -- a drop is a gate verdict, not a fact).")
    load_tensors(draws)
    assign_variants(draws)
    if args.screen:
        draws = screen_groups(draws)
        assign_variants(draws)
    results = analyse(draws)
    report(results)

    dest = out / "offset_profile.json"
    dest.write_text(json.dumps(
        {"task": "offset-profile", "machine": cfg.machine, "model": MODEL_ID,
         "arms": [list(a) for a in arms],
         "n_sound": len(draws), "n_dropped": dropped,
         "rejected": [{k: v for k, v in d.items() if k != "tensors"}
                      for d in rejected],
         "groups": results["groups"], "noise_rms": results["noise_rms"],
         "draws": [{k: v for k, v in d.items() if k not in ("arm", "tensors")}
                   for d in draws],
         "results": results["pairs"]}, indent=2, default=str) + "\n")
    print(f"\nwrote {dest}")

    if args.discard_tensors:
        for d in draws:
            Path(d["path"]).unlink(missing_ok=True)
        print("removed the saved Jacobians")
    else:
        held = sum(1 for d in draws if Path(d["path"]).exists())
        print(f"kept {held} Jacobians under {scratch} "
              f"(~{held * 96 / 1024:.1f} GB) so the analysis can be redone without "
              f"re-running the grid; --discard-tensors to drop them")


if __name__ == "__main__":
    main()
