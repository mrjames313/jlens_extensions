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
EXPECTED_IDENTITY = 0.5314
IDENTITY_TOL = 0.01

#: (compile-layer policy, dim_batch). `none` is uncompiled and single-variant, so it
#: anchors the configuration comparison; `linear-attn` is production and carries both
#: the variant question and the dim_batch one.
DEFAULT_ARMS = (("all", 8), ("linear-attn", 8), ("none", 8), ("linear-attn", 64))


def parse_arms(spec: str | None, default_draws: int) -> list[tuple[str, int, int]]:
    """``policy:dim_batch[:draws]``, comma-separated. Returns (policy, dim_batch, draws).

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
    """
    if not spec:
        return [(p, d, default_draws) for p, d in DEFAULT_ARMS]
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
        arms.append((policy, db, draws))
    return arms


# --------------------------------------------------------------------------- child


def child_draw(dim_batch: int, which: str, out_path: Path) -> dict:
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

    prompt = load_prompts(dataset="Salesforce/wikitext", config="wikitext-103-raw-v1",
                          split="train", text_field="text", n_prompts=1,
                          max_chars=2000)[0]
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
            "identity_distance": ident, "seq_len": seq_len, "n_valid": n_valid,
            "path": str(out_path)}


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


def pair_rels(draws: list[dict], same: bool, key) -> dict[int, list[float]]:
    """Per-layer relative differences over pairs that agree (or differ) on ``key``."""
    out: dict[int, list[float]] = {}
    for a, b in itertools.combinations(draws, 2):
        if (key(a) == key(b)) != same:
            continue
        A, B = a["tensors"], b["tensors"]
        for layer in sorted(A):
            out.setdefault(layer, []).append(rel_frobenius(A[layer], B[layer]))
    return out


def noise_pairs(draws: list[dict]) -> dict[int, list[float]]:
    """The pure-noise null: pairs matching on arm *and* variant.

    Everything else has to be measured against this rather than against its own
    same-group pairs. Grouping by configuration or by ``dim_batch`` alone leaves
    *different variants inside the same group*, so the "within" null then carries a
    variant term and the subtraction removes signal along with noise. The first run
    of this driver did exactly that: its configuration null at L22 read 4.685e-4
    against a true noise floor of 7.4e-7, a factor of 600, and every row came back
    "resolved" partly because the comparison was rigged against itself.
    """
    return pair_rels(draws, True, lambda d: (d["arm"], d["variant"]))


def decompose(draws: list[dict], key, label: str, *,
              null: dict[int, list[float]] | None = None,
              subtract: dict[int, float] | None = None) -> dict:
    """Between-group difference against a null, per layer.

    ``null`` supplies the pure-noise pairs (see :func:`noise_pairs`); without it the
    same-group pairs are used, which is only correct when the grouping key already
    pins every other source of difference.

    ``subtract`` removes a further per-layer term in quadrature -- used for the
    cross-arm comparisons, where a pair differs by ``dim_batch`` *and* by whichever
    variants the two processes drew. Subtracting the variant term measured in the
    same run leaves the quantity actually being asked about.
    """
    within = null if null is not None else pair_rels(draws, True, key)
    between = pair_rels(draws, False, key)
    layers = sorted(set(within) | set(between))
    rows = {}
    for l in layers:
        r = group_offset(within.get(l, []), between.get(l, []))
        if subtract and r.get("offset"):
            extra = subtract.get(l, 0.0)
            resid = r["offset"] ** 2 - extra * extra
            r["offset_before_subtraction"] = r["offset"]
            r["subtracted"] = extra
            r["offset"] = resid ** 0.5 if resid > 0 else 0.0
            r["resolved"] = r["offset"] > r["bound"]
        rows[l] = r
    return {"comparison": label, "layers": rows}


def report(name: str, result: dict) -> None:
    rows = result["layers"]
    if not rows or all(r.get("offset") is None for r in rows.values()):
        print(f"\n{name}: not enough draws on both sides -- skipped")
        return
    print(f"\n=== {name} ===")
    hdr = f"{'layer':>6} {'within (noise)':>15} {'between':>12} {'offset':>12}  verdict"
    print(hdr); print("-" * len(hdr))
    for layer in sorted(rows):
        r = rows[layer]
        if r.get("offset") is None:
            continue
        if r["resolved"]:
            verdict = "resolved"
            off = f"{r['offset']:.3e}"
        else:
            # Not "no offset" -- an offset the noise floor cannot separate. The
            # bound is a real exclusion; see compare.group_offset.
            verdict = f"unresolved, offset < {r['bound']:.3e}"
            off = f"({r['offset']:.3e})"
        print(f"{layer:>6} {r['within_rms']:>15.3e} {r['between_rms']:>12.3e} "
              f"{off:>12}  {verdict}")


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
    parser.add_argument("--keep", action="store_true",
                        help="keep the saved Jacobians instead of deleting them")
    parser.add_argument("--child")
    parser.add_argument("--dim-batch", type=int)
    parser.add_argument("--compile-layers")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.child:
        emit_result(child_draw(args.dim_batch, args.compile_layers, Path(args.out)))
        return

    arms = parse_arms(args.arms, args.draws)
    scratch = cfg.scratch_root / "offset-profile"
    out = cfg.artifact_root / "measurements" / "offset-profile"
    out.mkdir(parents=True, exist_ok=True)
    #: Per-draw cost, seconds, by policy -- compiled arms run at roughly half the
    #: uncompiled rate, and the estimate is what tells an unattended run whether it
    #: fits the budget before it starts rather than after.
    COST_S = {"all": 47, "linear-attn": 47, "full-attn": 60, "none": 72}
    total_draws = sum(n for _, _, n in arms)
    est_min = sum(n * COST_S.get(p, 55) for p, _, n in arms) / 60
    print(f"machine={cfg.machine}  model={MODEL_ID}")
    print(f"{len(arms)} arms, {total_draws} draws, each in its own process.")
    print(f"Arms: {', '.join(f'{p}:{d}x{n}' for p, d, n in arms)}")
    print(f"Estimated {est_min:.0f} minutes of draws, then a few minutes of analysis.")
    print(f"Analysis holds every draw in memory: ~{total_draws * 96 / 1024:.1f} GB.\n")

    draws: list[dict] = []
    dropped = 0
    # Written after every draw. An unattended run that dies partway then leaves both
    # the manifest and the tensors on disk, so the completed arms stay analysable
    # instead of costing the whole sitting.
    progress = out / "offset_profile_progress.json"
    for policy, db, n_draws in arms:
        for rep in range(n_draws):
            tag = f"{policy}-db{db}-r{rep}"
            dest = scratch / f"{tag}.pt"
            r = run_child(__file__, tag, ["--child", "draw", "--dim-batch", str(db),
                                          "--compile-layers", policy, "--out", str(dest)],
                          quiet=True)
            if not r:
                continue
            # Gate before the draw enters any group. A miscompiled Jacobian is not a
            # variant and not a datum -- averaging one in would contaminate every
            # comparison it appears in.
            if not identity_in_band(r["identity_distance"], EXPECTED_IDENTITY, IDENTITY_TOL):
                print(f"     {tag} MISCOMPILED (identity={r['identity_distance']:.6f})"
                      f" -- dropped", flush=True)
                dest.unlink(missing_ok=True)
                dropped += 1
                continue
            draws.append(r)
            progress.write_text(json.dumps(
                {"complete": False, "n_sound": len(draws), "n_dropped": dropped,
                 "arms": [list(a) for a in arms], "draws": draws}, indent=2,
                default=str) + "\n")

    print(f"\n{len(draws)} sound draws, {dropped} dropped as miscompiled.")
    load_tensors(draws)

    # Variants are per (configuration, dim_batch): the sound values differ between
    # configurations, so clustering across arms would split on configuration instead.
    for d in draws:
        d["arm"] = (d["compile_layers"], d["dim_batch"])
    for arm in {d["arm"] for d in draws}:
        members = [d for d in draws if d["arm"] == arm]
        clusters = cluster_variants([d["identity_distance"] for d in members])
        for vi, cluster in enumerate(clusters):
            for i in cluster:
                members[i]["variant"] = vi
        vals = ", ".join(f"v{i}: {members[c[0]]['identity_distance']:.6f} x{len(c)}"
                         for i, c in enumerate(clusters))
        print(f"  {arm[0]}:{arm[1]} -> {len(clusters)} variant(s)  [{vals}]")

    results = {}
    null = noise_pairs(draws)
    variant_term: dict[tuple, dict[int, float]] = {}

    # 1. variants, within one arm at a time. Same arm and same variant is the only
    #    grouping where "within" is pure noise, so this one needs no supplied null.
    for arm in sorted({d["arm"] for d in draws}):
        members = [d for d in draws if d["arm"] == arm]
        if len({d["variant"] for d in members}) < 2:
            print(f"\n  {arm[0]}:{arm[1]}: one variant only -- no variant comparison")
            continue
        name = f"variant offset, {arm[0]}:{arm[1]}"
        results[name] = decompose(members, lambda d: d["variant"], name)
        variant_term[arm] = {l: (r.get("offset") or 0.0)
                             for l, r in results[name]["layers"].items()}
        report(name, results[name])

    # A cross-arm pair differs by the arm AND by whichever variants it drew, so the
    # variant term is removed too. Use the largest of the arms involved: it is the
    # conservative choice, leaving the arm offset understated rather than inflated.
    def worst_variant_term(arms) -> dict[int, float]:
        terms = [variant_term[a] for a in arms if a in variant_term]
        if not terms:
            return {}
        return {l: max(t.get(l, 0.0) for t in terms) for l in terms[0]}

    # 2. configurations, at fixed dim_batch
    for db in sorted({d["dim_batch"] for d in draws}):
        members = [d for d in draws if d["dim_batch"] == db]
        if len({d["compile_layers"] for d in members}) < 2:
            continue
        name = f"configuration offset, dim_batch={db}"
        results[name] = decompose(members, lambda d: d["compile_layers"], name,
                                  null=null,
                                  subtract=worst_variant_term({d["arm"] for d in members}))
        report(name, results[name])

    # 3. dim_batch, at fixed configuration
    for policy in sorted({d["compile_layers"] for d in draws}):
        members = [d for d in draws if d["compile_layers"] == policy]
        if len({d["dim_batch"] for d in members}) < 2:
            continue
        name = f"dim_batch offset, {policy}"
        results[name] = decompose(members, lambda d: d["dim_batch"], name,
                                  null=null,
                                  subtract=worst_variant_term({d["arm"] for d in members}))
        report(name, results[name])

    dest = out / "offset_profile.json"
    dest.write_text(json.dumps(
        {"task": "offset-profile", "machine": cfg.machine, "model": MODEL_ID,
         "arms": [list(a) for a in arms],
         "n_sound": len(draws), "n_dropped": dropped,
         "draws": [{k: v for k, v in d.items() if k not in ("arm", "tensors")}
                   for d in draws],
         "results": results}, indent=2, default=str) + "\n")
    print(f"\nwrote {dest}")

    if not args.keep:
        for d in draws:
            Path(d["path"]).unlink(missing_ok=True)
        print(f"removed the saved Jacobians (pass --keep to retain them)")


if __name__ == "__main__":
    main()
