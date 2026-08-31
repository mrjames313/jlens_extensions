"""Is a compile policy sound on this model? N independent draws, one process each.

Spec: ``workspace-band-location``, T15.

`f-2026-08-28-compile-miscompilation` localised the fault on **Qwen3.5-0.8B** to the six
full-attention blocks, and that is a result about one model. The block *kinds* are an
architectural property, so the claim plausibly transfers -- but "plausibly" is not what
you want in front of a 32-hour fit, and this driver is what turns it into a measurement
on the model you are about to fit.

What it measures, and why draws rather than a single run
--------------------------------------------------------

The miscompilation is **per process**: a process either compiles correctly and every
prompt in it is right, or it does not and every prompt is wrong. So one draw tells you
nothing about the rate, and every draw needs a fresh process -- hence
:mod:`jlens_extensions.childproc` rather than a loop.

The readout is prompt-1 ``identity_distance``, computed by the same
``measure_prompt1_identity`` the gate reference uses, so a draw is directly comparable
against the profile's stored ``gate_identity``.

The value structure is the cheap signal
---------------------------------------

At 0.8B a *sound* configuration produced a small number of exactly-repeating values --
two for ``linear-attn`` (0.531430/0.531469), two for ``all`` (0.531268/0.531295), one for
uncompiled (0.531523) -- because a variant is a fixed set of kernels computing a
deterministic result. A *miscompiling* configuration scatters, and every observed failure
sat at least 2.3% from the reference against a 0.05% spread within sound variants.

That is why six draws is informative where six draws of a noisy statistic would not be:
we are not estimating a rate, we are asking whether the readings fall into a tight
discrete set near the reference. A full soak (``--draws 20``) bounds the rate; a reduced
one (``--draws 6``) checks the structure. Both are legitimate and the output says which
you ran, because the confidence attached to the policy is part of the result.

**0/N is not zero.** At 20 draws the 95% upper bound on the failure rate is still 14%.
This narrows the risk; the prompt-1 gate in the fit is what catches the remainder.

Run::

    uv run python experiments/compile_soak.py --model Qwen/Qwen3.5-4B --draws 6
    uv run python experiments/compile_soak.py --model Qwen/Qwen3.5-4B --policy all --draws 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import config as jx_config  # noqa: E402
from jlens_extensions.childproc import emit_result, run_child  # noqa: E402
from jlens_extensions.compile_policy import (  # noqa: E402
    IDENTITY_TOL,
    POLICIES,
    cluster_variants,
    identity_in_band,
)
from jlens_extensions.profile import MachineProfile, ProfileError  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
#: A 4B draw is ~15-20 min uncompiled; compiling adds inductor time on top. Generous,
#: because a timeout here silently discards a draw and biases the very structure we read.
DEFAULT_TIMEOUT_S = 3600


def child(args) -> int:
    """One draw, in its own process."""
    from fit_lens import load_prompts

    from jlens_extensions.fitcmd import measure_prompt1_identity

    prompt = load_prompts(
        dataset="Salesforce/wikitext", config="wikitext-103-raw-v1", split="train",
        text_field="text", n_prompts=1, max_chars=2000,
    )[0]
    started = time.time()
    result = measure_prompt1_identity(
        args.model, prompt=prompt, policy=args.policy,
        dim_batch=args.dim_batch, max_seq_len=args.max_seq_len,
    )
    result["elapsed_s"] = time.time() - started
    emit_result(result)
    return 0


def reference_for(model_id: str) -> float | None:
    """The stored ``gate_identity``, or None with a reason printed.

    Absent, the soak still measures the value structure -- which is most of the signal --
    but cannot say whether a cluster sits at the *right* place, only that it is tight.
    """
    try:
        facts = MachineProfile.load(cfg.profile_path).model(model_id)
    except (ProfileError, KeyError) as exc:
        print(f"  ! no profile entry for {model_id}: {exc}")
        return None
    if facts.gate_identity is None:
        print(f"  ! profile has no gate_identity for {model_id}; run probe_gate_identity.py")
    return facts.gate_identity


def report(values: list[float], reference: float | None, tol: float) -> dict:
    """The value structure of the *sound* draws, and where they sit vs the reference.

    **Clustering runs over in-band values only**, as ``cluster_variants`` requires: a
    miscompiled reading is not a variant, and letting one through would both invent a
    variant and, worse, inflate any null later computed from that grouping. Out-of-band
    draws are counted separately, which is the number that actually decides soundness.

    With no reference we cannot filter, so the clustering is over everything and is
    reported as tightness only -- it can say the readings repeat, not that they repeat
    in the right place.
    """
    out: dict = {"n_draws": len(values)}

    if reference is None:
        sound = list(values)
        out["reference"] = None
        out["filtered"] = False
    else:
        flags = [identity_in_band(v, reference, tol) for v in values]
        sound = [v for v, ok in zip(values, flags) if ok]
        out.update(
            reference=reference,
            tolerance=tol,
            filtered=True,
            n_in_band=sum(flags),
            n_out_of_band=len(flags) - sum(flags),
            out_of_band_values=[v for v, ok in zip(values, flags) if not ok],
            worst_rel_offset=(
                max(abs(v - reference) / abs(reference) for v in values) if values else None
            ),
        )

    clusters = cluster_variants(sound)
    out["n_sound_draws"] = len(sound)
    out["n_distinct_values"] = len(clusters)
    out["clusters"] = [{"value": sound[idx[0]], "count": len(idx)} for idx in clusters]
    out["spread_rel"] = (
        (max(sound) - min(sound)) / abs(statistics.median(sound))
        if sound and statistics.median(sound) else None
    )
    return out


def verdict(summary: dict, policy: str) -> str:
    """One line a reader can act on, plus what it does and does not license."""
    if summary.get("n_out_of_band"):
        return (
            f"UNSOUND: {summary['n_out_of_band']}/{summary['n_draws']} draws outside "
            f"{summary['tolerance']:.1%} of the reference (worst "
            f"{summary['worst_rel_offset']:.2%}). Do not fit with --compile_blocks "
            f"{policy}."
        )
    if summary["n_distinct_values"] > max(3, summary["n_sound_draws"] // 2):
        return (
            f"SUSPECT: {summary['n_distinct_values']} distinct values across "
            f"{summary['n_sound_draws']} in-band draws. A sound configuration repeats "
            f"exactly within a variant, so this many distinct readings is scatter, not "
            f"variants -- even though all sit within tolerance. Draw more before "
            f"trusting it."
        )
    if not summary.get("filtered"):
        return (
            f"UNVERIFIED: {summary['n_distinct_values']} distinct value(s) across "
            f"{summary['n_draws']} draws, so the readings are tight -- but with no "
            f"gate_identity to compare against, this cannot say they are tight in the "
            f"right place. Run probe_gate_identity.py, then re-read this JSON."
        )
    tail = "" if summary["n_draws"] >= 20 else (
        f" At {summary['n_draws']} draws this checks the value structure rather than "
        f"bounding the failure rate; the prompt-1 gate is the backstop."
    )
    return (
        f"SOUND so far: {summary['n_draws']}/{summary['n_draws']} in band, "
        f"{summary['n_distinct_values']} distinct value(s).{tail}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--policy", default="auto", choices=sorted(POLICIES))
    parser.add_argument("--draws", type=int, default=6,
                        help="6 checks the value structure; 20 bounds the rate")
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--tol", type=float, default=IDENTITY_TOL)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        raise SystemExit(child(args))

    print(f"machine={cfg.machine}  model={args.model}  policy={args.policy}  "
          f"draws={args.draws}")
    reference = reference_for(args.model)
    if reference is not None:
        print(f"  reference gate_identity = {reference:.6f}  (tol {args.tol:.1%})")
    print()

    draws: list[dict] = []
    for i in range(args.draws):
        payload = run_child(
            __file__, f"draw {i + 1}/{args.draws}",
            ["--child", "--model", args.model, "--policy", args.policy,
             "--dim-batch", str(args.dim_batch), "--max-seq-len", str(args.max_seq_len)],
            timeout_s=args.timeout_s, quiet=True,
        )
        if payload is None:
            continue
        draws.append(payload)
        mark = ""
        if reference is not None:
            mark = "  ok" if identity_in_band(payload["identity_distance"], reference,
                                              args.tol) else "  OUT OF BAND"
        print(f"     identity_distance = {payload['identity_distance']:.6f}"
              f"  ({payload['elapsed_s']:.0f}s){mark}")

    if not draws:
        raise SystemExit("no draws completed; nothing to report")

    values = [d["identity_distance"] for d in draws]
    summary = report(values, reference, args.tol)
    first = draws[0]

    print("\n--- value structure ---")
    for cluster in summary["clusters"]:
        print(f"  {cluster['value']:.6f}  x{cluster['count']}")
    if summary["spread_rel"] is not None:
        print(f"  spread across all draws: {summary['spread_rel']:.2%} relative")
    print(f"  compiled {first['compiled'].get('n_compiled')} of "
          f"{first['compiled'].get('n_blocks')} blocks "
          f"({first['compiled'].get('n_linear_attn')} linear-attn, "
          f"{first['compiled'].get('n_full_attn')} full-attn)")
    print(f"  median {statistics.median(d['elapsed_s'] for d in draws):.0f}s per draw")
    print(f"\n{verdict(summary, args.policy)}")

    out_dir = cfg.artifact_root / "measurements" / "compile_soak"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "_")
    out_path = out_dir / f"compile_soak_{slug}_{args.policy}_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(
        {"model": args.model, "policy": args.policy, "machine": cfg.machine,
         "summary": summary, "draws": draws}, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
