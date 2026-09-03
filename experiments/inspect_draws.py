"""Tensor health for a set of saved draws, when an offset looks wrong rather than large.

``offset_profile.py`` answers "how far apart are these execution paths". When that
distance comes back implausible it cannot say *why*, because every statistic it
reports is a ratio of norms and a ratio hides both of its terms. This reads the saved
Jacobians directly.

Written for 4B prompt 4, where the three surviving groups -- two compile variants and
the uncompiled path -- sat 0.60, 0.62 and 0.90 apart in relative Frobenius at L0
against a within-group noise of 9.9e-3. That is 60-91x, where prompts 0 and 3 gave
0.95x and 3.3x on the same model and configuration. The shape was not the surprise:
``analyse``'s cloud model already says every path lands roughly equidistant from
every other, and the spread table agreed at 1.5-3.0x max/min. The *radius* was.

Four causes produce a large radius, and they are distinguishable here:

* **Non-finite entries.** A NaN or Inf anywhere makes a norm meaningless and a
  difference arbitrary. Counted per layer per draw.
* **A collapsed denominator.** Relative Frobenius divides by ``||A||``, so a small
  ``||A||`` inflates the ratio without anything being wrong with the difference.
  Compare ``||J||`` across prompts, not just within one.
* **Few valid positions.** The Jacobian is a mean over valid positions; a prompt with
  few of them yields a fragile estimate. ``n_valid`` is recorded per draw.
* **A genuinely ill-conditioned input**, where each path is internally reproducible
  and they diverge deterministically. This is what is left when the other three are
  excluded, and it is the interesting answer rather than the default one.

Run::

    uv run python experiments/inspect_draws.py \
        $JLENS_ARTIFACT_ROOT/measurements/offset-profile/offset_profile_p4-diagnosis.json

    # against a prompt known to behave, for the comparison that makes it legible
    uv run python experiments/inspect_draws.py <p4.json> --against <p0.json>
"""

from __future__ import annotations

import argparse
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


def group_label(draw: dict) -> str:
    return (f"{draw.get('compile_layers')}:{draw.get('dim_batch')}"
            f"@p{draw.get('prompt_idx')}/v{draw.get('variant')}")


def summarise(manifest: Path) -> dict:
    """Per-draw norms, finiteness and n_valid, from the manifest's saved tensors."""
    import torch

    blob = json.loads(manifest.read_text())
    draws = [d for d in blob.get("draws", []) if not d.get("screened_out", False)]
    if not draws:
        draws = blob.get("draws", [])
    rows = []
    for draw in draws:
        path = Path(draw["path"])
        if not path.exists():
            print(f"  {group_label(draw)}: tensors gone from {path}, skipping")
            continue
        tensors = torch.load(path, map_location="cpu")
        layers = sorted(tensors)
        norms, nonfinite = {}, {}
        for layer in layers:
            t = tensors[layer].float()
            norms[layer] = t.norm().item()
            bad = int((~torch.isfinite(t)).sum().item())
            if bad:
                nonfinite[layer] = bad
        rows.append({
            "label": group_label(draw),
            "identity_distance": draw.get("identity_distance"),
            "n_valid": draw.get("n_valid"),
            "seq_len": draw.get("seq_len"),
            "norms": norms,
            "nonfinite": nonfinite,
            "layers": layers,
        })
        del tensors
    return {"model": blob.get("model"), "rows": rows}


def report(summary: dict, title: str) -> None:
    rows = summary["rows"]
    if not rows:
        print(f"\n{title}: nothing to inspect")
        return
    layers = rows[0]["layers"]
    shown = [layers[0], layers[len(layers) // 2], layers[-1]]

    print(f"\n=== {title} ({summary.get('model')}) ===")
    bad = {r["label"]: r["nonfinite"] for r in rows if r["nonfinite"]}
    if bad:
        print("  ** NON-FINITE ENTRIES — every norm and every offset here is void **")
        for label, counts in bad.items():
            print(f"    {label}: {counts}")
    else:
        print("  all entries finite")

    hdr = (f"{'group':>28} {'n_valid':>8} {'seq':>5} "
           + " ".join(f"{'||J_' + str(l) + '||':>13}" for l in shown))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:>28} {str(r['n_valid']):>8} {str(r['seq_len']):>5} "
              + " ".join(f"{r['norms'][l]:>13.4e}" for l in shown))


def by_group(summary: dict) -> dict[str, dict[int, float]]:
    """Mean ||J|| per layer, per group. Never pooled across groups -- see below."""
    buckets: dict[str, list[dict]] = {}
    for row in summary["rows"]:
        buckets.setdefault(row["label"], []).append(row)
    return {label: {layer: sum(r["norms"][layer] for r in rows) / len(rows)
                    for layer in rows[0]["layers"]}
            for label, rows in buckets.items()}


def compare_norms(primary: dict, other: dict) -> None:
    """Per-layer ||J||, per group, on both manifests.

    **Per group, never pooled.** The first version of this averaged ||J|| across
    every draw in a manifest and printed one ratio. On 4B prompt 4 that hid the
    result: the three groups there sit at 337, 447 and 833 at L0, and their mean --
    510 -- describes none of them and makes a 2.5x disagreement between execution
    paths look like a 1.75x difference between prompts. A statistic pooled over
    groups that disagree is a statistic about nothing.
    """
    groups = {"this": by_group(primary), "against": by_group(other)}
    print("\n=== ||J|| per layer, per group ===")
    print("A group whose norm collapses relative to the others has an inflated")
    print("relative-Frobenius denominator, and every offset involving it is inflated")
    print("with it. Groups that agree exclude that and point at the input instead.")
    print("Watch the depth trend: a spread that closes toward the last source layer")
    print("is a spread that identity_distance -- measured there -- cannot see.")

    for which, table in groups.items():
        if not table:
            continue
        layers = sorted(next(iter(table.values())))
        shown = [layers[0], layers[len(layers) // 4], layers[len(layers) // 2],
                 layers[-2], layers[-1]]
        print(f"\n  [{which}]")
        hdr = f"{'group':>28} " + " ".join(f"{'L' + str(l):>13}" for l in shown)
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for label, norms in sorted(table.items()):
            print(f"  {label:>28} " + " ".join(f"{norms[l]:>13.4e}" for l in shown))
        if len(table) > 1:
            print(f"  {'spread (max/min)':>28} " + " ".join(
                f"{max(n[l] for n in table.values()) / min(n[l] for n in table.values()):>12.2f}x"
                for l in shown))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path,
                        help="an offset_profile.json whose tensors are still on disk")
    parser.add_argument("--against", type=Path, default=None,
                        help="a second manifest, ideally a prompt that behaved, so the "
                             "norms and n_valid can be read comparatively")
    args = parser.parse_args()

    primary = summarise(args.manifest)
    report(primary, args.manifest.stem)
    if args.against:
        other = summarise(args.against)
        report(other, args.against.stem)

        compare_norms(primary, other)

    out = cfg.artifact_root / "measurements" / "inspect-draws"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.manifest.stem}_health.json"
    dest.write_text(json.dumps(primary, indent=2, default=str) + "\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
