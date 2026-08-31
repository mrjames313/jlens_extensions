"""Add a model entry to an existing machine profile, from projections rather than a sweep.

Spec: ``workspace-band-location``, T14 step 3.

`t13_write_profile.py` assembles an entry from a **completed bring-up** -- a full
`dim_batch` sweep plus a timed fit -- and refuses to invent anything. That is right for
the model you are bringing up and wrong for the second model on a box you already
understand, where the sweep costs hours and the 0.8B result already tells you what it
would say: per-prompt time *rises* with `dim_batch`, so the default of 8 is the optimum,
not the ceiling.

So this driver exists to add an entry whose cost and memory fields are **projections**,
and to make that impossible to mistake for measurement:

* ``dim_batch_basis`` takes a value naming the projection, not ``"optimum"``.
* ``measured_over_prompts`` is ``0``.
* ``notes`` records what was projected, from what, and what overwrites it.

`t15_validation_fit.py` reads only ``dim_batch``, ``compile`` and ``gate_identity``, so
a projected ``s_per_prompt`` cannot mis-steer a fit -- it is carried for costing, and
T16's own run replaces it with the measured value.

**This does not measure `force_bos_effective`.** That is a real tokenizer property and
is read from T10's per-model JSON; run ``t10_force_bos.py --model <id>`` first.

Run::

    uv run python experiments/add_model_profile.py --model Qwen/Qwen3.5-4B \\
        --dim-batch 8 --basis "assumed: 0.8B memory model, 36 GB at db=8" \\
        --s-per-prompt 278 --peak-alloc-gb 36.0 --no-compile
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import config as jx_config  # noqa: E402
from jlens_extensions.profile import MachineProfile, ModelFacts  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())


def read_force_bos(model_id: str) -> bool:
    """`force_bos_effective` from T10's per-model measurement, or a clear error."""
    slug = model_id.replace("/", "_")
    directory = cfg.artifact_root / "measurements" / "t10"
    candidates = [directory / f"t10_force_bos_{slug}.json"]
    if model_id == "Qwen/Qwen3.5-0.8B":
        candidates.append(directory / "t10_force_bos.json")
    for path in candidates:
        if path.exists():
            return bool(json.loads(path.read_text())["force_bos_effective"])
    raise SystemExit(
        f"no T10 measurement for {model_id} (looked for {', '.join(map(str, candidates))}).\n"
        f"force_bos_effective is a tokenizer property and must not be inherited from "
        f"another model -- Qwen3.5-0.8B's `false` holds only because that checkpoint has "
        f"no BOS token at all. Run:\n"
        f"    uv run python experiments/t10_force_bos.py --model {model_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dim-batch", type=int, required=True)
    parser.add_argument(
        "--basis", required=True,
        help='What kind of value dim_batch is. Say "assumed: ..." when projected.',
    )
    parser.add_argument("--s-per-prompt", type=float, default=0.0, help="projected")
    parser.add_argument("--peak-alloc-gb", type=float, default=0.0, help="projected")
    parser.add_argument("--peak-reserved-gb", type=float, default=0.0, help="projected")
    compile_group = parser.add_mutually_exclusive_group(required=True)
    compile_group.add_argument("--compile", dest="compile", action="store_true")
    compile_group.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing entry for this model")
    args = parser.parse_args()

    profile = MachineProfile.load(cfg.profile_path)
    if args.model in profile.models and not args.force:
        raise SystemExit(
            f"{cfg.profile_path} already has an entry for {args.model}. Pass --force to "
            f"replace it -- but if it carries measured numbers, replacing them with "
            f"projections is a downgrade."
        )

    force_bos = read_force_bos(args.model)
    profile.models[args.model] = ModelFacts(
        dim_batch=args.dim_batch,
        dim_batch_basis=args.basis,
        compile=args.compile,
        s_per_prompt=args.s_per_prompt,
        peak_alloc_gb=args.peak_alloc_gb,
        peak_reserved_gb=args.peak_reserved_gb,
        force_bos_effective=force_bos,
        measured_over_prompts=0,
        notes=(
            f"Added {date.today().isoformat()} by add_model_profile.py. "
            f"force_bos_effective is measured (T10). dim_batch, s_per_prompt and the "
            f"peak_* fields are PROJECTIONS from the 0.8B cost and memory models in "
            f"f-2026-08-26-gb10-bring-up, not measurements -- measured_over_prompts is 0 "
            f"for that reason. The fit replaces them with measured values."
        ),
    )
    profile.updated = date.today().isoformat()
    profile.write(cfg.profile_path)

    facts = profile.model(args.model)
    print(f"added {args.model} to {cfg.profile_path}")
    print(f"  dim_batch        {facts.dim_batch}  ({facts.dim_batch_basis})")
    print(f"  compile          {facts.compile}")
    print(f"  force_bos_effective {facts.force_bos_effective}   (measured, T10)")
    print(f"  s_per_prompt     {facts.s_per_prompt}  (projected)")
    print(f"  gate_identity    {facts.gate_identity}  <- probe_gate_identity.py writes this")


if __name__ == "__main__":
    main()
