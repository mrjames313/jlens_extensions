"""Measure a model's reference prompt-1 identity_distance, for the fit gate.

`torch.compile` miscompiles this model family in 30-50% of processes, silently, and a
fit defends itself by checking its first prompt against a known-good value. This
measures that value, uncompiled -- the only configuration that has never failed, 0/12
draws reproducing to six decimal places -- and writes it into the machine profile beside
`dim_batch`, where `t15_validation_fit.py` reads it.

About 90 seconds. It is per model, per corpus and per box, so run it once per model on
each box rather than copying a number between them.

Run::

    uv run python experiments/probe_gate_identity.py
    uv run python experiments/probe_gate_identity.py --model Qwen/Qwen3.5-4B
    uv run python experiments/probe_gate_identity.py --dry-run   # measure, do not write
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import config as jx_config  # noqa: E402
from jlens_extensions.fitcmd import probe_reference_identity  # noqa: E402
from jlens_extensions.profile import MachineProfile  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the measurement without writing the profile")
    args = parser.parse_args()

    from fit_lens import load_prompts

    prompt = load_prompts(
        dataset="Salesforce/wikitext", config="wikitext-103-raw-v1", split="train",
        text_field="text", n_prompts=1, max_chars=2000,
    )[0]

    print(f"machine={cfg.machine}  model={args.model}")
    print("measuring prompt-1 identity_distance UNCOMPILED (~90s) ...", flush=True)
    result = probe_reference_identity(args.model, prompt=prompt)
    print(f"  identity_distance = {result['identity_distance']:.6f}  "
          f"(seq_len={result['seq_len']}, n_valid={result['n_valid_positions']})")

    profile = MachineProfile.load(cfg.profile_path)
    facts = profile.model(args.model)
    if facts.gate_identity is not None:
        drift = abs(result["identity_distance"] - facts.gate_identity) / facts.gate_identity
        print(f"  profile already holds {facts.gate_identity:.6f} "
              f"({drift:.2%} from this measurement)")
        if drift > 0.01:
            print("  !! that is outside the gate's own tolerance. Either the corpus or "
                  "the model changed, or one of the two measurements was itself a bad "
                  "draw. Re-run before overwriting.")

    if args.dry_run:
        print("\n--dry-run: profile not written")
        return

    facts.gate_identity = result["identity_distance"]
    facts.gate_identity_basis = result["basis"]
    profile.write(cfg.profile_path)
    print(f"\nwrote gate_identity to {cfg.profile_path}")
    print("t15_validation_fit.py will now gate every compiled fit against it.")


if __name__ == "__main__":
    main()
