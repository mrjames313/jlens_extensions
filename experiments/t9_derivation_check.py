"""Validate the dictionary construction against a model's actual readout.

Run this for **every new model** before trusting any dictionary-derived number
from it. It is cheap, and it is what caught the bug that voided T9's first run:
``Qwen3_5RMSNorm`` applies ``x/rms(x) * (1 + w)`` rather than ``* w``, so reading
``.weight`` as the gain produced a dictionary wrong by a constant offset -- with
plausible output and no error, which is the failure mode the whole module exists
to prevent.

Two things can make our ranking disagree with the model's, and they have very
different consequences:

1. **Precision.** ``HFLensModel.unembed`` casts to the lm_head's dtype (bf16 here)
   and runs the norm and the unembedding matmul there, while ``lens_logits`` runs
   fp32 throughout. bf16 carries 8 mantissa bits (~0.4% relative), and across a
   248k vocabulary the top-1/top-2 gap is often smaller than that. Harmless: the
   instrument is blunt, the algebra is fine.

2. **Construction.** A convention mismatch, a transpose, a layer off-by-one. Every
   dictionary vector is then wrong and nothing raises.

Separating them has to be non-circular -- comparing our formula against our
formula proves nothing -- so the reference is the model's **own** norm module,
deep-copied to fp32: its math, at our precision.

    ours  vs  module-in-fp32   -> isolates CONSTRUCTION (expect ~1.0)
    ours  vs  module-in-bf16   -> isolates PRECISION (expect slightly below)

Then, for positions that still disagree, we ask whether they are near-ties: where
our top-1 ranks under the reference, and the relative logit gap. Near-ties mean
precision; rank displacement means something structural.

Run::

    uv run python experiments/t9_derivation_check.py
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import config as jx_config  # noqa: E402
from jlens_extensions.dictionary import effective_gain, lens_logits  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

import torch  # noqa: E402
import transformers  # noqa: E402

import jlens  # noqa: E402
from jlens.fitting import SKIP_FIRST_N_POSITIONS, valid_position_mask  # noqa: E402

from fit_lens import load_prompts  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-0.8B"
LENS_SUBPATH = Path("qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt")
MAX_SEQ_LEN = 128
N_PROMPTS = 4
PROBE_LAYERS = (0, 11, 22)


def main() -> None:
    lens_path = cfg.reference / LENS_SUBPATH
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    lm = jlens.from_hf(hf_model, tokenizer)

    norm = lm._final_norm
    raw = norm.weight.detach().to(torch.float32)
    # Probed, never read off .weight -- that is the whole point of this script.
    gamma = effective_gain(norm)

    # --- Q1. what IS this module, and did we recover its gain? --------------
    print("--- Q1. the norm module and its effective gain ---")
    print(f"  type            {type(norm).__module__}.{type(norm).__name__}")
    print(f"  weight          shape={tuple(norm.weight.shape)} dtype={norm.weight.dtype}")
    for attr in ("eps", "variance_epsilon", "epsilon"):
        if hasattr(norm, attr):
            print(f"  {attr:<15} {getattr(norm, attr)}")
    print(f"  raw .weight     mean {raw.mean().item():.4f}")
    print(f"  effective gain  mean {gamma.mean().item():.4f}")

    matches_plain = torch.allclose(gamma, raw, atol=1e-3)
    matches_offset = torch.allclose(gamma, raw + 1.0, atol=1e-3)
    convention = (
        "PLAIN  x/rms(x) * w" if matches_plain
        else "OFFSET x/rms(x) * (1 + w)" if matches_offset
        else "NEITHER -- some other diagonal convention"
    )
    print(f"  convention      {convention}")
    if not matches_plain:
        print("                  reading .weight as the gain here would be WRONG;")
        print("                  effective_gain() recovered it by probing instead.")

    # A deep copy at fp32: the module's own arithmetic, at our precision.
    norm32 = copy.deepcopy(norm).float()
    eps = float(getattr(norm, "eps", None) or torch.finfo(torch.float32).eps)
    probe = torch.randn(64, gamma.numel(), device=gamma.device, dtype=torch.float32) * 3.0
    module_out = norm32(probe)
    recovered = gamma * probe * torch.rsqrt(probe.pow(2).mean(-1, keepdim=True) + eps)
    err = ((module_out - recovered).norm() / module_out.norm()).item()
    print(f"\n  recovered gain reproduces the module to {err:.3e}  (expect ~1e-7)")

    # --- Q2/Q3. algebra vs precision ---------------------------------------
    W_U32 = lm._lm_head.weight.detach().to(torch.float32)
    lens = jlens.JacobianLens.load(str(lens_path))
    prompts = load_prompts(
        dataset="Salesforce/wikitext", config="wikitext-103-raw-v1", split="train",
        text_field="text", n_prompts=N_PROMPTS, max_chars=2000,
    )

    print("\n--- Q2. construction (fp32 reference) vs precision (bf16 reference) ---")
    header = f"{'L':>3}  {'vs fp32':>9}  {'vs bf16':>9}  {'median rank':>11}  {'p90 rank':>8}  {'median gap':>10}"
    print(header)
    print("-" * len(header))

    for layer in PROBE_LAYERS:
        J_bar = lens.jacobians[layer].cuda()
        agree32 = agree16 = total = 0
        ranks: list[int] = []
        gaps: list[float] = []
        for text in prompts:
            input_ids = lm.encode(text, max_length=MAX_SEQ_LEN)
            seq_len = int(input_ids.shape[1])
            if seq_len <= SKIP_FIRST_N_POSITIONS + 1:
                continue
            mask = valid_position_mask(seq_len).to(input_ids.device)
            with torch.no_grad():
                with jlens.ActivationRecorder(lm.layers, at=[layer]) as rec:
                    lm.forward(input_ids)
                    h = rec.activations[layer][0][mask].to(torch.float32)
                z = h @ J_bar.T
                ours = lens_logits(h, J_bar, W_U32, gamma, correct=True)
                ours_top1 = ours.argmax(dim=-1)
                # The module's own math, at fp32: isolates algebra.
                ref32 = norm32(z) @ W_U32.T
                # The shipped path, at bf16: isolates precision.
                ref16 = lm.unembed(z).to(torch.float32)

                agree32 += int((ref32.argmax(-1) == ours_top1).sum())
                agree16 += int((ref16.argmax(-1) == ours_top1).sum())
                total += int(h.shape[0])

                # Where bf16 disagrees, how far off is it really?
                bad = ref16.argmax(-1) != ours_top1
                if bad.any():
                    sub_ref, sub_ours = ref16[bad], ours_top1[bad]
                    order = sub_ref.argsort(dim=-1, descending=True)
                    rank = (order == sub_ours.unsqueeze(1)).float().argmax(dim=-1)
                    ranks.extend(rank.tolist())
                    top = sub_ref.max(dim=-1).values
                    mine = sub_ref.gather(1, sub_ours.unsqueeze(1)).squeeze(1)
                    gaps.extend(((top - mine).abs() / top.abs().clamp(min=1e-6)).tolist())

        r = torch.tensor(ranks, dtype=torch.float32) if ranks else torch.zeros(1)
        g = torch.tensor(gaps) if gaps else torch.zeros(1)
        print(
            f"{layer:>3}  {agree32/total:>9.4f}  {agree16/total:>9.4f}  "
            f"{r.median().item():>11.1f}  {r.quantile(0.9).item():>8.1f}  {g.median().item():>10.2e}"
        )

    print("\nreading:")
    print("  'vs fp32' ~1.0          -> construction is right; any shortfall vs bf16 is precision.")
    print("  'vs fp32' well below 1  -> structural, not precision. Stop and debug.")
    print("  median rank 1 (our top-1 is the reference's 2nd) with a tiny gap -> near-ties.")


if __name__ == "__main__":
    main()
