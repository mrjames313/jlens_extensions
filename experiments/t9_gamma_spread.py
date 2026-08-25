"""T9 -- the gamma norm-gain spread, and the dictionary divergence it causes.

Spec: ``environment-setup-and-first-fit``, stage 3 (bring-up). Needs no fit; the
published lens plus one weight tensor suffice.

Three things are measured, and they answer different questions:

A. **The gain spread.** ``max/min`` of ``final_norm.weight`` and its distribution.
   The two dictionary constructions agree iff the gain is uniform, so the spread
   is what decides how much the correction matters -- and it is model-specific,
   which is why research asked for it per model rather than once.

B. **Ranking overlap.** Top-k tokens under the corrected dictionary versus the
   paper's, over *real* residual streams at the fit's own valid positions. This
   replaces the simulated table in ``f-2026-08-18-jspace-construction-and-norm-gain``,
   whose stated caveat was that random ``W_U`` and ``h`` are unrealistic: real
   unembeddings and residuals are anisotropic, and large-gain dimensions
   plausibly correlate with high-variance residual dimensions.

C. **Vector divergence at a fixed token set.** B asks whether the two
   constructions pick the same tokens. C asks the question the hazard is actually
   about: having picked tokens correctly from proper logits, do you get the same
   *directions* for them -- cosine per vector, and the largest principal angle
   between the two k-dimensional spans, which is what a projection-out ablation
   would remove.

Also checks the derivation end-to-end: the corrected ranking should reproduce the
model's real ``unembed()`` top-1, because the omitted ``1/rms`` is positive and
shared across tokens. A low agreement rate means the algebra or the layer
indexing is wrong and nothing else here should be believed.

Run from anywhere::

    uv run python experiments/t9_gamma_spread.py
    T9_N_PROMPTS=16 uv run python experiments/t9_gamma_spread.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# harness/ is run in place so `import jlens` resolves to harness/jlens/, exactly as
# Neuronpedia's driver intends -- see PROVENANCE.md for why that is load-bearing.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))  # no-op when the package is installed

from jlens_extensions import config as jx_config  # noqa: E402
from jlens_extensions.dictionary import dictionary_vectors, gain_spread, lens_logits  # noqa: E402

cfg = jx_config.load()
# Confine HuggingFace downloads to our scratch root. Must precede the transformers
# and datasets imports -- both read these at import time.
os.environ.update(cfg.hf_env())

import torch  # noqa: E402
import transformers  # noqa: E402

import jlens  # noqa: E402
from jlens.fitting import SKIP_FIRST_N_POSITIONS, valid_position_mask  # noqa: E402

from fit_lens import load_prompts  # noqa: E402

# T1 resolved these; two independent sources agree on the HF id.
MODEL_ID = "Qwen/Qwen3.5-0.8B"
LENS_SUBPATH = Path("qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt")

# Corpus and truncation from the published recipe, so we read out over the same
# kind of activations the lens was fitted on.
DATASET, DATASET_CONFIG, SPLIT = "Salesforce/wikitext", "wikitext-103-raw-v1", "train"
MAX_CHARS, MAX_SEQ_LEN = 2000, 128

N_PROMPTS = int(os.environ.get("T9_N_PROMPTS", "8"))
# k is a per-experiment parameter in the paper, not a global constant: 10 for the
# ablation, 16 for verbal report, 25 for internal reasoning. 10 is also what the
# simulated table being replaced used.
K_VALUES = (10, 16, 25)
K_MAX = max(K_VALUES)
K_VECTORS = 10
# Positions sampled per (prompt, layer) for part C. Bounded to keep the QR/SVD
# work incidental; the statistic is stable well before this matters.
N_VECTOR_POSITIONS = 16


def main() -> None:
    started = time.time()
    lens_path = cfg.reference / LENS_SUBPATH
    print(f"machine={cfg.machine}  artifact_root={cfg.artifact_root}")
    if not lens_path.exists():
        raise SystemExit(f"published lens not found at {lens_path} -- run T8's fetcher first")

    print(f"loading {MODEL_ID} ...", flush=True)
    # torch_dtype= rather than dtype=, matching fit_lens.py:347 -- that spelling is
    # what already works against the installed transformers.
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    # Defaults match the fit (force_bos=True); no compile, this is forward-only.
    lm = jlens.from_hf(hf_model, tokenizer)
    print(f"  {lm!r}  layout={lm.layout}", flush=True)
    if not hasattr(lm._final_norm, "weight"):
        raise SystemExit(f"final norm {type(lm._final_norm).__name__} has no .weight")

    # ------------------------------------------------------------------ part A
    gamma = lm._final_norm.weight.detach().to(torch.float32)
    gamma_stats = gain_spread(gamma)
    gamma_stats["dtype"] = str(lm._final_norm.weight.dtype)
    print("\n--- A. gamma (final_norm.weight) ---")
    print(json.dumps(gamma_stats, indent=2))

    W_U = lm._lm_head.weight.detach().to(torch.float32)
    vocab = int(W_U.shape[0])
    lens = jlens.JacobianLens.load(str(lens_path))
    print(f"\n{lens!r}  vocab={vocab}", flush=True)

    print(f"\nstreaming {N_PROMPTS} prompts from {DATASET} ...", flush=True)
    prompts = load_prompts(
        dataset=DATASET,
        config=DATASET_CONFIG,
        split=SPLIT,
        text_field="text",
        n_prompts=N_PROMPTS,
        max_chars=MAX_CHARS,
    )

    layers = lens.source_layers
    jacobians = {layer: lens.jacobians[layer].cuda() for layer in layers}

    n_positions = {layer: 0 for layer in layers}
    overlap = {layer: {k: 0 for k in K_VALUES} for layer in layers}
    top1_agree = {layer: 0 for layer in layers}
    readout_agree = {layer: 0 for layer in layers}
    cosine_sum = {layer: 0.0 for layer in layers}
    cosine_min = {layer: 1.0 for layer in layers}
    cosine_n = {layer: 0 for layer in layers}
    angle_max = {layer: 0.0 for layer in layers}

    print("\nreading out ...", flush=True)
    for index, text in enumerate(prompts):
        input_ids = lm.encode(text, max_length=MAX_SEQ_LEN)
        seq_len = int(input_ids.shape[1])
        if seq_len <= SKIP_FIRST_N_POSITIONS + 1:
            print(f"  prompt {index}: {seq_len} tokens -- too short, skipped")
            continue
        # The fit's own mask: early positions are attention-sink dominated and the
        # final position has no next-token target.
        mask = valid_position_mask(seq_len).to(input_ids.device)

        with torch.no_grad():
            with jlens.ActivationRecorder(lm.layers, at=layers) as recorder:
                lm.forward(input_ids)
                # Block outputs, keyed by block index -- the fit's convention, taken
                # from the fit's own recorder rather than re-derived.
                residuals = {
                    layer: recorder.activations[layer][0][mask].to(torch.float32)
                    for layer in layers
                }

            for layer in layers:
                h = residuals[layer]
                J_bar = jacobians[layer]
                corrected = lens_logits(h, J_bar, W_U, gamma, correct=True)
                plain = lens_logits(h, J_bar, W_U, gamma, correct=False)

                top_corrected = corrected.topk(K_MAX, dim=-1).indices
                top_plain = plain.topk(K_MAX, dim=-1).indices
                n_here = int(h.shape[0])
                n_positions[layer] += n_here
                for k in K_VALUES:
                    a, b = top_corrected[:, :k], top_plain[:, :k]
                    overlap[layer][k] += int((a.unsqueeze(2) == b.unsqueeze(1)).any(2).sum())
                top1_agree[layer] += int((top_corrected[:, 0] == top_plain[:, 0]).sum())

                # Derivation check against the model's real norm + unembed.
                real_top1 = lm.unembed(h @ J_bar.T).argmax(dim=-1)
                readout_agree[layer] += int((real_top1 == top_corrected[:, 0]).sum())
                del corrected, plain

                # -------------------------------------------------------- part C
                step = max(1, n_here // N_VECTOR_POSITIONS)
                for position in range(0, n_here, step):
                    token_ids = top_corrected[position, :K_VECTORS]
                    v_corrected = dictionary_vectors(W_U, gamma, J_bar, token_ids, correct=True)
                    v_plain = dictionary_vectors(W_U, gamma, J_bar, token_ids, correct=False)
                    cosine = torch.nn.functional.cosine_similarity(v_corrected, v_plain, dim=-1)
                    cosine_sum[layer] += float(cosine.sum())
                    cosine_min[layer] = min(cosine_min[layer], float(cosine.min()))
                    cosine_n[layer] += int(cosine.numel())
                    # Largest principal angle between the two k-dim spans: the
                    # subspace a projection-out ablation would actually remove.
                    q_corrected, _ = torch.linalg.qr(v_corrected.T)
                    q_plain, _ = torch.linalg.qr(v_plain.T)
                    singular = torch.linalg.svdvals(q_corrected.T @ q_plain).clamp(-1.0, 1.0)
                    angle = float(torch.rad2deg(torch.arccos(singular.min())))
                    angle_max[layer] = max(angle_max[layer], angle)

        print(f"  prompt {index}: {seq_len} tokens, {int(mask.sum())} valid positions", flush=True)

    per_layer = []
    for layer in layers:
        total = n_positions[layer]
        per_layer.append(
            {
                "layer": layer,
                "n_positions": total,
                **{f"overlap@{k}": overlap[layer][k] / total / k for k in K_VALUES},
                "top1_agreement": top1_agree[layer] / total,
                "real_readout_top1_agreement": readout_agree[layer] / total,
                "mean_cosine_top10": cosine_sum[layer] / cosine_n[layer],
                "min_cosine_top10": cosine_min[layer],
                "max_principal_angle_deg": angle_max[layer],
            }
        )

    print("\n--- B/C. per layer (overlap@k = mean |intersection| / k) ---")
    header = (
        f"{'L':>3}  {'ovl@10':>7} {'ovl@16':>7} {'ovl@25':>7}  {'top1':>6}  "
        f"{'cos':>7} {'cosmin':>7}  {'angle':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in per_layer:
        print(
            f"{row['layer']:>3}  {row['overlap@10']:>7.3f} {row['overlap@16']:>7.3f} "
            f"{row['overlap@25']:>7.3f}  {row['top1_agreement']:>6.3f}  "
            f"{row['mean_cosine_top10']:>7.4f} {row['min_cosine_top10']:>7.4f}  "
            f"{row['max_principal_angle_deg']:>6.1f}"
        )

    agreement = [row["real_readout_top1_agreement"] for row in per_layer]
    print("\nderivation check -- corrected ranking vs real unembed() top-1:")
    print(f"  min over layers {min(agreement):.4f}, mean {sum(agreement)/len(agreement):.4f}  (expect ~1.0)")

    result = {
        "task": "T9",
        "machine": cfg.machine,
        "model": MODEL_ID,
        "lens_path": str(lens_path),
        "vocab": vocab,
        "n_prompts_requested": N_PROMPTS,
        "max_seq_len": MAX_SEQ_LEN,
        "skip_first": SKIP_FIRST_N_POSITIONS,
        "k_values": list(K_VALUES),
        "gamma": gamma_stats,
        "per_layer": per_layer,
        "elapsed_s": time.time() - started,
    }
    out_dir = cfg.artifact_root / "measurements" / "t9"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t9_gamma_spread.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}  ({result['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
