"""T10 -- does ``from_hf(force_bos=True)`` actually do anything on this tokenizer?

Spec: ``environment-setup-and-first-fit``, stage 3 (bring-up).

``HFLensModel.__init__`` sets ``tokenizer.add_bos_token = True``, but only behind a
three-part guard::

    force_bos
    and getattr(tokenizer, "bos_token_id", None) is not None
    and hasattr(tokenizer, "add_bos_token")

and upstream's docstring adds a second caveat the vendored copy dropped: "The
attribute may have no effect for some fast-tokenizer configurations." So there are
two independent ways this silently does nothing -- the guard declines to fire, or it
fires and the fast tokenizer's Rust post-processor ignores the Python attribute.

Why it matters even though the fit skips the first 16 positions: ``skip_first``
excludes early positions from the *Jacobian average*, but BOS changes the *forward
pass* everywhere, because every later position attends to position 0. An
attention-sink BOS and a content token at position 0 give different activations at
position 100. So this is not made moot by the mask.

Whatever the answer, it applies to Neuronpedia's published artifacts equally: same
driver, same guard, same tokenizer, and ``fit_lens.py`` never passes ``force_bos``,
so both they and we take the default.

Run::

    uv run python experiments/t10_force_bos.py
"""

from __future__ import annotations

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

import torch  # noqa: E402
import transformers  # noqa: E402

import jlens  # noqa: E402

from fit_lens import load_prompts  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LEN = 128
SAMPLE = "The quick brown fox jumps over the lazy dog."


def describe(tokenizer) -> dict:
    return {
        "add_bos_token": getattr(tokenizer, "add_bos_token", "<attribute absent>"),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
    }


def main() -> None:
    print(f"machine={cfg.machine}")
    print(f"loading {MODEL_ID} ...", flush=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).cuda()
    # Constructed exactly as fit_lens.py:359 does it.
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)

    print("\n--- the tokenizer, before from_hf ---")
    print(f"  class            {type(tokenizer).__name__}  (is_fast={tokenizer.is_fast})")
    before = describe(tokenizer)
    print(json.dumps(before, indent=2, default=str))
    print(f"  model config bos_token_id: {hf_model.config.get_text_config().bos_token_id}")

    # --- would the guard even fire? -----------------------------------------
    has_bos_id = getattr(tokenizer, "bos_token_id", None) is not None
    has_attr = hasattr(tokenizer, "add_bos_token")
    print("\n--- the guard in HFLensModel.__init__ ---")
    print(f"  force_bos (default)                     True")
    print(f"  bos_token_id is not None                {has_bos_id}")
    print(f"  hasattr(tokenizer, 'add_bos_token')     {has_attr}")
    guard_fires = has_bos_id and has_attr
    print(f"  => guard fires and sets the attribute:  {guard_fires}")

    if tokenizer.is_fast:
        try:
            post = tokenizer.backend_tokenizer.post_processor
            print(f"\n  fast backend post_processor: {post}")
        except Exception as exc:  # pragma: no cover - informational only
            print(f"\n  fast backend post_processor unavailable: {exc!r}")

    # --- encode before ------------------------------------------------------
    ids_before = tokenizer(SAMPLE).input_ids

    # --- run the real code path ---------------------------------------------
    lm = jlens.from_hf(hf_model, tokenizer)  # force_bos defaults to True
    after = describe(tokenizer)
    ids_after = tokenizer(SAMPLE).input_ids

    print("\n--- after from_hf(force_bos=True) ---")
    print(f"  add_bos_token: {before['add_bos_token']!r} -> {after['add_bos_token']!r}")
    print(f"  ids before:    {ids_before[:8]}")
    print(f"  ids after:     {ids_after[:8]}")
    ids_changed = ids_before != ids_after
    print(f"  encoding changed: {ids_changed}")

    # --- and does a real fit prompt begin with BOS? -------------------------
    prompts = load_prompts(
        dataset="Salesforce/wikitext", config="wikitext-103-raw-v1", split="train",
        text_field="text", n_prompts=1, max_chars=2000,
    )
    real_ids = lm.encode(prompts[0], max_length=MAX_SEQ_LEN)[0].tolist()
    first_id = real_ids[0]
    starts_with_bos = tokenizer.bos_token_id is not None and first_id == tokenizer.bos_token_id

    print("\n--- position 0 of an actual fit prompt ---")
    print(f"  first 6 ids:     {real_ids[:6]}")
    print(f"  decoded[0]:      {tokenizer.decode([first_id])!r}")
    print(f"  is that BOS?     {starts_with_bos}")

    effective = bool(guard_fires and ids_changed and starts_with_bos)
    print("\n" + "=" * 66)
    print(f"  force_bos_effective = {effective}")
    if not effective:
        print("  Position 0 is a content token, not an attention sink. This applies")
        print("  to Neuronpedia's published artifacts equally -- same driver, same")
        print("  guard, and fit_lens.py never passes force_bos.")
    print("=" * 66)

    result = {
        "task": "T10",
        "machine": cfg.machine,
        "model": MODEL_ID,
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": tokenizer.is_fast,
        "before": before,
        "after": after,
        "model_config_bos_token_id": hf_model.config.get_text_config().bos_token_id,
        "guard_bos_token_id_present": has_bos_id,
        "guard_add_bos_token_attr_present": has_attr,
        "guard_fires": guard_fires,
        "encoding_changed": ids_changed,
        "ids_before": ids_before[:16],
        "ids_after": ids_after[:16],
        "first_prompt_ids": real_ids[:16],
        "first_token_decoded": tokenizer.decode([first_id]),
        "starts_with_bos": starts_with_bos,
        "force_bos_effective": effective,
    }
    out_dir = cfg.artifact_root / "measurements" / "t10"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t10_force_bos.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
