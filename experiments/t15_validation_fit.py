"""T15 -- the pinned validation fit, run twice.

Spec: ``environment-setup-and-first-fit``, stage 4. Produces the lens T16 compares
against Neuronpedia's published artifact.

Why twice
---------

T18 measured the run-to-run nondeterminism envelope at ``--no_compile``,
``dim_batch=8``, and said explicitly that it does not transfer to this task's
configuration: ``torch.compile`` fuses reductions differently. Its projection to 233
prompts puts L0 at **7.5e-4** against the **~5e-4** fp16 storage floor -- a factor of
1.5 apart. So if compiled execution moves the noise even 2x, *which constraint binds
at L0 flips*, and L0 is the layer where the lens is most interesting.

Two independent fits at the production configuration settle it. The pair is also the
only measurement of our own noise draw that T16's "two noise draws, not one" caveat
can be quantified from.

**Sequentially, never concurrently.** Two fits at once contend for the GPU and perturb
the memory and kernel-selection conditions that are the variable under test.

Stored at fp32, and that is load-bearing
----------------------------------------

``lens.save()`` defaults to fp16 -- the published precision, and the right default for
matching a published artifact. It is the wrong default here, and the arithmetic is not
close.

The ~5e-4 fp16 floor is a **single**-quantisation figure: fp16 machine epsilon is
``2**-11 = 4.9e-4``, and the published lens is one quantised snapshot. Comparing two
fp16 lenses draws that error twice, putting the floor near ``sqrt(2) x 5e-4 = 7e-4``.
T18 projects the run-to-run envelope at 233 prompts to **7.5e-4 at L0** and below the
fp16 floor everywhere above L2. So an fp16 A/B would be measuring its own storage
quantisation at every layer, L0 included. Both runs therefore pass
``--save_dtype float32``.

This does not weaken the comparison against the published lens -- it protects it. Their
side is fp16 whatever we do; ours being fp32 keeps that comparison at the stated
single-draw ~5e-4 rather than inflating it to ~7e-4 by adding a second draw. And fp32 is
strictly more information: T16 can cast our tensor down to fp16 in memory if it ever
wants an exact like-for-like, which is not recoverable in the other direction.

What is pinned, and why
-----------------------

Early stopping is **off** -- ``--stop_at_delta`` is unset, which is ``fit_lens.py``'s
default, so ``--min_prompts`` and ``--stop_window`` are moot. ``--n_prompts`` is pinned
to the published ``results.prompts_fitted`` (233, confirmed against the artifact in T1).
The published lens is a running mean and early stopping is a threshold crossing on a
nearly flat curve, so a hardware-induced difference of a few prompts would show up as a
tensor discrepancy far above the fp16 floor and unrelated to fidelity.

``dim_batch`` and ``compile`` are read from the machine profile rather than typed here.
That is the whole point of T13: a fit that is hand-tuned is a fit whose parameters live
in someone's memory.

Where things land
-----------------

``--out_dir`` goes under ``$JLENS_SCRATCH_ROOT`` because ``fit_lens.py`` puts the
checkpoint inside it and the loop rewrites the entire running sum every prompt --
~96.5 MB x 233 prompts, ~22 GB of writes. (``cfg.checkpoints`` is unused here: the
driver exposes no checkpoint path of its own.) The lens, its convergence trace and the
provenance sidecar are then copied to ``$JLENS_ARTIFACT_ROOT/lenses/t15-<run>/``, which
is what T17's manifest points at.

Resume is refused by default
----------------------------

``jlens.fit(resume=True)`` is the *default* and ``fit_lens.py`` does not override it, so
a stale checkpoint in ``--out_dir`` would be picked up silently. For two runs that must
be independent, that failure would be invisible and would destroy the measurement --
run B would resume run A. Each run gets its own ``--out_dir``, and a pre-existing lens
or checkpoint is a hard error unless ``--resume`` or ``--fresh`` says otherwise.

Run::

    uv run python experiments/t15_validation_fit.py --runs a,b
    uv run python experiments/t15_validation_fit.py --runs a
    uv run python experiments/t15_validation_fit.py --runs b --fresh
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

import os  # noqa: E402

from jlens_extensions import config as jx_config  # noqa: E402
from jlens_extensions import provenance as jx_prov  # noqa: E402
from jlens_extensions.profile import MachineProfile  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

MODEL_ID = "Qwen/Qwen3.5-0.8B"
SLUG = "Qwen3.5-0.8B"          # fit_lens._slug(MODEL_ID); the published stem
N_PROMPTS = 233                # T1: results.prompts_fitted, read off the artifact
MAX_SEQ_LEN = 128
DTYPE = "bfloat16"
DEVICE_MAP = "cuda"
SAVE_DTYPE = "float32"         # see module docstring -- not the published fp16

DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"
TEXT_FIELD = "text"
MAX_CHARS = 2000

FIT_LENS = REPO / "harness" / "fit_lens.py"


def load_corpus() -> tuple[list[str], dict]:
    """Load the pinned corpus and fingerprint it, before spending an hour on a fit.

    ``load_prompts`` is deterministic and streams from the Hub, so both runs see the
    same 233 prompts and this call sees them too. Fingerprinting here rather than
    after the fit means a corpus that has shifted under us fails in seconds.
    """
    from fit_lens import load_prompts

    print(f"loading {N_PROMPTS} prompts from {DATASET} ({DATASET_CONFIG}) ...", flush=True)
    prompts = load_prompts(
        dataset=DATASET, config=DATASET_CONFIG, split=DATASET_SPLIT,
        text_field=TEXT_FIELD, n_prompts=N_PROMPTS, max_chars=MAX_CHARS,
    )
    if len(prompts) != N_PROMPTS:
        raise SystemExit(
            f"corpus returned {len(prompts)} prompts, not {N_PROMPTS}. Pinning "
            f"--n_prompts to the published count only reproduces the published fit if "
            f"the stream still yields that many. Do not fit against a short corpus -- "
            f"investigate the dataset revision first."
        )
    fingerprint = jx_prov.corpus_fingerprint(prompts)
    print(f"  {fingerprint['n_prompts']} prompts, corpus sha256 "
          f"{fingerprint['corpus_sha256'][:16]}...", flush=True)
    return prompts, fingerprint


def build_command(out_dir: Path, dim_batch: int, compile_model: bool,
                  gate_identity) -> list[str]:
    """The published recipe minus early stopping, via the shared builder.

    Built by ``jlens_extensions.fitcmd`` rather than assembled here, because the gate
    against the torch.compile miscompilation has to be on every fit and a per-driver
    argv is how it gets forgotten -- this driver and the dim_batch probe each had their
    own and neither carried it.
    """
    from jlens_extensions.fitcmd import build_fit_command

    return build_fit_command(
        fit_lens_path=FIT_LENS, model_id=MODEL_ID, out_dir=out_dir,
        n_prompts=N_PROMPTS, dim_batch=dim_batch, max_seq_len=MAX_SEQ_LEN,
        dtype=DTYPE, device_map=DEVICE_MAP, save_dtype=SAVE_DTYPE,
        dataset=DATASET, dataset_config=DATASET_CONFIG, dataset_split=DATASET_SPLIT,
        text_field=TEXT_FIELD, max_chars=MAX_CHARS,
        no_compile=not compile_model, gate_identity=gate_identity,
    )


def read_trace(csv_path: Path) -> dict:
    """Summarise the convergence trace, and check no prompt was silently skipped.

    T1 established that the published run has ``n_done == prompt_idx + 1`` on every
    row -- no prompt dropped by ``valid_position_mask`` -- and that this is what makes
    pinning ``--n_prompts`` exact rather than approximate. It is a property of the
    corpus, not a guarantee, so it is re-checked here on our own trace.
    """
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{csv_path} has no data rows")
    skipped = [r for r in rows if int(r["n_done"]) != int(r["prompt_idx"]) + 1]
    last = rows[-1]
    return {
        "rows": len(rows),
        "n_done_final": int(last["n_done"]),
        "identity_distance_final": float(last["identity_distance"]),
        "mean_rel_change_final": float(last["mean_rel_change"]),
        # The first prompt carries torch.compile; the median of the rest is the
        # steady-state figure comparable with T11's 15.53 s.
        "median_s_per_prompt": statistics.median(
            [float(r["elapsed_s"]) for r in rows[1:]] or [float(rows[0]["elapsed_s"])]
        ),
        "first_prompt_s": float(rows[0]["elapsed_s"]),
        "sum_elapsed_s": sum(float(r["elapsed_s"]) for r in rows),
        "prompts_skipped": len(skipped),
        "n_valid_positions": sorted({int(r["n_valid_positions"]) for r in rows}),
        "seq_len": sorted({int(r["seq_len"]) for r in rows}),
    }


def prepare_out_dir(out_dir: Path, label: str, resume: bool, fresh: bool) -> None:
    lens = out_dir / f"{SLUG}_jacobian_lens.pt"
    checkpoint = out_dir / f"{SLUG}_checkpoint.pt"
    existing = [p for p in (lens, checkpoint) if p.exists()]
    if not existing:
        out_dir.mkdir(parents=True, exist_ok=True)
        return
    if fresh:
        print(f"  --fresh: removing {len(existing)} existing file(s) in {out_dir}", flush=True)
        for path in existing:
            path.unlink()
        return
    if resume:
        print(f"  --resume: leaving {[p.name for p in existing]} in place; "
              f"fit() will pick the checkpoint up", flush=True)
        return
    raise SystemExit(
        f"run {label}: {out_dir} already contains {[p.name for p in existing]}.\n"
        f"jlens.fit(resume=True) is the default and fit_lens.py does not override it, "
        f"so continuing would silently resume rather than start a fresh fit -- and runs "
        f"a and b have to be independent for the envelope to mean anything.\n"
        f"Pass --fresh to discard and refit, or --resume if you are deliberately "
        f"continuing an interrupted run."
    )


def run_one(label: str, facts, corpus_fp: dict, resume: bool, fresh: bool,
            gate_identity=None) -> dict:
    out_dir = cfg.scratch_root / "fits" / f"t15-{label}"
    dest = cfg.lenses / f"t15-{label}"
    prepare_out_dir(out_dir, label, resume, fresh)

    cmd = build_command(out_dir, facts.dim_batch, facts.compile, gate_identity)
    print(f"\n=== run {label} ===")
    print("  " + " ".join(cmd), flush=True)
    projected_h = facts.s_per_prompt * N_PROMPTS / 3600.0
    print(f"  profile projects {facts.s_per_prompt:.2f} s/prompt -> {projected_h:.2f}h", flush=True)

    started = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO / "harness"))
    wall_s = time.time() - started
    if proc.returncode != 0:
        raise SystemExit(f"run {label}: fit_lens.py exited {proc.returncode} after {wall_s:.0f}s")

    lens_path = out_dir / f"{SLUG}_jacobian_lens.pt"
    csv_path = out_dir / f"{SLUG}_convergence.csv"
    for path in (lens_path, csv_path):
        if not path.exists():
            raise SystemExit(f"run {label}: expected {path} and it is not there")

    trace = read_trace(csv_path)
    if trace["rows"] != N_PROMPTS:
        print(f"  WARNING: trace has {trace['rows']} rows, expected {N_PROMPTS}", flush=True)
    if trace["prompts_skipped"]:
        print(f"  WARNING: {trace['prompts_skipped']} row(s) have n_done != prompt_idx+1 -- "
              f"a prompt was dropped, so this is not the published 233", flush=True)

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lens_path, dest / lens_path.name)
    shutil.copy2(csv_path, dest / csv_path.name)

    sidecar = jx_prov.build_sidecar(
        task="T15",
        run=label,
        machine=cfg.machine,
        model_id=MODEL_ID,
        command=cmd,
        fit_config={
            "n_prompts": N_PROMPTS, "dim_batch": facts.dim_batch,
            "max_seq_len": MAX_SEQ_LEN, "dtype": DTYPE, "device_map": DEVICE_MAP,
            "compile": facts.compile, "save_dtype": SAVE_DTYPE,
            "target_layer": None, "early_stopping": False,
            "gate_identity": gate_identity, "compile_blocks": "auto",
            "stop_at_delta": None, "checkpoint_every": 1, "resumed": resume,
        },
        profile_path=cfg.profile_path,
        profile_entry={
            "dim_batch": facts.dim_batch, "dim_batch_basis": facts.dim_batch_basis,
            "compile": facts.compile, "s_per_prompt": facts.s_per_prompt,
            "peak_alloc_gb": facts.peak_alloc_gb,
            "force_bos_effective": facts.force_bos_effective,
        },
        corpus={
            "dataset": DATASET, "config": DATASET_CONFIG, "split": DATASET_SPLIT,
            "text_field": TEXT_FIELD, "max_chars": MAX_CHARS, **corpus_fp,
        },
        repo=REPO,
        artifacts={
            "lens": {**jx_prov.artifact_facts(dest / lens_path.name), "dtype": SAVE_DTYPE},
            "convergence_csv": jx_prov.artifact_facts(dest / csv_path.name),
        },
        results={"wall_clock_s": round(wall_s, 1), **trace},
        notes=(
            "One of two independent fits at the production configuration. The pair "
            "measures the run-to-run envelope at compiled dim_batch=8, which T18's "
            "uncompiled measurement does not transfer to."
        ),
    )
    sidecar_path = jx_prov.write_sidecar(dest / f"{SLUG}_provenance.json", sidecar)

    print(f"  done in {wall_s / 3600:.2f}h ({wall_s / max(1, trace['rows']):.2f} s/prompt "
          f"wall, {trace['median_s_per_prompt']:.2f} s/prompt median in-loop)")
    print(f"  identity_distance {trace['identity_distance_final']:.6f}  "
          f"mean_rel_change {trace['mean_rel_change_final']:.8f}")
    print(f"  -> {dest}")
    return {"run": label, "dest": str(dest), "sidecar": str(sidecar_path),
            "wall_s": wall_s, **trace}


def resolve_gate(facts):
    """The gate reference from the profile, or a refusal that says how to get one."""
    from jlens_extensions.fitcmd import UNGATED

    if facts.gate_identity is not None:
        print(f"gate: prompt-1 identity_distance within 1% of {facts.gate_identity:.6f} "
              f"({facts.gate_identity_basis})")
        return facts.gate_identity
    if not facts.compile:
        return UNGATED  # uncompiled has never failed; nothing to guard
    raise SystemExit(
        "the machine profile has no gate_identity for this model, and the fit would "
        "run compiled.\n"
        "torch.compile miscompiles this model in 30-50% of processes and the lens it "
        "saves looks valid (f-2026-08-28-compile-miscompilation).\n"
        "Measure the reference with experiments/probe_gate_identity.py, or pass "
        "--ungated to fit without the check deliberately."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="a,b",
                        help="comma-separated run labels, fitted in order (default: a,b)")
    parser.add_argument("--resume", action="store_true",
                        help="allow an existing checkpoint to be resumed")
    parser.add_argument("--fresh", action="store_true",
                        help="delete any existing lens/checkpoint and refit")
    parser.add_argument("--ungated", action="store_true",
                        help="fit without the compile gate -- deliberate, and recorded "
                             "in the sidecar")
    args = parser.parse_args()
    if args.resume and args.fresh:
        raise SystemExit("--resume and --fresh are mutually exclusive")

    labels = [x.strip() for x in args.runs.split(",") if x.strip()]
    if not labels:
        raise SystemExit("--runs named no runs")

    profile = MachineProfile.load(cfg.profile_path)
    facts = profile.model(MODEL_ID)
    print(f"machine={cfg.machine}  model={MODEL_ID}  runs={labels}")
    print(f"profile={cfg.profile_path}")
    print(f"  dim_batch={facts.dim_batch} ({facts.dim_batch_basis})  compile={facts.compile}  "
          f"s_per_prompt={facts.s_per_prompt}")
    print(f"pinned: n_prompts={N_PROMPTS}, max_seq_len={MAX_SEQ_LEN}, dtype={DTYPE}, "
          f"save_dtype={SAVE_DTYPE}, early stopping OFF")

    _, corpus_fp = load_corpus()

    from jlens_extensions.fitcmd import UNGATED
    gate_identity = UNGATED if args.ungated else resolve_gate(facts)
    results = [run_one(label, facts, corpus_fp, args.resume, args.fresh, gate_identity)
               for label in labels]

    print("\n--- T15 summary ---")
    header = f"{'run':>4} {'rows':>5} {'wall_h':>7} {'s/prompt':>9} {'identity_distance':>18} {'mean_rel_change':>16}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['run']:>4} {r['rows']:>5} {r['wall_s'] / 3600:>7.2f} "
              f"{r['median_s_per_prompt']:>9.2f} {r['identity_distance_final']:>18.6f} "
              f"{r['mean_rel_change_final']:>16.8f}")

    out_dir = cfg.artifact_root / "measurements" / "t15"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t15_validation_fit.json"
    out_path.write_text(json.dumps(
        {"task": "T15", "machine": cfg.machine, "model": MODEL_ID,
         "n_prompts": N_PROMPTS, "save_dtype": SAVE_DTYPE,
         "corpus": corpus_fp, "runs": results}, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    if len(results) > 1:
        print("\nNext: T16 compares run a against run b for the production-config envelope,")
        print("then the pair against the published lens, each axis against the floor that binds it.")


if __name__ == "__main__":
    main()
