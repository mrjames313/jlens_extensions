"""Where does `dim_batch` change the answer? One prompt, no fit.

Diagnostic, not a measurement. Written because `dim_batch_neutrality.py` returned a
result that cannot be taken at face value.

Run 1 result, 2026-08-27, and why the question changed
------------------------------------------------------

**Uncompiled, everything is fine.** ``identity_distance`` is 0.5314-0.5315 at every
``dim_batch``, matching T15's 0.531422 and the published tensor's 0.531418, and the
Jacobian spread across ``dim_batch`` is 1e-2 to 2e-2 -- the same order as the forward's
own batch-dependence, so roughly 1.6x amplification and nothing more.

**Compiled, one configuration returned garbage**: ``identity_distance`` **8.41** at
``dim_batch=8`` against an expected 0.53, with a uniform ~1.0 relative difference at every
layer. Uniform-at-every-layer is not a ``dim_batch`` signature; it is one tensor being
wrong and the others right. And the cause was printed alongside it::

    torch._dynamo hit config.recompile_limit (8)
    last reason: 0/7: KeyError on self._modules['linear_attn']

Qwen3.5 is hybrid -- ``linear_attn`` state-space blocks with full attention every fourth
layer -- so dynamo guards on the module dict and treats the two layer kinds as separate
compile variants. Four ``dim_batch`` values on top of that exhausts a budget of 8.

Run 2 repeated it with each *check* in its own process and reproduced it: compiled
``dim_batch=8`` gave 7.83 again. So it is not cross-check contamination. But that run
still compiled all four ``dim_batch`` values inside the Check C process, and the pattern
across them is monotonic in the number of **retained-graph backward passes** rather than
in slicing width::

    dim_batch   backward passes   identity_distance
        8            128              7.826707     garbage
       16             64              0.545104     off by 2.6%
       32             32              0.531534     correct
       64             16              0.531402     correct

Which is where it stops adding up. T15 ran **compiled at dim_batch=8** -- 128 passes, the
worst cell -- over 233 prompts, twice, and produced 0.531422 both times, matching the
published tensor to 1.4e-3. A deterministically broken configuration cannot do that.

The one thing neither run isolated: **a real fit compiles exactly one `dim_batch`.** Both
diagnostics compiled four in the same process, which is precisely the condition that
exhausts dynamo's recompile budget on a hybrid model. So the remaining hypothesis is that
the breakage is caused by the diagnostic's own multiplicity, not by compile per se -- and
that decides whether T15 stands.

**The question is therefore no longer about `dim_batch`.** It is: *does `torch.compile`
change the answer at a single configuration in a clean process?* Our fits run compiled, so
if it does, T15, T16 and the envelope work are all suspect. Every configuration below now
runs in its own process, and compiled-vs-uncompiled at the same `dim_batch` is the first
comparison reported.

The original contradiction
--------------------------

Fitting at ``dim_batch=64`` and comparing against our ``dim_batch=8`` pair gave per-layer
relative Frobenius differences of **1.0 to 4.1** -- not a small numerical effect, but
essentially unrelated tensors. Yet T16 found our ``dim_batch=8`` lens agreed with
Neuronpedia's, fitted at ``dim_batch=128``, to **1.4e-3**. Both cannot be true of a
well-behaved estimator: if slicing width mattered at order 1, an 8-wide fit could not
match a 128-wide reference at 1e-3.

So the ``dim_batch=64`` result is a symptom of something, and what it is a symptom of has
to be established before any of it is written up.

What is already known, from the harness's own diagnostics rather than from any comparison
code of ours:

* ``identity_distance`` on the **first prompt** is 0.531268 at ``dim_batch=8`` and 0.543
  at 64. One prompt, one Jacobian -- so this is inside ``jacobian_for_prompt``, not in
  accumulation, and needs no 233-prompt fit to reproduce.

  Provenance note, added 2026-08-29: the 0.531268 here is **not** T15's value. Both T15
  runs read 0.531295 at prompt 1 -- the other of the two sound all-blocks compile variants
  (`f-2026-08-28-compile-miscompilation`). Both are sound and the argument above is
  unaffected, but do not use 0.531268 as T15's variant label: it would say the pair split
  when it did not.
* The whole convergence trajectory differs (Δmean < 0.01 at 45 prompts against 68).

What this separates
-------------------

Three candidates, and the three checks that tell them apart.

1. **The replicated forward differs by batch size.** ``jacobian_for_prompt`` runs the
   prompt replicated ``dim_batch`` times. Qwen3.5 is a **hybrid**: `linear_attn`
   state-space blocks with full attention every fourth layer
   (``full_attention_interval: 4``). SSM kernels commonly switch between chunked and
   sequential scans on batch size, and the observed error had a **period-4 sawtooth**
   across layers, which is suspicious in exactly that direction. Check A compares the
   forward's residual stream across batch sizes, and checks that replicas within one
   batch agree with each other.

2. **`torch.compile` specialises on batch size and one specialisation is wrong.**
   ``from_hf(compile=True)`` compiles per shape, and T11 already noted a new
   ``dim_batch`` recompiles. Check B runs the same comparison with compile off.

3. **The estimator itself mishandles a wide slice.** If the forward agrees across batch
   sizes and the effect survives with compile off, it is in the backward or the row
   assembly. Check C maps the effect across ``dim_batch`` to locate a threshold.

Each check runs in a **fresh subprocess**, which is T11's established pattern here and
is load-bearing rather than tidy. Tearing a CUDA model down and building another in one
process left the first run emitting ``cuBLAS ... there was no current CUDA context`` from
inside the backward -- recoverable, but it means the process is no longer in the state the
measurement assumes, and a diagnostic that cannot be trusted is worse than none. Separate
processes also mean one check crashing does not lose the others.

Cost: seconds to a couple of minutes. No fitting.

Run::

    uv run python experiments/dim_batch_diagnosis.py
    uv run python experiments/dim_batch_diagnosis.py --dim-batches 8,16,32,64
    uv run python experiments/dim_batch_diagnosis.py --soak 10 --dim-batches 16
    uv run python experiments/dim_batch_diagnosis.py --soak 10 --compile-layers full-attn
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import config as jx_config  # noqa: E402

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LEN = 128
D_MODEL = 1024
WARMUP_CONTEXT = False
DONATED_BUFFER = True
DEFAULT_DIM_BATCHES = "8,16,32,64"
#: T15's setting, and the cell with the highest observed failure rate.
SOAK_DEFAULT_DIM_BATCH = 8

#: Prompt-1 identity_distance for this model on this corpus, from runs independently
#: corroborated against Neuronpedia's published lens. Compiled sound runs give
#: 0.53127-0.53130 and uncompiled 0.531523; the miscompiled modes give ~0.5435 (subtle)
#: or 5-8 (catastrophic). This is the cheapest validity gate available -- every fit
#: writes it to row 1 of its convergence CSV.
EXPECTED_IDENTITY = 0.5314
IDENTITY_TOL = 0.01


def rel(a, b) -> float:
    n = b.norm().item()
    return (a - b).norm().item() / n if n else float("nan")


def maybe_warm_context(enabled: bool) -> None:
    if not enabled:
        return
    from jlens_extensions.compile_policy import ensure_cuda_context

    info = ensure_cuda_context()
    print(f"    warmed CUDA context and autograd workers: {info}", flush=True)


def maybe_disable_donated_buffer(enabled: bool) -> None:
    """Optionally turn off AOTAutograd's donated-buffer optimisation before compiling.

    A lead, on the same footing ``maybe_warm_context`` had before it was tested and
    closed -- and a better-founded one, because it is a documented incompatibility
    with the exact pattern our estimator uses rather than a warning that happens to
    co-occur.

    Donated buffers let the compiled backward free intermediate activations as it
    consumes them. That is sound when the backward runs once. ``jacobian_for_prompt``
    runs it ``d_model/dim_batch`` times -- 128 at ``dim_batch=8`` -- with the graph
    retained, seeding a different cotangent each time. Upstream states the two are
    incompatible: a backward compiled with non-empty donated buffers requires
    ``create_graph=False`` and ``retain_graph=False``, and the documented escape is
    exactly this flag (locuslab/torchdeq#11, pytorch/pytorch#139669, which notes the
    combination can go silently wrong rather than raising).

    Why it fits what we measured, better than block-kind selection does:

    * **Compile-only.** Donation is an AOTAutograd optimisation; eager has no such
      path. Uncompiled runs are not merely passing here, they are deterministic to
      six decimals, which a codegen story does not explain and this one does.
    * **Silent and catastrophic.** Reading a buffer that was freed and reused returns
      whatever now occupies it -- identity 0.83, 5.1, 7.4-9.3, not a clean offset.
    * **Per-process.** Whether the memory is overwritten before the next backward
      reads it depends on allocator and scheduling state, which varies per process
      and is stable within one. Our bad fit was wrong from its first row to its last.
    * **Block-kind localisation.** Which buffers are donatable depends on a block's
      activation structure, and full attention's intermediates look nothing like
      Gated DeltaNet's recurrent state. This would give the localisation in
      ``f-2026-08-28-compile-miscompilation`` a mechanism, which it currently lacks.

    **The objection, which this experiment exists to settle.** The incompatibility is
    supposed to *raise*. We have never seen that error, so either this path bypasses
    the guard or our graph donates nothing and the hypothesis is simply wrong. One
    soak at ``--donated-buffer off`` against the 10/20 all-blocks baseline decides it:
    0/20 is a mechanism, ~10/20 kills the lead for half an hour of GPU time.
    """
    if enabled:
        return
    import torch._functorch.config as fx_config

    before = getattr(fx_config, "donated_buffer", None)
    if before is None:
        print("    donated_buffer: NOT PRESENT in this torch build -- flag is a no-op,"
              " and this run does NOT test the hypothesis", flush=True)
        return
    fx_config.donated_buffer = False
    print(f"    donated_buffer: {before} -> {fx_config.donated_buffer}", flush=True)


def build(compile_model: bool, which: str = "all"):
    """Build the lens model, optionally compiling only some residual blocks.

    ``from_hf`` compiles per block -- ``self.layers[i] = torch.compile(...)`` -- so
    the two block kinds can be compiled independently. Qwen3.5 alternates Gated
    DeltaNet (linear attention, ``.linear_attn``) with full attention roughly 3:1,
    and the dynamo guard that kept failing named exactly that attribute
    (``KeyError on self._modules['linear_attn']``). Compiling one kind and not the
    other says which is implicated.

    ``which``: ``all`` (the default, what our fits do), ``full-attn`` (compile only
    blocks *without* ``.linear_attn``), ``linear-attn`` (only those *with* it), or
    ``none``.
    """
    import torch
    import transformers

    import jlens

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok, compile=(compile_model and which == "all"))

    if compile_model and which in ("full-attn", "linear-attn"):
        wanted_linear = which == "linear-attn"
        n = 0
        for i, block in enumerate(model.layers):
            is_linear = hasattr(block, "linear_attn")
            if is_linear == wanted_linear:
                model.layers[i] = torch.compile(block, mode="default", dynamic=False)
                n += 1
        print(f"    compiled {n}/{len(model.layers)} blocks ({which})", flush=True)
    return model


def one_prompt() -> str:
    from fit_lens import load_prompts

    return load_prompts(
        dataset="Salesforce/wikitext", config="wikitext-103-raw-v1", split="train",
        text_field="text", n_prompts=1, max_chars=2000,
    )[0]


def check_a_forward(model, prompt: str, batches: list[int]) -> dict:
    """Does the replicated forward depend on how many replicas there are?"""
    import torch

    from jlens.hooks import ActivationRecorder

    print("\n=== Check A: the replicated forward, across batch sizes ===")
    print("jacobian_for_prompt replicates the prompt dim_batch times. Every replica is")
    print("identical, so every batch size should give the same residual stream.")

    ids = model.encode(prompt, max_length=MAX_SEQ_LEN)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    layers = list(range(model.n_layers))
    out: dict[int, dict] = {}

    with torch.no_grad():
        for bs in batches:
            batched = ids.expand(bs, -1).contiguous()
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(batched)
                acts = {l: rec.activations[l].detach().float() for l in layers}
            # Within one batch: do the replicas agree with each other?
            spread = max(rel(acts[l][i], acts[l][0]) for l in layers for i in (1, bs - 1)) \
                if bs > 1 else 0.0
            out[bs] = {"row0": {l: acts[l][0].clone() for l in layers},
                       "within_batch_spread": spread}
            print(f"  batch={bs:>4}: replicas within the batch agree to {spread:.3e}")
            del acts

    base = batches[0]
    print(f"\n  residual stream vs batch={base}, per layer (max over layers shown):")
    cross = {}
    for bs in batches[1:]:
        per_layer = {l: rel(out[bs]["row0"][l], out[base]["row0"][l]) for l in layers}
        worst = max(per_layer, key=lambda l: per_layer[l])
        cross[bs] = per_layer
        print(f"  batch={bs:>4}: max {per_layer[worst]:.3e} at L{worst}, "
              f"L0 {per_layer[0]:.3e}, L{layers[-1]} {per_layer[layers[-1]]:.3e}")

    verdict = max((max(v.values()) for v in cross.values()), default=0.0)
    print(f"\n  VERDICT A: forward {'DIFFERS' if verdict > 1e-3 else 'agrees'} across "
          f"batch sizes (worst {verdict:.3e})")
    if verdict > 1e-3:
        print("  -> the model, not the estimator. A hybrid SSM kernel switching on batch")
        print("     size would do this, and it would make dim_batch unsafe on this family.")
    return {"within_batch": {str(k): v["within_batch_spread"] for k, v in out.items()},
            "cross_batch": {str(k): {str(l): x for l, x in v.items()} for k, v in cross.items()},
            "worst": verdict}


def compute_jacobian(dim_batch: int, compile_model: bool, out_path: Path,
                     which: str = "all") -> dict:
    """One prompt's Jacobian at one configuration, in this process, saved to disk.

    Saved rather than returned because each configuration runs in its own process --
    see the module docstring for why that is not optional here.
    """
    import torch

    from jlens.fitting import jacobian_for_prompt

    maybe_disable_donated_buffer(DONATED_BUFFER)
    maybe_warm_context(WARMUP_CONTEXT)
    model = build(compile_model, which)
    prompt = one_prompt()
    source_layers = list(range(model.n_layers - 1))
    J, seq_len, n_valid = jacobian_for_prompt(
        model, prompt, source_layers, target_layer=None,
        dim_batch=dim_batch, max_seq_len=MAX_SEQ_LEN,
    )
    late = max(source_layers)
    ident = (J[late].float() - torch.eye(D_MODEL)).norm().item() / D_MODEL**0.5

    recompiles = None
    try:  # how close this configuration ran to the dynamo recompile ceiling
        import torch._dynamo as dynamo

        recompiles = {"limit": dynamo.config.recompile_limit}
    except Exception:  # noqa: BLE001 - diagnostics must not fail the diagnostic
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({l: J[l].float() for l in source_layers}, str(out_path))
    return {"dim_batch": dim_batch, "compile": compile_model,
            "identity_distance": ident, "seq_len": seq_len, "n_valid": n_valid,
            "path": str(out_path), "dynamo": recompiles}


def compute_via_fit(dim_batch: int, compile_model: bool, out_path: Path,
                    which: str = "all") -> dict:
    """The same single-prompt Jacobian, but reached through ``fit()``.

    This is the discriminator. ``jacobian_for_prompt`` called directly and called by
    ``fit()`` receive identical arguments -- ``target_layer=None`` resolves to
    ``n_layers-1`` inside the callee, ``skip_first`` defaults to 16 either way -- so if
    the two paths disagree, the fault is in the call *context*, not the estimator.

    It matters because our fits take this path. T15 ran compiled at ``dim_batch=8``,
    twice, over 233 prompts, and produced ``identity_distance`` 0.531422 both times
    against the published tensor's 0.531418. The direct call at that same configuration
    returns garbage that varies run to run. Both observations cannot be describing the
    same code path, and which one describes our lenses is the whole question.
    """
    import torch

    from jlens.fitting import fit

    maybe_disable_donated_buffer(DONATED_BUFFER)
    maybe_warm_context(WARMUP_CONTEXT)
    model = build(compile_model, which)
    prompt = one_prompt()
    lens = fit(
        model, [prompt],
        dim_batch=dim_batch,
        max_seq_len=MAX_SEQ_LEN,
        checkpoint_path=None,
        resume=False,
    )
    late = max(lens.jacobians)
    ident = (lens.jacobians[late].float() - torch.eye(D_MODEL)).norm().item() / D_MODEL**0.5
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({l: v.float() for l, v in lens.jacobians.items()}, str(out_path))
    return {"dim_batch": dim_batch, "compile": compile_model, "via": "fit",
            "identity_distance": ident, "n_prompts": lens.n_prompts,
            "path": str(out_path)}


RESULT_PREFIX = "@@DIAG_RESULT@@ "


def child(args) -> None:
    if args.child == "forward":
        payload = check_a_forward(build(compile_model=False), one_prompt(),
                                  [int(x) for x in args.dim_batches.split(",")])
    elif args.child == "jacobian":
        payload = compute_jacobian(args.dim_batch, args.compiled, Path(args.out),
                                   args.compile_layers)
    elif args.child == "via_fit":
        payload = compute_via_fit(args.dim_batch, args.compiled, Path(args.out),
                                  args.compile_layers)
    else:
        raise SystemExit(f"unknown check {args.child!r}")
    print(RESULT_PREFIX + json.dumps(payload, default=str), flush=True)


#: Per-child ceiling. Measured children take 42-95s, so this is generous by ~4x.
#: It was 900s, which is worse than useless: nothing legitimate runs that long, and
#: the longer the ceiling the longer a hang looks like work.
CHILD_TIMEOUT_S = 300


def run_child(label: str, extra: list[str], quiet: bool = False) -> dict | None:
    """Run one child to completion, announcing it first.

    ``capture_output=True`` means nothing the child prints is visible until it exits,
    so without the announcement below a two-minute compile looks like a hang. The
    announcement is flushed explicitly because this output is usually piped to a log,
    where Python block-buffers and an unflushed line would not appear either.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), *extra]
    # A COMPLETE line, not a trailing "label ... " stub. A pager or a log reads by
    # line and holds a newline-less fragment until a newline arrives, so the
    # in-progress announcement was invisible through `| more` -- which made a run
    # that was merely slow look frozen. Two lines per child is a small price for
    # progress that survives being piped.
    print(f"  -> {label} starting", flush=True)
    started = time.time()

    # start_new_session puts the child in its own process GROUP so a timeout can kill
    # its descendants too. torch inductor spawns compile-worker subprocesses, and
    # killing only the direct child orphans them -- they keep a GPU context and burn
    # memory for the rest of the run.
    #
    # Note this is orphan hygiene, NOT a hang fix. subprocess.run(timeout=...) was
    # suspected of deadlocking in its post-kill drain; a reproduction with a
    # grandchild deliberately holding the pipe showed it returns promptly, because on
    # POSIX it calls process.wait() rather than communicate(). The theory was wrong
    # and the test is what said so.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"TIMED OUT after {CHILD_TIMEOUT_S}s -- killing the process group",
              flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError) as exc:
            print(f"     could not kill the group: {exc}", flush=True)
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = "", ""
            print("     drain still blocked after the group kill; giving up on it",
                  flush=True)
        print("     the other configurations still run; this one is recorded as missing.",
              flush=True)
        return None
    elapsed = time.time() - started

    payload = None
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX):])
        elif line.strip() and not quiet:
            print(line, flush=True)

    if payload is None:
        print(f"     {label} NO RESULT after {elapsed:.0f}s (exit {proc.returncode})",
              flush=True)
        for t in (stderr or "<no stderr>").strip().splitlines()[-4:]:
            print(f"     {t}", flush=True)
    else:
        note = ""
        if "recompile_limit" in (stderr or ""):
            note = "  !! dynamo hit its recompile limit -- output suspect"
            payload["dynamo_limit_hit"] = True
        ident = payload.get("identity_distance")
        detail = f"identity_distance={ident:.6f}  " if ident is not None else ""
        print(f"     {label} done: {detail}({elapsed:.0f}s){note}", flush=True)
    return payload


def rel_tensors(pa: str, pb: str) -> dict[int, float]:
    import torch

    A, B = torch.load(pa, map_location="cpu"), torch.load(pb, map_location="cpu")
    return {l: rel(A[l], B[l]) for l in sorted(A)}


def classify(ident: float) -> str:
    """Severity of one run's identity_distance against the known-good value."""
    off = abs(ident - EXPECTED_IDENTITY) / EXPECTED_IDENTITY
    if off <= IDENTITY_TOL:
        return "ok"
    return "subtle" if off < 0.5 else "catastrophic"


def soak(dim_batch: int, reps: int, scratch: Path, which: str = "all") -> dict:
    """Repeat ONE configuration many times on both call paths, and count failures.

    The question this settles. At compiled ``dim_batch=8`` the direct call has failed
    5 of 6 observations, while the ``fit()`` path is 5 for 5 clean -- T15's two runs,
    two envelope prompt-1s, and one via-fit. If both paths carried the same risk, five
    clean draws in a row would be about a 0.01% event, so they probably differ. But
    those five are heterogeneous (two 233-prompt finals, two first prompts, one single
    prompt) and n=5 is thin for a claim this load-bearing: it decides whether every
    lens we hold was produced by a path that intermittently corrupts them.

    So: same configuration, same process isolation, both paths, many draws, counted.
    """
    out: dict[str, list] = {"direct": [], "fit": []}
    for path_name, child_mode in (("direct", "jacobian"), ("fit", "via_fit")):
        for rep in range(reps):
            tag = f"soak-db{dim_batch}-{path_name}-r{rep}"
            dest = scratch / f"{tag}.pt"
            extra = ["--child", child_mode, "--dim-batch", str(dim_batch),
                     "--compiled", "--compile-layers", which, "--out", str(dest)]
            if WARMUP_CONTEXT:
                extra.append("--warmup-context")
            if not DONATED_BUFFER:
                extra.extend(["--donated-buffer", "off"])
            r = run_child(tag, extra, quiet=True)
            if r:
                out[path_name].append(r["identity_distance"])
            dest.unlink(missing_ok=True)

    print(f"\n=== soak at compiled dim_batch={dim_batch}, {reps} draws per path ===")
    hdr = f"{'path':>8} {'n':>4} {'ok':>4} {'subtle':>7} {'catastrophic':>13}  {'failure rate':>13}"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for path_name, vals in out.items():
        kinds = [classify(v) for v in vals]
        n = len(vals)
        ok = kinds.count("ok")
        sub = kinds.count("subtle")
        cat = kinds.count("catastrophic")
        rate = (n - ok) / n if n else float("nan")
        summary[path_name] = {"n": n, "ok": ok, "subtle": sub, "catastrophic": cat,
                              "failure_rate": rate, "values": vals}
        print(f"{path_name:>8} {n:>4} {ok:>4} {sub:>7} {cat:>13}  {rate:>12.0%}")

    d, f = summary.get("direct", {}), summary.get("fit", {})
    if d.get("n") and f.get("n"):
        print()
        if f["failure_rate"] == 0 and d["failure_rate"] > 0.5:
            print(f"  -> THE fit() PATH IS MATERIALLY SAFER. {f['n']}/{f['n']} clean against")
            print(f"     {d['ok']}/{d['n']} on the direct call. Our fits take the fit() path,")
            print(f"     so the lenses we hold are not produced by the failing route -- but")
            print(f"     'safer' is not 'safe', and {f['n']} draws bounds the rate at roughly")
            print(f"     1-in-{f['n']}, not at zero. Keep the prompt-1 gate on every fit.")
        elif f["failure_rate"] > 0:
            print(f"  -> BOTH PATHS FAIL. fit() at {f['failure_rate']:.0%}, direct at"
                  f" {d['failure_rate']:.0%}, in this configuration.")
            print(f"     Read that against the other arms, not against zero. This message")
            print(f"     used to say the only safe course was refitting uncompiled; that")
            print(f"     predates the localisation. Compiling only the linear-attention")
            print(f"     blocks measured 0/20 at full speed, and the prompt-1 gate catches")
            print(f"     every mode observed -- see f-2026-08-28-compile-miscompilation.")
            print(f"     Note {f['n'] + d['n']} draws is thin: a 20-draw arm's 95% CI on 50%")
            print(f"     spans 30-70%, so an intervention that looks like it halved the")
            print(f"     rate may have done nothing. Resolving 25% vs 50% needs ~58/arm.")
        else:
            n = f["n"] + d["n"]
            print(f"  -> NO FAILURES in {n} draws at this configuration.")
            print(f"     That is a result, not an anomaly: read it against the other")
            print(f"     configurations rather than against zero. Compiling only the")
            print(f"     linear-attention blocks measured 0/20 where full-attention-only")
            print(f"     measured 6/20 and all-blocks 10/20 (Fisher p = 0.020 and 0.0004),")
            print(f"     which localises the fault to the full-attention blocks.")
            print(f"     Note {n} clean draws bounds the rate near 1-in-{n}, not at zero.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim-batches", default=None,
                        help="comma-separated dim_batch values (default 8,16,32,64). "
                             "With --soak, pass exactly one.")
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the saved Jacobians")
    parser.add_argument("--repeat", type=int, default=2,
                        help="runs per configuration, for the reproducibility null (default 2)")
    parser.add_argument("--soak", type=int, metavar="REPS",
                        help="skip the matrix; repeat ONE dim_batch on both call paths "
                             "REPS times and count failures")
    parser.add_argument("--compile-layers", default="all",
                        choices=("all", "full-attn", "linear-attn", "none"),
                        help="which residual blocks to compile; localises the fault "
                             "to a block kind (default all, what our fits do)")
    parser.add_argument("--warmup-context", action="store_true",
                        help="initialise the CUDA context and autograd workers before "
                             "building the model, then measure whether the failure rate "
                             "changes (a lead, not a known fix)")
    parser.add_argument("--donated-buffer", default="on", choices=("on", "off"),
                        help="'off' disables AOTAutograd's donated-buffer "
                             "optimisation, which upstream documents as incompatible "
                             "with the retained-graph repeated backward this estimator "
                             "runs; compare the failure rate against the baseline for "
                             "the same --compile-layers (a lead, not a known fix)")
    parser.add_argument("--child")
    parser.add_argument("--dim-batch", type=int)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    raw = args.dim_batches
    batches = [int(x) for x in (raw or DEFAULT_DIM_BATCHES).split(",") if x.strip()]

    # --soak takes ONE dim_batch. It used to read a separate --soak-dim-batch, so
    # passing --dim-batches 16 --soak N silently soaked 8 instead. Two flags whose
    # names differ by a plural, and the natural one did nothing -- so there is now
    # one flag, and an explicit list of the wrong length is an error rather than a
    # value quietly discarded.
    soak_db = None
    if args.soak:
        if raw is None:
            soak_db = SOAK_DEFAULT_DIM_BATCH
        elif len(batches) != 1:
            raise SystemExit(
                f"--soak runs one configuration many times, so it needs exactly one "
                f"--dim-batches value; got {batches}. Pass e.g. --dim-batches 16."
            )
        else:
            soak_db = batches[0]

    global WARMUP_CONTEXT, DONATED_BUFFER
    WARMUP_CONTEXT = args.warmup_context
    DONATED_BUFFER = args.donated_buffer == "on"
    if args.child:
        child(args)
        return

    scratch_early = cfg.scratch_root / "diag-jacobians"
    if args.soak:
        print(f"machine={cfg.machine}  model={MODEL_ID}")
        origin = "default" if raw is None else "from --dim-batches"
        print(f"Soak: compiled dim_batch={soak_db} ({origin}), "
              f"{args.compile_layers} blocks compiled,")
        print(f"donated_buffer={args.donated_buffer}, "
              f"{args.soak} draws per path,")
        print(f"each in its own process. ~{2 * args.soak * 45 // 60} minutes.")
        res = soak(soak_db, args.soak, scratch_early, args.compile_layers)
        out = cfg.artifact_root / "measurements" / "dim-batch-diagnosis"
        out.mkdir(parents=True, exist_ok=True)
        # Name the file after the configuration, not just the dim_batch. Every soak
        # used to write soak_db8.json, so a second configuration overwrote the first
        # -- which matters most here, where the run is only meaningful read against
        # the baseline it would have replaced.
        stem = f"soak_db{soak_db}_{args.compile_layers}_donated-{args.donated_buffer}"
        (out / f"{stem}.json").write_text(
            json.dumps({"task": "compile-soak", "machine": cfg.machine,
                        "dim_batch": soak_db, "reps": args.soak,
                        "compile_layers": args.compile_layers,
                        "donated_buffer": args.donated_buffer == "on",
                        "expected_identity": EXPECTED_IDENTITY,
                        "summary": res}, indent=2, default=str) + "\n")
        print(f"\nwrote {out / f'{stem}.json'}")
        return

    print(f"machine={cfg.machine}  model={MODEL_ID}  dim_batches={batches}")
    print("One prompt, no fit. Every configuration in its own process.")
    print("\nThe question that matters: does torch.compile change the answer? Our fits")
    print("run compiled, so if it does, T15 and everything downstream is suspect.")

    scratch = cfg.scratch_root / "diag-jacobians"
    results: dict = {"task": "dim-batch-diagnosis", "machine": cfg.machine,
                     "model": MODEL_ID, "dim_batches": batches}

    if not args.skip_forward:
        results["forward"] = run_child("forward", ["--child", "forward",
                                                   "--dim-batches",
                                                   ",".join(str(b) for b in batches)])

    n_children = 2 * len(batches) * args.repeat
    print(f"\n=== computing Jacobians, ONE configuration per process ===")
    print(f"{n_children} child processes. Each loads the model and compiles from scratch,")
    print(f"so expect roughly 60-120s per child -- the first is slowest on a cold inductor")
    print(f"cache (T11 measured 81.8s there). Total on the order of {n_children * 90 // 60} minutes.")
    print(f"A child is killed and reported if it exceeds {CHILD_TIMEOUT_S}s rather than hanging.")
    print("This is what the previous run did not isolate: it compiled four dim_batch")
    print("values in a single process. A real fit compiles exactly one.")
    print(f"Each configuration runs {args.repeat}x, so within-configuration reproducibility")
    print("is measured rather than assumed -- without it the compile number has no null.")
    meta: dict[tuple[int, bool, int], dict] = {}
    for compiled in (False, True):
        for db in batches:
            for rep in range(args.repeat):
                tag = f"db{db}-{'c' if compiled else 'nc'}-r{rep}"
                path = scratch / f"{tag}.pt"
                extra = ["--child", "jacobian", "--dim-batch", str(db), "--out", str(path)]
                if compiled:
                    extra.append("--compiled")
                r = run_child(tag, extra, quiet=True)
                if r:
                    meta[(db, compiled, rep)] = r

    results["configs"] = {f"db{db}-{'c' if c else 'nc'}-r{r_}": v
                          for (db, c, r_), v in meta.items()}

    # --- The null: does one configuration reproduce itself? -------------------
    if args.repeat > 1:
        print("\n=== within-configuration reproducibility (the null) ===")
        hdr = f"{'config':>14} {'identity r0':>13} {'identity r1':>13} {'max rel diff':>14}"
        print(hdr); print("-" * len(hdr))
        selfdiff = {}
        for compiled in (False, True):
            for db in batches:
                if (db, compiled, 0) not in meta or (db, compiled, 1) not in meta:
                    continue
                tag = f"db{db}-{'c' if compiled else 'nc'}"
                d = rel_tensors(meta[(db, compiled, 0)]["path"],
                                meta[(db, compiled, 1)]["path"])
                selfdiff[tag] = max(d.values())
                print(f"{tag:>14} {meta[(db, compiled, 0)]['identity_distance']:>13.6f} "
                      f"{meta[(db, compiled, 1)]['identity_distance']:>13.6f} "
                      f"{selfdiff[tag]:>14.4e}")
        results["self_reproducibility"] = selfdiff
        # Flag against the measured null rather than a constant: the baseline here is
        # bf16 run-to-run variation, whose size is a property of the model, not a number
        # to hardcode. A fixed 1e-2 threshold flagged a config sitting at 2x the null.
        vals = sorted(selfdiff.values())
        null = vals[len(vals) // 2] if vals else 0.0
        flaky = {k: v for k, v in selfdiff.items() if null and v > 20 * null}
        print(f"\n  baseline null (median across configurations): {null:.3e}")
        if flaky:
            for k, v in sorted(flaky.items()):
                print(f"  !! {k} differs from ITSELF by {v:.3e} = {v / null:.0f}x the null")
            print("     A configuration that does not reproduce itself across two identical")
            print("     runs is nondeterministic, not merely different, and no comparison")
            print("     against it means anything until that is understood.")
        else:
            print("  every configuration reproduces itself within a small multiple of it.")
        results["null"] = null
        results["flaky"] = sorted(flaky)

    # Collapse to r0 for the remaining comparisons.
    meta = {(db, c): v for (db, c, r_), v in meta.items() if r_ == 0}

    # --- The critical axis: does compile change the answer? -------------------
    print("\n=== compiled vs uncompiled, SAME dim_batch ===")
    print("Both sides identical except for torch.compile. Any difference here is compile.")
    hdr = f"{'dim_batch':>10} {'identity nc':>13} {'identity c':>13} {'max rel diff':>14}"
    print(hdr); print("-" * len(hdr))
    compile_effect = {}
    for db in batches:
        if (db, False) not in meta or (db, True) not in meta:
            continue
        d = rel_tensors(meta[(db, False)]["path"], meta[(db, True)]["path"])
        compile_effect[db] = d
        print(f"{db:>10} {meta[(db, False)]['identity_distance']:>13.6f} "
              f"{meta[(db, True)]['identity_distance']:>13.6f} {max(d.values()):>14.4e}")
    results["compile_vs_uncompiled"] = {str(k): {str(l): x for l, x in v.items()}
                                        for k, v in compile_effect.items()}

    # --- dim_batch, within each compile setting -------------------------------
    for compiled in (False, True):
        tag = "compiled" if compiled else "uncompiled"
        avail = [db for db in batches if (db, compiled) in meta]
        if len(avail) < 2:
            continue
        base = avail[0]
        print(f"\n=== dim_batch effect, {tag} (vs dim_batch={base}) ===")
        hdr = f"{'dim_batch':>10}" + "".join(f"{('L' + str(l)):>11}" for l in (0, 3, 11, 22))
        print(hdr); print("-" * len(hdr))
        store = {}
        for db in avail[1:]:
            d = rel_tensors(meta[(base, compiled)]["path"], meta[(db, compiled)]["path"])
            store[db] = d
            print(f"{db:>10}" + "".join(f"{d[l]:>11.3e}" for l in (0, 3, 11, 22)))
        results[f"dim_batch_{tag}"] = {str(k): {str(l): x for l, x in v.items()}
                                       for k, v in store.items()}

    # --- verdict --------------------------------------------------------------
    print("\n=== reading the result ===")
    flaky = set(results.get("flaky") or [])
    null = results.get("null") or 0.0
    # Judge compile only where the configuration reproduces itself. A cell that is
    # nondeterministic tells us nothing about compile-vs-uncompiled; it tells us the
    # cell is broken, which is a different and larger finding.
    sound = {db: max(v.values()) for db, v in compile_effect.items()
             if f"db{db}-c" not in flaky and f"db{db}-nc" not in flaky}
    broken = sorted(flaky)

    if sound:
        worst_sound = max(sound.values())
        print(f"  where configurations reproduce themselves, compile changes the Jacobian")
        print(f"  by at most {worst_sound:.3e} (dim_batch {sorted(sound)}), against a")
        print(f"  self-reproducibility null of {null:.3e} and a forward batch-dependence of"
              f"{f' {fwd:.3e}' if (fwd := (results.get('forward') or {}).get('worst')) else ''}.")
        if worst_sound < 5e-2:
            print("  -> COMPILE IS BENIGN at those settings: the change it makes is the same")
            print("     order as the bf16 variation already present without it.")

    if broken:
        print(f"\n  BUT these configurations do not reproduce themselves: {broken}")
        print("     That is not a compile-versus-uncompiled result. It is a cell returning")
        print("     a different answer each time it is run, which has to be explained before")
        print("     anything measured at that setting is used.")
        if any(b.startswith("db8-") for b in broken):
            print("\n     dim_batch=8 compiled is T15's exact configuration -- so this cannot")
            print("     be left open. Weigh it against what T15 actually produced: 0.531422")
            print("     twice, matching the published tensor's 0.531418, two runs agreeing to")
            print("     4.8e-4, a clean power law across six prompt counts, and a stopping-rule")
            print("     replay landing on exactly 233. A nondeterministically broken estimator")
            print("     produces none of that, let alone all of it.")
            print("     The via-fit comparison below is what separates the two.")

    # --- The discriminator: same configuration, reached through fit() ---------
    print("\n=== direct jacobian_for_prompt vs the same thing through fit() ===")
    print("Identical arguments either way. Our fits take the fit() path, so if the two")
    print("disagree the fault is in the call context and our lenses are unaffected.")
    hdr = (f"{'config':>12} {'direct':>13} {'via fit()':>13} {'expected':>11}")
    print(hdr); print("-" * len(hdr))
    viafit = {}
    for compiled in (True, False):
        for db in batches:
            if (db, compiled) not in meta:
                continue
            tag = f"db{db}-{'c' if compiled else 'nc'}"
            path = scratch / f"{tag}-viafit.pt"
            extra = ["--child", "via_fit", "--dim-batch", str(db), "--out", str(path)]
            if compiled:
                extra.append("--compiled")
            r = run_child(f"{tag}-viafit", extra, quiet=True)
            if not r:
                continue
            viafit[tag] = r
            print(f"{tag:>12} {meta[(db, compiled)]['identity_distance']:>13.6f} "
                  f"{r['identity_distance']:>13.6f} {'~0.5314':>11}")
    results["via_fit"] = viafit

    # Judge EVERY via-fit result against the expected value, not only those that
    # disagree with their direct counterpart. The previous logic filtered to configs
    # where the two paths differed by >0.05 and then asked whether that subset was
    # correct -- so a via-fit result that was wrong in the same way as its direct
    # counterpart, or wrong by a subtle 2%, was never examined at all. It concluded
    # "every via-fit result is correct" from a set of one.
    print("\n  every via-fit result against the expected value:")
    bad_viafit = {}
    for tag, r in sorted(viafit.items()):
        ident = r["identity_distance"]
        off = abs(ident - EXPECTED_IDENTITY) / EXPECTED_IDENTITY
        flag = "" if off <= IDENTITY_TOL else ("  <- SUBTLE" if off < 0.5 else "  <- CATASTROPHIC")
        if off > IDENTITY_TOL:
            bad_viafit[tag] = ident
        print(f"    {tag:>12} {ident:>12.6f}  {off:>8.2%} off{flag}")
    results["via_fit_bad"] = bad_viafit

    if bad_viafit:
        print(f"\n  -> COMPILE IS NOT SAFE, INCLUDING THROUGH fit(). {sorted(bad_viafit)}")
        print("     came back wrong on the path our fits actually take, so the earlier")
        print("     reading that only the direct call was affected does not hold.")
        print("     Note the two severities: a catastrophic mode (identity 5-8, obviously")
        print("     wrong) and a subtle one (~0.5435, 2.3% high) that looks plausible and")
        print("     would pass unnoticed. The subtle mode is the dangerous one.")
    else:
        print("\n  -> every via-fit result is correct. On this evidence the fit() path is")
        print("     sound, though the direct-call failure remains a real bug.")

    print("\n=== the validity gate this implies ===")
    print(f"  A sound run has prompt-1 identity_distance within {IDENTITY_TOL:.0%} of")
    print(f"  {EXPECTED_IDENTITY:.6f} for this model and corpus. Every fit already writes")
    print("  that number to row 1 of its *_convergence.csv, so it is a retroactive check")
    print("  on every lens we have ever produced, at zero cost. Check it before trusting")
    print("  any fit, and refit uncompiled if a lens matters and cannot be re-verified.")

    out = cfg.artifact_root / "measurements" / "dim-batch-diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dim_batch_diagnosis.json"
    path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nwrote {path}")

    if not args.keep:
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)
        print(f"removed {scratch} (pass --keep to retain the tensors)")


if __name__ == "__main__":
    main()
