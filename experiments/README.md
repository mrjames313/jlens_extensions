# experiments/

One-shot measurement drivers. Each produces numbers that get written up in a kb
finding and, where they describe the hardware rather than the model, into the
machine profile.

**These are committed on purpose.** The repository layout argument for vendoring
the harness — that the code producing an artifact has to be answerable from a
fresh clone, rather than reassembled per-clone against an unpinned checkout —
applies to measurements too. A number that reaches a finding and gates a decision
should have a runnable, versioned provenance. So drivers live here rather than
under `$JLENS_ARTIFACT_ROOT`, and their *outputs* live under the artifact root
because those are large and machine-specific.

## Boundaries

- **Not packaged.** `[tool.setuptools.packages.find]` searches only `src/`, and
  `experiments*` is in its `exclude` alongside `harness*`. A driver is a script,
  never an import target.
- **Reusable logic belongs in `src/jlens_extensions/`, not here.** If two drivers
  would share a function, it is library code. The corollary: a driver should be
  mostly corpus wrangling, a loop, and a table.
- **Never assemble a `fit_lens.py` argv by hand.** Use
  `jlens_extensions.fitcmd.build_fit_command`, which requires the compile gate rather
  than defaulting it. Two drivers previously built their own and neither carried the
  gate — which is what a defence relying on every caller remembering actually does.
- **Never edit `harness/` to take a measurement.** The fork has to stay
  artifact-comparable. Where a measurement genuinely requires perturbing the
  harness — timing a diagnostic by removing it, say — that edit is a local
  scaffold, reverted and not committed, and the driver says so in its docstring.
- **Outputs go to `$JLENS_ARTIFACT_ROOT/measurements/<task>/`** as JSON, so a
  write-up can be regenerated without a re-run.

## Follow-on measurements

Not spec tasks. Named for what they measure rather than given a task number, so a
later spec's numbering cannot collide with them.

| Driver | Produces |
|---|---|
| `probe_gate_identity.py` | The reference prompt-1 `identity_distance` for a model, measured uncompiled and written into the machine profile. Run once per model per box; `t15_validation_fit.py` refuses to run a compiled fit without it. |
| `dim_batch_diagnosis.py` | **Diagnostic, run this before reading `dim_batch_neutrality.py`'s output.** Separates three explanations for that driver's implausible result — a batch-size-dependent forward, a `torch.compile` specialisation, or the estimator — on one prompt with no fit. |
| `dim_batch_neutrality.py` | Whether `dim_batch` changes the fitted tensor, tested against T15's pair as a pure-noise null so the variable is isolated on one box. Reports per-layer excess over that null and whether it is depth-independent — the shape T16's unexplained residual has. |
| `cross_path_scaling.py` | Whether a difference between two *execution paths* averages down at the same rate as run-to-run noise — the assumption every single-prompt prediction rests on. Fits an exponent per layer for within-variant, cross-variant and cross-configuration differences, with the within-variant fit as a control against the measured 0.473. |
| `offset_profile.py` | Per-layer Frobenius offsets between compile **variants**, between compile **configurations**, and between **`dim_batch`** values — three questions that are one experiment, since each subtracts a within-group noise null from a between-group difference. Feeds A1's unmeasured variant profile and the published-artifact bound. Supersedes `dim_batch_neutrality.py`, whose only run was miscompiled. Per-model via `--model`; `--out-tag` keeps several runs (one per prompt, say) from overwriting each other, which is how `t17_envelope.py` drives it. |
| `inspect_draws.py` | Tensor health for a set of saved draws — per-layer norms, non-finite entries, and `n_valid` — for when an `offset_profile.py` offset looks *wrong* rather than merely large. Every statistic that driver reports is a ratio of norms, and a ratio hides both of its terms; this reads the Jacobians directly. `--against` a manifest from a prompt that behaved makes the comparison legible. |
| `envelope_vs_n.py` | The run-to-run envelope against prompt count, at one fixed configuration (fp32, compiled, `dim_batch=8`), and a per-layer scaling exponent. Re-measures what T18 established through fp16 and at a different execution config. Gets six prompt counts out of a single 60-prompt pass by resuming the running sum. |

## Drivers

| Driver | Task | Produces |
| `compile_soak.py` | Whether a compile policy is sound on a given model: N draws, one process each (the miscompilation is per-process), reading prompt-1 `identity_distance` through the same `measure_prompt1_identity` the gate reference uses. Reports the discrete value structure of the in-band draws — a sound configuration repeats *exactly* within a variant — and refuses to claim soundness without a stored `gate_identity` to compare against. `--draws 6` checks the structure; `--draws 20` bounds the rate. |
| `add_model_profile.py` | A machine-profile entry for a second model on a box already brought up, where `t13_write_profile.py` does not apply because there is no `dim_batch` sweep to assemble from. Records `dim_batch`/`s_per_prompt`/`peak_*` as **projections** — `measured_over_prompts=0` and a basis string that names the projection — and refuses to project `force_bos_effective`, which it reads from T10. |
|---|---|---|
| `t9_gamma_spread.py` | T9 | The γ spread for a model, the top-k dictionary overlap between the corrected and the paper's construction, and the vector/subspace divergence at a fixed token set. |
| `t9_derivation_check.py` | T9 / any new model | Per-model validation: the norm's gain convention, and whether our dictionary construction reproduces the model's readout at equal precision. |
| `t10_force_bos.py` | T10 | Whether `from_hf(force_bos=True)` changes the encoded ids on this tokenizer, or silently no-ops. |
| `t11_dim_batch_sweep.py` | T11 | The `dim_batch` ceiling, per-prompt wall-clock and peak allocator memory, swept in fresh subprocesses. Projects the 233-prompt fit against T14's gate. |
| `t12_diagnostic_cost.py` | T12 | The share of per-prompt wall-clock taken by `mean_rel_change` and `identity_distance`, by importing a text-patched copy of `fitting.py` from a temp dir. Never edits the repo. |
| `t15_validation_fit.py` | T15 | The pinned 233-prompt validation fit, run twice sequentially at the production configuration. Reads `dim_batch` / `compile` from the machine profile, stores at fp32, and writes a provenance sidecar beside each lens. |
| `t16_compare.py` | T16 | A1's negative control (every layer matched against every published layer), the fp16 storage floor measured rather than assumed, and the four comparison axes: the a-vs-b production-config envelope (fp32, and again through fp16 for comparability with T18), ours vs published per layer against the floor that binds each, `identity_distance` recomputed from the tensor, and the convergence trace. Also replays the published early-stopping rule on our trace, recovering the `prompts_fitted` axis that pinning removes. |
| `t17_envelope.py` | T17 | The per-layer run-to-run envelope for a rung that was fitted **once**, predicted from single-prompt draws instead of a second full fit (~17 h at 4B). Runs `offset_profile.py` once per prompt — one subprocess per prompt, so the eager analysis never holds more than one prompt's draws, which at 4B's 812 MB/draw is the difference between 6.5 GB and 32 GB — then averages the within-group noise across prompts and divides by `n**alpha`. Emits the JSON `t16_compare.py --envelope` consumes, carrying the exponent's measured range and, where `n` falls outside it, which direction the error points. |
| `t13_write_profile.py` | T13 | Assembles `$JLENS_ARTIFACT_ROOT/profiles/$JLENS_MACHINE.toml` from the T10/T11 measurement JSONs, fits the memory model, and reads it back through the path T15 uses. |
