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
| `dim_batch_neutrality.py` | Whether `dim_batch` changes the fitted tensor, tested against T15's pair as a pure-noise null so the variable is isolated on one box. Reports per-layer excess over that null and whether it is depth-independent — the shape T16's unexplained residual has. |
| `envelope_vs_n.py` | The run-to-run envelope against prompt count, at one fixed configuration (fp32, compiled, `dim_batch=8`), and a per-layer scaling exponent. Re-measures what T18 established through fp16 and at a different execution config. Gets six prompt counts out of a single 60-prompt pass by resuming the running sum. |

## Drivers

| Driver | Task | Produces |
|---|---|---|
| `t9_gamma_spread.py` | T9 | The γ spread for a model, the top-k dictionary overlap between the corrected and the paper's construction, and the vector/subspace divergence at a fixed token set. |
| `t9_derivation_check.py` | T9 / any new model | Per-model validation: the norm's gain convention, and whether our dictionary construction reproduces the model's readout at equal precision. |
| `t10_force_bos.py` | T10 | Whether `from_hf(force_bos=True)` changes the encoded ids on this tokenizer, or silently no-ops. |
| `t11_dim_batch_sweep.py` | T11 | The `dim_batch` ceiling, per-prompt wall-clock and peak allocator memory, swept in fresh subprocesses. Projects the 233-prompt fit against T14's gate. |
| `t12_diagnostic_cost.py` | T12 | The share of per-prompt wall-clock taken by `mean_rel_change` and `identity_distance`, by importing a text-patched copy of `fitting.py` from a temp dir. Never edits the repo. |
| `t15_validation_fit.py` | T15 | The pinned 233-prompt validation fit, run twice sequentially at the production configuration. Reads `dim_batch` / `compile` from the machine profile, stores at fp32, and writes a provenance sidecar beside each lens. |
| `t16_compare.py` | T16 | A1's negative control (every layer matched against every published layer), the fp16 storage floor measured rather than assumed, and the four comparison axes: the a-vs-b production-config envelope (fp32, and again through fp16 for comparability with T18), ours vs published per layer against the floor that binds each, `identity_distance` recomputed from the tensor, and the convergence trace. Also replays the published early-stopping rule on our trace, recovering the `prompts_fitted` axis that pinning removes. |
| `t13_write_profile.py` | T13 | Assembles `$JLENS_ARTIFACT_ROOT/profiles/$JLENS_MACHINE.toml` from the T10/T11 measurement JSONs, fits the memory model, and reads it back through the path T15 uses. |
