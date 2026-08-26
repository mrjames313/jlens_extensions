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

## Drivers

| Driver | Task | Produces |
|---|---|---|
| `t9_gamma_spread.py` | T9 | The γ spread for a model, the top-k dictionary overlap between the corrected and the paper's construction, and the vector/subspace divergence at a fixed token set. |
| `t9_derivation_check.py` | T9 / any new model | Per-model validation: the norm's gain convention, and whether our dictionary construction reproduces the model's readout at equal precision. |
| `t10_force_bos.py` | T10 | Whether `from_hf(force_bos=True)` changes the encoded ids on this tokenizer, or silently no-ops. |
| `t11_dim_batch_sweep.py` | T11 | The `dim_batch` ceiling, per-prompt wall-clock and peak allocator memory, swept in fresh subprocesses. Projects the 233-prompt fit against T14's gate. |
| `t12_diagnostic_cost.py` | T12 | The share of per-prompt wall-clock taken by `mean_rel_change` and `identity_distance`, by importing a text-patched copy of `fitting.py` from a temp dir. Never edits the repo. |
| `t13_write_profile.py` | T13 | Assembles `$JLENS_ARTIFACT_ROOT/profiles/$JLENS_MACHINE.toml` from the T10/T11 measurement JSONs, fits the memory model, and reads it back through the path T15 uses. |
