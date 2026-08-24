# Provenance

## The vendored harness slice (`harness/`)

`harness/` is a **fork**, not a dependency. It is run in place and never installed,
so `fit_lens.py`'s bare `import jlens` resolves through `sys.path[0]` — its own
directory — to `harness/jlens/`, exactly as Neuronpedia's driver intends.

| | |
|---|---|
| Source repo | `https://github.com/hijohnnylin/neuronpedia` |
| Source commit | `93b8cae` — "fix: lint ts and python" |
| Source path | `utils/neuronpedia-utils/neuronpedia_utils/jlens/` |
| Copied | 2026-08-23 |
| Licence | Apache-2.0, © 2026 Anthropic PBC — SPDX headers intact, licence text at `harness/LICENSE` |

`93b8cae` is the **last commit to touch that path**. Our reference checkout sits at
`53db9fb`, which is later but leaves the slice untouched, so the copy was taken from
the working tree and verified byte-identical with `diff -r` rather than by checking
out the older commit.

### What was copied

`fit_lens.py` (421 lines) and `jlens/` (7 modules, 1100 lines) — 1521 lines, ~72 KB —
plus `LICENSE`, which Apache-2.0 §4(a) requires travel with the source.

### What was left behind, and why

| Not copied | Why |
|---|---|
| `run-all-fit-lens.py` | Orchestrates the 38-model fleet ladder; we fit one model at a time. It can follow if we do the ladder. Note it is also the only thing that writes `config.yaml`, so **our runs produce no `config.yaml`** — provenance is carried by the sidecar instead. |
| `pyproject.toml` | Theirs declares a script environment (`[tool.uv] package = false`). Ours is the root one here, which declares a real package. The dependency list was seeded from theirs verbatim. |
| `uv.lock` | 556 KB resolved for their platform against an explicit `pytorch-cu128` index; the GB10 is ARM64 Grace-Blackwell. We resolve our own. |
| `README.md` | Describes their repo layout, not ours. Its attribution is reproduced above. |

## Changes to the vendored slice

Four backports from upstream `anthropics/jacobian-lens` at `581d398`, applied on top of
the byte-identical vendoring commit `4fc2167`. `git diff 4fc2167 -- harness/` is the
permanent record of what we changed — no `patches/` directory to maintain, and no
reference checkout needed to answer "what did we change".

| Item | File | What |
|---|---|---|
| `checkpoint_every` | `jlens/fitting.py` | Cadence control on `write_checkpoint()`'s two call sites |
| Resume validation | `jlens/fitting.py` | Store `target_layer` / `skip_first` in the checkpoint **and** check them on resume |
| `skip_first` guard | `jlens/fitting.py` | Reject negative `skip_first` in `valid_position_mask` |
| `save(dtype=)` | `jlens/lens.py` | Parameterise save precision instead of hardcoding fp16 |

**Every one is behaviour-preserving at its default or opt-in** — `checkpoint_every=1` is
the vendored cadence, `dtype=torch.float16` is the vendored precision, and both
validations fire only on input that is already wrong. The estimator is untouched, so an
artifact-comparable fit cannot change. That is verified rather than asserted: a
before/after smoke fit must produce an identical lens tensor (T7), and the resume path
T7 cannot reach is covered by `tests/test_harness_backports.py`.

Upstream's `if key in state` guard on the resume check **was** kept. It tolerates
checkpoints written before those keys were stored — exactly the class of checkpoint our
own pre-patch runs produce — so any such checkpoint stays unprotected forever, which is
why landing this before long fits exist is worth strictly more than landing it after.

**One deliberate deviation from upstream: `checkpoint_every=None`.** Upstream reads
`None` as "skip per-iteration writes and save once at the end", which requires a third
`write_checkpoint()` call after the loop. We did not port that call, so for us `None`
disables checkpoint writes entirely. The reason is local: `fit_lens.py` deletes the
checkpoint as soon as the lens is saved, so a trailing write would be written and
immediately unlinked — negligible at 0.8B, ~6.7 GB at 27B, and pure waste at both. If a
caller ever wants a durable end-of-fit checkpoint, add the trailing call rather than
reaching for `None`.

**Deferred, with triggers.** `_check_layer_indices()` lands with the first patch that
passes `source_layers` explicitly — expected before the 27B fit — so the guard arrives
alongside the argument it guards. `apply(positions=…)` lands at the first
multi-position readout. Neither is in this commit.

**What stays Neuronpedia's, and must not be "fixed" from upstream:**
`metrics_callback` / `FitProgress`, `identity_distance`, and `mean_rel_change`. The
published `prompts_fitted` and the early-stopping rule are *defined* by their
`mean_rel_change`, and upstream's same-named statistic computes something different.
Porting it would break the comparability the vendored copy was chosen for.

## Deviations from Neuronpedia's dependency set

The seven runtime dependencies above are verbatim. Two deliberate differences in the
surrounding configuration:

- **`pytest>=8.0` added as a `dev` extra.** Not a runtime dependency; does not affect
  a fit.
- **Neuronpedia's `[[tool.uv.index]] pytorch-cu128` and `[tool.uv.sources] torch`
  routing is not carried over.** Their config routes `torch` to an explicit CUDA 12.8
  index under `sys_platform == 'linux'`, which selects the GB10 — but the GB10 is
  ARM64, not x86, so the wheel that index serves may not be the one we need. Resolving
  this is T3's job, and whatever it settles on gets recorded below.

### Resolved environment — T3, 2026-08-24, machine `gx10-ace5`

Resolved with `uv sync --extra dev` against PyPI with **no index configuration at all**.
PyPI served a CUDA-enabled `linux_aarch64` wheel directly, so the `pytorch-cu128`
routing we declined to inherit turned out to be unnecessary rather than merely wrong
for us. Python 3.12.3.

Against Neuronpedia's `uv.lock`:

| package | Neuronpedia | ours | |
|---|---|---|---|
| accelerate | 1.14.0 | 1.14.0 | identical |
| tokenizers | 0.22.2 | 0.22.2 | identical |
| sentencepiece | 0.2.1 | 0.2.2 | patch |
| datasets | 5.0.0 | 5.0.1 | patch |
| tqdm | 4.68.2 | 4.70.0 | minor |
| transformers | 5.11.0 | 5.15.1 | minor |
| numpy | 2.2.6 / 2.4.6 | 2.5.2 | minor |
| **torch** | **2.11.0+cu128** | **2.13.0+cu130** | **two minors + a CUDA major** |

`torch` is the only substantive difference, and it is **forced rather than chosen**:
the GB10 is compute capability `sm_121` behind a CUDA 13.0 driver (580.173.02).
Neuronpedia's doubled `numpy` and `torch` rows are uv locking per-platform
resolutions; the `+cu128` branch is the one their `sys_platform` marker selects.
(`importlib.metadata` reports our torch as `2.13.0` while `torch.__version__` carries
the local segment `+cu130` — same wheel.)

**What their lock does and does not establish.** It is a single commit dated
2026-07-06 — **25 days after** the Qwen3.5-0.8B lens was fit on 2026-06-11 — and the
published `config.yaml` records no library versions whatsoever. So the left column is
the best available evidence of their environment, not a record of what produced the
artifact; fit-time `transformers` was `<= 5.11.0` and could have been older. That
asymmetry is precisely what our own provenance sidecar exists to avoid on this side.

**If T16's comparison disagrees beyond the ~5e-4 fp16 floor, start here, and suspect
`torch` first.** The estimator is our vendored copy and byte-identical to theirs, so a
numerical difference cannot originate in the lens math — it has to enter through the
libraries that load the weights, tokenize the corpus, or execute the kernels.
