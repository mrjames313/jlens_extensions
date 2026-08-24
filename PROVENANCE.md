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

**None at this commit.** Commit 1 is byte-identical to the source, deliberately: the
`git diff` between it and the patch commit that follows is the permanent record of
what we changed, with no `patches/` directory to maintain and no reference checkout
needed to answer "what did we change".

Four backports from upstream `anthropics/jacobian-lens` at `581d398` land in the next
commit — `checkpoint_every`, resume validation of `target_layer`/`skip_first`, the
`skip_first < 0` guard, and `save(dtype=)`. Every one is behaviour-preserving at its
default or opt-in.

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

_(T3 appends the resolved environment's deviations here.)_
