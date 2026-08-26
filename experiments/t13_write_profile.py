"""T13 -- assemble this machine's profile from the bring-up measurements.

Spec: ``environment-setup-and-first-fit``, stage 3. Writes
``$JLENS_ARTIFACT_ROOT/profiles/$JLENS_MACHINE.toml``, which T15 reads for
``dim_batch`` and ``compile`` instead of being hand-tuned.

**Assembled from the measurement JSONs, not transcribed.** T10 and T11 already wrote
their raw results under ``$JLENS_ARTIFACT_ROOT/measurements/``; this reads them. A
profile typed out by hand is a second copy of the numbers that can drift from the
first, and the whole point of the profile is that stage 4 stops depending on someone
remembering a value.

Two judgement calls are baked in, both recorded in ``profile.py``'s docstring:

* ``dim_batch`` holds the **optimum**, not the ceiling. The plan expected the
  ceiling, on the reasoning that host transfers scale as ``1/dim_batch``; T11
  measured the opposite, so the largest value that fits is the *slowest* one that
  fits. ``dim_batch_basis`` records which kind of value it is.
* ``force_bos_effective`` is stored **per model**, though the plan listed it as a
  profile field alongside box facts. Whether ``force_bos`` fires is a tokenizer
  property, and Qwen3.5's ``false`` holds only because that checkpoint has no BOS at
  all -- a second model must not inherit it.

Run::

    uv run python experiments/t13_write_profile.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "src"))

from jlens_extensions import config as jx_config  # noqa: E402
from jlens_extensions.profile import (  # noqa: E402
    MachineProfile,
    ModelFacts,
    ProfileError,
    probe_host,
)

cfg = jx_config.load()
os.environ.update(cfg.hf_env())

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def read_measurement(task: str, filename: str) -> dict:
    path = cfg.artifact_root / "measurements" / task / filename
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun {task.upper()}'s driver first: "
            f"experiments/{task}_*.py -- the profile is assembled from the "
            f"measurement JSONs rather than typed in."
        )
    return json.loads(path.read_text())


def fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares intercept and slope. Plain Python -- it is four sums."""
    n = len(points)
    sum_x = sum(x for x, _ in points)
    sum_y = sum(y for _, y in points)
    sum_xy = sum(x * y for x, y in points)
    sum_xx = sum(x * x for x, _ in points)
    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        return (sum_y / n, 0.0)
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return ((sum_y - slope * sum_x) / n, slope)


def main() -> None:
    t10 = read_measurement("t10", "t10_force_bos.json")
    t11 = read_measurement("t11", "t11_dim_batch_sweep.json")

    best = t11.get("best_compiled")
    if not best:
        raise SystemExit("t11 JSON has no best_compiled entry; re-run the sweep")

    ok = [r for r in t11["results"] if r.get("compile") and not r.get("oom") and "error" not in r]
    oomed = [r["dim_batch"] for r in t11["results"] if r.get("oom")]
    intercept, slope = fit_line([(r["dim_batch"], r["peak_alloc_gb"]) for r in ok])

    host = probe_host(device_map="cuda")
    implied_ceiling = int((host.total_memory_gb - intercept) / slope) if slope > 0 else None

    facts = ModelFacts(
        dim_batch=best["dim_batch"],
        dim_batch_basis="optimum",
        compile=bool(best["compile"]),
        s_per_prompt=round(best["median_s_per_prompt"], 3),
        peak_alloc_gb=round(best["peak_alloc_gb"], 3),
        peak_reserved_gb=round(best["peak_reserved_gb"], 3),
        force_bos_effective=bool(t10["force_bos_effective"]),
        measured_over_prompts=best["n_prompts"],
        dim_batch_ceiling_measured=max(r["dim_batch"] for r in ok),
        dim_batch_oom_at=min(oomed) if oomed else None,
        memory_model_intercept_gb=round(intercept, 4),
        memory_model_slope_gb=round(slope, 4),
        notes=(
            f"dim_batch is the fastest measured, not the largest that fits: per-prompt "
            f"time rises with dim_batch on this box (implied ceiling ~{implied_ceiling} "
            f"from the memory model). force_bos_effective is false because this "
            f"checkpoint has no BOS token at all, not because the flag was ignored."
        ),
    )

    profile = MachineProfile(
        machine=cfg.machine,
        updated=date.today().isoformat(),
        host=host,
        models={MODEL_ID: facts},
    )
    path = profile.write(cfg.profile_path)

    print(f"wrote {path}\n")
    print(path.read_text())

    # Read it back through the same path T15 will use.
    reloaded = MachineProfile.load(cfg.profile_path)
    entry = reloaded.model(MODEL_ID)
    print(f"reload check: dim_batch={entry.dim_batch} compile={entry.compile} "
          f"({entry.dim_batch_basis}, ceiling measured at "
          f"{entry.dim_batch_ceiling_measured})")

    try:
        reloaded.model("Qwen/Qwen3.5-27B")
    except ProfileError:
        print("unknown-model guard: OK (raises rather than substituting)")
    else:
        raise SystemExit("unknown-model guard FAILED -- it returned something")


if __name__ == "__main__":
    main()
