"""Resume must *continue* a running sum, not refit from scratch.

T19 measures how the run-to-run envelope scales with prompt count, and it gets every
prompt count out of a single pass by exploiting exactly this: a lens at n prompts is
the running mean of the first n, so fitting to 5, resuming to 10, resuming to 20 and
so on should give the same lenses as five independent fits, while computing each
prompt once. That turns 165 prompts of compute into 60.

If the assumption is wrong the driver does not crash -- it silently produces lenses
over the wrong prompt sets, and every exponent fitted from them is wrong in a way
nothing downstream would catch. So it is asserted here against the real `fit()`, with
a stubbed per-prompt estimator so no GPU or model is involved.

This also covers the resume path on real accumulation rather than the
already-finished checkpoint used in `test_harness_backports.py`, which returns before
the loop body ever runs.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "harness"

D_MODEL, N_LAYERS = 4, 3
SOURCE_LAYERS = [0, 1]
TARGET_LAYER = 2


@pytest.fixture
def jlens():
    sys.path.insert(0, str(HARNESS))
    preexisting = set(sys.modules)
    try:
        yield importlib.import_module("jlens")
    finally:
        sys.path.remove(str(HARNESS))
        for name in set(sys.modules) - preexisting:
            if name == "jlens" or name.startswith("jlens."):
                del sys.modules[name]
        importlib.invalidate_caches()


@pytest.fixture
def deterministic_estimator(jlens, monkeypatch):
    """Replace the per-prompt Jacobian with a deterministic function of the prompt.

    Each prompt contributes a distinct, reproducible matrix, so a lens over the first
    n prompts has a value that can be predicted exactly -- which is what lets the test
    distinguish "continued" from "refitted" rather than merely "ran".
    """

    def fake(model, prompt, source_layers, **kwargs):
        seed = int(prompt.split("-")[1])
        per_layer = {
            layer: torch.full((D_MODEL, D_MODEL), float(seed + layer * 100))
            for layer in source_layers
        }
        return per_layer, 128, 111

    monkeypatch.setattr(jlens.fitting, "jacobian_for_prompt", fake)
    return fake


@pytest.fixture
def model():
    return SimpleNamespace(n_layers=N_LAYERS, d_model=D_MODEL)


def _fit(jlens, model, prompts, path, resume):
    return jlens.fit(
        model, prompts,
        source_layers=SOURCE_LAYERS,
        target_layer=TARGET_LAYER,
        dim_batch=2,
        checkpoint_path=str(path),
        checkpoint_every=1,
        resume=resume,
    )


PROMPTS = [f"p-{i}" for i in range(12)]


def test_staged_resume_matches_independent_fits(jlens, model, deterministic_estimator, tmp_path):
    """The property T19 depends on: staged lenses equal independently fitted ones."""
    staged_path = tmp_path / "staged.pt"
    staged = {}
    for i, n in enumerate((3, 6, 12)):
        staged[n] = _fit(jlens, model, PROMPTS[:n], staged_path, resume=(i > 0))

    for n, lens in staged.items():
        solo_path = tmp_path / f"solo-{n}.pt"
        solo = _fit(jlens, model, PROMPTS[:n], solo_path, resume=False)
        assert lens.n_prompts == solo.n_prompts == n
        for layer in SOURCE_LAYERS:
            assert torch.equal(lens.jacobians[layer], solo.jacobians[layer]), \
                f"staged lens at n={n} differs from an independent fit at layer {layer}"


def test_each_prompt_is_computed_exactly_once(jlens, model, monkeypatch, tmp_path):
    """The saving that makes the design cheap. Refitting would recompute prompts."""
    seen = []

    def counting(model_, prompt, source_layers, **kwargs):
        seen.append(prompt)
        return ({l: torch.ones(D_MODEL, D_MODEL) for l in source_layers}, 128, 111)

    monkeypatch.setattr(jlens.fitting, "jacobian_for_prompt", counting)
    path = tmp_path / "c.pt"
    for i, n in enumerate((3, 6, 12)):
        _fit(jlens, model, PROMPTS[:n], path, resume=(i > 0))

    assert len(seen) == 12, f"expected 12 prompt computations, got {len(seen)}"
    assert seen == PROMPTS, "prompts were computed out of order or repeated"


def test_the_lens_is_a_running_mean_over_the_first_n(jlens, model,
                                                     deterministic_estimator, tmp_path):
    """Anchors the arithmetic: prompt i contributes i, so the mean over the first n
    is (n-1)/2. If resume double-counted or dropped a prompt this would move."""
    path = tmp_path / "m.pt"
    lens = None
    for i, n in enumerate((3, 6)):
        lens = _fit(jlens, model, PROMPTS[:n], path, resume=(i > 0))

    expected = sum(range(6)) / 6.0
    assert lens.jacobians[0][0, 0].item() == pytest.approx(expected)


def test_resume_without_a_checkpoint_starts_clean(jlens, model,
                                                  deterministic_estimator, tmp_path):
    """resume=True on a missing checkpoint must fit from scratch, not fail --
    it is how the first stage behaves if the path was cleared."""
    lens = _fit(jlens, model, PROMPTS[:3], tmp_path / "absent.pt", resume=True)
    assert lens.n_prompts == 3


def test_a_stale_checkpoint_is_what_resume_false_guards_against(jlens, model,
                                                               deterministic_estimator,
                                                               tmp_path):
    """The failure mode the driver clears its checkpoint to avoid: with resume=True
    and a checkpoint already at n, a shorter fit returns the *stale* lens rather
    than refitting. Silent, and it would look like a result."""
    path = tmp_path / "stale.pt"
    _fit(jlens, model, PROMPTS[:6], path, resume=False)

    shorter = _fit(jlens, model, PROMPTS[:3], path, resume=True)

    assert shorter.n_prompts == 6, "stale state carried forward, as documented"
