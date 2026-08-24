"""Guards for the four upstream backports applied to the vendored harness.

These four changes are claimed to be *behaviour-preserving at their default or
opt-in*, which is what lets us keep calling our fits artifact-comparable. T7
verifies the strongest form of that claim — the lens tensor is unchanged — by
re-running a smoke fit before and after the patch.

What T7 cannot reach is the **resume path**. ``fit_lens.py`` passes
``resume=True``, but T7 runs into a fresh ``--out_dir``, so no checkpoint exists
at start and the validation never executes. Since the resume hazard is the whole
reason the patch set was scheduled before any long fit, it is checked here
instead.

The tests drive ``fit()`` with a stub model and a pre-written checkpoint whose
``next_idx`` has already consumed the prompt list, so the loop body never runs.
That exercises the resume branch exactly, with no model, no GPU and no autograd.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "harness"

D_MODEL = 8
N_LAYERS = 4
SOURCE_LAYERS = [0, 1, 2]
TARGET_LAYER = 3
SKIP_FIRST = 16


@pytest.fixture
def jlens():
    """Import the vendored ``jlens`` from ``harness/`` the way ``fit_lens.py`` does.

    ``fit_lens.py`` resolves its bare ``import jlens`` through ``sys.path[0]`` —
    its own directory — and the package is deliberately not installed. So the
    only way to reach it from a test is to put ``harness/`` on the path.

    The teardown is load-bearing rather than tidiness: ``test_scaffold`` asserts
    that ``jlens`` is *not* importable, and ``find_spec`` consults ``sys.modules``
    before the path. Leaving the import in place would break that guard from a
    different file, with the failure depending on collection order.
    """
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
def model():
    """The only two attributes ``fit()`` touches when the loop body never runs."""
    return SimpleNamespace(n_layers=N_LAYERS, d_model=D_MODEL)


def write_checkpoint(jlens, path, **overrides):
    """Write a checkpoint that leaves ``fit()`` nothing to do.

    ``next_idx`` past the end of the prompt list means every prompt is skipped by
    the ``prompt_idx < next_idx`` guard, so ``fit()`` runs its resume validation
    and then falls straight through to building the lens.
    """
    state = {
        "jacobian_sum": {
            layer: torch.eye(D_MODEL, dtype=torch.float32) * 3.0 for layer in SOURCE_LAYERS
        },
        "n_done": 3,
        "next_idx": 3,
        "source_layers": list(SOURCE_LAYERS),
        "target_layer": TARGET_LAYER,
        "skip_first": SKIP_FIRST,
    }
    state.update(overrides)
    jlens.fitting._atomic_save(state, str(path))
    return state


def resume(jlens, model, path, **kwargs):
    return jlens.fit(
        model,
        ["p0", "p1", "p2"],
        source_layers=SOURCE_LAYERS,
        target_layer=TARGET_LAYER,
        skip_first=SKIP_FIRST,
        checkpoint_path=str(path),
        **kwargs,
    )


# --- Patch 2: resume validation of target_layer / skip_first ----------------


def test_resume_accepts_matching_metadata(jlens, model, tmp_path):
    path = tmp_path / "checkpoint.pt"
    write_checkpoint(jlens, path)

    lens = resume(jlens, model, path)

    assert lens.n_prompts == 3
    assert lens.source_layers == SOURCE_LAYERS
    # jacobian_sum was 3*I over n_done=3, so the mean is the identity.
    assert torch.allclose(lens.jacobians[0], torch.eye(D_MODEL))


@pytest.mark.parametrize(
    "key, bad_value",
    [
        ("target_layer", TARGET_LAYER - 1),
        ("skip_first", SKIP_FIRST + 1),
        ("source_layers", [0, 1]),
    ],
)
def test_resume_rejects_changed_metadata(jlens, model, tmp_path, key, bad_value):
    """The hazard this patch set exists to close.

    Before the backport only ``source_layers`` was validated, so resuming after
    changing ``target_layer`` silently mixed Jacobians taken against two
    different target layers into one running sum — no error, and a lens that
    looks fine. We know we will vary ``target_layer`` (the paper's penultimate
    default against the published ``null``), which is what makes this live.
    """
    path = tmp_path / "checkpoint.pt"
    write_checkpoint(jlens, path, **{key: bad_value})

    with pytest.raises(ValueError, match=key):
        resume(jlens, model, path)


def test_resume_tolerates_checkpoint_written_before_the_patch(jlens, model, tmp_path):
    """Upstream's ``if key in state`` guard, and the reason we kept it.

    A checkpoint written by the unpatched vendored copy stores only
    ``jacobian_sum / n_done / next_idx / source_layers``. Validating the two new
    keys unconditionally would turn every such checkpoint into a hard failure on
    resume. They stay unvalidated instead — which is also why landing this before
    any long fit exists is worth strictly more than landing it after.
    """
    path = tmp_path / "checkpoint.pt"
    state = write_checkpoint(jlens, path)
    del state["target_layer"], state["skip_first"]
    jlens.fitting._atomic_save(state, str(path))

    lens = resume(jlens, model, path)

    assert lens.n_prompts == 3


def test_checkpoint_state_round_trips_under_weights_only(jlens, tmp_path):
    """``fit()`` loads with ``weights_only=True``; the new keys must survive it."""
    path = tmp_path / "checkpoint.pt"
    write_checkpoint(jlens, path)

    state = torch.load(str(path), map_location="cpu", weights_only=True)

    assert state["target_layer"] == TARGET_LAYER
    assert state["skip_first"] == SKIP_FIRST


# --- Patch 1: checkpoint_every ----------------------------------------------


def test_checkpoint_every_defaults_to_every_prompt(jlens):
    """``fit_lens.py`` never passes ``checkpoint_every``, so the default is what
    makes the cadence patch behaviour-preserving for an artifact-comparable fit."""
    import inspect

    assert inspect.signature(jlens.fit).parameters["checkpoint_every"].default == 1


# --- Patch 3: skip_first guard ----------------------------------------------


def test_negative_skip_first_is_rejected(jlens):
    with pytest.raises(ValueError, match="skip_first"):
        jlens.fitting.valid_position_mask(128, skip_first=-1)


def test_default_mask_matches_the_published_position_count(jlens):
    """128 − 16 skipped − 1 final = 111, the ``n_valid_positions`` on every row of
    the published Qwen3.5-0.8B convergence trace. Anchors the guard against
    shifting the mask it guards."""
    mask = jlens.fitting.valid_position_mask(128)

    assert int(mask.sum()) == 111


# --- Patch 4: save(dtype=) --------------------------------------------------


def test_save_still_defaults_to_fp16(jlens, tmp_path):
    """The published artifacts are fp16 and our comparison floor of ~5e-4 assumes
    both sides are. Parameterising the dtype must not have moved the default."""
    path = tmp_path / "lens.pt"
    lens = jlens.JacobianLens(
        jacobians={0: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL
    )

    lens.save(str(path))

    stored = torch.load(str(path), map_location="cpu", weights_only=True)
    assert stored["J"][0].dtype is torch.float16


def test_save_honours_an_explicit_dtype(jlens, tmp_path):
    """What the parameter buys: our own lenses need not be quantised, which
    matters for the 27B lens we fit once and reuse."""
    path = tmp_path / "lens.pt"
    lens = jlens.JacobianLens(
        jacobians={0: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL
    )

    lens.save(str(path), dtype=torch.float32)

    stored = torch.load(str(path), map_location="cpu", weights_only=True)
    assert stored["J"][0].dtype is torch.float32
    assert jlens.JacobianLens.load(str(path)).n_prompts == 1
