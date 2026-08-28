"""Tests for the single fit-command builder.

Its whole reason to exist is that the gate cannot be omitted by accident. T15 and the
dim_batch probe each built their own argv and neither carried it, which is how a
defence that depends on every caller remembering actually behaves.

So the property under test is not "the command is right" but "a command without a gate
cannot be built silently".
"""

import pytest

from jlens_extensions.fitcmd import UNGATED, MissingGateError, build_fit_command

BASE = dict(fit_lens_path="/h/fit_lens.py", model_id="Qwen/Qwen3.5-0.8B",
            out_dir="/scratch/out", n_prompts=233, dim_batch=8)


def cmd(**kw):
    return build_fit_command(**{**BASE, **kw})


# --- the point of the module ------------------------------------------------


def test_gate_is_required_not_defaulted():
    """Omitting it is a TypeError from the signature, not a silent ungated fit."""
    with pytest.raises(TypeError):
        build_fit_command(**BASE)


def test_none_is_rejected_with_an_actionable_message():
    with pytest.raises(MissingGateError) as exc:
        cmd(gate_identity=None)
    msg = str(exc.value)
    assert "probe_reference_identity" in msg
    assert "UNGATED" in msg
    assert "30-50%" in msg


def test_opting_out_must_be_explicit_and_greppable():
    out = cmd(gate_identity=UNGATED)
    assert "--gate_identity" not in out


def test_a_stray_string_is_not_mistaken_for_an_opt_out():
    with pytest.raises(MissingGateError, match="must be a number or UNGATED"):
        cmd(gate_identity="0.5314")


# --- what the command carries -----------------------------------------------


def test_gate_reaches_the_command_line():
    out = cmd(gate_identity=0.5314)
    assert "--gate_identity" in out
    assert float(out[out.index("--gate_identity") + 1]) == pytest.approx(0.5314)
    assert float(out[out.index("--gate_tol") + 1]) == pytest.approx(0.01)


def test_compile_policy_defaults_to_auto():
    out = cmd(gate_identity=0.5314)
    assert out[out.index("--compile_blocks") + 1] == "auto"


def test_no_compile_drops_both_compile_and_gate_flags():
    """Uncompiled has never failed, and --gate_identity would have nothing to guard."""
    out = cmd(gate_identity=0.5314, no_compile=True)
    assert "--no_compile" in out
    assert "--compile_blocks" not in out
    assert "--gate_identity" not in out


def test_hf_cache_dir_is_never_passed():
    """fit_lens.py rmtree's it in a finally, which would re-download every weight."""
    assert "--hf_cache_dir" not in cmd(gate_identity=0.5314)


def test_early_stopping_is_never_passed():
    """Validation fits pin --n_prompts and disable early stopping by construction."""
    assert "--stop_at_delta" not in cmd(gate_identity=0.5314)


def test_pinned_recipe_values_are_present():
    out = cmd(gate_identity=0.5314)
    for flag, value in (("--n_prompts", "233"), ("--dim_batch", "8"),
                        ("--max_seq_len", "128"), ("--dtype", "bfloat16"),
                        ("--save_dtype", "float32"), ("--device_map", "cuda"),
                        ("--dataset_config", "wikitext-103-raw-v1")):
        assert out[out.index(flag) + 1] == value


def test_float_values_survive_the_round_trip():
    """repr() rather than str() so a reference is not truncated into a false failure."""
    out = cmd(gate_identity=0.5314267891, gate_tol=0.005)
    assert float(out[out.index("--gate_identity") + 1]) == 0.5314267891
    assert float(out[out.index("--gate_tol") + 1]) == 0.005


def test_extra_args_are_appended():
    out = cmd(gate_identity=0.5314, extra=["--target_layer", "-2"])
    assert out[-2:] == ["--target_layer", "-2"]
