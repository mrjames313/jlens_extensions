"""The soak's reporting, checked against the 0.8B numbers it was designed from.

`compile_soak.py` needs a GPU to take a draw, but everything that decides the verdict --
filtering, clustering, the thresholds -- is pure and testable here. That matters because
the driver's whole job is to be trusted in front of a 32-hour fit, and its failure mode
is a wrong verdict on real-looking numbers rather than a crash.

The values are the measured ones from f-2026-08-28-compile-miscompilation.
"""

from __future__ import annotations

import pytest


def load_driver():
    from conftest import load_driver as _load

    return _load("compile_soak")


soak = load_driver()

# Measured on gx10-ace5, 2026-08-28. linear-attn-only draws two sound variants.
REF_08B = 0.531523
LINEAR_ATTN_SOUND = [0.531430, 0.531469, 0.531430, 0.531469, 0.531430, 0.531469]
# Every observed failure sat at least 2.3% from the reference.
MISCOMPILED = 0.519000


def test_sound_draws_report_their_two_variants_and_pass():
    summary = soak.report(LINEAR_ATTN_SOUND, REF_08B, 0.01)
    assert summary["n_out_of_band"] == 0
    assert summary["n_distinct_values"] == 2, "linear-attn-only has exactly two variants"
    assert {c["count"] for c in summary["clusters"]} == {3}
    assert soak.verdict(summary, "linear-attn").startswith("SOUND")


def test_a_miscompiled_draw_is_excluded_from_the_variant_structure():
    """The bug this guards: clustering an out-of-band value invents a third variant.

    `cluster_variants` documents that a miscompiled reading is not a variant and must be
    filtered first. If it leaks in it both inflates the variant count and, per that
    docstring, contaminates any null later computed from the grouping.
    """
    values = LINEAR_ATTN_SOUND + [MISCOMPILED]
    summary = soak.report(values, REF_08B, 0.01)
    assert summary["n_out_of_band"] == 1
    assert summary["n_distinct_values"] == 2, "the bad draw must not become a variant"
    assert summary["n_sound_draws"] == 6
    assert soak.verdict(summary, "linear-attn").startswith("UNSOUND")


def test_uncompiled_reference_is_a_single_repeating_value():
    summary = soak.report([REF_08B] * 12, REF_08B, 0.01)
    assert summary["n_distinct_values"] == 1
    assert summary["spread_rel"] == 0.0
    assert soak.verdict(summary, "none").startswith("SOUND")


def test_reduced_soak_says_it_did_not_bound_the_rate():
    """A 6-draw run must not read as equivalent to a 20-draw one."""
    reduced = soak.verdict(soak.report(LINEAR_ATTN_SOUND, REF_08B, 0.01), "linear-attn")
    assert "value structure rather than bounding the failure rate" in reduced

    full = soak.verdict(soak.report(LINEAR_ATTN_SOUND * 4, REF_08B, 0.01), "linear-attn")
    assert "value structure rather than bounding" not in full


def test_scatter_within_tolerance_is_flagged_rather_than_passed():
    """Tight-but-not-exact is the shape a new model's trouble would take.

    All in band, so the failure count says nothing; but a sound variant repeats
    *exactly*, so eight distinct readings from eight draws is scatter.
    """
    scattered = [REF_08B * (1 + i * 1e-4) for i in range(8)]
    summary = soak.report(scattered, REF_08B, 0.01)
    assert summary["n_out_of_band"] == 0
    assert soak.verdict(summary, "auto").startswith("SUSPECT")


def test_without_a_reference_the_verdict_refuses_to_claim_soundness():
    summary = soak.report(LINEAR_ATTN_SOUND, None, 0.01)
    assert summary["filtered"] is False
    assert summary["reference"] is None
    assert soak.verdict(summary, "auto").startswith("UNVERIFIED")


def test_worst_offset_is_measured_over_every_draw_not_just_sound_ones():
    summary = soak.report(LINEAR_ATTN_SOUND + [MISCOMPILED], REF_08B, 0.01)
    assert summary["worst_rel_offset"] == pytest.approx(
        abs(MISCOMPILED - REF_08B) / REF_08B
    )
    assert summary["worst_rel_offset"] > 0.02
