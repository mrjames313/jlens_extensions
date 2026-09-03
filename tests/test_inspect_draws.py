"""The norm table must not pool across groups that disagree."""

from __future__ import annotations

import pytest

import jlens_extensions.fetch  # noqa: F401


@pytest.fixture
def driver():
    from conftest import load_driver
    return load_driver("inspect_draws")


def _summary(rows):
    return {"model": "Qwen/Qwen3.5-4B",
            "rows": [{"label": lbl, "norms": norms, "layers": sorted(norms),
                      "identity_distance": 0.5, "n_valid": 111, "seq_len": 128,
                      "nonfinite": {}}
                     for lbl, norms in rows]}


def test_norms_are_reported_per_group_not_pooled(driver):
    """4B p4's three groups sit at 337/447/833 at L0; their mean describes none."""
    summary = _summary([("lin/v0", {0: 337.4, 30: 57.290}),
                        ("lin/v0", {0: 337.4, 30: 57.290}),
                        ("lin/v1", {0: 447.4, 30: 57.296}),
                        ("none/v0", {0: 833.2, 30: 57.304})])
    table = driver.by_group(summary)
    assert set(table) == {"lin/v0", "lin/v1", "none/v0"}
    assert table["lin/v0"][0] == pytest.approx(337.4)
    assert table["none/v0"][0] == pytest.approx(833.2)


def test_the_spread_closes_toward_the_last_layer(driver, capsys):
    """The shape that makes identity_distance blind: agreement exactly where it reads."""
    rows = [("lin/v0", {i: v for i, v in enumerate([337.4] + [80.6] * 29 + [57.290])}),
            ("none/v0", {i: v for i, v in enumerate([833.2] + [111.3] * 29 + [57.304])})]
    driver.compare_norms(_summary(rows), _summary(rows))
    out = capsys.readouterr().out
    assert "spread (max/min)" in out
    assert "2.47x" in out, "L0 spread must be visible"
    assert "1.00x" in out, "and the last layer's agreement with it"
