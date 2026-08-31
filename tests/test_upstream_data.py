"""The vendored data is present, complete, and reachable without the reference checkout.

The point of these tests is the *fresh clone* property. T1 vendored upstream's ``data/``
precisely so that an experiment citing ``ignition.json`` is re-runnable by someone who
has never cloned ``anthropics/jacobian-lens``. A test that only checked
``upstream_data.load`` returns JSON would pass just as happily if the loader were
quietly reading the per-clone checkout, so these assert on the vendored path itself.
"""

from __future__ import annotations

import json

import pytest

from jlens_extensions import upstream_data as ud

# The two files spec `workspace-band-location` actually consumes: ignition drives
# apparatus operation 1 (T11), multihop drives the C3 precondition (T4).
REQUIRED = [("experiments", "ignition"), ("evaluations", "lens-eval-multihop")]


def test_data_root_is_in_the_repo_not_the_reference_checkout():
    """The vendored copy must live under the repo, never under `reference/`.

    `reference/` is gitignored and per-clone. If DATA_ROOT ever resolved there the
    whole vendoring would be undone while every other test still passed.
    """
    assert ud.DATA_ROOT == ud.REPO / "data"
    assert ud.DATA_ROOT.is_dir()
    assert "reference" not in ud.DATA_ROOT.parts


@pytest.mark.parametrize("kind,name", REQUIRED)
def test_required_files_load(kind, name):
    payload = ud.load(kind, name)
    assert payload, f"{kind}/{name}.json parsed but is empty"


def test_suffix_is_optional():
    assert ud.load("experiments", "ignition") == ud.load("experiments", "ignition.json")


def test_ignition_carries_the_fields_the_driver_needs():
    """T11 builds the alpha sweep out of exactly these keys.

    Named individually rather than checked for non-emptiness because a silently
    renamed key upstream would otherwise surface as an empty sweep rather than an
    error, and an empty sweep looks like a null result.
    """
    data = ud.experiment("ignition")
    for key in (
        "countries_12",
        "alt_words",
        "ctx_templates",
        "noun_ctx_templates",
        "idiom_pairs",
        "scrambled_pairs",
    ):
        assert key in data, f"ignition.json lost {key!r}"
        assert data[key], f"ignition.json has {key!r} but it is empty"
    assert all("{W}" in t for t in data["ctx_templates"]), "a carrier lost its {W} slot"
    assert all("{W}" in t for t in data["noun_ctx_templates"])


def test_readmes_are_vendored_with_the_prompts():
    """The protocols live in the READMEs; prompts without them are close to unusable."""
    for kind in ud.KINDS:
        assert "##" in ud.readme(kind), f"{kind}/README.md is present but looks empty"
    assert "ignition" in ud.readme("experiments")


def test_available_lists_every_vendored_file():
    assert "ignition" in ud.available("experiments")
    assert "lens-eval-multihop" in ud.available("evaluations")
    counts = {k: len(ud.available(k)) for k in ud.KINDS}
    assert counts == {"experiments": 11, "evaluations": 6}, (
        f"vendored file count changed: {counts}. If upstream was re-vendored at a new "
        f"revision, update PROVENANCE.md and this expectation together."
    )


def test_licence_travels_with_the_data():
    """Apache-2.0 s4(a); same reason `harness/LICENSE` exists."""
    licence = ud.DATA_ROOT / "LICENSE"
    assert licence.exists(), "vendored data must carry its licence"
    assert "Apache License" in licence.read_text()


def test_unknown_kind_and_missing_file_raise_rather_than_return_empty():
    with pytest.raises(ud.DataError):
        ud.load("nonsense", "ignition")
    with pytest.raises(ud.DataError, match="no vendored data file"):
        ud.experiment("does-not-exist")


def test_every_vendored_json_parses():
    """Catches a truncated copy, which `cp -R` can produce on a full disk."""
    for kind in ud.KINDS:
        for stem in ud.available(kind):
            path = ud.DATA_ROOT / kind / f"{stem}.json"
            json.loads(path.read_text())
