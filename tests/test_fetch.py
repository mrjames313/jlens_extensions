"""Tests for the reference-artifact fetcher.

Network-free. The download path is driven through ``file://`` URLs, which go through
the same urllib machinery as an https fetch, so streaming, hashing, verification and
the atomic rename are all covered without touching the Hub or the 48 MB artifact.

The property most of these assert is **negative**: a download that fails verification
must leave nothing at the destination name. A partially-written or wrong-contents file
that looks complete is the failure that matters, because every later task reads these
artifacts as the comparison baseline and would have no reason to re-check them.

Not covered here: that a downloaded tensor actually opens with ``JacobianLens.load``.
That needs ``harness/`` on the path and a real artifact, so it runs on the machine
holding ``$JLENS_ARTIFACT_ROOT`` -- see T8's implementation notes.
"""

import hashlib

import pytest

from jlens_extensions.config import Config
from jlens_extensions.fetch import (
    QWEN35_08B,
    FetchError,
    PublishedLens,
    RemoteFile,
    destination,
    fetch,
    resolve_url,
    sha256_of,
)

PAYLOAD = b"jacobian lens bytes" * 500


@pytest.fixture
def cfg(tmp_path):
    return Config(
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        machine="test-box",
        scratch_is_derived=False,
    )


@pytest.fixture
def source(tmp_path):
    """A local stand-in for the published file, addressable as a URL."""
    path = tmp_path / "source.bin"
    path.write_bytes(PAYLOAD)
    return path


def make_lens(*, size=None, sha256=None):
    return PublishedLens(
        model="test-model",
        hf_model="Test/Model",
        path="test-model/jlens/corpus",
        files=(RemoteFile("artifact.bin", size=size, sha256=sha256),),
    )


def point_at(monkeypatch, url):
    monkeypatch.setattr("jlens_extensions.fetch.resolve_url", lambda *a, **k: url)


# --- URL construction -------------------------------------------------------


def test_resolve_url_uses_the_model_repo_form():
    """T1 established these live in a model repo, not a dataset repo. The dataset
    form carries a ``datasets/`` segment and 404s here."""
    url = resolve_url(QWEN35_08B, "config.yaml")

    assert "/neuronpedia/jacobian-lens/resolve/main/" in url
    assert "/datasets/" not in url
    assert url.endswith("qwen3.5-0.8b/jlens/Salesforce-wikitext/config.yaml")


def test_revision_is_honoured():
    assert "/resolve/abc123/" in resolve_url(QWEN35_08B, "config.yaml", revision="abc123")


def test_registry_carries_the_identifiers_recorded_in_t1():
    tensor = next(f for f in QWEN35_08B.files if f.name.endswith(".pt"))

    assert QWEN35_08B.hf_model == "Qwen/Qwen3.5-0.8B"
    assert tensor.size == 48_242_373
    assert tensor.sha256 == "aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48"
    # The two small files are not LFS, so the Hub publishes no oid for them and
    # only their size is known ahead of time.
    assert all(f.sha256 is None for f in QWEN35_08B.files if not f.name.endswith(".pt"))


def test_destination_mirrors_the_remote_layout(cfg):
    assert destination(QWEN35_08B, cfg) == cfg.reference / QWEN35_08B.path


# --- the download itself ----------------------------------------------------


def test_downloads_and_records_the_hash(cfg, source, monkeypatch):
    point_at(monkeypatch, source.as_uri())
    lens = make_lens(size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest())

    (item,) = fetch(lens, cfg)

    assert item.path.read_bytes() == PAYLOAD
    assert item.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert item.size == len(PAYLOAD)
    assert item.verified is True
    assert item.cached is False


def test_hash_is_recorded_when_none_was_known(cfg, source, monkeypatch):
    point_at(monkeypatch, source.as_uri())
    lens = make_lens(size=len(PAYLOAD))

    (item,) = fetch(lens, cfg)

    assert item.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert item.verified is False, "no independent value existed, so nothing was verified"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": len(PAYLOAD), "sha256": "0" * 64},
        {"size": len(PAYLOAD) + 1, "sha256": hashlib.sha256(PAYLOAD).hexdigest()},
    ],
    ids=["wrong-sha", "wrong-size"],
)
def test_a_failed_check_leaves_nothing_behind(cfg, source, monkeypatch, kwargs):
    """The property that matters: no file at the destination name, and no ``.part``
    litter either. A later task must never find a plausible-looking artifact that
    failed verification."""
    point_at(monkeypatch, source.as_uri())
    lens = make_lens(**kwargs)

    with pytest.raises(FetchError):
        fetch(lens, cfg)

    target = destination(lens, cfg)
    assert not (target / "artifact.bin").exists()
    assert list(target.glob("*.part")) == []


def test_a_missing_remote_file_is_reported_not_silently_skipped(cfg, tmp_path, monkeypatch):
    point_at(monkeypatch, (tmp_path / "absent.bin").as_uri())

    with pytest.raises(FetchError):
        fetch(make_lens(), cfg)


# --- idempotence ------------------------------------------------------------


def test_an_intact_file_is_not_refetched(cfg, source, monkeypatch):
    point_at(monkeypatch, source.as_uri())
    lens = make_lens(size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest())
    fetch(lens, cfg)

    # Make any real fetch impossible; success now can only mean it was skipped.
    point_at(monkeypatch, (cfg.artifact_root / "gone.bin").as_uri())
    (item,) = fetch(lens, cfg)

    assert item.cached is True
    assert item.sha256 == hashlib.sha256(PAYLOAD).hexdigest()


def test_force_refetches_an_intact_file(cfg, source, monkeypatch):
    point_at(monkeypatch, source.as_uri())
    lens = make_lens(size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest())
    fetch(lens, cfg)

    (item,) = fetch(lens, cfg, force=True)

    assert item.cached is False


def test_a_corrupt_existing_file_is_replaced(cfg, source, monkeypatch):
    """A truncated earlier attempt should be repaired rather than trusted or fatal."""
    point_at(monkeypatch, source.as_uri())
    lens = make_lens(size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest())
    target = destination(lens, cfg)
    target.mkdir(parents=True)
    (target / "artifact.bin").write_bytes(PAYLOAD[:100])

    (item,) = fetch(lens, cfg)

    assert item.cached is False
    assert item.path.read_bytes() == PAYLOAD


def test_sha256_of_streams_correctly(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(PAYLOAD)

    assert sha256_of(path) == hashlib.sha256(PAYLOAD).hexdigest()


# --- the 4B entry, added by workspace-band-location T14 ----------------------


def test_registry_carries_both_published_rungs():
    from jlens_extensions.fetch import REGISTRY

    assert set(REGISTRY) == {"qwen3.5-0.8b", "qwen3.5-4b"}


def test_4b_entry_has_an_independently_recorded_hash_for_the_tensor():
    """The .pt must carry a sha256 or the integrity check silently degrades.

    `fetch.py` verifies against a hash recorded independently of the download (the
    Hub's LFS oid). A RemoteFile with size but no sha256 still downloads, and still
    reports `verified=False` -- which is correct for the small inline files and would
    be a quiet loss of checking on the 406 MB tensor.
    """
    from jlens_extensions.fetch import QWEN35_4B

    tensor = next(f for f in QWEN35_4B.files if f.name.endswith(".pt"))
    assert tensor.sha256 and len(tensor.sha256) == 64
    assert tensor.size == 406_333_179


def test_4b_fetches_the_early_stopped_lens_not_the_n1000_variant():
    """Regime A compares against the artifact `config.yaml`'s results block describes.

    That is `prompts_fitted: 417`, i.e. the default file. The repo also publishes
    `_n1000.pt` (the un-early-stopped run), which has no results block of its own and
    no 0.8B counterpart; fetching it instead would break the analogy with the lens we
    already validated at 0.8B.
    """
    from jlens_extensions.fetch import QWEN35_4B

    names = [f.name for f in QWEN35_4B.files]
    assert "Qwen3.5-4B_jacobian_lens.pt" in names
    assert not any("n1000" in n for n in names)
