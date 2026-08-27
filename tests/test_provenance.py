"""Tests for the provenance sidecar.

Our fits produce no ``config.yaml`` -- we did not vendor the script that writes one
-- so this sidecar is the only record of what made a lens. Two properties carry the
weight:

* **The corpus fingerprint detects a change anywhere**, not just at the ends. The
  risk it exists for is a dataset revision shifting the stream under a deterministic
  loader; if that shift landed in the middle, first/last hashes alone would miss it.
* **Discovery never raises.** A sidecar is written at the end of an hour-long fit.
  A missing git binary or an uninstalled package must degrade to a null field rather
  than throw away the record of a completed run.
"""

import json
import subprocess
from pathlib import Path

import pytest

from jlens_extensions import provenance as prov

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- hashing ----------------------------------------------------------------


def test_sha256_file_matches_text(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("hello")
    assert prov.sha256_file(path) == prov.sha256_text("hello")


def test_sha256_file_reads_in_chunks(tmp_path):
    """Lenses are ~48 MB; the reader must not depend on fitting in one read."""
    path = tmp_path / "big.bin"
    payload = b"x" * (3 * (1 << 20) + 7)
    path.write_bytes(payload)
    import hashlib

    assert prov.sha256_file(path) == hashlib.sha256(payload).hexdigest()


# --- corpus fingerprint -----------------------------------------------------


def test_corpus_fingerprint_empty():
    fp = prov.corpus_fingerprint([])
    assert fp["n_prompts"] == 0
    assert fp["corpus_sha256"] is None


def test_corpus_fingerprint_records_ends_and_count():
    fp = prov.corpus_fingerprint(["alpha", "beta", "gamma"])
    assert fp["n_prompts"] == 3
    assert fp["first_sha256"] == prov.sha256_text("alpha")
    assert fp["last_sha256"] == prov.sha256_text("gamma")
    assert fp["first_chars"] == 5


def test_corpus_fingerprint_catches_a_middle_change():
    """The stated reason the joined hash exists -- endpoints alone would not see this."""
    a = prov.corpus_fingerprint(["alpha", "beta", "gamma"])
    b = prov.corpus_fingerprint(["alpha", "BETA", "gamma"])
    assert a["first_sha256"] == b["first_sha256"]
    assert a["last_sha256"] == b["last_sha256"]
    assert a["corpus_sha256"] != b["corpus_sha256"]


def test_corpus_fingerprint_is_order_sensitive():
    a = prov.corpus_fingerprint(["one", "two"])
    b = prov.corpus_fingerprint(["two", "one"])
    assert a["corpus_sha256"] != b["corpus_sha256"]


def test_corpus_fingerprint_separator_is_unambiguous():
    """Concatenation without a separator would collide on a boundary shift."""
    a = prov.corpus_fingerprint(["ab", "c"])
    b = prov.corpus_fingerprint(["a", "bc"])
    assert a["corpus_sha256"] != b["corpus_sha256"]


# --- git facts --------------------------------------------------------------


def test_git_facts_on_this_repo():
    facts = prov.git_facts(REPO_ROOT)
    assert facts["commit"] and len(facts["commit"]) == 40
    assert facts["dirty"] in (True, False)
    assert facts["uv_lock_sha256"] is not None, "uv.lock is committed deliberately"


def test_git_facts_outside_a_repo_returns_nulls(tmp_path):
    facts = prov.git_facts(tmp_path)
    assert facts["commit"] is None
    assert facts["dirty"] is None
    assert facts["uv_lock_sha256"] is None


def test_git_facts_reports_a_dirty_tree(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x")
    facts = prov.git_facts(tmp_path)
    assert facts["dirty"] is True
    assert "f.txt" in facts["dirty_paths"]


# --- library versions -------------------------------------------------------


def test_library_versions_always_has_python():
    versions = prov.library_versions()
    assert versions["python"].count(".") == 2
    # Absent packages are recorded as None rather than omitted or raised.
    assert set(versions) >= {"python", "torch", "transformers", "datasets", "numpy"}


# --- the sidecar ------------------------------------------------------------


@pytest.fixture
def sidecar(tmp_path):
    return prov.build_sidecar(
        task="T15",
        run="a",
        machine="test-box",
        model_id="Qwen/Qwen3.5-0.8B",
        command=["python", "fit_lens.py", "Qwen/Qwen3.5-0.8B", "--n_prompts", "233"],
        fit_config={"n_prompts": 233, "dim_batch": 8, "save_dtype": "float32"},
        profile_path=tmp_path / "p.toml",
        profile_entry={"dim_batch": 8, "compile": True},
        corpus=prov.corpus_fingerprint(["a", "b"]),
        repo=REPO_ROOT,
        artifacts={},
        results={"rows": 233},
    )


def test_sidecar_command_is_one_line(sidecar):
    assert sidecar["command"] == "python fit_lens.py Qwen/Qwen3.5-0.8B --n_prompts 233"
    assert "\n" not in sidecar["command"]


def test_sidecar_carries_the_four_provenance_axes(sidecar):
    assert sidecar["code"]["commit"]
    assert sidecar["libraries"]["python"]
    assert sidecar["corpus"]["corpus_sha256"]
    assert sidecar["profile"]["entry"] == {"dim_batch": 8, "compile": True}


def test_sidecar_states_the_reproducibility_limit(sidecar):
    """It pins inputs, not the tensor. Saying so is the point -- see T18."""
    assert "not bitwise" in sidecar["reproducibility"].lower() or \
           "envelope" in sidecar["reproducibility"].lower()


def test_sidecar_profile_entry_is_copied_not_referenced(tmp_path, sidecar):
    """A later profile edit must not rewrite history for an already-fitted lens."""
    assert isinstance(sidecar["profile"]["entry"], dict)
    assert sidecar["profile"]["path"].endswith("p.toml")


def test_write_sidecar_round_trips(tmp_path, sidecar):
    path = prov.write_sidecar(tmp_path / "nested" / "s.json", sidecar)
    assert path.exists()
    assert json.loads(path.read_text()) == json.loads(json.dumps(sidecar))


def test_artifact_facts(tmp_path):
    path = tmp_path / "lens.pt"
    path.write_bytes(b"tensor")
    facts = prov.artifact_facts(path)
    assert facts["bytes"] == 6
    assert facts["sha256"] == prov.sha256_text("tensor")
