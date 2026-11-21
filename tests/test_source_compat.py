import hashlib
import json
from pathlib import Path

import pytest

from causal_sandbox import source_compat as compat


@pytest.fixture
def renamed_source(tmp_path, monkeypatch):
    old, new = b"from .old_module import X\n", b"from .new_module import X\n"
    path = tmp_path / "causal_sandbox/new_module.py"
    path.parent.mkdir()
    path.write_bytes(new)
    record = {"path": "causal_sandbox/new_module.py",
              "original_sha256": hashlib.sha256(old).hexdigest(),
              "packaged_sha256": hashlib.sha256(new).hexdigest(),
              "reverse_edits": [{"start": 0, "end": 1, "original": [old.decode()]}]}
    monkeypatch.setattr(compat, "ROOT", tmp_path)
    monkeypatch.setattr(compat, "_manifest", lambda: {"source_transforms": {"causal_sandbox/old_module.py": record}})
    return path, record


def test_all_packaged_transformations_restore_original_hashes():
    assert compat.verify_source_transforms() > 0


def test_old_and_new_names_and_hashes_match(renamed_source):
    path, r = renamed_source
    assert compat.recorded_source_matches("causal_sandbox/old_module.py", r["original_sha256"])
    assert compat.recorded_source_matches(path, r["packaged_sha256"])
    assert compat.sources_match({"old_module.py": r["original_sha256"]},
                                {"new_module.py": r["packaged_sha256"]})


def test_windows_paths_match(renamed_source):
    _, r = renamed_source
    assert compat.sources_match({"causal_sandbox\\old_module.py": r["original_sha256"]},
                                {"causal_sandbox/new_module.py": r["packaged_sha256"]})


def test_tampered_source_is_rejected(renamed_source):
    path, r = renamed_source
    path.write_bytes(path.read_bytes() + b"X = 0\n")
    assert not compat.recorded_source_matches(path, r["original_sha256"])
    assert not compat.sources_match({"old_module.py": r["original_sha256"]},
                                    {"new_module.py": r["packaged_sha256"]})


def test_wrong_recorded_hash_is_rejected(renamed_source):
    _, r = renamed_source
    assert not compat.sources_match({"old_module.py": "0" * 64},
                                    {"new_module.py": r["packaged_sha256"]})


def test_missing_dependency_is_rejected(renamed_source):
    _, r = renamed_source
    assert not compat.sources_match({}, {"new_module.py": r["packaged_sha256"]})


def test_incorrect_reverse_edit_is_rejected(renamed_source):
    path, r = renamed_source
    r["reverse_edits"][0]["original"] = ["different\n"]
    assert not compat.recorded_source_matches(path, r["original_sha256"])


def test_duplicate_dependency_is_rejected(renamed_source):
    _, r = renamed_source
    assert not compat.sources_match({"old_module.py": r["original_sha256"],
                                     "new_module.py": r["packaged_sha256"]},
                                    {"new_module.py": r["packaged_sha256"]})


def test_lock_comparison_preserves_other_fields(renamed_source):
    _, r = renamed_source
    old = {"code_hashes": {"old_module.py": r["original_sha256"]}, "selected": "memory-0.9"}
    new = {"code_hashes": {"new_module.py": r["packaged_sha256"]}, "selected": "memory-0.9"}
    assert compat.records_match(old, new)
    new["selected"] = "memory-0.5"
    assert not compat.records_match(old, new)


@pytest.mark.skipif(not (compat.ROOT / "experiments/readout_conflict_20261114/results").is_dir(),
                    reason="Saved readout records are stored separately")
def test_recorded_readout_cache_is_accepted_without_acquisition(monkeypatch):
    from causal_sandbox import readout_conflict_followup as driver
    monkeypatch.chdir(compat.ROOT)
    settings = driver.settings()
    lock = json.loads((driver.ROOT / "confirmation_lock.json").read_text())
    assert compat.sources_match(driver.source_hashes(), lock["source_hashes"])
    seed = settings["confirmation_families"][0]
    path = driver.ROOT / "results/confirmation" / f"family_{seed}.json"
    assert path.is_file() and path.with_suffix(".npz").is_file()
    assert driver.run_family(seed, "confirmation", settings, lock) == (seed, json.loads(path.read_text())["seconds"])


@pytest.mark.skipif(not (compat.ROOT / "experiments/tdmpc2_sequential_20261116/results").is_dir(),
                    reason="Saved sequential records are stored separately")
def test_recorded_sequential_cache_is_accepted_without_training(monkeypatch):
    from causal_sandbox import tdmpc2_sequential as driver
    monkeypatch.chdir(compat.ROOT)
    protocol = json.loads((driver.ROOT / "protocol.json").read_text())
    lock = json.loads((driver.ROOT / "confirmation_lock.json").read_text())
    assert compat.sources_match(driver.code_hashes(), lock["code_hashes"])
    task, seed = protocol["tasks"][0], protocol["confirmation_seeds"][0]
    path = driver.ROOT / "results/confirmation" / f"{task}_{seed}.json"
    assert path.is_file() and path.with_suffix(".npz").is_file()
    def no_acquisition(*args, **kwargs):
        raise AssertionError("Saved-cache check must not acquire data")
    monkeypatch.setattr(driver, "collect_states", no_acquisition)
    assert driver.run_agent(task, seed, "confirmation", "cpu", protocol) == json.loads(path.read_text())
