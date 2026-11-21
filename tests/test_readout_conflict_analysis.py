import copy
import json
from pathlib import Path
import runpy

import pytest


@pytest.fixture(scope="module")
def material():
    root = Path("experiments/readout_conflict_20261114")
    analysis = runpy.run_path(str(root / "analyze.py"), run_name="verification_tests")
    run = json.loads((root / "results/confirmation/family_11201.json").read_text())
    lock = json.loads((root / "confirmation_lock.json").read_text())
    return analysis, run, lock


def test_exact_rescoring_accepts_saved_run(material):
    analysis, run, lock = material
    assert analysis["verify_filter"](run, analysis["settings"](), lock)["increment_checks"] > 0


@pytest.mark.parametrize("field", ["accuracy", "simple_accuracy", "increment"])
def test_small_saved_score_corruption_is_rejected(material, field):
    analysis, run, lock = material
    corrupted = copy.deepcopy(run)
    record = next(c for t in corrupted["trials"] for r in t["rounds"] for c in r["increment_candidates"])
    record[field] += 5e-8
    with pytest.raises(AssertionError):
        analysis["verify_filter"](corrupted, analysis["settings"](), lock)
