from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pytest

from causal_sandbox.observation import LearnedReadoutObservationAdapter
from causal_sandbox.readout_increment_followup import increment_candidates
from causal_sandbox.readout_conflict_followup import eligible_mask, conflict_mask, guard_details


def rule(target, sources, accuracy, simple):
    return {"target": target, "sources": sources, "accuracy": accuracy, "increment": accuracy - simple}


@pytest.mark.parametrize("margin", [.05, .1, .15])
def test_inclusive_increment_boundary(margin):
    assert eligible_mask([rule("r", ["a", "b"], 1., 1. - margin)], margin) == {"r"}


def test_reciprocal_veto_is_simultaneous_and_order_invariant():
    candidates = [rule("r", ["a", "b"], 1., .7), rule("a", ["r", "b"], .90, .65), rule("b", ["r", "a"], .88, .66)]
    for ordered in permutations(candidates):
        assert conflict_mask(ordered, .05)[0] == {"r"}


def test_tied_directions_abstain_and_one_way_edges_are_not_blocked():
    assert conflict_mask([rule("a", ["b", "x"], .9, .6), rule("b", ["a", "z"], .9, .6)], .05)[0] == set()
    assert conflict_mask([rule("a", ["b", "x"], .95, .6), rule("b", ["z", "w"], .9, .6)], .05)[0] == {"a", "b"}


def test_conflict_chain_uses_initial_set_but_three_cycle_is_untouched():
    chain = [rule("a", ["b", "x"], 1., .6), rule("b", ["a", "c"], .95, .6), rule("c", ["b", "x"], .9, .6)]
    assert conflict_mask(chain, .05)[0] == {"a"}
    cycle = [rule("a", ["b", "x"], 1., .6), rule("b", ["c", "y"], .95, .6), rule("c", ["a", "z"], .9, .6)]
    assert conflict_mask(cycle, .05)[0] == {"a", "b", "c"}


@pytest.mark.parametrize("inverse", [False, True])
def test_copy_protection_is_not_vacuous(inverse):
    rng = np.random.default_rng(817)
    adapter = LearnedReadoutObservationAdapter(min_accuracy=.85, refresh_interval=10 ** 9)
    for _ in range(500):
        x = bool(rng.integers(2))
        adapter.observe_state({"x": x, "q": x ^ inverse, "w": rng.random() < .97, "z": rng.random() < .96})
    adapter._refresh_rules()
    candidates = increment_candidates(adapter)
    assert {"x", "q"} <= {r["target"] for r in candidates}
    assert not ({"x", "q"} & conflict_mask(candidates, .05)[0])


def test_saved_clean_or_reverse_rule_regression():
    path = Path("experiments/noisy_readout_threshold_20261114/results/confirmation/family_10208.json")
    run = json.loads(path.read_text())
    trial = next(t for t in run["trials"] if t["condition"] == "or" and t["budget"] == 4)
    adapter = LearnedReadoutObservationAdapter(min_accuracy=.85, refresh_interval=10 ** 9)
    for row in trial["rounds"][0]["batch"]:
        adapter.observe_state(row["before"])
        adapter.observe_state(row["after"])
    adapter._refresh_rules()
    candidates = increment_candidates(adapter)
    roots = set(trial["evaluation_only_truth"]["roots"])
    assert eligible_mask(candidates, .15) & roots
    mask, conflicts = conflict_mask(candidates, .05)
    assert conflicts
    assert not (mask & roots)
    assert trial["evaluation_only_truth"]["readout"] in mask


def test_saved_clean_or_renaming_and_column_order():
    path = Path("experiments/noisy_readout_threshold_20261114/results/confirmation/family_10208.json")
    run = json.loads(path.read_text())
    trial = next(t for t in run["trials"] if t["condition"] == "or" and t["budget"] == 4)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        names = rng.permutation(trial["variables"]).tolist()
        mapping = dict(zip(trial["variables"], names))
        adapter = LearnedReadoutObservationAdapter(min_accuracy=.85, refresh_interval=10 ** 9)
        for row in trial["rounds"][0]["batch"]:
            for side in ("before", "after"):
                adapter.observe_state({mapping[k]: row[side][k] for k in reversed(trial["variables"])})
        adapter._refresh_rules()
        mask, _ = conflict_mask(increment_candidates(adapter), .05)
        assert not (mask & {mapping[r] for r in trial["evaluation_only_truth"]["roots"]})
        assert mapping[trial["evaluation_only_truth"]["readout"]] in mask


def test_retention_guards_check_suppression_even_if_ba_is_unchanged():
    table = {}
    for b in (4, 8):
        for c in ("or", "identity", "noise10"):
            for m in ("components original", "components calibrated", "components conflict"):
                table[c, b, m] = {"physical_ba": .95, "direct_ba": .95, "physical_suppression": 0., "readout_fpr": .01}
    table["identity", 4, "components conflict"]["physical_suppression"] = .01
    guards = guard_details(table, {"budgets_per_action_context": [4, 8]}, "components conflict")
    assert not next(g for g in guards if g["name"] == "identity_4_physical_suppression")["pass"]


def test_resume_rejects_a_changed_selection_lock(tmp_path, monkeypatch):
    from causal_sandbox import readout_conflict_followup as experiment
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    (tmp_path / "protocol.json").write_text("{}")
    lock_path = tmp_path / "confirmation_lock.json"
    lock_path.write_text('{"increment_index":0}')
    folder = tmp_path / "results/confirmation"
    folder.mkdir(parents=True)
    cache_path = folder / "family_7.npz"
    np.savez_compressed(cache_path, placeholder=np.zeros(1))
    cached = {"source_hashes": experiment.source_hashes(), "protocol_sha256": experiment.sha(tmp_path / "protocol.json"),
              "lock_sha256": experiment.sha(lock_path), "cache_sha256": experiment.sha(cache_path), "seconds": 1.}
    (folder / "family_7.json").write_text(json.dumps(cached))
    assert experiment.run_family(7, "confirmation", {}, {}) == (7, 1.)
    lock_path.write_text('{"increment_index":1}')
    with pytest.raises(AssertionError):
        experiment.run_family(7, "confirmation", {}, {})
