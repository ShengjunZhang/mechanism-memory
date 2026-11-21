import numpy as np
import pytest
from causal_sandbox.tdmpc2_sequential_rules import (
    estimate_sequence, loss_history, map_indices, method_grid, schedule_maps, sign_maps,
)


def test_maps_roundtrip():
    for d in (1, 2, 6):
        np.testing.assert_array_equal(map_indices(sign_maps(d)), np.arange(2 ** d))


def test_updates_compare_with_incumbent_and_can_revert():
    losses = np.array([[[0, 4], [0, 4]], [[4, 0], [0, 4]], [[0, 4], [0, 4]]])
    expected = np.array([[1, 1], [-1, 1], [1, 1]])
    np.testing.assert_array_equal(estimate_sequence(losses, "memory", .5), expected)


def test_weak_evidence_retains_previous_negative_entry():
    losses = np.array([[[4, 0]], [[.8, 1]]])
    np.testing.assert_array_equal(estimate_sequence(losses, "memory", .5), [[-1], [-1]])
    np.testing.assert_array_equal(estimate_sequence(losses, "reset", .5), [[-1], [1]])


def test_no_gate_equals_roundwise_matched_prior():
    x = np.random.default_rng(9).uniform(size=(12, 6, 2))
    np.testing.assert_array_equal(estimate_sequence(x, "memory", 1), estimate_sequence(x, "window", 1))
    np.testing.assert_array_equal(estimate_sequence(x, "ewma", 0), estimate_sequence(x, "window", 1))


def test_prefix_causality_for_every_family():
    rng = np.random.default_rng(11)
    x = rng.uniform(size=(12, 3, 2))
    changed = x.copy()
    changed[7:] = rng.uniform(0, 100, size=changed[7:].shape)
    for kind, parameter in (("memory", .5), ("reset", .5), ("window", 4),
                            ("ewma", .8), ("markov", (.1, 1.))):
        np.testing.assert_array_equal(estimate_sequence(x, kind, parameter)[:7],
                                      estimate_sequence(changed, kind, parameter)[:7])


def test_no_future_or_overbudget_measurements():
    rng = np.random.default_rng(22)
    p, y = rng.normal(size=(2, 3, 5, 4)), rng.normal(size=(2, 3, 5, 4))
    a = loss_history(p, y, np.ones(4), 3)
    y[:, :, 3:] = 1e10
    np.testing.assert_array_equal(a, loss_history(p, y, np.ones(4), 3))
    with pytest.raises(ValueError):
        loss_history(p, y, np.ones(4), 6)


def test_schedules_change_only_at_declared_rounds():
    for d in (1, 2, 6):
        maps = schedule_maps(d, 12, [3, 6, 9], np.random.default_rng(31))
        assert np.all(maps["stationary"] == 1)
        assert np.all(maps["recurrent"][6:9] == 1)
        for kind in maps:
            for t in range(1, 12):
                edits = np.sum(maps[kind][t] != maps[kind][t-1])
                if t not in (3, 6, 9):
                    assert edits == 0
                elif kind == "sparse":
                    assert edits == 1
                elif kind in ("dense", "recurrent"):
                    assert edits == (d+1)//2


def test_markov_has_correct_likelihood_direction():
    x = np.array([[[100., 0.]], [[100., 0.]], [[0., 100.]]])
    np.testing.assert_array_equal(estimate_sequence(x, "markov", (.1, 1.)), [[-1], [-1], [1]])


def test_ties_preserve_and_prefixes_do_not_reset_state():
    x = np.array([[[4., 0.]], [[0., 0.]], [[0., 0.]]])
    np.testing.assert_array_equal(estimate_sequence(x, "memory", .5), [[-1], [-1], [-1]])


def test_grid_counts_match_prespecified_search():
    p = dict(ratios=[.25,.5,.75,.9,1.], windows=[1,2,4,8,12],
             decays=[0.,.25,.5,.75,.9,1.], hazards=[.01,.05,.2,.5], likelihood_strengths=[.01,.1,1.])
    grid = method_grid(p)
    assert sum(k.startswith("memory-") for k in grid) == 5
    assert sum(k.startswith(("window-", "ewma-", "markov-")) for k in grid) == 23
