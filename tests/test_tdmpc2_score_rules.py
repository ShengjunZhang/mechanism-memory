import numpy as np
import pytest
from causal_sandbox.tdmpc2_score_rules import choose_scores, source_weights


def scores(p, y, budget=2):
    return choose_scores(np.asarray(p), np.asarray(y), budget, 2, [.5, .75, 1.], [0., .25, .5], np.ones((3, 2)))


def test_clean_flip_and_preservation():
    p = np.ones((4, 2))
    y = p.copy()
    y[:2] *= -1
    for gear in scores(p, y).values():
        assert np.all(gear[1:] == 1)
    assert np.array_equal(scores(p, y)["gain-w0-0.5"], [-1, 1])


def test_uninformative_ties_preserve():
    result = scores(np.zeros((2, 2)), np.ones((2, 2)))
    assert all(np.array_equal(g, [1]) for g in result.values())


def test_only_budget_prefix_used():
    p = np.ones((4, 2))
    y = p.copy()
    before = scores(p, y, 1)
    y[[1, 3]] = -1e9
    after = scores(p, y, 1)
    assert all(np.array_equal(before[k], after[k]) for k in before)


def test_zero_threshold_agrees_with_matched_prior():
    rng = np.random.default_rng(167)
    result = scores(rng.normal(size=(20, 2)), rng.normal(size=(20, 2)))
    assert np.array_equal(result["gain-w0-0"], result["matched prior"])
    assert np.array_equal(result["gain-w1-0"], result["weighted matched prior"])


def test_weights_source_only_and_finite():
    w = source_weights(np.array([[0, 1, 10], [0, 1, 10]]), 1)
    assert np.all(np.isfinite(w)) and np.all(w > 0)
    assert np.isclose(w.mean(), 1)
    assert w[1] > w[2]


def test_bad_budget():
    with pytest.raises(ValueError):
        scores(np.ones((4, 2)), np.ones((4, 2)), 3)
