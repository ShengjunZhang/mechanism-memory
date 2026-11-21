import numpy as np

from causal_sandbox.noisy_readout_filter_followup import ValidatedReadoutFilter, binary_ba


def records(seed=71, count=160, noise=.1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        before = {k: bool(rng.integers(2)) for k in ("x", "z", "w")}
        after = dict(before)
        after["x"] = bool(rng.integers(2))
        for state in (before, after):
            state["q"] = (state["x"] or state["z"]) ^ (rng.random() < noise)
        rows.append({"before": before, "after": after, "index": i})
    return rows


def test_pair_counts_and_opaque_noisy_or():
    filter_ = ValidatedReadoutFilter()
    filter_.observe(records())
    candidates = filter_.candidates()
    assert filter_.suppress(candidates, .75, .8) == {"q"}
    assert all(r["fit_pairs"] == 80 and r["validation_pairs"] == 80 for r in candidates)


def test_holdout_failure_prevents_suppression():
    filter_ = ValidatedReadoutFilter()
    rows = records(noise=0)
    for i, row in enumerate(rows):
        if i % 2:
            for side in ("before", "after"):
                row[side]["q"] = not row[side]["q"]
    filter_.observe(rows)
    assert "q" not in filter_.suppress(filter_.candidates(), .75, .8)


def test_small_sample_abstains():
    filter_ = ValidatedReadoutFilter()
    filter_.observe(records(count=20))
    assert filter_.candidates() == []


def test_constant_predictor_has_no_ba_advantage():
    np.testing.assert_array_equal(binary_ba([[1, 1, 1, 1]], [0, 1, 1, 1]), [.5])
