from itertools import product

import numpy as np
import pytest

from causal_sandbox.observation import LearnedReadoutObservationAdapter
from causal_sandbox.readout_increment_followup import increment_candidates, incremental_mask


def fit(states):
    adapter = LearnedReadoutObservationAdapter(min_accuracy=.85, refresh_interval=10 ** 9)
    for state in states:
        adapter.observe_state(state)
    adapter._refresh_rules()
    return increment_candidates(adapter)


def balanced_states(mode):
    states = []
    for _ in range(10):
        for x, z, w in product((False, True), repeat=3):
            q = x if mode == "copy" else not x if mode == "inverse" else x or z
            states.append({"x": x, "z": z, "w": w, "q": q})
    return states


@pytest.mark.parametrize("mode", ["copy", "inverse"])
def test_copies_cannot_pass_positive_increment(mode):
    candidates = fit(balanced_states(mode))
    assert not ({"x", "q"} & incremental_mask(candidates, .05))
    for candidate in candidates:
        if candidate["target"] in {"x", "q"}:
            assert candidate["simple_accuracy"] == 1
            assert candidate["increment"] <= 0


def test_clean_or_retains_physical_roots():
    candidates = fit(balanced_states("or"))
    assert incremental_mask(candidates, .10) == {"q"}
    readout = next(c for c in candidates if c["target"] == "q")
    assert readout["accuracy"] == 1
    assert readout["simple_accuracy"] == .75
    assert readout["increment"] == .25


def test_noisy_or_passes_on_large_sample():
    rng = np.random.default_rng(719)
    states = []
    for _ in range(800):
        x, z, w = map(bool, rng.integers(2, size=3))
        states.append({"x": x, "z": z, "w": w, "q": (x or z) ^ (rng.random() < .1)})
    assert incremental_mask(fit(states), .10) == {"q"}


def test_renaming_preserves_balanced_decision():
    mapping = {"x": "v600", "z": "v017", "w": "v811", "q": "v093"}
    states = [{mapping[k]: v for k, v in s.items()} for s in balanced_states("or")]
    assert incremental_mask(fit(states), .10) == {"v093"}


def test_small_sample_abstains_and_constant_is_uninformative():
    assert fit(balanced_states("or")[:10]) == []
    states = [s | {"constant": True} for s in balanced_states("or")]
    assert "constant" not in incremental_mask(fit(states), .1)


@pytest.mark.parametrize("margin", [0, -.1, 1.1])
def test_nonpositive_or_invalid_margin_rejected(margin):
    with pytest.raises(ValueError):
        incremental_mask([], margin)
