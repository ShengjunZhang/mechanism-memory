import numpy as np
from itertools import product

from causal_sandbox.core import ActionSpec
from causal_sandbox.context_readout_memory_followup import TupleComponent, recent_cells, balanced_accuracy
from causal_sandbox.observation import LearnedReadoutObservationAdapter


def record(action, context, before, after, round_idx=0):
    return {"action": action, "context": context, "before": before, "after": after, "round": round_idx, "index": 0}


def test_recent_replay_retains_unvisited_contexts():
    a = record("a", [0, 0], {}, {}, 0)
    b = record("a", [1, 0], {}, {}, 0)
    c = record("a", [0, 0], {}, {}, 1)
    assert recent_cells([a, b, c]) == [b, c]


def test_tuple_replacement_is_local_and_filter_is_shared_input():
    model = TupleComponent([ActionSpec("a", "apply")], ["y", "r"])
    model.update([record("a", [0, 0], {"y": False, "r": False}, {"y": True, "r": True}),
                  record("a", [1, 0], {"y": False, "r": False}, {"y": True, "r": True})], {"r"})
    untouched = model.memory.entries["a", (1, 0), "y"]
    model.update([record("a", [0, 0], {"y": True, "r": True}, {"y": False, "r": False}, 1)], {"r"})
    assert model.memory.entries["a", (1, 0), "y"] is untouched
    assert not model.memory.entries["a", (0, 0), "y"].value
    assert all(e.target != "r" for e in model.memory.entries.values())


def test_prediction_reads_no_query_after_or_truth():
    model = TupleComponent([ActionSpec("a", "apply")], ["y"])
    model.update([record("a", [0, 0], {"y": False}, {"y": True})], set())
    query = {"action": "a", "context": [0, 0], "before": {"y": False}}
    np.testing.assert_array_equal(model.predict([query], set()), [[True]])
    query["context"] = [1, 0]
    np.testing.assert_array_equal(model.predict([query], set()), [[False]])


def test_ba_does_not_reward_all_negative_predictions():
    assert balanced_accuracy([1, 0, 0, 0], [0, 0, 0, 0]) == .5


def test_original_readout_learner_does_not_need_semantic_names():
    adapter = LearnedReadoutObservationAdapter(refresh_interval=10 ** 9)
    for _ in range(6):
        for x, y, z in product((False, True), repeat=3):
            adapter.observe_state({"v13": x, "v29": y, "v47": z, "v81": x or y})
    adapter._refresh_rules()
    assert adapter.suppressed_variables == {"v81"}


def test_unsuppressed_ambiguous_effect_is_not_assigned_a_constant_value():
    model = TupleComponent([ActionSpec("a", "apply")], ["y"])
    model.update([record("a", [0, 0], {"y": False}, {"y": True}),
                  record("a", [0, 0], {"y": True}, {"y": False})], set())
    assert not model.memory.entries
