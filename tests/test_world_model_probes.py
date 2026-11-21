import unittest
import numpy as np

from causal_sandbox.world_model_probes import select_gear, cluster_summary, module_weights


class RevisionTDMPC2Tests(unittest.TestCase):
    def test_legacy_conversion_preserves_parameter_objects_and_norm_order(self):
        linear, norm = object(), object()
        values = {"_encoder.state.1.weight": linear, "_encoder.state.2.weight": norm}
        converted = module_weights(values, "_encoder")
        self.assertIs(converted["state.0.weight"], linear)
        self.assertIs(converted["state.0.ln.weight"], norm)

    def test_gate_and_matched_prior_have_same_candidates_but_different_acceptance(self):
        losses = [[1., .8], [1., .1], [.1, 1.]]
        np.testing.assert_array_equal(select_gear(losses), [1, -1, 1])
        np.testing.assert_array_equal(select_gear(losses, policy="matched-prior"), [-1, -1, 1])

    def test_one_flip_baseline_does_not_use_true_changed_axis(self):
        losses = [[1., .8], [2., .1], [.1, 1.]]
        np.testing.assert_array_equal(select_gear(losses, policy="one-flip"), [1, -1, 1])

    def test_confidence_interval_unit_is_agent_and_task_means_are_balanced(self):
        rows = []
        for task, values in (("a", [0., 1.]), ("b", [1., 1., 1.])):
            for seed, value in enumerate(values):
                rows.append(dict(task=task, seed=seed, condition="shifted", method="m", budget=5,
                                 ratio=None, alpha=None, decoder="d", nrmse=value,
                                 sign_accuracy=value, target_accuracy=value))
        result = cluster_summary(rows, bootstrap=200)[0]
        self.assertAlmostEqual(result["sign_accuracy"], .75)
        self.assertEqual(result["independent_agents"], 5)
        lo, hi = result["sign_accuracy_ci95"]
        self.assertLessEqual(lo, .75)
        self.assertGreaterEqual(hi, .75)


if __name__ == "__main__":
    unittest.main()
