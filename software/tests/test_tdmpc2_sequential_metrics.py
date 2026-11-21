import numpy as np
from causal_sandbox.tdmpc2_sequential import response_lookup, trajectory_metrics


def test_lookup_matches_direct_errors_for_every_map_pair():
    rng = np.random.default_rng(88)
    p, y = rng.normal(size=(8, 9, 4)), rng.normal(size=(8, 9, 4))
    nrmse, signs = response_lookup(p, y)
    for i in range(8):
        for j in range(8):
            expected = np.sqrt(np.mean(np.sum((p[i] - y[j]) ** 2, axis=1)) /
                               (np.mean(np.sum(y[j] ** 2, axis=1)) + 1e-12))
            assert np.isclose(nrmse[i, j], expected)
            d = np.argmax(np.abs(y[j]), axis=1)
            expected_sign = np.mean(np.sign(p[i, np.arange(9), d]) == np.sign(y[j, np.arange(9), d]))
            assert signs[i, j] == expected_sign


def test_metrics_distinguish_stable_errors_changed_entries_and_bad_writes():
    truth = np.array([[1, 1], [-1, 1], [-1, -1]])
    estimate = np.array([[1, -1], [-1, -1], [-1, -1]])
    metrics = trajectory_metrics(estimate, truth, (np.zeros((4,4)), np.ones((4,4))))
    np.testing.assert_allclose(metrics[:, 2], [.5, .5, 1.])
    np.testing.assert_allclose(metrics[:, 4], [.5, 1., 0.])
    assert np.isnan(metrics[0, 5])
    np.testing.assert_allclose(metrics[1:, 5], [1., 1.])
    np.testing.assert_allclose(metrics[:, 6], [.5, 0., 0.])
