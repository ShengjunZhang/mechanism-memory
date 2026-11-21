"""Independent cache checks, checkpoint-level comparisons and complete summaries."""
from causal_sandbox.source_compat import sources_match, recorded_source_matches
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("experiments/tdmpc2_sequential_20261116")


def mean_finite(x, axis=None):
    x = np.asarray(x, float)
    n = np.isfinite(x).sum(axis=axis)
    return np.divide(np.nansum(x, axis=axis), n,
                     out=np.full_like(np.asarray(np.nansum(x, axis=axis)), np.nan), where=n > 0)


def independently_estimate(loss, method, variance, budget):
    name, *params = method.split("-")
    current = np.ones(loss.shape[1], dtype=np.int8)
    saved, score = [], np.zeros((loss.shape[1], 2))
    negative = np.full(loss.shape[1], .01)
    for t in range(len(loss)):
        if name in ("memory", "reset"):
            if name == "reset":
                current[:] = 1
            for a in range(len(current)):
                chosen = 0 if current[a] == 1 else 1
                if loss[t, a, 1 - chosen] < float(params[0]) * loss[t, a, chosen]:
                    current[a] = -current[a]
        elif name in ("window", "ewma"):
            if name == "window":
                evidence = np.sum(loss[max(0, t + 1 - int(params[0])):t + 1], axis=0)
            else:
                score *= float(params[0])
                score += loss[t]
                evidence = score
            current[evidence[:, 0] < evidence[:, 1]] = 1
            current[evidence[:, 1] < evidence[:, 0]] = -1
        elif name == "markov":
            h, strength = map(float, params)
            predictive = negative * (1 - h) + (1 - negative) * h
            predictive = np.clip(predictive, 1e-12, 1 - 1e-12)
            # Log unnormalized forward messages, independent of the runner's odds update.
            log_negative = np.log(predictive) - strength * budget * loss[t, :, 1] / (2 * variance)
            log_positive = np.log1p(-predictive) - strength * budget * loss[t, :, 0] / (2 * variance)
            odds = np.clip(log_negative - log_positive, -30, 30)
            negative = np.exp(odds - np.logaddexp(0, odds))
            current[negative > .5] = -1
            current[negative < .5] = 1
        else:
            raise ValueError(method)
        saved.append(current.copy())
    return np.stack(saved)


def verify_run(path, protocol, lock):
    metadata = json.loads(path.read_text())
    for source, expected in lock["code_hashes"].items():
        assert recorded_source_matches(source, expected)
    assert sources_match(metadata["code_hashes"], lock["code_hashes"])
    assert hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() == metadata["cache_sha256"]
    parts = metadata["physical_state_hashes"]
    for a in parts:
        for b in parts:
            if a != b:
                assert not set(parts[a]) & set(parts[b])
    with np.load(path.with_suffix(".npz")) as data:
        methods = metadata["method_names"]
        p, y, weights = data["adapt_prediction"], data["adapt_truth_plus"], data["weights"]
        e = np.mean(data["train_targets"].astype(float) ** 2, axis=0)
        nonzero = e[e > 1e-12]
        floor = max(np.median(nonzero) * .01, 1e-12) if len(nonzero) else 1.
        independently_weighted = 1 / np.maximum(e, floor)
        independently_weighted /= independently_weighted.mean()
        np.testing.assert_allclose(weights, independently_weighted)
        val = data["validation_targets"]
        variance = max(np.mean(weights * (data["validation_odd_prediction"] - val) ** 2),
                       np.mean(weights * val ** 2) * .01, 1e-12)
        assert np.isclose(variance, metadata["source_likelihood_variance"])
        decisions, metrics = data["decisions"], data["metrics"]
        predicted, truth_bank = data["predicted_test_bank"].astype(float), data["true_test_bank"].astype(float)
        nrmse, sign = np.empty((len(predicted), len(truth_bank))), np.empty((len(predicted), len(truth_bank)))
        for j, true_y in enumerate(truth_bank):
            difference = predicted - true_y
            energy = np.mean(np.sum(true_y ** 2, axis=1)) + 1e-12
            nrmse[:, j] = np.sqrt(np.mean(np.sum(difference ** 2, axis=2), axis=1) / energy)
            d = np.argmax(np.abs(true_y), axis=1)
            sign[:, j] = (np.sign(predicted[:, np.arange(len(d)), d]) == np.sign(true_y[np.arange(len(d)), d])).mean(axis=1)
        np.testing.assert_allclose(data["nrmse_lookup"], nrmse, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(data["sign_lookup"], sign)
        cells = 0
        powers = 2 ** np.arange(metadata["action_dim"])
        for s, _ in enumerate(protocol["schedules"]):
            for n, level in enumerate(protocol["noise_levels"]):
                for b, budget in enumerate(protocol["budgets"]):
                    for r in range(protocol["repeats"]):
                        truth = data["truth_maps"][s, r]
                        observed = y[r] * truth[:, :, None, None] + level * data["noise_draws"][r]
                        losses = np.empty((protocol["rounds"], metadata["action_dim"], 2))
                        for k, sign_value in enumerate((1, -1)):
                            residual = sign_value * p[r, :, :, :budget].astype(float) - observed[:, :, :budget]
                            losses[:, :, k] = (weights * residual ** 2).mean(axis=(2, 3))
                        truth_id = ((truth < 0) * powers).sum(axis=-1)
                        previous_truth = np.vstack([np.ones_like(truth[:1]), truth[:-1]])
                        stable, changed = truth == previous_truth, truth != previous_truth
                        for m, method in enumerate(methods):
                            if method == "true-map":
                                expected = truth
                            elif method == "frozen":
                                expected = np.ones_like(truth)
                            else:
                                expected = independently_estimate(losses, method, variance, budget)
                            np.testing.assert_array_equal(expected, decisions[m, s, n, b, r])
                            predicted_id = ((expected < 0) * powers).sum(axis=-1)
                            correct = expected == truth
                            previous = np.vstack([np.ones_like(expected[:1]), expected[:-1]])
                            independently_scored = np.column_stack([
                                nrmse[predicted_id, truth_id], sign[predicted_id, truth_id], correct.mean(axis=1),
                                correct.all(axis=1), mean_finite(np.where(stable, ~correct, np.nan), axis=1),
                                mean_finite(np.where(changed, correct, np.nan), axis=1),
                                ((expected != previous) & ~correct).mean(axis=1)])
                            np.testing.assert_allclose(independently_scored, metrics[m, s, n, b, r],
                                                       atol=1e-8, rtol=1e-8, equal_nan=True)
                            cells += len(truth)
        primary_b = protocol["budgets"].index(protocol["primary_budget"])
        primary = metrics[:, :, :, primary_b, :, :, 0].mean(axis=(1, 2, 3, 4))
        np.testing.assert_allclose(primary, [metadata["primary_nrmse"][m] for m in methods])
        output = metrics.copy()
    print(f"VERIFIED {metadata['task']} seed={metadata['seed']}: {cells} round decisions", flush=True)
    return metadata, output, cells


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    lock = json.loads((ROOT / "confirmation_lock.json").read_text())
    verified = [verify_run(ROOT / "results" / "confirmation" / f"{task}_{seed}.json", protocol, lock)
                for task in protocol["tasks"] for seed in protocol["confirmation_seeds"]]
    arrays = np.stack([r[1] for r in verified])
    methods, metrics = verified[0][0]["method_names"], verified[0][0]["metric_names"]
    selected = lock["selected"]
    names = {"Selective memory": selected["memory"], "Strongest development-selected baseline": selected["baseline"],
             "Windowed matched prior": selected["window"], "Discounted matched prior": selected["ewma"],
             "Markov matched prior": selected["markov"], "Default memory gate": "memory-0.5",
             "No gate": "memory-1", "Reset memory": selected["memory"].replace("memory", "reset"),
             "Frozen map": "frozen", "True map": "true-map"}
    primary_b = protocol["budgets"].index(protocol["primary_budget"])
    agent_primary = mean_finite(arrays[:, :, :, :, primary_b], axis=(2, 3, 4, 5))
    primary = {label: {metric: float(mean_finite(agent_primary[:, methods.index(method), k]))
                       for k, metric in enumerate(metrics)} for label, method in names.items()}
    rng = np.random.default_rng(20261117)
    indices = rng.integers(0, 2, size=(20000, len(protocol["tasks"]), 2))
    def interval(values):
        clustered = np.asarray(values).reshape(len(protocol["tasks"]), 2)
        sampled = np.take_along_axis(np.broadcast_to(clustered, (len(indices),) + clustered.shape), indices, axis=2)
        return {"mean": float(clustered.mean()), "ci95": np.quantile(sampled.mean(axis=(1, 2)), [.025, .975]).tolist(),
                "per_checkpoint": clustered.ravel().tolist()}
    ours = methods.index(selected["memory"])
    baseline = methods.index(selected["baseline"])
    comparisons = {label: {metric: interval(agent_primary[:, ours, k] - agent_primary[:, methods.index(method), k])
                            for k, metric in enumerate(metrics)} for label, method in names.items()
                   if label != "Selective memory"}
    conditions = []
    for s, schedule in enumerate(protocol["schedules"]):
        for n, noise in enumerate(protocol["noise_levels"]):
            for b, budget in enumerate(protocol["budgets"]):
                a = arrays[:, ours, s, n, b, :, :, 0].mean(axis=(1, 2))
                reference = arrays[:, baseline, s, n, b, :, :, 0].mean(axis=(1, 2))
                conditions.append({"schedule": schedule, "noise": noise, "budget": budget,
                                   "memory_nrmse": float(a.mean()), "baseline_nrmse": float(reference.mean()),
                                   "difference": interval(a - reference)})
    data_summary = {"selected": selected, "primary": primary, "comparisons": comparisons, "conditions": conditions,
                    "round_decisions_verified": sum(r[2] for r in verified), "networks": len(verified),
                    "primary_budget": protocol["primary_budget"],
                    "scope": "Checkpoint-stratified percentile intervals for five fixed tasks; two confirmation checkpoints per task. Sequences, rounds, and query coordinates are not independent trained networks. One prespecified primary comparison; all other intervals are descriptive.",
                    "checkpoint_reuse": "Fresh records on previously used independently trained checkpoints; selection held out within this follow-up."}
    (ROOT / "verified_analysis.json").write_text(json.dumps(data_summary, indent=2) + "\n")
    with (ROOT / "all_conditions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "seed", "method", "schedule", "noise", "budget", *metrics])
        writer.writeheader()
        for a, (meta, _, _) in enumerate(verified):
            for m, method in enumerate(methods):
                for s, schedule in enumerate(protocol["schedules"]):
                    for n, noise in enumerate(protocol["noise_levels"]):
                        for b, budget in enumerate(protocol["budgets"]):
                            means = mean_finite(arrays[a, m, s, n, b], axis=(0, 1))
                            writer.writerow({"task": meta["task"], "seed": meta["seed"], "method": method,
                                             "schedule": schedule, "noise": noise, "budget": budget,
                                             **{k: (float(v) if np.isfinite(v) else "") for k, v in zip(metrics, means)}})
    print(json.dumps({"primary": primary, "primary_difference": comparisons["Strongest development-selected baseline"]["nrmse"],
                      "verified_rounds": data_summary["round_decisions_verified"]}, indent=2))


if __name__ == "__main__":
    main()
