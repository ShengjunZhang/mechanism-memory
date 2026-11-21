"""Fresh-record sequential actuator-map adaptation with history-aware controls."""
from __future__ import annotations
from causal_sandbox.source_compat import sources_match, records_match

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .world_model_probes import collect_states, latent_features, load_model, sha256
from .tdmpc2_contrast_followup import project_pair, query_arrays
from .tdmpc2_retention import HORIZONS, NeuralResponseDecoder, RidgeResponseDecoder, _sample_queries, _true_responses
from .tdmpc2_score_rules import source_weights
from .tdmpc2_sequential_rules import estimate_sequence, loss_history, map_indices, method_grid, schedule_maps, sign_maps

ROOT = Path("experiments/tdmpc2_sequential_20261116")
METRICS = ("nrmse", "sign_accuracy", "map_accuracy", "exact_map", "stable_error", "change_accuracy", "incorrect_writes")
SOURCES = [Path("causal_sandbox") / (name + ".py") for name in
           ("tdmpc2_sequential", "tdmpc2_sequential_rules", "tdmpc2_score_rules", "world_model_probes",
            "tdmpc2_contrast_followup", "tdmpc2_retention")]


def code_hashes():
    return {str(p): sha256(p) for p in SOURCES} | {str(ROOT / "protocol.json"): sha256(ROOT / "protocol.json")}


def finite_mean(values, axis=None):
    values = np.asarray(values, float)
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(total, count, out=np.full_like(np.asarray(total), np.nan), where=count > 0)


def response_lookup(predicted, truth):
    p = predicted.reshape(len(predicted), -1).astype(float)
    y = truth.reshape(len(truth), -1).astype(float)
    energy = np.square(y).sum(axis=1)
    error = np.square(p).sum(axis=1)[:, None] + energy[None] - 2 * p @ y.T
    nrmse = np.sqrt(np.maximum(0, error) / (energy[None] + 1e-12 * truth.shape[1]))
    sign = np.empty_like(nrmse)
    for target_index, target in enumerate(truth):
        dominant = np.abs(target).argmax(axis=1)
        selected_p = predicted[:, np.arange(len(target)), dominant]
        selected_y = target[np.arange(len(target)), dominant]
        sign[:, target_index] = np.mean(np.sign(selected_p) == np.sign(selected_y), axis=1)
    return nrmse, sign


def trajectory_metrics(estimate, truth, lookup):
    predicted_ids, truth_ids = map_indices(estimate), map_indices(truth)
    correct = estimate == truth
    previous_truth = np.concatenate((np.ones_like(truth[:1]), truth[:-1]))
    previous_estimate = np.concatenate((np.ones_like(estimate[:1]), estimate[:-1]))
    stable = truth == previous_truth
    changed = ~stable
    stable_error = finite_mean(np.where(stable, ~correct, np.nan), axis=1)
    change_accuracy = finite_mean(np.where(changed, correct, np.nan), axis=1)
    incorrect_writes = np.mean((estimate != previous_estimate) & ~correct, axis=1)
    return np.stack((lookup[0][predicted_ids, truth_ids], lookup[1][predicted_ids, truth_ids],
                     correct.mean(axis=1), correct.all(axis=1), stable_error, change_accuracy, incorrect_writes), axis=-1)


def run_agent(task, seed, phase, device, protocol):
    import gymnasium as gym
    import torch
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale

    folder = ROOT / "results" / phase
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{task}_{seed}.json"
    if path.exists():
        result = json.loads(path.read_text())
        assert sources_match(result["code_hashes"], code_hashes())
        assert result["cache_sha256"] == sha256(path.with_suffix(".npz"))
        return result
    started = time.monotonic()
    if not hasattr(gym.logger, "set_level"):
        gym.logger.set_level = lambda level: None
    index = protocol["tasks"].index(task)
    acquisition_seed = (701001 if phase == "development" else 801001) + 100 * index + seed
    environment_seed = (4711 if phase == "development" else 4911) + seed
    rng = np.random.default_rng(acquisition_seed)
    checkpoint = Path(f"external/tdmpc2/checkpoints/revision_single_task/{task}-{seed}.pt")
    env = action_scale.Wrapper(suite.load(*task.split("-", 1), task_kwargs={"random": environment_seed}), minimum=-1, maximum=1)
    model, _ = load_model(Path("external/tdmpc2"), checkpoint, task, env, device)
    states, reward = collect_states(model, env, sum(protocol["state_counts"].values()), rng, device)
    rng.shuffle(states)
    pools, hashes, start = {}, {}, 0
    for name, count in protocol["state_counts"].items():
        pools[name] = states[start:start + count]
        hashes[name] = [hashlib.sha256(s.physics_state.tobytes()).hexdigest() for s in pools[name]]
        start += count
    for a in hashes:
        for b in hashes:
            if a != b:
                assert not set(hashes[a]) & set(hashes[b])
    dim, pulse = int(np.prod(env.action_spec().shape)), protocol["pulse"]
    identity = np.ones(dim, dtype=np.float32)
    train = _sample_queries(pools["train"], protocol["train_queries"], dim, rng, "mixed")
    validation = _sample_queries(pools["validation"], protocol["validation_queries"], dim, rng, "random")
    tx, vx = [latent_features(model, q, identity, pulse, device) for q in (train, validation)]
    ty, vy = [_true_responses(env, q, pulse, identity) for q in (train, validation)]
    decoders = {"ridge20": RidgeResponseDecoder(20).fit(tx, ty), "ridge100": RidgeResponseDecoder(100).fit(tx, ty),
                "neural256": NeuralResponseDecoder(tx.shape[1], ty.shape[1], 1987, device).fit(tx, ty)}
    validation_mse = {k: float(np.mean((d.predict(vx) - vy) ** 2)) for k, d in decoders.items()}
    selected = min(validation_mse, key=validation_mse.get)
    decoder = decoders[selected]
    def predict(queries, gear):
        return decoder.predict(latent_features(model, queries, gear, pulse, device))

    weights = source_weights(ty, 1)
    scale = np.sqrt(np.mean(ty.astype(float) ** 2, axis=0))
    valid_odd = project_pair(predict(validation, identity), predict(validation, -identity))
    variance = max(float(np.mean(weights * (valid_odd - vy) ** 2)), float(np.mean(weights * vy ** 2)) * .01, 1e-12)
    rounds, repeats, maximum = protocol["rounds"], protocol["repeats"], max(protocol["budgets"])
    adapt = []
    for sequence in range(repeats):
        for t in range(rounds):
            for actuator in range(dim):
                indices = rng.choice(len(pools["adapt"]), maximum, replace=False)
                direction = np.eye(dim, dtype=np.float32)[actuator]
                for j, idx in enumerate(indices):
                    adapt.append({"state": pools["adapt"][int(idx)], "direction": direction,
                                  "horizon": HORIZONS[(j + t + sequence) % len(HORIZONS)]})
    ap = project_pair(predict(adapt, identity), predict(adapt, -identity))
    ay = _true_responses(env, adapt, pulse, identity)
    # A one-axis pulse with sign -1 swaps the two physical reset branches.
    checked_indices = np.linspace(0, len(adapt) - 1, 24, dtype=int)
    reverse = _true_responses(env, [adapt[i] for i in checked_indices], pulse, -identity)
    np.testing.assert_allclose(reverse, -ay[checked_indices], atol=1e-6, rtol=1e-6)
    ap = ap.reshape(repeats, rounds, dim, maximum, -1)
    ay = ay.reshape(ap.shape)
    noise = rng.normal(size=ap.shape) * scale
    truths = np.stack([np.stack([s[k] for k in protocol["schedules"]]) for s in
                       [schedule_maps(dim, rounds, protocol["change_rounds_zero_based"], rng) for _ in range(repeats)]], axis=1)
    tests = _sample_queries(pools["test"], protocol["test_queries"], dim, rng, "random")
    gears = sign_maps(dim)
    predicted, true = {}, {}
    for code, gear in enumerate(gears):
        if code in predicted:
            continue
        opposite = int(map_indices(-gear))
        predicted[code] = project_pair(predict(tests, gear), predict(tests, -gear))
        true[code] = _true_responses(env, tests, pulse, gear)
        predicted[opposite], true[opposite] = -predicted[code], -true[code]
    predicted_bank = np.stack([predicted[i] for i in range(len(gears))])
    truth_bank = np.stack([true[i] for i in range(len(gears))])
    lookup = response_lookup(predicted_bank, truth_bank)
    grid = method_grid(protocol)
    names = list(grid) + ["frozen", "true-map"]
    shape = (len(names), len(protocol["schedules"]), len(protocol["noise_levels"]), len(protocol["budgets"]), repeats, rounds)
    decisions = np.empty(shape + (dim,), dtype=np.int8)
    metrics = np.empty(shape + (len(METRICS),), dtype=float)
    for s, schedule in enumerate(protocol["schedules"]):
        for n, level in enumerate(protocol["noise_levels"]):
            for b, budget in enumerate(protocol["budgets"]):
                for r in range(repeats):
                    truth = truths[s, r]
                    observed = ay[r] * truth[:, :, None, None] + level * noise[r]
                    losses = loss_history(ap[r], observed, weights, budget)
                    maps = {name: estimate_sequence(losses, kind, parameter, variance, budget)
                            for name, (kind, parameter) in grid.items()}
                    maps["frozen"], maps["true-map"] = np.ones_like(truth), truth
                    for m, name in enumerate(names):
                        decisions[m, s, n, b, r] = maps[name]
                        metrics[m, s, n, b, r] = trajectory_metrics(maps[name], truth, lookup)
    arrays = {"train_features": tx, "train_targets": ty, "validation_features": vx, "validation_targets": vy,
              "validation_odd_prediction": valid_odd, "adapt_prediction": ap, "adapt_truth_plus": ay,
              "weights": weights, "noise_scale": scale, "noise_draws": noise, "truth_maps": truths,
              "predicted_test_bank": predicted_bank, "true_test_bank": truth_bank, "decisions": decisions,
              "metrics": metrics, "nrmse_lookup": lookup[0], "sign_lookup": lookup[1]}
    for label, queries in (("train", train), ("validation", validation), ("adapt", adapt), ("test", tests)):
        arrays.update({label + "__" + k: v for k, v in query_arrays(queries).items()})
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    primary_b = protocol["budgets"].index(protocol["primary_budget"])
    primary = metrics[:, :, :, primary_b, :, :, 0].mean(axis=(1, 2, 3, 4))
    result = {"task": task, "seed": seed, "phase": phase, "acquisition_seed": acquisition_seed,
              "environment_seed": environment_seed, "action_dim": dim, "selected_decoder": selected,
              "validation_mse": validation_mse, "source_likelihood_variance": variance,
              "physical_state_hashes": hashes, "checkpoint_sha256": sha256(checkpoint),
              "code_hashes": code_hashes(), "cache_sha256": sha256(path.with_suffix(".npz")),
              "method_names": names, "metric_names": METRICS,
              "array_axes": ["method", "schedule", "noise", "budget", "sequence", "round", "metric_or_actuator"],
              "primary_nrmse": dict(zip(names, map(float, primary))), "collection_reward": reward,
              "target_pairs_per_round": {str(b): b * dim for b in protocol["budgets"]},
              "symmetry_checks": 24, "seconds": time.monotonic() - started}
    path.write_text(json.dumps(result, indent=2) + "\n")
    env.close()
    del model, decoder, decoders
    torch.cuda.empty_cache()
    print(f"DONE {phase} {task} seed={seed}, {dim} actuators, {result['seconds']:.1f}s", flush=True)
    return result


def summarize(runs, protocol):
    names = runs[0]["method_names"]
    return {name: float(np.mean([np.mean([r["primary_nrmse"][name] for r in runs if r["task"] == task])
                                for task in protocol["tasks"]])) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    lock_path = ROOT / "confirmation_lock.json"
    if args.phase == "confirmation":
        lock = json.loads(lock_path.read_text())
        assert sources_match(lock["code_hashes"], code_hashes())
    runs = [run_agent(task, seed, args.phase, args.device, protocol)
            for task in protocol["tasks"] for seed in protocol[args.phase + "_seeds"]]
    summary = summarize(runs, protocol)
    (ROOT / "results" / args.phase / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.phase == "development":
        selected = {kind: min((k for k in summary if k.startswith(kind + "-")), key=lambda k: (summary[k], k))
                    for kind in ("memory", "window", "ewma", "markov")}
        selected["baseline"] = min((selected[k] for k in ("window", "ewma", "markov")), key=lambda k: (summary[k], k))
        lock = {"selected": selected, "code_hashes": code_hashes(), "development_summary": summary}
        if lock_path.exists():
            assert records_match(json.loads(lock_path.read_text()), lock)
        else:
            lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"selected": lock["selected"], "selected_results": {k: summary[v] for k, v in lock["selected"].items()},
                      "controls": {k: summary[k] for k in ("memory-0.5", "window-1", "frozen", "true-map")}}, indent=2))


if __name__ == "__main__":
    main()
