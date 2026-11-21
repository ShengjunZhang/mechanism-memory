"""Fresh-record comparison of paired-response projection and local calibration."""
from __future__ import annotations
from causal_sandbox.source_compat import recorded_source_matches

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .world_model_probes import (
    adaptation_queries, collect_states, download_checkpoint, latent_features,
    load_model, select_gear, sha256,
)
from .tdmpc2_retention import (
    HORIZONS, NeuralResponseDecoder, RidgeResponseDecoder, _calibration_features,
    _metric_row, _sample_queries, _true_responses,
)

DEFAULT_ROOT = Path("experiments/tdmpc2_contrast_followup_20261114")


def project_pair(forward, reverse):
    forward, reverse = np.asarray(forward), np.asarray(reverse)
    if forward.shape != reverse.shape:
        raise ValueError("Paired predictions must have matching shapes")
    return (forward - reverse) / 2


def balanced_predictions(forward, reverse, projected=False):
    if projected:
        odd = project_pair(forward, reverse)
        return np.concatenate((odd, -odd))
    return np.concatenate((forward, reverse))


def choose_maps(raw_losses, projected_losses, ratio):
    return {
        "raw frozen": np.ones(len(raw_losses), dtype=np.float32),
        "raw matched prior": select_gear(raw_losses, policy="matched-prior"),
        "raw selective": select_gear(raw_losses, ratio=ratio),
        "projected matched prior": select_gear(projected_losses, policy="matched-prior"),
        "projected selective": select_gear(projected_losses, ratio=ratio),
    }


def query_arrays(queries):
    return {
        "observations": np.stack([q["state"].observation for q in queries]),
        "physics_states": np.stack([q["state"].physics_state for q in queries]),
        "directions": np.stack([q["direction"] for q in queries]),
        "horizons": np.asarray([q["horizon"] for q in queries]),
    }


def run_one(task, seed, args, protocol):
    import gymnasium as gym
    import torch
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale

    if not hasattr(gym.logger, "set_level"):
        gym.logger.set_level = lambda _level: None
    out = Path(args.output) / args.phase
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{task}_{seed}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["protocol_sha256"] != sha256(Path(args.protocol)):
            raise ValueError("Existing result belongs to a different protocol")
        return result
    started = time.monotonic()
    task_index = protocol["tasks"].index(task)
    acquisition_seed = (41001 if args.phase == "development" else 61001) + 100 * task_index + seed
    environment_seed = (311 if args.phase == "development" else 733) + seed
    checkpoint = download_checkpoint(task, seed, Path(args.checkpoints))
    env = action_scale.Wrapper(suite.load(*task.split("-", 1), task_kwargs={"random": environment_seed}),
                               minimum=-1, maximum=1)
    model, checkpoint_metadata = load_model(Path(args.model_root), checkpoint, task, env, args.device)
    action_dim = int(np.prod(env.action_spec().shape))
    rng = np.random.default_rng(acquisition_seed)
    counts = protocol["state_counts"]
    states, collection_reward = collect_states(model, env, sum(counts.values()), rng, args.device)
    rng.shuffle(states)
    pools, left, hashes = {}, 0, {}
    for name, count in counts.items():
        pools[name] = states[left:left + count]
        hashes[name] = [hashlib.sha256(s.physics_state.tobytes()).hexdigest() for s in pools[name]]
        left += count
    for name in hashes:
        for other in hashes:
            if name != other and set(hashes[name]) & set(hashes[other]):
                raise ValueError("Physical states overlap between partitions")
    pulse = protocol["pulse"]
    identity = np.ones(action_dim, dtype=np.float32)
    truth = identity.copy()
    truth[0] = -1
    train = _sample_queries(pools["train"], protocol["source_train_queries"], action_dim, rng, "mixed")
    validation = _sample_queries(pools["validation"], protocol["source_validation_queries"], action_dim, rng, "random")
    train_x = latent_features(model, train, identity, pulse, args.device)
    train_y = _true_responses(env, train, pulse, identity)
    validation_x = latent_features(model, validation, identity, pulse, args.device)
    validation_y = _true_responses(env, validation, pulse, identity)
    decoders = {
        "ridge20": RidgeResponseDecoder(alpha=20).fit(train_x, train_y),
        "ridge100": RidgeResponseDecoder(alpha=100).fit(train_x, train_y),
        "neural256": NeuralResponseDecoder(train_x.shape[1], train_y.shape[1], 1987, args.device).fit(train_x, train_y),
    }
    validation_mse = {k: float(np.mean((d.predict(validation_x) - validation_y) ** 2)) for k, d in decoders.items()}
    selected_decoder = min(validation_mse, key=validation_mse.get)
    decoder = decoders[selected_decoder]
    del decoders
    adapt = adaptation_queries(pools["adapt"], action_dim, max(protocol["budgets"]), rng)
    adapt_y = _true_responses(env, adapt, pulse, truth)
    tests = {"target-random": _sample_queries(pools["test"], protocol["test_base_queries_per_condition"], action_dim, rng, "random"),
             "target-shifted": _sample_queries(pools["test"], protocol["test_base_queries_per_condition"], action_dim, rng, "shifted", 0)}
    if action_dim > 1:
        tests["target-stable"] = _sample_queries(pools["test"], protocol["test_base_queries_per_condition"], action_dim, rng, "stable", 0)
    test_y = {c: _true_responses(env, q, pulse, truth) for c, q in tests.items()}
    arrays = {"train_features": train_x, "train_targets": train_y,
              "validation_features": validation_x, "validation_targets": validation_y,
              "adapt_targets": adapt_y}
    for label, queries in {"train": train, "validation": validation, "adapt": adapt, **tests}.items():
        arrays.update({label + "__" + key: value for key, value in query_arrays(queries).items()})
    for condition, target in test_y.items():
        arrays[condition + "__targets"] = target
    prediction_cache = {}

    def predict(label, queries, gear):
        key = (label, tuple(int(x) for x in gear))
        if key not in prediction_cache:
            features = latent_features(model, queries, gear, pulse, args.device)
            prediction_cache[key] = decoder.predict(features)
        return prediction_cache[key]

    raw_candidates, odd_candidates = {}, {}
    for axis in range(action_dim):
        flipped = identity.copy()
        flipped[axis] = -1
        for sign, gear in enumerate((identity, flipped)):
            forward = predict("adapt", adapt, gear)
            reverse = predict("adapt", adapt, -gear)
            raw_candidates[(axis, sign)] = forward
            odd_candidates[(axis, sign)] = project_pair(forward, reverse)
    rows, decisions = [], []
    for budget in protocol["budgets"]:
        selected = np.array([q["repeat"] < budget for q in adapt])
        raw_losses, odd_losses = [], []
        for axis in range(action_dim):
            idx = selected & np.array([q["actuator"] == axis for q in adapt])
            raw_losses.append([float(np.mean((raw_candidates[(axis, s)][idx] - adapt_y[idx]) ** 2)) for s in (0, 1)])
            odd_losses.append([float(np.mean((odd_candidates[(axis, s)][idx] - adapt_y[idx]) ** 2)) for s in (0, 1)])
        maps = choose_maps(raw_losses, odd_losses, protocol["acceptance_ratio"])
        maps.update({"raw oracle map": truth, "projected oracle map": truth})
        for method, gear in maps.items():
            projected = method.startswith("projected")
            decisions.append({"budget": budget, "method": method, "gear": gear.tolist(),
                              "localized": bool(np.array_equal(gear, truth)),
                              "false_flips": int(np.sum(gear[1:] < 0))})
            for condition, queries in tests.items():
                forward = predict(condition, queries, gear)
                reverse = predict(condition, queries, -gear)
                prediction = balanced_predictions(forward, reverse, projected)
                target = np.concatenate((test_y[condition], -test_y[condition]))
                arrays[f"pred__{budget}__{method}__{condition}"] = prediction
                row = _metric_row(task, seed, condition, method, prediction, target)
                rows.append(row | {"budget": budget, "localized": bool(np.array_equal(gear, truth)),
                                   "false_flips": int(np.sum(gear[1:] < 0))})
        raw_adapt = raw_candidates[(0, 0)]
        local_queries = [q for q, use in zip(adapt, selected) if use]
        residual = RidgeResponseDecoder(alpha=protocol["global_ridge_alpha"]).fit(
            _calibration_features(raw_adapt[selected], local_queries), adapt_y[selected] - raw_adapt[selected])
        for condition, queries in tests.items():
            forward = predict(condition, queries, identity)
            reverse = predict(condition, queries, -identity)
            reverse_queries = [q | {"direction": -q["direction"]} for q in queries]
            prediction = np.concatenate((forward + residual.predict(_calibration_features(forward, queries)),
                                         reverse + residual.predict(_calibration_features(reverse, reverse_queries))))
            target = np.concatenate((test_y[condition], -test_y[condition]))
            arrays[f"pred__{budget}__global residual__{condition}"] = prediction
            rows.append(_metric_row(task, seed, condition, "global residual", prediction, target) | {"budget": budget})
        arrays[f"losses_raw__{budget}"] = np.asarray(raw_losses)
        arrays[f"losses_projected__{budget}"] = np.asarray(odd_losses)
    cache_name = path.with_suffix(".npz")
    np.savez_compressed(cache_name, **arrays)
    result = {"task": task, "training_seed": seed, "phase": args.phase,
              "protocol_sha256": sha256(Path(args.protocol)), "driver_sha256": sha256(Path(__file__)),
              "checkpoint_sha256": sha256(checkpoint), "checkpoint_metadata": {k: str(v) for k, v in checkpoint_metadata.items()},
              "acquisition_seed": acquisition_seed, "environment_seed": environment_seed,
              "selected_decoder": selected_decoder, "validation_mse": validation_mse,
              "physical_state_hashes": hashes, "state_partitions_disjoint": True,
              "action_dim": action_dim, "pulse": pulse, "shifted_actuator": 0,
              "target_pairs_per_budget": {str(b): b * action_dim for b in protocol["budgets"]},
              "additional_target_interactions_for_projection": 0,
              "projection_compute_note": "The same paired candidate predictions are available to every control. Projection adds no simulator pulse; model calls and the raw-matched projection control are reported separately.",
              "test_base_queries_per_condition": protocol["test_base_queries_per_condition"],
              "paired_test_queries_per_condition": protocol["paired_test_queries_per_condition"],
              "source_train_queries": len(train), "source_validation_queries": len(validation),
              "collection_mean_reward": collection_reward, "decisions": decisions, "rows": rows,
              "cache": cache_name.name, "cache_sha256": sha256(cache_name),
              "seconds": time.monotonic() - started}
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    env.close()
    del model, decoder
    torch.cuda.empty_cache()
    print(f"DONE {args.phase} {task} seed={seed} decoder={selected_decoder} seconds={result['seconds']:.1f}", flush=True)
    return result


def summarize(runs, protocol, bootstrap=2000):
    grouped = defaultdict(list)
    for run in runs:
        for row in run["rows"]:
            grouped[(row["condition"], row["method"], row["budget"])].append(row)
    summary = []
    for (condition, method, budget), rows in sorted(grouped.items()):
        result = {"condition": condition, "method": method, "budget": budget, "agents": len(rows)}
        tasks = sorted({r["task"] for r in rows})
        rng = np.random.default_rng(7441)
        for metric in ("nrmse", "sign_accuracy", "target_accuracy"):
            arrays = [np.array([r[metric] for r in rows if r["task"] == t]) for t in tasks]
            values = np.stack([a[rng.integers(0, len(a), size=(bootstrap, len(a)))].mean(axis=1) for a in arrays]).mean(axis=0)
            result[metric] = float(np.mean([a.mean() for a in arrays]))
            result[metric + "_ci95"] = np.quantile(values, [.025, .975]).tolist()
        summary.append(result)
    comparisons = []
    primary = protocol["primary_method"]
    for condition in ("target-random", "target-shifted", "target-stable"):
        for baseline in (protocol["primary_baseline"], "projected matched prior", "raw selective"):
            pairs = []
            for run in runs:
                items = {r["method"]: r for r in run["rows"] if r["condition"] == condition and r["budget"] == protocol["primary_budget"]}
                if primary in items and baseline in items:
                    pairs.append({"task": run["task"], "seed": run["training_seed"],
                                  "nrmse_difference": items[primary]["nrmse"] - items[baseline]["nrmse"],
                                  "sign_accuracy_difference": items[primary]["sign_accuracy"] - items[baseline]["sign_accuracy"]})
            if not pairs:
                continue
            tasks = sorted({p["task"] for p in pairs})
            item = {"condition": condition, "method": primary, "baseline": baseline, "agents": len(pairs), "pairs": pairs}
            for metric in ("nrmse_difference", "sign_accuracy_difference"):
                arrays = [np.array([p[metric] for p in pairs if p["task"] == t]) for t in tasks]
                rng = np.random.default_rng(7442)
                values = np.stack([a[rng.integers(0, len(a), size=(bootstrap, len(a)))].mean(axis=1) for a in arrays]).mean(axis=0)
                item[metric] = float(np.mean([a.mean() for a in arrays]))
                item[metric + "_ci95"] = np.quantile(values, [.025, .975]).tolist()
            item["nrmse_wins_ties_losses"] = [sum(p["nrmse_difference"] < -1e-7 for p in pairs),
                                              sum(abs(p["nrmse_difference"]) <= 1e-7 for p in pairs),
                                              sum(p["nrmse_difference"] > 1e-7 for p in pairs)]
            comparisons.append(item)
    return {"runs": len(runs), "summary": summary, "paired_primary_comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_ROOT / "protocol.json"))
    parser.add_argument("--output", default=str(DEFAULT_ROOT / "results"))
    parser.add_argument("--model-root", default="external/tdmpc2")
    parser.add_argument("--checkpoints", default="external/tdmpc2/checkpoints/revision_single_task")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if args.phase == "confirmation":
        lock = Path(args.output).parent / "confirmation_lock.json"
        if not lock.exists():
            raise ValueError("Freeze the protocol and implementation after development before confirmation")
        record = json.loads(lock.read_text())
        if record["protocol_sha256"] != sha256(Path(args.protocol)) or not recorded_source_matches(__file__, record["driver_sha256"]):
            raise ValueError("Confirmation lock differs from protocol/driver")
    import torch
    torch.set_num_threads(2)
    seeds = protocol[args.phase + "_training_seeds"]
    tasks = args.tasks or protocol["tasks"]
    for task in tasks:
        if task not in protocol["tasks"]:
            raise ValueError(task)
        for seed in seeds:
            if not args.summarize_only:
                print(f"START {args.phase} {task} seed={seed}", flush=True)
                run_one(task, seed, args, protocol)
    out = Path(args.output) / args.phase
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.name != "summary.json"]
    result = summarize(runs, protocol)
    result["protocol_sha256"] = sha256(Path(args.protocol))
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    primary = [r for r in result["summary"] if r["condition"] == "target-random" and r["budget"] == protocol["primary_budget"]]
    print(json.dumps(primary, indent=2), flush=True)


if __name__ == "__main__":
    main()
