"""Independent-checkpoint intervention calibration for the JMLR revision."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import numpy as np

from .tdmpc2_retention import (
    HORIZONS, SavedState, RidgeResponseDecoder, NeuralResponseDecoder,
    _flatten_observation, _metric_row, _sample_queries, _true_responses,
)

HF_REVISION = "8fb2a82efb3bae96941da440128fe1332e4394fd"
TASKS = ("cheetah-run", "walker-walk", "reacher-easy", "cartpole-swingup", "finger-spin")
BUDGETS = (1, 3, 5, 10, 20)
RATIOS = (0.45, 0.50, 0.60, 0.75, 0.90)
ALPHAS = (0.1, 1.0, 5.0, 20.0, 100.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_weights(source, prefix):
    """Translate legacy Linear/LayerNorm keys without changing layer order."""
    state = {k[len(prefix) + 1:]: v for k, v in source.items() if k.startswith(prefix + ".")}
    if prefix == "_encoder" and "state.2.weight" in state:
        mapping = {"state.1.": "state.0.", "state.2.": "state.0.ln.",
                   "state.4.": "state.1.", "state.5.": "state.1.ln."}
    elif prefix == "_dynamics" and "0.0.weight" in state:
        mapping = {"0.0.": "0.", "0.1.": "0.ln.", "0.3.": "1.",
                   "0.4.": "1.ln.", "0.6.": "2.", "1.": "2.ln."}
    elif prefix == "_pi" and "6.weight" in state:
        mapping = {"0.": "0.", "1.": "0.ln.", "3.": "1.",
                   "4.": "1.ln.", "6.": "2."}
    else:
        return state
    result = {}
    for key, value in state.items():
        matches = [old for old in mapping if key.startswith(old)]
        if len(matches) != 1:
            raise ValueError(f"Unknown legacy parameter: {prefix}.{key}")
        old = matches[0]
        result[mapping[old] + key[len(old):]] = value
    return result


def download_checkpoint(task: str, seed: int, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task}-{seed}.pt"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    url = f"https://huggingface.co/nicklashansen/tdmpc2/resolve/{HF_REVISION}/dmcontrol/{path.name}"
    temporary = path.with_suffix(f".partial.{os.getpid()}")
    with urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
    if temporary.stat().st_size < 1_000_000:
        raise ValueError(f"Checkpoint download too small: {path}")
    temporary.replace(path)
    return path


def load_model(root: Path, checkpoint: Path, task: str, env, device: str):
    import pathlib
    import torch
    from omegaconf import OmegaConf

    module_path = str(root / "tdmpc2")
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    from common import MODEL_SIZE
    from common.parser import cfg_to_dataclass
    from common.world_model import WorldModel

    cfg = OmegaConf.load(root / "tdmpc2" / "config.yaml")
    cfg.task, cfg.model_size, cfg.multitask = task, 5, False
    cfg.compile, cfg.task_dim, cfg.tasks = False, 0, [task]
    obs_dim = len(_flatten_observation(env.reset().observation))
    action_dim = int(np.prod(env.action_spec().shape))
    cfg.obs_shape, cfg.action_dim = {"state": (obs_dim,)}, action_dim
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    for key, value in MODEL_SIZE[5].items():
        cfg[key] = value
    model = WorldModel(cfg_to_dataclass(cfg)).to(device)
    original_posix = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original_posix
    source = payload["model"]
    for prefix in ("_encoder", "_dynamics", "_pi"):
        state = module_weights(source, prefix)
        getattr(model, prefix).load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload.get("metadata", {})


def collect_states(model, env, count: int, rng, device: str):
    import torch
    records, rewards = [], []
    timestep = env.reset()
    while len(records) < count:
        obs = _flatten_observation(timestep.observation)
        records.append(SavedState(obs.copy(), env.physics.get_state().copy()))
        with torch.inference_mode():
            z = model.encode(torch.from_numpy(obs).to(device)[None], None)
            _, info = model.pi(z, None)
        action = info["mean"][0].cpu().numpy()
        action = np.clip(action + rng.normal(0, 0.12, size=action.shape), -1, 1)
        for _ in range(2):
            timestep = env.step(action)
            rewards.append(float(timestep.reward or 0))
            if timestep.last():
                timestep = env.reset()
                break
    return records, float(np.mean(rewards))


def latent_features(model, queries, gear, pulse, device):
    import torch
    obs = np.stack([q["state"].observation for q in queries]).astype(np.float32)
    direction = (np.stack([q["direction"] for q in queries]) * gear).astype(np.float32)
    horizons = np.asarray([q["horizon"] for q in queries])
    onehot = np.asarray([[int(h == t) for t in HORIZONS] for h in horizons], dtype=np.float32)
    with torch.inference_mode():
        z0 = model.encode(torch.from_numpy(obs).to(device), None)
        actions = torch.from_numpy(direction).to(device)
        plus, minus = z0.clone(), z0.clone()
        delta = torch.empty_like(z0)
        for step in range(1, max(HORIZONS) + 1):
            act = actions * pulse if step == 1 else torch.zeros_like(actions)
            plus, minus = model.next(plus, act, None), model.next(minus, -act, None)
            mask = torch.from_numpy(horizons == step).to(device)
            delta[mask] = (plus[mask] - minus[mask]) / (2 * pulse)
        result = torch.cat((z0, delta, actions, torch.from_numpy(onehot).to(device)), dim=1)
    return result.cpu().numpy()


def adaptation_queries(states, action_dim, maximum, rng):
    result = []
    for actuator in range(action_dim):
        indices = rng.choice(len(states), maximum, replace=False)
        for repeat, idx in enumerate(indices):
            direction = np.zeros(action_dim, dtype=np.float32)
            direction[actuator] = 1
            result.append({"state": states[int(idx)], "direction": direction,
                           "horizon": HORIZONS[repeat % len(HORIZONS)],
                           "actuator": actuator, "repeat": repeat})
    return result


def select_gear(losses, ratio=0.5, policy="gated"):
    losses = np.asarray(losses)
    gear = np.ones(len(losses), dtype=np.float32)
    if policy == "gated":
        gear[losses[:, 1] < ratio * losses[:, 0]] = -1
    elif policy == "matched-prior":
        gear[losses[:, 1] < losses[:, 0]] = -1
    elif policy == "one-flip":
        gains = losses[:, 0] - losses[:, 1]
        best = int(np.argmax(gains))
        if gains[best] > 0:
            gear[best] = -1
    else:
        raise ValueError(policy)
    return gear


def cluster_summary(rows, bootstrap=2000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["method"], row["budget"], row["ratio"], row["alpha"], row["decoder"])].append(row)
    result = []
    for key, group in sorted(grouped.items(), key=lambda x: str(x[0])):
        item = dict(zip(("condition", "method", "budget", "ratio", "alpha", "decoder"), key))
        item["independent_agents"] = len(group)
        tasks = sorted({r["task"] for r in group})
        rng = np.random.default_rng(1927)
        for metric in ("nrmse", "sign_accuracy", "target_accuracy"):
            arrays = [np.array([r[metric] for r in group if r["task"] == task]) for task in tasks]
            values = np.stack([arr[rng.integers(0, len(arr), (bootstrap, len(arr)))].mean(axis=1) for arr in arrays])
            means = values.mean(axis=0)
            item[metric] = float(np.mean([arr.mean() for arr in arrays]))
            item[metric + "_ci95"] = np.quantile(means, [0.025, 0.975]).tolist()
        result.append(item)
    return result


def run_one(task, training_seed, args):
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale
    import torch

    output = Path(args.output) / f"tdmpc2_{task}_{training_seed}.json"
    if output.exists() and not args.overwrite:
        return json.loads(output.read_text(encoding="utf-8"))
    started = time.monotonic()
    checkpoint = download_checkpoint(task, training_seed, Path(args.checkpoints))
    env = action_scale.Wrapper(suite.load(*task.split("-", 1), task_kwargs={"random": 112}), minimum=-1, maximum=1)
    model, metadata = load_model(Path(args.root), checkpoint, task, env, args.device)
    action_dim = int(np.prod(env.action_spec().shape))
    rng = np.random.default_rng(2227)
    states, policy_reward = collect_states(model, env, 1000, rng, args.device)
    rng.shuffle(states)
    pools = {"train": states[:400], "validation": states[400:550], "adapt": states[550:750], "test": states[750:]}
    fingerprints = {name: {hashlib.sha256(s.physics_state.tobytes()).hexdigest() for s in pool} for name, pool in pools.items()}
    for name, values in fingerprints.items():
        for other, other_values in fingerprints.items():
            if name != other and values.intersection(other_values):
                raise ValueError("Physical state overlap between partitions")
    train_queries = _sample_queries(pools["train"], args.train_queries, action_dim, rng, "mixed")
    val_queries = _sample_queries(pools["validation"], 180, action_dim, rng, "random")
    ones = np.ones(action_dim, dtype=np.float32)
    train_x = latent_features(model, train_queries, ones, args.pulse, args.device)
    train_y = _true_responses(env, train_queries, args.pulse, ones)
    val_x = latent_features(model, val_queries, ones, args.pulse, args.device)
    val_y = _true_responses(env, val_queries, args.pulse, ones)
    decoders = {
        "ridge20": RidgeResponseDecoder(alpha=20).fit(train_x, train_y),
        "ridge100": RidgeResponseDecoder(alpha=100).fit(train_x, train_y),
        "neural256": NeuralResponseDecoder(train_x.shape[1], train_y.shape[1], 1987, args.device).fit(train_x, train_y),
    }
    validation = {name: float(np.mean((decoder.predict(val_x) - val_y) ** 2)) for name, decoder in decoders.items()}
    chosen = min(validation, key=validation.get)
    shift_dim = 0
    target = ones.copy()
    target[shift_dim] = -1
    adapt = adaptation_queries(pools["adapt"], action_dim, max(BUDGETS), rng)
    adapt_y = _true_responses(env, adapt, args.pulse, target)
    queries = {"target-random": _sample_queries(pools["test"], args.test_queries, action_dim, rng, "random"),
               "target-shifted": _sample_queries(pools["test"], args.test_queries, action_dim, rng, "shifted", shift_dim)}
    if action_dim > 1:
        queries["target-stable"] = _sample_queries(pools["test"], args.test_queries, action_dim, rng, "stable", shift_dim)
    targets = {name: _true_responses(env, qs, args.pulse, target) for name, qs in queries.items()}
    # Cache all binary per-axis responses; pure-axis calibration makes other signs irrelevant.
    maps = {"identity": ones, "oracle": target}
    for axis in range(action_dim):
        gear = ones.copy()
        gear[axis] = -1
        maps[f"flip-{axis}"] = gear
    adapt_features = {name: latent_features(model, adapt, gear, args.pulse, args.device) for name, gear in maps.items()}
    test_features = {condition: {name: latent_features(model, qs, gear, args.pulse, args.device) for name, gear in maps.items()} for condition, qs in queries.items()}
    rows, diagnostics = [], []
    for decoder_name, decoder in decoders.items():
        source_pred = decoder.predict(adapt_features["identity"])
        for budget in BUDGETS:
            selected = np.array([q["repeat"] < budget for q in adapt])
            local_queries = [q for q, use in zip(adapt, selected) if use]
            losses = []
            for axis in range(action_dim):
                idx = np.array([q["actuator"] == axis for q in adapt]) & selected
                losses.append([float(np.mean((decoder.predict(adapt_features[name][idx]) - adapt_y[idx]) ** 2)) for name in ("identity", f"flip-{axis}")])
            settings = [("selective update", ratio, None, select_gear(losses, ratio)) for ratio in RATIOS]
            settings += [("matched sign prior", None, None, select_gear(losses, policy="matched-prior")),
                         ("one-flip prior", None, None, select_gear(losses, policy="one-flip")),
                         ("frozen", None, None, ones), ("oracle map", None, None, target)]
            for method, ratio, alpha, gear in settings:
                changed = np.flatnonzero(gear != 1)
                if len(changed) <= 1:
                    map_name = "identity" if not len(changed) else f"flip-{changed[0]}"
                    feature_by_condition = {c: values[map_name] for c, values in test_features.items()}
                else:
                    feature_by_condition = {c: latent_features(model, qs, gear, args.pulse, args.device) for c, qs in queries.items()}
                for condition, feature in feature_by_condition.items():
                    row = _metric_row(task, training_seed, condition, method, decoder.predict(feature), targets[condition])
                    rows.append(row | {"budget": budget, "ratio": ratio, "alpha": alpha, "decoder": decoder_name,
                                       "source_selected_decoder": decoder_name == chosen,
                                       "localized": bool(np.array_equal(gear, target))})
            for alpha in ALPHAS:
                from .tdmpc2_retention import _calibration_features
                x = _calibration_features(source_pred[selected], local_queries)
                residual = RidgeResponseDecoder(alpha=alpha).fit(x, adapt_y[selected] - source_pred[selected])
                for condition, qs in queries.items():
                    base = decoder.predict(test_features[condition]["identity"])
                    pred = base + residual.predict(_calibration_features(base, qs))
                    row = _metric_row(task, training_seed, condition, "global residual", pred, targets[condition])
                    rows.append(row | {"budget": budget, "ratio": None, "alpha": alpha, "decoder": decoder_name,
                                       "source_selected_decoder": decoder_name == chosen})
            diagnostics.append({"decoder": decoder_name, "budget": budget, "losses": losses})
    payload = {"task": task, "training_seed": training_seed, "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256(checkpoint), "checkpoint_metadata": {k: str(v) for k, v in metadata.items()},
               "hf_revision": HF_REVISION, "action_dim": action_dim, "shifted_actuator": shift_dim,
               "independent_training": "Official single-task agent checkpoint; no finetuning in this experiment",
               "source_decoder_validation_mse": validation, "selected_decoder": chosen,
               "policy_prior_mean_step_reward": policy_reward, "state_partition_counts": {k: len(v) for k, v in pools.items()},
               "physical_state_partitions_disjoint": True, "train_queries": args.train_queries,
               "test_queries_per_condition": args.test_queries, "horizons": list(HORIZONS), "pulse": args.pulse,
               "budgets": list(BUDGETS), "target_pair_budget": {str(b): b * action_dim for b in BUDGETS},
               "target_rollout_budget": {str(b): 2 * b * action_dim for b in BUDGETS},
               "target_wrapper_step_budget": {str(b): 2 * action_dim * sum(HORIZONS[i % len(HORIZONS)] for i in range(b)) for b in BUDGETS},
               "cost_note": "Each paired response uses two reset branches; each wrapper step repeats two physical control steps. State collection and heldout evaluation costs are excluded from the adaptation budget and reported separately.",
               "max_adaptation_pairs_acquired_for_nested_sweep": len(adapt),
               "rows": rows, "diagnostics": diagnostics, "seconds": time.monotonic() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    env.close()
    del model, decoders
    torch.cuda.empty_cache()
    print(f"DONE {task} training_seed={training_seed} decoder={chosen} seconds={payload['seconds']:.1f}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="external/tdmpc2")
    parser.add_argument("--checkpoints", default="external/tdmpc2/checkpoints/revision_single_task")
    parser.add_argument("--output", default="jmlr_revision_20261112/results/tdmpc2")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-queries", type=int, default=900)
    parser.add_argument("--test-queries", type=int, default=240)
    parser.add_argument("--pulse", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    runs = []
    for task in args.tasks:
        for seed in args.training_seeds:
            print(f"START {task} training_seed={seed}", flush=True)
            runs.append(run_one(task, seed, args))
    selected_rows = [r | {"decoder": "source-validation-selected"}
                     for run in runs for r in run["rows"] if r["source_selected_decoder"]]
    all_rows = [r for run in runs for r in run["rows"]]
    summary = {"independent_agents": len(runs), "hf_revision": HF_REVISION,
               "tasks": args.tasks, "training_seeds": args.training_seeds,
               "ci_method": "Task-stratified bootstrap over independently trained agents, 2000 replicates; tasks fixed, three released training seeds per task",
               "primary": cluster_summary(selected_rows), "sensitivity": cluster_summary(all_rows)}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"COMPLETE {len(runs)} independent agents", flush=True)


if __name__ == "__main__":
    main()
