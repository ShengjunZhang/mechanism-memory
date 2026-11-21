from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HORIZONS = (1, 2, 4)


@dataclass(frozen=True)
class SavedState:
    observation: np.ndarray
    physics_state: np.ndarray


class RidgeResponseDecoder:
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.x_scaler: Any = None
        self.y_scaler: Any = None
        self.model: Any = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RidgeResponseDecoder:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self.x_scaler = StandardScaler().fit(features)
        self.y_scaler = StandardScaler().fit(targets)
        x = self.x_scaler.transform(features)
        y = self.y_scaler.transform(targets)
        self.model = Ridge(alpha=self.alpha).fit(x, y)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        y = self.model.predict(self.x_scaler.transform(features))
        return self.y_scaler.inverse_transform(y).astype(np.float32)


class NeuralResponseDecoder:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int,
        device: str,
    ) -> None:
        import torch

        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, output_dim),
        ).to(self.device)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 180,
    ) -> NeuralResponseDecoder:
        import torch

        rng = np.random.default_rng(991)
        self.x_mean = features.mean(axis=0, keepdims=True)
        self.x_std = features.std(axis=0, keepdims=True) + 1e-5
        self.y_mean = targets.mean(axis=0, keepdims=True)
        self.y_std = targets.std(axis=0, keepdims=True) + 1e-5
        x = ((features - self.x_mean) / self.x_std).astype(np.float32)
        y = ((targets - self.y_mean) / self.y_std).astype(np.float32)
        order = rng.permutation(len(x))
        split = max(1, int(0.85 * len(order)))
        train_idx, val_idx = order[:split], order[split:]
        tx = torch.from_numpy(x[train_idx]).to(self.device)
        ty = torch.from_numpy(y[train_idx]).to(self.device)
        vx = torch.from_numpy(x[val_idx]).to(self.device)
        vy = torch.from_numpy(y[val_idx]).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-4)
        best_state = None
        best_loss = float("inf")
        stale = 0
        batch_size = min(256, len(tx))
        generator = torch.Generator(device="cpu").manual_seed(1987)
        for _epoch in range(epochs):
            self.model.train()
            permutation = torch.randperm(len(tx), generator=generator)
            for start in range(0, len(tx), batch_size):
                idx = permutation[start : start + batch_size].to(self.device)
                prediction = self.model(tx[idx])
                loss = torch.nn.functional.smooth_l1_loss(prediction, ty[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = float(torch.nn.functional.mse_loss(self.model(vx), vy).item())
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= 24:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        import torch

        x = ((features - self.x_mean) / self.x_std).astype(np.float32)
        with torch.no_grad():
            y = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        return (y * self.y_std + self.y_mean).astype(np.float32)


def run_tdmpc2_distributional_retention(
    tdmpc2_root: str,
    checkpoint: str,
    tasks: tuple[str, ...] = ("cheetah-run", "walker-walk", "reacher-easy"),
    seeds: int = 3,
    source_train_queries: int = 900,
    source_test_queries: int = 240,
    target_test_queries: int = 240,
    adaptation_states_per_actuator: int = 5,
    pulse: float = 0.25,
    output_json: str | None = None,
    figure_dir: str | None = None,
) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale
    from omegaconf import OmegaConf

    if not hasattr(gym.logger, "set_level"):
        gym.logger.set_level = lambda _level: None

    root = Path(tdmpc2_root).resolve()
    module_root = root / "tdmpc2"
    checkpoint_path = Path(checkpoint).resolve()
    sys.path.insert(0, str(module_root))
    try:
        from common import MODEL_SIZE, TASK_SET
        from common.parser import cfg_to_dataclass
        from common.world_model import WorldModel
    finally:
        sys.path.pop(0)

    cfg = OmegaConf.load(module_root / "config.yaml")
    cfg.task = "mt30"
    cfg.model_size = 5
    cfg.compile = False
    cfg.multitask = True
    cfg.tasks = TASK_SET["mt30"]
    cfg.task_dim = 64
    cfg.task_title = "MT30"
    cfg.obs_shape = {"state": (24,)}
    cfg.action_dim = 6
    cfg.action_dims = [6] * len(cfg.tasks)
    cfg.obs_shapes = [24] * len(cfg.tasks)
    cfg.episode_lengths = [500] * len(cfg.tasks)
    cfg.episode_length = 500
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    for key, value in MODEL_SIZE[5].items():
        cfg[key] = value
    parsed = cfg_to_dataclass(cfg)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = WorldModel(parsed).to(device)
    _load_world_model_modules(model, checkpoint_path)
    model.eval()

    task_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for task_name in tasks:
        task_idx = parsed.tasks.index(task_name)
        domain, task = task_name.split("-", 1)
        for seed in range(1, seeds + 1):
            rng = np.random.default_rng(10_000 * task_idx + seed)
            env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
                visualize_reward=False,
            )
            env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
            action_dim = int(np.prod(env.action_spec().shape))
            required_states = max(
                700,
                source_train_queries // 2
                + source_test_queries // 3
                + target_test_queries // 3
                + adaptation_states_per_actuator * action_dim
                + 80,
            )
            states, policy_return = _collect_states(
                model=model,
                env=env,
                task_idx=task_idx,
                action_dim=action_dim,
                count=required_states,
                rng=rng,
                device=device,
            )
            rng.shuffle(states)
            train_pool = states[: required_states // 2]
            heldout_pool = states[required_states // 2 :]
            train_queries = _sample_queries(
                train_pool,
                source_train_queries,
                action_dim,
                rng,
                mode="mixed",
            )
            source_test = _sample_queries(
                heldout_pool,
                source_test_queries,
                action_dim,
                rng,
                mode="random",
            )
            shift_dim = (seed - 1) % action_dim
            source_gear = np.ones(action_dim, dtype=np.float32)
            target_gear = source_gear.copy()
            target_gear[shift_dim] = -1.0
            adapt_queries = _adaptation_queries(
                heldout_pool,
                action_dim,
                adaptation_states_per_actuator,
                rng,
            )
            target_random = _sample_queries(
                heldout_pool,
                target_test_queries,
                action_dim,
                rng,
                mode="random",
            )
            target_stable = _sample_queries(
                heldout_pool,
                max(120, target_test_queries // 2),
                action_dim,
                rng,
                mode="stable",
                shift_dim=shift_dim,
            )
            target_shifted = _sample_queries(
                heldout_pool,
                max(120, target_test_queries // 2),
                action_dim,
                rng,
                mode="shifted",
                shift_dim=shift_dim,
            )

            train_features = _latent_features(
                model,
                train_queries,
                task_idx,
                action_dim,
                pulse,
                source_gear,
                device,
            )
            train_targets = _true_responses(env, train_queries, pulse, source_gear)
            output_dim = train_targets.shape[1]
            ridge_decoder = RidgeResponseDecoder(alpha=20.0).fit(train_features, train_targets)
            neural_decoder = NeuralResponseDecoder(
                input_dim=train_features.shape[1],
                output_dim=output_dim,
                seed=seed + 100 * task_idx,
                device=device,
            ).fit(train_features, train_targets)

            source_features = _latent_features(
                model,
                source_test,
                task_idx,
                action_dim,
                pulse,
                source_gear,
                device,
            )
            source_targets = _true_responses(env, source_test, pulse, source_gear)
            source_predictions = {
                "TD-MPC2 linear probe": ridge_decoder.predict(source_features),
                "TD-MPC2 nonlinear probe": neural_decoder.predict(source_features),
            }
            for method, prediction in source_predictions.items():
                task_rows.append(
                    _metric_row(task_name, seed, "source", method, prediction, source_targets)
                )

            adapt_source_features = _latent_features(
                model,
                adapt_queries,
                task_idx,
                action_dim,
                pulse,
                source_gear,
                device,
            )
            adapt_source_prediction = neural_decoder.predict(adapt_source_features)
            adapt_targets = _true_responses(env, adapt_queries, pulse, target_gear)
            estimated_gear, diagnostic = _estimate_selective_gear(
                model=model,
                decoder=neural_decoder,
                queries=adapt_queries,
                targets=adapt_targets,
                task_idx=task_idx,
                action_dim=action_dim,
                pulse=pulse,
                device=device,
            )
            global_update = _fit_global_residual(
                adapt_source_prediction,
                adapt_queries,
                adapt_targets,
            )

            condition_sets = {
                "target-random": target_random,
                "target-stable": target_stable,
                "target-shifted": target_shifted,
            }
            for condition, queries in condition_sets.items():
                true_response = _true_responses(env, queries, pulse, target_gear)
                frozen_features = _latent_features(
                    model,
                    queries,
                    task_idx,
                    action_dim,
                    pulse,
                    source_gear,
                    device,
                )
                frozen_prediction = neural_decoder.predict(frozen_features)
                selective_features = _latent_features(
                    model,
                    queries,
                    task_idx,
                    action_dim,
                    pulse,
                    estimated_gear,
                    device,
                )
                oracle_features = _latent_features(
                    model,
                    queries,
                    task_idx,
                    action_dim,
                    pulse,
                    target_gear,
                    device,
                )
                predictions = {
                    "TD-MPC2 frozen": frozen_prediction,
                    "global probe-head update": _apply_global_residual(
                        global_update,
                        frozen_prediction,
                        queries,
                    ),
                    "selective mechanism update": neural_decoder.predict(selective_features),
                    "oracle actuator map": neural_decoder.predict(oracle_features),
                }
                for method, prediction in predictions.items():
                    task_rows.append(
                        _metric_row(task_name, seed, condition, method, prediction, true_response)
                    )

            run_summaries.append(
                {
                    "task": task_name,
                    "seed": seed,
                    "action_dim": action_dim,
                    "shifted_actuator": shift_dim,
                    "true_gear": target_gear.tolist(),
                    "estimated_gear": estimated_gear.tolist(),
                    "localized": bool(np.array_equal(estimated_gear, target_gear)),
                    "diagnostic_losses": diagnostic,
                    "policy_return": policy_return,
                }
            )

    summary = _aggregate_rows(task_rows)
    diagnostic_sensitivity = _diagnostic_sensitivity(run_summaries)
    payload = {
        "experiment": "TD-MPC2 distributional causal-retention probes",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_metadata": _checkpoint_metadata(checkpoint_path),
        "tasks": list(tasks),
        "seeds": seeds,
        "horizons": list(HORIZONS),
        "pulse": pulse,
        "source_train_queries": source_train_queries,
        "source_test_queries": source_test_queries,
        "target_test_queries": target_test_queries,
        "adaptation_states_per_actuator": adaptation_states_per_actuator,
        "rows": task_rows,
        "summary": summary,
        "runs": run_summaries,
        "localization_rate": float(np.mean([run["localized"] for run in run_summaries])),
        "diagnostic_margin_sensitivity": diagnostic_sensitivity,
    }
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if figure_dir:
        _make_figure(payload, Path(figure_dir))
    return payload


def _load_world_model_modules(model: Any, checkpoint: Path) -> None:
    import pathlib
    import torch

    original_posix = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original_posix
    source = payload["model"]
    for prefix, module in (
        ("_task_emb", model._task_emb),
        ("_encoder", model._encoder),
        ("_dynamics", model._dynamics),
        ("_pi", model._pi),
    ):
        state = {
            key[len(prefix) + 1 :]: value
            for key, value in source.items()
            if key.startswith(prefix + ".")
        }
        module.load_state_dict(state, strict=True)


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    import pathlib
    import torch

    original_posix = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original_posix
    metadata = payload.get("metadata", {})
    return {str(key): value for key, value in metadata.items() if key != "tasks"}


def _collect_states(
    model: Any,
    env: Any,
    task_idx: int,
    action_dim: int,
    count: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[list[SavedState], float]:
    import torch

    records: list[SavedState] = []
    completed_returns: list[float] = []
    timestep = env.reset()
    episode_return = 0.0
    wrapper_steps = 0
    while len(records) < count:
        observation = _flatten_observation(timestep.observation)
        records.append(
            SavedState(
                observation=observation.copy(),
                physics_state=env.physics.get_state().copy(),
            )
        )
        padded = _pad(observation, 24)
        with torch.no_grad():
            z = model.encode(torch.from_numpy(padded).to(device).unsqueeze(0), task_idx)
            _sampled, info = model.pi(z, task_idx)
            action = info["mean"][0, :action_dim].detach().cpu().numpy()
        action = np.clip(action + rng.normal(0.0, 0.12, size=action_dim), -1.0, 1.0)
        for _ in range(2):
            timestep = env.step(action)
            episode_return += float(timestep.reward or 0.0)
            if timestep.last():
                break
        wrapper_steps += 1
        if timestep.last() or wrapper_steps >= 500:
            completed_returns.append(episode_return)
            timestep = env.reset()
            episode_return = 0.0
            wrapper_steps = 0
    if not completed_returns:
        completed_returns.append(episode_return * 500.0 / max(wrapper_steps, 1))
    return records, float(np.mean(completed_returns))


def _sample_queries(
    states: list[SavedState],
    count: int,
    action_dim: int,
    rng: np.random.Generator,
    mode: str,
    shift_dim: int | None = None,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for index in range(count):
        record = states[int(rng.integers(0, len(states)))]
        if mode == "mixed" and index % 3 == 0:
            direction = np.zeros(action_dim, dtype=np.float32)
            direction[index % action_dim] = 1.0
        elif mode == "shifted":
            direction = np.zeros(action_dim, dtype=np.float32)
            direction[int(shift_dim)] = 1.0 if index % 2 == 0 else -1.0
        else:
            direction = rng.normal(size=action_dim).astype(np.float32)
            if mode == "stable":
                direction[int(shift_dim)] = 0.0
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                direction[(int(shift_dim) + 1) % action_dim] = 1.0
                norm = 1.0
            direction /= norm
        queries.append(
            {
                "state": record,
                "direction": direction,
                "horizon": int(HORIZONS[index % len(HORIZONS)]),
            }
        )
    return queries


def _adaptation_queries(
    states: list[SavedState],
    action_dim: int,
    per_actuator: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    queries = []
    for actuator in range(action_dim):
        for repeat in range(per_actuator):
            direction = np.zeros(action_dim, dtype=np.float32)
            direction[actuator] = 1.0
            queries.append(
                {
                    "state": states[int(rng.integers(0, len(states)))],
                    "direction": direction,
                    "horizon": int(HORIZONS[repeat % len(HORIZONS)]),
                    "actuator": actuator,
                }
            )
    return queries


def _latent_features(
    model: Any,
    queries: list[dict[str, Any]],
    task_idx: int,
    action_dim: int,
    pulse: float,
    model_gear: np.ndarray,
    device: str,
) -> np.ndarray:
    import torch

    observations = np.stack([_pad(query["state"].observation, 24) for query in queries])
    directions = np.stack([query["direction"] for query in queries]) * model_gear[None, :]
    padded_directions = np.zeros((len(queries), 6), dtype=np.float32)
    padded_directions[:, :action_dim] = directions
    horizons = np.asarray([query["horizon"] for query in queries], dtype=np.int64)
    horizon_onehot = np.zeros((len(queries), len(HORIZONS)), dtype=np.float32)
    for row, horizon in enumerate(horizons):
        horizon_onehot[row, HORIZONS.index(int(horizon))] = 1.0
    with torch.no_grad():
        obs_tensor = torch.from_numpy(observations.astype(np.float32)).to(device)
        direction_tensor = torch.from_numpy(padded_directions).to(device)
        z0 = model.encode(obs_tensor, task_idx)
        z_plus = z0.clone()
        z_minus = z0.clone()
        outputs_plus: dict[int, Any] = {}
        outputs_minus: dict[int, Any] = {}
        zeros = torch.zeros_like(direction_tensor)
        for step in range(1, max(HORIZONS) + 1):
            plus_action = pulse * direction_tensor if step == 1 else zeros
            minus_action = -pulse * direction_tensor if step == 1 else zeros
            z_plus = model.next(z_plus, plus_action, task_idx)
            z_minus = model.next(z_minus, minus_action, task_idx)
            if step in HORIZONS:
                outputs_plus[step] = z_plus.clone()
                outputs_minus[step] = z_minus.clone()
        delta = torch.empty_like(z0)
        for horizon in HORIZONS:
            mask = torch.from_numpy(horizons == horizon).to(device)
            delta[mask] = (outputs_plus[horizon][mask] - outputs_minus[horizon][mask]) / (2.0 * pulse)
        feature = torch.cat(
            [
                z0,
                delta,
                direction_tensor,
                torch.from_numpy(horizon_onehot).to(device),
            ],
            dim=-1,
        )
    return feature.cpu().numpy().astype(np.float32)


def _true_responses(
    env: Any,
    queries: list[dict[str, Any]],
    pulse: float,
    gear: np.ndarray,
) -> np.ndarray:
    responses = []
    for query in queries:
        plus = _branch_observation(
            env,
            query["state"].physics_state,
            pulse * query["direction"],
            int(query["horizon"]),
            gear,
        )
        minus = _branch_observation(
            env,
            query["state"].physics_state,
            -pulse * query["direction"],
            int(query["horizon"]),
            gear,
        )
        responses.append((plus - minus) / (2.0 * pulse))
    return np.asarray(responses, dtype=np.float32)


def _branch_observation(
    env: Any,
    physics_state: np.ndarray,
    commanded_action: np.ndarray,
    horizon: int,
    gear: np.ndarray,
) -> np.ndarray:
    physics = env.physics
    with physics.reset_context():
        physics.set_state(physics_state)
    zero = np.zeros_like(commanded_action)
    for model_step in range(horizon):
        command = commanded_action if model_step == 0 else zero
        physical_action = np.clip(command * gear, -1.0, 1.0)
        for _ in range(2):
            env.task.before_step(physical_action, physics)
            physics.step(env._n_sub_steps)
            env.task.after_step(physics)
    return _flatten_observation(env.task.get_observation(physics))


def _estimate_selective_gear(
    model: Any,
    decoder: NeuralResponseDecoder,
    queries: list[dict[str, Any]],
    targets: np.ndarray,
    task_idx: int,
    action_dim: int,
    pulse: float,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    estimated = np.ones(action_dim, dtype=np.float32)
    diagnostics: dict[str, Any] = {}
    for actuator in range(action_dim):
        indices = [i for i, query in enumerate(queries) if query["actuator"] == actuator]
        candidate_losses = {}
        for candidate in (1.0, -1.0):
            gear = np.ones(action_dim, dtype=np.float32)
            gear[actuator] = candidate
            features = _latent_features(
                model,
                [queries[i] for i in indices],
                task_idx,
                action_dim,
                pulse,
                gear,
                device,
            )
            prediction = decoder.predict(features)
            candidate_losses[str(int(candidate))] = float(
                np.mean((prediction - targets[indices]) ** 2)
            )
        base = candidate_losses["1"]
        alternative = candidate_losses["-1"]
        if alternative < 0.5 * base:
            estimated[actuator] = -1.0
        diagnostics[str(actuator)] = {
            "loss_positive": base,
            "loss_negative": alternative,
            "selected": float(estimated[actuator]),
        }
    return estimated, diagnostics


def _fit_global_residual(
    source_prediction: np.ndarray,
    queries: list[dict[str, Any]],
    targets: np.ndarray,
) -> dict[str, Any]:
    features = _calibration_features(source_prediction, queries)
    residual = targets - source_prediction
    decoder = RidgeResponseDecoder(alpha=5.0).fit(features, residual)
    return {"decoder": decoder}


def _apply_global_residual(
    update: dict[str, Any],
    source_prediction: np.ndarray,
    queries: list[dict[str, Any]],
) -> np.ndarray:
    residual = update["decoder"].predict(_calibration_features(source_prediction, queries))
    return source_prediction + residual


def _calibration_features(
    source_prediction: np.ndarray,
    queries: list[dict[str, Any]],
) -> np.ndarray:
    directions = np.stack([query["direction"] for query in queries]).astype(np.float32)
    horizon = np.zeros((len(queries), len(HORIZONS)), dtype=np.float32)
    for row, query in enumerate(queries):
        horizon[row, HORIZONS.index(int(query["horizon"]))] = 1.0
    return np.concatenate([source_prediction, directions, horizon], axis=1)


def _metric_row(
    task: str,
    seed: int,
    condition: str,
    method: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    error = prediction - target
    target_energy = float(np.mean(np.sum(target**2, axis=1))) + 1e-12
    nrmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1)) / target_energy))
    sample_error = np.linalg.norm(error, axis=1) / np.sqrt(target_energy)
    cutoff = np.quantile(sample_error, 0.9)
    tail = float(np.mean(sample_error[sample_error >= cutoff]))
    target_norm = np.linalg.norm(target, axis=1)
    prediction_norm = np.linalg.norm(prediction, axis=1)
    cosine = np.sum(target * prediction, axis=1) / (
        target_norm * prediction_norm + 1e-8
    )
    dominant = np.argmax(np.abs(target), axis=1)
    predicted_dominant = np.argmax(np.abs(prediction), axis=1)
    rows = np.arange(len(target))
    sign_correct = np.sign(prediction[rows, dominant]) == np.sign(target[rows, dominant])
    return {
        "task": task,
        "seed": seed,
        "condition": condition,
        "method": method,
        "nrmse": nrmse,
        "tail_error": tail,
        "cosine": float(np.mean(cosine)),
        "target_accuracy": float(np.mean(predicted_dominant == dominant)),
        "sign_accuracy": float(np.mean(sign_correct)),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["method"]), []).append(row)
    summary = []
    for (condition, method), values in sorted(grouped.items()):
        item: dict[str, Any] = {
            "condition": condition,
            "method": method,
            "runs": len(values),
        }
        for metric in ("nrmse", "tail_error", "cosine", "target_accuracy", "sign_accuracy"):
            array = np.asarray([value[metric] for value in values], dtype=np.float64)
            item[metric] = float(array.mean())
            item[f"{metric}_se"] = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
        summary.append(item)
    return summary


def _diagnostic_sensitivity(runs: list[dict[str, Any]]) -> list[dict[str, float]]:
    output = []
    for threshold in (0.45, 0.50, 0.60, 0.75, 0.90):
        exact = []
        for run in runs:
            estimate = np.ones(int(run["action_dim"]), dtype=np.float32)
            for actuator, values in run["diagnostic_losses"].items():
                if values["loss_negative"] < threshold * values["loss_positive"]:
                    estimate[int(actuator)] = -1.0
            exact.append(float(np.array_equal(estimate, np.asarray(run["true_gear"]))))
        output.append({"loss_ratio_threshold": threshold, "localization_rate": float(np.mean(exact))})
    return output


def _make_figure(payload: dict[str, Any], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "TD-MPC2 frozen": "#A8B0BA",
        "global probe-head update": "#E26A4A",
        "selective mechanism update": "#16A085",
        "oracle actuator map": "#3B6FB6",
    }
    methods = list(colors)
    conditions = ["target-random", "target-stable", "target-shifted"]
    lookup = {
        (row["condition"], row["method"]): row
        for row in payload["summary"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
    x = np.arange(len(conditions))
    width = 0.18
    for index, method in enumerate(methods):
        values = [lookup[(condition, method)]["nrmse"] for condition in conditions]
        errors = [lookup[(condition, method)]["nrmse_se"] for condition in conditions]
        axes[0].bar(
            x + (index - 1.5) * width,
            values,
            width,
            yerr=errors,
            color=colors[method],
            edgecolor="white",
            linewidth=0.5,
            capsize=2,
            label=method,
        )
    axes[0].set_xticks(x, ["random", "stable", "shifted axis"])
    axes[0].set_ylabel("normalized response error")
    axes[0].set_title("Continuous intervention response")
    axes[0].spines[["top", "right"]].set_visible(False)
    sign_values = [lookup[("target-shifted", method)]["sign_accuracy"] for method in methods]
    sign_errors = [lookup[("target-shifted", method)]["sign_accuracy_se"] for method in methods]
    axes[1].bar(
        np.arange(len(methods)),
        sign_values,
        yerr=sign_errors,
        color=[colors[method] for method in methods],
        edgecolor="white",
        linewidth=0.5,
        capsize=2,
    )
    axes[1].set_xticks(
        np.arange(len(methods)),
        ["frozen", "global", "selective", "oracle"],
        rotation=0,
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("effect-sign accuracy")
    axes[1].set_title("Shifted actuator")
    axes[1].spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.2)
    fig.savefig(figure_dir / "tdmpc2_distributional_retention.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "tdmpc2_distributional_retention.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def _flatten_observation(observation: Any) -> np.ndarray:
    return np.concatenate(
        [np.asarray(value, dtype=np.float32).reshape(-1) for value in observation.values()]
    ).astype(np.float32)


def _pad(array: np.ndarray, width: int) -> np.ndarray:
    output = np.zeros(width, dtype=np.float32)
    output[: len(array)] = array
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_tdmpc2_retention(payload: dict[str, Any]) -> str:
    rows = ["TD-MPC2 distributional causal-retention probes"]
    for row in payload["summary"]:
        if row["condition"] in {"source", "target-random", "target-stable", "target-shifted"}:
            rows.append(
                f"{row['condition']:15s} | {row['method']:28s} | "
                f"NRMSE={row['nrmse']:.3f} | sign={row['sign_accuracy']:.3f}"
            )
    localized = np.mean([float(run["localized"]) for run in payload["runs"]])
    rows.append(f"exact shifted-actuator localization={localized:.3f}")
    return "\n".join(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tdmpc2-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(("cheetah-run", "walker-walk", "reacher-easy")))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--source-train-queries", type=int, default=900)
    parser.add_argument("--source-test-queries", type=int, default=240)
    parser.add_argument("--target-test-queries", type=int, default=240)
    parser.add_argument("--adaptation-states-per-actuator", type=int, default=5)
    parser.add_argument("--pulse", type=float, default=0.25)
    parser.add_argument("--output-json", default="tmp/tdmpc2_distributional_retention.json")
    parser.add_argument("--figure-dir", default="tmp/tdmpc2_retention_figures")
    args = parser.parse_args()
    result = run_tdmpc2_distributional_retention(
        tdmpc2_root=args.tdmpc2_root,
        checkpoint=args.checkpoint,
        tasks=tuple(args.tasks),
        seeds=args.seeds,
        source_train_queries=args.source_train_queries,
        source_test_queries=args.source_test_queries,
        target_test_queries=args.target_test_queries,
        adaptation_states_per_actuator=args.adaptation_states_per_actuator,
        pulse=args.pulse,
        output_json=args.output_json,
        figure_dir=args.figure_dir,
    )
    print(format_tdmpc2_retention(result))
