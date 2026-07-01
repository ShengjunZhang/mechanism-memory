from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProbeAtom:
    cell: tuple[int, ...]
    action_dim: int
    action_sign: int
    target: int
    direction: int

    def key_without_cell(self) -> tuple[int, int, int, int]:
        return (self.action_dim, self.action_sign, self.target, self.direction)


@dataclass(frozen=True)
class MethodScore:
    method: str
    target_accuracy: float
    exact_accuracy: float
    mean_rank: float
    mean_effect_error: float
    cell_coverage: float

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target_accuracy": self.target_accuracy,
            "exact_accuracy": self.exact_accuracy,
            "mean_rank": self.mean_rank,
            "mean_effect_error": self.mean_effect_error,
            "cell_coverage": self.cell_coverage,
        }


@dataclass(frozen=True)
class ReadoutScore:
    method: str
    direct_accuracy: float
    exact_accuracy: float
    readout_false_positive_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "direct_accuracy": self.direct_accuracy,
            "exact_accuracy": self.exact_accuracy,
            "readout_false_positive_rate": self.readout_false_positive_rate,
        }


class ContextBinner:
    def __init__(self, states: np.ndarray, dims: tuple[int, ...], bins: int = 3) -> None:
        self.dims = dims
        self.thresholds = []
        for dim in dims:
            qs = np.linspace(0.0, 1.0, bins + 1)[1:-1]
            self.thresholds.append(np.quantile(states[:, dim], qs))

    def cell(self, state: np.ndarray) -> tuple[int, ...]:
        return tuple(
            int(np.searchsorted(thresholds, state[dim], side="right"))
            for dim, thresholds in zip(self.dims, self.thresholds)
        )


def run_mujoco_probe_experiment(
    env_id: str = "Reacher-v5",
    seeds: int = 5,
    train_states: int = 240,
    test_states: int = 120,
    random_transitions: int = 3000,
    val_transitions: int = 800,
    epochs: int = 60,
    pulse: float = 0.8,
    context_dims: tuple[int, ...] = (0, 1),
    save_json: str | None = None,
    figure_dir: str | None = "docs/figures",
) -> dict[str, object]:
    import gymnasium as gym

    all_scores: dict[str, list[MethodScore]] = defaultdict(list)
    all_training_curves: list[dict[str, list[float]]] = []
    heatmap_counts: Counter[tuple[int, int, int]] = Counter()
    target_names: list[str] | None = None
    action_dim_count = 0

    for seed in range(1, seeds + 1):
        env = gym.make(env_id)
        action_dim_count = int(env.action_space.shape[0])
        state_dim = _state_dim(env)
        target_names = _target_names(env)
        train_probe_states = _sample_states(env, train_states, seed=10_000 + seed)
        test_probe_states = _sample_states(env, test_states, seed=20_000 + seed)
        binner = ContextBinner(train_probe_states, dims=context_dims)

        train_x, train_y, val_x, val_y = _transition_dataset(
            env,
            train_count=random_transitions,
            val_count=val_transitions,
            seed=30_000 + seed,
        )
        linear_model = _fit_linear(train_x, train_y)
        neural_model, curve = _fit_neural(train_x, train_y, val_x, val_y, seed=seed, epochs=epochs)
        all_training_curves.append(curve)

        true_atoms, true_effects = _probe_truth(
            env,
            test_probe_states,
            binner,
            pulse=pulse,
        )
        for atom in true_atoms:
            heatmap_counts[(atom.action_dim, atom.action_sign, atom.target)] += 1

        metric_memory, coverage_cells = _fit_metric_memory(
            env,
            train_probe_states,
            binner,
            pulse=pulse,
        )
        predictors = {
            "random-correlation": _CorrelationPredictor(train_x, train_y),
            "global-linear-dynamics": _LinearPredictor(linear_model),
            "neural-dynamics": _NeuralPredictor(neural_model),
            "metric-causal-core": _MetricMemoryPredictor(metric_memory, coverage_cells),
        }
        for method, predictor in predictors.items():
            predictions = []
            effects = []
            covered = []
            for state in test_probe_states:
                cell = binner.cell(state)
                for action_dim in range(action_dim_count):
                    for action_sign in (-1, 1):
                        atom, effect, is_covered = predictor.predict(
                            state=state,
                            cell=cell,
                            action_dim=action_dim,
                            action_sign=action_sign,
                            pulse=pulse,
                            state_dim=state_dim,
                        )
                        predictions.append(atom)
                        effects.append(effect)
                        covered.append(is_covered)
            all_scores[method].append(
                _score_method(
                    method,
                    predictions,
                    true_atoms,
                    effects,
                    true_effects,
                    covered,
                )
            )
        env.close()

    summary = [_aggregate_scores(method, scores) for method, scores in all_scores.items()]
    payload: dict[str, object] = {
        "task": "mujoco frozen mechanism probe",
        "env_id": env_id,
        "seeds": seeds,
        "train_states": train_states,
        "test_states": test_states,
        "random_transitions": random_transitions,
        "val_transitions": val_transitions,
        "epochs": epochs,
        "pulse": pulse,
        "context_dims": list(context_dims),
        "target_names": target_names or [],
        "summary": summary,
        "per_seed": {
            method: [score.to_dict() for score in scores]
            for method, scores in all_scores.items()
        },
        "training_curves": all_training_curves,
        "heatmap_counts": [
            {
                "action_dim": action_dim,
                "action_sign": action_sign,
                "target": target,
                "count": count,
            }
            for (action_dim, action_sign, target), count in sorted(heatmap_counts.items())
        ],
    }
    if save_json:
        output = Path(save_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if figure_dir:
        _make_figures(payload, Path(figure_dir))
        _try_render_frame(env_id, Path(figure_dir), seed=1)
    return payload


def run_mujoco_causal_checks(
    env_id: str = "HalfCheetah-v5",
    seeds: int = 5,
    train_states: int = 240,
    test_states: int = 120,
    adapt_states: int = 30,
    random_transitions: int = 3000,
    val_transitions: int = 800,
    epochs: int = 60,
    pulse: float = 0.8,
    context_dims: tuple[int, ...] = (0, 1),
    readout_scale: float = 4.0,
    save_json: str | None = None,
    figure_dir: str | None = "docs/figures",
) -> dict[str, object]:
    import gymnasium as gym

    context_scores: dict[str, list[MethodScore]] = defaultdict(list)
    readout_scores: dict[str, list[ReadoutScore]] = defaultdict(list)

    for seed in range(1, seeds + 1):
        source_env = gym.make(env_id)
        shifted_env = gym.make(env_id)
        _apply_actuator_polarity_shift(shifted_env)

        state_dim = _state_dim(source_env)
        source_train_states = _sample_states(source_env, train_states, seed=10_000 + seed)
        shifted_test_states = _sample_states(shifted_env, test_states, seed=20_000 + seed)
        shifted_adapt_states = _sample_states(shifted_env, adapt_states, seed=25_000 + seed)
        binner = ContextBinner(source_train_states, dims=context_dims)

        train_x, train_y, val_x, val_y = _transition_dataset(
            source_env,
            train_count=random_transitions,
            val_count=val_transitions,
            seed=30_000 + seed,
        )
        linear_model = _fit_linear(train_x, train_y)
        neural_model, _curve = _fit_neural(
            train_x,
            train_y,
            val_x,
            val_y,
            seed=seed,
            epochs=epochs,
        )
        source_metric_memory, source_cells = _fit_metric_memory(
            source_env,
            source_train_states,
            binner,
            pulse=pulse,
        )
        adapted_metric_memory, adapted_cells = _fit_metric_memory(
            shifted_env,
            shifted_adapt_states,
            binner,
            pulse=pulse,
        )

        shifted_truth, shifted_effects = _probe_truth(
            shifted_env,
            shifted_test_states,
            binner,
            pulse=pulse,
        )
        context_predictors = {
            "source-linear dynamics": _LinearPredictor(linear_model),
            "source-neural dynamics": _NeuralPredictor(neural_model),
            "source-metric core": _MetricMemoryPredictor(source_metric_memory, source_cells),
            "few-shot metric core": _MetricMemoryPredictor(
                adapted_metric_memory,
                adapted_cells,
            ),
        }
        for method, predictor in context_predictors.items():
            predictions, effects, covered = _predict_probe_grid(
                predictor,
                shifted_test_states,
                binner,
                action_dim_count=int(shifted_env.action_space.shape[0]),
                pulse=pulse,
                state_dim=state_dim,
            )
            context_scores[method].append(
                _score_method(
                    method,
                    predictions,
                    shifted_truth,
                    effects,
                    shifted_effects,
                    covered,
                )
            )

        source_truth, _source_effects = _probe_truth(
            source_env,
            shifted_test_states,
            binner,
            pulse=pulse,
        )
        readout_predictors = {
            "linear observed selector": _LinearPredictor(linear_model),
            "neural observed selector": _NeuralPredictor(neural_model),
            "metric core observed selector": _MetricMemoryPredictor(
                source_metric_memory,
                source_cells,
            ),
            "metric core with readout filter": _MetricMemoryPredictor(
                source_metric_memory,
                source_cells,
            ),
        }
        for method, predictor in readout_predictors.items():
            predictions, effects, _covered = _predict_probe_grid(
                predictor,
                shifted_test_states,
                binner,
                action_dim_count=int(source_env.action_space.shape[0]),
                pulse=pulse,
                state_dim=state_dim,
            )
            readout_scores[method].append(
                _score_readout_targets(
                    method,
                    predictions,
                    source_truth,
                    effects,
                    state_dim=state_dim,
                    readout_scale=readout_scale,
                    filter_readouts=method.endswith("readout filter"),
                )
            )

        source_env.close()
        shifted_env.close()

    payload: dict[str, object] = {
        "task": "mujoco causal checks",
        "env_id": env_id,
        "seeds": seeds,
        "train_states": train_states,
        "test_states": test_states,
        "adapt_states": adapt_states,
        "random_transitions": random_transitions,
        "val_transitions": val_transitions,
        "epochs": epochs,
        "pulse": pulse,
        "context_dims": list(context_dims),
        "readout_scale": readout_scale,
        "context_shift": {
            "description": "all actuator gear signs are reversed at test time",
            "summary": [
                _aggregate_scores(method, scores)
                for method, scores in context_scores.items()
            ],
            "per_seed": {
                method: [score.to_dict() for score in scores]
                for method, scores in context_scores.items()
            },
        },
        "readout_shift": {
            "description": (
                "the observation appends amplified deterministic readouts of "
                "every physical state coordinate"
            ),
            "summary": [
                _aggregate_readout_scores(method, scores)
                for method, scores in readout_scores.items()
            ],
            "per_seed": {
                method: [score.to_dict() for score in scores]
                for method, scores in readout_scores.items()
            },
        },
    }
    if save_json:
        output = Path(save_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if figure_dir:
        _make_causal_check_figures(payload, Path(figure_dir))
    return payload


def format_mujoco_probe(payload: dict[str, object]) -> str:
    lines = [
        "MuJoCo frozen mechanism probe",
        (
            f"Env={payload['env_id']}, seeds={payload['seeds']}, "
            f"transitions={payload['random_transitions']}, epochs={payload['epochs']}"
        ),
        "",
        (
            f"{'method':<25} {'target-acc':>10} {'exact':>7} "
            f"{'rank':>7} {'effect-err':>11} {'coverage':>9}"
        ),
        "-" * 76,
    ]
    for row in payload["summary"]:  # type: ignore[index]
        lines.append(
            f"{row['method']:<25}"
            f"{float(row['target_accuracy']):>10.3f}"
            f"{float(row['exact_accuracy']):>7.3f}"
            f"{float(row['mean_rank']):>7.2f}"
            f"{float(row['mean_effect_error']):>11.4f}"
            f"{float(row['cell_coverage']):>9.3f}"
        )
    return "\n".join(lines)


def format_mujoco_causal_checks(payload: dict[str, object]) -> str:
    lines = [
        "MuJoCo causal checks",
        (
            f"Env={payload['env_id']}, seeds={payload['seeds']}, "
            f"adapt-states={payload['adapt_states']}, epochs={payload['epochs']}"
        ),
        "",
        "Actuator-polarity context shift",
        (
            f"{'method':<28} {'target-acc':>10} {'exact':>7} "
            f"{'rank':>7} {'effect-err':>11} {'coverage':>9}"
        ),
        "-" * 79,
    ]
    context = payload["context_shift"]  # type: ignore[index]
    for row in context["summary"]:  # type: ignore[index]
        lines.append(
            f"{row['method']:<28}"
            f"{float(row['target_accuracy']):>10.3f}"
            f"{float(row['exact_accuracy']):>7.3f}"
            f"{float(row['mean_rank']):>7.2f}"
            f"{float(row['mean_effect_error']):>11.4f}"
            f"{float(row['cell_coverage']):>9.3f}"
        )
    lines.extend(
        [
            "",
            "Amplified deterministic readouts",
            (
                f"{'method':<32} {'direct':>8} {'exact':>8} "
                f"{'readout-fp':>11}"
            ),
            "-" * 64,
        ]
    )
    readout = payload["readout_shift"]  # type: ignore[index]
    for row in readout["summary"]:  # type: ignore[index]
        lines.append(
            f"{row['method']:<32}"
            f"{float(row['direct_accuracy']):>8.3f}"
            f"{float(row['exact_accuracy']):>8.3f}"
            f"{float(row['readout_false_positive_rate']):>11.3f}"
        )
    return "\n".join(lines)


def _sample_states(env: Any, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = []
    obs, _ = env.reset(seed=seed)
    del obs
    while len(states) < count:
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        states.append(_state_vector(env))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=seed + len(states))
    return np.asarray(states, dtype=np.float32)


def _predict_probe_grid(
    predictor: Any,
    states: np.ndarray,
    binner: ContextBinner,
    action_dim_count: int,
    pulse: float,
    state_dim: int,
) -> tuple[list[ProbeAtom], list[np.ndarray], list[bool]]:
    predictions = []
    effects = []
    covered = []
    for state in states:
        cell = binner.cell(state)
        for action_dim in range(action_dim_count):
            for action_sign in (-1, 1):
                atom, effect, is_covered = predictor.predict(
                    state=state,
                    cell=cell,
                    action_dim=action_dim,
                    action_sign=action_sign,
                    pulse=pulse,
                    state_dim=state_dim,
                )
                predictions.append(atom)
                effects.append(effect)
                covered.append(is_covered)
    return predictions, effects, covered


def _transition_dataset(
    env: Any,
    train_count: int,
    val_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = _collect_transitions(env, train_count + val_count, seed)
    return x[:train_count], y[:train_count], x[train_count:], y[train_count:]


def _collect_transitions(env: Any, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    env.reset(seed=seed)
    while len(xs) < count:
        before = _state_vector(env)
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        after = _state_vector(env)
        xs.append(np.concatenate([before, action]))
        ys.append(after - before)
        if terminated or truncated:
            env.reset(seed=seed + len(xs))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _probe_truth(
    env: Any,
    states: np.ndarray,
    binner: ContextBinner,
    pulse: float,
) -> tuple[list[ProbeAtom], list[np.ndarray]]:
    atoms = []
    effects = []
    for state in states:
        cell = binner.cell(state)
        for action_dim in range(env.action_space.shape[0]):
            for action_sign in (-1, 1):
                effect = _finite_difference_effect(env, state, action_dim, action_sign, pulse)
                target = int(np.argmax(np.abs(effect)))
                direction = 1 if effect[target] >= 0 else -1
                atoms.append(ProbeAtom(cell, action_dim, action_sign, target, direction))
                effects.append(effect)
    return atoms, effects


def _fit_metric_memory(
    env: Any,
    states: np.ndarray,
    binner: ContextBinner,
    pulse: float,
) -> tuple[dict[tuple[tuple[int, ...], int, int], ProbeAtom], set[tuple[int, ...]]]:
    votes: dict[tuple[tuple[int, ...], int, int], Counter[tuple[int, int]]] = defaultdict(Counter)
    effects: dict[tuple[tuple[int, ...], int, int, int, int], list[float]] = defaultdict(list)
    for state in states:
        cell = binner.cell(state)
        for action_dim in range(env.action_space.shape[0]):
            for action_sign in (-1, 1):
                effect = _finite_difference_effect(env, state, action_dim, action_sign, pulse)
                target = int(np.argmax(np.abs(effect)))
                direction = 1 if effect[target] >= 0 else -1
                key = (cell, action_dim, action_sign)
                votes[key][(target, direction)] += 1
                effects[(cell, action_dim, action_sign, target, direction)].append(abs(float(effect[target])))
    memory = {}
    for key, counter in votes.items():
        target, direction = max(
            counter,
            key=lambda pair: (
                counter[pair],
                mean(effects[(key[0], key[1], key[2], pair[0], pair[1])]),
            ),
        )
        memory[key] = ProbeAtom(key[0], key[1], key[2], target, direction)
    return memory, {key[0] for key in votes}


class _CorrelationPredictor:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.state_dim = y.shape[1]
        self.action_dim = x.shape[1] - self.state_dim
        actions = x[:, self.state_dim :]
        self.effects = np.zeros((self.action_dim, self.state_dim), dtype=np.float32)
        for action_dim in range(self.action_dim):
            a = actions[:, action_dim]
            for target in range(self.state_dim):
                self.effects[action_dim, target] = _corr(a, y[:, target])

    def predict(
        self,
        state: np.ndarray,
        cell: tuple[int, ...],
        action_dim: int,
        action_sign: int,
        pulse: float,
        state_dim: int,
    ) -> tuple[ProbeAtom, np.ndarray, bool]:
        del state, pulse, state_dim
        effect = self.effects[action_dim] * float(action_sign)
        target = int(np.argmax(np.abs(effect)))
        direction = 1 if effect[target] >= 0 else -1
        return ProbeAtom(cell, action_dim, action_sign, target, direction), effect, True


class _LinearPredictor:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    def predict(
        self,
        state: np.ndarray,
        cell: tuple[int, ...],
        action_dim: int,
        action_sign: int,
        pulse: float,
        state_dim: int,
    ) -> tuple[ProbeAtom, np.ndarray, bool]:
        action = np.zeros(self.weights.shape[0] - state_dim - 1, dtype=np.float32)
        action[action_dim] = pulse * action_sign
        zero = np.zeros_like(action)
        x1 = np.concatenate([state, action, [1.0]]).astype(np.float32)
        x0 = np.concatenate([state, zero, [1.0]]).astype(np.float32)
        effect = (x1 @ self.weights) - (x0 @ self.weights)
        target = int(np.argmax(np.abs(effect)))
        direction = 1 if effect[target] >= 0 else -1
        return ProbeAtom(cell, action_dim, action_sign, target, direction), effect, True


class _NeuralPredictor:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle

    def predict(
        self,
        state: np.ndarray,
        cell: tuple[int, ...],
        action_dim: int,
        action_sign: int,
        pulse: float,
        state_dim: int,
    ) -> tuple[ProbeAtom, np.ndarray, bool]:
        import torch

        model = self.bundle["model"]
        x_mean = self.bundle["x_mean"]
        x_std = self.bundle["x_std"]
        y_mean = self.bundle["y_mean"]
        y_std = self.bundle["y_std"]
        action_dim_count = x_mean.shape[-1] - state_dim
        action = np.zeros(action_dim_count, dtype=np.float32)
        action[action_dim] = pulse * action_sign
        zero = np.zeros_like(action)
        batch = np.stack(
            [
                np.concatenate([state, action]),
                np.concatenate([state, zero]),
            ]
        ).astype(np.float32)
        x = torch.from_numpy((batch - x_mean) / x_std).to(self.bundle["device"])
        model.eval()
        with torch.no_grad():
            pred = model(x).cpu().numpy() * y_std + y_mean
        effect = pred[0] - pred[1]
        target = int(np.argmax(np.abs(effect)))
        direction = 1 if effect[target] >= 0 else -1
        return ProbeAtom(cell, action_dim, action_sign, target, direction), effect, True


class _MetricMemoryPredictor:
    def __init__(
        self,
        memory: dict[tuple[tuple[int, ...], int, int], ProbeAtom],
        coverage_cells: set[tuple[int, ...]],
    ) -> None:
        self.memory = memory
        self.coverage_cells = coverage_cells

    def predict(
        self,
        state: np.ndarray,
        cell: tuple[int, ...],
        action_dim: int,
        action_sign: int,
        pulse: float,
        state_dim: int,
    ) -> tuple[ProbeAtom, np.ndarray, bool]:
        del state, pulse
        key = (cell, action_dim, action_sign)
        atom = self.memory.get(key)
        covered = atom is not None
        if atom is None:
            candidates = [
                (candidate_key, candidate_atom)
                for candidate_key, candidate_atom in self.memory.items()
                if candidate_key[1] == action_dim and candidate_key[2] == action_sign
            ]
            nearest_key, atom = min(candidates, key=lambda item: _cell_distance(cell, item[0][0]))
            del nearest_key
        effect = np.zeros(state_dim, dtype=np.float32)
        effect[atom.target] = float(atom.direction)
        return ProbeAtom(cell, action_dim, action_sign, atom.target, atom.direction), effect, covered


def _fit_linear(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    lhs = xb.T @ xb + ridge * np.eye(xb.shape[1], dtype=np.float32)
    rhs = xb.T @ y
    return np.linalg.solve(lhs, rhs)


def _fit_neural(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    epochs: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True) + 1e-6
    y_mean = train_y.mean(axis=0, keepdims=True)
    y_std = train_y.std(axis=0, keepdims=True) + 1e-6
    tx = torch.from_numpy((train_x - x_mean) / x_std).float().to(device)
    ty = torch.from_numpy((train_y - y_mean) / y_std).float().to(device)
    vx = torch.from_numpy((val_x - x_mean) / x_std).float().to(device)
    vy = torch.from_numpy((val_y - y_mean) / y_std).float().to(device)
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, train_y.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_curve = []
    val_curve = []
    for _ in range(epochs):
        perm = torch.randperm(tx.shape[0], device=device)
        model.train()
        for start in range(0, tx.shape[0], 256):
            idx = perm[start : start + 256]
            pred = model(tx[idx])
            loss = torch.mean((pred - ty[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            train_curve.append(float(torch.mean((model(tx) - ty) ** 2).cpu()))
            val_curve.append(float(torch.mean((model(vx) - vy) ** 2).cpu()))
    return (
        {
            "model": model,
            "device": device,
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "y_mean": y_mean.astype(np.float32),
            "y_std": y_std.astype(np.float32),
        },
        {"train": train_curve, "val": val_curve},
    )


def _score_method(
    method: str,
    predictions: list[ProbeAtom],
    truth: list[ProbeAtom],
    pred_effects: list[np.ndarray],
    true_effects: list[np.ndarray],
    covered: list[bool],
) -> MethodScore:
    target_correct = []
    exact_correct = []
    ranks = []
    effect_errors = []
    for pred, true, pred_effect, true_effect in zip(
        predictions,
        truth,
        pred_effects,
        true_effects,
    ):
        target_correct.append(pred.target == true.target)
        exact_correct.append(pred.key_without_cell() == true.key_without_cell())
        order = list(np.argsort(-np.abs(pred_effect)))
        ranks.append(float(order.index(true.target) + 1 if true.target in order else len(order)))
        pred_norm = pred_effect / (np.linalg.norm(pred_effect) + 1e-8)
        true_norm = true_effect / (np.linalg.norm(true_effect) + 1e-8)
        effect_errors.append(float(np.linalg.norm(pred_norm - true_norm)))
    return MethodScore(
        method=method,
        target_accuracy=mean(target_correct),
        exact_accuracy=mean(exact_correct),
        mean_rank=mean(ranks),
        mean_effect_error=mean(effect_errors),
        cell_coverage=mean(covered),
    )


def _score_readout_targets(
    method: str,
    predictions: list[ProbeAtom],
    truth: list[ProbeAtom],
    pred_effects: list[np.ndarray],
    state_dim: int,
    readout_scale: float,
    filter_readouts: bool,
) -> ReadoutScore:
    direct_correct = []
    exact_correct = []
    readout_false_positive = []
    for pred, true, pred_effect in zip(predictions, truth, pred_effects):
        if filter_readouts:
            selected_target = pred.target
            selected_direction = pred.direction
            selected_readout = False
        else:
            observed_effect = np.concatenate([pred_effect, readout_scale * pred_effect])
            observed_target = int(np.argmax(np.abs(observed_effect)))
            selected_readout = observed_target >= state_dim
            selected_target = observed_target if not selected_readout else observed_target - state_dim
            selected_direction = 1 if observed_effect[observed_target] >= 0 else -1
        direct_correct.append((not selected_readout) and selected_target == true.target)
        exact_correct.append(
            (not selected_readout)
            and pred.action_dim == true.action_dim
            and pred.action_sign == true.action_sign
            and selected_target == true.target
            and selected_direction == true.direction
        )
        readout_false_positive.append(selected_readout and selected_target == true.target)
    return ReadoutScore(
        method=method,
        direct_accuracy=mean(direct_correct),
        exact_accuracy=mean(exact_correct),
        readout_false_positive_rate=mean(readout_false_positive),
    )


def _aggregate_scores(method: str, scores: list[MethodScore]) -> dict[str, object]:
    return {
        "method": method,
        "target_accuracy": mean(score.target_accuracy for score in scores),
        "exact_accuracy": mean(score.exact_accuracy for score in scores),
        "mean_rank": mean(score.mean_rank for score in scores),
        "mean_effect_error": mean(score.mean_effect_error for score in scores),
        "cell_coverage": mean(score.cell_coverage for score in scores),
    }


def _aggregate_readout_scores(
    method: str,
    scores: list[ReadoutScore],
) -> dict[str, object]:
    return {
        "method": method,
        "direct_accuracy": mean(score.direct_accuracy for score in scores),
        "exact_accuracy": mean(score.exact_accuracy for score in scores),
        "readout_false_positive_rate": mean(
            score.readout_false_positive_rate for score in scores
        ),
    }


def _apply_actuator_polarity_shift(env: Any) -> None:
    env.unwrapped.model.actuator_gear[:, 0] *= -1.0


def _finite_difference_effect(
    env: Any,
    state: np.ndarray,
    action_dim: int,
    action_sign: int,
    pulse: float,
) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[action_dim] = pulse * action_sign
    zero = np.zeros_like(action)
    next_action = _step_from_state(env, state, action)
    next_zero = _step_from_state(env, state, zero)
    return next_action - next_zero


def _step_from_state(env: Any, state: np.ndarray, action: np.ndarray) -> np.ndarray:
    nq = int(env.unwrapped.model.nq)
    qpos = state[:nq].copy()
    qvel = state[nq:].copy()
    env.unwrapped.set_state(qpos, qvel)
    env.step(action)
    return _state_vector(env)


def _state_vector(env: Any) -> np.ndarray:
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    return np.concatenate([qpos, qvel]).astype(np.float32)


def _state_dim(env: Any) -> int:
    return int(env.unwrapped.model.nq + env.unwrapped.model.nv)


def _target_names(env: Any) -> list[str]:
    nq = int(env.unwrapped.model.nq)
    nv = int(env.unwrapped.model.nv)
    return [f"qpos{i}" for i in range(nq)] + [f"qvel{i}" for i in range(nv)]


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom <= 1e-12:
        return 0.0
    return float(x0 @ y0 / denom)


def _cell_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(abs(x - y) for x, y in zip(a, b))


def _make_figures(payload: dict[str, object], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = payload["summary"]  # type: ignore[assignment]
    methods = [str(row["method"]) for row in rows]  # type: ignore[index]
    target = [float(row["target_accuracy"]) for row in rows]  # type: ignore[index]
    exact = [float(row["exact_accuracy"]) for row in rows]  # type: ignore[index]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.36
    ax.bar(x - width / 2, target, width, label="target")
    ax.bar(x + width / 2, exact, width, label="target+sign")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(_short_method(method) for method in methods)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.set_title("MuJoCo frozen mechanism probes")
    fig.tight_layout()
    fig.savefig(figure_dir / "mujoco_probe_metrics.png", dpi=220)
    fig.savefig(figure_dir / "mujoco_probe_metrics.pdf")
    plt.close(fig)

    curves = payload["training_curves"]  # type: ignore[assignment]
    train = np.asarray([curve["train"] for curve in curves], dtype=float)  # type: ignore[index]
    val = np.asarray([curve["val"] for curve in curves], dtype=float)  # type: ignore[index]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(train.mean(axis=0), label="train")
    ax.plot(val.mean(axis=0), label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("standardized MSE")
    ax.set_title("Neural dynamics predictor training")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "mujoco_neural_training.png", dpi=220)
    fig.savefig(figure_dir / "mujoco_neural_training.pdf")
    plt.close(fig)

    target_names = payload["target_names"]  # type: ignore[assignment]
    action_dim = max(item["action_dim"] for item in payload["heatmap_counts"]) + 1  # type: ignore[index]
    matrix = np.zeros((2 * action_dim, len(target_names)), dtype=float)
    for item in payload["heatmap_counts"]:  # type: ignore[assignment]
        row = int(item["action_dim"]) * 2 + (0 if int(item["action_sign"]) < 0 else 1)
        matrix[row, int(item["target"])] = float(item["count"])
    matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(2 * action_dim))
    ax.set_yticklabels(
        f"a{dim}{sign}" for dim in range(action_dim) for sign in ("-", "+")
    )
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_title("Ground-truth top response frequency")
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "mujoco_probe_heatmap.png", dpi=220)
    fig.savefig(figure_dir / "mujoco_probe_heatmap.pdf")
    plt.close(fig)


def _make_causal_check_figures(payload: dict[str, object], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    context = payload["context_shift"]  # type: ignore[index]
    context_rows = context["summary"]  # type: ignore[index]
    methods = [str(row["method"]) for row in context_rows]  # type: ignore[index]
    target = [float(row["target_accuracy"]) for row in context_rows]  # type: ignore[index]
    exact = [float(row["exact_accuracy"]) for row in context_rows]  # type: ignore[index]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    width = 0.36
    ax.bar(x - width / 2, target, width, label="target")
    ax.bar(x + width / 2, exact, width, label="target+sign")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(_short_causal_check_method(method) for method in methods)
    ax.set_title("Actuator-polarity context shift")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / "mujoco_context_shift.png", dpi=220)
    fig.savefig(figure_dir / "mujoco_context_shift.pdf")
    plt.close(fig)

    readout = payload["readout_shift"]  # type: ignore[index]
    readout_rows = readout["summary"]  # type: ignore[index]
    methods = [str(row["method"]) for row in readout_rows]  # type: ignore[index]
    direct = [float(row["direct_accuracy"]) for row in readout_rows]  # type: ignore[index]
    fp = [
        float(row["readout_false_positive_rate"])
        for row in readout_rows  # type: ignore[index]
    ]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    ax.bar(x - width / 2, direct, width, label="direct target")
    ax.bar(x + width / 2, fp, width, label="readout false target")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_xticks(x)
    ax.set_xticklabels(_short_causal_check_method(method) for method in methods)
    ax.set_title("Prediction under amplified readouts")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / "mujoco_readout_shift.png", dpi=220)
    fig.savefig(figure_dir / "mujoco_readout_shift.pdf")
    plt.close(fig)


def _try_render_frame(env_id: str, figure_dir: Path, seed: int) -> None:
    try:
        import gymnasium as gym
        from PIL import Image

        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed)
        frame = env.render()
        if frame is not None:
            safe_env = env_id.lower().replace("-", "_")
            Image.fromarray(frame).save(figure_dir / f"mujoco_{safe_env}_frame.png")
        env.close()
    except Exception as exc:
        (figure_dir / "mujoco_render_error.txt").write_text(str(exc), encoding="utf-8")


def _short_method(method: str) -> str:
    return {
        "random-correlation": "corr.",
        "global-linear-dynamics": "linear",
        "neural-dynamics": "neural",
        "metric-causal-core": "metric core",
    }.get(method, method)


def _short_causal_check_method(method: str) -> str:
    return {
        "source-linear dynamics": "source linear",
        "source-neural dynamics": "source neural",
        "source-metric core": "source metric",
        "few-shot metric core": "few-shot metric",
        "linear observed selector": "linear observed",
        "neural observed selector": "neural observed",
        "metric core observed selector": "metric observed",
        "metric core with readout filter": "metric filtered",
    }.get(method, method)
