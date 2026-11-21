"""Causal-prefix local updates and history-aware binary-map controls."""
import numpy as np


def sign_maps(dim):
    return np.array([[1 - 2 * ((code >> j) & 1) for j in range(dim)]
                     for code in range(2 ** dim)], dtype=np.int8)


def map_indices(gears):
    gears = np.asarray(gears)
    return ((gears < 0) * (2 ** np.arange(gears.shape[-1]))).sum(axis=-1)


def schedule_maps(dim, rounds, change_rounds, rng):
    result = {k: np.ones((rounds, dim), dtype=np.int8)
              for k in ("stationary", "sparse", "dense", "recurrent")}
    recurring = rng.choice(dim, (dim + 1) // 2, replace=False)
    for t in range(1, rounds):
        for name in ("sparse", "dense"):
            result[name][t] = result[name][t - 1]
            if t in change_rounds:
                indices = rng.choice(dim, 1 if name == "sparse" else (dim + 1) // 2, replace=False)
                result[name][t, indices] *= -1
        if sum(t >= c for c in change_rounds) % 2:
            result["recurrent"][t, recurring] = -1
    return result


def loss_history(predictions, observations, weights, budget):
    """Inputs have shape [round, actuator, maximum budget, response dimension]."""
    p, y = np.asarray(predictions, float), np.asarray(observations, float)
    if p.shape != y.shape or p.ndim != 4 or not 1 <= budget <= p.shape[2]:
        raise ValueError("Invalid shapes or budget")
    p, y = p[:, :, :budget], y[:, :, :budget]
    return np.stack([np.mean(weights * (s * p - y) ** 2, axis=(2, 3)) for s in (1, -1)], axis=-1)


def estimate_sequence(losses, kind, parameter, variance=1., budget=1):
    """No future rows or evaluator metadata are accessible at an update."""
    losses = np.asarray(losses, float)
    if losses.ndim != 3 or losses.shape[-1] != 2 or not np.all(np.isfinite(losses)):
        raise ValueError("Expected finite [round, actuator, two signs] losses")
    rounds, dim, _ = losses.shape
    current = np.ones(dim, dtype=np.int8)
    output = []
    accumulated = np.zeros((dim, 2))
    probability_negative = np.full(dim, .01)
    for t in range(rounds):
        if kind in ("memory", "reset"):
            if kind == "reset":
                current[:] = 1
            incumbent = (current < 0).astype(int)
            keep = losses[t, np.arange(dim), incumbent]
            replace = losses[t, np.arange(dim), 1 - incumbent]
            accept = replace < parameter * keep
            current[accept] *= -1
        elif kind in ("window", "ewma"):
            if kind == "window":
                score = losses[max(0, t + 1 - parameter):t + 1].sum(axis=0)
            else:
                accumulated = parameter * accumulated + losses[t]
                score = accumulated
            current = np.where(score[:, 0] < score[:, 1], 1,
                               np.where(score[:, 1] < score[:, 0], -1, current)).astype(np.int8)
        elif kind == "markov":
            hazard, strength = parameter
            prior = hazard + (1 - 2 * hazard) * probability_negative
            prior = np.clip(prior, 1e-12, 1 - 1e-12)
            log_odds = np.log(prior) - np.log1p(-prior)
            log_odds += strength * budget * (losses[t, :, 0] - losses[t, :, 1]) / (2 * variance)
            # Bounded log odds avoid numerical saturation without using outcomes.
            log_odds = np.clip(log_odds, -30, 30)
            probability_negative = 1 / (1 + np.exp(-log_odds))
            current = np.where(probability_negative > .5, -1,
                               np.where(probability_negative < .5, 1, current)).astype(np.int8)
        else:
            raise ValueError(kind)
        output.append(current.copy())
    return np.stack(output)


def method_grid(protocol):
    methods = {}
    for ratio in protocol["ratios"]:
        methods[f"memory-{ratio:g}"] = ("memory", ratio)
        methods[f"reset-{ratio:g}"] = ("reset", ratio)
    for window in protocol["windows"]:
        methods[f"window-{window}"] = ("window", window)
    for decay in protocol["decays"]:
        methods[f"ewma-{decay:g}"] = ("ewma", decay)
    for hazard in protocol["hazards"]:
        for strength in protocol["likelihood_strengths"]:
            methods[f"markov-{hazard:g}-{strength:g}"] = ("markov", (hazard, strength))
    return methods
