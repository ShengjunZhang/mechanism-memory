"""Local paired-loss scores with source-only scaling and unchanged sign maps."""
import numpy as np


def source_weights(source_targets, exponent):
    energy = np.mean(np.asarray(source_targets, dtype=np.float64) ** 2, axis=0)
    positive = energy[energy > 1e-12]
    floor = max(float(np.median(positive)) * .01, 1e-12) if len(positive) else 1.
    weights = np.maximum(energy, floor) ** (-exponent)
    return weights / weights.mean()


def choose_scores(predictions, observed, budget, maximum, ratios, thresholds, source_targets):
    if budget < 1 or budget > maximum or len(predictions) % maximum:
        raise ValueError("Invalid per-actuator budget")
    dim = len(predictions) // maximum
    p = np.asarray(predictions, dtype=np.float64).reshape(dim, maximum, -1)[:, :budget]
    y = np.asarray(observed, dtype=np.float64).reshape(dim, maximum, -1)[:, :budget]
    maps = {"frozen map": np.ones(dim, dtype=np.float32)}
    for exponent in (0, 1):
        w = source_weights(source_targets, exponent)
        keep = np.mean(w * (p - y) ** 2, axis=(1, 2))
        flip = np.mean(w * (-p - y) ** 2, axis=(1, 2))
        energy = np.mean(w * p ** 2, axis=(1, 2))
        for ratio in ratios:
            maps[f"ratio-w{exponent}-{ratio:g}"] = np.where(flip < ratio * keep, -1., 1.).astype(np.float32)
        # Squared measurement noise cancels in the paired loss difference.
        # Normalization uses predictions, not latent noise levels or test labels.
        for threshold in thresholds:
            maps[f"gain-w{exponent}-{threshold:g}"] = np.where(
                (keep - flip) > 4 * threshold * energy + 1e-12, -1., 1.).astype(np.float32)
    maps["matched prior"] = maps["ratio-w0-1"]
    maps["weighted matched prior"] = maps["ratio-w1-1"]
    maps["fixed gate"] = maps["ratio-w0-0.5"]
    return maps
