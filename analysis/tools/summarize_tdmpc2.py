"""Build tables and figures from completed checkpoint experiments."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from causal_sandbox.world_model_probes import cluster_summary


def main():
    revision = ROOT / "data/controlled"
    results = revision / "results" / "tdmpc2"
    figure_dir = revision / "figures"
    figure_dir.mkdir(exist_ok=True)
    all_runs, all_summaries = {}, {}
    for setting in ("default", "pulse010", "pulse040"):
        folder = results / setting
        if not folder.exists():
            continue
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("tdmpc2_*.json"))]
        if not runs:
            continue
        hashes = [r["checkpoint_sha256"] for r in runs]
        if len(set(hashes)) != len(hashes):
            raise ValueError("Expected distinct independently trained checkpoint hashes")
        rows = [row | {"decoder": "source-validation-selected"}
                for run in runs for row in run["rows"] if row["source_selected_decoder"]]
        summary = cluster_summary(rows)
        all_runs[setting], all_summaries[setting] = runs, summary
    if "default" not in all_runs:
        raise ValueError("No default experiments available")

    summaries = all_summaries["default"]
    default = [r for r in summaries if r["budget"] == 5
               and (r["ratio"] in (None, 0.5)) and (r["alpha"] in (None, 5.0))]
    columns = ["condition", "method", "budget", "independent_agents", "nrmse", "nrmse_lo", "nrmse_hi",
               "sign_accuracy", "sign_lo", "sign_hi", "target_accuracy"]
    with (results / "main_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in default:
            writer.writerow({k: row[k] for k in columns if k in row} | {
                "nrmse_lo": row["nrmse_ci95"][0], "nrmse_hi": row["nrmse_ci95"][1],
                "sign_lo": row["sign_accuracy_ci95"][0], "sign_hi": row["sign_accuracy_ci95"][1]})

    manifest = {"settings": {}, "interpretation": [
        "Confidence intervals resample independent training checkpoints within each fixed task.",
        "There are three independently trained official agents per task, not repeated evaluation seeds of one agent.",
        "Matched sign and one-flip baselines use the same target pairs as the selective update.",
        "Effect-sign accuracy concerns the largest-magnitude true-response coordinate.",
        "No target evaluation labels are used for selecting decoder, ratio, or regularization.",
        "Cartpole has one actuator and therefore has no nonzero direction excluding the shifted actuator; stable-direction summaries exclude it."]}
    for setting, runs in all_runs.items():
        manifest["settings"][setting] = {"independent_agents": len(runs),
            "tasks": sorted({r["task"] for r in runs}),
            "checkpoint_hashes": [r["checkpoint_sha256"] for r in runs],
            "primary": all_summaries[setting]}
    (results / "combined_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.linewidth": .8, "axes.grid": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    methods = ["frozen", "global residual", "matched sign prior", "one-flip prior", "selective update", "oracle map"]
    labels = ["Frozen", "Global", "Sign calibration", "One-flip calibration", "Selective", "Oracle"]
    colors = ["#A6ADB4", "#E26A4A", "#7D8FA9", "#64A3A2", "#159A81", "#3479B8"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for ax, condition, metric, ylabel in zip(axes, ("target-shifted", "target-stable"),
                                           ("sign_accuracy", "nrmse"),
                                           ("Shifted-axis sign accuracy", "Stable-direction NRMSE")):
        lookup = {r["method"]: r for r in default if r["condition"] == condition}
        means = np.array([lookup[m][metric] for m in methods])
        intervals = np.array([lookup[m][metric + "_ci95"] for m in methods])
        ax.bar(range(len(methods)), means, color=colors, width=.7)
        ax.errorbar(range(len(methods)), means,
                    yerr=np.maximum(0, np.stack((means - intervals[:, 0], intervals[:, 1] - means))),
                    fmt="none", color="#222222", capsize=2, linewidth=.8)
        ax.set_xticks(range(len(methods)), labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
    fig.savefig(figure_dir / "tdmpc2_independent_agents.pdf")
    fig.savefig(figure_dir / "tdmpc2_independent_agents.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), layout="constrained")
    axes = axes.ravel()
    lookup = {(r["condition"], r["method"], r["budget"], r["ratio"], r["alpha"]): r for r in summaries}
    for method, ratio, alpha, color, label in [
        ("selective update", .5, None, "#159A81", "Selective"),
        ("global residual", None, 5., "#E26A4A", "Global"),
        ("matched sign prior", None, None, "#7D8FA9", "Sign calibration")]:
        budgets = [1, 3, 5, 10, 20]
        axes[0].plot(budgets, [lookup[("target-shifted", method, b, ratio, alpha)]["sign_accuracy"] for b in budgets],
                     marker="o", ms=3, color=color, label=label)
    axes[0].set(xlabel="Target states per actuator", ylabel="Shifted-axis sign accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    ratios = [.45, .5, .6, .75, .9]
    axes[1].plot(ratios, [lookup[("target-shifted", "selective update", 5, r, None)]["sign_accuracy"] for r in ratios],
                 marker="o", ms=3, color="#159A81")
    axes[1].set(xlabel="Selective loss ratio", ylabel="Shifted-axis sign accuracy")
    alphas = [.1, 1, 5, 20, 100]
    axes[2].plot(alphas, [lookup[("target-shifted", "global residual", 5, None, a)]["sign_accuracy"] for a in alphas],
                 marker="o", ms=3, color="#E26A4A")
    axes[2].set(xscale="log", xlabel="Global ridge regularization", ylabel="Shifted-axis sign accuracy")
    pulse_settings = [("pulse010", .1), ("default", .25), ("pulse040", .4)]
    for method, ratio, alpha, color, label in [
        ("selective update", .5, None, "#159A81", "Selective"),
        ("global residual", None, 5., "#E26A4A", "Global"),
        ("matched sign prior", None, None, "#7D8FA9", "Sign calibration")]:
        points = []
        for setting, pulse in pulse_settings:
            if setting not in all_summaries:
                continue
            match = [r for r in all_summaries[setting] if r["condition"] == "target-shifted"
                     and r["budget"] == 5 and r["method"] == method
                     and r["ratio"] == ratio and r["alpha"] == alpha]
            if match:
                points.append((pulse, match[0]["sign_accuracy"]))
        axes[3].plot([p[0] for p in points], [p[1] for p in points], marker="o", ms=3, color=color, label=label)
    axes[3].set(xlabel="Intervention pulse amplitude", ylabel="Shifted-axis sign accuracy")
    for index, axis in enumerate(axes):
        axis.text(-.15, 1.04, chr(97 + index), transform=axis.transAxes, fontweight="bold")
    fig.savefig(figure_dir / "tdmpc2_sensitivity.pdf")
    fig.savefig(figure_dir / "tdmpc2_sensitivity.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"completed_agents": {k: len(v) for k, v in all_runs.items()}, "primary": default}, indent=2))


if __name__ == "__main__":
    main()
