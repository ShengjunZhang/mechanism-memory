"""Plot the locked comparison, its tradeoffs and all primary-budget conditions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path("experiments/tdmpc2_sequential_20261116")


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    result = json.loads((ROOT / "verified_analysis.json").read_text())
    selected = result["selected"]
    arrays = []
    for task in protocol["tasks"]:
        for seed in protocol["confirmation_seeds"]:
            path = ROOT / "results" / "confirmation" / f"{task}_{seed}"
            meta = json.loads(path.with_suffix(".json").read_text())
            with np.load(path.with_suffix(".npz")) as data:
                arrays.append(data["metrics"])
    data = np.stack(arrays)
    method_index = {m: i for i, m in enumerate(meta["method_names"])}
    ours, reference = method_index[selected["memory"]], method_index[selected["baseline"]]
    b = protocol["budgets"].index(protocol["primary_budget"])
    rng = np.random.default_rng(20261117)
    indices = rng.integers(0, 2, size=(20000, 5, 2))
    def ci(values):
        v = np.asarray(values).reshape(5, 2)
        samples = np.take_along_axis(np.broadcast_to(v, (len(indices), 5, 2)), indices, axis=2).mean(axis=(1, 2))
        return v.mean(), *np.quantile(samples, [.025, .975])
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.linewidth": .8,
                         "xtick.major.width": .8, "ytick.major.width": .8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "svg.fonttype": "none", "savefig.dpi": 220})
    teal, red, blue, grey, purple = "#008775", "#C94A29", "#4A7BB0", "#70777D", "#8B6BA5"
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    fig.subplots_adjust(left=.09, right=.96, bottom=.10, top=.92, wspace=.36, hspace=.48)
    for ax, letter in zip(axes.ravel(), "abcd"):
        ax.text(-.14, 1.10, letter, transform=ax.transAxes, fontweight="bold", fontsize=13)
    primary_curve = []
    for i, budget in enumerate(protocol["budgets"]):
        diff = (data[:, ours, :, :, i, :, :, 0] - data[:, reference, :, :, i, :, :, 0]).mean(axis=(1, 2, 3, 4))
        point, low, high = ci(diff)
        primary_curve.append({"budget": budget, "difference": float(point), "ci95": [float(low), float(high)]})
    curve = np.array([[r["difference"], *r["ci95"]] for r in primary_curve])
    ax = axes[0, 0]
    ax.axhline(0, color=grey, lw=.8, ls="--")
    ax.errorbar(protocol["budgets"], curve[:, 0], yerr=[curve[:, 0] - curve[:, 1], curve[:, 2] - curve[:, 0]],
                color=teal, lw=1.4, marker="o", ms=5, capsize=4)
    ax.set(xlabel="Pairs per actuator per round", ylabel="NRMSE difference", xticks=protocol["budgets"],
           title="Memory minus selected baseline")
    ax.text(.49, .06, "Lower values favor memory", transform=ax.transAxes, color=grey, fontsize=9)
    ax = axes[0, 1]
    points = [("Selective memory", "Memory", teal, (-10, -18)),
              ("Markov matched prior", "Markov", red, (7, -3)),
              ("No gate", "No gate", blue, (8, -1)),
              ("Reset memory", "Reset", purple, (-32, -14)),
              ("Default memory gate", "Default gate", grey, (-55, 8))]
    for key, label, color, offset in points:
        values = result["primary"][key]
        x, y = 100 * values["stable_error"], 100 * values["change_accuracy"]
        ax.scatter(x, y, s=42, c=color, edgecolors="white", linewidths=.6, zorder=3)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", color=color, fontsize=9)
    ax.set(xlabel="Stable-entry error (%)", ylabel="Changed-entry accuracy (%)", xlim=(.3, 4.1), ylim=(72, 102),
           title="Preservation and adaptation")
    ax = axes[1, 0]
    condition = np.array([[next(c["difference"]["mean"] for c in result["conditions"]
                                if c["schedule"] == s and c["noise"] == n and c["budget"] == protocol["primary_budget"])
                           for n in protocol["noise_levels"]] for s in protocol["schedules"]])
    limit = max(abs(condition.min()), abs(condition.max()))
    im = ax.imshow(condition, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), cmap="RdBu_r", aspect="auto")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{condition[i,j]:+.4f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(condition[i,j]) > limit * .65 else "#222222")
    ax.set(xticks=range(4), xticklabels=protocol["noise_levels"], yticks=range(4),
           yticklabels=[s.capitalize() for s in protocol["schedules"]], xlabel="Measurement-noise scale",
           title="NRMSE difference at budget 3")
    ax.tick_params(length=0)
    colorbar = fig.colorbar(im, ax=ax, fraction=.046, pad=.035)
    colorbar.outline.set_linewidth(.8)
    ax = axes[1, 1]
    curves = {}
    for name, label, color in ((selected["memory"], "Memory", teal), (selected["baseline"], "Markov", red),
                               ("window-1", "Roundwise", blue),
                               (selected["memory"].replace("memory", "reset"), "Reset", purple)):
        values = data[:, method_index[name], :, :, b, :, :, 0].mean(axis=(0, 1, 2, 3))
        curves[label] = values.tolist()
        ax.plot(np.arange(1, 13), values, color=color, lw=1.35, marker="o", ms=3, label=label)
    for change in protocol["change_rounds_zero_based"]:
        ax.axvline(change + 1, lw=.7, ls=":", color="#BBBBBB", zorder=0)
    ax.set(xlabel="Round", ylabel="Response NRMSE", xticks=[1, 4, 7, 10, 12], title="Sequential response prediction")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.savefig(ROOT / "sequential_comparison.png")
    fig.savefig(ROOT / "sequential_comparison.svg")
    plt.close(fig)
    (ROOT / "figure_data.json").write_text(json.dumps({"budget_comparison": primary_curve, "round_curves": curves,
                                                       "condition_matrix": condition.tolist()}, indent=2) + "\n")
    print(json.dumps(primary_curve, indent=2))


if __name__ == "__main__":
    main()
