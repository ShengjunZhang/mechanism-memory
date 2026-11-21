from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "SourceData_Fig4_llm_hidden_retention.csv"
workspace_figures = HERE.parent / "manuscript" / "figures"
OUT = workspace_figures if workspace_figures.exists() else HERE / "generated_figures"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "TF-IDF query only",
    "Qwen calibrated Yes/No logits",
    "Qwen linear layer 28",
    "Qwen nonlinear RBF",
    "paired-evidence rule",
    "gated mechanism state",
]
LABELS = [
    "query only",
    "Qwen logits",
    "linear hidden",
    "nonlinear hidden",
    "paired evidence",
    "gated state",
]
COLORS = ["#A7B0B9", "#E76F51", "#3F6FB6", "#20A37D", "#64748B", "#172033"]


def load_rows() -> dict[tuple[str, str], dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8-sig") as handle:
        return {
            (row["method"], row["group"]): row
            for row in csv.DictReader(handle)
        }


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.7),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )

    groups = ["source-clean", "renamed-shift", "semantic-readout"]
    group_labels = ["source", "renamed shift", "unseen readout"]
    x = np.arange(len(groups))
    width = 0.13
    for index, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        values = [number(rows[(method, group)], "balanced_accuracy") for group in groups]
        errors = [number(rows[(method, group)], "family_se") for group in groups]
        axes[0].bar(
            x + (index - (len(METHODS) - 1) / 2) * width,
            values,
            width,
            label=label,
            color=color,
            yerr=errors,
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
    axes[0].axhline(0.5, color="#64748B", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(x, group_labels)
    axes[0].set_ylim(0.45, 1.02)
    axes[0].set_ylabel("balanced accuracy")
    axes[0].set_title("Frozen mechanism probes", fontsize=8.8)

    shifted = [number(rows[(method, "shifted-delay")], "balanced_accuracy") for method in METHODS]
    shifted_se = [number(rows[(method, "shifted-delay")], "family_se") for method in METHODS]
    readout = [
        number(rows[(method, "semantic-readout-candidates")], "false_positive_rate")
        for method in METHODS
    ]
    readout_se = [
        number(rows[(method, "semantic-readout-candidates")], "family_se")
        for method in METHODS
    ]
    y = np.arange(len(METHODS))[::-1]
    axes[1].barh(
        y + 0.16,
        shifted,
        height=0.28,
        color="#20A37D",
        label="shifted delay BA",
        xerr=shifted_se,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    axes[1].barh(
        y - 0.16,
        readout,
        height=0.28,
        color="#D9474F",
        label="readout FPR",
        xerr=readout_se,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    axes[1].set_yticks(y, LABELS)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_xlabel("rate")
    axes[1].set_title("Mechanism shift and readout", fontsize=8.8)
    axes[1].legend(
        frameon=False,
        fontsize=6.6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(width=0.8, labelsize=7.1)
    axes[0].legend(
        frameon=False,
        fontsize=6.5,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.30, wspace=0.35)
    fig.savefig(OUT / "llm_hidden_retention.pdf", bbox_inches="tight")
    fig.savefig(OUT / "llm_hidden_retention.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
