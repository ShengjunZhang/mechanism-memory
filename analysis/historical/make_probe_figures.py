from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE if (HERE / "SourceData_Fig2_fixed_probe_outcomes.csv").exists() else HERE.parent / "source_data"
workspace_figures = HERE.parent / "manuscript" / "figures"
OUT = workspace_figures if workspace_figures.exists() else HERE / "generated_figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#172033"
MUTED = "#64748B"
AXIS = "#334155"
BLUE = "#1479B8"
GREEN = "#20A37D"
CORAL = "#E76F51"
RED = "#D9474F"
GRAY = "#A7B0B9"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.linewidth": 0.8,
        "axes.edgecolor": AXIS,
        "axes.facecolor": "white",
        "axes.grid": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style_axis(axis: plt.Axes, *, keep_left: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(keep_left)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.spines["left"].set_linewidth(0.8)
    axis.tick_params(colors=AXIS, labelsize=7.1, width=0.8)


def annotate_bars(axis: plt.Axes, bars, values, *, fmt: str, limit: float) -> None:
    for bar, value in zip(bars, values):
        x = float(bar.get_width())
        inside = x > 0.35 * limit
        axis.text(
            x - 0.018 * limit if inside else x + 0.018 * limit,
            bar.get_y() + bar.get_height() / 2,
            format(value, fmt),
            ha="right" if inside else "left",
            va="center",
            color="white" if inside else INK,
            fontsize=7.0,
        )


def make_overview() -> None:
    rows = read_rows(DATA / "SourceData_Fig2_fixed_probe_outcomes.csv")
    values = {(row["panel"], row["method"], row["metric"]): float(row["value"]) for row in rows}

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    coarse = [
        ("passive", "passive", CORAL),
        ("reward", "reward", CORAL),
        ("conditional_oracle", "conditional", GREEN),
        ("tuple_oracle", "tuple", MUTED),
        ("world_model_oracle", "world model", "#64A70B"),
        ("transition_oracle", "transition", "#8296C3"),
        ("causal_core", "Causal Core", BLUE),
    ]
    y = np.arange(len(coarse))[::-1]
    vals = [values[("coarse", key, "target_edge_f1")] for key, _, _ in coarse]
    bars = ax_a.barh(y, vals, color=[color for _, _, color in coarse], height=0.55)
    ax_a.set_yticks(y, [label for _, label, _ in coarse])
    ax_a.set_xlim(0, 0.9)
    ax_a.set_xlabel("target-edge F1")
    ax_a.set_title("Coarse interfaces", fontsize=9.0, pad=6)
    annotate_bars(ax_a, bars, vals, fmt=".2f", limit=0.9)
    style_axis(ax_a)

    write = [
        ("ungated_qwen", "ungated Qwen"),
        ("causal_prompt", "causal prompt"),
        ("control_gate", "Qwen + gate"),
    ]
    y = np.arange(len(write))[::-1]
    fp = [values[("write", key, "false_edges")] for key, _ in write]
    ro = [values[("write", key, "readout_false_edges")] for key, _ in write]
    bars_fp = ax_b.barh(y + 0.16, fp, height=0.28, color=GRAY, label="false edges")
    bars_ro = ax_b.barh(y - 0.16, ro, height=0.28, color=RED, label="readout false edges")
    ax_b.set_yticks(y, [label for _, label in write])
    ax_b.set_xlim(0, 105)
    ax_b.set_xlabel("count")
    ax_b.set_title("Write validity", fontsize=9.0, pad=6)
    for bars, vals in ((bars_fp, fp), (bars_ro, ro)):
        for bar, value in zip(bars, vals):
            if value > 0:
                ax_b.text(value + 2, bar.get_y() + bar.get_height() / 2, f"{value:.0f}", va="center", fontsize=7.0, color=INK)
    ax_b.legend(frameon=False, fontsize=6.6, loc="lower right", handlelength=1.2)
    style_axis(ax_b)

    adapt = [
        ("ranking_renamed", "ranking, renamed", CORAL),
        ("core_renamed", "core, renamed", BLUE),
        ("ranking_readout", "ranking, readout", CORAL),
        ("core_readout", "core, readout", GREEN),
    ]
    y = np.arange(len(adapt))[::-1]
    vals = [values[("adapt", key, "adapt_score")] for key, _, _ in adapt]
    bars = ax_c.barh(y, vals, color=[color for _, _, color in adapt], height=0.55)
    ax_c.set_yticks(y, [label for _, label, _ in adapt])
    ax_c.set_xlim(0, 1.05)
    ax_c.set_xlabel("AdaptScore")
    ax_c.set_title("Selective adaptation", fontsize=9.0, pad=6)
    annotate_bars(ax_c, bars, vals, fmt=".2f", limit=1.05)
    style_axis(ax_c)

    proposal = [
        ("causal_prompt", "causal prompt", CORAL),
        ("llm_context", "LLM + context", GREEN),
        ("llm_control", "LLM + control", BLUE),
        ("context_planner", "context planner", GRAY),
        ("control_planner", "control planner", MUTED),
    ]
    y = np.arange(len(proposal))[::-1]
    vals = [values[("proposal", key, "target_edge_f1")] for key, _, _ in proposal]
    bars = ax_d.barh(y, vals, color=[color for _, _, color in proposal], height=0.55)
    ax_d.set_yticks(y, [label for _, label, _ in proposal])
    ax_d.set_xlim(0, 0.8)
    ax_d.set_xlabel("target-edge F1")
    ax_d.set_title("Proposal policies", fontsize=9.0, pad=6)
    annotate_bars(ax_d, bars, vals, fmt=".2f", limit=0.8)
    style_axis(ax_d)

    for label, axis in zip("abcd", axes.ravel()):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontsize=10.0, fontweight="bold", color=INK)

    fig.subplots_adjust(left=0.20, right=0.98, top=0.94, bottom=0.10, hspace=0.48, wspace=0.48)
    fig.savefig(OUT / "main_results_v10.pdf", bbox_inches="tight")
    fig.savefig(OUT / "main_results_v10.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_llm() -> None:
    rows = read_rows(DATA / "SourceData_Table5_llm_proposal_policies.csv")
    by_variant = {row["variant"]: row for row in rows}
    variants = [
        ("llm-vanilla", "LLM-only", GRAY),
        ("llm-causal-prompt", "causal\nprompt", CORAL),
        ("llm-context-search-gated", "LLM +\ncontext gate", GREEN),
        ("llm-control-planner-gated", "LLM +\ncontrol gate", GREEN),
        ("causal-core-context-search-planner", "context\nplanner", BLUE),
        ("causal-core-control-experiment-planner", "control\nplanner", BLUE),
    ]
    labels = [label for _, label, _ in variants]
    f1 = np.array([float(by_variant[key]["f1"]) for key, _, _ in variants])
    fp = np.array([float(by_variant[key]["false_positive"]) for key, _, _ in variants])
    ro = np.array([float(by_variant[key]["readout_false_positive"]) for key, _, _ in variants])
    y = np.arange(len(variants))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.35), gridspec_kw={"width_ratios": [1.05, 1.0]})
    bars = ax1.barh(y, f1, color=[color for _, _, color in variants], height=0.62)
    ax1.set_yticks(y, labels)
    ax1.set_xlim(0, 0.82)
    ax1.set_xlabel("target-edge F1")
    ax1.set_title("Frozen target probes", fontsize=8.6, pad=5)
    annotate_bars(ax1, bars, f1, fmt=".2f", limit=0.82)
    style_axis(ax1, keep_left=False)
    ax1.tick_params(axis="y", length=0)

    ax2.barh(y + 0.15, fp, height=0.28, color=GRAY, label="false edges")
    ax2.barh(y - 0.15, ro, height=0.28, color=RED, label="readout false edges")
    for yi, value in zip(y + 0.15, fp):
        ax2.text(value + 1.5, yi, f"{value:.0f}", va="center", fontsize=6.8, color=INK)
    for yi, value in zip(y - 0.15, ro):
        if value > 0:
            ax2.text(value + 1.5, yi, f"{value:.0f}", va="center", fontsize=6.8, color=INK)
    ax2.set_yticks(y, [])
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("count")
    ax2.set_title("False writes", fontsize=8.6, pad=5)
    ax2.legend(frameon=False, fontsize=6.5, loc="lower right", handlelength=1.2)
    style_axis(ax2, keep_left=False)

    fig.subplots_adjust(left=0.20, right=0.98, top=0.84, bottom=0.22, wspace=0.28)
    fig.savefig(OUT / "llm_causal_suite.pdf", bbox_inches="tight")
    fig.savefig(OUT / "llm_causal_suite.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_overview()
    make_llm()
