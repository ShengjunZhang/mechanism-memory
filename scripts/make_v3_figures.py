from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INK = "#0F172A"
MUTED = "#64748B"
AXIS = "#334155"
BORDER = "#CBD5E1"
PALE = "#F8FAFC"
BLUE = "#3B82F6"
BLUE_DARK = "#1D4ED8"
BLUE_LIGHT = "#EFF6FF"
GREEN = "#10B981"
GREEN_DARK = "#065F46"
GREEN_LIGHT = "#ECFDF5"
CORAL = "#F43F5E"
CORAL_DARK = "#BE123C"
CORAL_LIGHT = "#FFF1F2"
GRAY = "#94A3B8"
GRAY_DARK = "#475569"
GRAY_LIGHT = "#F1F5F9"
GRID = "#E2E8F0"
RED = CORAL
RED_LIGHT = CORAL_LIGHT
ORANGE = CORAL
ORANGE_LIGHT = CORAL_LIGHT


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
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def _box(ax, x, y, w, h, text, fc, ec=INK, lw=1.0, size=8.5, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        weight=weight,
        linespacing=1.12,
    )
    return patch


def _arrow(ax, start, end, color=INK, lw=1.0, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    return arrow


def _chip(ax, x, y, text, fc, ec, color=INK, size=7.2):
    patch = FancyBboxPatch(
        (x, y),
        0.118,
        0.046,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=0.75,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + 0.059, y + 0.023, text, ha="center", va="center", fontsize=size, color=color)


def _style_axes(ax, *, left=True, bottom=True):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=MUTED, width=0.8, length=3.0, labelsize=7.2)
    ax.grid(False)


def _value_label(ax, x, y, text, *, color=INK, ha="left", va="center", size=7.0):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color)


def draw_mechanism_memory():
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.020, 0.960, "a", fontsize=10, weight="bold", color=INK)
    ax.text(0.515, 0.960, "b", fontsize=10, weight="bold", color=INK)
    ax.text(0.055, 0.955, "Behavioural equivalence", fontsize=8.8, weight="bold", color=INK)
    ax.text(0.550, 0.955, "Frozen mechanism answers", fontsize=8.8, weight="bold", color=INK)

    # Panel a: quotient obstruction.
    _box(ax, 0.060, 0.690, 0.160, 0.120, "SCM\n$M_0$", BLUE_LIGHT, BLUE, size=9.5, weight="bold")
    _box(ax, 0.060, 0.415, 0.160, 0.120, "SCM\n$M_1$", BLUE_LIGHT, BLUE, size=9.5, weight="bold")
    _box(ax, 0.285, 0.545, 0.170, 0.150, "same\ninteraction law", GRAY_LIGHT, "#8c96a0", size=8.8, weight="bold")
    _arrow(ax, (0.220, 0.750), (0.285, 0.640), color="#7d8790", rad=-0.08)
    _arrow(ax, (0.220, 0.475), (0.285, 0.600), color="#7d8790", rad=0.08)
    ax.text(0.370, 0.505, r"$M_0\sim_{I,\Pi}M_1$", fontsize=9.8, ha="center", va="center", color=INK)

    _chip(ax, 0.278, 0.760, "reward", "#f7f7f3", "#b7bdc4", MUTED)
    _chip(ax, 0.278, 0.705, "prediction", "#f7f7f3", "#b7bdc4", MUTED)
    _chip(ax, 0.278, 0.650, "ranking", "#f7f7f3", "#b7bdc4", MUTED)
    ax.text(0.445, 0.728, "can all agree", fontsize=7.6, color=MUTED, va="center")

    _box(ax, 0.277, 0.275, 0.185, 0.120, "different\nprobe answers", RED_LIGHT, RED, size=8.8, weight="bold")
    _arrow(ax, (0.365, 0.545), (0.365, 0.395), color=RED, lw=1.05)
    ax.text(0.370, 0.228, r"$G_{M_0}(z)\ne G_{M_1}(z)$", fontsize=9.6, ha="center", color=RED)
    ax.text(
        0.055,
        0.135,
        "A decoder cannot recover a coordinate that the\ninteraction interface never separated.",
        ha="left",
        va="center",
        fontsize=7.6,
        color=MUTED,
        linespacing=1.15,
    )

    ax.plot([0.500, 0.500], [0.125, 0.920], color=GRID, lw=1.0)

    # Panel b: constructive path.
    _box(ax, 0.550, 0.640, 0.160, 0.130, "controlled\ncontrasts", BLUE_LIGHT, BLUE, size=8.8, weight="bold")
    _box(ax, 0.550, 0.435, 0.160, 0.130, "readout and\ndelay checks", ORANGE_LIGHT, ORANGE, size=8.8, weight="bold")
    _box(ax, 0.550, 0.230, 0.160, 0.130, "context\ncoverage", GREEN_LIGHT, GREEN, size=8.8, weight="bold")

    _box(ax, 0.780, 0.500, 0.145, 0.165, "evidence\ngate", PALE, INK, size=9.0, weight="bold")
    for y, col in [(0.705, BLUE), (0.500, ORANGE), (0.295, GREEN)]:
        _arrow(ax, (0.710, y), (0.780, 0.585), color=col, lw=1.05, rad=0.03 if y != 0.500 else 0.0)

    _box(ax, 0.780, 0.235, 0.145, 0.135, r"frozen map" + "\n" + r"$S=\{\theta_j\}$", GREEN_LIGHT, GREEN, size=8.8, weight="bold")
    _arrow(ax, (0.852, 0.500), (0.852, 0.370), color=INK, lw=1.05)
    probe_x = [0.565, 0.655, 0.745, 0.835, 0.925]
    probe_labels = ["action", "context", "target", "value", "delay"]
    probe_colors = [BLUE, GREEN, RED, ORANGE, "#6b5aa6"]
    for x, label, col in zip(probe_x, probe_labels, probe_colors):
        ax.add_patch(Circle((x, 0.095), 0.029, facecolor="white", edgecolor=col, lw=1.25))
        ax.text(x, 0.095, label, ha="center", va="center", fontsize=6.6, color=INK)
        _arrow(ax, (0.852, 0.235), (x, 0.127), color=col, lw=0.8, rad=(0.855 - x) * 0.25)
    ax.text(0.740, 0.045, "held-out mechanism probe", ha="center", va="center", fontsize=7.5, color=MUTED)

    fig.savefig(OUT / "mechanism_memory_nmi_v3.pdf", bbox_inches="tight")
    fig.savefig(OUT / "mechanism_memory_nmi_v3.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def _read_json(name):
    with (ROOT / "results" / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def _row_by(rows, key, value):
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(value)


def collect_gap_data():
    matched = _read_json("icml_matched_mechanism_baselines_8fam_2seed_120_worldmodel_v1.json")
    matched_rows = matched["matched_mechanism_baselines"]["summary"]
    world = _row_by(matched_rows, "method", "latent-world-model")
    core = _row_by(matched_rows, "method", "causal-core-context-search")

    ranking = _read_json("icml_ranking_loss_algorithm_comparison_8fam_2seed_v1.json")
    rank_hidden = _row_by(ranking["summary"], "task", "procedural noisy-hidden ranking")

    continuous = _read_json("icml_continuous_metric_poc_20seed_v1.json")
    cont_linear = _row_by(continuous["summary"], "agent", "global-linear-regression")
    cont_core = _row_by(continuous["summary"], "agent", "metric-causal-core")

    mujoco = _read_json("icml_mujoco_causal_checks_halfcheetah_5seed_v1.json")
    mujoco_linear = _row_by(mujoco["context_shift"]["summary"], "method", "source-linear dynamics")
    mujoco_core = _row_by(mujoco["context_shift"]["summary"], "method", "few-shot metric core")

    llm = _read_json("icml_llm_algorithm_comparison_100x3_v1.json")
    llm_world = llm["worlds"]["panel-complex-noisy-hidden"]
    llm_prompt = llm_world["llm-causal-prompt"]["summary"]
    llm_gate = llm_world["llm-control-planner-gated"]["summary"]

    return [
        {
            "setting": "Symbolic SCM\nworld model",
            "baseline": world["f1"],
            "memory": core["f1"],
            "baseline_name": "latent model",
            "memory_name": "memory",
            "signal": f"next-bit acc. {world['model_next_bit_accuracy']:.3f}",
        },
        {
            "setting": "Hidden-context\nranking",
            "baseline": rank_hidden["ranking_predictor_mechanism_f1"],
            "memory": rank_hidden["causal_core_mechanism_f1"],
            "baseline_name": "ranker",
            "memory_name": "memory",
            "signal": f"rank top-1 {rank_hidden['ranking_predictor_rank_top1']:.3f}",
        },
        {
            "setting": "Continuous\nmetric cells",
            "baseline": cont_linear["f1"],
            "memory": cont_core["f1"],
            "baseline_name": "linear",
            "memory_name": "memory",
            "signal": f"readout FP {cont_linear['readout_false_positive']:.1f} -> 0",
        },
        {
            "setting": "HalfCheetah\npolarity shift",
            "baseline": mujoco_linear["exact_accuracy"],
            "memory": mujoco_core["exact_accuracy"],
            "baseline_name": "source dyn.",
            "memory_name": "few-shot",
            "signal": f"target acc. {mujoco_linear['target_accuracy']:.3f}",
        },
        {
            "setting": "Qwen proposals\nlanguage",
            "baseline": llm_prompt["f1"],
            "memory": llm_gate["f1"],
            "baseline_name": "prompt",
            "memory_name": "gated",
            "signal": f"readout FP {llm_prompt['readout_false_positive']:.0f} -> 0",
        },
    ]


def draw_cross_domain_gap():
    data = collect_gap_data()
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    y = np.arange(len(data))[::-1]
    baseline = np.array([d["baseline"] for d in data])
    memory = np.array([d["memory"] for d in data])

    ax.axvspan(0.0, 0.35, color=CORAL_LIGHT, alpha=0.38, zorder=0)
    ax.axvspan(0.70, 1.0, color=GREEN_LIGHT, alpha=0.46, zorder=0)
    ax.text(0.02, len(data) - 0.10, "mechanism omitted", ha="left", va="bottom", fontsize=7.0, color=CORAL_DARK, weight="bold")
    ax.text(0.985, len(data) - 0.10, "answer retained", ha="right", va="bottom", fontsize=7.0, color=GREEN_DARK, weight="bold")

    for yi, b, m, d in zip(y, baseline, memory, data):
        ax.hlines(yi, 0, 1.0, color=GRID, lw=0.8, zorder=0)
        ax.plot([b, m], [yi, yi], color=BORDER, lw=2.0, solid_capstyle="round", zorder=1)
        gap_mid = (b + m) / 2
        ax.plot([gap_mid - 0.018, gap_mid + 0.018], [yi - 0.10, yi + 0.10], color=CORAL_DARK, lw=1.0, zorder=2)
        ax.plot([gap_mid + 0.028, gap_mid + 0.064], [yi - 0.10, yi + 0.10], color=CORAL_DARK, lw=1.0, zorder=2)

        ax.scatter([b], [yi], s=72, color=GRAY, edgecolor=GRAY_DARK, linewidth=0.85, zorder=3)
        ax.scatter([m], [yi], s=96, color=GREEN, edgecolor=GREEN_DARK, linewidth=0.95, zorder=4)

        b_label_x = max(b - 0.032, 0.018)
        b_ha = "right" if b > 0.08 else "left"
        ax.text(b_label_x, yi - 0.24, f"{b:.2f}", ha=b_ha, va="center", fontsize=6.9, color=GRAY_DARK)
        m_label_x = min(m + 0.026, 1.02)
        m_ha = "left" if m < 0.96 else "right"
        ax.text(m_label_x, yi + 0.20, f"{m:.2f}", ha=m_ha, va="center", fontsize=6.9, color=GREEN_DARK, weight="bold")

        chip = FancyBboxPatch(
            (1.055, yi - 0.155),
            0.165,
            0.31,
            boxstyle="round,pad=0.014,rounding_size=0.035",
            linewidth=0.75,
            edgecolor=CORAL_DARK,
            facecolor="white",
            clip_on=False,
        )
        ax.add_patch(chip)
        ax.text(1.137, yi, d["signal"], va="center", ha="center", fontsize=6.25, color=CORAL_DARK, clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels([d["setting"] for d in data], fontsize=7.8, color=INK)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.65, len(data) - 0.05)
    ax.set_xlabel("frozen mechanism probe score", fontsize=8.0, color=INK)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    _style_axes(ax, left=False, bottom=True)
    ax.tick_params(axis="y", length=0)
    ax.text(1.137, len(data) - 0.10, "task signal", ha="center", va="bottom", fontsize=7.0, color=CORAL_DARK, weight="bold", clip_on=False)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor=GRAY_DARK, markersize=6.5, label="task-oriented baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor=GREEN_DARK, markersize=7.4, label="mechanism memory"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.16),
        frameon=False,
        fontsize=6.9,
        handlelength=1.3,
        ncol=2,
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.215, right=0.82, top=0.91, bottom=0.24)
    fig.savefig(OUT / "cross_domain_mechanism_gap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "cross_domain_mechanism_gap.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def draw_mujoco_checks():
    mujoco = _read_json("icml_mujoco_causal_checks_halfcheetah_5seed_v1.json")

    ctx_methods = [
        ("source-linear dynamics", "linear"),
        ("source-neural dynamics", "neural"),
        ("source-metric core", "metric"),
        ("few-shot metric core", "few-shot"),
    ]
    ctx = {_row["method"]: _row for _row in mujoco["context_shift"]["summary"]}
    x = np.arange(len(ctx_methods))
    w = 0.34

    fig, ax = plt.subplots(figsize=(3.25, 1.70))
    target = [ctx[m]["target_accuracy"] for m, _ in ctx_methods]
    exact = [ctx[m]["exact_accuracy"] for m, _ in ctx_methods]
    ax.bar(x - w / 2, target, w, color=BLUE, edgecolor=BLUE_DARK, linewidth=0.8, label="target")
    ax.bar(x + w / 2, exact, w, color=GREEN, edgecolor=GREEN_DARK, linewidth=0.8, label="signed")
    for xi, val in zip(x + w / 2, exact):
        ax.text(xi, min(val + 0.035, 1.02), f"{val:.2f}", ha="center", va="bottom", fontsize=6.1, color=INK)
    ax.set_ylabel("accuracy", fontsize=6.8, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in ctx_methods], fontsize=6.2, color=INK)
    _style_axes(ax, left=True, bottom=True)
    ax.legend(
        frameon=False,
        fontsize=6.2,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        handlelength=1.0,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.78, bottom=0.22)
    fig.savefig(OUT / "mujoco_context_shift.pdf", bbox_inches="tight")
    fig.savefig(OUT / "mujoco_context_shift.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    ro_methods = [
        ("linear observed selector", "linear"),
        ("neural observed selector", "neural"),
        ("metric core observed selector", "metric"),
        ("metric core with readout filter", "filtered"),
    ]
    ro = {_row["method"]: _row for _row in mujoco["readout_shift"]["summary"]}
    x = np.arange(len(ro_methods))
    direct = [ro[m]["direct_accuracy"] for m, _ in ro_methods]
    readout = [ro[m]["readout_false_positive_rate"] for m, _ in ro_methods]

    fig, ax = plt.subplots(figsize=(3.25, 1.70))
    ax.bar(x - w / 2, direct, w, color=GREEN, edgecolor=GREEN_DARK, linewidth=0.8, label="direct")
    ax.bar(x + w / 2, readout, w, color=CORAL, edgecolor=CORAL_DARK, linewidth=0.8, label="readout")
    for xi, val in zip(x + w / 2, readout):
        ax.text(xi, min(val + 0.035, 1.02), f"{val:.2f}", ha="center", va="bottom", fontsize=6.1, color=INK)
    ax.set_ylabel("rate", fontsize=6.8, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in ro_methods], fontsize=6.2, color=INK)
    _style_axes(ax, left=True, bottom=True)
    ax.legend(
        frameon=False,
        fontsize=6.2,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        handlelength=1.0,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.78, bottom=0.22)
    fig.savefig(OUT / "mujoco_readout_shift.pdf", bbox_inches="tight")
    fig.savefig(OUT / "mujoco_readout_shift.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def draw_llm_suite():
    llm = _read_json("icml_llm_algorithm_comparison_100x3_v1.json")
    world = llm["worlds"]["panel-complex-noisy-hidden"]
    variants = [
        ("llm-vanilla", "LLM-only", GRAY),
        ("llm-causal-prompt", "causal\nprompt", CORAL),
        ("llm-context-search-gated", "LLM +\ncontext gate", GREEN),
        ("llm-control-planner-gated", "LLM +\ncontrol gate", GREEN),
        ("causal-core-context-search-planner", "context\nplanner", BLUE),
        ("causal-core-control-experiment-planner", "control\nplanner", BLUE),
    ]
    summaries = [world[key]["summary"] for key, _, _ in variants]
    labels = [label for _, label, _ in variants]
    colors = [color for _, _, color in variants]
    f1 = np.array([s["f1"] for s in summaries])
    false_edges = np.array([s["false_positive"] for s in summaries])
    readout_fp = np.array([s["readout_false_positive"] for s in summaries])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.35), gridspec_kw={"width_ratios": [1.05, 1.0]})
    y = np.arange(len(labels))[::-1]

    edge_colors = [GRAY_DARK, CORAL_DARK, GREEN_DARK, GREEN_DARK, BLUE_DARK, BLUE_DARK]
    ax1.barh(y, f1, color=colors, alpha=0.94, edgecolor=edge_colors, linewidth=0.8)
    for yi, val in zip(y, f1):
        ax1.text(val + 0.018, yi, f"{val:.2f}", va="center", fontsize=6.6, color=INK)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=6.8, color=INK)
    ax1.set_xlim(0, 0.82)
    ax1.set_xlabel("mechanism F1", fontsize=6.9, color=INK)
    ax1.set_title("Frozen mechanism score", fontsize=7.5, color=INK, pad=4)
    _style_axes(ax1, left=False, bottom=True)
    ax1.tick_params(axis="y", length=0)

    ax2.barh(y + 0.15, false_edges, height=0.28, color=GRAY, edgecolor=GRAY_DARK, linewidth=0.8, label="false edges")
    ax2.barh(y - 0.15, readout_fp, height=0.28, color=CORAL, edgecolor=CORAL_DARK, linewidth=0.8, label="readout false edges")
    for yi, val in zip(y + 0.15, false_edges):
        ax2.text(val + 1.5, yi, f"{val:.0f}", va="center", fontsize=6.3, color=INK)
    for yi, val in zip(y - 0.15, readout_fp):
        if val > 0:
            ax2.text(val + 1.5, yi, f"{val:.0f}", va="center", fontsize=6.3, color=INK)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("count", fontsize=6.9, color=INK)
    ax2.set_title("Corrupted writes", fontsize=7.5, color=INK, pad=4)
    _style_axes(ax2, left=False, bottom=True)
    ax2.tick_params(axis="y", length=0)
    ax2.legend(frameon=False, fontsize=6.1, loc="lower right", handlelength=1.2)

    fig.subplots_adjust(left=0.20, right=0.98, top=0.84, bottom=0.22, wspace=0.28)
    fig.savefig(OUT / "llm_causal_suite.pdf", bbox_inches="tight")
    fig.savefig(OUT / "llm_causal_suite.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_mechanism_memory()
    draw_cross_domain_gap()
    draw_mujoco_checks()
    draw_llm_suite()
