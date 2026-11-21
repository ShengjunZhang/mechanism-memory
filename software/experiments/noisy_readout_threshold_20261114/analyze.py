"""Rescore all fixed-threshold predictions and check fitted readout rules."""
from pathlib import Path
import csv
import importlib.util
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from causal_sandbox.source_compat import sources_match
from causal_sandbox.noisy_readout_threshold_followup import settings, source_hashes, sha
from causal_sandbox.observation import _candidate_formulas

SPEC = importlib.util.spec_from_file_location("filter_analysis", ROOT.parent / "noisy_readout_filter_20261114/analyze.py")
PRIOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRIOR)


def verify_rules(run, p):
    count = 0
    for trial in run["trials"]:
        states = []
        for current in trial["rounds"]:
            states.extend(row[side] for row in current["batch"] for side in ("before", "after"))
            for tag, rules in current["readout_rules"].items():
                assert {r["target"] for r in rules} == set(current["masks"][tag])
                for rule in rules:
                    library = {expr: fn for expr, _, fn in _candidate_formulas(rule["target"], [rule["target"]] + rule["sources"])}
                    fn = library[rule["expression"]]
                    matches = [fn(state) == state[rule["target"]] for state in states]
                    agreement = sum(matches) / len(matches)
                    np.testing.assert_allclose(agreement, rule["accuracy"], atol=1e-12)
                    assert agreement >= (.98 if tag == "original" else p["min_accuracy"])
                    count += 1
    return count


def interval(values):
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(117419)
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975])


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": .8,
                         "xtick.major.width": .8, "ytick.major.width": .8,
                         "axes.spines.top": False, "axes.spines.right": False})
    table = {(r["condition"], r["budget"], r["method"]): r for r in summary}
    methods = [("components original", "Memory: original threshold", "#8EB7B0", "--"),
               ("stratified tree original", "Tree: original threshold", "#D5A19E", "--"),
               ("components calibrated", "Memory: calibrated threshold", "#21877F", "-"),
               ("stratified tree calibrated", "Tree: same calibrated threshold", "#B44B48", "-")]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8))
    for method, label, color, style in methods:
        for ax, metric in zip(axes[:2], ("readout_fpr", "direct_ba")):
            selected = [table["noise10", b, method] for b in (4, 8)]
            means = np.array([r[metric] for r in selected]) * 100
            ci = np.array([interval([f[metric] for f in r["families"]]) for r in selected]) * 100
            ax.errorbar([4, 8], means, yerr=np.maximum(0, np.stack([means - ci[:, 0], ci[:, 1] - means])),
                        fmt="o", linestyle=style, color=color, markersize=4, capsize=2.5, linewidth=1.3, label=label)
        if method.startswith("components"):
            selected = [table[c, 4, method] for c in ("or", "noise05", "noise10", "noise20")]
            axes[2].plot([0, .05, .1, .2], [r["physical_suppression"] * 100 for r in selected],
                         marker="o", linestyle=style, color=color, markersize=4, linewidth=1.3,
                         label="Original filter" if method.endswith("original") else "Calibrated filter")
    axes[0].set(title="a  Readout FPR, 10% noise", xlabel="Records per action-context cell", ylabel="False-positive rate (%)", xticks=[4, 8])
    axes[1].set(title="b  Direct-effect BA, 10% noise", xlabel="Records per action-context cell", ylabel="Balanced accuracy (%)", xticks=[4, 8])
    axes[2].set(title="c  Physical suppression, budget 4", xlabel="Readout noise probability", ylabel="Physical variables suppressed (%)", xticks=[0, .05, .1, .2])
    axes[0].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes[:2]:
        ax.set_xlim(3.6, 8.4)
    fig.suptitle("48 new finite environments; equal interaction budgets and shared filtering", fontsize=12)
    fig.subplots_adjust(left=.065, right=.98, top=.84, bottom=.31, wspace=.34)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.savefig(ROOT / "confirmation_comparison.png", dpi=220)
    fig.savefig(ROOT / "confirmation_comparison.svg")
    plt.close(fig)


def main():
    p = settings()
    lock = json.loads((ROOT / "confirmation_lock.json").read_text())
    assert sources_match(source_hashes(), lock["source_hashes"])
    assert sha(ROOT / "protocol.json") == lock["protocol_sha256"]
    assert sha(Path(p["parent_protocol"])) == lock["parent_protocol_sha256"]
    assert sha(Path(p["parent_selection"])) == lock["parent_selection_sha256"]
    rows, checks = [], []
    for seed in p["confirmation_families"]:
        run, check = PRIOR.verify(ROOT / "results/confirmation" / f"family_{seed}.json", p, lock)
        check["readout_rule_checks"] = verify_rules(run, p)
        checks.append(check)
        rows.extend(run["rows"])
    summary = json.loads((ROOT / "results/confirmation/summary.json").read_text())
    PRIOR.verify_summary(summary, rows, p)
    comparisons = [PRIOR.paired(summary, c["name"], b, "components calibrated", baseline)
                   for c in p["conditions"] for b in p["budgets_per_action_context"]
                   for baseline in ("components original", "stratified tree calibrated", "stratified tree original", "stratified next-state calibrated")]
    high = next(r for r in comparisons if r["condition"] == "noise10" and r["budget"] == 8 and r["baseline"] == "stratified tree calibrated")
    low = next(r for r in comparisons if r["condition"] == "noise10" and r["budget"] == 4 and r["baseline"] == "stratified tree calibrated")
    guard = next(r for r in comparisons if r["condition"] == "noise10" and r["budget"] == 4 and r["baseline"] == "components original")
    report = {"verification_scope": "Independent prediction rescoring, fitted-rule agreement, summary, budget, hashes and memory-locality checks; not independent full retraining.",
              "checks": checks, "rows_checked": len(rows), "comparisons": comparisons,
              "strict_readout_win": high["readout_fpr_difference"] < -1e-10,
              "readout_difference_ci_below_zero": high["readout_fpr_ci95"][1] < 0,
              "low_budget_ba_win": low["direct_ba_difference"] > 1e-10,
              "low_budget_ba_ci_above_zero": low["direct_ba_ci95"][0] > 0,
              "physical_retention_guard": guard["physical_ba_difference"] >= -.01,
              "summary": summary}
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "confirmation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(summary)
    print(json.dumps({k: v for k, v in report.items() if k not in ("checks", "comparisons", "summary")}, indent=2))
    print(json.dumps([high, low, guard], indent=2))


if __name__ == "__main__":
    main()
