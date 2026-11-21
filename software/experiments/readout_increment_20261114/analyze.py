"""Independent rescoring and checks of the incremental readout rule."""
from pathlib import Path
import csv
import importlib.util
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from causal_sandbox.source_compat import sources_match
from causal_sandbox.readout_increment_followup import settings, source_hashes, sha, select_settings, guard_details

SPEC = importlib.util.spec_from_file_location("threshold_analysis", ROOT.parent / "noisy_readout_threshold_20261114/analyze.py")
THRESHOLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(THRESHOLD)
PRIOR = THRESHOLD.PRIOR


def verify_increments(run, p, lock):
    checks, copy_checks = 0, 0
    for trial in run["trials"]:
        states = []
        for current in trial["rounds"]:
            states.extend(row[side] for row in current["batch"] for side in ("before", "after"))
            rules = {r["target"]: r for r in current["readout_rules"]["calibrated"]}
            assert set(rules) == {c["target"] for c in current["increment_candidates"]}
            for c in current["increment_candidates"]:
                y = [state[c["target"]] for state in states]
                simple_scores = [sum(y) / len(y), 1 - sum(y) / len(y)]
                for other in states[0]:
                    if other != c["target"]:
                        equal = sum(state[other] == state[c["target"]] for state in states) / len(states)
                        simple_scores.extend([equal, 1 - equal])
                best = max(simple_scores)
                np.testing.assert_allclose(c["simple_accuracy"], best, atol=1e-12)
                assert c["expression"] == rules[c["target"]]["expression"]
                np.testing.assert_allclose(c["accuracy"], rules[c["target"]]["accuracy"], atol=1e-12)
                np.testing.assert_allclose(c["increment"], c["accuracy"] - best, atol=1e-12)
                assert c["state_count"] == len(states)
                checks += 1
            for tag, mask in current["masks"].items():
                if tag.startswith("gain"):
                    margin = p["increment_grid"][int(tag[4:])]
                    expected = {c["target"] for c in current["increment_candidates"] if c["increment"] >= margin}
                    assert set(mask) == expected
                    # Exact equality or inversion makes the simple predictor perfect.
                    for target in states[0]:
                        for other in states[0]:
                            if target != other and (all(s[target] == s[other] for s in states) or all(s[target] != s[other] for s in states)):
                                assert target not in mask
                                copy_checks += 1
    if run["phase"] == "confirmation":
        assert {r["method"] for r in run["rows"]} == set(p["methods"])
        for row in run["rows"]:
            if row["method"] in ("components guarded", "stratified tree guarded", "stratified next-state guarded"):
                assert row["filter_key"] == f"gain{lock['increment_index']}"
            if row["method"] == "stratified tree tuned":
                assert row["filter_key"] == lock["tuned_tree_filter"]
    return checks, copy_checks


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": .8,
                         "xtick.major.width": .8, "ytick.major.width": .8,
                         "axes.spines.top": False, "axes.spines.right": False})
    table = {(r["condition"], r["budget"], r["method"]): r for r in summary}
    methods = [("components original", "Memory: original filter", "#A6ACB0", "--"),
               ("components calibrated", "Memory: threshold only", "#B39552", "--"),
               ("components guarded", "Memory: increment gate", "#21877F", "-"),
               ("stratified tree guarded", "Tree: same increment gate", "#B44B48", "-")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9))
    for i, (method, label, color, style) in enumerate(methods):
        if i < 3:
            selected = [table[c, 4, method] for c in ("or", "identity")]
            axes[0].bar(np.arange(2) + (i - 1) * .23, [100*r["physical_suppression"] for r in selected],
                        width=.22, color=color)
        for ax, metric, selected, x in (
            (axes[1], "direct_ba", [table["noise10", b, method] for b in (4, 8)], [4, 8]),
            (axes[2], "readout_fpr", [table[c, 8, method] for c in ("or", "noise05", "noise10", "noise20")], [0, .05, .1, .2]),
        ):
            means = np.array([r[metric] for r in selected]) * 100
            ci = np.array([THRESHOLD.interval([f[metric] for f in r["families"]]) for r in selected]) * 100
            ax.errorbar(x, means, yerr=np.maximum(0, np.stack([means - ci[:, 0], ci[:, 1] - means])),
                        fmt="o", linestyle=style, color=color, markersize=4, capsize=2.5, linewidth=1.3, label=label)
    axes[0].set(title="a  Physical suppression, budget 4", ylabel="Physical variables suppressed (%)", xticks=[0, 1], xticklabels=["Clean OR", "Exact copy"])
    axes[1].set(title="b  Direct-effect BA, 10% noise", xlabel="Records per action-context cell", ylabel="Balanced accuracy (%)", xticks=[4, 8], xlim=(3.6, 8.4))
    axes[2].set(title="c  Readout FPR, budget 8", xlabel="Readout noise probability", ylabel="False-positive rate (%)", xticks=[0, .05, .1, .2])
    axes[0].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    fig.suptitle("48 new finite environments; shared evidence and interaction budgets", fontsize=12)
    fig.subplots_adjust(left=.065, right=.98, top=.84, bottom=.31, wspace=.34)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.savefig(ROOT / "confirmation_comparison.png", dpi=220)
    fig.savefig(ROOT / "confirmation_comparison.svg")
    plt.close(fig)


def main():
    p = settings()
    lock = json.loads((ROOT / "confirmation_lock.json").read_text())
    assert sources_match(source_hashes(), lock["source_hashes"])
    assert sha(ROOT / "protocol.json") == lock["protocol_sha256"]
    assert sha(Path(p["parent_protocol"])) == lock["parent_protocol_sha256"]
    assert sha(ROOT / "results/development/summary.json") == lock["development_summary_sha256"]
    assert sha(ROOT / "development_selection.json") == lock["development_selection_sha256"]
    development = json.loads((ROOT / "results/development/summary.json").read_text())
    selected = select_settings(development, p)
    assert selected["increment_index"] == lock["increment_index"]
    assert selected["tuned_tree_filter"] == lock["tuned_tree_filter"]
    phase_rows, checks = {}, []
    for phase in ("development", "confirmation"):
        phase_rows[phase] = []
        for seed in p[phase + "_families"]:
            run, check = PRIOR.verify(ROOT / "results" / phase / f"family_{seed}.json", p, lock)
            check["readout_rule_checks"] = THRESHOLD.verify_rules(run, p)
            check["increment_checks"], check["copy_checks"] = verify_increments(run, p, lock)
            checks.append(check)
            phase_rows[phase].extend(run["rows"])
        s = json.loads((ROOT / "results" / phase / "summary.json").read_text())
        PRIOR.verify_summary(s, phase_rows[phase], p | {"confirmation_families": p[phase + "_families"]})
    summary = json.loads((ROOT / "results/confirmation/summary.json").read_text())
    comparisons = [PRIOR.paired(summary, c["name"], b, "components guarded", baseline)
                   for c in p["conditions"] for b in p["budgets_per_action_context"]
                   for baseline in ("components original", "components calibrated", "stratified tree guarded", "stratified tree tuned", "stratified next-state guarded")]
    table = {(r["condition"], r["budget"], r["method"]): r for r in summary}
    guards = guard_details(table, p, "components guarded")
    main_comparisons = [r for r in comparisons if r["condition"] == "noise10" and r["budget"] == 4 and r["baseline"] in ("stratified tree guarded", "stratified tree tuned")]
    report = {"verification_scope": "Independent prediction rescoring, formula and simple-predictor scores, increment decisions, exact-copy exclusions, budget, summary and memory-locality checks; not independent full retraining.",
              "checks": checks, "rows_checked": sum(len(rows) for rows in phase_rows.values()), "confirmation_rows": len(phase_rows["confirmation"]),
              "comparisons": comparisons, "retention_guards": guards, "retention_guards_pass": all(g["pass"] for g in guards),
              "low_budget_ba_win_vs_both": all(r["direct_ba_difference"] > 0 for r in main_comparisons),
              "low_budget_ba_ci_above_zero_vs_both": all(r["direct_ba_ci95"][0] > 0 for r in main_comparisons),
              "summary": summary}
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "confirmation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phase_rows["confirmation"][0]))
        writer.writeheader()
        writer.writerows(phase_rows["confirmation"])
    plot(summary)
    print(json.dumps({k: v for k, v in report.items() if k not in ("checks", "comparisons", "summary")}, indent=2))
    print(json.dumps(main_comparisons, indent=2))


if __name__ == "__main__":
    main()
