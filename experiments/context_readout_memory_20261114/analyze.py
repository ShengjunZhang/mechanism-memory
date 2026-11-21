"""Independent scoring and family-level intervals for the frozen component study."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from causal_sandbox.source_compat import sources_match
from causal_sandbox.context_readout_memory_followup import sha, source_hashes

METRICS = ("direct_ba", "physical_ba", "readout_fpr", "unchanged_ba", "changed_ba", "exact_set")


def ba(y, p):
    positive, negative = y.astype(bool), ~y.astype(bool)
    if not positive.any() or not negative.any():
        return float(np.mean(y == p))
    return float((np.sum(p & positive) / positive.sum() + np.sum(~p & negative) / negative.sum()) / 2)


def gold_and_masks(trial, current, round_idx):
    targets = trial["targets"]
    truth = trial["evaluation_only_truth"]
    gold = np.zeros((len(current["queries"]), len(targets)), bool)
    changed = np.zeros(len(gold), bool)
    for i, query in enumerate(current["queries"]):
        a = trial["actions"].index(query["action"])
        enabled = query["context"] == truth["active_contexts"][a]
        if enabled:
            target = truth["roots"][truth["root_indices"][a]]
            value = bool(truth["initial_values"][a])
            if a == 0 and round_idx in (1, 2) or a == 1 and round_idx >= 2:
                value = not value
            gold[i, targets.index(target)] = query["before"][target] != value
        current_action = 1 if round_idx == 2 else 0
        changed[i] = round_idx > 0 and a == current_action and enabled
        assert "after" not in query
    return gold, changed


def rescore(trial, current, round_idx, prediction):
    gold, changed = gold_and_masks(trial, current, round_idx)
    physical = [trial["targets"].index(v) for v in trial["evaluation_only_truth"]["roots"]]
    readout = trial["evaluation_only_truth"]["readout"]
    return {"direct_ba": ba(gold, prediction), "physical_ba": ba(gold[:, physical], prediction[:, physical]),
            "readout_fpr": float(prediction[:, trial["targets"].index(readout)].mean()) if readout in trial["targets"] else None,
            "unchanged_ba": ba(gold[~changed], prediction[~changed]),
            "changed_ba": ba(gold[changed], prediction[changed]) if changed.any() else None,
            "exact_set": float(np.mean(np.all(gold == prediction, axis=1)))}


def verify(path, p, lock):
    run = json.loads(path.read_text())
    assert sources_match(run["source_hashes"], lock["source_hashes"])
    assert run["protocol_sha256"] == lock["protocol_sha256"]
    assert sha(path.with_suffix(".npz")) == run["cache_sha256"]
    extra_path = path.parent.parent / (run["phase"] + "_stratified") / path.name
    extra = json.loads(extra_path.read_text())
    assert extra["source_sha256"] == sha(path)
    assert extra["implementation_sha256"] == lock["stratified_controls_sha256"]
    assert sha(extra_path.with_suffix(".npz")) == extra["cache_sha256"]
    trials = {(t["readout"], t["budget"]): t for t in run["trials"]}
    n = 0
    for result, cache_path in ((run, path.with_suffix(".npz")), (extra, extra_path.with_suffix(".npz"))):
        with np.load(cache_path, allow_pickle=False) as cache:
            for row in result["rows"]:
                trial = trials[row["readout"], row["budget"]]
                current = trial["rounds"][row["round"]]
                key = f"{row['readout']}__{row['budget']}__round{row['round']}"
                pred = cache[key + "__pred__" + row["method"]]
                expected = rescore(trial, current, row["round"], pred)
                if key + "__gold" in cache:
                    np.testing.assert_array_equal(cache[key + "__gold"], gold_and_masks(trial, current, row["round"])[0])
                for metric in METRICS:
                    if expected[metric] is None:
                        assert row[metric] is None
                    else:
                        np.testing.assert_allclose(row[metric], expected[metric], rtol=1e-12, atol=1e-12)
                assert row["interventions_so_far"] == (12 + row["round"]) * row["budget"]
                n += 1
    # Memory locality is checked on the recorded tuples, without using probe outcomes.
    locality = 0
    for trial in run["trials"]:
        previous = {}
        for current in trial["rounds"]:
            visited = {(r["action"], tuple(r["context"])) for r in current["batch"]}
            entries = {(e[0], tuple(e[1]), e[2]): e for e in current["memory"]}
            for key, value in previous.items():
                if key[:2] not in visited:
                    assert entries[key] == value
                    locality += 1
            previous = entries
    return run["rows"] + extra["rows"], {"family": run["family"], "phase": run["phase"], "rows_rescored": n, "unvisited_entry_checks": locality}


def average(rows, metric):
    values = [r[metric] for r in rows if r[metric] is not None]
    return float(np.mean(values)) if values else None


def aggregate(rows, p):
    result = []
    for readout in p["readouts"]:
        for budget in p["budgets_per_action_context"]:
            for method in sorted({r["method"] for r in rows}):
                selected = [r for r in rows if (r["readout"], r["budget"], r["method"]) == (readout, budget, method)]
                family = [{"family": f, **{m: average([r for r in selected if r["family"] == f], m) for m in METRICS}}
                          for f in sorted({r["family"] for r in selected})]
                result.append({"readout": readout, "budget": budget, "method": method, "families": family,
                               **{m: average(family, m) for m in METRICS}})
    return result


def figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["full-history tree", "context replay + shared filter", "stratified change + shared filter", "tuple components"]
    labels = ["Full-history tree", "Context replay + filter", "Stratified tree + filter", "Mechanism-memory components"]
    colors = ["#969CA2", "#C56C65", "#4A7BB0", "#21877F"]
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": .8, "xtick.major.width": .8,
                         "ytick.major.width": .8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    for i, (method, color, label) in enumerate(zip(methods, colors, labels)):
        r = next(r for r in summary if r["readout"] == "or" and r["budget"] == 8 and r["method"] == method)
        values = np.array([f["direct_ba"] for f in r["families"]])
        rng = np.random.default_rng(91)
        lo, hi = np.quantile(values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1), [.025, .975])
        axes[0].errorbar(values.mean() * 100, i, xerr=[[100 * (values.mean() - lo)], [100 * (hi - values.mean())]], fmt="o", color=color, capsize=3)
        x = np.arange(4)
        ys = [next(r["direct_ba"] * 100 for r in summary if r["readout"] == readout and r["budget"] == 8 and r["method"] == method)
              for readout in ("none", "or", "noisy-or", "identity")]
        axes[1].plot(x, ys, "o-", color=color, linewidth=1.2, markersize=4, label=label)
    axes[0].set(yticks=range(4), yticklabels=labels, xlabel="Direct-effect balanced accuracy (%)", xlim=(70, 101), ylim=(3.6, -.6))
    axes[0].set_title("a  OR readout: primary comparison", loc="left", fontsize=11)
    axes[1].set(xticks=range(4), xticklabels=["None", "OR", "Noisy OR", "Copy"], xlabel="Readout condition", ylabel="Direct-effect balanced accuracy (%)", ylim=(45, 102))
    axes[1].set_title("b  All mandatory scope controls", loc="left", fontsize=11)
    fig.suptitle("Sixteen held-out finite families; observed contexts and equal intervention budgets", fontsize=11)
    fig.subplots_adjust(left=.245, right=.985, top=.82, bottom=.25, wspace=.4)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.savefig(ROOT / "confirmation_comparison.png", dpi=220)
    fig.savefig(ROOT / "confirmation_comparison.svg")
    plt.close(fig)


def main():
    p = json.loads((ROOT / "protocol.json").read_text())
    lock = json.loads((ROOT / "confirmation_lock.json").read_text())
    assert sources_match(source_hashes(), lock["source_hashes"])
    assert sha(ROOT / "protocol.json") == lock["protocol_sha256"]
    assert sha(ROOT / "stratified_controls.py") == lock["stratified_controls_sha256"]
    assert sha(ROOT / "baseline_amendment.json") == lock["baseline_amendment_sha256"]
    assert sha(ROOT / "results/development/summary.json") == lock["development_summary_sha256"]
    assert sha(ROOT / "results/development_stratified/summary.json") == lock["stratified_development_summary_sha256"]
    all_rows, checks = [], []
    for phase in ("development", "confirmation"):
        for seed in p[phase + "_families"]:
            rows, check = verify(ROOT / "results" / phase / f"family_{seed}.json", p, lock)
            checks.append(check)
            if phase == "confirmation":
                all_rows.extend(rows)
    summary = aggregate(all_rows, p)
    primary = {r["method"]: r for r in summary if r["readout"] == p["primary_readout"] and r["budget"] == p["primary_budget"]}
    comparisons = []
    ours = {r["family"]: r["direct_ba"] for r in primary["tuple components"]["families"]}
    for method, row in primary.items():
        if method == "tuple components":
            continue
        differences = np.array([ours[r["family"]] - r["direct_ba"] for r in row["families"]])
        rng = np.random.default_rng(91641)
        draws = differences[rng.integers(0, len(differences), (10000, len(differences)))].mean(axis=1)
        nonzero = differences[abs(differences) > 1e-10]
        comparisons.append({"baseline": method, "direct_ba_difference": float(differences.mean()),
                            "ci95": np.quantile(draws, [.025, .975]).tolist(),
                            "two_sided_sign_test_p": float(binomtest(int(sum(nonzero > 0)), len(nonzero)).pvalue) if len(nonzero) else 1.,
                            "wins_ties_losses": [int(sum(differences > 1e-10)), int(sum(abs(differences) <= 1e-10)), int(sum(differences < -1e-10))]})
    secondary = []
    for readout in p["readouts"]:
        for budget in p["budgets_per_action_context"]:
            selected = {r["method"]: r for r in summary if r["readout"] == readout and r["budget"] == budget}
            left = {r["family"]: r for r in selected["tuple components"]["families"]}
            right = {r["family"]: r for r in selected["stratified change + shared filter"]["families"]}
            item = {"readout": readout, "budget": budget, "comparator": "stratified change + shared filter"}
            for metric in METRICS:
                differences = np.array([left[f][metric] - right[f][metric] for f in left if left[f][metric] is not None and right[f][metric] is not None])
                if not len(differences):
                    continue
                rng = np.random.default_rng(99161)
                draws = differences[rng.integers(0, len(differences), (10000, len(differences)))].mean(axis=1)
                item[metric + "_difference"] = float(differences.mean())
                item[metric + "_ci95"] = np.quantile(draws, [.025, .975]).tolist()
            secondary.append(item)
    report = {"verification": "Independent rescoring from cached predictions and regenerated truth; locality and hash checks. Not a full independent retraining.",
              "checks": checks, "rows_rescored": sum(r["rows_rescored"] for r in checks),
              "primary_comparisons": comparisons, "secondary_comparisons": secondary, "summary": summary}
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "confirmation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    figure(summary)
    print(json.dumps({"rows_rescored": report["rows_rescored"], "primary": [{k: v for k, v in r.items() if k != "families"} for r in primary.values()], "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
