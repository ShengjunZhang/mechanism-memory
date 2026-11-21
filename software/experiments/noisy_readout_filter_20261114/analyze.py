"""Independent prediction and formula scoring for the noisy-readout follow-up."""
from pathlib import Path
import csv
import importlib.util
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from causal_sandbox.source_compat import sources_match
from causal_sandbox.noisy_readout_filter_followup import sha, source_hashes, METRICS, select_settings
from causal_sandbox.observation import _candidate_formulas

SPEC = importlib.util.spec_from_file_location("previous_rescore", ROOT.parent / "context_readout_memory_20261114/analyze.py")
PREVIOUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIOUS)


def formula_ba(pred, gold):
    positive, negative = gold.astype(bool), ~gold.astype(bool)
    return float((np.mean(pred[positive]) + np.mean(~pred[negative])) / 2)


def verify(path, protocol, lock):
    run = json.loads(path.read_text())
    assert sources_match(run["source_hashes"], lock["source_hashes"])
    assert run["protocol_sha256"] == lock["protocol_sha256"]
    assert sha(path.with_suffix(".npz")) == run["cache_sha256"]
    lookup = {(t["condition"], t["budget"]): t for t in run["trials"]}
    checks = {"family": run["family"], "phase": run["phase"], "rows": 0, "formula_checks": 0, "locality_checks": 0}
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as cache:
        for row in run["rows"]:
            trial = lookup[row["condition"], row["budget"]]
            current = trial["rounds"][row["round"]]
            key = f"{row['condition']}__{row['budget']}__round{row['round']}"
            pred = cache[key + "__pred__" + row["method"]]
            expected = PREVIOUS.rescore(trial, current, row["round"], pred)
            np.testing.assert_array_equal(cache[key + "__gold"], PREVIOUS.gold_and_masks(trial, current, row["round"])[0])
            for metric, value in expected.items():
                if value is None:
                    assert row[metric] is None
                else:
                    np.testing.assert_allclose(row[metric], value, atol=1e-12, rtol=1e-12)
            mask = set(current["masks"][row["filter_key"]])
            roots = set(trial["evaluation_only_truth"]["roots"])
            np.testing.assert_allclose(row["physical_suppression"], len(roots & mask) / 3)
            if trial["readout"] != "none":
                assert row["readout_identified"] == int(trial["evaluation_only_truth"]["readout"] in mask)
            assert row["interventions_so_far"] == (12 + row["round"]) * row["budget"]
            assert not any(pred[:, i].any() for i, target in enumerate(trial["targets"]) if target in mask)
            checks["rows"] += 1
    for trial in run["trials"]:
        history, previous_memories = [], {}
        for current in trial["rounds"]:
            history.extend(current["batch"])
            states = [row[side] for row in history for side in ("before", "after")]
            fit = np.repeat(np.arange(len(history)) % 2 == 0, 2)
            for record in current["filter_candidates"]:
                target = record["target"]
                formulas = {expression: (sources, fn) for expression, sources, fn in _candidate_formulas(target, sorted(states[0]))}
                sources, fn = formulas[record["expression"]]
                assert list(sources) == record["sources"]
                pred, gold = np.array([fn(s) for s in states], bool), np.array([s[target] for s in states], bool)
                for split, metric in ((fit, "fit_ba"), (~fit, "validation_ba"), (np.ones(len(fit), bool), "pooled_ba")):
                    np.testing.assert_allclose(record[metric], formula_ba(pred[split], gold[split]), atol=1e-12)
                assert record["fit_pairs"] == int(fit.sum() // 2)
                assert record["validation_pairs"] == int((~fit).sum() // 2)
                checks["formula_checks"] += 1
            for name, mask in current["masks"].items():
                if name.startswith("validated"):
                    cfg = protocol["filter_grid"][int(name.removeprefix("validated"))]
                    expected = {r["target"] for r in current["filter_candidates"] if
                                min(r["fit_ba"], r["validation_ba"]) >= cfg["validation_floor"] and r["pooled_ba"] >= cfg["pooled_floor"]}
                    assert set(mask) == expected
            visited = {(r["action"], tuple(r["context"])) for r in current["batch"]}
            for variant, memory in current["memories"].items():
                entries = {(e[0], tuple(e[1]), e[2]): e for e in memory}
                for key, old in previous_memories.get(variant, {}).items():
                    if key[:2] not in visited:
                        assert entries[key] == old
                        checks["locality_checks"] += 1
                previous_memories[variant] = entries
    return run, checks


def paired(summary, condition, budget, method, baseline):
    rows = {r["method"]: r for r in summary if r["condition"] == condition and r["budget"] == budget}
    left, right = [{f["family"]: f for f in rows[m]["families"]} for m in (method, baseline)]
    result = {"condition": condition, "budget": budget, "method": method, "baseline": baseline}
    for metric in METRICS:
        d = np.array([left[f][metric] - right[f][metric] for f in left if left[f][metric] is not None and right[f][metric] is not None])
        if not len(d):
            continue
        rng = np.random.default_rng(117419)
        draws = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
        result[metric + "_difference"] = float(d.mean())
        result[metric + "_ci95"] = np.quantile(draws, [.025, .975]).tolist()
        result[metric + "_positive_tie_negative"] = [int(sum(d > 1e-10)), int(sum(abs(d) <= 1e-10)), int(sum(d < -1e-10))]
    return result


def verify_summary(summary, rows, protocol):
    for result in summary:
        selected = [r for r in rows if all(r[k] == result[k] for k in ("condition", "budget", "method"))]
        assert len(selected) == len(protocol["confirmation_families"]) * protocol["rounds"]
        assert {f["family"] for f in result["families"]} == set(protocol["confirmation_families"])
        for metric in METRICS:
            means = []
            for family in result["families"]:
                group = [r for r in selected if r["family"] == family["family"]]
                assert sorted(r["round"] for r in group) == list(range(protocol["rounds"]))
                values = [r[metric] for r in group if r[metric] is not None]
                if values:
                    value = sum(values) / len(values)
                    np.testing.assert_allclose(family[metric], value, atol=1e-12, rtol=1e-12)
                    means.append(value)
                else:
                    assert family[metric] is None
            if means:
                np.testing.assert_allclose(result[metric], sum(means) / len(means), atol=1e-12, rtol=1e-12)
            else:
                assert result[metric] is None


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": .8, "xtick.major.width": .8,
                         "ytick.major.width": .8, "axes.spines.top": False, "axes.spines.right": False})
    methods = ["components original", "stratified tree original", "components validated", "stratified tree validated", "stratified tree relaxed"]
    labels = ["Memory: original filter", "Tree: original filter", "Memory: validated filter", "Tree: same validated filter", "Tree: tuned threshold"]
    colors = ["#A6ACB0", "#B16C66", "#21877F", "#4A7BB0", "#9F8B40"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8))
    for method, label, color in zip(methods, labels, colors):
        selected = [next(r for r in summary if r["condition"] == "noise10" and r["budget"] == b and r["method"] == method) for b in (4, 8)]
        axes[0].plot([4, 8], [r["readout_fpr"] * 100 for r in selected], "o-", color=color, markersize=4, label=label)
        axes[1].plot([4, 8], [r["direct_ba"] * 100 for r in selected], "o-", color=color, markersize=4)
        scope = [next(r for r in summary if r["condition"] == c and r["budget"] == 4 and r["method"] == method)
                 for c in ("or", "noise05", "noise10", "noise20")]
        axes[2].plot([0, .05, .1, .2], [r["physical_suppression"] * 100 for r in scope], "o-", color=color, markersize=4)
    axes[0].set(title="a  Readout false positives", xlabel="Records per action-context cell", ylabel="False-positive rate (%)", xticks=[4, 8])
    axes[1].set(title="b  Direct-effect prediction", xlabel="Records per action-context cell", ylabel="Balanced accuracy (%)", xticks=[4, 8])
    axes[2].set(title="c  Physical-variable protection", xlabel="Readout noise probability", ylabel="Physical variables suppressed (%)", xticks=[0, .05, .1, .2])
    fig.suptitle("Fresh confirmation: 24 finite environments; shared data and filtering", fontsize=12)
    fig.subplots_adjust(left=.065, right=.98, top=.84, bottom=.31, wspace=.34)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.savefig(ROOT / "confirmation_comparison.png", dpi=220)
    fig.savefig(ROOT / "confirmation_comparison.svg")
    plt.close(fig)


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    lock = json.loads((ROOT / "confirmation_lock.json").read_text())
    assert sources_match(source_hashes(), lock["source_hashes"])
    assert sha(ROOT / "protocol.json") == lock["protocol_sha256"]
    assert sha(ROOT / "results/development/summary.json") == lock["development_summary_sha256"]
    assert sha(ROOT / "development_selection.json") == lock["development_selection_sha256"]
    assert sha(ROOT / "run_confirmation.py") == lock["launcher_sha256"]
    development = json.loads((ROOT / "results/development/summary.json").read_text())
    selection = select_settings(development, protocol)
    assert selection["filter_index"] == lock["filter_index"]
    assert selection["relaxed_index"] == lock["relaxed_index"]
    all_rows, checks = [], []
    for phase in ("development", "confirmation"):
        for seed in protocol[phase + "_families"]:
            run, check = verify(ROOT / "results" / phase / f"family_{seed}.json", protocol, lock)
            checks.append(check)
            if phase == "confirmation":
                all_rows.extend(run["rows"])
    summary = json.loads((ROOT / "results/confirmation/summary.json").read_text())
    verify_summary(summary, all_rows, protocol)
    comparisons = [paired(summary, c["name"], b, "components validated", m) for c in protocol["conditions"]
                   for b in protocol["budgets_per_action_context"] for m in
                   ("stratified tree validated", "stratified tree original", "stratified tree relaxed", "components original")]
    primary = [r for r in comparisons if r["condition"] == "noise10"]
    high = next(r for r in primary if r["budget"] == 8 and r["baseline"] == "stratified tree validated")
    low = next(r for r in primary if r["budget"] == 4 and r["baseline"] == "stratified tree validated")
    retention = next(r for r in primary if r["budget"] == 4 and r["baseline"] == "components original")
    report = {"verification_scope": "Independent rescoring of predictions and accepted formula scores; hash, pair separation, budget and memory-locality checks. Not independent full retraining.",
              "checks": checks, "rows_checked": sum(c["rows"] for c in checks), "comparisons": comparisons,
              "strict_readout_win": high["readout_fpr_difference"] < -1e-10,
              "low_budget_ba_win": low["direct_ba_difference"] > 1e-10,
              "physical_retention_guard": retention["physical_ba_difference"] >= -.01,
              "summary": summary}
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "confirmation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    plot(summary)
    print(json.dumps({k: v for k, v in report.items() if k not in ("checks", "comparisons", "summary")}, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != "families"} for r in summary if r["condition"] == "noise10"], indent=2))


if __name__ == "__main__":
    main()
