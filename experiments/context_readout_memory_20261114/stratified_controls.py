"""Additional key-conditioned baselines, fixed before confirmation."""
from collections import defaultdict
from pathlib import Path
import json
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from causal_sandbox.context_readout_memory_followup import Family, feature_matrix, observed_targets, recent_cells, measure


def evaluate(run, protocol):
    rows, predictions = [], {}
    for trial in run["trials"]:
        family = Family(run["family"], trial["readout"], protocol)
        history = []
        prefix = f"{trial['readout']}__{trial['budget']}"
        for round_idx, current in enumerate(trial["rounds"]):
            history.extend(current["batch"])
            grouped = defaultdict(list)
            for row in recent_cells(history):
                grouped[row["action"], tuple(row["context"])].append(row)
            queries = current["queries"]
            gold = []
            for query in queries:
                y = np.zeros(len(family.targets), bool)
                a = family.actions.index(query["action"])
                if tuple(query["context"]) == family.active_contexts[a]:
                    target = family.roots[family.root_indices[a]]
                    y[family.targets.index(target)] = query["before"][target] != family.value(a, round_idx)
                gold.append(y)
            gold = np.asarray(gold)
            for next_state in (False, True):
                method = "stratified next-state + shared filter" if next_state else "stratified change + shared filter"
                models = {}
                for key, records in grouped.items():
                    x = feature_matrix(records, trial["actions"], trial["variables"])
                    y = np.asarray([[r["after"][v] for v in trial["targets"]] for r in records], int) if next_state else observed_targets(records, trial["targets"])
                    if not next_state:
                        y[:, [j for j, v in enumerate(trial["targets"]) if v in current["suppressed"]]] = 0
                    models[key] = DecisionTreeClassifier(random_state=run["family"]).fit(x, y)
                pred = np.zeros_like(gold)
                query_groups = defaultdict(list)
                for i, q in enumerate(queries):
                    query_groups[q["action"], tuple(q["context"])].append(i)
                for key, indices in query_groups.items():
                    qs = [queries[i] for i in indices]
                    values = models[key].predict(feature_matrix(qs, trial["actions"], trial["variables"])).astype(bool)
                    if next_state:
                        values = values != np.asarray([[q["before"][v] for v in trial["targets"]] for q in qs])
                    values[:, [j for j, v in enumerate(trial["targets"]) if v in current["suppressed"]]] = False
                    pred[indices] = values
                predictions[f"{prefix}__round{round_idx}__pred__{method}"] = pred
                rows.append({"family": run["family"], "readout": trial["readout"], "budget": trial["budget"], "round": round_idx,
                             "method": method, "interventions_so_far": len(history), **measure(gold, pred, queries, family, round_idx)})
    return rows, predictions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    from causal_sandbox.context_readout_memory_followup import summary, sha
    if args.phase == "confirmation":
        lock = json.loads((ROOT / "confirmation_lock.json").read_text())
        assert lock["stratified_controls_sha256"] == sha(Path(__file__))
        assert lock["baseline_amendment_sha256"] == sha(ROOT / "baseline_amendment.json")
    results = []
    folder = ROOT / "results" / (args.phase + "_stratified")
    folder.mkdir(parents=True, exist_ok=True)
    for seed in protocol[args.phase + "_families"]:
        path = ROOT / "results" / args.phase / f"family_{seed}.json"
        run = json.loads(path.read_text())
        rows, predictions = evaluate(run, protocol)
        output = folder / path.name
        np.savez_compressed(output.with_suffix(".npz"), **predictions)
        result = {"family": seed, "source_sha256": sha(path), "implementation_sha256": sha(Path(__file__)),
                  "cache_sha256": sha(output.with_suffix(".npz")), "rows": rows}
        output.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
    value = summary(results, protocol)
    (folder / "summary.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "families"} for r in value], indent=2))


if __name__ == "__main__":
    main()
