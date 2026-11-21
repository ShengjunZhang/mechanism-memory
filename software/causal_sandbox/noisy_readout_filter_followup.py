"""Validation-based noise tolerance for the existing readout formula library."""
from __future__ import annotations
from causal_sandbox.source_compat import sources_match

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .core import ActionSpec
from .observation import LearnedReadoutObservationAdapter, _candidate_formulas
from .context_readout_memory_followup import (
    Family, TupleComponent, feature_matrix, observed_targets, recent_cells,
    measure, sha, source_hashes as original_hashes,
)

ROOT = Path("experiments/noisy_readout_filter_20261114")


def source_hashes():
    return original_hashes() | {"noisy_readout_filter_followup.py": sha(Path(__file__))}


def binary_ba(predictions, target):
    predictions = np.asarray(predictions, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not target.any() or target.all():
        return np.full(predictions.shape[0], np.nan)
    return .5 * (predictions[:, target].mean(axis=1) + (~predictions[:, ~target]).mean(axis=1))


class ValidatedReadoutFilter:
    """Pair-disjoint fit/validation, using the existing formula family."""
    def __init__(self):
        self.history = []

    def observe(self, batch):
        self.history.extend(batch)

    def candidates(self):
        states = [r[side] for r in self.history for side in ("before", "after")]
        fit = np.array([i % 2 == 0 for i in range(len(self.history)) for _ in (0, 1)])
        if fit.sum() < 40 or (~fit).sum() < 40:
            return []
        variables = sorted(states[0])
        records = []
        for target in variables:
            formulas = _candidate_formulas(target, variables)
            y = np.array([s[target] for s in states], dtype=bool)
            if len(np.unique(y[fit])) < 2 or len(np.unique(y[~fit])) < 2:
                continue
            predictions = np.array([[bool(evaluator(s)) for s in states] for _, _, evaluator in formulas])
            scores = binary_ba(predictions[:, fit], y[fit])
            best = min(range(len(formulas)), key=lambda i: (-scores[i], len(formulas[i][1]), formulas[i][0]))
            expression, sources, _ = formulas[best]
            selected = predictions[best:best + 1]
            records.append({"target": target, "expression": expression, "sources": list(sources),
                            "fit_ba": float(scores[best]),
                            "validation_ba": float(binary_ba(selected[:, ~fit], y[~fit])[0]),
                            "pooled_ba": float(binary_ba(selected, y)[0]),
                            "fit_pairs": int(fit.sum() // 2), "validation_pairs": int((~fit).sum() // 2)})
        return records

    @staticmethod
    def suppress(candidates, validation_floor, pooled_floor):
        return {r["target"] for r in candidates if r["fit_ba"] >= validation_floor
                and r["validation_ba"] >= validation_floor and r["pooled_ba"] >= pooled_floor}


def stratified_predict(history, queries, actions, variables, targets, suppressed, seed, next_state=False):
    grouped, query_groups = defaultdict(list), defaultdict(list)
    for r in recent_cells(history):
        grouped[r["action"], tuple(r["context"])].append(r)
    for i, r in enumerate(queries):
        query_groups[r["action"], tuple(r["context"])].append(i)
    prediction = np.zeros((len(queries), len(targets)), dtype=bool)
    blocked = [j for j, v in enumerate(targets) if v in suppressed]
    for key, records in grouped.items():
        x = feature_matrix(records, actions, variables)
        y = np.array([[r["after"][v] for v in targets] for r in records], int) if next_state else observed_targets(records, targets)
        if not next_state:
            y[:, blocked] = 0
        tree = DecisionTreeClassifier(random_state=seed).fit(x, y)
        indices = query_groups[key]
        qs = [queries[i] for i in indices]
        values = tree.predict(feature_matrix(qs, actions, variables)).astype(bool)
        if next_state:
            values ^= np.asarray([[q["before"][v] for v in targets] for q in qs], bool)
        values[:, blocked] = False
        prediction[indices] = values
    return prediction


def run_family(seed, phase, p, lock):
    out = ROOT / "results" / phase
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"family_{seed}.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert sources_match(old["source_hashes"], source_hashes())
        assert old["protocol_sha256"] == sha(ROOT / "protocol.json")
        return old
    start = time.monotonic()
    rows, trials, arrays = [], [], {}
    filter_ids = range(len(p["filter_grid"])) if phase == "development" else [lock["filter_index"]]
    relaxed_ids = range(len(p["relaxed_accuracy_grid"])) if phase == "development" else [lock["relaxed_index"]]
    for condition in p["conditions"]:
        for budget in p["budgets_per_action_context"]:
            family = Family(seed, condition["readout"], p | {"noise_probability": condition["noise_probability"]})
            train_rng = np.random.default_rng(seed * 100 + budget)
            query_rng = np.random.default_rng(seed * 100 + 73)
            actions = [ActionSpec(a, "Apply intervention") for a in family.actions]
            components = {"original": TupleComponent(actions, family.targets),
                          **{f"validated{i}": TupleComponent(actions, family.targets) for i in filter_ids}}
            original = LearnedReadoutObservationAdapter(refresh_interval=10 ** 9)
            relaxed = {i: LearnedReadoutObservationAdapter(min_accuracy=p["relaxed_accuracy_grid"][i], refresh_interval=10 ** 9) for i in relaxed_ids}
            validated = ValidatedReadoutFilter()
            history, round_records = [], []
            for round_idx in range(p["rounds"]):
                batch = []
                for action, context in family.cells(round_idx):
                    for _ in range(budget):
                        row, _ = family.record(action, context, round_idx, train_rng, len(history) + len(batch))
                        batch.append(row)
                        for adapter in (original, *relaxed.values()):
                            adapter.observe_state(row["before"])
                            adapter.observe_state(row["after"])
                history.extend(batch)
                validated.observe(batch)
                candidates = validated.candidates()
                for adapter in (original, *relaxed.values()):
                    adapter._refresh_rules()
                masks = {"original": set(original.suppressed_variables)}
                masks.update({f"validated{i}": validated.suppress(candidates, **p["filter_grid"][i]) for i in filter_ids})
                masks.update({f"relaxed{i}": set(adapter.suppressed_variables) for i, adapter in relaxed.items()})
                for key, component in components.items():
                    component.update(batch, masks[key])
                queries, gold = [], []
                for action in range(3):
                    for context in family.contexts:
                        for _ in range(p["query_count_per_action_context"]):
                            q, y = family.record(action, context, round_idx, query_rng, len(queries))
                            queries.append({k: v for k, v in q.items() if k != "after"})
                            gold.append(y)
                gold = np.asarray(gold)
                prediction = {}
                method_masks = {}
                for key, component in components.items():
                    tag = key if phase == "development" else ("original" if key == "original" else "validated")
                    for name, pred in (
                        (f"components {tag}", component.predict(queries, masks[key])),
                        (f"stratified tree {tag}", stratified_predict(history, queries, family.actions, family.variables, family.targets, masks[key], seed)),
                    ):
                        prediction[name], method_masks[name] = pred, key
                    if key != "original":
                        name = f"stratified next-state {tag}"
                        prediction[name] = stratified_predict(history, queries, family.actions, family.variables, family.targets, masks[key], seed, next_state=True)
                        method_masks[name] = key
                for i in relaxed_ids:
                    key = f"relaxed{i}"
                    name = f"stratified tree {key}" if phase == "development" else "stratified tree relaxed"
                    prediction[name] = stratified_predict(history, queries, family.actions, family.variables, family.targets, masks[key], seed)
                    method_masks[name] = key
                key = f"{condition['name']}__{budget}__round{round_idx}"
                arrays[key + "__gold"] = gold
                for method, pred in prediction.items():
                    mask = masks[method_masks[method]]
                    arrays[key + "__pred__" + method] = pred
                    rows.append({"family": seed, "condition": condition["name"], "budget": budget, "round": round_idx,
                                 "method": method, "filter_key": method_masks[method], "interventions_so_far": len(history),
                                 "physical_suppression": len(set(family.roots) & mask) / 3,
                                 "readout_identified": int(family.readout_name in mask) if family.readout != "none" else None,
                                 **measure(gold, pred, queries, family, round_idx)})
                round_records.append({"batch": batch, "queries": queries, "filter_candidates": candidates,
                                      "masks": {k: sorted(v) for k, v in masks.items()},
                                      "memories": {k: [list(e) for e in c.memory.entries.values()] for k, c in components.items()}})
            trials.append({"condition": condition["name"], "readout": condition["readout"], "budget": budget,
                           "variables": family.variables, "targets": family.targets, "actions": family.actions,
                           "evaluation_only_truth": {"roots": family.roots, "readout": family.readout_name,
                                                     "root_indices": family.root_indices.tolist(), "active_contexts": family.active_contexts,
                                                     "initial_values": family.initial_values.tolist()}, "rounds": round_records})
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    result = {"family": seed, "phase": phase, "protocol_sha256": sha(ROOT / "protocol.json"), "source_hashes": source_hashes(),
              "cache_sha256": sha(path.with_suffix(".npz")), "rows": rows, "trials": trials, "seconds": time.monotonic() - start}
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"DONE {phase} family={seed} seconds={result['seconds']:.1f}", flush=True)
    return result


METRICS = ("direct_ba", "physical_ba", "readout_fpr", "physical_suppression", "readout_identified", "unchanged_ba", "changed_ba", "exact_set")


def summarize(runs, p):
    groups = defaultdict(list)
    for run in runs:
        for row in run["rows"]:
            groups[row["condition"], row["budget"], row["method"]].append(row)
    results = []
    for (condition, budget, method), rows in sorted(groups.items()):
        families = []
        for seed in sorted({r["family"] for r in rows}):
            selected = [r for r in rows if r["family"] == seed]
            item = {"family": seed}
            for m in METRICS:
                values = [r[m] for r in selected if r[m] is not None]
                item[m] = float(np.mean(values)) if values else None
            families.append(item)
        results.append({"condition": condition, "budget": budget, "method": method, "families": families,
                        **{m: float(np.mean([f[m] for f in families])) if families[0][m] is not None else None for m in METRICS}})
    return results


def select_settings(summary, p):
    table = {(r["condition"], r["budget"], r["method"]): r for r in summary}
    condition = p["primary_condition"]
    old = table[condition, 4, "components original"]["physical_ba"]
    candidates = []
    for i in range(len(p["filter_grid"])):
        low = table[condition, 4, f"components validated{i}"]
        high = table[condition, 8, f"components validated{i}"]
        loss = old - low["physical_ba"]
        candidates.append({"index": i, "physical_ba_loss": loss, "guard_pass": loss <= .01,
                           "objective": low["direct_ba"] - high["readout_fpr"]})
    passed = [r for r in candidates if r["guard_pass"]]
    chosen = min(passed, key=lambda r: (-r["objective"], r["index"])) if passed else min(candidates, key=lambda r: (r["physical_ba_loss"], r["index"]))
    relaxed = [{"index": i, "objective": table[condition, 4, f"stratified tree relaxed{i}"]["direct_ba"] -
                 table[condition, 8, f"stratified tree relaxed{i}"]["readout_fpr"]} for i in range(len(p["relaxed_accuracy_grid"]))]
    return {"filter_index": chosen["index"], "relaxed_index": min(relaxed, key=lambda r: (-r["objective"], r["index"]))["index"],
            "development_guard_pass": chosen["guard_pass"], "filter_candidates": candidates, "relaxed_candidates": relaxed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    p = json.loads((ROOT / "protocol.json").read_text())
    lock = None
    if args.phase == "confirmation":
        lock = json.loads((ROOT / "confirmation_lock.json").read_text())
        assert sources_match(lock["source_hashes"], source_hashes())
        assert lock["protocol_sha256"] == sha(ROOT / "protocol.json")
    runs = [run_family(seed, args.phase, p, lock) for seed in p[args.phase + "_families"]]
    summary = summarize(runs, p)
    (ROOT / "results" / args.phase / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.phase == "development":
        selection = select_settings(summary, p)
        (ROOT / "development_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != "families"} for r in summary if r["condition"] == p["primary_condition"]], indent=2))


if __name__ == "__main__":
    main()
