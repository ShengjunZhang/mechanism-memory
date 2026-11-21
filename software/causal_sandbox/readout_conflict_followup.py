"""Incremental readout evidence with a local reciprocal-rule conflict veto."""
from __future__ import annotations
from causal_sandbox.source_compat import sources_match

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from .core import ActionSpec
from .observation import LearnedReadoutObservationAdapter
from .context_readout_memory_followup import Family, TupleComponent, measure, sha
from .noisy_readout_filter_followup import stratified_predict, summarize
from .readout_increment_followup import increment_candidates, settings as parent_settings, source_hashes as parent_hashes

ROOT = Path("experiments/readout_conflict_20261114")
TOL = 1e-12


def source_hashes():
    return parent_hashes() | {"readout_conflict_followup.py": sha(Path(__file__))}


def settings():
    return parent_settings() | json.loads((ROOT / "protocol.json").read_text())


def eligible_mask(candidates, margin):
    if not 0 < margin <= 1:
        raise ValueError("A strictly positive increment is required.")
    return {r["target"] for r in candidates if r["increment"] + TOL >= margin}


def conflict_mask(candidates, margin):
    eligible = eligible_mask(candidates, margin)
    by_target = {r["target"]: r for r in candidates}
    blocked, conflicts = set(), []
    for left, right in combinations(sorted(eligible), 2):
        a, b = by_target[left], by_target[right]
        if right not in a["sources"] or left not in b["sources"]:
            continue
        if a["accuracy"] > b["accuracy"] + TOL:
            veto = [right]
        elif b["accuracy"] > a["accuracy"] + TOL:
            veto = [left]
        else:
            veto = [left, right]
        blocked.update(veto)
        conflicts.append({"left": left, "right": right, "left_accuracy": a["accuracy"],
                          "right_accuracy": b["accuracy"], "veto": veto})
    return eligible - blocked, conflicts


def guard_details(table, p, method):
    guards = []
    def add(name, loss, limit):
        guards.append({"name": name, "loss": float(loss), "limit": limit, "pass": bool(loss <= limit + TOL)})
    for b in p["budgets_per_action_context"]:
        for c in ("or", "identity", "noise10"):
            new, old = table[c, b, method], table[c, b, "components original"]
            add(f"{c}_{b}_physical_loss", old["physical_ba"] - new["physical_ba"], 0.)
            add(f"{c}_{b}_physical_suppression", new["physical_suppression"], 0.)
        new, old = table["noise10", b, method], table["noise10", b, "components calibrated"]
        add(f"noise10_{b}_direct_loss", old["direct_ba"] - new["direct_ba"], .01)
        add(f"noise10_{b}_fpr_increase", new["readout_fpr"] - old["readout_fpr"], .02)
    return guards


def select_settings(summary, p):
    table = {(r["condition"], r["budget"], r["method"]): r for r in summary}
    def objective(method):
        return float(np.mean([table[c["name"], 4, method]["direct_ba"] for c in p["conditions"]]) - table["noise10", 8, method]["readout_fpr"])
    candidates = []
    for i, margin in enumerate(p["increment_grid"]):
        name = f"components conflict{i}"
        guards = guard_details(table, p, name)
        candidates.append({"index": i, "margin": margin, "guards": guards, "guard_pass": all(g["pass"] for g in guards),
                           "guard_excess": sum(max(0., g["loss"] - g["limit"]) for g in guards), "objective": objective(name)})
    passed = [r for r in candidates if r["guard_pass"]]
    selected = min(passed, key=lambda r: (-r["objective"], r["index"])) if passed else min(candidates, key=lambda r: (r["guard_excess"], -r["objective"], r["index"]))
    tags = ["original", "calibrated"] + [f"{kind}{i}" for kind in ("gain", "conflict") for i in range(len(p["increment_grid"]))]
    trees = [{"filter": tag, "objective": objective(f"stratified tree {tag}")} for tag in tags]
    tree = max(trees, key=lambda r: r["objective"])
    return {"increment_index": selected["index"], "development_guard_pass": selected["guard_pass"],
            "tuned_tree_filter": tree["filter"], "candidates": candidates, "tree_candidates": trees}


def run_family(seed, phase, p, lock):
    out = ROOT / "results" / phase
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"family_{seed}.json"
    lock_sha = sha(ROOT / (phase + "_lock.json"))
    if path.exists():
        old = json.loads(path.read_text())
        assert sources_match(old["source_hashes"], source_hashes())
        assert old["protocol_sha256"] == sha(ROOT / "protocol.json")
        assert old["lock_sha256"] == lock_sha
        assert old["cache_sha256"] == sha(path.with_suffix(".npz"))
        return seed, old["seconds"]
    start = time.monotonic()
    rows, trials, arrays = [], [], {}
    gains = list(range(len(p["increment_grid"]))) if phase == "development" else [lock["increment_index"]]
    if phase == "confirmation" and lock["tuned_tree_filter"] not in ("original", "calibrated"):
        gains = sorted(set(gains + [int(lock["tuned_tree_filter"][-1])]))
    for condition in p["conditions"]:
        for budget in p["budgets_per_action_context"]:
            family = Family(seed, condition["readout"], p | {"noise_probability": condition["noise_probability"]})
            train_rng, query_rng = np.random.default_rng(seed * 100 + budget), np.random.default_rng(seed * 100 + 73)
            actions = [ActionSpec(a, "Apply intervention") for a in family.actions]
            tags = ["original", "calibrated"] + [f"{kind}{i}" for kind in ("gain", "conflict") for i in gains]
            components = {tag: TupleComponent(actions, family.targets) for tag in tags}
            adapters = {tag: LearnedReadoutObservationAdapter(min_accuracy=accuracy, refresh_interval=10 ** 9)
                        for tag, accuracy in (("original", .98), ("calibrated", p["min_accuracy"]))}
            history, round_records = [], []
            for round_idx in range(p["rounds"]):
                batch = []
                for action, context in family.cells(round_idx):
                    for _ in range(budget):
                        row, _ = family.record(action, context, round_idx, train_rng, len(history) + len(batch))
                        batch.append(row)
                        for adapter in adapters.values():
                            adapter.observe_state(row["before"])
                            adapter.observe_state(row["after"])
                history.extend(batch)
                for adapter in adapters.values():
                    adapter._refresh_rules()
                candidates = increment_candidates(adapters["calibrated"])
                masks = {tag: set(adapter.suppressed_variables) for tag, adapter in adapters.items()}
                conflicts = {}
                for i in gains:
                    masks[f"gain{i}"] = eligible_mask(candidates, p["increment_grid"][i])
                    masks[f"conflict{i}"], conflicts[f"conflict{i}"] = conflict_mask(candidates, p["increment_grid"][i])
                for tag, component in components.items():
                    component.update(batch, masks[tag])
                queries, gold = [], []
                for action in range(3):
                    for context in family.contexts:
                        for _ in range(p["query_count_per_action_context"]):
                            query, truth = family.record(action, context, round_idx, query_rng, len(queries))
                            queries.append({k: v for k, v in query.items() if k != "after"})
                            gold.append(truth)
                gold = np.asarray(gold)
                key = f"{condition['name']}__{budget}__round{round_idx}"
                arrays[key + "__gold"] = gold
                prediction, keys = {}, {}
                for tag in tags:
                    label = tag
                    if phase == "confirmation" and tag not in ("original", "calibrated"):
                        if int(tag[-1]) != lock["increment_index"]:
                            continue
                        label = "increment" if tag.startswith("gain") else "conflict"
                    prediction[f"components {label}"] = components[tag].predict(queries, masks[tag])
                    prediction[f"stratified tree {label}"] = stratified_predict(history, queries, family.actions, family.variables, family.targets, masks[tag], seed)
                    keys[f"components {label}"] = keys[f"stratified tree {label}"] = tag
                    if label == "conflict":
                        name = "stratified next-state conflict"
                        prediction[name] = stratified_predict(history, queries, family.actions, family.variables, family.targets, masks[tag], seed, next_state=True)
                        keys[name] = tag
                if phase == "confirmation":
                    tag = lock["tuned_tree_filter"]
                    prediction["stratified tree tuned"] = stratified_predict(history, queries, family.actions, family.variables, family.targets, masks[tag], seed)
                    keys["stratified tree tuned"] = tag
                for method, pred in prediction.items():
                    tag = keys[method]
                    arrays[key + "__pred__" + method] = pred
                    rows.append({"family": seed, "condition": condition["name"], "budget": budget, "round": round_idx,
                                 "method": method, "filter_key": tag, "interventions_so_far": len(history),
                                 "physical_suppression": len(set(family.roots) & masks[tag]) / 3,
                                 "readout_identified": int(family.readout_name in masks[tag]) if family.readout != "none" else None,
                                 **measure(gold, pred, queries, family, round_idx)})
                round_records.append({"batch": batch, "queries": queries, "filter_candidates": [], "increment_candidates": candidates,
                                      "masks": {tag: sorted(mask) for tag, mask in masks.items()}, "conflicts": conflicts,
                                      "readout_rules": {tag: [asdict(rule) for rule in adapter.readout_rules.values()] for tag, adapter in adapters.items()},
                                      "memories": {tag: [list(entry) for entry in component.memory.entries.values()] for tag, component in components.items()}})
            trials.append({"condition": condition["name"], "readout": condition["readout"], "budget": budget,
                           "variables": family.variables, "targets": family.targets, "actions": family.actions,
                           "evaluation_only_truth": {"roots": family.roots, "readout": family.readout_name,
                                                     "root_indices": family.root_indices.tolist(), "active_contexts": family.active_contexts,
                                                     "initial_values": family.initial_values.tolist()}, "rounds": round_records})
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    result = {"family": seed, "phase": phase, "protocol_sha256": sha(ROOT / "protocol.json"), "source_hashes": source_hashes(),
              "lock_sha256": lock_sha, "cache_sha256": sha(path.with_suffix(".npz")), "rows": rows, "trials": trials, "seconds": time.monotonic() - start}
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return seed, result["seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    p = settings()
    assert p["numeric_tolerance"] == TOL
    lock = json.loads((ROOT / (args.phase + "_lock.json")).read_text())
    assert sources_match(source_hashes(), lock["source_hashes"])
    assert sha(ROOT / "protocol.json") == lock["protocol_sha256"]
    assert sha(Path(p["parent_protocol"])) == lock["parent_protocol_sha256"]
    assert sha(Path("experiments/noisy_readout_filter_20261114/protocol.json")) == lock["ancestor_protocol_sha256"]
    if args.phase == "confirmation":
        assert sha(ROOT / "results/development/summary.json") == lock["development_summary_sha256"]
        assert sha(ROOT / "development_selection.json") == lock["development_selection_sha256"]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_family, seed, args.phase, p, lock) for seed in p[args.phase + "_families"]]
        for future in as_completed(futures):
            seed, seconds = future.result()
            print(f"DONE {args.phase} family={seed} elapsed={seconds:.1f}s", flush=True)
    folder = ROOT / "results" / args.phase
    runs = [json.loads((folder / f"family_{seed}.json").read_text()) for seed in p[args.phase + "_families"]]
    summary = summarize(runs, p)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.phase == "development":
        selected = select_settings(summary, p)
        (ROOT / "development_selection.json").write_text(json.dumps(selected, indent=2) + "\n")
        print(json.dumps(selected, indent=2))
    else:
        print(json.dumps([{k: v for k, v in r.items() if k != "families"} for r in summary if r["condition"] == "noise10"], indent=2))


if __name__ == "__main__":
    main()
