"""Shared calibration of the existing adapter; no learner changes."""
from __future__ import annotations
from causal_sandbox.source_compat import sources_match

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .core import ActionSpec
from .observation import LearnedReadoutObservationAdapter
from .context_readout_memory_followup import Family, TupleComponent, measure, sha
from .noisy_readout_filter_followup import stratified_predict, summarize, source_hashes as parent_hashes

ROOT = Path("experiments/noisy_readout_threshold_20261114")


def source_hashes():
    return parent_hashes() | {"noisy_readout_threshold_followup.py": sha(Path(__file__))}


def settings():
    local = json.loads((ROOT / "protocol.json").read_text())
    parent = json.loads(Path(local["parent_protocol"]).read_text())
    selection = json.loads(Path(local["parent_selection"]).read_text())
    assert local["min_accuracy"] == parent["relaxed_accuracy_grid"][selection["relaxed_index"]]
    return parent | local


def run_family(seed, p):
    out = ROOT / "results/confirmation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"family_{seed}.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert sources_match(old["source_hashes"], source_hashes())
        assert old["protocol_sha256"] == sha(ROOT / "protocol.json")
        assert old["cache_sha256"] == sha(path.with_suffix(".npz"))
        return seed, old["seconds"]
    start = time.monotonic()
    rows, trials, arrays = [], [], {}
    for condition in p["conditions"]:
        for budget in p["budgets_per_action_context"]:
            family = Family(seed, condition["readout"], p | {"noise_probability": condition["noise_probability"]})
            train_rng = np.random.default_rng(seed * 100 + budget)
            query_rng = np.random.default_rng(seed * 100 + 73)
            actions = [ActionSpec(a, "Apply intervention") for a in family.actions]
            components = {tag: TupleComponent(actions, family.targets) for tag in ("original", "calibrated")}
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
                masks = {tag: set(adapter.suppressed_variables) for tag, adapter in adapters.items()}
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
                for tag, component in components.items():
                    predictions = {
                        f"components {tag}": component.predict(queries, masks[tag]),
                        f"stratified tree {tag}": stratified_predict(history, queries, family.actions, family.variables, family.targets, masks[tag], seed),
                    }
                    if tag == "calibrated":
                        predictions["stratified next-state calibrated"] = stratified_predict(
                            history, queries, family.actions, family.variables, family.targets, masks[tag], seed, next_state=True)
                    for method, pred in predictions.items():
                        arrays[key + "__pred__" + method] = pred
                        rows.append({"family": seed, "condition": condition["name"], "budget": budget, "round": round_idx,
                                     "method": method, "filter_key": tag, "interventions_so_far": len(history),
                                     "physical_suppression": len(set(family.roots) & masks[tag]) / 3,
                                     "readout_identified": int(family.readout_name in masks[tag]) if family.readout != "none" else None,
                                     **measure(gold, pred, queries, family, round_idx)})
                round_records.append({"batch": batch, "queries": queries, "filter_candidates": [],
                                      "masks": {tag: sorted(mask) for tag, mask in masks.items()},
                                      "readout_rules": {tag: [asdict(rule) for rule in adapter.readout_rules.values()] for tag, adapter in adapters.items()},
                                      "memories": {tag: [list(entry) for entry in component.memory.entries.values()] for tag, component in components.items()}})
            trials.append({"condition": condition["name"], "readout": condition["readout"], "budget": budget,
                           "variables": family.variables, "targets": family.targets, "actions": family.actions,
                           "evaluation_only_truth": {"roots": family.roots, "readout": family.readout_name,
                                                     "root_indices": family.root_indices.tolist(), "active_contexts": family.active_contexts,
                                                     "initial_values": family.initial_values.tolist()}, "rounds": round_records})
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    result = {"family": seed, "phase": "confirmation", "protocol_sha256": sha(ROOT / "protocol.json"), "source_hashes": source_hashes(),
              "cache_sha256": sha(path.with_suffix(".npz")), "rows": rows, "trials": trials, "seconds": time.monotonic() - start}
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return seed, result["seconds"]


def main():
    p = settings()
    lock = json.loads((ROOT / "confirmation_lock.json").read_text())
    assert sources_match(source_hashes(), lock["source_hashes"])
    assert sha(ROOT / "protocol.json") == lock["protocol_sha256"]
    assert sha(Path(p["parent_protocol"])) == lock["parent_protocol_sha256"]
    assert sha(Path(p["parent_selection"])) == lock["parent_selection_sha256"]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_family, seed, p) for seed in p["confirmation_families"]]
        for future in as_completed(futures):
            seed, seconds = future.result()
            print(f"CONFIRMED family={seed} elapsed={seconds:.1f}s", flush=True)
    folder = ROOT / "results/confirmation"
    runs = [json.loads((folder / f"family_{seed}.json").read_text()) for seed in p["confirmation_families"]]
    summary = summarize(runs, p)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "families"} for r in summary if r["condition"] == "noise10"], indent=2))


if __name__ == "__main__":
    main()
