"""Same-history component evaluation for observed contexts, readouts and retention."""
from __future__ import annotations
from causal_sandbox.source_compat import sources_match

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

from .core import ActionSpec, Transition
from .causal_core import CausalCore
from .observation import LearnedReadoutObservationAdapter
from .mechanism_memory import IdentityMemory, MechanismTuple

ROOT = Path("experiments/context_readout_memory_20261114")
DEPENDENCIES = ["causal_sandbox/core.py", "causal_sandbox/causal_core.py", "causal_sandbox/memory.py",
                "causal_sandbox/observation.py", "causal_sandbox/mechanism_memory.py", __file__]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {Path(p).name: sha(p) for p in DEPENDENCIES}


def balanced_accuracy(gold, predicted):
    gold, predicted = np.asarray(gold, bool).ravel(), np.asarray(predicted, bool).ravel()
    recalls = [np.mean(predicted[gold]), np.mean(~predicted[~gold])] if gold.any() and (~gold).any() else [np.mean(gold == predicted)]
    return float(np.mean(recalls))


def recent_cells(history):
    latest = {}
    for row in history:
        latest[row["action"], tuple(row["context"])] = row["round"]
    return [r for r in history if r["round"] == latest[r["action"], tuple(r["context"])]]


class TupleComponent:
    def __init__(self, actions, targets):
        self.actions, self.targets = actions, targets
        self.memory = IdentityMemory()

    def update(self, batch, suppressed):
        grouped = defaultdict(list)
        for row in batch:
            grouped[row["action"], tuple(row["context"])].append(row)
        for (action, context), records in grouped.items():
            core = CausalCore()
            core.reset(self.actions)
            for row in records:
                before = {k: v for k, v in row["before"].items() if k not in suppressed}
                after = {k: v for k, v in row["after"].items() if k not in suppressed}
                core.update(Transition(row["index"], action, before, after, frozenset(), "shared intervention record"))
            for source, target in core.discovered_edges():
                value = core._edge_effect_value(source, target)
                if value is None or target not in self.targets:
                    continue
                identity = (source, context, target)
                entry = MechanismTuple(source, context, target, value, 0)
                if identity in self.memory.entries:
                    self.memory.local_update({identity: entry})
                else:
                    self.memory.replace(identity, entry)

    def predict(self, queries, suppressed):
        output = np.zeros((len(queries), len(self.targets)), bool)
        for i, row in enumerate(queries):
            for j, target in enumerate(self.targets):
                entry = self.memory.entries.get((row["action"], tuple(row["context"]), target))
                output[i, j] = bool(entry is not None and target not in suppressed and row["before"][target] != entry.value)
        return output


class Family:
    def __init__(self, seed, readout, protocol):
        self.seed, self.readout, self.p = seed, readout, protocol
        self.rng = np.random.default_rng(seed)
        names = [f"v{v:05d}" for v in self.rng.choice(90000, 8, replace=False)]
        self.roots, self.context_names, self.nuisance = names[:3], names[3:5], names[5:7]
        self.readout_name = names[7]
        self.variables = sorted(names[:-1] if readout == "none" else names)
        self.targets = [v for v in self.variables if v not in self.context_names]
        self.actions = [f"a{v:05d}" for v in self.rng.choice(90000, 3, replace=False)]
        self.root_indices = self.rng.permutation(3)
        self.active_contexts = [tuple(map(int, self.rng.integers(0, 2, 2))) for _ in self.actions]
        self.initial_values = self.rng.integers(0, 2, 3)
        self.contexts = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def state(self, context, rng):
        result = {v: bool(rng.integers(2)) for v in self.roots + self.nuisance}
        result.update({v: bool(c) for v, c in zip(self.context_names, context)})
        self.read(result, rng)
        return result

    def read(self, state, rng):
        if self.readout == "none":
            return
        if self.readout == "identity":
            value = state[self.roots[0]]
        else:
            value = state[self.roots[0]] or state[self.roots[1]]
        if self.readout == "noisy-or" and rng.random() < self.p["noise_probability"]:
            value = not value
        state[self.readout_name] = value

    def value(self, action_idx, round_idx):
        changed = (action_idx == 0 and round_idx in (1, 2)) or (action_idx == 1 and round_idx >= 2)
        return bool(self.initial_values[action_idx]) ^ changed

    def record(self, action_idx, context, round_idx, rng, index):
        before = self.state(context, rng)
        after = dict(before)
        physical = np.zeros(len(self.targets), bool)
        if tuple(context) == self.active_contexts[action_idx]:
            target = self.roots[self.root_indices[action_idx]]
            after[target] = self.value(action_idx, round_idx)
            physical[self.targets.index(target)] = before[target] != after[target]
        self.read(after, rng)
        return {"action": self.actions[action_idx], "context": list(context), "round": round_idx,
                "before": before, "after": after, "index": index}, physical

    def cells(self, round_idx):
        if round_idx == 0:
            return [(a, c) for a in range(3) for c in self.contexts]
        action = 1 if round_idx == 2 else 0
        return [(action, self.active_contexts[action])]


def feature_matrix(rows, actions, variables):
    return np.asarray([[float(row["action"] == a) for a in actions] + [float(row["before"][v]) for v in variables] for row in rows])


def observed_targets(rows, targets):
    return np.asarray([[r["before"][v] != r["after"][v] for v in targets] for r in rows], dtype=int)


def measure(gold, pred, queries, family, round_idx):
    physical_idx = [family.targets.index(v) for v in family.roots]
    current = set((family.actions[a], c) for a, c in family.cells(round_idx)) if round_idx else set()
    changed = np.array([(q["action"], tuple(q["context"])) in current for q in queries])
    return {"direct_ba": balanced_accuracy(gold, pred),
            "physical_ba": balanced_accuracy(gold[:, physical_idx], pred[:, physical_idx]),
            "readout_fpr": float(pred[:, family.targets.index(family.readout_name)].mean()) if family.readout != "none" else None,
            "unchanged_ba": balanced_accuracy(gold[~changed], pred[~changed]),
            "changed_ba": balanced_accuracy(gold[changed], pred[changed]) if changed.any() else None,
            "exact_set": float(np.mean(np.all(gold == pred, axis=1)))}


def run_family(seed, phase, p, lock):
    out = ROOT / "results" / phase
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"family_{seed}.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert sources_match(old["source_hashes"], source_hashes())
        assert old["protocol_sha256"] == sha(ROOT / "protocol.json")
        return old
    started = time.monotonic()
    all_rows, trials, arrays = [], [], {}
    for readout in p["readouts"]:
        for budget in p["budgets_per_action_context"]:
            family = Family(seed, readout, p)
            prefix = f"{readout}__{budget}"
            train_rng = np.random.default_rng(seed * 100 + budget)
            query_rng = np.random.default_rng(seed * 100 + 73)
            component = TupleComponent([ActionSpec(a, "Apply intervention") for a in family.actions], family.targets)
            adapter = LearnedReadoutObservationAdapter(refresh_interval=10 ** 9)
            history, trial_rounds = [], []
            for round_idx in range(p["rounds"]):
                batch = []
                for action, context in family.cells(round_idx):
                    for _ in range(budget):
                        row, _ = family.record(action, context, round_idx, train_rng, len(history) + len(batch))
                        batch.append(row)
                        adapter.observe_state(row["before"])
                        adapter.observe_state(row["after"])
                history.extend(batch)
                adapter._refresh_rules()
                suppressed = adapter.suppressed_variables
                component.update(batch, suppressed)
                queries, gold = [], []
                for action in range(3):
                    for context in family.contexts:
                        for _ in range(p["query_count_per_action_context"]):
                            q, y = family.record(action, context, round_idx, query_rng, len(queries))
                            queries.append({k: v for k, v in q.items() if k != "after"})
                            gold.append(y)
                gold = np.asarray(gold)
                features = feature_matrix(queries, family.actions, family.variables)
                prediction = {"tuple components": component.predict(queries, suppressed)}
                replay = recent_cells(history)
                for name, records in (("full-history tree", history), ("context replay tree", replay),
                                      ("context replay + shared filter", replay), ("context replay next-state + shared filter", replay)):
                    settings = list(enumerate(p["tree_grid"])) if phase == "development" else [(lock["settings"][name], p["tree_grid"][lock["settings"][name]])]
                    x, y = feature_matrix(records, family.actions, family.variables), observed_targets(records, family.targets)
                    next_state = "next-state" in name
                    if next_state:
                        y = np.asarray([[r["after"][v] for v in family.targets] for r in records], dtype=int)
                    if "shared filter" in name and not next_state:
                        y[:, [i for i, v in enumerate(family.targets) if v in suppressed]] = 0
                    for setting_idx, setting in settings:
                        tree = DecisionTreeClassifier(random_state=seed, **setting).fit(x, y)
                        label = f"{name}__setting{setting_idx}" if phase == "development" else name
                        predicted = tree.predict(features).astype(bool)
                        if next_state:
                            predicted = predicted != np.asarray([[r["before"][v] for v in family.targets] for r in queries])
                            predicted[:, [i for i, v in enumerate(family.targets) if v in suppressed]] = False
                        prediction[label] = predicted
                x, y = feature_matrix(history, family.actions, family.variables), observed_targets(history, family.targets)
                y[:, [i for i, v in enumerate(family.targets) if v in suppressed]] = 0
                forest = ExtraTreesClassifier(n_estimators=p["forest_trees"], random_state=seed, n_jobs=1).fit(x, y)
                prediction["full-history forest + shared filter"] = forest.predict(features).astype(bool)
                key = f"{prefix}__round{round_idx}"
                arrays[key + "__gold"] = gold
                for method, pred in prediction.items():
                    arrays[key + "__pred__" + method] = pred
                    all_rows.append({"family": seed, "readout": readout, "budget": budget, "round": round_idx,
                                     "method": method, "interventions_so_far": len(history), **measure(gold, pred, queries, family, round_idx)})
                trial_rounds.append({"batch": batch, "queries": queries, "suppressed": sorted(suppressed),
                                     "memory": [list(entry) for entry in component.memory.entries.values()]})
            trials.append({"readout": readout, "budget": budget, "variables": family.variables, "targets": family.targets,
                           "context_names": family.context_names, "actions": family.actions,
                           "evaluation_only_truth": {"roots": family.roots, "readout": family.readout_name,
                                                     "root_indices": family.root_indices.tolist(), "active_contexts": family.active_contexts,
                                                     "initial_values": family.initial_values.tolist()}, "rounds": trial_rounds})
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    result = {"family": seed, "phase": phase, "protocol_sha256": sha(ROOT / "protocol.json"), "source_hashes": source_hashes(),
              "cache_sha256": sha(path.with_suffix(".npz")), "rows": all_rows, "trials": trials, "seconds": time.monotonic() - started}
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"DONE {phase} family={seed} seconds={result['seconds']:.1f}", flush=True)
    return result


def summary(runs, p):
    groups = defaultdict(list)
    for run in runs:
        for row in run["rows"]:
            if row["budget"] == p["primary_budget"] and row["readout"] == p["primary_readout"]:
                groups[row["method"]].append(row)
    result = []
    for method, rows in sorted(groups.items()):
        families = [{"family": f, "direct_ba": float(np.mean([r["direct_ba"] for r in rows if r["family"] == f]))} for f in sorted({r["family"] for r in rows})]
        result.append({"method": method, "direct_ba": float(np.mean([v["direct_ba"] for v in families])), "families": families})
    return result


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
    result = summary(runs, p)
    (ROOT / "results" / args.phase / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "families"} for r in result], indent=2))


if __name__ == "__main__":
    main()
