"""Re-score complete cellwise answer sets without changing experimental records."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "jmlr_revision_20261112/results"


def context_key(value):
    return tuple(sorted((str(name), int(state)) for name, state in value))


def mechanism_key(row):
    return (row["action"], context_key(row["context"]), row["target"], row["value"], row["delay"])


def summarize(runs):
    totals = defaultdict(lambda: defaultdict(int))
    for run in runs:
        universe = run["candidate_universe"]
        cells = {(c["action"], context_key(c["context"])) for c in universe}
        assert len(cells) == 2
        candidates = {(c["action"], context_key(c["context"]), c["target"]) for c in universe}
        truth = {mechanism_key(row) for row in run["truth_evaluator_only"]}
        for method, output in run["methods"].items():
            predicted = {mechanism_key(row) for row in output["predicted"]}
            assert {p[:3] for p in predicted | truth} <= candidates
            stats = totals[method]
            stats["runs"] += 1
            stats["memory_errors"] += predicted != truth
            for action, context in cells:
                target = {t for t in truth if t[:2] == (action, context)}
                answer = {p for p in predicted if p[:2] == (action, context)}
                stats["answer_queries"] += 1
                stats["answer_errors"] += target != answer
                stats["null_answer_queries"] += not target
                stats["false_nonempty_answers"] += not target and bool(answer)
                stats["missing_positive_answers"] += bool(target) and not answer
                stats["incorrect_nonempty_answers"] += bool(answer) and answer != target
            for candidate in candidates:
                stats["candidate_queries"] += 1
                stats["candidate_errors"] += ({p for p in predicted if p[:3] == candidate}
                                              != {t for t in truth if t[:3] == candidate})
    return {method: {**dict(stats),
                     "answer_risk": stats["answer_errors"] / stats["answer_queries"],
                     "candidate_risk": stats["candidate_errors"] / stats["candidate_queries"],
                     "memory_error_rate": stats["memory_errors"] / stats["runs"]}
            for method, stats in totals.items()}


def main():
    paths = [RESULTS / name for name in ("finite_results.json", "finite_budget_curve.json")]
    primary, budgets = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = summarize(primary["controlled_runs"])
    expected = {"contrast_peak_formula_emulation": 36, "paired_modal_first_without_direct": 36,
                "paired_modal_first_direct": 9, "paired_direct_candidate_omission": 24}
    assert all(rows[method]["answer_errors"] == errors for method, errors in expected.items())
    curves = []
    for budget in budgets["budgets"]:
        runs = budget["runs"]
        curves.append({"pairs_per_cell": budget["pairs_per_cell"],
                       "all_conditions": summarize(runs),
                       "in_class": summarize([r for r in runs if r["scenario"] != "weak_early"]),
                       "weak_early": summarize([r for r in runs if r["scenario"] == "weak_early"])})
    result = {"estimand": "Full action/context answer-set disagreement, uniform over both cells in each run; cellwise answers are state-homogeneous.",
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              "primary": rows, "budgets": curves,
              "independence_note": "36 runs reuse 12 SCM families; 72 cell queries are not 72 independent families."}
    (RESULTS / "finite_answer_risk.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
