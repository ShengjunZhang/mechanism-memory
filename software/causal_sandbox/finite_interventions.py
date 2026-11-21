"""Isolated reviewer experiments; historical writers are imported, never changed.

The new writer targets natural-law modal outcomes and first *detectable* response. Its
directness certificate requires surgical interventions on every other observed
endogenous node. Structural first response can be below the detection threshold;
the weak_early condition measures that failure rather than redefining the truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations, product
import json
from math import log, sqrt
from pathlib import Path
from random import Random
import sys
import time
from typing import Iterable

import numpy as np

from .agents import CausalCoreAgent, ContextSearchControlExperimentPlannerAgent
from .core import ActionSpec, Transition
from .evaluation import score_edges
from .worlds import make_procedural_complex_hidden_world


ROOT = Path(__file__).resolve().parents[1]
Context = tuple[tuple[str, int], ...]
REQUIRED_SOURCES = (
    "causal_sandbox/core.py", "causal_sandbox/evaluation.py", "causal_sandbox/causal_core.py",
    "causal_sandbox/agents.py", "causal_sandbox/worlds.py", "causal_sandbox/observation.py",
    "causal_sandbox/finite_interventions.py", "tests/test_finite_interventions.py",
)
OPTIONAL_REVIEWS = ("review/review-1.txt", "review/review-2.txt", "review/AE-review.txt")


@dataclass(frozen=True, order=True)
class Candidate:
    action: str
    context: Context
    target: str


@dataclass(frozen=True, order=True)
class Mechanism:
    action: str
    context: Context
    target: str
    value: int
    delay: int

    @property
    def candidate(self) -> Candidate:
        return Candidate(self.action, self.context, self.target)


@dataclass(frozen=True)
class PairStats:
    n: int
    p1: float
    p0: float
    paired_difference: float
    formula_agreement: float


@dataclass(frozen=True)
class DirectEvidence:
    # All assignments of the other observed nodes must be present, at every lag.
    clamped_variables: tuple[str, ...]
    cells: dict[tuple[int, tuple[int, ...]], PairStats]


@dataclass
class Evidence:
    variables: tuple[str, ...]
    lags: tuple[int, ...]
    natural: dict[tuple[Candidate, int], PairStats]
    direct: dict[Candidate, DirectEvidence]
    ledger: dict[str, int]


@dataclass(frozen=True)
class GateConfig:
    effect_threshold: float = 0.2
    familywise_alpha: float = 0.05
    formula_agreement_threshold: float = 0.9


@dataclass
class WriteResult:
    mechanisms: set[Mechanism]
    decisions: list[dict]


class FiniteSCM:
    """Two-node, two-lag acyclic SCM with finite Bernoulli exogenous support.

    Y_l = 1[U_l < q(A,C,l)]; R_l = Y_l XOR N_l. Clamps replace
    equations from pre-context setup through the queried endpoint, not observed
    values after sampling. The learner sees anonymous
    node names, context assignments and measurements, never root/readout roles.
    Each endpoint is a fresh trial; lags are queried separately, not a rollout.
    """

    def __init__(self, family_seed: int, scenario: str, varied: bool = True):
        rng = Random(family_seed)
        names = [f"v_{family_seed}_{i}" for i in range(2)]
        rng.shuffle(names)
        self.root, self.descendant = names
        self.variables = tuple(sorted(names))
        self.action = f"a_{family_seed}"
        self.context_variable = f"c_{family_seed}"
        self.active_context = rng.randrange(2)
        self.family_seed, self.scenario = family_seed, scenario
        self.lags = (1, 2)
        f = Fraction
        parameters = {
            "modal": ((f(1, 10), f(1, 10)), (f(2, 5), f(2, 5))),
            "late_peak": ((f(1, 10), f(1, 10)), (f(7, 10), f(19, 20))),
            "noisy_descendant": ((f(0), f(0)), (f(1), f(1))),
            "weak_early": ((f(1, 10), f(1, 10)), (f(7, 50), f(9, 10))),
        }
        if scenario not in parameters:
            raise ValueError(f"unknown scenario: {scenario}")
        self.q0, self.q1 = parameters[scenario]
        if varied and scenario != "noisy_descendant":
            # A prespecified grid, drawn without any learner results.
            offset = rng.choice([f(-1, 50), f(0), f(1, 50)])
            self.q0 = tuple(q + offset for q in self.q0)
            self.q1 = tuple(q + offset for q in self.q1)
        self.noise = rng.choice([f(3, 20), f(1, 5), f(1, 4)]) if varied else f(1, 5)

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(
            Candidate(self.action, ((self.context_variable, c),), v)
            for c in (0, 1) for v in self.variables
        )

    def probability(self, arm: int, context: int, lag: int) -> Fraction:
        return self.q1[lag - 1] if arm and context == self.active_context else self.q0[lag - 1]

    def endpoint(self, arm: int, context: int, lag: int, clamps: dict[str, int],
                 u: np.ndarray, noise: np.ndarray) -> dict[str, np.ndarray]:
        y = (u < float(self.probability(arm, context, lag)))
        if self.root in clamps:
            y = np.full(u.shape, bool(clamps[self.root]))
        r = np.logical_xor(y, noise < float(self.noise))
        if self.descendant in clamps:
            r = np.full(u.shape, bool(clamps[self.descendant]))
        return {self.root: y, self.descendant: r}

    def truth(self) -> set[Mechanism]:
        lag = next((l for l in self.lags if self.q1[l - 1] != self.q0[l - 1]), None)
        if lag is None:
            return set()
        p = self.q1[lag - 1]
        if p == Fraction(1, 2):
            raise ValueError("modal tie is outside this protocol")
        return {Mechanism(self.action, ((self.context_variable, self.active_context),),
                          self.root, int(p > Fraction(1, 2)), lag)}

    def population(self, arm: int, context: int, lag: int,
                   clamps: dict[str, int] | None = None) -> dict[str, Fraction]:
        """Exact evaluator-only enumeration of the finite exogenous support."""
        clamps = clamps or {}
        p = self.probability(arm, context, lag)
        result = {v: Fraction(0) for v in self.variables}
        agreement = Fraction(0)
        for y0, noise in product((0, 1), repeat=2):
            weight = (p if y0 else 1 - p) * (self.noise if noise else 1 - self.noise)
            y = clamps.get(self.root, y0)
            r = clamps.get(self.descendant, y ^ noise)
            result[self.root] += weight * y
            result[self.descendant] += weight * r
            agreement += weight * (y == r)
        result["copy_agreement"] = agreement
        return result


def _stats(one: dict[str, np.ndarray], zero: dict[str, np.ndarray], target: str,
           other: str) -> PairStats:
    x, y = one[target], zero[target]
    agreement = np.concatenate((x == one[other], y == zero[other])).mean()
    return PairStats(len(x), float(x.mean()), float(y.mean()),
                     float((x.astype(float) - y.astype(float)).mean()),
                     float(max(agreement, 1 - agreement)))


def collect_evidence(scm: FiniteSCM, seed: int, n: int, direct: bool = True) -> Evidence:
    if n < 1:
        raise ValueError("pairs per cell must be positive")
    rng = np.random.default_rng(seed)
    natural, separated = {}, {}
    ledger: Counter[str] = Counter()

    def pair(c: int, lag: int, clamps: dict[str, int], phase: str):
        u, noise = rng.random(n), rng.random(n)
        one = scm.endpoint(1, c, lag, clamps, u, noise)
        zero = scm.endpoint(0, c, lag, clamps, u, noise)
        # A vectorized endpoint is n intervention calls, not one call.
        ledger[f"{phase}_joint_do_calls"] += 2 * n
        ledger[f"{phase}_paired_trials"] += n
        ledger[f"{phase}_atomic_assignments"] += 2 * n * (2 + len(clamps))
        ledger["fresh_endpoint_instances"] += 2 * n
        return one, zero

    for c in (0, 1):
        for lag in scm.lags:
            one, zero = pair(c, lag, {}, "natural")
            for v in scm.variables:
                candidate = Candidate(scm.action, ((scm.context_variable, c),), v)
                other = next(x for x in scm.variables if x != v)
                natural[candidate, lag] = _stats(one, zero, v, other)
        if direct:
            for v in scm.variables:
                candidate = Candidate(scm.action, ((scm.context_variable, c),), v)
                others = tuple(x for x in scm.variables if x != v)
                cells = {}
                for lag, assignment in product(scm.lags, product((0, 1), repeat=len(others))):
                    one, zero = pair(c, lag, dict(zip(others, assignment)), "separation")
                    cells[lag, assignment] = _stats(one, zero, v, others[0])
                separated[candidate] = DirectEvidence(others, cells)
    ledger["total_joint_do_calls"] = ledger["natural_joint_do_calls"] + ledger["separation_joint_do_calls"]
    ledger["total_atomic_assignments"] = ledger["natural_atomic_assignments"] + ledger["separation_atomic_assignments"]
    ledger["extra_reset_calls"] = 0
    ledger["extra_setup_calls"] = 0  # Context is included in every joint do.
    return Evidence(scm.variables, scm.lags, natural, separated, dict(ledger))


def confidence_radii(evidence: Evidence, config: GateConfig) -> tuple[float, float]:
    stats = list(evidence.natural.values())
    stats += [s for d in evidence.direct.values() for s in d.cells.values()]
    if not stats:
        return float("inf"), float("inf")
    # Simultaneous two-sided Hoeffding bounds for differences in [-1,1] and p1.
    # Cells can share random numbers; the union bound needs no cell independence.
    n = min((s.n for s in stats if s.n > 0), default=1)
    k = 2 * len(stats)
    term = log(2 * k / config.familywise_alpha)
    return sqrt(2 * term / n), sqrt(term / (2 * n))


def paired_writer(evidence: Evidence, candidates: Iterable[Candidate], config: GateConfig,
                  require_direct: bool = True) -> WriteResult:
    candidates = tuple(candidates)
    delta_radius, mode_radius = confidence_radii(evidence, config)
    mechanisms, decisions = set(), []
    for candidate in candidates:
        row = {"candidate": asdict(candidate), "status": "abstained"}
        cells = [evidence.natural.get((candidate, lag)) for lag in evidence.lags]
        row["natural"] = [asdict(s) if s else None for s in cells]
        row["natural_diagnostics"] = [
            {"lag": lag, "tv_lower": max(0.0, abs(s.paired_difference) - delta_radius),
             "tv_upper": min(1.0, abs(s.paired_difference) + delta_radius),
             "detectability_gap_lower": abs(s.paired_difference) - delta_radius - config.effect_threshold,
             "modal_gap_lower": abs(s.p1 - 0.5) - mode_radius}
            for lag, s in zip(evidence.lags, cells) if s is not None]
        row["onset_law"] = row["modal_value_law"] = "natural"
        row["directness_certified"] = False
        row["contrast_radius"] = delta_radius
        row["modal_probability_radius"] = mode_radius
        row["eligible"] = all(s is not None and s.n > 0 for s in cells)
        reason = "missing_natural_cell"
        natural_response = any(s is not None and abs(s.paired_difference) - delta_radius > config.effect_threshold
                               for s in cells)
        if row["eligible"] and not require_direct:
            # Observed response alone never authorizes a structural-direct write.
            reason = "missing_direct_separation" if natural_response else "no_detectable_response"
        elif row["eligible"]:
            certificate = evidence.direct.get(candidate)
            others = tuple(v for v in evidence.variables if v != candidate.target)
            assignments = tuple(product((0, 1), repeat=len(others)))
            expected = set(product(evidence.lags, assignments))
            if (certificate is None or certificate.clamped_variables != others
                    or set(certificate.cells) != expected):
                row["eligible"] = False
                reason = "incomplete_direct_separation"
            else:
                row["precontext_clamps_held_through_endpoint"] = True
                row["direct_separation"] = [
                    {"lag": l, "clamps": dict(zip(others, a)), **asdict(cs),
                     "tv_lower": max(0.0, abs(cs.paired_difference) - delta_radius),
                     "tv_upper": min(1.0, abs(cs.paired_difference) + delta_radius),
                     "detectability_gap_lower": abs(cs.paired_difference) - delta_radius - config.effect_threshold,
                     "modal_gap_lower": abs(cs.p1 - 0.5) - mode_radius}
                    for (l, a), cs in sorted(certificate.cells.items())
                ]
                supported = [key for key, cs in certificate.cells.items()
                             if cs.n > 0 and abs(cs.paired_difference) - delta_radius > config.effect_threshold]
                if any(cs.n <= 0 for cs in certificate.cells.values()):
                    row["eligible"] = False
                    reason = "ineligible_blocked_cell"
                elif not supported:
                    reason = "direct_effect_not_supported_under_clamping" if natural_response else "no_detectable_response"
                    if any(abs(cs.paired_difference) + delta_radius > config.effect_threshold
                           for cs in certificate.cells.values()):
                        reason = "unresolved_direct_separation"
                else:
                    # Blocking certifies directness only. It never chooses the
                    # query's onset or value, both defined by the natural law.
                    row["directness_certified"] = True
                    row["directness_support"] = [{"lag": l, "clamps": dict(zip(others, a))}
                                                 for l, a in sorted(supported)]
                    reason = "no_detectable_response"
                    for lag, s in zip(evidence.lags, cells):
                        if abs(s.paired_difference) - delta_radius > config.effect_threshold:
                            row["selected_delay"] = lag
                            if abs(s.p1 - 0.5) <= mode_radius:
                                reason = "unresolved_natural_interventional_mode"
                            else:
                                m = Mechanism(candidate.action, candidate.context, candidate.target,
                                              int(s.p1 > 0.5), lag)
                                mechanisms.add(m)
                                row.update(status="written", mechanism=asdict(m))
                                reason = "natural_modal_first_detectable_after_direct_separation"
                            break
                        if abs(s.paired_difference) + delta_radius > config.effect_threshold:
                            reason = "unresolved_earlier_natural_lag"
                            break
        row["reason"] = reason
        decisions.append(row)
    return WriteResult(mechanisms, decisions)


def contrast_peak_emulation(evidence: Evidence, candidates: Iterable[Candidate],
                            config: GateConfig) -> WriteResult:
    """Emulate the criticized population decision, NOT any historical code version."""
    mechanisms, decisions = set(), []
    for c in candidates:
        lag, s = max(((l, evidence.natural[c, l]) for l in evidence.lags),
                     key=lambda item: (abs(item[1].paired_difference), -item[0]))
        reason = "no_threshold_response"
        if abs(s.paired_difference) >= config.effect_threshold:
            reason = "formula_filtered" if s.formula_agreement >= config.formula_agreement_threshold else "contrast_peak_written"
            if reason == "contrast_peak_written":
                mechanisms.add(Mechanism(c.action, c.context, c.target,
                                         int(s.paired_difference > 0), lag))
        decisions.append({"candidate": asdict(c), "status": "written" if reason == "contrast_peak_written" else "abstained",
                          "reason": reason, "selected_delay": lag, "statistic": asdict(s)})
    return WriteResult(mechanisms, decisions)


def _pr(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": p, "recall": r,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def tuple_scores(predicted: Iterable[Mechanism], truth: Iterable[Mechanism],
                 universe: Iterable[Candidate], generated: Iterable[Candidate]) -> dict:
    predicted, truth, universe, generated = set(predicted), set(truth), set(universe), set(generated)
    if any(m.candidate not in universe for m in truth | predicted):
        raise ValueError("scoring universe must include all true and predicted candidates")
    if not generated <= universe or any(m.candidate not in generated for m in predicted):
        raise ValueError("writes must come from generated candidates")
    tk, pk = {m.candidate for m in truth}, {m.candidate for m in predicted}
    exact_queries = sum({m for m in truth if m.candidate == c}
                        == {m for m in predicted if m.candidate == c} for c in universe)
    coordinates = {}
    for field in ("context", "target", "value", "delay"):
        t = {(m.action, getattr(m, field)) for m in truth}
        p = {(m.action, getattr(m, field)) for m in predicted}
        coordinates[field] = {**_pr(len(t & p), len(p - t), len(t - p)),
                              "projected_set_exact": p == t}
    return {
        "full_tuple": _pr(len(predicted & truth), len(predicted - truth), len(truth - predicted)),
        "all_query_accuracy": exact_queries / len(universe) if universe else None,
        "query_set_risk": (len(universe) - exact_queries) / len(universe) if universe else None,
        "all_query_correct": exact_queries, "all_query_count": len(universe),
        "positive_query_accuracy": len(predicted & truth) / len(truth) if truth else None,
        "memory_exact": predicted == truth,
        "coordinate_projections": coordinates,
        "coverage": {"true_candidates": len(tk), "universe_candidates": len(universe),
                     "generated_candidates": len(generated), "written_candidates": len(pk),
                     "true_candidates_generated": len(tk & generated),
                     "true_candidates_answered": len(tk & pk),
                     "generation_coverage": len(tk & generated) / len(tk) if tk else None,
                     "answer_coverage": len(tk & pk) / len(tk) if tk else None,
                     "candidate_omission_errors": len(tk - generated),
                     "true_candidate_abstentions": len((tk & generated) - pk),
                     "wrong_tuple_writes_on_true_candidates": sum(m.candidate in tk for m in predicted - truth),
                     "spurious_candidate_writes": len(pk - tk)},
    }


def coverage_counterexample(reviewer: str) -> dict:
    if reviewer == "R1":
        shifted = (frozenset((1, 2)), frozenset((1, 3)), frozenset((2, 4)))
    elif reviewer == "R2":
        shifted = (frozenset((1, 2)), frozenset((3, 4)))
    else:
        raise ValueError(reviewer)
    budget = 3
    lists = [frozenset(c) for n in range(budget + 1) for c in combinations(range(1, 5), n)]
    best = max(Fraction(sum(b <= l for b in shifted), len(shifted)) for l in lists)
    paths = []
    for b in shifted:
        tested = [1]
        observations = [1 in b]
        if reviewer == "R1":
            if 1 not in b:
                tested.extend((2, 4))
            else:
                tested.append(2)
                if 2 not in b:
                    tested.append(3)
        elif 1 in b:
            tested.append(2)
        else:
            tested.extend((3, 4))
        observations.extend(j in b for j in tested[1:])
        paths.append({"shifted": sorted(b), "tested": tested,
                      "retest_outcomes": observations, "covered": b <= set(tested)})
    adaptive = Fraction(sum(p["covered"] for p in paths), len(paths))
    return {"reviewer": reviewer, "static_optimum": str(best),
            "adaptive_coverage": str(adaptive), "budget": budget, "paths": paths,
            "best_static_lists": [sorted(l) for l in lists
                                  if Fraction(sum(b <= l for b in shifted), len(shifted)) == best]}


def exact_population_examples() -> dict:
    f = Fraction
    descendant = FiniteSCM(0, "noisy_descendant", varied=False)
    c = descendant.active_context
    one, zero = descendant.population(1, c, 1), descendant.population(0, c, 1)
    r, y = descendant.descendant, descendant.root
    return {
        "R1_modal": {"p1_y0": str(f(3, 5)), "p0_y0": str(f(9, 10)),
                     "delta_y0": str(f(3, 5) - f(9, 10)),
                     "delta_y1": str(f(2, 5) - f(1, 10)),
                     "modal_value": 0, "positive_contrast_value": 1},
        "R2_delay": {"p1": ["7/10", "19/20"], "p0": ["1/10", "1/10"],
                     "contrasts": [str(f(7, 10) - f(1, 10)), str(f(19, 20) - f(1, 10))],
                     "structural_first_delay": 1, "maximum_contrast_delay": 2},
        "R2_descendant": {"direct_contrast": str(one[y] - zero[y]), "descendant_contrast": str(one[r] - zero[r]),
                          "best_copy_accuracy": str(one["copy_agreement"]), "target_threshold": "1/5",
                          "formula_threshold": "9/10", "descendant_passes_contrast": True,
                          "descendant_filtered_by_formula": False,
                          "descendant_contrast_under_do_y0": str(descendant.population(1, c, 1, {y: 0})[r] - descendant.population(0, c, 1, {y: 0})[r]),
                          "descendant_contrast_under_do_y1": str(descendant.population(1, c, 1, {y: 1})[r] - descendant.population(0, c, 1, {y: 1})[r]),
                          "direct_contrast_under_do_r0_or_r1": str(descendant.population(1, c, 1, {r: 0})[y] - descendant.population(0, c, 1, {r: 0})[y])},
        "symmetric_copy": {"r_equals_y": True, "copy_y_to_r_accuracy": "1",
                           "copy_r_to_y_accuracy": "1",
                           "interpretation": "Formula agreement alone does not orient the edge."},
        "static_adaptive": [coverage_counterexample(r) for r in ("R1", "R2")],
    }


def run_historical(family_seed: int, seed: int, protocol: dict) -> dict:
    world = make_procedural_complex_hidden_world(
        family_seed, protocol["historical_mechanisms"], protocol["historical_visible_variables"],
        protocol["historical_observation_noise"], protocol["historical_readouts"])
    observation = world.reset(seed)
    # Original descriptions embed visible_conditions from the hidden SCM.
    # Strip all descriptions, retaining only the public action identifiers.
    actions = [ActionSpec(a.name, "Available intervention.") for a in world.actions()]
    agent = ContextSearchControlExperimentPlannerAgent()
    replay = CausalCoreAgent()
    agent.reset(actions)
    replay.reset(actions)
    history = []
    for step in range(1, protocol["historical_steps"] + 1):
        decision = agent.choose_action(observation, tuple(history))
        before = world.observe()
        after = world.step(decision.action)
        transition = Transition(step, decision.action, before, after, decision.prediction, decision.hypothesis)
        agent.observe_transition(transition)
        replay.observe_transition(transition)
        history.append(transition)
        observation = after
    # Only the evaluator accesses source labels, after both learners are frozen.
    truth = world.true_edges()
    readouts = world.readout_variables()
    methods = {}
    for method in (agent, replay):
        edges = method.discovered_edges()
        score = asdict(score_edges(edges, truth, tuple(history)))
        hints = [{"action": a, "target": t, "hints": h}
                 for (a, t), h in sorted(method.condition_hints().items())]
        methods[method.name] = {
            "target_edge": score, "readout_false_positives": sum(t in readouts for a, t in edges - truth),
            "discovered_edges": [list(e) for e in sorted(edges)], "condition_hints": hints,
            "full_tuple": None, "full_tuple_accuracy": None, "tuple_coverage": None,
            "coordinate_status": {"target": "scored as action-target edge projection",
                                  "context": "prose hints; no complete typed context answer",
                                  "value": "some prose sets hints; no complete typed outcome answer",
                                  "delay": "absent from this agent's memory/output schema"},
            "schema_gap": "discovered_edges() returns (action,target); condition_hints() returns list[str]. "
                          "No complete typed (action,context,target,value,delay) decoder. Delay=0 and "
                          "unconditional contexts cannot be filled in by the evaluator.",
            "intervention_calls_attributable_to_method": len(history) if method is agent else 0,
            "matched_training_records": len(history),
        }
    return {"family_seed": family_seed, "seed": seed, "world": world.name,
            "protocol": "context-search policy; base CausalCore replay receives identical transitions",
            "actions": [asdict(a) for a in actions],
            "trajectory": [asdict(t) | {"prediction": sorted(t.prediction)} for t in history],
            "true_edges_evaluator_only": [list(e) for e in sorted(truth)],
            "methods": methods,
            "ledger": {"total_step_intervention_calls": len(history), "reset_calls": 1,
                       "observation_calls": 1 + 2 * len(history),
                       "truth_label_calls_after_freeze": 2,
                       "extra_diagnostic_or_test_interventions": 0}}


def _bootstrap(values: list[float], seed: int, replicates: int) -> dict:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, len(data), (replicates, len(data)))].mean(axis=1)
    return {"mean": float(data.mean()), "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "independent_families": len(data), "interval": "family-cluster percentile bootstrap; descriptive"}


def aggregate_controlled(runs: list[dict], protocol: dict) -> dict:
    output = {}
    for name in runs[0]["methods"]:
        rows = [r["methods"][name] for r in runs]
        counts = {k: sum(r["metrics"]["full_tuple"][k] for r in rows)
                  for k in ("true_positive", "false_positive", "false_negative")}
        by_family = {}
        for family in sorted({r["family_seed"] for r in runs}):
            scores = [r["methods"][name]["metrics"] for r in runs if r["family_seed"] == family]
            by_family[family] = float(np.mean([s["positive_query_accuracy"] for s in scores]))
        output[name] = {
            "full_tuple_micro": _pr(counts["true_positive"], counts["false_positive"], counts["false_negative"]),
            "positive_query_accuracy": _bootstrap(list(by_family.values()), protocol["bootstrap_seed"], protocol["bootstrap_replicates"]),
            "all_query_accuracy": sum(r["metrics"]["all_query_correct"] for r in rows) / sum(r["metrics"]["all_query_count"] for r in rows),
            "query_set_risk": sum(r["metrics"]["all_query_count"] - r["metrics"]["all_query_correct"] for r in rows) / sum(r["metrics"]["all_query_count"] for r in rows),
            "exact_memories": sum(r["metrics"]["memory_exact"] for r in rows),
            "total_memories": len(rows),
            "coverage_counts": {k: sum(r["metrics"]["coverage"][k] for r in rows)
                                for k in ("true_candidates", "generated_candidates", "written_candidates",
                                          "true_candidates_generated", "true_candidates_answered", "candidate_omission_errors",
                                          "true_candidate_abstentions", "wrong_tuple_writes_on_true_candidates", "spurious_candidate_writes")},
            "abstention_reasons": dict(Counter(d["reason"] for r in rows for d in r["decisions"] if d["status"] == "abstained")),
            "coordinate_projections_micro": {},
        }
        for coordinate in ("context", "target", "value", "delay"):
            c = [r["metrics"]["coordinate_projections"][coordinate] for r in rows]
            totals = [sum(x[k] for x in c) for k in ("true_positive", "false_positive", "false_negative")]
            output[name]["coordinate_projections_micro"][coordinate] = {
                **_pr(*totals), "projected_set_exact_rate": sum(x["projected_set_exact"] for x in c) / len(c)}
    return output


def controlled_strata(runs: list[dict], protocol: dict) -> dict:
    groups = {"all_conditions": runs,
              "in_class": [r for r in runs if r["scenario"] in ("modal", "late_peak", "noisy_descendant")],
              "weak_early": [r for r in runs if r["scenario"] == "weak_early"]}
    return {name: {"runs": len(rows), "families": len({r['family_seed'] for r in rows}),
                   "methods": aggregate_controlled(rows, protocol)}
            for name, rows in groups.items() if rows}


def aggregate_historical(runs: list[dict], protocol: dict) -> dict:
    output = {}
    for name in runs[0]["methods"]:
        family_scores = [float(np.mean([r["methods"][name]["target_edge"]["f1"] for r in runs
                                       if r["family_seed"] == f])) for f in protocol["family_seeds"]]
        counts = [sum(r["methods"][name]["target_edge"][k] for r in runs)
                  for k in ("true_positive", "false_positive", "false_negative")]
        output[name] = {"target_edge_micro": _pr(*counts),
                        "target_edge_family_mean_f1": _bootstrap(family_scores, protocol["bootstrap_seed"], protocol["bootstrap_replicates"]),
                        "full_tuple": None, "full_tuple_accuracy": None, "tuple_coverage": None,
                        "readout_false_positives": sum(r["methods"][name]["readout_false_positives"] for r in runs)}
    return output


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in sorted(value)]
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_provenance(root: Path) -> dict:
    return {
        "source_sha256": {p: _digest(root / p) for p in REQUIRED_SOURCES},
        "optional_review_sha256": {p: _digest(root / p) for p in OPTIONAL_REVIEWS if (root / p).is_file()},
    }


def write_findings(result: dict, path: Path) -> None:
    p, a = result["protocol"], result["controlled_aggregate"]
    lines = ["# Finite Revision Experiments", "", "## Protocol",
             "- Type: finite SCM simulation and historical-agent re-evaluation",
             "- Status: COMPLETED; CPU execution; descriptive simulation results",
             f"- Protocol: `{p['protocol_id']}`; SHA-256 `{result['protocol_sha256']}`",
             f"- Protocol file: `{result['protocol_path']}`",
             "- Scientific source hashes are required; confidential review hashes are optional provenance only.",
             "- Source labels: evaluator only; historical action descriptions redacted", "",
             "## Fixed Design",
             f"{len(p['family_seeds'])} independently generated families, {len(p['replicate_seeds'])} seed replicates per family. "
             f"Each controlled cell uses {p['pairs_per_cell']} paired trials; effect threshold {p['effect_threshold']}, "
             f"familywise alpha {p['familywise_alpha']}. Thresholds were fixed before outcome inspection.", "",
             "Controlled SCMs have two anonymous endogenous nodes, two context cells and two lags. "
             "The root is action-responsive in one context; the second node is its XOR-noise descendant. "
             "Each paired trial shares exogenous draws between do(A=1) and do(A=0). "
             "Non-target nodes are fixed by pre-context interventions held through each endpoint. Complete "
             "blocking support and at least one blocked TV lower bound above the fixed threshold certify "
             "directness in this fully observed surgical-intervention class. AFTER that certificate, the writer "
             "scans NATURAL Bernoulli TV contrasts in lag order and estimates the value from NATURAL p1 "
             "at the selected natural lag. Blocked onset and blocked modes do not define the query answer; "
             "blocked modal invariance is not required. An unresolved earlier natural threshold decision "
             "or unresolved natural mode causes abstention. Eligibility, both natural/blocked TV intervals, "
             "detectability gaps and natural modal gaps are saved per candidate.", "",
             "## Joint Results",
             "Positive-query accuracy is exact tuple recall, including abstentions as failures. "
             "All-query accuracy additionally scores empty answers on inactive/descendant candidates. "
             "Memory exactness requires set equality. No joint metric is a product of coordinate metrics.", "",
             "| Method | Tuple precision | Tuple recall / positive accuracy | Query-set risk | Exact memories |",
             "|---|---:|---:|---:|---:|"]
    def fmt(x):
        return "undefined (no writes)" if x is None else f"{x:.4f}"
    for name, summary in a.items():
        m = summary["full_tuple_micro"]
        lines.append(f"| {name} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(summary['query_set_risk'])} | "
                     f"{summary['exact_memories']}/{summary['total_memories']} |")
    lines += ["", "## In-Class and Detectability-Violation Results",
              "The scenario split is fixed by generator condition, not by outcomes. In-class includes "
              "modal, late_peak and noisy_descendant; weak_early violates the minimum early-effect "
              "detectability condition. Both the complete mixture above and every stratum below are reported.", "",
              "| Condition | Runs | Tuple precision | Tuple recall | Answer coverage | Query-set risk | Wrong true-candidate tuples |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, stratum in result["controlled_strata"].items():
        summary = stratum["methods"]["paired_modal_first_direct"]
        m, c = summary["full_tuple_micro"], summary["coverage_counts"]
        lines.append(f"| {name} | {stratum['runs']} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                     f"{c['true_candidates_answered']}/{c['true_candidates']} | {fmt(summary['query_set_risk'])} | "
                     f"{c['wrong_tuple_writes_on_true_candidates']} |")
    lines += ["", "## Controlled Sample-Budget Curve",
              "Budgets 64/256/1024/4096 were fixed for this follow-up before inspecting curve outcomes. "
              "All budgets use the same 12 families and three seeds, with unchanged gates. Datasets are "
              "separately sampled rather than nested; the 4096-pair primary data are reused. "
              "Historical trajectories are not rerun. Per-candidate decisions and ledgers are in `finite_budget_curve.json`.", "",
              "| Pairs per cell | Condition | Tuple precision | Tuple recall | True-candidate answer coverage | Query-set risk | Abstentions | Wrong true-candidate tuples |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for budget in result["sample_budget_curve"]:
        for condition, group in budget["strata"].items():
            summary = group["methods"]["paired_modal_first_direct"]
            m, c = summary["full_tuple_micro"], summary["coverage_counts"]
            lines.append(f"| {budget['pairs_per_cell']} | {condition} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                         f"{c['true_candidates_answered']}/{c['true_candidates']} | {fmt(summary['query_set_risk'])} | "
                         f"{c['true_candidate_abstentions']} | {c['wrong_tuple_writes_on_true_candidates']} |")
    lines += ["", "Coordinate context/target/value/delay precision, recall and projected-set accuracy, "
              "family-cluster descriptive bootstrap intervals, and every decision are in `finite_results.json`. "
              "Coordinate projections can agree while the full tuple fails. Every action/context/target "
              "candidate is a query; inactive contexts and descendants have the null answer, so spurious "
              "writes reduce query-set accuracy (one minus uniform query-set risk).", "", "## Coverage and Unfavorable Results"]
    for name, summary in a.items():
        c = summary["coverage_counts"]
        lines.append(f"- {name}: true candidates answered {c['true_candidates_answered']}/{c['true_candidates']}; "
                     f"generation omissions {c['candidate_omission_errors']}; true-candidate abstentions {c['true_candidate_abstentions']}; "
                     f"wrong tuples on true candidates {c['wrong_tuple_writes_on_true_candidates']}; "
                     f"spurious candidate writes {c['spurious_candidate_writes']}.")
    lines += ["", "The weak_early condition has a structural response of 0.04 at lag 1, below the fixed 0.2 gate threshold, "
              "followed by a large response at lag 2. The new estimator can certify the first detectable lag while "
              "returning the wrong structural delay and modal value. These are scored as full-tuple errors. "
              "Recovering structural first response requires an explicit lower bound on every nonzero early effect "
              "and enough samples to resolve it. Natural contrast without a clamping certificate produces abstentions. "
              "The candidate-omission stress drops the lexicographically first public node before writing, "
              "without consulting truth. Its omitted true mechanisms stay in the query universe and recall denominator. "
              "Generation omissions and evidence abstentions are counted separately.", "",
              "The contrast/peak/formula row is an explicit emulation of the criticized rule, not a rerun or "
              "renaming of the historical Causal Core implementation. The corrected paired writer is new code "
              "and has a stronger intervention interface. Its performance is not a matched-cost comparison "
              "with the historical agent.", "", "## Historical Matched Procedural Runs"]
    for name, summary in result["historical_aggregate"].items():
        m = summary["target_edge_micro"]
        lines.append(f"- {name}: target-edge precision {fmt(m['precision'])}, recall {fmt(m['recall'])}, F1 {fmt(m['f1'])}; "
                     f"readout false positives {summary['readout_false_positives']}; joint tuple accuracy unavailable.")
    if result["historical_provenance"]["reused"]:
        lines += ["", "The 36 historical runs are reused byte-for-byte at the record level from the prior completed "
                  "dataset. Historical configuration and six historical module hashes were checked before reuse. "
                  "Original runtime/source hashes, input-file hash and the reused-records hash are retained in "
                  "`historical_provenance`; current writer/test hashes describe this corrected controlled rerun. "
                  "This execution uses zero new historical intervention calls."]
    lines += ["", "The existing context-search agent selects the trajectory; base CausalCore replays identical records. "
              "These are fresh source-world discovery runs, not a renamed transfer or historical table reproduction. "
              "World identifiers and public action names are unchanged. Original action descriptions contain hidden "
              "visible-setup conditions, so all descriptions are replaced with a generic string before either learner sees them. "
              "No true_edges, hidden_context_edges, readout labels, or procedural_spec are passed to a learner.", "",
              "The historical schema is `set[(action,target)]` plus `dict[(action,target),list[str]]` prose hints. "
              "The context-search and base agents lack a complete typed context/value answer and any delay field. "
              "The evaluator therefore records null full-tuple accuracy, precision/recall and coverage. "
              "It does not insert delay=0, unconditional context, or derive joint accuracy from marginals. "
              "The temporal agent separately has private delay/value maps, but is not the context-search implementation evaluated here.", "",
              "## Exact Reviewer Counterexamples",
              "- R1: modal value 0 has contrast -3/10; value 1 has contrast +3/10.",
              "- R2: contrasts 3/5 and 17/20 select peak lag 2 although the first response is lag 1.",
              "- R2 noisy descendant: contrast 3/5 passes 1/5; copy accuracy 4/5 fails the 9/10 readout filter. "
              "Clamping its parent removes the action contrast.",
              "- R1 coverage: optimal static coverage 2/3; retest-informed adaptive coverage 1 within three tests.",
              "- R2 coverage: optimal static coverage 1/2; retest-informed adaptive coverage 1 within three tests.",
              "All static lists are enumerated, all adaptive paths and outcomes are saved. "
              "These finite calculations are executable counterexamples, not a new general proof.", "", "## Cost and Scope",
              f"Physical controlled joint-do calls: {result['costs']['controlled_joint_do_calls']:,}; "
              f"atomic assignments: {result['costs']['controlled_atomic_assignments']:,}. "
              f"Historical step interventions: {result['costs']['historical_step_interventions']:,}; "
              f"historical resets: {result['costs']['historical_resets']}.", "",
              f"Additional smaller-budget joint-do calls: {result['costs']['additional_budget_joint_do_calls']:,}. "
              f"Current execution total controlled joint-do calls: {result['costs']['current_execution_controlled_joint_do_calls']:,}; "
              f"atomic assignments: {result['costs']['current_execution_controlled_atomic_assignments']:,}. "
              "The primary 4096 budget is counted once, not twice. Historical counts above are retained dataset "
              "costs, not newly incurred costs when historical_provenance.reused is true.", "",
              "Every vectorized endpoint counts one joint intervention per sampled arm, including reference arms, "
              "inactive contexts and clamps. Context setup is included in the joint do; no uncharged rollout, "
              "reset or follow-up action is used in the stateless controlled SCM. Atomic-assignment counts "
              "also count A, C and each clamp separately. Natural samples are shared by the emulation and "
              "no-direct ablation; separation trials are charged only to the direct writer. Replay has zero new calls. "
              "Bootstrap and exact population calculations require no environment interventions.", "",
              "Directness interpretation is restricted to this fully observed acyclic two-node surgical-intervention "
              "class, with independent exogenous noise and complete simultaneous clamps. There is no general "
              "directness guarantee for latent mediators, incomplete clamp support, cancellation or observational evidence alone.", "",
              "## Reproduction",
              f"Repository root: `{ROOT.as_posix()}`. The canonical protocol file above exists under "
              "`jmlr_revision_20261112/experiments`; commands below use its absolute path.", "",
              "`D:/software/miniconda3/envs/zocosmo/python.exe -m causal_sandbox.finite_interventions "
              f"--protocol {Path(result['protocol_path']).as_posix()}`", "",
              "Controlled-only rerun, reusing the saved historical records:", "",
              "`D:/software/miniconda3/envs/zocosmo/python.exe -m causal_sandbox.finite_interventions "
              f"--protocol {Path(result['protocol_path']).as_posix()} "
              f"--reuse-historical {(path.parent / 'finite_results.json').resolve().as_posix()}`", "",
              "`D:/software/miniconda3/envs/zocosmo/python.exe -m pytest -q -p no:cacheprovider tests/test_finite_interventions.py`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def controlled_run(scm: FiniteSCM, seed: int, protocol: dict, n: int) -> dict:
    evidence = collect_evidence(scm, seed, n)
    candidates = scm.candidates
    gate = GateConfig(protocol["effect_threshold"], protocol["familywise_alpha"], protocol["formula_agreement_threshold"])
    methods = {
        "contrast_peak_formula_emulation": contrast_peak_emulation(evidence, candidates, gate),
        "paired_modal_first_without_direct": paired_writer(evidence, candidates, gate, require_direct=False),
        "paired_modal_first_direct": paired_writer(evidence, candidates, gate),
    }
    omitted = tuple(c for c in candidates if c.target != evidence.variables[0])
    methods["paired_direct_candidate_omission"] = paired_writer(evidence, omitted, gate)
    # Truth is accessed only after candidate generation and writer decisions.
    truth = scm.truth()
    entries = {}
    for name, writer in methods.items():
        generated = omitted if name == "paired_direct_candidate_omission" else candidates
        entries[name] = {"predicted": [asdict(m) for m in sorted(writer.mechanisms)],
                         "decisions": writer.decisions,
                         "generated_candidates": [asdict(c) for c in generated],
                         "metrics": tuple_scores(writer.mechanisms, truth, candidates, generated),
                         "required_joint_do_calls": evidence.ledger["total_joint_do_calls"] if name in ("paired_modal_first_direct", "paired_direct_candidate_omission")
                         else evidence.ledger["natural_joint_do_calls"]}
    return {"family_seed": scm.family_seed, "seed": seed, "scenario": scm.scenario,
            "pairs_per_cell": n,
            "structural_parameters_evaluator_only": {"root": scm.root, "descendant": scm.descendant,
                "q1": list(map(str, scm.q1)), "q0": list(map(str, scm.q0)), "noise": str(scm.noise)},
            "truth_evaluator_only": [asdict(m) for m in sorted(truth)],
            "candidate_universe": [asdict(c) for c in candidates],
            "ledger": evidence.ledger, "methods": entries}


def reuse_historical(path: Path, protocol: dict) -> tuple[list[dict], dict]:
    previous = json.loads(path.read_text(encoding="utf-8"))
    keys = ("family_seeds", "replicate_seeds", "historical_steps", "historical_mechanisms",
            "historical_visible_variables", "historical_readouts", "historical_observation_noise")
    if any(previous["protocol"][k] != protocol[k] for k in keys):
        raise ValueError("historical cache protocol differs")
    for module in ("core", "evaluation", "causal_core", "agents", "worlds", "observation"):
        source = f"causal_sandbox/{module}.py"
        if _digest(ROOT / source) != previous["source_sha256"][source]:
            raise ValueError(f"historical cache source changed: {source}")
    expected = set(product(protocol["family_seeds"], protocol["replicate_seeds"]))
    rows = previous["historical_runs"]
    if len(rows) != len(expected) or {(r["family_seed"], r["seed"]) for r in rows} != expected:
        raise ValueError("historical cache does not contain the fixed family/seed design")
    provenance = {"reused": True, "input_path": str(path.resolve()), "input_sha256": _digest(path),
                  "original_protocol_sha256": previous["protocol_sha256"],
                  "original_runtime": previous["runtime"], "original_source_sha256": previous["source_sha256"],
                  "historical_records_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                  "new_historical_intervention_calls": 0,
                  "previous_recorded_controlled_joint_do_calls": previous["costs"]["controlled_joint_do_calls"]}
    return rows, provenance


def sample_budget_curve(primary: list[dict], protocol: dict) -> dict:
    entries = []
    for n in protocol.get("sample_budgets", [protocol["pairs_per_cell"]]):
        if n == protocol["pairs_per_cell"]:
            rows = primary
        else:
            rows = []
            for i, family in enumerate(protocol["family_seeds"]):
                scm = FiniteSCM(family, protocol["scenarios"][i % len(protocol["scenarios"])] )
                for seed in protocol["replicate_seeds"]:
                    rows.append(controlled_run(scm, seed, protocol, n))
        thin = [dict(r, methods={"paired_modal_first_direct": r["methods"]["paired_modal_first_direct"]}) for r in rows]
        entries.append({"pairs_per_cell": n, "reuses_primary_samples": n == protocol["pairs_per_cell"],
                        "strata": controlled_strata(thin, protocol), "runs": thin,
                        "joint_do_calls": sum(r["ledger"]["total_joint_do_calls"] for r in rows),
                        "atomic_assignments": sum(r["ledger"]["total_atomic_assignments"] for r in rows)})
        print(f"budget={n} completed {len(rows)} controlled runs", flush=True)
    return {"protocol": protocol, "budgets": entries,
            "sampling": "Same fixed families and seeds at every budget; separately sampled non-nested datasets. "
                        "The primary 4096-pair dataset is reused, not sampled twice. No historical reruns or gate tuning.",
            "alpha_scope": "0.05 simultaneous natural/blocked gate bounds within each family-seed-budget run"}


def run(protocol_path: Path, output: Path, historical_cache: Path | None = None) -> dict:
    start = time.perf_counter()
    provenance = source_provenance(ROOT)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["device"] != "cpu":
        raise ValueError("this runner is CPU only")
    if len(set(protocol["family_seeds"])) != len(protocol["family_seeds"]):
        raise ValueError("independent family seeds must be distinct")
    controlled, historical = [], []
    historical_provenance = {"reused": False}
    if historical_cache is not None:
        historical, historical_provenance = reuse_historical(historical_cache, protocol)
    for i, family in enumerate(protocol["family_seeds"]):
        scenario = protocol["scenarios"][i % len(protocol["scenarios"])]
        scm = FiniteSCM(family, scenario)
        for seed in protocol["replicate_seeds"]:
            print(f"controlled family={family} seed={seed} scenario={scenario}", flush=True)
            controlled.append(controlled_run(scm, seed, protocol, protocol["pairs_per_cell"]))
            if historical_cache is None:
                print(f"historical family={family} seed={seed} steps={protocol['historical_steps']}", flush=True)
                historical.append(run_historical(family, seed, protocol))
    curve = sample_budget_curve(controlled, protocol)
    result = {
        "status": "completed", "protocol": protocol, "protocol_sha256": _digest(protocol_path),
        "protocol_path": str(protocol_path.resolve()),
        "writer_semantics": {"onset": "first detectable natural Bernoulli TV contrast",
                             "value": "modal natural p1 at the selected natural lag",
                             "directness": "separate complete blocking-support gate; at least one detectable blocked contrast",
                             "blocked_modal_invariance_required": False},
        "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                    "device": "cpu", "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.perf_counter() - start},
        **provenance,
        "exact_population_examples": exact_population_examples(),
        "controlled_runs": controlled, "historical_runs": historical,
        "historical_provenance": historical_provenance,
        "controlled_aggregate": aggregate_controlled(controlled, protocol),
        "controlled_strata": controlled_strata(controlled, protocol),
        "sample_budget_curve": [{k: v for k, v in e.items() if k != "runs"} for e in curve["budgets"]],
        "historical_aggregate": aggregate_historical(historical, protocol),
        "costs": {"controlled_joint_do_calls": sum(r["ledger"]["total_joint_do_calls"] for r in controlled),
                  "controlled_atomic_assignments": sum(r["ledger"]["total_atomic_assignments"] for r in controlled),
                  "historical_step_interventions": sum(r["ledger"]["total_step_intervention_calls"] for r in historical),
                  "historical_resets": len(historical),
                  "current_execution_historical_step_interventions": 0 if historical_cache else sum(len(r["trajectory"]) for r in historical),
                  "additional_budget_joint_do_calls": sum(e["joint_do_calls"] for e in curve["budgets"] if not e["reuses_primary_samples"]),
                  "current_execution_controlled_joint_do_calls": sum(e["joint_do_calls"] for e in curve["budgets"]),
                  "current_execution_controlled_atomic_assignments": sum(e["atomic_assignments"] for e in curve["budgets"])},
        "interpretation": "Exact tuple set comparisons; historical tuples unavailable; no inferred joint metrics or universal structural guarantee.",
    }
    output.mkdir(parents=True, exist_ok=True)
    curve["protocol_path"] = result["protocol_path"]
    curve["protocol_sha256"] = result["protocol_sha256"]
    curve["source_sha256"] = result["source_sha256"]
    curve["optional_review_sha256"] = result["optional_review_sha256"]
    (output / "finite_budget_curve.json").write_text(json.dumps(_jsonable(curve), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "finite_results.json").write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_findings(result, output / "finite_findings.md")
    print(f"COMPLETED {output / 'finite_results.json'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "jmlr_revision_20261112/experiments/finite_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "jmlr_revision_20261112/results")
    parser.add_argument("--reuse-historical", type=Path, help="Reuse a completed finite_results.json; verify historical source/config hashes.")
    args = parser.parse_args()
    run(args.protocol, args.output, args.reuse_historical)


if __name__ == "__main__":
    main()
