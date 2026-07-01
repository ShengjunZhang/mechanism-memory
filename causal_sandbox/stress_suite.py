from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from math import exp, log
from pathlib import Path
from random import Random
from statistics import mean

import numpy as np

from .agents import ContextSearchControlExperimentPlannerAgent, make_agent
from .continuous import run_continuous_metric_poc
from .core import Edge, State, Transition
from .evaluation import EpisodeResult, run_episode, score_edges
from .worlds import make_procedural_complex_hidden_world


@dataclass(frozen=True)
class MechanismBaselineResult:
    name: str
    discovered_edges: set[Edge]
    hidden_labels: set[Edge]
    threshold: float | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)


def run_icml_stress_suite(
    families: int = 6,
    seeds: int = 2,
    steps: int = 160,
    mechanisms: int = 4,
    visible: int = 5,
    readouts: int = 6,
    noise: float = 0.02,
    thresholds: tuple[float, ...] = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30),
    budget_steps: tuple[int, ...] = (80, 120, 160, 240),
    continuous_noises: tuple[float, ...] = (0.02, 0.05, 0.10),
    neural_epochs: int = 24,
) -> dict[str, object]:
    """Run reviewer-facing stress checks for the mechanism-memory paper."""

    matched = _run_matched_mechanism_baselines(
        families=families,
        seeds=seeds,
        steps=steps,
        mechanisms=mechanisms,
        visible=visible,
        readouts=readouts,
        noise=noise,
        thresholds=thresholds,
        neural_epochs=neural_epochs,
    )
    budget = _run_candidate_budget_sensitivity(
        families=min(families, 6),
        seeds=seeds,
        steps_list=budget_steps,
        mechanisms=mechanisms,
        visible=visible,
        readouts=readouts,
        noise=noise,
    )
    continuous = _run_continuous_noise_sweep(
        seeds=max(8, families * seeds),
        noise_levels=continuous_noises,
    )
    return {
        "suite": "icml-mechanism-stress-suite",
        "families": families,
        "seeds": seeds,
        "steps": steps,
        "mechanisms": mechanisms,
        "visible": visible,
        "readouts": readouts,
        "noise": noise,
        "neural_epochs": neural_epochs,
        "matched_mechanism_baselines": matched,
        "candidate_budget_sensitivity": budget,
        "continuous_noise_sweep": continuous,
    }


def run_matched_mechanism_baseline_suite(
    families: int = 8,
    seeds: int = 3,
    steps: int = 200,
    mechanisms: int = 4,
    visible: int = 5,
    readouts: int = 6,
    noise: float = 0.02,
    thresholds: tuple[float, ...] = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30),
    neural_epochs: int = 24,
) -> dict[str, object]:
    """Run the matched mechanism baselines without auxiliary sweeps."""

    matched = _run_matched_mechanism_baselines(
        families=families,
        seeds=seeds,
        steps=steps,
        mechanisms=mechanisms,
        visible=visible,
        readouts=readouts,
        noise=noise,
        thresholds=thresholds,
        neural_epochs=neural_epochs,
    )
    return {
        "suite": "matched-mechanism-baselines",
        "families": families,
        "seeds": seeds,
        "steps": steps,
        "mechanisms": mechanisms,
        "visible": visible,
        "readouts": readouts,
        "noise": noise,
        "neural_epochs": neural_epochs,
        "matched_mechanism_baselines": matched,
    }


def format_icml_stress_suite(payload: dict[str, object]) -> str:
    matched = payload["matched_mechanism_baselines"]  # type: ignore[index]
    budget = payload["candidate_budget_sensitivity"]  # type: ignore[index]
    continuous = payload["continuous_noise_sweep"]  # type: ignore[index]

    lines = [
        "ICML mechanism stress suite",
        (
            f"Families={payload['families']}, seeds={payload['seeds']}, "
            f"steps={payload['steps']}, noise={payload['noise']}"
        ),
        "",
        "Matched mechanism baselines",
        (
            f"{'method':<34} {'f1':>5} {'fp':>6} {'readout-fp':>10} "
            f"{'hidden-r':>9} {'hidden-fp':>9} {'threshold':>9}"
        ),
        "-" * 88,
    ]
    for row in matched["summary"]:  # type: ignore[index]
        lines.append(
            f"{row['method']:<34} "
            f"{float(row['f1']):>5.2f} "
            f"{float(row['false_positive']):>6.2f} "
            f"{float(row['readout_false_positive']):>10.2f} "
            f"{float(row['hidden_context_recall']):>9.2f} "
            f"{float(row['hidden_context_false_positive']):>9.2f} "
            f"{str(row.get('threshold', '--')):>9}"
        )

    lines.extend(
        [
            "",
            "Candidate-budget sensitivity",
            (
                f"{'config':<20} {'steps':>6} {'f1':>5} {'fp':>6} "
                f"{'readout-fp':>10} {'hidden-r':>9} {'hidden-fp':>9}"
            ),
            "-" * 76,
        ]
    )
    for row in budget["summary"]:  # type: ignore[index]
        lines.append(
            f"{row['config']:<20} "
            f"{int(row['steps']):>6} "
            f"{float(row['f1']):>5.2f} "
            f"{float(row['false_positive']):>6.2f} "
            f"{float(row['readout_false_positive']):>10.2f} "
            f"{float(row['hidden_context_recall']):>9.2f} "
            f"{float(row['hidden_context_false_positive']):>9.2f}"
        )

    lines.extend(
        [
            "",
            "Continuous noise sweep",
            (
                f"{'noise':>6} {'random-f1':>10} {'linear-f1':>10} "
                f"{'metric-core-f1':>14} {'metric-readout-fp':>17}"
            ),
            "-" * 70,
        ]
    )
    for row in continuous["summary"]:  # type: ignore[index]
        lines.append(
            f"{float(row['noise']):>6.2f} "
            f"{float(row['random_correlation_f1']):>10.2f} "
            f"{float(row['global_linear_f1']):>10.2f} "
            f"{float(row['metric_causal_core_f1']):>14.2f} "
            f"{float(row['metric_causal_core_readout_fp']):>17.2f}"
        )
    return "\n".join(lines)


def format_matched_mechanism_baseline_suite(payload: dict[str, object]) -> str:
    matched = payload["matched_mechanism_baselines"]  # type: ignore[index]
    lines = [
        "Matched mechanism baselines",
        (
            f"Families={payload['families']}, seeds={payload['seeds']}, "
            f"steps={payload['steps']}, noise={payload['noise']}"
        ),
        "",
        (
            f"{'method':<34} {'prec':>6} {'rec':>6} {'f1':>6} "
            f"{'fp':>6} {'readout-fp':>10} {'hidden-r':>9} "
            f"{'hidden-fp':>9} {'threshold':>9}"
        ),
        "-" * 107,
    ]
    for row in matched["summary"]:  # type: ignore[index]
        lines.append(
            f"{row['method']:<34} "
            f"{float(row['precision']):>6.2f} "
            f"{float(row['recall']):>6.2f} "
            f"{float(row['f1']):>6.2f} "
            f"{float(row['false_positive']):>6.2f} "
            f"{float(row['readout_false_positive']):>10.2f} "
            f"{float(row['hidden_context_recall']):>9.2f} "
            f"{float(row['hidden_context_false_positive']):>9.2f} "
            f"{str(row.get('threshold', '--')):>9}"
        )
    return "\n".join(lines)


def save_icml_stress_suite(payload: dict[str, object], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_matched_mechanism_baselines(
    families: int,
    seeds: int,
    steps: int,
    mechanisms: int,
    visible: int,
    readouts: int,
    noise: float,
    thresholds: tuple[float, ...],
    neural_epochs: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    methods = (
        "causal-core-context-search",
        "conditional-discovery",
        "conditional-discovery-readout-oracle",
        "tuple-lift",
        "tuple-lift-readout-oracle",
        "latent-world-model",
        "latent-world-model-readout-oracle",
        "neural-transition",
        "neural-transition-readout-oracle",
    )
    runs: dict[str, list[dict[str, object]]] = {method: [] for method in methods}
    detailed: list[dict[str, object]] = []

    for family_seed in range(1, families + 1):
        spec_world = make_procedural_complex_hidden_world(
            family_seed=family_seed,
            mechanism_count=mechanisms,
            visible_count=visible,
            noise_probability=noise,
            readout_count=readouts,
        )
        readout_variables = spec_world.readout_variables()
        hidden_edges = spec_world.hidden_context_edges()

        for seed in range(1, seeds + 1):
            world = make_procedural_complex_hidden_world(
                family_seed=family_seed,
                mechanism_count=mechanisms,
                visible_count=visible,
                noise_probability=noise,
                readout_count=readouts,
            )
            core_result = run_episode(
                world,
                make_agent("causal-core-context-search-planner", seed=seed),
                steps=steps,
                seed=seed,
            )
            core_hidden = _hidden_context_hint_edges(core_result.condition_hints)
            core_row = _metric_row(
                "causal-core-context-search",
                core_result.discovered_edges,
                core_result.true_edges,
                readout_variables,
                hidden_edges,
                core_hidden,
                transitions=core_result.transitions,
            )
            runs["causal-core-context-search"].append(core_row)
            detailed.append(
                core_row
                | {
                    "family_seed": family_seed,
                    "seed": seed,
                    "shared_data": "context-search trajectory",
                }
            )

            lift_candidates = [
                _tuple_lift_writer(
                    core_result.transitions,
                    readout_variables,
                    threshold=threshold,
                    remove_readouts=False,
                )
                for threshold in thresholds
            ]
            lift_oracle_candidates = [
                _tuple_lift_writer(
                    core_result.transitions,
                    readout_variables,
                    threshold=threshold,
                    remove_readouts=True,
                )
                for threshold in thresholds
            ]
            discovery_candidates = _conditional_discovery_grid(
                core_result.transitions,
                readout_variables,
                thresholds=thresholds,
                remove_readouts=False,
            )
            discovery_oracle_candidates = _conditional_discovery_grid(
                core_result.transitions,
                readout_variables,
                thresholds=thresholds,
                remove_readouts=True,
            )
            neural_candidates = _neural_transition_probe_grid(
                core_result.transitions,
                readout_variables,
                thresholds=thresholds,
                remove_readouts=False,
                seed=seed + 10_000 * family_seed,
                epochs=neural_epochs,
            )
            neural_oracle_candidates = _neural_transition_probe_grid(
                core_result.transitions,
                readout_variables,
                thresholds=thresholds,
                remove_readouts=True,
                seed=seed + 20_000 * family_seed,
                epochs=neural_epochs,
            )
            world_model_candidates = _latent_world_model_probe_grid(
                core_result.transitions,
                readout_variables,
                thresholds=thresholds,
                remove_readouts=False,
                seed=seed + 30_000 * family_seed,
                epochs=neural_epochs,
            )
            world_model_oracle_candidates = _latent_world_model_probe_grid(
                core_result.transitions,
                readout_variables,
                thresholds=thresholds,
                remove_readouts=True,
                seed=seed + 40_000 * family_seed,
                epochs=neural_epochs,
            )

            for candidate_group in (
                discovery_candidates,
                discovery_oracle_candidates,
                lift_candidates,
                lift_oracle_candidates,
                world_model_candidates,
                world_model_oracle_candidates,
                neural_candidates,
                neural_oracle_candidates,
            ):
                for candidate in candidate_group:
                    row = _metric_row(
                        candidate.name,
                        candidate.discovered_edges,
                        core_result.true_edges,
                        readout_variables,
                        hidden_edges,
                        candidate.hidden_labels,
                        threshold=candidate.threshold,
                        transitions=core_result.transitions,
                        diagnostics=candidate.diagnostics,
                    )
                    threshold_rows.append(
                        row
                        | {
                            "family_seed": family_seed,
                            "seed": seed,
                        }
                    )

            for candidate in (
                _best_candidate(discovery_candidates, core_result.true_edges),
                _best_candidate(discovery_oracle_candidates, core_result.true_edges),
                _best_candidate(lift_candidates, core_result.true_edges),
                _best_candidate(lift_oracle_candidates, core_result.true_edges),
                _best_candidate(world_model_candidates, core_result.true_edges),
                _best_candidate(world_model_oracle_candidates, core_result.true_edges),
                _best_candidate(neural_candidates, core_result.true_edges),
                _best_candidate(neural_oracle_candidates, core_result.true_edges),
            ):
                row = _metric_row(
                    candidate.name,
                    candidate.discovered_edges,
                    core_result.true_edges,
                    readout_variables,
                    hidden_edges,
                    candidate.hidden_labels,
                    threshold=candidate.threshold,
                    transitions=core_result.transitions,
                    diagnostics=candidate.diagnostics,
                )
                runs[candidate.name].append(row)
                detailed.append(row | {"family_seed": family_seed, "seed": seed})

    for method in methods:
        rows.append(_summary_row(method, runs[method]))

    return {
        "protocol": "all writers use the same context-search trajectories",
        "summary": rows,
        "threshold_sweep": _threshold_summary(threshold_rows),
        "runs": detailed,
    }


def _tuple_lift_writer(
    transitions: tuple[Transition, ...],
    readout_variables: set[str],
    threshold: float,
    remove_readouts: bool,
) -> MechanismBaselineResult:
    action_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    edge_counts: Counter[Edge] = Counter()
    effect_values: dict[Edge, Counter[bool]] = {}
    all_targets = sorted(
        {
            target
            for transition in transitions
            for target in set(transition.before) | set(transition.after)
        }
    )
    for transition in transitions:
        action_counts[transition.action] += 1
        for target in transition.changed:
            target_counts[target] += 1
            edge = (transition.action, target)
            edge_counts[edge] += 1
            effect_values.setdefault(edge, Counter())[transition.after[target]] += 1

    total = len(transitions)
    discovered: set[Edge] = set()
    hidden_labels: set[Edge] = set()
    for action, action_count in action_counts.items():
        if action_count < 3:
            continue
        for target in all_targets:
            if remove_readouts and target in readout_variables:
                continue
            edge = (action, target)
            effect_count = edge_counts[edge]
            if effect_count == 0:
                continue
            action_rate = effect_count / action_count
            background_count = target_counts[target] - effect_count
            background_total = max(total - action_count, 1)
            background_rate = background_count / background_total
            if action_rate - background_rate < threshold:
                continue
            discovered.add(edge)
            failures = action_count - effect_count
            if effect_count >= 2 and failures >= 2 and 0.15 <= action_rate <= 0.85:
                values = effect_values.get(edge, Counter())
                if not values or values.most_common(1)[0][1] / effect_count >= 0.70:
                    hidden_labels.add(edge)

    suffix = "-readout-oracle" if remove_readouts else ""
    return MechanismBaselineResult(
        name=f"tuple-lift{suffix}",
        discovered_edges=discovered,
        hidden_labels=hidden_labels,
        threshold=threshold,
    )


def _conditional_discovery_grid(
    transitions: tuple[Transition, ...],
    readout_variables: set[str],
    thresholds: tuple[float, ...],
    remove_readouts: bool,
) -> list[MechanismBaselineResult]:
    if not transitions:
        return [
            MechanismBaselineResult(
                name="conditional-discovery",
                discovered_edges=set(),
                hidden_labels=set(),
                threshold=threshold,
            )
            for threshold in thresholds
        ]
    variables = sorted(
        {
            target
            for transition in transitions
            for target in set(transition.before) | set(transition.after)
        }
    )
    actions = sorted({transition.action for transition in transitions})
    context_variables = [
        variable for variable in variables if variable not in readout_variables
    ]
    scored_edges: list[tuple[Edge, float, float]] = []
    for action in actions:
        for target in variables:
            if remove_readouts and target in readout_variables:
                continue
            candidate_contexts = [()]
            candidate_contexts.extend(
                (variable,)
                for variable in context_variables
                if variable != target
            )
            score = max(
                _conditional_mutual_information(
                    transitions,
                    action=action,
                    target=target,
                    context_variables=context,
                )
                for context in candidate_contexts
            )
            heterogeneity = _conditional_effect_heterogeneity(
                transitions,
                action=action,
                target=target,
                context_variables=tuple(
                    variable for variable in context_variables if variable != target
                ),
            )
            scored_edges.append(((action, target), score, heterogeneity))

    suffix = "-readout-oracle" if remove_readouts else ""
    candidates: list[MechanismBaselineResult] = []
    for threshold in thresholds:
        discovered = {
            edge for edge, score, _ in scored_edges if score >= threshold
        }
        hidden_labels = {
            edge
            for edge, score, heterogeneity in scored_edges
            if score >= threshold and heterogeneity >= 0.35
        }
        candidates.append(
            MechanismBaselineResult(
                name=f"conditional-discovery{suffix}",
                discovered_edges=discovered,
                hidden_labels=hidden_labels,
                threshold=threshold,
            )
        )
    return candidates


def _conditional_mutual_information(
    transitions: tuple[Transition, ...],
    action: str,
    target: str,
    context_variables: tuple[str, ...],
) -> float:
    counts: dict[tuple[tuple[str, bool], ...], Counter[tuple[bool, bool]]] = {}
    total = 0
    for transition in transitions:
        if target not in transition.after:
            continue
        key = (("prev_target", bool(transition.before.get(target, False))),) + tuple(
            (variable, bool(transition.before.get(variable, False)))
            for variable in context_variables
        )
        treatment = transition.action == action
        outcome = bool(transition.after[target])
        counts.setdefault(key, Counter())[(treatment, outcome)] += 1
        total += 1
    if total == 0:
        return 0.0

    cmi = 0.0
    for joint_counts in counts.values():
        stratum_total = sum(joint_counts.values())
        if stratum_total < 4:
            continue
        treatment_counts: Counter[bool] = Counter()
        outcome_counts: Counter[bool] = Counter()
        for (treatment, outcome), count in joint_counts.items():
            treatment_counts[treatment] += count
            outcome_counts[outcome] += count
        if len(treatment_counts) < 2 or len(outcome_counts) < 2:
            continue
        stratum_weight = stratum_total / total
        for (treatment, outcome), count in joint_counts.items():
            if count == 0:
                continue
            p_joint = count / stratum_total
            p_treatment = treatment_counts[treatment] / stratum_total
            p_outcome = outcome_counts[outcome] / stratum_total
            cmi += stratum_weight * p_joint * log(
                p_joint / (p_treatment * p_outcome)
            )
    return cmi


def _conditional_effect_heterogeneity(
    transitions: tuple[Transition, ...],
    action: str,
    target: str,
    context_variables: tuple[str, ...],
) -> float:
    action_rows = [
        transition
        for transition in transitions
        if transition.action == action and target in transition.after
    ]
    if len(action_rows) < 6:
        return 0.0
    base_rate = mean(float(transition.after[target]) for transition in action_rows)
    best_gap = 0.0
    for variable in context_variables:
        splits: dict[bool, list[float]] = {False: [], True: []}
        for transition in action_rows:
            splits[bool(transition.before.get(variable, False))].append(
                float(transition.after[target])
            )
        if min(len(values) for values in splits.values()) < 2:
            continue
        gap = max(abs(mean(values) - base_rate) for values in splits.values())
        best_gap = max(best_gap, gap)
    return best_gap


def _neural_transition_probe_grid(
    transitions: tuple[Transition, ...],
    readout_variables: set[str],
    thresholds: tuple[float, ...],
    remove_readouts: bool,
    seed: int,
    epochs: int,
) -> list[MechanismBaselineResult]:
    if not transitions:
        return [
            MechanismBaselineResult(
                name="neural-transition",
                discovered_edges=set(),
                hidden_labels=set(),
                threshold=threshold,
            )
            for threshold in thresholds
        ]
    variables = sorted(
        {
            target
            for transition in transitions
            for target in set(transition.before) | set(transition.after)
        }
    )
    actions = sorted({transition.action for transition in transitions})
    baseline_action = "wait" if "wait" in actions else actions[0]
    model = _VectorizedTransitionProbe(variables, actions, seed=seed)
    model.fit(transitions, epochs=epochs)
    states = _unique_states([transition.before for transition in transitions], limit=96)
    scored_edges: list[tuple[Edge, float, float]] = []
    for action in actions:
        for target in variables:
            if remove_readouts and target in readout_variables:
                continue
            diffs = [
                model.predict_probability(target, state, action)
                - model.predict_probability(target, state, baseline_action)
                for state in states
            ]
            if not diffs:
                continue
            mean_abs = mean(abs(diff) for diff in diffs)
            state_changes = [
                abs(model.predict_probability(target, state, action) - float(state[target]))
                for state in states
                if target in state
            ]
            mid_rate = 0.0
            if state_changes:
                mid_rate = sum(
                    0.20 <= change <= 0.80 for change in state_changes
                ) / len(state_changes)
            scored_edges.append(((action, target), mean_abs, mid_rate))
    suffix = "-readout-oracle" if remove_readouts else ""
    candidates: list[MechanismBaselineResult] = []
    for threshold in thresholds:
        discovered = {
            edge for edge, score, _ in scored_edges if score >= threshold
        }
        hidden_labels = {
            edge
            for edge, score, mid_rate in scored_edges
            if score >= threshold and mid_rate >= 0.25
        }
        candidates.append(
            MechanismBaselineResult(
                name=f"neural-transition{suffix}",
                discovered_edges=discovered,
                hidden_labels=hidden_labels,
                threshold=threshold,
            )
        )
    return candidates


def _latent_world_model_probe_grid(
    transitions: tuple[Transition, ...],
    readout_variables: set[str],
    thresholds: tuple[float, ...],
    remove_readouts: bool,
    seed: int,
    epochs: int,
) -> list[MechanismBaselineResult]:
    if not transitions:
        return [
            MechanismBaselineResult(
                name="latent-world-model",
                discovered_edges=set(),
                hidden_labels=set(),
                threshold=threshold,
            )
            for threshold in thresholds
        ]
    variables = sorted(
        {
            target
            for transition in transitions
            for target in set(transition.before) | set(transition.after)
        }
    )
    actions = sorted({transition.action for transition in transitions})
    baseline_action = "wait" if "wait" in actions else actions[0]
    try:
        model = _TorchLatentWorldModel(variables, actions, seed=seed)
        model.fit(transitions, epochs=epochs)
    except Exception:
        model = _VectorizedTransitionProbe(variables, actions, seed=seed)
        model.fit(transitions, epochs=epochs)
    diagnostics = _transition_prediction_diagnostics(model, transitions, variables)
    states = _unique_states([transition.before for transition in transitions], limit=96)
    scored_edges: list[tuple[Edge, float, float]] = []
    for action in actions:
        for target in variables:
            if remove_readouts and target in readout_variables:
                continue
            diffs = [
                model.predict_probability(target, state, action)
                - model.predict_probability(target, state, baseline_action)
                for state in states
            ]
            if not diffs:
                continue
            mean_abs = mean(abs(diff) for diff in diffs)
            active_cut = max(0.05, min(thresholds) if thresholds else 0.05)
            active_rate = sum(abs(diff) >= active_cut for diff in diffs) / len(diffs)
            sign_balance = min(
                sum(diff > active_cut for diff in diffs),
                sum(diff < -active_cut for diff in diffs),
            ) / len(diffs)
            hidden_score = min(active_rate, 1.0 - active_rate) + sign_balance
            scored_edges.append(((action, target), mean_abs, hidden_score))
    suffix = "-readout-oracle" if remove_readouts else ""
    candidates: list[MechanismBaselineResult] = []
    for threshold in thresholds:
        discovered = {
            edge for edge, score, _ in scored_edges if score >= threshold
        }
        hidden_labels = {
            edge
            for edge, score, hidden_score in scored_edges
            if score >= threshold and hidden_score >= 0.20
        }
        candidates.append(
            MechanismBaselineResult(
                name=f"latent-world-model{suffix}",
                discovered_edges=discovered,
                hidden_labels=hidden_labels,
                threshold=threshold,
                diagnostics=diagnostics,
            )
        )
    return candidates


def _transition_prediction_diagnostics(
    model: object,
    transitions: tuple[Transition, ...],
    variables: list[str],
) -> dict[str, float]:
    if not transitions or not variables:
        return {"model_next_bit_accuracy": 0.0, "model_next_bce": 0.0}
    correct = 0
    total = 0
    bce = 0.0
    eps = 1e-6
    for transition in transitions:
        for variable in variables:
            if variable not in transition.after:
                continue
            label = 1.0 if transition.after[variable] else 0.0
            probability = float(
                model.predict_probability(variable, transition.before, transition.action)
            )
            probability = min(max(probability, eps), 1.0 - eps)
            correct += int((probability >= 0.5) == bool(label))
            total += 1
            bce -= label * log(probability) + (1.0 - label) * log(1.0 - probability)
    return {
        "model_next_bit_accuracy": _safe_div(correct, total),
        "model_next_bce": _safe_div(bce, total),
    }


class _PerTargetTransitionModel:
    def __init__(self, variables: list[str], actions: list[str], seed: int) -> None:
        self.variables = variables
        self.actions = actions
        self.rng = Random(seed)
        self.weights: dict[str, Counter[str]] = {
            target: Counter() for target in variables
        }

    def fit(
        self,
        transitions: tuple[Transition, ...],
        epochs: int,
        learning_rate: float,
    ) -> None:
        rows = list(transitions)
        for _ in range(epochs):
            self.rng.shuffle(rows)
            for transition in rows:
                features = self._features(transition.before, transition.action)
                for target in self.variables:
                    if target not in transition.after:
                        continue
                    label = 1.0 if transition.after[target] else 0.0
                    score = sum(
                        self.weights[target][feature] * value
                        for feature, value in features.items()
                    )
                    prediction = _sigmoid(score)
                    error = label - prediction
                    for feature, value in features.items():
                        self.weights[target][feature] += learning_rate * error * value

    def predict_probability(self, target: str, state: State, action: str) -> float:
        features = self._features(state, action)
        score = sum(
            self.weights[target][feature] * value
            for feature, value in features.items()
        )
        return _sigmoid(score)

    def _features(self, state: State, action: str) -> dict[str, float]:
        features = {
            "bias": 1.0,
            f"action:{action}": 1.0,
        }
        for variable, value in sorted(state.items()):
            bit = "1" if value else "0"
            features[f"state:{variable}={bit}"] = 0.35
            features[f"action:{action}|{variable}={bit}"] = 1.0
        return features


class _TorchLatentWorldModel:
    def __init__(self, variables: list[str], actions: list[str], seed: int) -> None:
        import torch

        self.torch = torch
        self.variables = variables
        self.actions = actions
        self.variable_index = {variable: idx for idx, variable in enumerate(variables)}
        self.action_index = {action: idx for idx, action in enumerate(actions)}
        self.state_dim = len(variables)
        self.action_dim = len(actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass
        self.net = self._build_network().to(self.device)

    def fit(self, transitions: tuple[Transition, ...], epochs: int) -> None:
        torch = self.torch
        x = torch.tensor(
            [self._state_vector(transition.before) for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        a = torch.tensor(
            [self._action_vector(transition.action) for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        y = torch.tensor(
            [self._state_vector(transition.after) for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.01, weight_decay=1e-4)
        batch_size = min(64, len(transitions))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(17 + len(transitions) + self.state_dim)
        for _ in range(max(epochs, 1)):
            order = torch.randperm(len(transitions), generator=generator, device=self.device)
            for start in range(0, len(transitions), batch_size):
                idx = order[start : start + batch_size]
                logits, rec_logits = self.net(x[idx], a[idx])
                next_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    y[idx],
                )
                rec_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    rec_logits,
                    x[idx],
                )
                loss = next_loss + 0.25 * rec_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict_probability(self, target: str, state: State, action: str) -> float:
        torch = self.torch
        with torch.no_grad():
            x = torch.tensor(
                [self._state_vector(state)],
                dtype=torch.float32,
                device=self.device,
            )
            a = torch.tensor(
                [self._action_vector(action)],
                dtype=torch.float32,
                device=self.device,
            )
            logits, _ = self.net(x, a)
            probability = torch.sigmoid(logits)[0, self.variable_index[target]]
        return float(probability.detach().cpu())

    def _build_network(self):
        torch = self.torch
        latent_dim = min(16, max(4, self.state_dim // 2 + self.action_dim // 2))
        hidden_dim = min(64, max(16, 2 * (self.state_dim + self.action_dim)))

        class LatentDynamics(torch.nn.Module):
            def __init__(self, state_dim: int, action_dim: int) -> None:
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, hidden_dim),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_dim, latent_dim),
                    torch.nn.Tanh(),
                )
                self.transition = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim + action_dim, hidden_dim),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_dim, latent_dim),
                    torch.nn.Tanh(),
                )
                self.decoder = torch.nn.Linear(latent_dim, state_dim)
                self.reconstruction = torch.nn.Linear(latent_dim, state_dim)

            def forward(self, state, action):
                z = self.encoder(state)
                z_next = self.transition(torch.cat([z, action], dim=-1))
                return self.decoder(z_next), self.reconstruction(z)

        return LatentDynamics(self.state_dim, self.action_dim)

    def _state_vector(self, state: State) -> list[float]:
        return [1.0 if state.get(variable, False) else 0.0 for variable in self.variables]

    def _action_vector(self, action: str) -> list[float]:
        vector = [0.0] * len(self.actions)
        vector[self.action_index[action]] = 1.0
        return vector


class _VectorizedTransitionProbe:
    """Multi-output transition probe with the same interaction features.

    The previous implementation updated one target at a time in Python. This
    version fits all targets with a single ridge solve, which preserves the
    baseline interface while making large matched runs feasible.
    """

    def __init__(self, variables: list[str], actions: list[str], seed: int) -> None:
        del seed
        self.variables = variables
        self.actions = actions
        self.variable_index = {variable: idx for idx, variable in enumerate(variables)}
        self.action_index = {action: idx for idx, action in enumerate(actions)}
        self.feature_count = 1 + len(actions) + 2 * len(variables) * (1 + len(actions))
        self.weights = np.zeros((self.feature_count, len(variables)), dtype=np.float64)

    def fit(self, transitions: tuple[Transition, ...], epochs: int) -> None:
        if not transitions:
            return
        x = np.vstack(
            [self._features(transition.before, transition.action) for transition in transitions]
        )
        y = np.zeros((len(transitions), len(self.variables)), dtype=np.float64)
        for row, transition in enumerate(transitions):
            for target, value in transition.after.items():
                target_idx = self.variable_index.get(target)
                if target_idx is not None:
                    y[row, target_idx] = 1.0 if value else 0.0
        ridge = 1.0 / max(float(epochs), 1.0)
        lhs = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float64)
        rhs = x.T @ y
        self.weights = np.linalg.solve(lhs, rhs)

    def predict_probability(self, target: str, state: State, action: str) -> float:
        target_idx = self.variable_index[target]
        score = float(self._features(state, action) @ self.weights[:, target_idx])
        return min(max(score, 0.0), 1.0)

    def _features(self, state: State, action: str) -> np.ndarray:
        features = np.zeros(self.feature_count, dtype=np.float64)
        features[0] = 1.0
        action_idx = self.action_index[action]
        action_offset = 1
        state_offset = action_offset + len(self.actions)
        interaction_offset = state_offset + 2 * len(self.variables)

        features[action_offset + action_idx] = 1.0
        for variable, variable_idx in self.variable_index.items():
            bit_idx = 1 if state.get(variable, False) else 0
            features[state_offset + 2 * variable_idx + bit_idx] = 0.35
            interaction_idx = (
                interaction_offset
                + action_idx * 2 * len(self.variables)
                + 2 * variable_idx
                + bit_idx
            )
            features[interaction_idx] = 1.0
        return features


def _run_candidate_budget_sensitivity(
    families: int,
    seeds: int,
    steps_list: tuple[int, ...],
    mechanisms: int,
    visible: int,
    readouts: int,
    noise: float,
) -> dict[str, object]:
    configs = {
        "small": (8, 2, 8),
        "default": (32, 4, 20),
        "wide": (64, 5, 32),
    }
    summary_inputs: dict[tuple[str, int], list[dict[str, object]]] = {
        (config, steps): [] for config in configs for steps in steps_list
    }
    runs: list[dict[str, object]] = []
    for config, (candidate_cap, variable_cap, setup_cap) in configs.items():
        for steps in steps_list:
            for family_seed in range(1, families + 1):
                spec_world = make_procedural_complex_hidden_world(
                    family_seed=family_seed,
                    mechanism_count=mechanisms,
                    visible_count=visible,
                    noise_probability=noise,
                    readout_count=readouts,
                )
                readout_variables = spec_world.readout_variables()
                hidden_edges = spec_world.hidden_context_edges()
                for seed in range(1, seeds + 1):
                    agent = ContextSearchControlExperimentPlannerAgent()
                    agent._max_context_search_candidates = candidate_cap
                    agent._context_search_variable_limit = variable_cap
                    agent._context_search_setup_limit = setup_cap
                    result = run_episode(
                        make_procedural_complex_hidden_world(
                            family_seed=family_seed,
                            mechanism_count=mechanisms,
                            visible_count=visible,
                            noise_probability=noise,
                            readout_count=readouts,
                        ),
                        agent,
                        steps=steps,
                        seed=seed,
                    )
                    row = _metric_row(
                        config,
                        result.discovered_edges,
                        result.true_edges,
                        readout_variables,
                        hidden_edges,
                        _hidden_context_hint_edges(result.condition_hints),
                        transitions=result.transitions,
                    )
                    row = row | {
                        "config": config,
                        "steps": steps,
                        "family_seed": family_seed,
                        "seed": seed,
                        "candidate_cap": candidate_cap,
                        "variable_cap": variable_cap,
                        "setup_cap": setup_cap,
                    }
                    summary_inputs[(config, steps)].append(row)
                    runs.append(row)
    summary = []
    for config in configs:
        for steps in steps_list:
            summary.append(
                _summary_row(
                    config,
                    summary_inputs[(config, steps)],
                )
                | {
                    "config": config,
                    "steps": steps,
                }
            )
    return {
        "configs": {
            config: {
                "candidate_cap": values[0],
                "variable_cap": values[1],
                "setup_cap": values[2],
            }
            for config, values in configs.items()
        },
        "summary": summary,
        "runs": runs,
    }


def _run_continuous_noise_sweep(
    seeds: int,
    noise_levels: tuple[float, ...],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    payloads: list[dict[str, object]] = []
    for noise in noise_levels:
        payload = run_continuous_metric_poc(
            seeds=seeds,
            random_trials=160,
            regression_trials=240,
            paired_samples=8,
            noise=noise,
        )
        payloads.append(payload)
        by_agent = {
            row["agent"]: row
            for row in payload["summary"]  # type: ignore[index]
        }
        rows.append(
            {
                "noise": noise,
                "random_correlation_f1": by_agent["random-correlation"]["f1"],
                "global_linear_f1": by_agent["global-linear-regression"]["f1"],
                "metric_causal_core_f1": by_agent["metric-causal-core"]["f1"],
                "metric_causal_core_readout_fp": by_agent["metric-causal-core"][
                    "readout_false_positive"
                ],
            }
        )
    return {
        "seeds": seeds,
        "summary": rows,
        "payloads": payloads,
    }


def _metric_row(
    method: str,
    discovered_edges: set[Edge],
    true_edges: set[Edge],
    readout_variables: set[str],
    hidden_edges: set[Edge],
    hidden_labels: set[Edge],
    threshold: float | None = None,
    transitions: tuple[Transition, ...] = (),
    diagnostics: dict[str, float] | None = None,
) -> dict[str, object]:
    score = score_edges(discovered_edges, true_edges, transitions)
    false_edges = discovered_edges - true_edges
    row: dict[str, object] = {
        "method": method,
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "false_positive": len(false_edges),
        "readout_false_positive": sum(
            1 for _, target in false_edges if target in readout_variables
        ),
        "non_readout_false_positive": sum(
            1 for _, target in false_edges if target not in readout_variables
        ),
        "hidden_context_recall": _safe_div(
            len(hidden_labels & hidden_edges),
            len(hidden_edges),
        ),
        "hidden_context_false_positive": len(hidden_labels - hidden_edges),
        "edges": len(discovered_edges),
        "threshold": threshold,
    }
    if diagnostics:
        row.update(diagnostics)
    return row


def _summary_row(method: str, rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = (
        "precision",
        "recall",
        "f1",
        "false_positive",
        "readout_false_positive",
        "non_readout_false_positive",
        "hidden_context_recall",
        "hidden_context_false_positive",
        "edges",
    )
    summary: dict[str, object] = {"method": method}
    for key in numeric_keys:
        summary[key] = _mean([float(row[key]) for row in rows])
    extra_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("model_") and isinstance(row[key], (int, float))
        }
    )
    for key in extra_keys:
        values = [float(row[key]) for row in rows if key in row]
        summary[key] = _mean(values)
    thresholds = [
        float(row["threshold"])
        for row in rows
        if row.get("threshold") is not None
    ]
    if thresholds:
        summary["threshold"] = round(_mean(thresholds), 3)
    return summary


def _threshold_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        threshold = row.get("threshold")
        if threshold is None:
            continue
        grouped.setdefault((str(row["method"]), float(threshold)), []).append(row)
    summary = []
    for (method, threshold), group in sorted(grouped.items()):
        summary.append(_summary_row(method, group) | {"threshold": threshold})
    return summary


def _best_candidate(
    candidates: list[MechanismBaselineResult],
    true_edges: set[Edge],
) -> MechanismBaselineResult:
    return max(
        candidates,
        key=lambda candidate: (
            score_edges(candidate.discovered_edges, true_edges, ()).f1,
            -len(candidate.discovered_edges - true_edges),
            -(candidate.threshold or 0.0),
        ),
    )


def _hidden_context_hint_edges(
    condition_hints: dict[Edge, list[str]]
) -> set[Edge]:
    hidden_edges: set[Edge] = set()
    for edge, hints in condition_hints.items():
        hint_text = " ".join(hints).lower()
        if "hidden" in hint_text or "latent" in hint_text:
            hidden_edges.add(edge)
    return hidden_edges


def _unique_states(states: list[State], limit: int) -> list[State]:
    unique: dict[tuple[tuple[str, bool], ...], State] = {}
    for state in states:
        unique.setdefault(tuple(sorted(state.items())), dict(state))
        if len(unique) >= limit:
            break
    return list(unique.values())


def _sigmoid(value: float) -> float:
    if value >= 30.0:
        return 1.0
    if value <= -30.0:
        return 0.0
    return 1.0 / (1.0 + exp(-value))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
