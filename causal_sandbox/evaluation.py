from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .core import Agent, Edge, EdgeScore, Transition, World
from .causal_core import state_signature
from .reward import task_utility


@dataclass(frozen=True)
class EpisodeResult:
    world_name: str
    agent_name: str
    transitions: tuple[Transition, ...]
    discovered_edges: set[Edge]
    true_edges: set[Edge]
    condition_hints: dict[Edge, list[str]]
    score: EdgeScore

    def to_dict(self) -> dict[str, object]:
        return {
            "world": self.world_name,
            "agent": self.agent_name,
            "steps": len(self.transitions),
            "score": {
                "true_positive": self.score.true_positive,
                "false_positive": self.score.false_positive,
                "false_negative": self.score.false_negative,
                "precision": self.score.precision,
                "recall": self.score.recall,
                "f1": self.score.f1,
                "prediction_jaccard": self.score.prediction_jaccard,
            },
            "discovered_edges": [
                {"action": action, "target": target}
                for action, target in sorted(self.discovered_edges)
            ],
            "condition_hints": {
                f"{action}->{target}": hints
                for (action, target), hints in sorted(self.condition_hints.items())
            },
            "transitions": [
                {
                    "step": transition.step,
                    "action": transition.action,
                    "changed": sorted(transition.changed),
                    "prediction": sorted(transition.prediction),
                    "before": transition.before,
                    "after": transition.after,
                    "hypothesis": transition.hypothesis,
                }
                for transition in self.transitions
            ],
        }


@dataclass(frozen=True)
class InterventionTestCase:
    action: str
    state: dict[str, bool]
    expected: frozenset[str]
    predicted: frozenset[str]
    seen_during_exploration: bool
    mechanism_shifted: bool = False

    @property
    def exact_match(self) -> bool:
        return self.expected == self.predicted

    @property
    def jaccard(self) -> float:
        union = self.expected | self.predicted
        if not union:
            return 1.0
        return len(self.expected & self.predicted) / len(union)


@dataclass(frozen=True)
class InterventionTestResult:
    exploration: EpisodeResult
    cases: tuple[InterventionTestCase, ...]
    exact_match_rate: float
    prediction_precision: float
    prediction_recall: float
    prediction_f1: float
    mean_jaccard: float

    def to_dict(self) -> dict[str, object]:
        return {
            "world": self.exploration.world_name,
            "agent": self.exploration.agent_name,
            "explore_steps": len(self.exploration.transitions),
            "exploration_score": self.exploration.to_dict()["score"],
            "heldout_cases": len(self.cases),
            "intervention_test": {
                "exact_match_rate": self.exact_match_rate,
                "prediction_precision": self.prediction_precision,
                "prediction_recall": self.prediction_recall,
                "prediction_f1": self.prediction_f1,
                "mean_jaccard": self.mean_jaccard,
            },
            "cases": [
                {
                    "action": case.action,
                    "state": case.state,
                    "expected": sorted(case.expected),
                    "predicted": sorted(case.predicted),
                    "exact_match": case.exact_match,
                    "jaccard": case.jaccard,
                    "seen_during_exploration": case.seen_during_exploration,
                    "mechanism_shifted": case.mechanism_shifted,
                }
                for case in self.cases
            ],
        }


@dataclass(frozen=True)
class TransferInterventionTestResult:
    exploration: EpisodeResult
    target_world_name: str
    cases: tuple[InterventionTestCase, ...]
    exact_match_rate: float
    prediction_precision: float
    prediction_recall: float
    prediction_f1: float
    mean_jaccard: float
    shifted_cases: tuple[InterventionTestCase, ...]
    shifted_exact_match_rate: float
    shifted_prediction_f1: float

    def to_dict(self) -> dict[str, object]:
        return {
            "source_world": self.exploration.world_name,
            "target_world": self.target_world_name,
            "agent": self.exploration.agent_name,
            "explore_steps": len(self.exploration.transitions),
            "exploration_score": self.exploration.to_dict()["score"],
            "transfer_cases": len(self.cases),
            "transfer_intervention_test": {
                "exact_match_rate": self.exact_match_rate,
                "prediction_precision": self.prediction_precision,
                "prediction_recall": self.prediction_recall,
                "prediction_f1": self.prediction_f1,
                "mean_jaccard": self.mean_jaccard,
                "shifted_cases": len(self.shifted_cases),
                "shifted_exact_match_rate": self.shifted_exact_match_rate,
                "shifted_prediction_f1": self.shifted_prediction_f1,
            },
        }


@dataclass(frozen=True)
class TransferAdaptationTestResult:
    before_adaptation: TransferInterventionTestResult
    after_adaptation: TransferInterventionTestResult
    adaptation_steps: int
    adaptation_mode: str

    def to_dict(self) -> dict[str, object]:
        return {
            "before_adaptation": self.before_adaptation.to_dict(),
            "after_adaptation": self.after_adaptation.to_dict(),
            "adaptation_steps": self.adaptation_steps,
            "adaptation_mode": self.adaptation_mode,
        }


@dataclass(frozen=True)
class Level3TestResult:
    counterfactual: CounterfactualTestResult
    stable_transfer: TransferInterventionTestResult
    repair: TransferAdaptationTestResult

    @property
    def counterfactual_delta_f1(self) -> float:
        return self.counterfactual.delta_f1

    @property
    def stable_transfer_f1(self) -> float:
        return self.stable_transfer.prediction_f1

    @property
    def repair_before_shift_f1(self) -> float:
        return self.repair.before_adaptation.shifted_prediction_f1

    @property
    def repair_after_shift_f1(self) -> float:
        return self.repair.after_adaptation.shifted_prediction_f1

    @property
    def repair_shift_gain(self) -> float:
        return self.repair_after_shift_f1 - self.repair_before_shift_f1

    @property
    def level3_score(self) -> float:
        return (
            self.counterfactual_delta_f1
            + self.stable_transfer_f1
            + self.repair_after_shift_f1
        ) / 3.0

    def to_dict(self) -> dict[str, object]:
        return {
            "counterfactual": self.counterfactual.to_dict(),
            "stable_transfer": self.stable_transfer.to_dict(),
            "repair": self.repair.to_dict(),
            "level3": {
                "counterfactual_delta_f1": self.counterfactual_delta_f1,
                "stable_transfer_f1": self.stable_transfer_f1,
                "repair_before_shift_f1": self.repair_before_shift_f1,
                "repair_after_shift_f1": self.repair_after_shift_f1,
                "repair_shift_gain": self.repair_shift_gain,
                "level3_score": self.level3_score,
            },
        }


@dataclass(frozen=True)
class RewardEpisodeResult:
    exploration: EpisodeResult
    total_return: float
    mean_return: float
    final_utility: float

    def to_dict(self) -> dict[str, object]:
        return {
            "world": self.exploration.world_name,
            "agent": self.exploration.agent_name,
            "steps": len(self.exploration.transitions),
            "score": self.exploration.to_dict()["score"],
            "reward": {
                "total_return": self.total_return,
                "mean_return": self.mean_return,
                "final_utility": self.final_utility,
            },
        }


@dataclass(frozen=True)
class GateDisambiguationCase:
    gate_variable: str
    gate_value: bool
    state: dict[str, bool]
    expected_effect: bool
    predicted_effect: bool

    @property
    def correct(self) -> bool:
        return self.expected_effect == self.predicted_effect


@dataclass(frozen=True)
class Level4DisambiguationResult:
    exploration: EpisodeResult
    target_action: str
    target_variable: str
    candidate_gates: tuple[tuple[str, bool], ...]
    cases: tuple[GateDisambiguationCase, ...]
    first_disambiguating_step: int | None
    first_correct_step: int | None
    disambiguating_tests: int
    final_correct: bool
    level4_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "world": self.exploration.world_name,
            "agent": self.exploration.agent_name,
            "steps": len(self.exploration.transitions),
            "target_action": self.target_action,
            "target_variable": self.target_variable,
            "candidate_gates": [
                {"variable": variable, "value": value}
                for variable, value in self.candidate_gates
            ],
            "first_disambiguating_step": self.first_disambiguating_step,
            "first_correct_step": self.first_correct_step,
            "disambiguating_tests": self.disambiguating_tests,
            "final_correct": self.final_correct,
            "level4_score": self.level4_score,
            "cases": [
                {
                    "gate_variable": case.gate_variable,
                    "gate_value": case.gate_value,
                    "state": case.state,
                    "expected_effect": case.expected_effect,
                    "predicted_effect": case.predicted_effect,
                    "correct": case.correct,
                }
                for case in self.cases
            ],
        }


@dataclass(frozen=True)
class DelayedEffectCase:
    action: str
    followup_action: str
    delay_steps: int
    state: dict[str, bool]
    expected_effect: bool
    predicted_effect: bool

    @property
    def correct(self) -> bool:
        return self.expected_effect == self.predicted_effect


@dataclass(frozen=True)
class TemporalCreditResult:
    exploration: EpisodeResult
    target_action: str
    target_variable: str
    followup_action: str
    delay_steps: int
    cases: tuple[DelayedEffectCase, ...]
    first_correct_step: int | None
    final_correct: bool
    delayed_edge_learned: bool
    followup_misattribution: bool
    temporal_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "world": self.exploration.world_name,
            "agent": self.exploration.agent_name,
            "steps": len(self.exploration.transitions),
            "target_action": self.target_action,
            "target_variable": self.target_variable,
            "followup_action": self.followup_action,
            "delay_steps": self.delay_steps,
            "first_correct_step": self.first_correct_step,
            "final_correct": self.final_correct,
            "delayed_edge_learned": self.delayed_edge_learned,
            "followup_misattribution": self.followup_misattribution,
            "temporal_score": self.temporal_score,
            "cases": [
                {
                    "action": case.action,
                    "followup_action": case.followup_action,
                    "delay_steps": case.delay_steps,
                    "state": case.state,
                    "expected_effect": case.expected_effect,
                    "predicted_effect": case.predicted_effect,
                    "correct": case.correct,
                }
                for case in self.cases
            ],
        }


@dataclass(frozen=True)
class TemporalTransferResult:
    exploration: EpisodeResult
    target_world_name: str
    target_action: str
    target_variable: str
    followup_action: str
    delay_steps: int
    cases: tuple[DelayedEffectCase, ...]
    final_correct: bool
    delayed_edge_learned: bool
    followup_misattribution: bool
    mechanism_shifted: bool
    temporal_transfer_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "source_world": self.exploration.world_name,
            "target_world": self.target_world_name,
            "agent": self.exploration.agent_name,
            "explore_steps": len(self.exploration.transitions),
            "target_action": self.target_action,
            "target_variable": self.target_variable,
            "followup_action": self.followup_action,
            "delay_steps": self.delay_steps,
            "final_correct": self.final_correct,
            "delayed_edge_learned": self.delayed_edge_learned,
            "followup_misattribution": self.followup_misattribution,
            "mechanism_shifted": self.mechanism_shifted,
            "temporal_transfer_score": self.temporal_transfer_score,
            "cases": [
                {
                    "action": case.action,
                    "followup_action": case.followup_action,
                    "delay_steps": case.delay_steps,
                    "state": case.state,
                    "expected_effect": case.expected_effect,
                    "predicted_effect": case.predicted_effect,
                    "correct": case.correct,
                }
                for case in self.cases
            ],
        }


@dataclass(frozen=True)
class TemporalAdaptationResult:
    before_adaptation: TemporalTransferResult
    after_adaptation: TemporalTransferResult
    adaptation_steps: int
    adaptation_mode: str

    @property
    def repair_gain(self) -> float:
        return (
            self.after_adaptation.temporal_transfer_score
            - self.before_adaptation.temporal_transfer_score
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "before_adaptation": self.before_adaptation.to_dict(),
            "after_adaptation": self.after_adaptation.to_dict(),
            "adaptation_steps": self.adaptation_steps,
            "adaptation_mode": self.adaptation_mode,
            "repair_gain": self.repair_gain,
        }


@dataclass(frozen=True)
class TemporalTargetResult:
    target_world_name: str
    target_action: str
    target_variable: str
    followup_action: str
    delay_steps: int
    cases: tuple[DelayedEffectCase, ...]
    final_correct: bool
    delayed_edge_learned: bool
    followup_misattribution: bool
    mechanism_shifted: bool
    temporal_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "target_world": self.target_world_name,
            "target_action": self.target_action,
            "target_variable": self.target_variable,
            "followup_action": self.followup_action,
            "delay_steps": self.delay_steps,
            "final_correct": self.final_correct,
            "delayed_edge_learned": self.delayed_edge_learned,
            "followup_misattribution": self.followup_misattribution,
            "mechanism_shifted": self.mechanism_shifted,
            "temporal_score": self.temporal_score,
            "cases": [
                {
                    "action": case.action,
                    "followup_action": case.followup_action,
                    "delay_steps": case.delay_steps,
                    "state": case.state,
                    "expected_effect": case.expected_effect,
                    "predicted_effect": case.predicted_effect,
                    "correct": case.correct,
                }
                for case in self.cases
            ],
        }


@dataclass(frozen=True)
class TemporalSelectiveAdaptationResult:
    exploration: EpisodeResult
    target_world_name: str
    before_targets: tuple[TemporalTargetResult, ...]
    after_targets: tuple[TemporalTargetResult, ...]
    adaptation_steps: int
    adaptation_mode: str

    @property
    def before_score(self) -> float:
        return _temporal_target_mean(self.before_targets)

    @property
    def after_score(self) -> float:
        return _temporal_target_mean(self.after_targets)

    @property
    def repair_gain(self) -> float:
        return self.after_score - self.before_score

    @property
    def after_all_success(self) -> bool:
        return all(target.temporal_score == 1.0 for target in self.after_targets)

    @property
    def shifted_before_score(self) -> float:
        return _temporal_target_mean(self.before_targets, mechanism_shifted=True)

    @property
    def shifted_after_score(self) -> float:
        return _temporal_target_mean(self.after_targets, mechanism_shifted=True)

    @property
    def stable_before_score(self) -> float:
        return _temporal_target_mean(self.before_targets, mechanism_shifted=False)

    @property
    def stable_after_score(self) -> float:
        return _temporal_target_mean(self.after_targets, mechanism_shifted=False)

    @property
    def shifted_count(self) -> int:
        return sum(1 for target in self.after_targets if target.mechanism_shifted)

    @property
    def stable_count(self) -> int:
        return sum(1 for target in self.after_targets if not target.mechanism_shifted)

    @property
    def selective_score(self) -> float:
        if self.shifted_count == 0 or self.stable_count == 0:
            return 0.0
        if not self.after_all_success:
            return 0.0
        if self.shifted_after_score <= self.shifted_before_score:
            return 0.0
        return 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "source_world": self.exploration.world_name,
            "target_world": self.target_world_name,
            "agent": self.exploration.agent_name,
            "explore_steps": len(self.exploration.transitions),
            "adaptation_steps": self.adaptation_steps,
            "adaptation_mode": self.adaptation_mode,
            "before_score": self.before_score,
            "after_score": self.after_score,
            "repair_gain": self.repair_gain,
            "after_all_success": self.after_all_success,
            "shifted_before_score": self.shifted_before_score,
            "shifted_after_score": self.shifted_after_score,
            "stable_before_score": self.stable_before_score,
            "stable_after_score": self.stable_after_score,
            "selective_score": self.selective_score,
            "targets": {
                "before": [target.to_dict() for target in self.before_targets],
                "after": [target.to_dict() for target in self.after_targets],
            },
        }


@dataclass(frozen=True)
class Level6SchemaTransferResult:
    exploration: EpisodeResult
    target_world_name: str
    targets: tuple[TemporalTargetResult, ...]
    target_steps: int
    transfer_mode: str

    @property
    def target_score(self) -> float:
        return _temporal_target_mean(self.targets)

    @property
    def all_success(self) -> bool:
        return all(target.temporal_score == 1.0 for target in self.targets)

    @property
    def delayed_edge_rate(self) -> float:
        return _safe_div(
            sum(1 for target in self.targets if target.delayed_edge_learned),
            len(self.targets),
        )

    @property
    def followup_misattribution_rate(self) -> float:
        return _safe_div(
            sum(1 for target in self.targets if target.followup_misattribution),
            len(self.targets),
        )

    @property
    def level6_score(self) -> float:
        if not self.all_success:
            return 0.0
        if self.followup_misattribution_rate > 0.0:
            return 0.0
        return 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "source_world": self.exploration.world_name,
            "target_world": self.target_world_name,
            "agent": self.exploration.agent_name,
            "source_steps": len(self.exploration.transitions),
            "target_steps": self.target_steps,
            "transfer_mode": self.transfer_mode,
            "target_score": self.target_score,
            "all_success": self.all_success,
            "delayed_edge_rate": self.delayed_edge_rate,
            "followup_misattribution_rate": self.followup_misattribution_rate,
            "level6_score": self.level6_score,
            "targets": [target.to_dict() for target in self.targets],
        }


@dataclass(frozen=True)
class Level6SchemaRepairResult:
    exploration: EpisodeResult
    target_world_name: str
    before_targets: tuple[TemporalTargetResult, ...]
    after_targets: tuple[TemporalTargetResult, ...]
    target_steps: int
    transfer_mode: str
    shifted_roles: frozenset[str]

    @property
    def before_score(self) -> float:
        return _temporal_target_mean(self.before_targets)

    @property
    def after_score(self) -> float:
        return _temporal_target_mean(self.after_targets)

    @property
    def repair_gain(self) -> float:
        return self.after_score - self.before_score

    @property
    def all_success(self) -> bool:
        return all(target.temporal_score == 1.0 for target in self.after_targets)

    @property
    def delayed_edge_rate(self) -> float:
        return _safe_div(
            sum(1 for target in self.after_targets if target.delayed_edge_learned),
            len(self.after_targets),
        )

    @property
    def followup_misattribution_rate(self) -> float:
        return _safe_div(
            sum(1 for target in self.after_targets if target.followup_misattribution),
            len(self.after_targets),
        )

    @property
    def shifted_before_score(self) -> float:
        return _schema_temporal_target_mean(
            self.before_targets, self.shifted_roles, schema_shifted=True
        )

    @property
    def shifted_after_score(self) -> float:
        return _schema_temporal_target_mean(
            self.after_targets, self.shifted_roles, schema_shifted=True
        )

    @property
    def stable_before_score(self) -> float:
        return _schema_temporal_target_mean(
            self.before_targets, self.shifted_roles, schema_shifted=False
        )

    @property
    def stable_after_score(self) -> float:
        return _schema_temporal_target_mean(
            self.after_targets, self.shifted_roles, schema_shifted=False
        )

    @property
    def shifted_count(self) -> int:
        return _schema_temporal_target_count(
            self.after_targets, self.shifted_roles, schema_shifted=True
        )

    @property
    def stable_count(self) -> int:
        return _schema_temporal_target_count(
            self.after_targets, self.shifted_roles, schema_shifted=False
        )

    @property
    def level6_repair_score(self) -> float:
        if self.shifted_count == 0 or self.stable_count == 0:
            return 0.0
        if not self.all_success:
            return 0.0
        if self.followup_misattribution_rate > 0.0:
            return 0.0
        if self.shifted_after_score <= self.shifted_before_score:
            return 0.0
        if self.stable_after_score < 1.0:
            return 0.0
        return 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "source_world": self.exploration.world_name,
            "target_world": self.target_world_name,
            "agent": self.exploration.agent_name,
            "source_steps": len(self.exploration.transitions),
            "target_steps": self.target_steps,
            "transfer_mode": self.transfer_mode,
            "shifted_roles": sorted(self.shifted_roles),
            "before_score": self.before_score,
            "after_score": self.after_score,
            "repair_gain": self.repair_gain,
            "all_success": self.all_success,
            "delayed_edge_rate": self.delayed_edge_rate,
            "followup_misattribution_rate": self.followup_misattribution_rate,
            "shifted_before_score": self.shifted_before_score,
            "shifted_after_score": self.shifted_after_score,
            "stable_before_score": self.stable_before_score,
            "stable_after_score": self.stable_after_score,
            "level6_repair_score": self.level6_repair_score,
            "targets": {
                "before": [target.to_dict() for target in self.before_targets],
                "after": [target.to_dict() for target in self.after_targets],
            },
        }


@dataclass(frozen=True)
class CounterfactualTestCase:
    factual_step: int
    factual_action: str
    factual_before: dict[str, bool]
    factual_after: dict[str, bool]
    edit_type: str
    edit_name: str
    counterfactual_action: str
    counterfactual_before: dict[str, bool]
    expected_after: dict[str, bool]
    predicted_after: dict[str, bool]

    @property
    def exact_match(self) -> bool:
        return self.expected_after == self.predicted_after

    @property
    def state_accuracy(self) -> float:
        keys = set(self.expected_after) | set(self.predicted_after)
        if not keys:
            return 1.0
        matches = sum(
            self.expected_after.get(key) == self.predicted_after.get(key)
            for key in keys
        )
        return matches / len(keys)

    @property
    def expected_delta_from_fact(self) -> frozenset[str]:
        return _changed_keys(self.factual_after, self.expected_after)

    @property
    def predicted_delta_from_fact(self) -> frozenset[str]:
        return _changed_keys(self.factual_after, self.predicted_after)

    @property
    def delta_jaccard(self) -> float:
        expected = self.expected_delta_from_fact
        predicted = self.predicted_delta_from_fact
        union = expected | predicted
        if not union:
            return 1.0
        return len(expected & predicted) / len(union)


@dataclass(frozen=True)
class CounterfactualTestResult:
    exploration: EpisodeResult
    cases: tuple[CounterfactualTestCase, ...]
    exact_match_rate: float
    mean_state_accuracy: float
    mean_delta_jaccard: float
    delta_precision: float
    delta_recall: float
    delta_f1: float

    def to_dict(self) -> dict[str, object]:
        return {
            "world": self.exploration.world_name,
            "agent": self.exploration.agent_name,
            "explore_steps": len(self.exploration.transitions),
            "exploration_score": self.exploration.to_dict()["score"],
            "counterfactual_cases": len(self.cases),
            "counterfactual_test": {
                "exact_match_rate": self.exact_match_rate,
                "mean_state_accuracy": self.mean_state_accuracy,
                "mean_delta_jaccard": self.mean_delta_jaccard,
                "delta_precision": self.delta_precision,
                "delta_recall": self.delta_recall,
                "delta_f1": self.delta_f1,
            },
            "cases": [
                {
                    "factual_step": case.factual_step,
                    "factual_action": case.factual_action,
                    "factual_before": case.factual_before,
                    "factual_after": case.factual_after,
                    "edit_type": case.edit_type,
                    "edit_name": case.edit_name,
                    "counterfactual_action": case.counterfactual_action,
                    "counterfactual_before": case.counterfactual_before,
                    "expected_after": case.expected_after,
                    "predicted_after": case.predicted_after,
                    "exact_match": case.exact_match,
                    "state_accuracy": case.state_accuracy,
                    "expected_delta_from_fact": sorted(case.expected_delta_from_fact),
                    "predicted_delta_from_fact": sorted(case.predicted_delta_from_fact),
                    "delta_jaccard": case.delta_jaccard,
                }
                for case in self.cases
            ],
        }


def run_episode(
    world: World, agent: Agent, steps: int = 20, seed: int | None = None
) -> EpisodeResult:
    observation = world.reset(seed=seed)
    agent.reset(world.actions())
    history: list[Transition] = []

    for step in range(1, steps + 1):
        decision = agent.choose_action(observation, tuple(history))
        before = world.observe()
        after = world.step(decision.action)
        transition = Transition(
            step=step,
            action=decision.action,
            before=before,
            after=after,
            prediction=decision.prediction,
            hypothesis=decision.hypothesis,
        )
        agent.observe_transition(transition)
        history.append(transition)
        observation = after

    discovered = agent.discovered_edges()
    true_edges = world.true_edges()
    score = score_edges(discovered, true_edges, tuple(history))
    return EpisodeResult(
        world_name=world.name,
        agent_name=agent.name,
        transitions=tuple(history),
        discovered_edges=discovered,
        true_edges=true_edges,
        condition_hints=_condition_hints_for_agent(agent),
        score=score,
    )


def run_reward_episode(
    world: World, agent: Agent, steps: int = 20, seed: int | None = None
) -> RewardEpisodeResult:
    exploration = run_episode(world, agent, steps=steps, seed=seed)
    rewards = [task_utility(transition.after) for transition in exploration.transitions]
    total_return = sum(rewards)
    mean_return = _safe_div(total_return, len(rewards))
    final_utility = rewards[-1] if rewards else 0.0
    return RewardEpisodeResult(
        exploration=exploration,
        total_return=total_return,
        mean_return=mean_return,
        final_utility=final_utility,
    )


def run_level4_disambiguation_test(
    world: World,
    agent: Agent,
    steps: int = 8,
    seed: int | None = None,
    target_action: str | None = None,
    target_variable: str | None = None,
    effect_value: bool | None = None,
    candidate_gates: tuple[tuple[str, bool], ...] | None = None,
) -> Level4DisambiguationResult:
    """Evaluate active selection of interventions that separate causal gates."""

    (
        target_action,
        target_variable,
        effect_value,
        candidate_gates,
    ) = _resolve_level4_spec(
        world=world,
        target_action=target_action,
        target_variable=target_variable,
        effect_value=effect_value,
        candidate_gates=candidate_gates,
    )
    initial_state = world.reset(seed=seed)
    case_templates = _gate_case_templates(
        world=world,
        base_state=initial_state,
        target_action=target_action,
        target_variable=target_variable,
        effect_value=effect_value,
        candidate_gates=candidate_gates,
    )

    observation = world.reset(seed=seed)
    agent.reset(world.actions())
    history: list[Transition] = []
    first_disambiguating_step: int | None = None
    first_correct_step: int | None = None
    disambiguating_tests = 0

    for step in range(1, steps + 1):
        decision = agent.choose_action(observation, tuple(history))
        before = world.observe()
        after = world.step(decision.action)
        transition = Transition(
            step=step,
            action=decision.action,
            before=before,
            after=after,
            prediction=decision.prediction,
            hypothesis=decision.hypothesis,
        )
        agent.observe_transition(transition)
        history.append(transition)
        observation = after

        if _is_disambiguating_transition(
            transition,
            target_action=target_action,
            target_variable=target_variable,
            effect_value=effect_value,
            candidate_gates=candidate_gates,
        ):
            disambiguating_tests += 1
            if first_disambiguating_step is None:
                first_disambiguating_step = step

        if first_correct_step is None and _gate_predictions_correct(
            agent=agent,
            cases=case_templates,
            target_action=target_action,
            target_variable=target_variable,
        ):
            first_correct_step = step

    final_cases = _with_gate_predictions(
        agent=agent,
        cases=case_templates,
        target_action=target_action,
        target_variable=target_variable,
    )
    final_correct = all(case.correct for case in final_cases)
    level4_score = 0.0
    if first_correct_step is not None:
        level4_score = max(0.0, 1.0 - ((first_correct_step - 1) / steps))

    transitions = tuple(history)
    discovered = agent.discovered_edges()
    true_edges = world.true_edges()
    exploration = EpisodeResult(
        world_name=world.name,
        agent_name=agent.name,
        transitions=transitions,
        discovered_edges=discovered,
        true_edges=true_edges,
        condition_hints=_condition_hints_for_agent(agent),
        score=score_edges(discovered, true_edges, transitions),
    )
    return Level4DisambiguationResult(
        exploration=exploration,
        target_action=target_action,
        target_variable=target_variable,
        candidate_gates=candidate_gates,
        cases=final_cases,
        first_disambiguating_step=first_disambiguating_step,
        first_correct_step=first_correct_step,
        disambiguating_tests=disambiguating_tests,
        final_correct=final_correct,
        level4_score=level4_score,
    )


def run_temporal_credit_test(
    world: World,
    agent: Agent,
    steps: int = 8,
    seed: int | None = None,
    target_action: str | None = None,
    target_variable: str | None = None,
    effect_value: bool | None = None,
    followup_action: str | None = None,
    delay_steps: int | None = None,
) -> TemporalCreditResult:
    """Evaluate whether delayed effects are credited to their initiating action."""

    (
        target_action,
        target_variable,
        effect_value,
        followup_action,
        delay_steps,
    ) = _resolve_temporal_spec(
        world=world,
        target_action=target_action,
        target_variable=target_variable,
        effect_value=effect_value,
        followup_action=followup_action,
        delay_steps=delay_steps,
    )
    initial_state = world.reset(seed=seed)
    case_templates = _temporal_case_templates(
        world=world,
        base_state=initial_state,
        target_action=target_action,
        target_variable=target_variable,
        effect_value=effect_value,
        followup_action=followup_action,
        delay_steps=delay_steps,
    )

    observation = world.reset(seed=seed)
    agent.reset(world.actions())
    history: list[Transition] = []
    first_correct_step: int | None = None

    for step in range(1, steps + 1):
        decision = agent.choose_action(observation, tuple(history))
        before = world.observe()
        after = world.step(decision.action)
        transition = Transition(
            step=step,
            action=decision.action,
            before=before,
            after=after,
            prediction=decision.prediction,
            hypothesis=decision.hypothesis,
        )
        agent.observe_transition(transition)
        history.append(transition)
        observation = after

        if first_correct_step is None and _temporal_predictions_correct(
            agent=agent,
            cases=case_templates,
            target_variable=target_variable,
        ):
            first_correct_step = step

    final_cases = _with_temporal_predictions(
        agent=agent,
        cases=case_templates,
        target_variable=target_variable,
    )
    final_correct = all(case.correct for case in final_cases)
    temporal_score = 0.0
    if first_correct_step is not None:
        temporal_score = max(0.0, 1.0 - ((first_correct_step - 1) / steps))

    transitions = tuple(history)
    discovered = agent.discovered_edges()
    true_edges = world.true_edges()
    exploration = EpisodeResult(
        world_name=world.name,
        agent_name=agent.name,
        transitions=transitions,
        discovered_edges=discovered,
        true_edges=true_edges,
        condition_hints=_condition_hints_for_agent(agent),
        score=score_edges(discovered, true_edges, transitions),
    )
    return TemporalCreditResult(
        exploration=exploration,
        target_action=target_action,
        target_variable=target_variable,
        followup_action=followup_action,
        delay_steps=delay_steps,
        cases=final_cases,
        first_correct_step=first_correct_step,
        final_correct=final_correct,
        delayed_edge_learned=(target_action, target_variable) in discovered,
        followup_misattribution=(followup_action, target_variable) in discovered,
        temporal_score=temporal_score,
    )


def run_temporal_transfer_test(
    source_world: World,
    target_world: World,
    agent: Agent,
    explore_steps: int = 8,
    seed: int | None = None,
) -> TemporalTransferResult:
    """Explore a source world, then test delayed-effect credit in a target world."""

    exploration = run_episode(source_world, agent, steps=explore_steps, seed=seed)
    return _evaluate_temporal_transfer(
        source_world=source_world,
        target_world=target_world,
        agent=agent,
        exploration=exploration,
        seed=seed,
    )


def run_temporal_adaptation_test(
    source_world: World,
    target_world: World,
    agent: Agent,
    explore_steps: int = 8,
    adapt_steps: int = 4,
    seed: int | None = None,
    adaptation_mode: str = "structural-prior",
) -> TemporalAdaptationResult:
    """Test whether a delayed causal mechanism can be repaired after transfer."""

    exploration = run_episode(source_world, agent, steps=explore_steps, seed=seed)
    before = _evaluate_temporal_transfer(
        source_world=source_world,
        target_world=target_world,
        agent=agent,
        exploration=exploration,
        seed=seed,
    )

    if adaptation_mode == "fresh":
        agent.reset(target_world.actions())
    elif adaptation_mode == "structural-prior":
        starter = getattr(agent, "start_new_environment", None)
        if starter is None:
            agent.reset(target_world.actions())
        else:
            starter(keep_priors=True)
    elif adaptation_mode != "continue":
        raise ValueError(
            "adaptation_mode must be 'continue', 'fresh', or 'structural-prior'"
        )

    _run_episode_without_agent_reset(
        target_world,
        agent,
        steps=adapt_steps,
        seed=seed,
    )
    after = _evaluate_temporal_transfer(
        source_world=source_world,
        target_world=target_world,
        agent=agent,
        exploration=exploration,
        seed=seed,
    )
    return TemporalAdaptationResult(
        before_adaptation=before,
        after_adaptation=after,
        adaptation_steps=adapt_steps,
        adaptation_mode=adaptation_mode,
    )


def run_temporal_selective_adaptation_test(
    source_world: World,
    target_world: World,
    agent: Agent,
    explore_steps: int = 9,
    adapt_steps: int = 6,
    seed: int | None = None,
    adaptation_mode: str = "structural-prior",
) -> TemporalSelectiveAdaptationResult:
    """Test selective repair when only some delayed mechanisms change."""

    exploration = run_episode(source_world, agent, steps=explore_steps, seed=seed)
    before_targets = _evaluate_temporal_targets(
        source_world=source_world,
        target_world=target_world,
        agent=agent,
        seed=seed,
    )

    if adaptation_mode == "fresh":
        agent.reset(target_world.actions())
    elif adaptation_mode == "structural-prior":
        starter = getattr(agent, "start_new_environment", None)
        if starter is None:
            agent.reset(target_world.actions())
        else:
            starter(keep_priors=True)
    elif adaptation_mode != "continue":
        raise ValueError(
            "adaptation_mode must be 'continue', 'fresh', or 'structural-prior'"
        )

    _run_episode_without_agent_reset(
        target_world,
        agent,
        steps=adapt_steps,
        seed=seed,
    )
    after_targets = _evaluate_temporal_targets(
        source_world=source_world,
        target_world=target_world,
        agent=agent,
        seed=seed,
    )
    return TemporalSelectiveAdaptationResult(
        exploration=exploration,
        target_world_name=target_world.name,
        before_targets=before_targets,
        after_targets=after_targets,
        adaptation_steps=adapt_steps,
        adaptation_mode=adaptation_mode,
    )


def run_level6_schema_transfer_test(
    source_world: World,
    target_world: World,
    agent: Agent,
    explore_steps: int = 9,
    target_steps: int = 6,
    seed: int | None = None,
    transfer_mode: str = "schema-prior",
) -> Level6SchemaTransferResult:
    """Test few-shot causal transfer under a renamed action/observation schema."""

    exploration = run_episode(source_world, agent, steps=explore_steps, seed=seed)
    target_initial = target_world.reset(seed=seed)

    if transfer_mode == "fresh":
        agent.reset(target_world.actions())
    elif transfer_mode == "schema-prior":
        starter = getattr(agent, "start_new_environment_with_schema", None)
        if starter is None:
            agent.reset(target_world.actions())
        else:
            starter(
                target_world.actions(),
                target_initial,
                keep_priors=True,
            )
    elif transfer_mode != "continue":
        raise ValueError("transfer_mode must be 'schema-prior', 'fresh', or 'continue'")

    _run_episode_without_agent_reset(
        target_world,
        agent,
        steps=target_steps,
        seed=seed,
    )
    targets = _evaluate_temporal_targets(
        source_world=target_world,
        target_world=target_world,
        agent=agent,
        seed=seed,
    )
    return Level6SchemaTransferResult(
        exploration=exploration,
        target_world_name=target_world.name,
        targets=targets,
        target_steps=target_steps,
        transfer_mode=transfer_mode,
    )


def run_level6_schema_repair_test(
    source_world: World,
    target_world: World,
    agent: Agent,
    explore_steps: int = 9,
    target_steps: int = 6,
    seed: int | None = None,
    transfer_mode: str = "schema-prior",
) -> Level6SchemaRepairResult:
    """Test schema transfer plus selective repair of a shifted mechanism."""

    exploration = run_episode(source_world, agent, steps=explore_steps, seed=seed)
    target_initial = target_world.reset(seed=seed)

    if transfer_mode == "fresh":
        agent.reset(target_world.actions())
    elif transfer_mode == "schema-prior":
        starter = getattr(agent, "start_new_environment_with_schema", None)
        if starter is None:
            agent.reset(target_world.actions())
        else:
            starter(
                target_world.actions(),
                target_initial,
                keep_priors=True,
            )
    elif transfer_mode != "continue":
        raise ValueError("transfer_mode must be 'schema-prior', 'fresh', or 'continue'")

    shifted_roles = _schema_shifted_temporal_roles(source_world, target_world)
    before_targets = _evaluate_temporal_targets(
        source_world=target_world,
        target_world=target_world,
        agent=agent,
        seed=seed,
    )
    _run_episode_without_agent_reset(
        target_world,
        agent,
        steps=target_steps,
        seed=seed,
    )
    after_targets = _evaluate_temporal_targets(
        source_world=target_world,
        target_world=target_world,
        agent=agent,
        seed=seed,
    )
    return Level6SchemaRepairResult(
        exploration=exploration,
        target_world_name=target_world.name,
        before_targets=before_targets,
        after_targets=after_targets,
        target_steps=target_steps,
        transfer_mode=transfer_mode,
        shifted_roles=shifted_roles,
    )


def run_level6_active_diagnostic_repair_test(
    source_world: World,
    target_world: World,
    agent: Agent,
    explore_steps: int = 12,
    repair_steps: int = 1,
    seed: int | None = None,
    transfer_mode: str = "schema-prior",
) -> Level6SchemaRepairResult:
    """Test whether prediction error focuses repair on the shifted mechanism."""

    exploration = run_episode(source_world, agent, steps=explore_steps, seed=seed)
    target_initial = target_world.reset(seed=seed)

    if transfer_mode == "fresh":
        agent.reset(target_world.actions())
    elif transfer_mode == "schema-prior":
        starter = getattr(agent, "start_new_environment_with_schema", None)
        if starter is None:
            agent.reset(target_world.actions())
        else:
            starter(
                target_world.actions(),
                target_initial,
                keep_priors=True,
            )
    elif transfer_mode != "continue":
        raise ValueError("transfer_mode must be 'schema-prior', 'fresh', or 'continue'")

    shifted_roles = _schema_shifted_temporal_roles(source_world, target_world)
    before_targets = _evaluate_temporal_targets(
        source_world=target_world,
        target_world=target_world,
        agent=agent,
        seed=seed,
    )

    observation = target_world.reset(seed=seed)
    diagnostic_actions = _diagnostic_temporal_script(
        source_world=source_world,
        target_world=target_world,
        shifted_roles=shifted_roles,
    )
    history = _run_scripted_actions_without_agent_reset(
        target_world,
        agent,
        actions=diagnostic_actions,
        initial_observation=observation,
    )
    if history:
        observation = history[-1].after

    repair_history = _continue_episode_without_agent_reset(
        target_world,
        agent,
        steps=repair_steps,
        seed=seed,
        initial_observation=observation,
        initial_history=history,
    )
    after_targets = _evaluate_temporal_targets(
        source_world=target_world,
        target_world=target_world,
        agent=agent,
        seed=seed,
    )
    return Level6SchemaRepairResult(
        exploration=exploration,
        target_world_name=target_world.name,
        before_targets=before_targets,
        after_targets=after_targets,
        target_steps=len(history) + len(repair_history),
        transfer_mode=f"{transfer_mode}-active-diagnostic",
        shifted_roles=shifted_roles,
    )


def run_counterfactual_test(
    world: World,
    agent: Agent,
    explore_steps: int = 20,
    seed: int | None = None,
    case_source: str = "history",
    include_state_edits: bool = True,
    include_action_edits: bool = True,
) -> CounterfactualTestResult:
    exploration = run_episode(world, agent, steps=explore_steps, seed=seed)
    variables = sorted(world.observe())
    actions = [action.name for action in world.actions()]
    cases: list[CounterfactualTestCase] = []

    for transition in _counterfactual_sources(
        world, exploration, variables, actions, case_source
    ):
        if include_state_edits:
            for variable in variables:
                counterfactual_before = dict(transition.before)
                counterfactual_before[variable] = not counterfactual_before[variable]
                cases.append(
                    _counterfactual_case(
                        world=world,
                        agent=agent,
                        transition=transition,
                        edit_type="state",
                        edit_name=variable,
                        counterfactual_action=transition.action,
                        counterfactual_before=counterfactual_before,
                    )
                )

        if include_action_edits:
            for action in actions:
                if action == transition.action:
                    continue
                cases.append(
                    _counterfactual_case(
                        world=world,
                        agent=agent,
                        transition=transition,
                        edit_type="action",
                        edit_name=action,
                        counterfactual_action=action,
                        counterfactual_before=transition.before,
                    )
                )

    exact_match_rate = _safe_div(
        sum(case.exact_match for case in cases), len(cases)
    )
    mean_state_accuracy = _safe_div(
        sum(case.state_accuracy for case in cases), len(cases)
    )
    mean_delta_jaccard = _safe_div(
        sum(case.delta_jaccard for case in cases), len(cases)
    )
    delta_precision, delta_recall, delta_f1 = _micro_delta_scores(tuple(cases))
    return CounterfactualTestResult(
        exploration=exploration,
        cases=tuple(cases),
        exact_match_rate=exact_match_rate,
        mean_state_accuracy=mean_state_accuracy,
        mean_delta_jaccard=mean_delta_jaccard,
        delta_precision=delta_precision,
        delta_recall=delta_recall,
        delta_f1=delta_f1,
    )


def _counterfactual_sources(
    world: World,
    exploration: EpisodeResult,
    variables: list[str],
    actions: list[str],
    case_source: str,
) -> tuple[Transition, ...]:
    if case_source == "history":
        return exploration.transitions
    if case_source != "all-states":
        raise ValueError("case_source must be 'history' or 'all-states'")

    transitions: list[Transition] = []
    step = 1
    for state in all_boolean_states(variables):
        for action in actions:
            after = _expected_after(world, dict(state), action)
            transitions.append(
                Transition(
                    step=step,
                    action=action,
                    before=dict(state),
                    after=after,
                    prediction=frozenset(),
                    hypothesis="counterfactual source",
                )
            )
            step += 1
    return tuple(transitions)


def format_counterfactual_test(result: CounterfactualTestResult) -> str:
    exploration_score = result.exploration.score
    lines = [
        f"World: {result.exploration.world_name}",
        f"Agent: {result.exploration.agent_name}",
        f"Explore steps: {len(result.exploration.transitions)}",
        (
            "Exploration graph: "
            f"precision={exploration_score.precision:.2f}, "
            f"recall={exploration_score.recall:.2f}, "
            f"f1={exploration_score.f1:.2f}"
        ),
        (
            "Counterfactuals: "
            f"cases={len(result.cases)}, "
            f"exact={result.exact_match_rate:.2f}, "
            f"state-acc={result.mean_state_accuracy:.2f}, "
            f"delta-jaccard={result.mean_delta_jaccard:.2f}, "
            f"delta-f1={result.delta_f1:.2f}"
        ),
        "",
        "Failed cases:",
    ]
    failed = [case for case in result.cases if not case.exact_match]
    for case in failed[:12]:
        before = _format_state(case.counterfactual_before)
        expected = _format_state(case.expected_after)
        predicted = _format_state(case.predicted_after)
        lines.append(
            f"  - step {case.factual_step}, {case.edit_type}:{case.edit_name}, "
            f"{case.counterfactual_action} @ {before}: "
            f"expected [{expected}], predicted [{predicted}]"
        )
    if len(failed) > 12:
        lines.append(f"  ... {len(failed) - 12} more")
    if not failed:
        lines.append("  none")
    return "\n".join(lines)


def run_intervention_test(
    world: World,
    agent: Agent,
    explore_steps: int = 20,
    seed: int | None = None,
    include_seen: bool = False,
) -> InterventionTestResult:
    exploration = run_episode(world, agent, steps=explore_steps, seed=seed)
    variables = sorted(world.observe())
    actions = [action.name for action in world.actions()]
    seen = {
        (state_signature(transition.before), transition.action)
        for transition in exploration.transitions
    }

    cases: list[InterventionTestCase] = []
    for state in all_boolean_states(variables):
        signature = state_signature(state)
        for action in actions:
            seen_case = (signature, action) in seen
            if seen_case and not include_seen:
                continue
            expected = _expected_changes(world, state, action)
            predicted = agent.predict(action, state)
            cases.append(
                InterventionTestCase(
                    action=action,
                    state=dict(state),
                    expected=expected,
                    predicted=predicted,
                    seen_during_exploration=seen_case,
                    mechanism_shifted=False,
                )
            )

    exact_match_rate = _safe_div(
        sum(case.exact_match for case in cases), len(cases)
    )
    mean_jaccard = _safe_div(sum(case.jaccard for case in cases), len(cases))
    precision, recall, f1 = _micro_prediction_scores(tuple(cases))
    return InterventionTestResult(
        exploration=exploration,
        cases=tuple(cases),
        exact_match_rate=exact_match_rate,
        prediction_precision=precision,
        prediction_recall=recall,
        prediction_f1=f1,
        mean_jaccard=mean_jaccard,
    )


def run_transfer_intervention_test(
    source_world: World,
    target_world: World,
    agent: Agent,
    explore_steps: int = 20,
    seed: int | None = None,
    include_seen: bool = False,
) -> TransferInterventionTestResult:
    exploration = run_episode(source_world, agent, steps=explore_steps, seed=seed)
    return _evaluate_transfer_intervention(
        source_world=source_world,
        target_world=target_world,
        agent=agent,
        exploration=exploration,
        seed=seed,
        include_seen=include_seen,
    )


def run_transfer_adaptation_test(
    source_world: World,
    target_world: World,
    agent: Agent,
    explore_steps: int = 20,
    adapt_steps: int = 8,
    seed: int | None = None,
    adaptation_mode: str = "structural-prior",
    include_seen: bool = False,
) -> TransferAdaptationTestResult:
    exploration = run_episode(source_world, agent, steps=explore_steps, seed=seed)
    before = _evaluate_transfer_intervention(
        source_world=source_world,
        target_world=target_world,
        agent=agent,
        exploration=exploration,
        seed=seed,
        include_seen=include_seen,
    )

    if adaptation_mode == "fresh":
        agent.reset(target_world.actions())
    elif adaptation_mode == "structural-prior":
        starter = getattr(agent, "start_new_environment", None)
        if starter is None:
            agent.reset(target_world.actions())
        else:
            starter(keep_priors=True)
    elif adaptation_mode != "continue":
        raise ValueError(
            "adaptation_mode must be 'continue', 'fresh', or 'structural-prior'"
        )

    _run_episode_without_agent_reset(
        target_world, agent, steps=adapt_steps, seed=seed
    )
    after = _evaluate_transfer_intervention(
        source_world=source_world,
        target_world=target_world,
        agent=agent,
        exploration=exploration,
        seed=seed,
        include_seen=include_seen,
    )
    return TransferAdaptationTestResult(
        before_adaptation=before,
        after_adaptation=after,
        adaptation_steps=adapt_steps,
        adaptation_mode=adaptation_mode,
    )


def run_level3_test(
    source_world: World,
    stable_target_world: World,
    repair_target_world: World,
    agent: Agent,
    explore_steps: int = 20,
    adapt_steps: int = 12,
    seed: int | None = None,
    case_source: str = "all-states",
    adaptation_mode: str = "structural-prior",
) -> Level3TestResult:
    """Evaluate the Level 3 ladder: counterfactual, transfer, and repair."""
    counterfactual = run_counterfactual_test(
        source_world,
        agent,
        explore_steps=explore_steps,
        seed=seed,
        case_source=case_source,
    )
    stable_transfer = run_transfer_intervention_test(
        source_world,
        stable_target_world,
        agent,
        explore_steps=explore_steps,
        seed=seed,
    )
    repair = run_transfer_adaptation_test(
        source_world,
        repair_target_world,
        agent,
        explore_steps=explore_steps,
        adapt_steps=adapt_steps,
        seed=seed,
        adaptation_mode=adaptation_mode,
    )
    return Level3TestResult(
        counterfactual=counterfactual,
        stable_transfer=stable_transfer,
        repair=repair,
    )


def _evaluate_temporal_transfer(
    source_world: World,
    target_world: World,
    agent: Agent,
    exploration: EpisodeResult,
    seed: int | None,
) -> TemporalTransferResult:
    target = _evaluate_temporal_target(
        source_world=source_world,
        target_world=target_world,
        agent=agent,
        seed=seed,
        spec=_resolve_temporal_spec(
            world=target_world,
            target_action=None,
            target_variable=None,
            effect_value=None,
            followup_action=None,
            delay_steps=None,
        ),
    )
    return TemporalTransferResult(
        exploration=exploration,
        target_world_name=target.target_world_name,
        target_action=target.target_action,
        target_variable=target.target_variable,
        followup_action=target.followup_action,
        delay_steps=target.delay_steps,
        cases=target.cases,
        final_correct=target.final_correct,
        delayed_edge_learned=target.delayed_edge_learned,
        followup_misattribution=target.followup_misattribution,
        mechanism_shifted=target.mechanism_shifted,
        temporal_transfer_score=target.temporal_score,
    )


def _evaluate_temporal_targets(
    source_world: World,
    target_world: World,
    agent: Agent,
    seed: int | None,
) -> tuple[TemporalTargetResult, ...]:
    return tuple(
        _evaluate_temporal_target(
            source_world=source_world,
            target_world=target_world,
            agent=agent,
            seed=seed,
            spec=spec,
        )
        for spec in _resolve_temporal_specs(target_world)
    )


def _evaluate_temporal_target(
    source_world: World,
    target_world: World,
    agent: Agent,
    seed: int | None,
    spec: tuple[str, str, bool, str, int],
) -> TemporalTargetResult:
    (
        target_action,
        target_variable,
        effect_value,
        followup_action,
        delay_steps,
    ) = spec
    target_initial = target_world.reset(seed=seed)
    case_templates = _temporal_case_templates(
        world=target_world,
        base_state=target_initial,
        target_action=target_action,
        target_variable=target_variable,
        effect_value=effect_value,
        followup_action=followup_action,
        delay_steps=delay_steps,
    )
    final_cases = _with_temporal_predictions(
        agent=agent,
        cases=case_templates,
        target_variable=target_variable,
    )
    final_correct = all(case.correct for case in final_cases)
    discovered = agent.discovered_edges()
    delayed_edge_learned = (target_action, target_variable) in discovered
    followup_misattribution = (followup_action, target_variable) in discovered
    temporal_transfer_score = 1.0 if (
        final_correct and delayed_edge_learned and not followup_misattribution
    ) else 0.0

    mechanism_shifted = _temporal_mechanism_shifted(
        source_world=source_world,
        target_world=target_world,
        cases=case_templates,
        target_action=target_action,
        target_variable=target_variable,
        effect_value=effect_value,
        followup_action=followup_action,
        delay_steps=delay_steps,
    )
    return TemporalTargetResult(
        target_world_name=target_world.name,
        target_action=target_action,
        target_variable=target_variable,
        followup_action=followup_action,
        delay_steps=delay_steps,
        cases=final_cases,
        final_correct=final_correct,
        delayed_edge_learned=delayed_edge_learned,
        followup_misattribution=followup_misattribution,
        mechanism_shifted=mechanism_shifted,
        temporal_score=temporal_transfer_score,
    )


def _evaluate_transfer_intervention(
    source_world: World,
    target_world: World,
    agent: Agent,
    exploration: EpisodeResult,
    seed: int | None,
    include_seen: bool,
) -> TransferInterventionTestResult:
    target_world.reset(seed=seed)
    variables = sorted(target_world.observe())
    actions = [action.name for action in target_world.actions()]
    seen = {
        (state_signature(transition.before), transition.action)
        for transition in exploration.transitions
    }

    cases: list[InterventionTestCase] = []
    for state in all_boolean_states(variables):
        signature = state_signature(state)
        for action in actions:
            seen_case = (signature, action) in seen
            if seen_case and not include_seen:
                continue
            source_expected = _expected_changes(source_world, state, action)
            expected = _expected_changes(target_world, state, action)
            predicted = agent.predict(action, state)
            cases.append(
                InterventionTestCase(
                    action=action,
                    state=dict(state),
                    expected=expected,
                    predicted=predicted,
                    seen_during_exploration=seen_case,
                    mechanism_shifted=source_expected != expected,
                )
            )

    exact_match_rate = _safe_div(
        sum(case.exact_match for case in cases), len(cases)
    )
    mean_jaccard = _safe_div(sum(case.jaccard for case in cases), len(cases))
    precision, recall, f1 = _micro_prediction_scores(tuple(cases))
    shifted_cases = tuple(case for case in cases if case.mechanism_shifted)
    shifted_exact = _safe_div(
        sum(case.exact_match for case in shifted_cases), len(shifted_cases)
    )
    _, _, shifted_f1 = _micro_prediction_scores(shifted_cases)
    return TransferInterventionTestResult(
        exploration=exploration,
        target_world_name=target_world.name,
        cases=tuple(cases),
        exact_match_rate=exact_match_rate,
        prediction_precision=precision,
        prediction_recall=recall,
        prediction_f1=f1,
        mean_jaccard=mean_jaccard,
        shifted_cases=shifted_cases,
        shifted_exact_match_rate=shifted_exact,
        shifted_prediction_f1=shifted_f1,
    )


def _run_episode_without_agent_reset(
    world: World, agent: Agent, steps: int, seed: int | None = None
) -> tuple[Transition, ...]:
    observation = world.reset(seed=seed)
    history: list[Transition] = []
    for step in range(1, steps + 1):
        decision = agent.choose_action(observation, tuple(history))
        before = world.observe()
        after = world.step(decision.action)
        transition = Transition(
            step=step,
            action=decision.action,
            before=before,
            after=after,
            prediction=decision.prediction,
            hypothesis=decision.hypothesis,
        )
        agent.observe_transition(transition)
        history.append(transition)
        observation = after
    return tuple(history)


def _continue_episode_without_agent_reset(
    world: World,
    agent: Agent,
    steps: int,
    seed: int | None = None,
    initial_observation: dict[str, bool] | None = None,
    initial_history: tuple[Transition, ...] = (),
) -> tuple[Transition, ...]:
    observation = initial_observation
    if observation is None:
        observation = world.reset(seed=seed)
    history = list(initial_history)
    new_transitions: list[Transition] = []
    for _ in range(steps):
        step = len(history) + 1
        decision = agent.choose_action(observation, tuple(history))
        before = world.observe()
        after = world.step(decision.action)
        transition = Transition(
            step=step,
            action=decision.action,
            before=before,
            after=after,
            prediction=decision.prediction,
            hypothesis=decision.hypothesis,
        )
        agent.observe_transition(transition)
        history.append(transition)
        new_transitions.append(transition)
        observation = after
    return tuple(new_transitions)


def _run_scripted_actions_without_agent_reset(
    world: World,
    agent: Agent,
    actions: tuple[str, ...],
    initial_observation: dict[str, bool],
) -> tuple[Transition, ...]:
    observation = initial_observation
    history: list[Transition] = []
    for step, action in enumerate(actions, start=1):
        before = world.observe()
        after = world.step(action)
        transition = Transition(
            step=step,
            action=action,
            before=before,
            after=after,
            prediction=agent.predict(action, observation),
            hypothesis="Scripted diagnostic exposure.",
        )
        agent.observe_transition(transition)
        history.append(transition)
        observation = after
    return tuple(history)


def _diagnostic_temporal_script(
    source_world: World,
    target_world: World,
    shifted_roles: frozenset[str],
) -> tuple[str, ...]:
    source_delays = _temporal_role_delays(source_world)
    fallback: tuple[str, str, int] | None = None
    for action, variable, _, followup_action, target_delay in _resolve_temporal_specs(
        target_world
    ):
        role = _schema_temporal_role(action, variable)
        source_delay = source_delays.get(role, 1)
        if role in shifted_roles:
            return (action,) + (followup_action,) * max(1, source_delay)
        if fallback is None:
            fallback = (action, followup_action, target_delay)
    if fallback is None:
        return ()
    action, followup_action, _ = fallback
    return (action, followup_action)


def format_transfer_intervention_test(
    result: TransferInterventionTestResult,
) -> str:
    exploration_score = result.exploration.score
    lines = [
        f"Source world: {result.exploration.world_name}",
        f"Target world: {result.target_world_name}",
        f"Agent: {result.exploration.agent_name}",
        f"Explore steps: {len(result.exploration.transitions)}",
        (
            "Source graph: "
            f"precision={exploration_score.precision:.2f}, "
            f"recall={exploration_score.recall:.2f}, "
            f"f1={exploration_score.f1:.2f}"
        ),
        (
            "Transfer interventions: "
            f"cases={len(result.cases)}, "
            f"exact={result.exact_match_rate:.2f}, "
            f"jaccard={result.mean_jaccard:.2f}, "
            f"f1={result.prediction_f1:.2f}"
        ),
        (
            "Shift-only: "
            f"cases={len(result.shifted_cases)}, "
            f"exact={result.shifted_exact_match_rate:.2f}, "
            f"f1={result.shifted_prediction_f1:.2f}"
        ),
        "",
        "Failed cases:",
    ]
    failed = [case for case in result.cases if not case.exact_match]
    for case in failed[:12]:
        state = _format_state(case.state)
        expected = ", ".join(sorted(case.expected)) or "nothing"
        predicted = ", ".join(sorted(case.predicted)) or "nothing"
        lines.append(
            f"  - {case.action} @ {state}: "
            f"expected [{expected}], predicted [{predicted}]"
        )
    if len(failed) > 12:
        lines.append(f"  ... {len(failed) - 12} more")
    if not failed:
        lines.append("  none")
    return "\n".join(lines)


def format_intervention_test(result: InterventionTestResult) -> str:
    exploration_score = result.exploration.score
    lines = [
        f"World: {result.exploration.world_name}",
        f"Agent: {result.exploration.agent_name}",
        f"Explore steps: {len(result.exploration.transitions)}",
        (
            "Exploration graph: "
            f"precision={exploration_score.precision:.2f}, "
            f"recall={exploration_score.recall:.2f}, "
            f"f1={exploration_score.f1:.2f}"
        ),
        (
            "Held-out interventions: "
            f"cases={len(result.cases)}, "
            f"exact={result.exact_match_rate:.2f}, "
            f"jaccard={result.mean_jaccard:.2f}, "
            f"f1={result.prediction_f1:.2f}"
        ),
        "",
        "Failed cases:",
    ]
    failed = [case for case in result.cases if not case.exact_match]
    for case in failed[:12]:
        state = ", ".join(f"{key}={value}" for key, value in sorted(case.state.items()))
        expected = ", ".join(sorted(case.expected)) or "nothing"
        predicted = ", ".join(sorted(case.predicted)) or "nothing"
        lines.append(
            f"  - {case.action} @ {state}: "
            f"expected [{expected}], predicted [{predicted}]"
        )
    if len(failed) > 12:
        lines.append(f"  ... {len(failed) - 12} more")
    if not failed:
        lines.append("  none")
    return "\n".join(lines)


def score_edges(
    discovered_edges: set[Edge],
    true_edges: set[Edge],
    transitions: tuple[Transition, ...],
) -> EdgeScore:
    true_positive = len(discovered_edges & true_edges)
    false_positive = len(discovered_edges - true_edges)
    false_negative = len(true_edges - discovered_edges)
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    prediction_jaccard = _mean_prediction_jaccard(transitions)
    return EdgeScore(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        prediction_jaccard=prediction_jaccard,
    )


def format_episode(result: EpisodeResult) -> str:
    score = result.score
    lines = [
        f"World: {result.world_name}",
        f"Agent: {result.agent_name}",
        f"Steps: {len(result.transitions)}",
        (
            "Graph score: "
            f"precision={score.precision:.2f}, "
            f"recall={score.recall:.2f}, "
            f"f1={score.f1:.2f}"
        ),
        f"Prediction jaccard: {score.prediction_jaccard:.2f}",
        "",
        "Learned causal claims:",
    ]

    hints = result.condition_hints
    for action, target in sorted(result.discovered_edges):
        suffix = ""
        if (action, target) in hints:
            suffix = " when " + ", ".join(hints[(action, target)])
        lines.append(f"  - {action} -> {target}{suffix}")

    lines.extend(["", "Recent transitions:"])
    for transition in result.transitions[-8:]:
        changed = ", ".join(sorted(transition.changed)) or "nothing"
        prediction = ", ".join(sorted(transition.prediction)) or "unknown"
        lines.append(
            f"  {transition.step:02d}. {transition.action}: "
            f"predicted [{prediction}], changed [{changed}]"
        )
    return "\n".join(lines)


def _condition_hints_for_agent(agent: Agent) -> dict[Edge, list[str]]:
    condition_hints_fn = getattr(agent, "condition_hints", None)
    if condition_hints_fn is None:
        return getattr(agent, "memory").condition_hints()
    return condition_hints_fn()


def _resolve_level4_spec(
    world: World,
    target_action: str | None,
    target_variable: str | None,
    effect_value: bool | None,
    candidate_gates: tuple[tuple[str, bool], ...] | None,
) -> tuple[str, str, bool, tuple[tuple[str, bool], ...]]:
    spec_fn = getattr(world, "level4_spec", None)
    spec = spec_fn() if spec_fn is not None else {}
    resolved_action = target_action or str(spec.get("target_action", "press_a"))
    resolved_variable = target_variable or str(spec.get("target_variable", "lamp_on"))
    resolved_effect = (
        effect_value if effect_value is not None else bool(spec.get("effect_value", True))
    )
    resolved_gates = candidate_gates
    if resolved_gates is None:
        raw_gates = spec.get(
            "candidate_gates",
            (("dark", True), ("door_open", True)),
        )
        resolved_gates = tuple(
            (str(variable), bool(value)) for variable, value in raw_gates
        )
    return resolved_action, resolved_variable, resolved_effect, resolved_gates


def _gate_case_templates(
    world: World,
    base_state: dict[str, bool],
    target_action: str,
    target_variable: str,
    effect_value: bool,
    candidate_gates: tuple[tuple[str, bool], ...],
) -> tuple[GateDisambiguationCase, ...]:
    cases: list[GateDisambiguationCase] = []
    full_state = dict(base_state)
    for variable, value in candidate_gates:
        full_state[variable] = value
    full_state[target_variable] = not effect_value
    full_expected = target_variable in _expected_changes(
        world, full_state, target_action
    )
    cases.append(
        GateDisambiguationCase(
            gate_variable="all",
            gate_value=True,
            state=dict(full_state),
            expected_effect=full_expected,
            predicted_effect=False,
        )
    )
    for focus_variable, focus_value in candidate_gates:
        state = dict(base_state)
        for variable, value in candidate_gates:
            state[variable] = value
        state[focus_variable] = not focus_value
        state[target_variable] = not effect_value
        expected = target_variable in _expected_changes(world, state, target_action)
        cases.append(
            GateDisambiguationCase(
                gate_variable=focus_variable,
                gate_value=focus_value,
                state=dict(state),
                expected_effect=expected,
                predicted_effect=False,
            )
        )
    return tuple(cases)


def _with_gate_predictions(
    agent: Agent,
    cases: tuple[GateDisambiguationCase, ...],
    target_action: str,
    target_variable: str,
) -> tuple[GateDisambiguationCase, ...]:
    predicted_cases: list[GateDisambiguationCase] = []
    for case in cases:
        predicted = target_variable in agent.predict(target_action, dict(case.state))
        predicted_cases.append(
            GateDisambiguationCase(
                gate_variable=case.gate_variable,
                gate_value=case.gate_value,
                state=dict(case.state),
                expected_effect=case.expected_effect,
                predicted_effect=predicted,
            )
        )
    return tuple(predicted_cases)


def _gate_predictions_correct(
    agent: Agent,
    cases: tuple[GateDisambiguationCase, ...],
    target_action: str,
    target_variable: str,
) -> bool:
    return all(
        case.correct
        for case in _with_gate_predictions(
            agent=agent,
            cases=cases,
            target_action=target_action,
            target_variable=target_variable,
        )
    )


def _resolve_temporal_spec(
    world: World,
    target_action: str | None,
    target_variable: str | None,
    effect_value: bool | None,
    followup_action: str | None,
    delay_steps: int | None,
) -> tuple[str, str, bool, str, int]:
    spec_fn = getattr(world, "temporal_spec", None)
    spec = spec_fn() if spec_fn is not None else {}
    resolved_action = target_action or str(spec.get("target_action", "press_delay"))
    resolved_variable = target_variable or str(spec.get("target_variable", "lamp_on"))
    resolved_effect = (
        effect_value if effect_value is not None else bool(spec.get("effect_value", True))
    )
    resolved_followup = followup_action or str(spec.get("followup_action", "wait"))
    resolved_delay = delay_steps if delay_steps is not None else int(
        spec.get("delay_steps", 1)
    )
    return (
        resolved_action,
        resolved_variable,
        resolved_effect,
        resolved_followup,
        resolved_delay,
    )


def _resolve_temporal_specs(world: World) -> tuple[tuple[str, str, bool, str, int], ...]:
    specs_fn = getattr(world, "temporal_specs", None)
    if specs_fn is None:
        return (
            _resolve_temporal_spec(
                world=world,
                target_action=None,
                target_variable=None,
                effect_value=None,
                followup_action=None,
                delay_steps=None,
            ),
        )
    specs = []
    for spec in specs_fn():
        specs.append(
            (
                str(spec.get("target_action", "press_delay")),
                str(spec.get("target_variable", "lamp_on")),
                bool(spec.get("effect_value", True)),
                str(spec.get("followup_action", "wait")),
                int(spec.get("delay_steps", 1)),
            )
        )
    return tuple(specs)


def _matching_temporal_spec(
    world: World, target_action: str, target_variable: str
) -> tuple[str, str, bool, str, int] | None:
    for spec in _resolve_temporal_specs(world):
        spec_action, spec_variable, _, _, _ = spec
        if spec_action == target_action and spec_variable == target_variable:
            return spec
    return None


def _temporal_case_templates(
    world: World,
    base_state: dict[str, bool],
    target_action: str,
    target_variable: str,
    effect_value: bool,
    followup_action: str,
    delay_steps: int,
) -> tuple[DelayedEffectCase, ...]:
    active_state = dict(base_state)
    active_state[target_variable] = not effect_value
    active_expected_after = _expected_temporal_after(
        world=world,
        state=active_state,
        action=target_action,
        followup_action=followup_action,
        delay_steps=delay_steps,
    )
    active_expected_effect = active_expected_after.get(target_variable) == effect_value

    already_state = dict(base_state)
    already_state[target_variable] = effect_value
    already_expected_after = _expected_temporal_after(
        world=world,
        state=already_state,
        action=target_action,
        followup_action=followup_action,
        delay_steps=delay_steps,
    )
    already_expected_effect = (
        already_expected_after.get(target_variable) != already_state[target_variable]
        and already_expected_after.get(target_variable) == effect_value
    )
    return (
        DelayedEffectCase(
            action=target_action,
            followup_action=followup_action,
            delay_steps=delay_steps,
            state=dict(active_state),
            expected_effect=active_expected_effect,
            predicted_effect=False,
        ),
        DelayedEffectCase(
            action=target_action,
            followup_action=followup_action,
            delay_steps=delay_steps,
            state=dict(already_state),
            expected_effect=already_expected_effect,
            predicted_effect=False,
        ),
    )


def _with_temporal_predictions(
    agent: Agent,
    cases: tuple[DelayedEffectCase, ...],
    target_variable: str,
) -> tuple[DelayedEffectCase, ...]:
    predicted_cases: list[DelayedEffectCase] = []
    for case in cases:
        predicted = _predict_temporal_effect(agent, case, target_variable)
        predicted_cases.append(
            DelayedEffectCase(
                action=case.action,
                followup_action=case.followup_action,
                delay_steps=case.delay_steps,
                state=dict(case.state),
                expected_effect=case.expected_effect,
                predicted_effect=predicted,
            )
        )
    return tuple(predicted_cases)


def _temporal_predictions_correct(
    agent: Agent,
    cases: tuple[DelayedEffectCase, ...],
    target_variable: str,
) -> bool:
    return all(
        case.correct
        for case in _with_temporal_predictions(
            agent=agent,
            cases=cases,
            target_variable=target_variable,
        )
    )


def _predict_temporal_effect(
    agent: Agent, case: DelayedEffectCase, target_variable: str
) -> bool:
    predictor = getattr(agent, "predict_temporal", None)
    if predictor is not None:
        return target_variable in predictor(
            case.action,
            dict(case.state),
            followup_action=case.followup_action,
            delay_steps=case.delay_steps,
        )
    return target_variable in agent.predict(case.action, dict(case.state))


def _temporal_mechanism_shifted(
    source_world: World,
    target_world: World,
    cases: tuple[DelayedEffectCase, ...],
    target_action: str,
    target_variable: str,
    effect_value: bool,
    followup_action: str,
    delay_steps: int,
) -> bool:
    source_spec = _matching_temporal_spec(
        source_world,
        target_action=target_action,
        target_variable=target_variable,
    )
    target_spec = (
        target_action,
        target_variable,
        effect_value,
        followup_action,
        delay_steps,
    )
    if source_spec is None or source_spec != target_spec:
        return True

    for case in cases:
        source_effect = _expected_temporal_effect_flag(
            world=source_world,
            state=dict(case.state),
            action=case.action,
            target_variable=target_variable,
            effect_value=effect_value,
            followup_action=case.followup_action,
            delay_steps=case.delay_steps,
        )
        if source_effect != case.expected_effect:
            return True
    return False


def _is_disambiguating_transition(
    transition: Transition,
    target_action: str,
    target_variable: str,
    effect_value: bool,
    candidate_gates: tuple[tuple[str, bool], ...],
) -> bool:
    if transition.action != target_action:
        return False
    if transition.before.get(target_variable) == effect_value:
        return False
    gate_truths = {
        transition.before.get(variable) == value
        for variable, value in candidate_gates
    }
    return len(gate_truths) > 1


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _temporal_target_mean(
    targets: tuple[TemporalTargetResult, ...],
    mechanism_shifted: bool | None = None,
) -> float:
    scores = [
        target.temporal_score
        for target in targets
        if mechanism_shifted is None or target.mechanism_shifted == mechanism_shifted
    ]
    return _safe_div(sum(scores), len(scores))


def _schema_shifted_temporal_roles(
    source_world: World, target_world: World
) -> frozenset[str]:
    source_delays = _temporal_role_delays(source_world)
    shifted_roles: set[str] = set()
    for role, target_delay in _temporal_role_delays(target_world).items():
        if source_delays.get(role) != target_delay:
            shifted_roles.add(role)
    return frozenset(shifted_roles)


def _temporal_role_delays(world: World) -> dict[str, int]:
    delays: dict[str, int] = {}
    for action, variable, _, _, delay_steps in _resolve_temporal_specs(world):
        role = _schema_temporal_role(action, variable)
        delays[role] = delay_steps
    return delays


def _schema_temporal_target_mean(
    targets: tuple[TemporalTargetResult, ...],
    shifted_roles: frozenset[str],
    schema_shifted: bool,
) -> float:
    selected = [
        target.temporal_score
        for target in targets
        if (_schema_temporal_role(target.target_action, target.target_variable)
            in shifted_roles)
        == schema_shifted
    ]
    return _safe_div(sum(selected), len(selected))


def _schema_temporal_target_count(
    targets: tuple[TemporalTargetResult, ...],
    shifted_roles: frozenset[str],
    schema_shifted: bool,
) -> int:
    return sum(
        1
        for target in targets
        if (_schema_temporal_role(target.target_action, target.target_variable)
            in shifted_roles)
        == schema_shifted
    )


def _schema_temporal_role(action: str, variable: str) -> str:
    text = f"{action} {variable}".lower()
    tokens = {
        token
        for token in text.replace("-", "_").split("_")
        for token in token.split()
        if token
    }
    if tokens & {"lamp", "light", "glow", "beacon"}:
        return "lamp"
    if tokens & {"alarm", "siren", "alert"}:
        return "alarm"
    if tokens & {"door", "access"}:
        return "door"
    if tokens & {"fan", "rotor", "vent", "blower", "motion"}:
        return "fan"
    return f"{action}->{variable}"


def _mean_prediction_jaccard(transitions: tuple[Transition, ...]) -> float:
    if not transitions:
        return 0.0
    scores: list[float] = []
    for transition in transitions:
        predicted = set(transition.prediction)
        changed = set(transition.changed)
        union = predicted | changed
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(predicted & changed) / len(union))
    return sum(scores) / len(scores)


def all_boolean_states(variables: list[str]) -> list[dict[str, bool]]:
    states: list[dict[str, bool]] = []
    for values in product([False, True], repeat=len(variables)):
        states.append(dict(zip(variables, values)))
    return states


def _expected_changes(world: World, state: dict[str, bool], action: str) -> frozenset[str]:
    setter = getattr(world, "set_state", None)
    if setter is None:
        raise TypeError(f"{world.name} does not support set_state")
    setter(state)
    before = world.observe()
    after = world.step(action)
    return frozenset(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    )


def _micro_prediction_scores(
    cases: tuple[InterventionTestCase, ...]
) -> tuple[float, float, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for case in cases:
        true_positive += len(case.expected & case.predicted)
        false_positive += len(case.predicted - case.expected)
        false_negative += len(case.expected - case.predicted)
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _counterfactual_case(
    world: World,
    agent: Agent,
    transition: Transition,
    edit_type: str,
    edit_name: str,
    counterfactual_action: str,
    counterfactual_before: dict[str, bool],
) -> CounterfactualTestCase:
    expected_after = _expected_after(
        world, dict(counterfactual_before), counterfactual_action
    )
    predicted_after = agent.predict_next_state(
        counterfactual_action, dict(counterfactual_before)
    )
    return CounterfactualTestCase(
        factual_step=transition.step,
        factual_action=transition.action,
        factual_before=dict(transition.before),
        factual_after=dict(transition.after),
        edit_type=edit_type,
        edit_name=edit_name,
        counterfactual_action=counterfactual_action,
        counterfactual_before=dict(counterfactual_before),
        expected_after=expected_after,
        predicted_after=predicted_after,
    )


def _expected_after(
    world: World, state: dict[str, bool], action: str
) -> dict[str, bool]:
    setter = getattr(world, "set_state", None)
    if setter is None:
        raise TypeError(f"{world.name} does not support set_state")
    setter(state)
    return world.step(action)


def _expected_temporal_after(
    world: World,
    state: dict[str, bool],
    action: str,
    followup_action: str,
    delay_steps: int,
) -> dict[str, bool]:
    setter = getattr(world, "set_state", None)
    if setter is None:
        raise TypeError(f"{world.name} does not support set_state")
    setter(state)
    world.step(action)
    after = world.observe()
    for _ in range(delay_steps):
        after = world.step(followup_action)
    return after


def _expected_temporal_effect_flag(
    world: World,
    state: dict[str, bool],
    action: str,
    target_variable: str,
    effect_value: bool,
    followup_action: str,
    delay_steps: int,
) -> bool:
    after = _expected_temporal_after(
        world=world,
        state=state,
        action=action,
        followup_action=followup_action,
        delay_steps=delay_steps,
    )
    return (
        after.get(target_variable) != state.get(target_variable)
        and after.get(target_variable) == effect_value
    )


def _changed_keys(left: dict[str, bool], right: dict[str, bool]) -> frozenset[str]:
    return frozenset(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    )


def _micro_delta_scores(
    cases: tuple[CounterfactualTestCase, ...]
) -> tuple[float, float, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for case in cases:
        expected = case.expected_delta_from_fact
        predicted = case.predicted_delta_from_fact
        true_positive += len(expected & predicted)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _format_state(state: dict[str, bool]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(state.items()))
