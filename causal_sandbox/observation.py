from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

from .core import State, Transition


@dataclass(frozen=True)
class ReadoutRule:
    target: str
    expression: str
    sources: tuple[str, ...]
    accuracy: float


@dataclass(frozen=True)
class ActionTargetLift:
    action: str
    target: str
    action_count: int
    effect_count: int
    other_count: int
    total_count: int
    action_rate: float
    background_rate: float
    lift: float
    dominant_value: bool | None = None
    dominant_value_rate: float = 0.0


@dataclass(frozen=True)
class LatentContextGate:
    action: str
    target: str
    latent_name: str
    effect_value: bool
    eligible_count: int
    success_count: int
    failure_count: int
    success_rate: float
    best_observed_gate: str | None
    best_observed_gate_accuracy: float
    lift: float

    @property
    def state_label(self) -> str:
        if self.success_rate >= 0.65:
            return "likely enabled"
        if self.success_rate <= 0.35:
            return "likely disabled"
        return "mixed or switching"


@dataclass(frozen=True)
class LatentStateBelief:
    gate: LatentContextGate
    enabled_probability: float
    uncertainty: float
    posterior_alpha: float
    posterior_beta: float
    recent_successes: int
    recent_failures: int
    last_observation: str
    probe_priority: float
    total_successes: int = 0
    total_failures: int = 0
    confidence: float = 0.0
    recent_drift: float = 0.0
    belief_source: str = "recent-window"


class SensorFilteringObservationAdapter:
    """A minimal observation adapter for separating sensors from causal state.

    This is intentionally small and conservative. It treats variables with a
    sensor-like suffix as readouts, not intervention targets, and forwards the
    remaining variables to the causal core. The goal is not to solve causal
    representation learning, but to create a first experiment that separates
    derived observation variables from candidate causal state variables.
    """

    def __init__(self, sensor_suffix: str = "_sensor") -> None:
        self.sensor_suffix = sensor_suffix
        self.suppressed_variables: set[str] = set()

    def reset(self) -> None:
        self.suppressed_variables = set()

    def transform_state(self, state: State, learn: bool = True) -> State:
        if learn:
            self._observe_variables(state)
        return {
            variable: value
            for variable, value in state.items()
            if variable not in self.suppressed_variables
        }

    def transform_transition(
        self, transition: Transition, learn: bool = True
    ) -> Transition:
        before = self.transform_state(transition.before, learn=learn)
        after = self.transform_state(transition.after, learn=learn)
        prediction = frozenset(
            target
            for target in transition.prediction
            if target not in self.suppressed_variables
        )
        return Transition(
            step=transition.step,
            action=transition.action,
            before=before,
            after=after,
            prediction=prediction,
            hypothesis=transition.hypothesis,
        )

    def _observe_variables(self, state: State) -> None:
        for variable in state:
            if variable.endswith(self.sensor_suffix):
                self.suppressed_variables.add(variable)


class LearnedReadoutObservationAdapter:
    """Learn deterministic composite readouts from observed boolean states.

    The adapter suppresses variables whose values can be predicted by a simple
    non-identity boolean expression over other variables. Avoiding identity
    rules is deliberate: if two variables are perfect copies, observations
    alone cannot tell which one is the causal state and which one is the
    readout.
    """

    def __init__(
        self,
        min_observations: int = 40,
        min_accuracy: float = 0.98,
        refresh_interval: int = 4,
        allow_identity_from_protected: bool = False,
        max_candidate_sources: int = 8,
    ) -> None:
        self.min_observations = min_observations
        self.min_accuracy = min_accuracy
        self.refresh_interval = refresh_interval
        self.allow_identity_from_protected = allow_identity_from_protected
        self.max_candidate_sources = max_candidate_sources
        self.suppressed_variables: set[str] = set()
        self.protected_variables: set[str] = set()
        self.readout_rules: dict[str, ReadoutRule] = {}
        self._observed_states: list[State] = []

    def reset(self) -> None:
        self.suppressed_variables = set()
        self.protected_variables = set()
        self.readout_rules = {}
        self._observed_states = []

    def set_protected_variables(self, variables: set[str]) -> None:
        self.protected_variables = set(variables)
        self.suppressed_variables -= self.protected_variables
        self.readout_rules = {
            target: rule
            for target, rule in self.readout_rules.items()
            if target not in self.protected_variables
        }

    def observe_state(self, state: State) -> None:
        self._observed_states.append(dict(state))
        if (
            len(self._observed_states) >= self.min_observations
            and len(self._observed_states) % self.refresh_interval == 0
        ):
            self._refresh_rules()

    def observe_transition(self, transition: Transition) -> None:
        self.observe_state(transition.before)
        self.observe_state(transition.after)

    def transform_state(self, state: State, learn: bool = True) -> State:
        if learn:
            self.observe_state(state)
        return self.filter_state(state)

    def filter_state(self, state: State) -> State:
        return {
            variable: value
            for variable, value in state.items()
            if variable not in self.suppressed_variables
        }

    def transform_transition(
        self, transition: Transition, learn: bool = True
    ) -> Transition:
        if learn:
            self.observe_transition(transition)
        before = self.filter_state(transition.before)
        after = self.filter_state(transition.after)
        prediction = frozenset(
            target
            for target in transition.prediction
            if target not in self.suppressed_variables
        )
        return Transition(
            step=transition.step,
            action=transition.action,
            before=before,
            after=after,
            prediction=prediction,
            hypothesis=transition.hypothesis,
        )

    def _refresh_rules(self) -> None:
        if len(self._observed_states) < self.min_observations:
            return

        variables = sorted(
            set().union(*(state.keys() for state in self._observed_states))
        )
        learned_rules: dict[str, ReadoutRule] = {}
        for target in variables:
            if self._is_protected(target):
                continue
            rule = self._best_rule_for(target, variables)
            if rule is not None:
                learned_rules[target] = rule

        self.readout_rules = learned_rules
        self.suppressed_variables = set(learned_rules)

    def _best_rule_for(
        self, target: str, variables: list[str]
    ) -> ReadoutRule | None:
        target_states = [
            state for state in self._observed_states if target in state
        ]
        if len(target_states) < self.min_observations:
            return None
        if len({state[target] for state in target_states}) < 2:
            return None

        source_variables = self._ranked_sources_for(
            target=target,
            variables=variables,
            target_states=target_states,
        )
        best: ReadoutRule | None = None
        for expression, sources, evaluator in _candidate_formulas(
            target=target,
            variables=[target] + source_variables[: self.max_candidate_sources],
            protected_variables=self.protected_variables,
            allow_identity_from_protected=self.allow_identity_from_protected,
        ):
            evaluable_count = 0
            match_count = 0
            positive_predictions = 0
            for state in target_states:
                if not all(source in state for source in sources):
                    continue
                prediction = evaluator(state)
                evaluable_count += 1
                positive_predictions += int(prediction)
                match_count += int(state[target] == prediction)

            if evaluable_count < self.min_observations:
                continue
            if positive_predictions == 0 or positive_predictions == evaluable_count:
                continue

            accuracy = match_count / evaluable_count
            if accuracy < self.min_accuracy:
                continue

            rule = ReadoutRule(
                target=target,
                expression=expression,
                sources=sources,
                accuracy=accuracy,
            )
            if best is None or _rule_rank(rule) > _rule_rank(best):
                best = rule

        return best

    def _ranked_sources_for(
        self,
        target: str,
        variables: list[str],
        target_states: list[State],
    ) -> list[str]:
        ranked: list[tuple[int, float, str]] = []
        for source in variables:
            if source == target:
                continue
            evaluable = [
                state
                for state in target_states
                if target in state and source in state
            ]
            if len(evaluable) < self.min_observations:
                continue
            source_values = [state[source] for state in evaluable]
            if len(set(source_values)) < 2:
                continue
            target_values = [state[target] for state in evaluable]
            same = sum(
                int(source_value == target_value)
                for source_value, target_value in zip(source_values, target_values)
            )
            inverse = len(evaluable) - same
            source_true = sum(int(value) for value in source_values)
            target_true = sum(int(value) for value in target_values)
            joint_true = sum(
                int(source_value and target_value)
                for source_value, target_value in zip(source_values, target_values)
            )
            implication_precision = _safe_div(joint_true, source_true)
            implication_recall = _safe_div(joint_true, target_true)
            score = max(
                same / len(evaluable),
                inverse / len(evaluable),
                implication_precision,
                implication_recall,
            )
            protected = int(_variable_matches_protected(source, self.protected_variables))
            ranked.append((protected, score, source))
        return [
            source
            for _, _, source in sorted(ranked, reverse=True)
        ]

    def _is_protected(self, variable: str) -> bool:
        return _variable_matches_protected(variable, self.protected_variables)


class CausalObservationAdapterV2:
    """Learn readouts and filter low-lift changes before causal-memory updates.

    This adapter is an intentionally small v2 prototype. It combines two
    mechanisms:

    1. Learn deterministic or near-deterministic readouts, including oriented
       identity copies from action-referenced variables.
    2. Estimate action-conditioned change rates against background rates and
       remove observed changes whose lift is too weak to be treated as an
       intervention effect.

    The adapter does not delete noisy variables from the state by default. It
    only suppresses variables learned as readouts and neutralizes low-lift
    changes in transitions before they reach Causal Core.
    """

    def __init__(
        self,
        min_readout_observations: int = 40,
        min_readout_accuracy: float = 0.90,
        readout_refresh_interval: int = 4,
        min_total_transitions: int = 40,
        min_action_observations: int = 4,
        min_effect_observations: int = 2,
        min_effect_lift: float = 0.08,
        strong_action_rate: float = 0.45,
        sensor_suffixes: tuple[str, ...] = ("_sensor",),
        min_effect_value_consistency: float = 0.0,
        hidden_context_lift_floor: float = 0.04,
        hidden_context_min_action_rate: float = 0.15,
        hidden_context_max_background_rate: float = 0.25,
        enable_hidden_context_hypotheses: bool = False,
    ) -> None:
        self.readout_adapter = LearnedReadoutObservationAdapter(
            min_observations=min_readout_observations,
            min_accuracy=min_readout_accuracy,
            refresh_interval=readout_refresh_interval,
            allow_identity_from_protected=True,
        )
        self.min_total_transitions = min_total_transitions
        self.min_action_observations = min_action_observations
        self.min_effect_observations = min_effect_observations
        self.min_effect_lift = min_effect_lift
        self.strong_action_rate = strong_action_rate
        self.sensor_suffixes = sensor_suffixes
        self.min_effect_value_consistency = min_effect_value_consistency
        self.hidden_context_lift_floor = hidden_context_lift_floor
        self.hidden_context_min_action_rate = hidden_context_min_action_rate
        self.hidden_context_max_background_rate = hidden_context_max_background_rate
        self.enable_hidden_context_hypotheses = enable_hidden_context_hypotheses
        self.prior_suppressed_variables: set[str] = set()
        self._raw_transitions: list[Transition] = []
        self._action_counts: Counter[str] = Counter()
        self._target_counts: Counter[str] = Counter()
        self._edge_counts: Counter[tuple[str, str]] = Counter()
        self._edge_value_counts: Counter[tuple[str, str, bool]] = Counter()

    @property
    def suppressed_variables(self) -> set[str]:
        return set(self.readout_adapter.suppressed_variables) | set(
            self.prior_suppressed_variables
        )

    @property
    def readout_rules(self) -> dict[str, ReadoutRule]:
        return dict(self.readout_adapter.readout_rules)

    def reset(self) -> None:
        self.readout_adapter.reset()
        self.prior_suppressed_variables = set()
        self._raw_transitions = []
        self._reset_lift_stats()

    def set_protected_variables(self, variables: set[str]) -> None:
        self.readout_adapter.set_protected_variables(variables)
        self._refresh_lift_stats()

    def observe_transition(self, transition: Transition) -> None:
        self._raw_transitions.append(transition)
        self._observe_prior_suppressed(transition.before)
        self._observe_prior_suppressed(transition.after)
        self.readout_adapter.observe_transition(
            self._prior_filtered_transition(transition)
        )
        self._refresh_lift_stats()

    def transform_state(self, state: State, learn: bool = False) -> State:
        if learn:
            self._observe_prior_suppressed(state)
        return self.filter_state(state)

    def filter_state(self, state: State) -> State:
        return {
            variable: value
            for variable, value in state.items()
            if variable not in self.suppressed_variables
        }

    def transform_transition(
        self, transition: Transition, learn: bool = False
    ) -> Transition:
        if learn:
            self.observe_transition(transition)

        before = self.filter_state(transition.before)
        after = self.filter_state(transition.after)
        filtered_after = dict(after)
        for target in _changed_variables(before, after):
            if not self._change_is_causal_candidate(transition.action, target):
                filtered_after[target] = before[target]

        prediction = frozenset(
            target
            for target in transition.prediction
            if target not in self.suppressed_variables
        )
        return Transition(
            step=transition.step,
            action=transition.action,
            before=before,
            after=filtered_after,
            prediction=prediction,
            hypothesis=transition.hypothesis,
        )

    def lift_for(self, action: str, target: str) -> ActionTargetLift:
        action_count = self._action_counts[action]
        edge_count = self._edge_counts[(action, target)]
        total_count = len(self._raw_transitions)
        target_count = self._target_counts[target]
        other_total = max(total_count - action_count, 0)
        other_count = max(target_count - edge_count, 0)
        action_rate = _safe_div(edge_count, action_count)
        background_rate = _safe_div(other_count, other_total)
        value_counts = {
            value: self._edge_value_counts[(action, target, value)]
            for value in (False, True)
        }
        dominant_value, dominant_count = max(
            value_counts.items(), key=lambda item: (item[1], item[0])
        )
        if edge_count == 0:
            dominant_value = None
        return ActionTargetLift(
            action=action,
            target=target,
            action_count=action_count,
            effect_count=edge_count,
            other_count=other_count,
            total_count=total_count,
            action_rate=action_rate,
            background_rate=background_rate,
            lift=action_rate - background_rate,
            dominant_value=dominant_value,
            dominant_value_rate=_safe_div(dominant_count, edge_count),
        )

    def _change_is_causal_candidate(self, action: str, target: str) -> bool:
        total_count = len(self._raw_transitions)
        if total_count < self.min_total_transitions:
            return True

        stats = self.lift_for(action, target)
        if stats.action_count < self.min_action_observations:
            return True
        if stats.effect_count < self.min_effect_observations:
            return False
        if stats.dominant_value_rate < self.min_effect_value_consistency:
            return False
        if stats.lift >= self.min_effect_lift:
            return True
        if self.enable_hidden_context_hypotheses and self._looks_like_hidden_context_effect(stats):
            return True
        return stats.action_rate >= self.strong_action_rate and stats.lift > 0

    def hidden_context_hypotheses(self) -> dict[tuple[str, str], ActionTargetLift]:
        if not self.enable_hidden_context_hypotheses:
            return {}
        hypotheses: dict[tuple[str, str], ActionTargetLift] = {}
        targets = {
            target for _, target in self._edge_counts if target not in self.suppressed_variables
        }
        for action in self._action_counts:
            for target in targets:
                stats = self.lift_for(action, target)
                if self._looks_like_hidden_context_effect(stats):
                    hypotheses[(action, target)] = stats
        return hypotheses

    def _looks_like_hidden_context_effect(self, stats: ActionTargetLift) -> bool:
        if stats.effect_count < self.min_effect_observations + 1:
            return False
        if stats.dominant_value_rate < self.min_effect_value_consistency:
            return False
        if stats.lift < self.hidden_context_lift_floor:
            return False
        if stats.action_rate < self.hidden_context_min_action_rate:
            return False
        return stats.background_rate <= self.hidden_context_max_background_rate

    def _refresh_lift_stats(self) -> None:
        self._reset_lift_stats()
        for transition in self._raw_transitions:
            before = self.filter_state(transition.before)
            after = self.filter_state(transition.after)
            self._action_counts[transition.action] += 1
            for target in _changed_variables(before, after):
                self._target_counts[target] += 1
                self._edge_counts[(transition.action, target)] += 1
                self._edge_value_counts[(transition.action, target, after[target])] += 1

    def _reset_lift_stats(self) -> None:
        self._action_counts = Counter()
        self._target_counts = Counter()
        self._edge_counts = Counter()
        self._edge_value_counts = Counter()

    def _observe_prior_suppressed(self, state: State) -> None:
        for variable in state:
            if variable.endswith(self.sensor_suffixes):
                self.prior_suppressed_variables.add(variable)

    def _prior_filtered_transition(self, transition: Transition) -> Transition:
        before = {
            variable: value
            for variable, value in transition.before.items()
            if variable not in self.prior_suppressed_variables
        }
        after = {
            variable: value
            for variable, value in transition.after.items()
            if variable not in self.prior_suppressed_variables
        }
        prediction = frozenset(
            target
            for target in transition.prediction
            if target not in self.prior_suppressed_variables
        )
        return Transition(
            step=transition.step,
            action=transition.action,
            before=before,
            after=after,
            prediction=prediction,
            hypothesis=transition.hypothesis,
        )


class HiddenContextObservationAdapter(CausalObservationAdapterV2):
    """Observation Adapter v3 with effect stability and hidden-context hints."""

    def __init__(self) -> None:
        super().__init__(
            min_readout_observations=40,
            min_readout_accuracy=0.90,
            readout_refresh_interval=4,
            min_total_transitions=40,
            min_action_observations=4,
            min_effect_observations=2,
            min_effect_lift=0.14,
            strong_action_rate=0.50,
            min_effect_value_consistency=0.75,
            hidden_context_lift_floor=0.04,
            hidden_context_min_action_rate=0.15,
            hidden_context_max_background_rate=0.25,
            enable_hidden_context_hypotheses=True,
        )


class LatentContextObservationAdapter(CausalObservationAdapterV2):
    """Observation Adapter v4 with explicit latent context gates.

    The key difference from HiddenContextObservationAdapter is that failures
    are counted only on eligible attempts: cases where the target was not
    already at the learned effect value. This avoids treating normal no-op
    interventions, such as closing an already closed door, as hidden context.
    """

    def __init__(self) -> None:
        super().__init__(
            min_readout_observations=40,
            min_readout_accuracy=0.90,
            readout_refresh_interval=4,
            min_total_transitions=40,
            min_action_observations=4,
            min_effect_observations=2,
            min_effect_lift=0.14,
            strong_action_rate=0.55,
            min_effect_value_consistency=0.78,
            hidden_context_lift_floor=0.04,
            hidden_context_min_action_rate=0.15,
            hidden_context_max_background_rate=0.25,
            enable_hidden_context_hypotheses=False,
        )
        self.min_latent_eligible_attempts = 8
        self.min_latent_successes = 2
        self.min_latent_failures = 2
        self.min_latent_success_rate = 0.20
        self.max_latent_success_rate = 0.80
        self.max_observed_gate_accuracy = 0.88
        self.min_eligible_effect_attempts = 3
        self.min_eligible_effect_success_rate = 0.70
        self._latent_gate_cache_signature: tuple[int, tuple[str, ...]] | None = None
        self._latent_gate_cache: dict[tuple[str, str], LatentContextGate] = {}
        self._eligible_examples_cache_signature: (
            tuple[int, tuple[str, ...]] | None
        ) = None
        self._eligible_examples_cache: dict[
            tuple[str, str, bool],
            tuple[list[State], list[State]],
        ] = {}

    def latent_context_gates(self) -> dict[tuple[str, str], LatentContextGate]:
        signature = self._cache_signature()
        if self._latent_gate_cache_signature == signature:
            return dict(self._latent_gate_cache)

        gates: dict[tuple[str, str], LatentContextGate] = {}
        targets = {
            target
            for _, target in self._edge_counts
            if target not in self.suppressed_variables
        }
        for action in self._action_counts:
            for target in targets:
                gate = self._latent_context_gate_for(action, target)
                if gate is not None:
                    gates[(action, target)] = gate
        self._latent_gate_cache_signature = signature
        self._latent_gate_cache = gates
        return gates

    def _cache_signature(self) -> tuple[int, tuple[str, ...]]:
        return (
            len(self._raw_transitions),
            tuple(sorted(self.suppressed_variables)),
        )

    def _change_is_causal_candidate(self, action: str, target: str) -> bool:
        total_count = len(self._raw_transitions)
        if total_count < self.min_total_transitions:
            return True

        stats = self.lift_for(action, target)
        if stats.action_count < self.min_action_observations:
            return True
        if stats.effect_count < self.min_effect_observations:
            return False
        if stats.dominant_value_rate < self.min_effect_value_consistency:
            return False
        if stats.lift >= self.min_effect_lift:
            return True
        if self._eligible_effect_is_stable(
            action=action,
            target=target,
            effect_value=stats.dominant_value,
        ):
            return True
        if self._latent_context_gate_for(action, target) is not None:
            return True
        return stats.action_rate >= self.strong_action_rate and stats.lift > 0

    def _eligible_effect_is_stable(
        self, action: str, target: str, effect_value: bool | None
    ) -> bool:
        if effect_value is None:
            return False
        successes, failures = self._eligible_effect_examples(
            action=action,
            target=target,
            effect_value=effect_value,
        )
        eligible_count = len(successes) + len(failures)
        if eligible_count < self.min_eligible_effect_attempts:
            return False
        if len(successes) < self.min_effect_observations:
            return False
        success_rate = _safe_div(len(successes), eligible_count)
        return success_rate >= self.min_eligible_effect_success_rate

    def _latent_context_gate_for(
        self, action: str, target: str
    ) -> LatentContextGate | None:
        if target in self.suppressed_variables:
            return None

        stats = self.lift_for(action, target)
        if stats.dominant_value is None:
            return None
        if stats.lift < self.hidden_context_lift_floor:
            return None
        if stats.background_rate > self.hidden_context_max_background_rate:
            return None

        successes, failures = self._eligible_effect_examples(
            action=action,
            target=target,
            effect_value=stats.dominant_value,
        )
        eligible_count = len(successes) + len(failures)
        if eligible_count < self.min_latent_eligible_attempts:
            return None
        if len(successes) < self.min_latent_successes:
            return None
        if len(failures) < self.min_latent_failures:
            return None

        success_rate = _safe_div(len(successes), eligible_count)
        if success_rate < self.min_latent_success_rate:
            return None
        if success_rate > self.max_latent_success_rate:
            return None

        observed_gate, observed_accuracy = self._best_observed_separator(
            target=target,
            successes=successes,
            failures=failures,
        )
        if observed_accuracy >= self.max_observed_gate_accuracy:
            return None

        latent_name = f"latent:{action}->{target}:enabled"
        return LatentContextGate(
            action=action,
            target=target,
            latent_name=latent_name,
            effect_value=stats.dominant_value,
            eligible_count=eligible_count,
            success_count=len(successes),
            failure_count=len(failures),
            success_rate=success_rate,
            best_observed_gate=observed_gate,
            best_observed_gate_accuracy=observed_accuracy,
            lift=stats.lift,
        )

    def _eligible_effect_examples(
        self, action: str, target: str, effect_value: bool
    ) -> tuple[list[State], list[State]]:
        signature = self._cache_signature()
        if self._eligible_examples_cache_signature != signature:
            self._eligible_examples_cache_signature = signature
            self._eligible_examples_cache = {}

        cache_key = (action, target, effect_value)
        cached = self._eligible_examples_cache.get(cache_key)
        if cached is not None:
            successes, failures = cached
            return list(successes), list(failures)

        successes: list[State] = []
        failures: list[State] = []
        for transition in self._raw_transitions:
            if transition.action != action:
                continue
            before = self.filter_state(transition.before)
            after = self.filter_state(transition.after)
            if target not in before or target not in after:
                continue
            if before[target] == effect_value:
                continue
            if after[target] == effect_value:
                successes.append(before)
            else:
                failures.append(before)
        self._eligible_examples_cache[cache_key] = (successes, failures)
        return successes, failures

    def _best_observed_separator(
        self,
        target: str,
        successes: list[State],
        failures: list[State],
    ) -> tuple[str | None, float]:
        examples = successes + failures
        variables = sorted(set().union(*(state.keys() for state in examples)))
        best_gate: str | None = None
        best_accuracy = 0.0
        for variable in variables:
            if variable == target or variable in self.suppressed_variables:
                continue
            for value in (False, True):
                correct_successes = sum(
                    state.get(variable) == value for state in successes
                )
                correct_failures = sum(
                    state.get(variable) != value for state in failures
                )
                accuracy = _safe_div(
                    correct_successes + correct_failures,
                    len(successes) + len(failures),
                )
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_gate = f"{variable}={value}"
        return best_gate, best_accuracy


class StatefulLatentContextObservationAdapter(LatentContextObservationAdapter):
    """Observation Adapter v5 with stateful latent beliefs and probe priority."""

    def __init__(self) -> None:
        super().__init__()
        self.latent_belief_window = 8
        self.min_probe_uncertainty = 0.25
        self._eligible_outcomes_cache_signature: (
            tuple[int, tuple[str, ...]] | None
        ) = None
        self._eligible_outcomes_cache: dict[tuple[str, str, bool], list[bool]] = {}

    def latent_state_beliefs(self) -> dict[tuple[str, str], LatentStateBelief]:
        beliefs: dict[tuple[str, str], LatentStateBelief] = {}
        for edge, gate in self.latent_context_gates().items():
            beliefs[edge] = self._latent_state_belief_for(gate)
        return beliefs

    def probe_candidate(self, state: State) -> LatentStateBelief | None:
        filtered_state = self.filter_state(state)
        candidates = sorted(
            self.latent_state_beliefs().values(),
            key=lambda belief: belief.probe_priority,
            reverse=True,
        )
        for belief in candidates:
            gate = belief.gate
            if gate.target not in filtered_state:
                continue
            if filtered_state[gate.target] == gate.effect_value:
                continue
            if belief.uncertainty < self.min_probe_uncertainty:
                continue
            return belief
        return None

    def _latent_state_belief_for(
        self, gate: LatentContextGate
    ) -> LatentStateBelief:
        outcomes = self._eligible_effect_outcomes(
            action=gate.action,
            target=gate.target,
            effect_value=gate.effect_value,
        )
        recent = outcomes[-self.latent_belief_window :]
        recent_successes = sum(recent)
        recent_failures = len(recent) - recent_successes
        posterior_alpha = 1.0 + recent_successes
        posterior_beta = 1.0 + recent_failures
        enabled_probability = _safe_div(
            posterior_alpha,
            posterior_alpha + posterior_beta,
        )
        ambiguity = 1.0 - abs(enabled_probability - 0.5) * 2.0
        evidence_gap = 1.0 - min(len(recent) / self.latent_belief_window, 1.0)
        uncertainty = max(ambiguity, evidence_gap)
        last_observation = "none"
        if outcomes:
            last_observation = "success" if outcomes[-1] else "failure"
        probe_priority = uncertainty * (1.0 + min(gate.eligible_count / 20.0, 1.0))
        return LatentStateBelief(
            gate=gate,
            enabled_probability=enabled_probability,
            uncertainty=uncertainty,
            posterior_alpha=posterior_alpha,
            posterior_beta=posterior_beta,
            recent_successes=recent_successes,
            recent_failures=recent_failures,
            last_observation=last_observation,
            probe_priority=probe_priority,
        )

    def _eligible_effect_outcomes(
        self, action: str, target: str, effect_value: bool
    ) -> list[bool]:
        signature = self._cache_signature()
        if self._eligible_outcomes_cache_signature != signature:
            self._eligible_outcomes_cache_signature = signature
            self._eligible_outcomes_cache = {}

        cache_key = (action, target, effect_value)
        cached = self._eligible_outcomes_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        outcomes: list[bool] = []
        for transition in self._raw_transitions:
            if transition.action != action:
                continue
            before = self.filter_state(transition.before)
            after = self.filter_state(transition.after)
            if target not in before or target not in after:
                continue
            if before[target] == effect_value:
                continue
            outcomes.append(after[target] == effect_value)
        self._eligible_outcomes_cache[cache_key] = outcomes
        return outcomes


class PersistentLatentContextObservationAdapter(
    StatefulLatentContextObservationAdapter
):
    """Observation Adapter v6 with persistent latent posterior estimates."""

    def __init__(self) -> None:
        super().__init__()
        self.min_persistent_evidence = 8
        self.min_probe_uncertainty = 0.70

    def _latent_state_belief_for(
        self, gate: LatentContextGate
    ) -> LatentStateBelief:
        outcomes = self._eligible_effect_outcomes(
            action=gate.action,
            target=gate.target,
            effect_value=gate.effect_value,
        )
        recent = outcomes[-self.latent_belief_window :]
        recent_successes = sum(recent)
        recent_failures = len(recent) - recent_successes
        total_successes = sum(outcomes)
        total_failures = len(outcomes) - total_successes
        posterior_alpha = 1.0 + total_successes
        posterior_beta = 1.0 + total_failures
        enabled_probability = _safe_div(
            posterior_alpha,
            posterior_alpha + posterior_beta,
        )
        persistent_ambiguity = 1.0 - abs(enabled_probability - 0.5) * 2.0
        evidence_gap = 1.0 - min(
            len(outcomes) / self.min_persistent_evidence,
            1.0,
        )
        recent_probability = _safe_div(1.0 + recent_successes, 2.0 + len(recent))
        recent_drift = abs(enabled_probability - recent_probability)
        uncertainty = max(
            persistent_ambiguity,
            evidence_gap,
            min(recent_drift * 2.0, 1.0),
        )
        confidence = (1.0 - persistent_ambiguity) * (1.0 - evidence_gap)
        last_observation = "none"
        if outcomes:
            last_observation = "success" if outcomes[-1] else "failure"
        probe_priority = uncertainty * (1.0 + min(len(outcomes) / 20.0, 1.0))
        return LatentStateBelief(
            gate=gate,
            enabled_probability=enabled_probability,
            uncertainty=uncertainty,
            posterior_alpha=posterior_alpha,
            posterior_beta=posterior_beta,
            recent_successes=recent_successes,
            recent_failures=recent_failures,
            last_observation=last_observation,
            probe_priority=probe_priority,
            total_successes=total_successes,
            total_failures=total_failures,
            confidence=confidence,
            recent_drift=recent_drift,
            belief_source="persistent-posterior",
        )


FormulaEvaluator = Callable[[State], bool]


def _candidate_formulas(
    target: str,
    variables: list[str],
    protected_variables: set[str] | None = None,
    allow_identity_from_protected: bool = False,
) -> list[tuple[str, tuple[str, ...], FormulaEvaluator]]:
    formulas: list[tuple[str, tuple[str, ...], FormulaEvaluator]] = []
    sources = [variable for variable in variables if variable != target]
    protected_variables = protected_variables or set()
    if allow_identity_from_protected and not _variable_matches_protected(
        target, protected_variables
    ):
        for source in sources:
            if _variable_matches_protected(source, protected_variables):
                formulas.append(
                    (
                        source,
                        (source,),
                        lambda state, source=source: state[source],
                    )
                )
    for left, right in combinations(sources, 2):
        formulas.extend(
            [
                (
                    f"{left} OR {right}",
                    (left, right),
                    lambda state, left=left, right=right: state[left]
                    or state[right],
                ),
                (
                    f"{left} AND {right}",
                    (left, right),
                    lambda state, left=left, right=right: state[left]
                    and state[right],
                ),
                (
                    f"{left} XOR {right}",
                    (left, right),
                    lambda state, left=left, right=right: state[left]
                    != state[right],
                ),
                (
                    f"{left} == {right}",
                    (left, right),
                    lambda state, left=left, right=right: state[left]
                    == state[right],
                ),
                (
                    f"{left} OR NOT {right}",
                    (left, right),
                    lambda state, left=left, right=right: state[left]
                    or (not state[right]),
                ),
                (
                    f"NOT {left} OR {right}",
                    (left, right),
                    lambda state, left=left, right=right: (not state[left])
                    or state[right],
                ),
                (
                    f"{left} AND NOT {right}",
                    (left, right),
                    lambda state, left=left, right=right: state[left]
                    and (not state[right]),
                ),
                (
                    f"NOT {left} AND {right}",
                    (left, right),
                    lambda state, left=left, right=right: (not state[left])
                    and state[right],
                ),
            ]
        )
    for first, second, third in combinations(sources, 3):
        triples = (
            (first, second, third),
            (second, first, third),
            (third, first, second),
        )
        for outer, inner_left, inner_right in triples:
            formulas.append(
                (
                    f"{outer} OR ({inner_left} AND {inner_right})",
                    (outer, inner_left, inner_right),
                    lambda state, outer=outer, inner_left=inner_left, inner_right=inner_right: (
                        state[outer] or (state[inner_left] and state[inner_right])
                    ),
                )
            )
            formulas.append(
                (
                    f"{outer} AND ({inner_left} OR {inner_right})",
                    (outer, inner_left, inner_right),
                    lambda state, outer=outer, inner_left=inner_left, inner_right=inner_right: (
                        state[outer] and (state[inner_left] or state[inner_right])
                    ),
                )
            )
    return formulas


def _rule_rank(rule: ReadoutRule) -> tuple[float, int, str]:
    return (rule.accuracy, len(rule.sources), rule.expression)


def _variable_matches_protected(variable: str, protected_variables: set[str]) -> bool:
    if variable in protected_variables:
        return True
    pieces = {piece for piece in variable.split("_") if len(piece) >= 3}
    return bool(pieces & protected_variables)


def _changed_variables(before: State, after: State) -> frozenset[str]:
    keys = set(before) | set(after)
    return frozenset(key for key in keys if before.get(key) != after.get(key))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
