from __future__ import annotations

from collections import Counter
from random import Random
import re

from .causal_core import CausalCore
from .core import ActionSpec, AgentDecision, Edge, State, Transition
from .memory import CausalMemory
from .observation import (
    CausalObservationAdapterV2,
    HiddenContextObservationAdapter,
    LatentContextObservationAdapter,
    LatentStateBelief,
    LearnedReadoutObservationAdapter,
    PersistentLatentContextObservationAdapter,
    SensorFilteringObservationAdapter,
    StatefulLatentContextObservationAdapter,
)
from .reward import task_utility


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = Random(seed)
        self._actions: list[str] = []
        self.memory = CausalMemory()

    def reset(self, actions: list[ActionSpec]) -> None:
        self._rng = Random(self._seed)
        self._actions = [action.name for action in actions]
        self.memory = CausalMemory()

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        action = self._rng.choice(self._actions)
        prediction = self.memory.predicted_changes_for(action)
        return AgentDecision(
            action=action,
            prediction=prediction,
            hypothesis="Try a random intervention and compare the outcome.",
        )

    def observe_transition(self, transition: Transition) -> None:
        self.memory.record(transition)

    def discovered_edges(self) -> set[Edge]:
        return self.memory.discovered_edges()

    def predict(self, action: str, state: State) -> frozenset[str]:
        return self.memory.predicted_changes_for(action)

    def predict_next_state(self, action: str, state: State) -> State:
        return self.memory.predicted_state_for(action, state)


class ActiveCausalAgent:
    name = "active"

    _preferred_plan = [
        "reset_lamp",
        "set_dark",
        "press_a",
        "wait",
        "reset_lamp",
        "set_bright",
        "press_a",
        "wait",
        "close_door",
        "open_door",
        "close_door",
        "press_b",
        "wait",
        "open_door",
        "press_b",
        "wait",
        "set_dark",
        "set_bright",
        "open_door",
        "close_door",
        "reset_lamp",
    ]

    def __init__(self) -> None:
        self._actions: list[str] = []
        self._plan: list[str] = []
        self._next_plan_index = 0
        self.memory = CausalMemory()

    def reset(self, actions: list[ActionSpec]) -> None:
        self._actions = [action.name for action in actions]
        self._plan = [
            action for action in self._preferred_plan if action in self._actions
        ]
        self._next_plan_index = 0
        self.memory = CausalMemory()

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        action = self._choose_planned_or_underexplored_action(history)
        prediction = self.memory.predicted_changes_for(action)
        if prediction:
            predicted = ", ".join(sorted(prediction))
            hypothesis = f"Based on prior trials, {action} should change: {predicted}."
        else:
            hypothesis = f"Probe {action}; its direct effects are still uncertain."
        return AgentDecision(action=action, prediction=prediction, hypothesis=hypothesis)

    def observe_transition(self, transition: Transition) -> None:
        self.memory.record(transition)

    def discovered_edges(self) -> set[Edge]:
        return self.memory.discovered_edges()

    def predict(self, action: str, state: State) -> frozenset[str]:
        return self.memory.predicted_changes_for(action)

    def predict_next_state(self, action: str, state: State) -> State:
        return self.memory.predicted_state_for(action, state)

    def _choose_planned_or_underexplored_action(
        self, history: tuple[Transition, ...]
    ) -> str:
        if self._next_plan_index < len(self._plan):
            action = self._plan[self._next_plan_index]
            self._next_plan_index += 1
            return action

        counts = Counter(transition.action for transition in history)
        return min(self._actions, key=lambda action: (counts[action], action))


class PassiveCorrelationAgent:
    name = "passive-correlation"

    _natural_plan = [
        "press_a",
        "wait",
        "reset_lamp",
        "press_a",
        "wait",
        "reset_lamp",
        "close_door",
        "press_b",
        "wait",
        "close_door",
        "press_b",
        "wait",
        "press_a",
        "wait",
        "reset_lamp",
    ]

    def __init__(self) -> None:
        self._actions: list[str] = []
        self._plan: list[str] = []
        self._next_plan_index = 0
        self._state_edges: set[Edge] = set()
        self.memory = CausalMemory()

    def reset(self, actions: list[ActionSpec]) -> None:
        self._actions = [action.name for action in actions]
        self._plan = [
            action for action in self._natural_plan if action in self._actions
        ]
        self._next_plan_index = 0
        self._state_edges = set()
        self.memory = CausalMemory()

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        if self._next_plan_index < len(self._plan):
            action = self._plan[self._next_plan_index]
            self._next_plan_index += 1
        else:
            action = "wait" if "wait" in self._actions else self._actions[0]

        prediction = self._predict_from_correlations(observation)
        return AgentDecision(
            action=action,
            prediction=prediction,
            hypothesis=(
                "Passively observe the next natural transition and infer causes "
                "from state correlations."
            ),
        )

    def observe_transition(self, transition: Transition) -> None:
        self.memory.record(transition)
        true_before = [
            source for source, value in transition.before.items() if value
        ]
        for source in true_before:
            for target in transition.changed:
                if source != target:
                    self._state_edges.add((source, target))

    def discovered_edges(self) -> set[Edge]:
        return set(self._state_edges)

    def predict(self, action: str, state: State) -> frozenset[str]:
        return self._predict_from_correlations(state)

    def predict_next_state(self, action: str, state: State) -> State:
        prediction = dict(state)
        for target in self.predict(action, state):
            prediction[target] = not prediction[target]
        return prediction

    def _predict_from_correlations(self, observation: State) -> frozenset[str]:
        return frozenset(
            target
            for source, target in self._state_edges
            if observation.get(source, False)
        )


class RewardSeekingAgent:
    name = "reward-rl"

    def __init__(
        self,
        seed: int | None = None,
        epsilon: float = 0.20,
        learning_rate: float = 0.35,
    ) -> None:
        self._seed = seed
        self._rng = Random(seed)
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._actions: list[str] = []
        self._action_values: dict[str, float] = {}
        self._action_counts: Counter[str] = Counter()
        self.memory = CausalMemory()

    def reset(self, actions: list[ActionSpec]) -> None:
        self._rng = Random(self._seed)
        self._actions = [action.name for action in actions]
        self._action_values = {action: 0.0 for action in self._actions}
        self._action_counts = Counter()
        self.memory = CausalMemory()

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        untried = [
            action for action in self._actions if self._action_counts[action] == 0
        ]
        if untried:
            action = untried[0]
        elif self._rng.random() < self._epsilon:
            action = self._rng.choice(self._actions)
        else:
            action = max(
                self._actions,
                key=lambda candidate: (
                    self._action_values.get(candidate, 0.0),
                    -self._action_counts[candidate],
                    candidate,
                ),
            )
        prediction = self.memory.predicted_changes_for(action)
        return AgentDecision(
            action=action,
            prediction=prediction,
            hypothesis=(
                "Reward RL chooses actions by learned scalar return, not by "
                "mechanism uncertainty."
            ),
        )

    def observe_transition(self, transition: Transition) -> None:
        self.memory.record(transition)
        action = transition.action
        reward = task_utility(transition.after)
        previous = self._action_values.get(action, 0.0)
        self._action_values[action] = previous + self._learning_rate * (
            reward - previous
        )
        self._action_counts[action] += 1

    def discovered_edges(self) -> set[Edge]:
        return self.memory.discovered_edges()

    def predict(self, action: str, state: State) -> frozenset[str]:
        return self.memory.predicted_changes_for(action)

    def predict_next_state(self, action: str, state: State) -> State:
        return self.memory.predicted_state_for(action, state)


class RewardTransferAgent(RewardSeekingAgent):
    name = "reward-transfer"

    def start_new_environment(self, keep_priors: bool = True) -> None:
        if not keep_priors:
            self._action_values = {action: 0.0 for action in self._actions}
            self._action_counts = Counter()
        self.memory = CausalMemory()


class RankingLossPredictorAgent:
    """Scalar intervention-ranking predictor without mechanism memory."""

    name = "ranking-loss-predictor"

    def __init__(
        self,
        seed: int | None = None,
        epsilon: float = 0.10,
        learning_rate: float = 0.40,
    ) -> None:
        self._seed = seed
        self._rng = Random(seed)
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._actions: list[str] = []
        self._action_scores: dict[str, float] = {}
        self._action_counts: Counter[str] = Counter()
        self.memory = CausalMemory()

    def reset(self, actions: list[ActionSpec]) -> None:
        self._rng = Random(self._seed)
        self._actions = [action.name for action in actions]
        self._action_scores = {action: 0.0 for action in self._actions}
        self._action_counts = Counter()
        self.memory = CausalMemory()

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        untried = [
            action for action in self._actions if self._action_counts[action] == 0
        ]
        if untried:
            action = untried[0]
        elif self._rng.random() < self._epsilon:
            action = self._rng.choice(self._actions)
        else:
            action = max(
                self._actions,
                key=lambda candidate: (
                    self._action_scores.get(candidate, 0.0),
                    -self._action_counts[candidate],
                    candidate,
                ),
            )
        return AgentDecision(
            action=action,
            prediction=frozenset(),
            hypothesis=(
                "Ranking predictor optimizes scalar intervention preference "
                "rather than action-target mechanisms."
            ),
        )

    def observe_transition(self, transition: Transition) -> None:
        self.memory.record(transition)
        action = transition.action
        utility = task_utility(transition.after)
        previous = self._action_scores.get(action, 0.0)
        self._action_scores[action] = previous + self._learning_rate * (
            utility - previous
        )
        self._action_counts[action] += 1

    def discovered_edges(self) -> set[Edge]:
        return set()

    def predict(self, action: str, state: State) -> frozenset[str]:
        return frozenset()

    def predict_next_state(self, action: str, state: State) -> State:
        return dict(state)

    def start_new_environment(self, keep_priors: bool = True) -> None:
        self.memory = CausalMemory()
        if not keep_priors:
            self._action_scores = {action: 0.0 for action in self._actions}
            self._action_counts = Counter()


class CausalCoreAgent:
    name = "causal-core"

    def __init__(
        self,
        min_effect_observations: int = 1,
        min_action_observations: int = 1,
        min_effect_lift: float = 0.0,
    ) -> None:
        self.core = CausalCore(
            min_effect_observations=min_effect_observations,
            min_action_observations=min_action_observations,
            min_effect_lift=min_effect_lift,
        )
        self.memory = self.core.memory

    def reset(self, actions: list[ActionSpec]) -> None:
        self.core.reset(actions)
        self.memory = self.core.memory

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        decision = self.core.choose_action(observation)
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=f"Causal core chose {decision.action}: {decision.reason}.",
        )

    def observe_transition(self, transition: Transition) -> None:
        self.core.update(transition)
        self.memory = self.core.memory

    def discovered_edges(self) -> set[Edge]:
        return self.core.discovered_edges()

    def predict(self, action: str, state: State) -> frozenset[str]:
        return self.core.predict(action, state)

    def predict_next_state(self, action: str, state: State) -> State:
        return self.core.predict_next_state(action, state)

    def condition_hints(self) -> dict[Edge, list[str]]:
        return self.core.condition_hints()

    def start_new_environment(self, keep_priors: bool = True) -> None:
        self.core.start_new_environment(keep_priors=keep_priors)
        self.memory = self.core.memory


class HypothesisTestingCausalAgent(CausalCoreAgent):
    name = "causal-core-active"

    def __init__(self) -> None:
        super().__init__()
        self._available_actions: set[str] = set()
        self._setter_priors: dict[tuple[str, bool], str] = {}
        self._planned_actions: list[str] = []
        self._active_reason = ""

    def reset(self, actions: list[ActionSpec]) -> None:
        super().reset(actions)
        self._available_actions = {action.name for action in actions}
        self._setter_priors = _setter_priors_from_actions(actions)
        self._planned_actions = []
        self._active_reason = ""

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        if self._planned_actions:
            action = self._planned_actions.pop(0)
            return AgentDecision(
                action=action,
                prediction=self.core.predict(action, observation),
                hypothesis=self._active_reason,
            )

        plan = self._disambiguation_plan(observation)
        if plan:
            action = plan[0]
            self._planned_actions = plan[1:]
            return AgentDecision(
                action=action,
                prediction=self.core.predict(action, observation),
                hypothesis=self._active_reason,
            )

        decision = self.core.choose_action(observation)
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Hypothesis-testing causal core fell back to "
                f"{decision.action}: {decision.reason}."
            ),
        )

    def start_new_environment(self, keep_priors: bool = True) -> None:
        super().start_new_environment(keep_priors=keep_priors)
        self._planned_actions = []
        self._active_reason = ""

    def _disambiguation_plan(self, state: State) -> list[str]:
        for (
            action,
            target,
            effect_value,
            candidates,
            test_candidates,
        ) in self._ambiguous_edges():
            plan = self._best_gate_test_plan(
                state=state,
                action=action,
                target=target,
                effect_value=effect_value,
                candidates=candidates,
                test_candidates=test_candidates,
            )
            if plan:
                candidate_text = ", ".join(
                    f"{variable}={value}" for variable, value in candidates
                )
                self._active_reason = (
                    "Actively disambiguate competing causal gates for "
                    f"{action}->{target}: {candidate_text}."
                )
                return plan
        return []

    def _ambiguous_edges(
        self,
    ) -> list[
        tuple[
            str,
            str,
            bool,
            tuple[tuple[str, bool], ...],
            tuple[tuple[str, bool], ...],
        ]
    ]:
        transitions = self.core.memory.transitions
        edges: list[
            tuple[
                str,
                str,
                bool,
                tuple[tuple[str, bool], ...],
                tuple[tuple[str, bool], ...],
            ]
        ] = []
        changed_targets = sorted(
            {
                (transition.action, target)
                for transition in transitions
                for target in transition.changed
            },
            key=lambda item: (not item[0].startswith("press_"), item[0], item[1]),
        )
        for action, target in changed_targets:
            if target == "sound":
                continue
            positives = [
                transition
                for transition in transitions
                if transition.action == action and target in transition.changed
            ]
            if not positives:
                continue
            effect_values = {transition.after[target] for transition in positives}
            if len(effect_values) != 1:
                continue
            effect_value = next(iter(effect_values))
            negatives = [
                transition.before
                for transition in transitions
                if transition.action == action
                and target not in transition.changed
                and transition.before.get(target) != effect_value
            ]
            changed_with_target = set().union(
                *(transition.changed for transition in positives)
            )
            variables = sorted(
                set().union(*(transition.before.keys() for transition in positives))
            )
            candidates: list[tuple[str, bool]] = []
            for variable in variables:
                if variable == target or variable in changed_with_target:
                    continue
                values = {transition.before.get(variable) for transition in positives}
                if len(values) != 1:
                    continue
                value = next(iter(values))
                if not isinstance(value, bool):
                    continue
                if self._setter_action(variable, not value, disallowed=action) is None:
                    continue
                candidates.append((variable, value))

            if len(candidates) >= 2:
                candidate_tuple = tuple(candidates)
                test_candidates = tuple(
                    candidate
                    for candidate in candidate_tuple
                    if not self._has_disambiguating_test(
                        action=action,
                        target=target,
                        effect_value=effect_value,
                        candidate=candidate,
                        candidates=candidate_tuple,
                    )
                )
                if test_candidates:
                    edges.append(
                        (
                            action,
                            target,
                            effect_value,
                            candidate_tuple,
                            test_candidates,
                        )
                    )
        return edges

    def _has_disambiguating_test(
        self,
        action: str,
        target: str,
        effect_value: bool,
        candidate: tuple[str, bool],
        candidates: tuple[tuple[str, bool], ...],
    ) -> bool:
        candidate_variable, candidate_value = candidate
        for transition in self.core.memory.transitions:
            if transition.action != action:
                continue
            if transition.before.get(target) == effect_value:
                continue
            if transition.before.get(candidate_variable) == candidate_value:
                continue
            other_candidates_match = all(
                transition.before.get(variable) == value
                for variable, value in candidates
                if variable != candidate_variable
            )
            if other_candidates_match:
                return True
        return False

    def _best_gate_test_plan(
        self,
        state: State,
        action: str,
        target: str,
        effect_value: bool,
        candidates: tuple[tuple[str, bool], ...],
        test_candidates: tuple[tuple[str, bool], ...],
    ) -> list[str]:
        possible_plans: list[list[str]] = []
        for flip_variable, flip_value in test_candidates:
            assignment = []
            for variable, value in candidates:
                if variable == flip_variable:
                    assignment.append((variable, not flip_value))
                else:
                    assignment.append((variable, value))
            plan = self._assignment_plan(
                state=state,
                assignment=tuple(assignment),
                action=action,
                target=target,
                effect_value=effect_value,
            )
            if plan:
                possible_plans.append(plan)
        if not possible_plans:
            return []
        shortest_length = min(len(plan) for plan in possible_plans)
        for plan in possible_plans:
            if len(plan) == shortest_length:
                return plan
        return possible_plans[0]

    def _assignment_plan(
        self,
        state: State,
        assignment: tuple[tuple[str, bool], ...],
        action: str,
        target: str,
        effect_value: bool,
    ) -> list[str]:
        simulated = dict(state)
        plan: list[str] = []
        for variable, value in assignment:
            if simulated.get(variable) == value:
                continue
            setter = self._setter_action(variable, value, disallowed=action)
            if setter is None:
                return []
            plan.append(setter)
            simulated[variable] = value

        if simulated.get(target) == effect_value:
            setter = self._setter_action(target, not effect_value, disallowed=action)
            if setter is None:
                return []
            plan.append(setter)
            simulated[target] = not effect_value

        plan.append(action)
        return plan

    def _setter_action(
        self, variable: str, value: bool, disallowed: str | None = None
    ) -> str | None:
        learned = self._learned_setters()
        candidates = [
            learned.get((variable, value)),
            self._setter_priors.get((variable, value)),
        ]
        for action in candidates:
            if (
                action is not None
                and action in self._available_actions
                and action != disallowed
            ):
                return action
        return None

    def _learned_setters(self) -> dict[tuple[str, bool], str]:
        setters: dict[tuple[str, bool], str] = {}
        for transition in self.core.memory.transitions:
            for variable in sorted(transition.changed):
                setters.setdefault(
                    (variable, transition.after[variable]),
                    transition.action,
                )
        return setters


class TemporalCausalCoreAgent(CausalCoreAgent):
    name = "causal-core-temporal"

    def __init__(self, max_delay_steps: int = 2) -> None:
        super().__init__()
        self.max_delay_steps = max_delay_steps
        self._available_actions: set[str] = set()
        self._planned_actions: list[str] = []
        self._delayed_edges: set[Edge] = set()
        self._delayed_effect_values: dict[Edge, bool] = {}
        self._delayed_followups: dict[Edge, tuple[str, int]] = {}
        self._suppressed_followup_edges: set[Edge] = set()
        self._temporal_retest_edges: list[Edge] = []

    def reset(self, actions: list[ActionSpec]) -> None:
        super().reset(actions)
        self._available_actions = {action.name for action in actions}
        self._planned_actions = []
        self._delayed_edges = set()
        self._delayed_effect_values = {}
        self._delayed_followups = {}
        self._suppressed_followup_edges = set()
        self._temporal_retest_edges = []

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        if self._planned_actions:
            action = self._planned_actions.pop(0)
            return AgentDecision(
                action=action,
                prediction=self.predict(action, observation),
                hypothesis="Resolve a planned temporal intervention sequence.",
            )

        plan = self._temporal_probe_plan(observation)
        if plan:
            action = plan[0]
            self._planned_actions = plan[1:]
            return AgentDecision(
                action=action,
                prediction=self.predict(action, observation),
                hypothesis="Probe whether an earlier action has a delayed effect.",
            )

        decision = self.core.choose_action(observation)
        return AgentDecision(
            action=decision.action,
            prediction=self.predict(decision.action, observation),
            hypothesis=f"Temporal causal core chose {decision.action}: {decision.reason}.",
        )

    def observe_transition(self, transition: Transition) -> None:
        self.core.update(transition)
        self.memory = self.core.memory
        self._assign_delayed_effects(transition)

    def discovered_edges(self) -> set[Edge]:
        return (self.core.discovered_edges() - self._suppressed_followup_edges) | set(
            self._delayed_edges
        )

    def predict(self, action: str, state: State) -> frozenset[str]:
        suppressed_targets = {
            target
            for followup_action, target in self._suppressed_followup_edges
            if followup_action == action
        }
        return frozenset(self.core.predict(action, state) - suppressed_targets)

    def predict_temporal(
        self,
        action: str,
        state: State,
        followup_action: str = "wait",
        delay_steps: int = 1,
    ) -> frozenset[str]:
        predictions = set()
        for edge in self._delayed_edges:
            source, target = edge
            if source != action:
                continue
            learned_followup, learned_delay = self._delayed_followups.get(
                edge, (followup_action, delay_steps)
            )
            if learned_followup != followup_action or learned_delay != delay_steps:
                continue
            effect_value = self._delayed_effect_values.get(edge)
            if effect_value is not None and state.get(target) == effect_value:
                continue
            predictions.add(target)
        return frozenset(predictions)

    def predict_next_state(self, action: str, state: State) -> State:
        return self.core.predict_next_state(action, state)

    def condition_hints(self) -> dict[Edge, list[str]]:
        hints = self.core.condition_hints()
        for edge in sorted(self._delayed_edges):
            target = edge[1]
            followup, delay_steps = self._delayed_followups.get(edge, ("wait", 1))
            edge_hints = [f"after {followup} x{delay_steps}"]
            effect_value = self._delayed_effect_values.get(edge)
            if effect_value is not None:
                edge_hints.append(f"sets {target}={effect_value}")
            hints[edge] = edge_hints
        for edge in self._suppressed_followup_edges:
            hints.pop(edge, None)
        return hints

    def start_new_environment(self, keep_priors: bool = True) -> None:
        suppressed_followup_edges = set(self._suppressed_followup_edges)
        super().start_new_environment(keep_priors=keep_priors)
        if keep_priors:
            self.core.prior_edges -= suppressed_followup_edges
            for edge in suppressed_followup_edges:
                self.core.prior_effect_values.pop(edge, None)
                self.core.prior_context_gates.pop(edge, None)
            self._suppressed_followup_edges = suppressed_followup_edges
            self._temporal_retest_edges = sorted(self._delayed_edges)
        else:
            self._delayed_edges = set()
            self._delayed_effect_values = {}
            self._delayed_followups = {}
            self._suppressed_followup_edges = set()
            self._temporal_retest_edges = []
        self._planned_actions = []

    def _temporal_probe_plan(self, state: State) -> list[str]:
        followup_action = self._temporal_followup_action()
        if followup_action is None:
            return []
        retest_plan = self._temporal_retest_plan(state)
        if retest_plan:
            return retest_plan
        delayed_actions = self._delayed_probe_actions()
        for action in delayed_actions:
            if any(edge[0] == action for edge in self._delayed_edges):
                continue
            if self._temporal_probe_was_unproductive(action):
                continue
            if state.get("lamp_on") is True and "reset_lamp" in self._available_actions:
                return ["reset_lamp", action] + [followup_action] * self.max_delay_steps
            return [action] + [followup_action] * self.max_delay_steps
        return []

    def _delayed_probe_actions(self) -> list[str]:
        return sorted(
            action
            for action in self._available_actions
            if "delay" in action and not self._is_reset_action(action)
        )

    def _temporal_probe_was_unproductive(self, action: str) -> bool:
        if self.core.action_counts[action] == 0:
            return False
        return not any(edge[0] == action for edge in self._delayed_edges)

    def _temporal_retest_plan(self, state: State) -> list[str]:
        while self._temporal_retest_edges:
            action, target = self._temporal_retest_edges.pop(0)
            if action not in self._available_actions:
                continue
            plan: list[str] = []
            effect_value = self._delayed_effect_values.get((action, target), True)
            if state.get(target) == effect_value:
                resetter = self._resetter_for(target)
                if resetter is None:
                    continue
                plan.append(resetter)
            followup_action, _ = self._delayed_followups.get(
                (action, target),
                (self._temporal_followup_action(), self.max_delay_steps),
            )
            if followup_action is None:
                continue
            plan.append(action)
            plan.extend([followup_action] * self.max_delay_steps)
            return plan
        return []

    def _resetter_for(self, target: str) -> str | None:
        priors = {
            "lamp_on": "reset_lamp",
            "alarm_on": "reset_alarm",
        }
        resetter = priors.get(target)
        if resetter in self._available_actions:
            return resetter
        return None

    def _temporal_followup_action(self) -> str | None:
        if "wait" in self._available_actions:
            return "wait"
        return None

    def _is_reset_action(self, action: str) -> bool:
        return action.startswith("reset_")

    def _assign_delayed_effects(self, transition: Transition) -> None:
        followup_action = self._temporal_followup_action()
        if followup_action is None or transition.action != followup_action:
            return
        changed_targets = [
            target
            for target in sorted(transition.changed)
            if target != "sound" and transition.before.get(target) is not None
        ]
        if not changed_targets:
            return

        trigger = self._recent_temporal_trigger()
        if trigger is None:
            return

        delay_steps = transition.step - trigger.step
        if delay_steps < 1 or delay_steps > self.max_delay_steps:
            return

        for target in changed_targets:
            edge = (trigger.action, target)
            self._delayed_edges.add(edge)
            self._delayed_effect_values[edge] = transition.after[target]
            self._delayed_followups[edge] = (transition.action, delay_steps)
            self._suppressed_followup_edges.add((transition.action, target))

    def _recent_temporal_trigger(self) -> Transition | None:
        followup_action = self._temporal_followup_action()
        for previous in reversed(self.core.memory.transitions[:-1]):
            if previous.action == followup_action or self._is_reset_action(
                previous.action
            ):
                continue
            if self.core.memory.transitions[-1].step - previous.step > self.max_delay_steps:
                return None
            return previous
        return None


def _semantic_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))


def _semantic_roles(text: str) -> set[str]:
    tokens = _semantic_tokens(text)
    roles: set[str] = set()
    if tokens & {"lamp", "light", "glow", "beacon"}:
        roles.add("lamp")
    if tokens & {"alarm", "siren", "alert"}:
        roles.add("alarm")
    if tokens & {"fan", "rotor", "vent", "blower"}:
        roles.add("fan")
    if tokens & {"sound", "tone", "click", "chime"}:
        roles.add("sound")
    if tokens & {"wait", "settle", "resolve", "pause"}:
        roles.add("wait")
    if tokens & {"reset", "clear", "off", "silence", "dim"}:
        roles.add("reset")
    if tokens & {"delay", "delayed", "pending"}:
        roles.add("delayed")
    if tokens & {"decoy", "dummy"}:
        roles.add("decoy")
    return roles


def _action_roles(action: ActionSpec) -> set[str]:
    return _semantic_roles(f"{action.name} {action.description}")


def _primary_causal_role(text: str) -> str | None:
    roles = _semantic_roles(text)
    for role in ("lamp", "alarm", "fan", "sound"):
        if role in roles:
            return role
    return None


def _semantic_followup_action(actions: list[ActionSpec]) -> str | None:
    for action in actions:
        roles = _action_roles(action)
        if "wait" in roles and "reset" not in roles:
            return action.name
    return None


def _semantic_delayed_action_for_role(
    actions: list[ActionSpec], role: str
) -> str | None:
    fallback: str | None = None
    for action in actions:
        roles = _action_roles(action)
        if role not in roles or "reset" in roles or "decoy" in roles:
            continue
        if "delayed" in roles:
            return action.name
        if fallback is None:
            fallback = action.name
    return fallback


def _semantic_variables_by_role(
    state: State, prefer_non_readout: bool = True
) -> dict[str, str]:
    candidates: dict[str, list[str]] = {}
    for variable in sorted(state):
        role = _primary_causal_role(variable)
        if role is not None:
            candidates.setdefault(role, []).append(variable)
    variables: dict[str, str] = {}
    for role, role_candidates in sorted(candidates.items()):
        preferred = role_candidates
        if prefer_non_readout:
            preferred = [
                variable
                for variable in role_candidates
                if not _looks_like_readout_variable(variable)
            ]
        variables[role] = sorted(preferred or role_candidates)[0]
    return variables


def _looks_like_readout_variable(variable: str) -> bool:
    tokens = _semantic_tokens(variable)
    return bool(tokens & {"readout", "sensor", "indicator", "meter", "aux"})


def _semantic_resetters_by_variable(
    actions: list[ActionSpec], variables_by_role: dict[str, str]
) -> dict[str, str]:
    resetters: dict[str, str] = {}
    for action in actions:
        roles = _action_roles(action)
        if "reset" not in roles:
            continue
        for role, variable in variables_by_role.items():
            if role in roles:
                resetters[variable] = action.name
    return resetters


class PortableTemporalCausalCoreAgent(TemporalCausalCoreAgent):
    """Temporal causal core with a tiny schema-alignment layer."""

    name = "causal-core-temporal-portable"

    def __init__(self, max_delay_steps: int = 2) -> None:
        super().__init__(max_delay_steps=max_delay_steps)
        self._action_specs: list[ActionSpec] = []
        self._portable_followup_action: str | None = None
        self._portable_resetters: dict[str, str] = {}
        self._reset_actions: set[str] = set()

    def reset(self, actions: list[ActionSpec]) -> None:
        super().reset(actions)
        self._action_specs = list(actions)
        self._portable_followup_action = _semantic_followup_action(actions)
        self._portable_resetters = {}
        self._reset_actions = {
            action.name for action in actions if "reset" in _action_roles(action)
        }

    def start_new_environment_with_schema(
        self,
        actions: list[ActionSpec],
        initial_state: State,
        keep_priors: bool = True,
    ) -> None:
        source_edges = sorted(self._delayed_edges)
        source_effect_values = dict(self._delayed_effect_values)
        source_followups = dict(self._delayed_followups)

        self.core.reset(actions)
        self.memory = self.core.memory
        self._available_actions = {action.name for action in actions}
        self._planned_actions = []
        self._action_specs = list(actions)
        self._delayed_edges = set()
        self._delayed_effect_values = {}
        self._delayed_followups = {}
        self._suppressed_followup_edges = set()
        self._temporal_retest_edges = []

        variables_by_role = self._schema_variables_by_role(initial_state)
        self._portable_followup_action = _semantic_followup_action(actions)
        self._portable_resetters = _semantic_resetters_by_variable(
            actions, variables_by_role
        )
        self._reset_actions = {
            action.name for action in actions if "reset" in _action_roles(action)
        } | set(self._portable_resetters.values())

        if not keep_priors or self._portable_followup_action is None:
            return

        for source_edge in source_edges:
            source_action, source_target = source_edge
            if not self._should_transfer_source_target(source_target):
                continue
            role = _primary_causal_role(source_target)
            if role is None:
                continue
            target_action = _semantic_delayed_action_for_role(actions, role)
            target_variable = variables_by_role.get(role)
            if target_action is None or target_variable is None:
                continue
            _, delay_steps = source_followups.get(source_edge, ("wait", 1))
            target_edge = (target_action, target_variable)
            self._delayed_edges.add(target_edge)
            self._delayed_effect_values[target_edge] = source_effect_values.get(
                source_edge, True
            )
            self._delayed_followups[target_edge] = (
                self._portable_followup_action,
                delay_steps,
            )
            self._suppressed_followup_edges.add(
                (self._portable_followup_action, target_variable)
            )

        self._temporal_retest_edges = sorted(self._delayed_edges)

    def _schema_variables_by_role(self, initial_state: State) -> dict[str, str]:
        return _semantic_variables_by_role(initial_state, prefer_non_readout=True)

    def _should_transfer_source_target(self, source_target: str) -> bool:
        return not _looks_like_readout_variable(source_target)

    def _temporal_followup_action(self) -> str | None:
        if (
            self._portable_followup_action is not None
            and self._portable_followup_action in self._available_actions
        ):
            return self._portable_followup_action
        return super()._temporal_followup_action()

    def _resetter_for(self, target: str) -> str | None:
        resetter = self._portable_resetters.get(target)
        if resetter in self._available_actions:
            return resetter
        return super()._resetter_for(target)

    def _delayed_probe_actions(self) -> list[str]:
        semantic_actions = [
            action.name
            for action in self._action_specs
            if action.name in self._available_actions
            and "delayed" in _action_roles(action)
            and not self._is_reset_action(action.name)
        ]
        if semantic_actions:
            return sorted(semantic_actions)
        return super()._delayed_probe_actions()

    def _is_reset_action(self, action: str) -> bool:
        return action in self._reset_actions or super()._is_reset_action(action)


class DiagnosticPortableTemporalCausalCoreAgent(PortableTemporalCausalCoreAgent):
    """Portable temporal core that prioritizes mechanisms after surprise."""

    name = "causal-core-temporal-diagnostic"

    def __init__(self, max_delay_steps: int = 2) -> None:
        super().__init__(max_delay_steps=max_delay_steps)
        self._expected_temporal_effects: list[dict[str, object]] = []

    def reset(self, actions: list[ActionSpec]) -> None:
        super().reset(actions)
        self._expected_temporal_effects = []

    def start_new_environment_with_schema(
        self,
        actions: list[ActionSpec],
        initial_state: State,
        keep_priors: bool = True,
    ) -> None:
        super().start_new_environment_with_schema(
            actions,
            initial_state,
            keep_priors=keep_priors,
        )
        self._expected_temporal_effects = []

    def observe_transition(self, transition: Transition) -> None:
        self._record_temporal_expectations(transition)
        super().observe_transition(transition)
        self._handle_temporal_surprise(transition)

    def _record_temporal_expectations(self, transition: Transition) -> None:
        for edge in sorted(self._delayed_edges):
            action, target = edge
            if action != transition.action:
                continue
            followup_action, learned_delay = self._delayed_followups.get(
                edge,
                (self._temporal_followup_action(), 1),
            )
            if followup_action is None:
                continue
            self._expected_temporal_effects.append(
                {
                    "edge": edge,
                    "target": target,
                    "effect_value": self._delayed_effect_values.get(edge, True),
                    "followup_action": followup_action,
                    "due_step": transition.step + learned_delay,
                    "learned_delay": learned_delay,
                }
            )

    def _handle_temporal_surprise(self, transition: Transition) -> None:
        remaining: list[dict[str, object]] = []
        for expectation in self._expected_temporal_effects:
            due_step = int(expectation["due_step"])
            if transition.step < due_step:
                remaining.append(expectation)
                continue
            if transition.step > due_step:
                continue
            followup_action = str(expectation["followup_action"])
            target = str(expectation["target"])
            effect_value = bool(expectation["effect_value"])
            if transition.action != followup_action:
                remaining.append(expectation)
                continue
            if transition.after.get(target) == effect_value:
                continue

            edge = expectation["edge"]
            if isinstance(edge, tuple) and len(edge) == 2:
                self._promote_suspect_edge((str(edge[0]), str(edge[1])))
            extra_followups = self.max_delay_steps - int(expectation["learned_delay"])
            if extra_followups > 0:
                self._planned_actions = [followup_action] * extra_followups
        self._expected_temporal_effects = remaining

    def _promote_suspect_edge(self, edge: Edge) -> None:
        self._temporal_retest_edges = [
            candidate for candidate in self._temporal_retest_edges if candidate != edge
        ]
        self._temporal_retest_edges.insert(0, edge)


class UnsafeReadoutDiagnosticPortableTemporalCausalCoreAgent(
    DiagnosticPortableTemporalCausalCoreAgent
):
    """Diagnostic portable core without readout-safe schema alignment."""

    name = "causal-core-temporal-diagnostic-unsafe-readout"

    def _schema_variables_by_role(self, initial_state: State) -> dict[str, str]:
        return _semantic_variables_by_role(initial_state, prefer_non_readout=False)

    def _should_transfer_source_target(self, source_target: str) -> bool:
        return True


class RobustCausalCoreAgent(CausalCoreAgent):
    name = "causal-core-robust"

    def __init__(self) -> None:
        super().__init__(min_effect_observations=2)


class NoiseAwareCausalCoreAgent(CausalCoreAgent):
    name = "causal-core-noise-aware"

    def __init__(self) -> None:
        super().__init__(
            min_effect_observations=1,
            min_action_observations=3,
            min_effect_lift=0.15,
        )


class ObservationAdapterCausalCoreAgent(CausalCoreAgent):
    name = "causal-core-observation-adapter"

    def __init__(self) -> None:
        super().__init__(
            min_effect_observations=1,
            min_action_observations=3,
            min_effect_lift=0.15,
        )
        self.adapter = SensorFilteringObservationAdapter()

    def reset(self, actions: list[ActionSpec]) -> None:
        self.adapter.reset()
        self.core.reset(actions)
        self.memory = self.core.memory

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        adapted_observation = self.adapter.transform_state(observation)
        decision = self.core.choose_action(adapted_observation)
        suppressed = sorted(self.adapter.suppressed_variables)
        suffix = ""
        if suppressed:
            suffix = f" Suppressed readouts: {', '.join(suppressed)}."
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Observation adapter + causal core chose {decision.action}: "
                f"{decision.reason}.{suffix}"
            ),
        )

    def observe_transition(self, transition: Transition) -> None:
        adapted_transition = self.adapter.transform_transition(transition)
        self.core.update(adapted_transition)
        self.memory = self.core.memory

    def predict(self, action: str, state: State) -> frozenset[str]:
        return self.core.predict(
            action, self.adapter.transform_state(state, learn=False)
        )

    def predict_next_state(self, action: str, state: State) -> State:
        adapted_state = self.adapter.transform_state(state, learn=False)
        adapted_prediction = self.core.predict_next_state(action, adapted_state)
        prediction = dict(state)
        for variable, value in adapted_prediction.items():
            prediction[variable] = value
        return prediction

    def start_new_environment(self, keep_priors: bool = True) -> None:
        self.adapter.reset()
        self.core.start_new_environment(keep_priors=keep_priors)
        self.memory = self.core.memory


class LearnedObservationAdapterCausalCoreAgent(CausalCoreAgent):
    name = "causal-core-learned-observation-adapter"

    def __init__(self) -> None:
        super().__init__(
            min_effect_observations=1,
            min_action_observations=3,
            min_effect_lift=0.15,
        )
        self.adapter = LearnedReadoutObservationAdapter()
        self._raw_transitions: list[Transition] = []

    def reset(self, actions: list[ActionSpec]) -> None:
        self.adapter.reset()
        self.adapter.set_protected_variables(_action_referenced_variables(actions))
        self._raw_transitions = []
        self.core.reset(actions)
        self.memory = self.core.memory

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        adapted_observation = self.adapter.transform_state(observation)
        decision = self.core.choose_action(adapted_observation)
        suppressed = sorted(self.adapter.suppressed_variables)
        suffix = ""
        if suppressed:
            suffix = f" Learned readouts: {', '.join(suppressed)}."
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Learned observation adapter + causal core chose "
                f"{decision.action}: {decision.reason}.{suffix}"
            ),
        )

    def observe_transition(self, transition: Transition) -> None:
        self._raw_transitions.append(transition)
        previous_suppressed = set(self.adapter.suppressed_variables)
        self.adapter.observe_transition(transition)
        if self.adapter.suppressed_variables != previous_suppressed:
            adapted_transitions = tuple(
                self.adapter.transform_transition(raw, learn=False)
                for raw in self._raw_transitions
            )
            self.core.rebuild(adapted_transitions)
        else:
            self.core.update(
                self.adapter.transform_transition(transition, learn=False)
            )
        self.memory = self.core.memory

    def predict(self, action: str, state: State) -> frozenset[str]:
        return self.core.predict(action, self.adapter.filter_state(state))

    def predict_next_state(self, action: str, state: State) -> State:
        adapted_state = self.adapter.filter_state(state)
        adapted_prediction = self.core.predict_next_state(action, adapted_state)
        prediction = dict(state)
        for variable, value in adapted_prediction.items():
            prediction[variable] = value
        return prediction

    def condition_hints(self) -> dict[Edge, list[str]]:
        hints = self.core.condition_hints()
        for target, rule in sorted(self.adapter.readout_rules.items()):
            hints[("observation", target)] = [
                f"readout of {rule.expression}",
                f"accuracy={rule.accuracy:.2f}",
            ]
        return hints

    def start_new_environment(self, keep_priors: bool = True) -> None:
        self.adapter.reset()
        self._raw_transitions = []
        self.core.start_new_environment(keep_priors=keep_priors)
        self.memory = self.core.memory


class CausalObservationAdapterV2Agent(CausalCoreAgent):
    name = "causal-core-observation-adapter-v2"

    def __init__(self) -> None:
        super().__init__(
            min_effect_observations=1,
            min_action_observations=3,
            min_effect_lift=0.10,
        )
        self.adapter = CausalObservationAdapterV2()
        self._raw_transitions: list[Transition] = []

    def reset(self, actions: list[ActionSpec]) -> None:
        self.adapter.reset()
        self.adapter.set_protected_variables(_action_referenced_variables(actions))
        self._raw_transitions = []
        self.core.reset(actions)
        self.memory = self.core.memory

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        adapted_observation = self.adapter.transform_state(observation)
        decision = self.core.choose_action(adapted_observation)
        suppressed = sorted(self.adapter.suppressed_variables)
        suffix = ""
        if suppressed:
            suffix = f" Suppressed readouts: {', '.join(suppressed)}."
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Observation adapter v2 + causal core chose "
                f"{decision.action}: {decision.reason}.{suffix}"
            ),
        )

    def observe_transition(self, transition: Transition) -> None:
        self._raw_transitions.append(transition)
        self.adapter.observe_transition(transition)
        adapted_transitions = tuple(
            self.adapter.transform_transition(raw, learn=False)
            for raw in self._raw_transitions
        )
        self.core.rebuild(adapted_transitions)
        self.memory = self.core.memory

    def predict(self, action: str, state: State) -> frozenset[str]:
        return self.core.predict(action, self.adapter.filter_state(state))

    def predict_next_state(self, action: str, state: State) -> State:
        adapted_state = self.adapter.filter_state(state)
        adapted_prediction = self.core.predict_next_state(action, adapted_state)
        prediction = dict(state)
        for variable, value in adapted_prediction.items():
            prediction[variable] = value
        return prediction

    def condition_hints(self) -> dict[Edge, list[str]]:
        hints = self.core.condition_hints()
        for target, rule in sorted(self.adapter.readout_rules.items()):
            hints[("observation", target)] = [
                f"readout of {rule.expression}",
                f"accuracy={rule.accuracy:.2f}",
            ]
        return hints

    def start_new_environment(self, keep_priors: bool = True) -> None:
        self.adapter.reset()
        self._raw_transitions = []
        self.core.start_new_environment(keep_priors=keep_priors)
        self.memory = self.core.memory


class HiddenContextObservationAdapterAgent(CausalObservationAdapterV2Agent):
    name = "causal-core-hidden-context-adapter"

    def __init__(self) -> None:
        CausalCoreAgent.__init__(
            self,
            min_effect_observations=1,
            min_action_observations=3,
            min_effect_lift=0.10,
        )
        self.adapter = HiddenContextObservationAdapter()
        self._raw_transitions: list[Transition] = []

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        adapted_observation = self.adapter.transform_state(observation)
        decision = self.core.choose_action(adapted_observation)
        hidden = sorted(self.adapter.hidden_context_hypotheses())
        suffix = ""
        if hidden:
            examples = ", ".join(f"{action}->{target}" for action, target in hidden[:3])
            suffix = f" Hidden-context candidates: {examples}."
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Hidden-context observation adapter + causal core chose "
                f"{decision.action}: {decision.reason}.{suffix}"
            ),
        )

    def condition_hints(self) -> dict[Edge, list[str]]:
        hints = super().condition_hints()
        for (action, target), stats in sorted(
            self.adapter.hidden_context_hypotheses().items()
        ):
            if (action, target) in self.core.discovered_edges():
                hints.setdefault((action, target), []).append(
                    "possible hidden context"
                )
                hints[(action, target)].append(
                    f"success-rate={stats.action_rate:.2f}"
                )
                hints[(action, target)].append(
                    f"lift={stats.lift:.2f}"
                )
        return hints


class LatentContextObservationAdapterAgent(CausalObservationAdapterV2Agent):
    name = "causal-core-latent-context-adapter"

    def __init__(self) -> None:
        CausalCoreAgent.__init__(
            self,
            min_effect_observations=1,
            min_action_observations=3,
            min_effect_lift=0.10,
        )
        self.adapter = LatentContextObservationAdapter()
        self._raw_transitions: list[Transition] = []

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        adapted_observation = self.adapter.transform_state(observation)
        decision = self.core.choose_action(adapted_observation)
        gates = sorted(self.adapter.latent_context_gates().values(), key=_gate_key)
        suffix = ""
        if gates:
            examples = ", ".join(
                f"{gate.action}->{gate.target} ({gate.state_label})"
                for gate in gates[:3]
            )
            suffix = f" Latent gates: {examples}."
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Latent-context observation adapter + causal core chose "
                f"{decision.action}: {decision.reason}.{suffix}"
            ),
        )

    def condition_hints(self) -> dict[Edge, list[str]]:
        hints = super().condition_hints()
        for (action, target), gate in sorted(
            self.adapter.latent_context_gates().items()
        ):
            if (action, target) not in self.core.discovered_edges():
                continue
            hints.setdefault((action, target), []).extend(
                [
                    f"latent gate {gate.latent_name}=enabled",
                    f"latent-state={gate.state_label}",
                    f"eligible-success-rate={gate.success_rate:.2f}",
                    (
                        "eligible-attempts="
                        f"{gate.eligible_count}, successes={gate.success_count}, "
                        f"failures={gate.failure_count}"
                    ),
                    f"best-observed-gate-accuracy={gate.best_observed_gate_accuracy:.2f}",
                ]
            )
        return hints


class StatefulLatentContextObservationAdapterAgent(LatentContextObservationAdapterAgent):
    name = "causal-core-stateful-latent-context-adapter"

    def __init__(self) -> None:
        CausalCoreAgent.__init__(
            self,
            min_effect_observations=1,
            min_action_observations=3,
            min_effect_lift=0.10,
        )
        self.adapter = StatefulLatentContextObservationAdapter()
        self._raw_transitions: list[Transition] = []

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        adapted_observation = self.adapter.transform_state(observation)
        probe = self._probe_candidate(adapted_observation, history)
        if probe is not None:
            gate = probe.gate
            prediction = self.core.predict(gate.action, adapted_observation)
            return AgentDecision(
                action=gate.action,
                prediction=prediction,
                hypothesis=(
                    "Stateful latent-context adapter probes "
                    f"{gate.latent_name}: p_enabled="
                    f"{probe.enabled_probability:.2f}, uncertainty="
                    f"{probe.uncertainty:.2f}, recent="
                    f"{probe.recent_successes}/"
                    f"{probe.recent_successes + probe.recent_failures}."
                ),
            )

        decision = self.core.choose_action(adapted_observation)
        beliefs = sorted(
            self.adapter.latent_state_beliefs().values(),
            key=lambda belief: belief.probe_priority,
            reverse=True,
        )
        suffix = ""
        if beliefs:
            examples = ", ".join(
                (
                    f"{belief.gate.action}->{belief.gate.target} "
                    f"p={belief.enabled_probability:.2f}, "
                    f"u={belief.uncertainty:.2f}"
                )
                for belief in beliefs[:3]
            )
            suffix = f" Latent beliefs: {examples}."
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Stateful latent-context observation adapter + causal core "
                f"chose {decision.action}: {decision.reason}.{suffix}"
            ),
        )

    def _probe_candidate(
        self, adapted_observation: State, history: tuple[Transition, ...]
    ) -> LatentStateBelief | None:
        if sum("probes" in transition.hypothesis for transition in history[-8:]) > 0:
            return None
        discovered = self.core.discovered_edges()
        candidates = sorted(
            self.adapter.latent_state_beliefs().values(),
            key=lambda belief: belief.probe_priority,
            reverse=True,
        )
        for belief in candidates:
            gate = belief.gate
            if (gate.action, gate.target) not in discovered:
                continue
            if gate.target not in adapted_observation:
                continue
            if adapted_observation[gate.target] == gate.effect_value:
                continue
            if belief.uncertainty < self.adapter.min_probe_uncertainty:
                continue
            return belief
        return None

    def condition_hints(self) -> dict[Edge, list[str]]:
        hints = super().condition_hints()
        for (action, target), belief in sorted(
            self.adapter.latent_state_beliefs().items()
        ):
            if (action, target) not in self.core.discovered_edges():
                continue
            hints.setdefault((action, target), []).extend(
                [
                    f"latent-belief-p-enabled={belief.enabled_probability:.2f}",
                    f"latent-uncertainty={belief.uncertainty:.2f}",
                    (
                        "recent-eligible="
                        f"{belief.recent_successes}/"
                        f"{belief.recent_successes + belief.recent_failures}"
                    ),
                    f"last-latent-observation={belief.last_observation}",
                    f"probe-priority={belief.probe_priority:.2f}",
                ]
            )
        return hints


class PersistentLatentContextObservationAdapterAgent(
    StatefulLatentContextObservationAdapterAgent
):
    name = "causal-core-persistent-latent-context-adapter"

    def __init__(self) -> None:
        CausalCoreAgent.__init__(
            self,
            min_effect_observations=1,
            min_action_observations=3,
            min_effect_lift=0.10,
        )
        self.adapter = PersistentLatentContextObservationAdapter()
        self._raw_transitions: list[Transition] = []

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        adapted_observation = self.adapter.transform_state(observation)
        probe = self._probe_candidate(adapted_observation, history)
        if probe is not None:
            gate = probe.gate
            prediction = self.core.predict(gate.action, adapted_observation)
            return AgentDecision(
                action=gate.action,
                prediction=prediction,
                hypothesis=(
                    "Persistent latent-context adapter probes "
                    f"{gate.latent_name}: p_enabled="
                    f"{probe.enabled_probability:.2f}, confidence="
                    f"{probe.confidence:.2f}, uncertainty="
                    f"{probe.uncertainty:.2f}, drift="
                    f"{probe.recent_drift:.2f}."
                ),
            )

        decision = self.core.choose_action(adapted_observation)
        beliefs = sorted(
            self.adapter.latent_state_beliefs().values(),
            key=lambda belief: belief.probe_priority,
            reverse=True,
        )
        suffix = ""
        if beliefs:
            examples = ", ".join(
                (
                    f"{belief.gate.action}->{belief.gate.target} "
                    f"p={belief.enabled_probability:.2f}, "
                    f"c={belief.confidence:.2f}, "
                    f"d={belief.recent_drift:.2f}"
                )
                for belief in beliefs[:3]
            )
            suffix = f" Persistent latent beliefs: {examples}."
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Persistent latent-context observation adapter + causal core "
                f"chose {decision.action}: {decision.reason}.{suffix}"
            ),
        )

    def condition_hints(self) -> dict[Edge, list[str]]:
        hints = super().condition_hints()
        for (action, target), belief in sorted(
            self.adapter.latent_state_beliefs().items()
        ):
            if (action, target) not in self.core.discovered_edges():
                continue
            hints.setdefault((action, target), []).extend(
                [
                    f"persistent-p-enabled={belief.enabled_probability:.2f}",
                    f"persistent-confidence={belief.confidence:.2f}",
                    f"persistent-total={belief.total_successes}/"
                    f"{belief.total_successes + belief.total_failures}",
                    f"recent-drift={belief.recent_drift:.2f}",
                ]
            )
        return hints


class ProactiveLatentContextObservationAdapterAgent(
    PersistentLatentContextObservationAdapterAgent
):
    name = "causal-core-proactive-latent-context-adapter"

    def __init__(self) -> None:
        super().__init__()
        self._available_actions: set[str] = set()
        self._setter_priors: dict[tuple[str, bool], str] = {}
        self._action_text_by_name: dict[str, str] = {}
        self._planned_actions: list[str] = []
        self._active_reason = ""
        self._candidate_refresh_interval = 4
        self._max_pre_gate_candidates = 40
        self._pre_gate_cache_step = -1
        self._pre_gate_candidate_cache: list[dict[str, object]] = []
        self._learned_setters_cache_step = -1
        self._learned_setters_cache: dict[tuple[str, bool], str] = {}

    def reset(self, actions: list[ActionSpec]) -> None:
        super().reset(actions)
        self._available_actions = {action.name for action in actions}
        self._setter_priors = _setter_priors_from_actions(actions)
        self._action_text_by_name = {
            action.name: f"{action.name} {action.description}" for action in actions
        }
        self._planned_actions = []
        self._active_reason = ""
        self._clear_planning_caches()

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        adapted_observation = self.adapter.transform_state(observation)
        if self._planned_actions:
            action = self._planned_actions.pop(0)
            return AgentDecision(
                action=action,
                prediction=self.core.predict(action, adapted_observation),
                hypothesis=self._active_reason,
            )

        probe = self._probe_candidate(adapted_observation, history)
        if probe is not None:
            gate = probe.gate
            prediction = self.core.predict(gate.action, adapted_observation)
            return AgentDecision(
                action=gate.action,
                prediction=prediction,
                hypothesis=(
                    "Proactive latent-context adapter probes established gate "
                    f"{gate.latent_name}: p_enabled="
                    f"{probe.enabled_probability:.2f}, confidence="
                    f"{probe.confidence:.2f}, uncertainty="
                    f"{probe.uncertainty:.2f}."
                ),
            )

        plan = self._pre_gate_probe_plan(adapted_observation, history)
        if plan:
            action = plan[0]
            self._planned_actions = plan[1:]
            return AgentDecision(
                action=action,
                prediction=self.core.predict(action, adapted_observation),
                hypothesis=self._active_reason,
            )

        decision = self.core.choose_action(adapted_observation)
        beliefs = sorted(
            self.adapter.latent_state_beliefs().values(),
            key=lambda belief: belief.probe_priority,
            reverse=True,
        )
        suffix = ""
        if beliefs:
            examples = ", ".join(
                (
                    f"{belief.gate.action}->{belief.gate.target} "
                    f"p={belief.enabled_probability:.2f}, "
                    f"c={belief.confidence:.2f}, "
                    f"d={belief.recent_drift:.2f}"
                )
                for belief in beliefs[:3]
            )
            suffix = f" Persistent latent beliefs: {examples}."
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Proactive latent-context observation adapter + causal core "
                f"chose {decision.action}: {decision.reason}.{suffix}"
            ),
        )

    def start_new_environment(self, keep_priors: bool = True) -> None:
        super().start_new_environment(keep_priors=keep_priors)
        self._planned_actions = []
        self._active_reason = ""
        self._clear_planning_caches()

    def _clear_planning_caches(self) -> None:
        self._pre_gate_cache_step = -1
        self._pre_gate_candidate_cache = []
        self._learned_setters_cache_step = -1
        self._learned_setters_cache = {}

    def _pre_gate_probe_plan(
        self, state: State, history: tuple[Transition, ...]
    ) -> list[str]:
        if self._recently_ran_pre_gate_probe(history):
            return []

        formed_gates = set(self.adapter.latent_context_gates())
        for candidate in self._pre_gate_candidates():
            action = candidate["action"]
            target = candidate["target"]
            effect_value = candidate["effect_value"]
            edge = (action, target)
            if edge in formed_gates:
                continue
            if target not in state:
                continue
            resetter = self._setter_action(target, not effect_value, disallowed=action)
            if resetter is None:
                continue
            if action not in self._available_actions:
                continue

            plan: list[str] = []
            for variable, value in candidate["setup"]:
                if state.get(variable) == value:
                    continue
                setter = self._setter_action(variable, value, disallowed=action)
                if setter is not None and setter not in plan:
                    plan.append(setter)
            if state.get(target) == effect_value:
                plan.append(resetter)
            plan.append(action)
            if len(plan) == 1 and state.get(target) == effect_value:
                continue

            self._active_reason = (
                "Proactive latent-context pre-gate probe for "
                f"{action}->{target}: eligible="
                f"{candidate['eligible_count']}, successes="
                f"{candidate['success_count']}, failures="
                f"{candidate['failure_count']}, lift="
                f"{candidate['lift']:.2f}."
            )
            return plan
        return []

    def _recently_ran_pre_gate_probe(self, history: tuple[Transition, ...]) -> bool:
        recent = history[-6:]
        return any(
            "Proactive latent-context pre-gate probe" in item.hypothesis
            for item in recent
        )

    def _pre_gate_candidates(self) -> list[dict[str, object]]:
        transition_count = len(self._raw_transitions)
        if (
            self._pre_gate_cache_step >= 0
            and transition_count - self._pre_gate_cache_step
            < self._candidate_refresh_interval
        ):
            return list(self._pre_gate_candidate_cache)

        candidates: list[dict[str, object]] = []
        actions = sorted({transition.action for transition in self._raw_transitions})
        targets = sorted(
            {
                target
                for transition in self._raw_transitions
                for target in set(transition.before) | set(transition.after)
                if target not in self.adapter.suppressed_variables
            }
        )
        for action in actions:
            for target in targets:
                candidate = self._pre_gate_candidate(action, target)
                if candidate is not None:
                    candidates.append(candidate)
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate["missing_eligible"],
                -candidate["success_count"],
                -candidate["failure_count"],
                candidate["action"],
                candidate["target"],
            ),
        )[: self._max_pre_gate_candidates]
        self._pre_gate_cache_step = transition_count
        self._pre_gate_candidate_cache = ranked
        return list(ranked)

    def _pre_gate_candidate(
        self, action: str, target: str
    ) -> dict[str, object] | None:
        if action in self._direct_control_actions():
            return None
        stats = self.adapter.lift_for(action, target)
        if stats.dominant_value is None:
            return None
        if stats.lift < self.adapter.hidden_context_lift_floor:
            return None
        if stats.background_rate > self.adapter.hidden_context_max_background_rate:
            return None
        if stats.dominant_value_rate < 0.70:
            return None
        if (
            self._setter_action(
                target,
                not stats.dominant_value,
                disallowed=action,
            )
            is None
        ):
            return None

        successes, failures = self.adapter._eligible_effect_examples(
            action=action,
            target=target,
            effect_value=stats.dominant_value,
        )
        success_count = len(successes)
        failure_count = len(failures)
        eligible_count = success_count + failure_count
        if eligible_count < 3:
            return None
        if success_count < 1 or failure_count < 1:
            return None
        success_rate = success_count / eligible_count
        if success_rate < 0.15 or success_rate > 0.85:
            return None
        setup = self._observed_setup_requirements(
            target=target,
            successes=successes,
            failures=failures,
            disallowed=action,
        )
        return {
            "action": action,
            "target": target,
            "effect_value": stats.dominant_value,
            "eligible_count": eligible_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "lift": stats.lift,
            "setup": setup,
            "missing_eligible": max(
                self.adapter.min_latent_eligible_attempts - eligible_count,
                0,
            ),
        }

    def _observed_setup_requirements(
        self,
        target: str,
        successes: list[State],
        failures: list[State],
        disallowed: str,
    ) -> tuple[tuple[str, bool], ...]:
        if not successes:
            return ()
        variables = sorted(set().union(*(state.keys() for state in successes)))
        candidates: list[tuple[int, str, bool]] = []
        for variable in variables:
            if variable == target or variable in self.adapter.suppressed_variables:
                continue
            values = {state.get(variable) for state in successes}
            if len(values) != 1:
                continue
            value = next(iter(values))
            if value is None:
                continue
            if self._setter_action(variable, value, disallowed=disallowed) is None:
                continue
            rejected_failures = sum(state.get(variable) != value for state in failures)
            candidates.append((rejected_failures, variable, value))

        selected: list[tuple[str, bool]] = []
        remaining = list(failures)
        for _, variable, value in sorted(candidates, reverse=True):
            rejected_failures = sum(
                state.get(variable) != value for state in remaining
            )
            selected.append((variable, value))
            if rejected_failures:
                remaining = [
                    state for state in remaining if state.get(variable) == value
                ]
            if len(selected) >= 4:
                break
        return tuple(sorted(selected))

    def _setter_action(
        self, variable: str, value: bool, disallowed: str | None = None
    ) -> str | None:
        candidates = [
            self._learned_setters().get((variable, value)),
            self._setter_priors.get((variable, value)),
        ]
        for action in candidates:
            if (
                action is not None
                and action in self._available_actions
                and action != disallowed
            ):
                return action
        return None

    def _learned_setters(self) -> dict[tuple[str, bool], str]:
        transition_count = len(self.core.memory.transitions)
        if self._learned_setters_cache_step == transition_count:
            return self._learned_setters_cache

        setters: dict[tuple[str, bool], str] = {}
        for transition in self.core.memory.transitions:
            for variable in sorted(transition.changed):
                setters.setdefault(
                    (variable, transition.after[variable]),
                    transition.action,
                )
        self._learned_setters_cache_step = transition_count
        self._learned_setters_cache = setters
        return setters

    def _direct_control_actions(self) -> set[str]:
        return set(self._setter_priors.values())


class ControlExperimentPlannerLatentContextAgent(
    ProactiveLatentContextObservationAdapterAgent
):
    name = "causal-core-control-experiment-planner"

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        adapted_observation = self.adapter.transform_state(observation)

        probe = self._probe_candidate(adapted_observation, history)
        if probe is not None:
            gate = probe.gate
            prediction = self.core.predict(gate.action, adapted_observation)
            return AgentDecision(
                action=gate.action,
                prediction=prediction,
                hypothesis=(
                    "Control experiment planner probes established gate "
                    f"{gate.latent_name}: p_enabled="
                    f"{probe.enabled_probability:.2f}, confidence="
                    f"{probe.confidence:.2f}, uncertainty="
                    f"{probe.uncertainty:.2f}."
                ),
            )

        control_action = self._control_experiment_action(
            adapted_observation,
            history,
        )
        if control_action is not None:
            action, reason = control_action
            return AgentDecision(
                action=action,
                prediction=self.core.predict(action, adapted_observation),
                hypothesis=reason,
            )

        decision = self.core.choose_action(adapted_observation)
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Control experiment planner fell back to "
                f"{decision.action}: {decision.reason}."
            ),
        )

    def _control_experiment_action(
        self,
        state: State,
        history: tuple[Transition, ...],
    ) -> tuple[str, str] | None:
        formed_gates = set(self.adapter.latent_context_gates())
        for candidate in self._pre_gate_candidates():
            action = str(candidate["action"])
            target = str(candidate["target"])
            effect_value = bool(candidate["effect_value"])
            edge = (action, target)
            if edge in formed_gates:
                continue
            trial_action = self._controlled_trial_action(
                state=state,
                action=action,
                target=target,
                effect_value=effect_value,
                setup=tuple(candidate["setup"]),
                reason_prefix="Control experiment planner",
                trial_suffix=(
                    f"eligible={candidate['eligible_count']}, successes="
                    f"{candidate['success_count']}, failures="
                    f"{candidate['failure_count']}."
                ),
            )
            if trial_action is not None:
                return trial_action

        return None

    def _controlled_trial_action(
        self,
        state: State,
        action: str,
        target: str,
        effect_value: bool,
        setup: tuple[tuple[str, bool], ...],
        reason_prefix: str,
        trial_suffix: str,
    ) -> tuple[str, str] | None:
        if action not in self._available_actions or target not in state:
            return None
        for variable, value in setup:
            plan = self._control_plan_to_value(
                state=state,
                variable=variable,
                value=value,
                disallowed={action},
                depth=3,
                visited=set(),
            )
            if plan is None:
                return None
            if plan:
                return (
                    plan[0],
                    (
                        f"{reason_prefix}: drive {variable}={value} "
                        f"before testing {action}->{target}."
                    ),
                )

        reset_plan = self._control_plan_to_value(
            state=state,
            variable=target,
            value=not effect_value,
            disallowed={action},
            depth=1,
            visited=set(),
        )
        if reset_plan is None:
            return None
        if reset_plan:
            return (
                reset_plan[0],
                (
                    f"{reason_prefix}: reset {target}={not effect_value} "
                    f"before testing {action}->{target}."
                ),
            )
        return (
            action,
            (
                f"{reason_prefix}: perform eligible trial for "
                f"{action}->{target}; {trial_suffix}"
            ),
        )

    def _visible_gate_stress_candidates(self) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        formed_gates = set(self.adapter.latent_context_gates())
        actions = sorted({transition.action for transition in self._raw_transitions})
        targets = sorted(
            {
                target
                for transition in self._raw_transitions
                for target in set(transition.before) | set(transition.after)
                if target not in self.adapter.suppressed_variables
            }
        )
        for action in actions:
            if action in self._direct_control_actions():
                continue
            if action not in self._available_actions:
                continue
            for target in targets:
                if (action, target) in formed_gates:
                    continue
                if target == "sound":
                    continue
                stats = self.adapter.lift_for(action, target)
                if stats.dominant_value is not True:
                    continue
                if stats.action_count < 3 or stats.effect_count < 1:
                    continue
                if stats.lift < self.adapter.hidden_context_lift_floor:
                    continue
                if self._direct_setter_action(target, False, {action}) is None:
                    continue
                successes, failures = self.adapter._eligible_effect_examples(
                    action=action,
                    target=target,
                    effect_value=True,
                )
                if not successes:
                    continue
                eligible_count = len(successes) + len(failures)
                if eligible_count >= 18 and len(failures) >= 2:
                    continue
                setup = self._stress_setup_requirements(
                    action=action,
                    target=target,
                    successes=successes,
                    failures=failures,
                )
                if not setup:
                    continue
                candidates.append(
                    {
                        "action": action,
                        "target": target,
                        "effect_value": True,
                        "eligible_count": eligible_count,
                        "success_count": len(successes),
                        "failure_count": len(failures),
                        "lift": stats.lift,
                        "setup": setup,
                        "missing_eligible": max(
                            self.adapter.min_latent_eligible_attempts
                            - eligible_count,
                            0,
                        ),
                    }
                )
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate["missing_eligible"],
                -candidate["success_count"],
                candidate["failure_count"],
                -candidate["lift"],
                candidate["action"],
                candidate["target"],
            ),
        )

    def _stress_setup_requirements(
        self,
        action: str,
        target: str,
        successes: list[State],
        failures: list[State],
    ) -> tuple[tuple[str, bool], ...]:
        co_changed = {
            variable
            for transition in self.adapter._raw_transitions
            if transition.action == action and target in transition.changed
            for variable in transition.changed
        }
        variables = sorted(set().union(*(state.keys() for state in successes)))
        candidates: list[tuple[int, int, str, bool]] = []
        for variable in variables:
            if variable == target or variable == "sound":
                continue
            if variable in co_changed or variable in self.adapter.suppressed_variables:
                continue
            values = {state.get(variable) for state in successes}
            if len(values) != 1:
                continue
            value = next(iter(values))
            if value is None:
                continue
            if self._setter_action(variable, value, disallowed=action) is None:
                continue
            rejected_failures = sum(state.get(variable) != value for state in failures)
            directness = 1 if self._direct_setter_action(variable, value, {action}) else 0
            candidates.append((rejected_failures, directness, variable, value))

        selected: list[tuple[str, bool]] = []
        for _, _, variable, value in sorted(candidates, reverse=True):
            selected.append((variable, value))
            if len(selected) >= 4:
                break
        return tuple(sorted(selected))

    def _control_plan_to_value(
        self,
        state: State,
        variable: str,
        value: bool,
        disallowed: set[str],
        depth: int,
        visited: set[tuple[str, bool]],
    ) -> list[str] | None:
        if state.get(variable) == value:
            return []

        direct = self._direct_setter_action(variable, value, disallowed)
        if direct is not None:
            return [direct]

        if depth <= 0:
            return None
        key = (variable, value)
        if key in visited:
            return None
        visited = set(visited)
        visited.add(key)

        for controller in self._controller_candidates(
            variable=variable,
            value=value,
            disallowed=disallowed,
        ):
            action = controller["action"]
            setup = tuple(controller["setup"])
            for setup_variable, setup_value in setup:
                plan = self._control_plan_to_value(
                    state=state,
                    variable=setup_variable,
                    value=setup_value,
                    disallowed=disallowed | {action},
                    depth=depth - 1,
                    visited=visited,
                )
                if plan is None:
                    break
                if plan:
                    return plan
            else:
                return [action]
        return None

    def _controller_candidates(
        self,
        variable: str,
        value: bool,
        disallowed: set[str],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        for action in sorted({transition.action for transition in self._raw_transitions}):
            if action in disallowed or action not in self._available_actions:
                continue
            if action in self._direct_control_actions():
                continue
            successes, failures = self.adapter._eligible_effect_examples(
                action=action,
                target=variable,
                effect_value=value,
            )
            if not successes:
                continue
            stats = self.adapter.lift_for(action, variable)
            if stats.dominant_value != value:
                continue
            if stats.lift < self.adapter.hidden_context_lift_floor:
                continue
            setup = self._observed_setup_requirements(
                target=variable,
                successes=successes,
                failures=failures,
                disallowed=action,
            )
            candidates.append(
                {
                    "action": action,
                    "setup": setup,
                    "success_count": len(successes),
                    "failure_count": len(failures),
                    "lift": stats.lift,
                }
            )
        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate["success_count"],
                candidate["failure_count"],
                -candidate["lift"],
                len(candidate["setup"]),
                candidate["action"],
            ),
        )

    def _direct_setter_action(
        self,
        variable: str,
        value: bool,
        disallowed: set[str],
    ) -> str | None:
        action = self._setter_priors.get((variable, value))
        if action is None:
            return None
        if action not in self._available_actions or action in disallowed:
            return None
        return action

    def discovered_edges(self) -> set[Edge]:
        edges = super().discovered_edges()
        direct_actions = self._direct_control_actions() - {"wait"}
        expected_targets: dict[str, set[str]] = {}
        for (variable, _), action in self._setter_priors.items():
            expected_targets.setdefault(action, set()).add(variable)

        guarded: set[Edge] = set()
        for action, target in edges:
            if action in direct_actions and target not in expected_targets.get(
                action,
                set(),
            ):
                continue
            guarded.add((action, target))
        return guarded

    def condition_hints(self) -> dict[Edge, list[str]]:
        guarded_edges = self.discovered_edges()
        hints = {
            edge: hints
            for edge, hints in super().condition_hints().items()
            if edge in guarded_edges or edge[0] == "observation"
        }
        for edge in list(hints):
            action, _ = edge
            if action not in self._direct_control_actions():
                continue
            hints[edge] = [
                hint
                for hint in hints[edge]
                if "hidden" not in hint.lower() and "latent" not in hint.lower()
            ]
        for edge in list(hints):
            action, target = edge
            if self._hidden_context_hint_allowed(
                action,
                target,
            ) and not self._has_competing_stable_process_target(
                action,
                target,
                hints,
                guarded_edges,
            ):
                continue
            hints[edge] = [
                hint
                for hint in hints[edge]
                if "hidden" not in hint.lower() and "latent" not in hint.lower()
            ]
        for action, target in sorted(guarded_edges):
            if action in self._direct_control_actions() or target == "sound":
                continue
            if not self._hidden_context_hint_allowed(action, target):
                continue
            if self._has_competing_stable_process_target(
                action,
                target,
                hints,
                guarded_edges,
            ):
                continue
            if target in self.adapter.suppressed_variables:
                continue
            stats = self.adapter.lift_for(action, target)
            if stats.dominant_value is None:
                continue
            if stats.background_rate > self.adapter.hidden_context_max_background_rate:
                continue
            successes, failures = self.adapter._eligible_effect_examples(
                action=action,
                target=target,
                effect_value=stats.dominant_value,
            )
            eligible_count = len(successes) + len(failures)
            if (
                eligible_count < self.adapter.min_latent_eligible_attempts
                or len(successes) < self.adapter.min_latent_successes
                or len(failures) < self.adapter.min_latent_failures
            ):
                continue
            success_rate = len(successes) / eligible_count
            if (
                success_rate < self.adapter.min_latent_success_rate
                or success_rate > self.adapter.max_latent_success_rate
            ):
                continue
            _, observed_accuracy = self.adapter._best_observed_separator(
                target=target,
                successes=successes,
                failures=failures,
            )
            if observed_accuracy < self.adapter.max_observed_gate_accuracy:
                continue
            hint_text = " ".join(hints.get((action, target), [])).lower()
            if "hidden" in hint_text or "latent" in hint_text:
                continue
            hints.setdefault((action, target), []).extend(
                [
                    "residual latent evidence after visible separator",
                    (
                        "residual eligible-attempts="
                        f"{eligible_count}, successes={len(successes)}, "
                        f"failures={len(failures)}"
                    ),
                    f"best-visible-separator-accuracy={observed_accuracy:.2f}",
                ]
            )
        return hints

    def _hidden_context_hint_allowed(self, action: str, target: str) -> bool:
        if target == "sound" or target in self.adapter.suppressed_variables:
            return False
        action_roles = _semantic_roles(action)
        target_roles = _semantic_roles(target)
        if action_roles & {"decoy"}:
            return False
        if target_roles & {"sound", "decoy"}:
            return False
        if self._looks_like_visible_context_variable(target):
            return False
        if _looks_like_readout_variable(target):
            return False
        if self._indexed_action_target_conflict(action, target):
            return False
        if self._action_target_semantic_conflict(action, target):
            return False
        can_set_true = self._setter_priors.get((target, True)) is not None
        can_set_false = self._setter_priors.get((target, False)) is not None
        if can_set_true and can_set_false:
            return self._action_target_semantically_aligned(action, target)
        return True

    def _looks_like_visible_context_variable(self, target: str) -> bool:
        tokens = _semantic_tokens(target)
        return (
            target.startswith("ctx_")
            or target.endswith("_ctx")
            or "context" in tokens
        )

    def _action_target_semantically_aligned(self, action: str, target: str) -> bool:
        action_tokens = self._informative_semantic_tokens(
            self._action_text_by_name.get(action, action)
        )
        target_tokens = self._informative_semantic_tokens(target)
        return bool(action_tokens & target_tokens)

    def _action_target_semantic_conflict(self, action: str, target: str) -> bool:
        action_tokens = self._informative_semantic_tokens(
            self._action_text_by_name.get(action, action)
        )
        target_tokens = self._informative_semantic_tokens(target)
        variable_tokens: set[str] = set()
        for variable, _ in self._setter_priors:
            variable_tokens |= self._informative_semantic_tokens(variable)
        action_variable_tokens = action_tokens & variable_tokens
        return bool(action_variable_tokens and not (action_variable_tokens & target_tokens))

    def _indexed_action_target_conflict(self, action: str, target: str) -> bool:
        action_numbers = re.findall(r"\d+", action)
        target_numbers = re.findall(r"\d+", target)
        if len(action_numbers) < 2 or len(target_numbers) < 2:
            return False
        same_family = action_numbers[0] == target_numbers[0]
        different_mechanism = action_numbers[-1] != target_numbers[-1]
        return same_family and different_mechanism

    def _informative_semantic_tokens(self, text: str) -> set[str]:
        ignored = {
            "a",
            "b",
            "c",
            "d",
            "e",
            "on",
            "off",
            "true",
            "false",
            "press",
            "button",
            "direct",
            "directly",
            "set",
            "turn",
        }
        return {
            token
            for token in _semantic_tokens(text)
            if token not in ignored and not token.isdigit() and len(token) > 1
        }

    def _has_competing_stable_process_target(
        self,
        action: str,
        target: str,
        hints: dict[Edge, list[str]],
        discovered_edges: set[Edge],
    ) -> bool:
        if self._action_target_semantically_aligned(action, target):
            return False
        for candidate_action, candidate_target in discovered_edges:
            if candidate_action != action or candidate_target == target:
                continue
            if candidate_target == "sound":
                continue
            if candidate_target in self.adapter.suppressed_variables:
                continue
            if _looks_like_readout_variable(candidate_target):
                continue
            hint_text = " ".join(hints.get((candidate_action, candidate_target), []))
            if "hidden" in hint_text.lower() or "latent" in hint_text.lower():
                continue
            return True
        return False


class ContextSearchControlExperimentPlannerAgent(
    ControlExperimentPlannerLatentContextAgent
):
    name = "causal-core-context-search-planner"

    def __init__(self) -> None:
        super().__init__()
        self._context_search_attempts: set[
            tuple[str, str, tuple[tuple[str, bool], ...]]
        ] = set()
        self._context_search_edges: set[Edge] = set()
        self._max_context_search_candidates = 32
        self._context_search_variable_limit = 4
        self._context_search_setup_limit = 20
        self._context_search_cache_step = -1
        self._context_search_candidate_cache: list[dict[str, object]] = []

    def reset(self, actions: list[ActionSpec]) -> None:
        super().reset(actions)
        self._context_search_attempts = set()
        self._context_search_edges = set()
        self._clear_context_search_cache()

    def start_new_environment(self, keep_priors: bool = True) -> None:
        super().start_new_environment(keep_priors=keep_priors)
        self._context_search_attempts = set()
        self._context_search_edges = set()
        self._clear_context_search_cache()

    def _clear_context_search_cache(self) -> None:
        self._context_search_cache_step = -1
        self._context_search_candidate_cache = []

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        adapted_observation = self.adapter.transform_state(observation)

        if self._planned_actions:
            action = self._planned_actions.pop(0)
            return AgentDecision(
                action=action,
                prediction=self.core.predict(action, adapted_observation),
                hypothesis=self._active_reason,
            )

        probe = self._probe_candidate(adapted_observation, history)
        if probe is not None:
            gate = probe.gate
            prediction = self.core.predict(gate.action, adapted_observation)
            return AgentDecision(
                action=gate.action,
                prediction=prediction,
                hypothesis=(
                    "Context-search planner probes established gate "
                    f"{gate.latent_name}: p_enabled="
                    f"{probe.enabled_probability:.2f}, confidence="
                    f"{probe.confidence:.2f}, uncertainty="
                    f"{probe.uncertainty:.2f}."
                ),
            )

        control_action = self._control_experiment_action(
            adapted_observation,
            history,
        )
        if control_action is not None:
            action, reason = control_action
            return AgentDecision(
                action=action,
                prediction=self.core.predict(action, adapted_observation),
                hypothesis=reason,
            )

        plan = self._context_search_plan(adapted_observation, history)
        if plan:
            action = plan[0]
            self._planned_actions = plan[1:]
            return AgentDecision(
                action=action,
                prediction=self.core.predict(action, adapted_observation),
                hypothesis=self._active_reason,
            )

        decision = self.core.choose_action(adapted_observation)
        return AgentDecision(
            action=decision.action,
            prediction=decision.prediction,
            hypothesis=(
                f"Context-search planner fell back to "
                f"{decision.action}: {decision.reason}."
            ),
        )

    def _context_search_plan(
        self,
        state: State,
        history: tuple[Transition, ...],
    ) -> list[str]:
        if len(history) < 80:
            return []
        if any("Context search" in item.hypothesis for item in history[-2:]):
            return []

        formed_gates = set(self.adapter.latent_context_gates())
        for candidate in self._context_search_candidates():
            action = str(candidate["action"])
            target = str(candidate["target"])
            effect_value = bool(candidate["effect_value"])
            if (action, target) in formed_gates:
                continue
            for setup in self._context_search_setups(candidate):
                signature = (action, target, setup)
                if signature in self._context_search_attempts:
                    continue
                plan = self._context_search_trial_plan(
                    state=state,
                    action=action,
                    target=target,
                    effect_value=effect_value,
                    setup=setup,
                )
                if not plan:
                    continue
                self._context_search_attempts.add(signature)
                self._context_search_edges.add((action, target))
                setup_text = ", ".join(
                    f"{variable}={value}" for variable, value in setup
                )
                if not setup_text:
                    setup_text = "no explicit setup"
                self._active_reason = (
                    "Context search constructs controlled setting for "
                    f"{action}->{target}: {setup_text}; eligible="
                    f"{candidate['eligible_count']}, successes="
                    f"{candidate['success_count']}, failures="
                    f"{candidate['failure_count']}, lift="
                    f"{candidate['lift']:.2f}."
                )
                return plan
        return []

    def condition_hints(self) -> dict[Edge, list[str]]:
        hints = super().condition_hints()
        discovered = self.discovered_edges()
        for action, target in sorted(self._context_search_edges & discovered):
            if action in self._direct_control_actions():
                continue
            if not self._hidden_context_hint_allowed(action, target):
                continue
            if self._has_competing_stable_process_target(
                action,
                target,
                hints,
                discovered,
            ):
                continue
            stats = self.adapter.lift_for(action, target)
            if stats.dominant_value is None:
                continue
            if stats.background_rate > 0.30:
                continue
            successes, failures = self.adapter._eligible_effect_examples(
                action=action,
                target=target,
                effect_value=stats.dominant_value,
            )
            eligible_count = len(successes) + len(failures)
            if eligible_count < 6 or not successes or not failures:
                continue
            success_rate = len(successes) / eligible_count
            if success_rate > 0.85:
                continue
            hints.setdefault((action, target), []).extend(
                [
                    "context-search latent evidence",
                    (
                        "context-search eligible-attempts="
                        f"{eligible_count}, successes={len(successes)}, "
                        f"failures={len(failures)}"
                    ),
                ]
            )
        return hints

    def _context_search_candidates(self) -> list[dict[str, object]]:
        transition_count = len(self._raw_transitions)
        if (
            self._context_search_cache_step >= 0
            and transition_count - self._context_search_cache_step
            < self._candidate_refresh_interval
        ):
            return list(self._context_search_candidate_cache)

        candidates: list[dict[str, object]] = []
        formed_gates = set(self.adapter.latent_context_gates())
        actions = sorted({transition.action for transition in self._raw_transitions})
        targets = sorted(
            {
                target
                for transition in self._raw_transitions
                for target in set(transition.before) | set(transition.after)
                if target not in self.adapter.suppressed_variables
            }
        )
        for action in actions:
            if action in self._direct_control_actions():
                continue
            if action not in self._available_actions:
                continue
            for target in targets:
                if target == "sound" or (action, target) in formed_gates:
                    continue
                stats = self.adapter.lift_for(action, target)
                if stats.dominant_value is None:
                    continue
                if stats.lift < self.adapter.hidden_context_lift_floor:
                    continue
                if stats.action_count < 3 or stats.effect_count < 1:
                    continue
                if stats.background_rate > 0.30:
                    continue
                if stats.dominant_value_rate < 0.80:
                    continue
                if (
                    self._setter_action(
                        target,
                        not stats.dominant_value,
                        disallowed=action,
                    )
                    is None
                ):
                    continue
                successes, failures = self.adapter._eligible_effect_examples(
                    action=action,
                    target=target,
                    effect_value=stats.dominant_value,
                )
                eligible_count = len(successes) + len(failures)
                if len(successes) < 1:
                    continue
                if eligible_count < 1:
                    continue
                if len(successes) >= self.adapter.min_latent_successes:
                    continue
                candidates.append(
                    {
                        "action": action,
                        "target": target,
                        "effect_value": stats.dominant_value,
                        "eligible_count": eligible_count,
                        "success_count": len(successes),
                        "failure_count": len(failures),
                        "lift": stats.lift,
                        "background_rate": stats.background_rate,
                        "successes": successes,
                        "failures": failures,
                    }
                )
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate["success_count"],
                -candidate["lift"],
                -candidate["eligible_count"],
                candidate["background_rate"],
                candidate["action"],
                candidate["target"],
            ),
        )[: self._max_context_search_candidates]
        self._context_search_cache_step = transition_count
        self._context_search_candidate_cache = ranked
        return list(ranked)

    def _context_search_setups(
        self,
        candidate: dict[str, object],
    ) -> list[tuple[tuple[str, bool], ...]]:
        action = str(candidate["action"])
        target = str(candidate["target"])
        successes = list(candidate["successes"])
        failures = list(candidate["failures"])
        setups: list[tuple[tuple[str, bool], ...]] = [()]

        observed = self._observed_setup_requirements(
            target=target,
            successes=successes,
            failures=failures,
            disallowed=action,
        )
        if observed:
            setups.append(observed)

        variables = self._context_search_variables(
            action=action,
            target=target,
            successes=successes,
            failures=failures,
        )
        search_variables = variables[: self._context_search_variable_limit]
        for variable in search_variables:
            for value in self._context_search_values(variable, successes):
                setups.append(((variable, value),))

        pair_values = ((True, False), (False, True), (True, True), (False, False))
        pair_variables = search_variables
        for index, left in enumerate(pair_variables):
            for right in pair_variables[index + 1 :]:
                for left_value, right_value in pair_values:
                    setups.append(
                        tuple(sorted(((left, left_value), (right, right_value))))
                    )

        deduped: list[tuple[tuple[str, bool], ...]] = []
        seen: set[tuple[tuple[str, bool], ...]] = set()
        for setup in setups:
            if setup in seen:
                continue
            seen.add(setup)
            deduped.append(setup)
            if len(deduped) >= self._context_search_setup_limit:
                break
        return deduped

    def _context_search_variables(
        self,
        action: str,
        target: str,
        successes: list[State],
        failures: list[State],
    ) -> list[str]:
        variables = {
            variable
            for variable, _ in self._setter_priors
            if variable != target
            and variable != "sound"
            and variable not in self.adapter.suppressed_variables
        }
        if successes:
            variables |= set().union(*(state.keys() for state in successes))

        ranked: list[tuple[int, int, str]] = []
        for variable in variables:
            if variable == target or variable == "sound":
                continue
            values = {state.get(variable) for state in successes}
            constant_success = len(values) == 1 and None not in values
            rejected_failures = 0
            if constant_success:
                success_value = next(iter(values))
                rejected_failures = sum(
                    state.get(variable) != success_value for state in failures
                )
            controllable_values = sum(
                self._setter_action(variable, value, disallowed=action) is not None
                for value in (False, True)
            )
            if controllable_values == 0 and not constant_success:
                continue
            ranked.append(
                (
                    rejected_failures,
                    controllable_values + int(constant_success),
                    variable,
                )
            )
        return [
            variable
            for _, _, variable in sorted(ranked, reverse=True)
        ]

    def _context_search_values(
        self,
        variable: str,
        successes: list[State],
    ) -> tuple[bool, ...]:
        preferred: list[bool] = []
        values = {state.get(variable) for state in successes}
        if len(values) == 1:
            value = next(iter(values))
            if isinstance(value, bool):
                preferred.append(value)
                preferred.append(not value)
        for value in (True, False):
            if (
                value not in preferred
                and self._setter_action(variable, value, disallowed=None) is not None
            ):
                preferred.append(value)
        return tuple(preferred)

    def _context_search_trial_plan(
        self,
        state: State,
        action: str,
        target: str,
        effect_value: bool,
        setup: tuple[tuple[str, bool], ...],
    ) -> list[str]:
        if action not in self._available_actions or target not in state:
            return []
        plan: list[str] = []
        disallowed = {action}
        for variable, value in setup:
            subplan = self._control_plan_to_value(
                state=state,
                variable=variable,
                value=value,
                disallowed=disallowed,
                depth=3,
                visited=set(),
            )
            if subplan is None:
                return []
            plan.extend(subplan)
        reset_plan = self._control_plan_to_value(
            state=state,
            variable=target,
            value=not effect_value,
            disallowed=disallowed,
            depth=1,
            visited=set(),
        )
        if reset_plan is None:
            return []
        plan.extend(reset_plan)
        plan.append(action)
        return plan


class UnguardedControlExperimentPlannerLatentContextAgent(
    ControlExperimentPlannerLatentContextAgent
):
    """Diagnostic ablation without hidden-label eligibility filtering."""

    name = "causal-core-control-experiment-planner-unguarded"

    def _hidden_context_hint_allowed(self, action: str, target: str) -> bool:
        return target not in self.adapter.suppressed_variables


class UnguardedContextSearchControlExperimentPlannerAgent(
    ContextSearchControlExperimentPlannerAgent
):
    """Diagnostic ablation without hidden-label eligibility filtering."""

    name = "causal-core-context-search-planner-unguarded"

    def _hidden_context_hint_allowed(self, action: str, target: str) -> bool:
        return target not in self.adapter.suppressed_variables


def _gate_key(gate: object) -> tuple[str, str]:
    action = getattr(gate, "action")
    target = getattr(gate, "target")
    return (action, target)


def _action_referenced_variables(actions: list[ActionSpec]) -> set[str]:
    action_text = " ".join(f"{action.name} {action.description}" for action in actions)
    action_text = action_text.lower().replace("_", " ")
    protected: set[str] = set()
    for token in action_text.split():
        clean = token.strip(".,;:!?()[]{}")
        if len(clean) >= 3:
            protected.add(clean)
    return protected


def _setter_priors_from_actions(actions: list[ActionSpec]) -> dict[tuple[str, bool], str]:
    priors = {
        "set_dark": ("dark", True),
        "set_bright": ("dark", False),
        "open_door": ("door_open", True),
        "close_door": ("door_open", False),
        "reset_lamp": ("lamp_on", False),
        "start_fan": ("fan_on", True),
        "stop_fan": ("fan_on", False),
        "reset_alarm": ("alarm_on", False),
        "start_coolant": ("coolant_on", True),
        "stop_coolant": ("coolant_on", False),
        "start_heater": ("heater_on", True),
        "stop_heater": ("heater_on", False),
        "open_valve": ("valve_open", True),
        "close_valve": ("valve_open", False),
        "reset_pressure": ("pressure_high", False),
        "enable_backup": ("backup_on", True),
        "disable_backup": ("backup_on", False),
        "lock_panel": ("locked", True),
        "unlock_panel": ("locked", False),
        "wait": ("sound", False),
    }
    available = {action.name for action in actions}
    parsed_priors: dict[tuple[str, bool], str] = {}
    for action in actions:
        name = action.name
        for pattern, value in (
            (r"^set_(.+)_on$", True),
            (r"^set_(.+)_off$", False),
            (r"^enable_(.+)$", True),
            (r"^disable_(.+)$", False),
            (r"^reset_(.+)$", False),
        ):
            match = re.match(pattern, name)
            if match is not None:
                parsed_priors.setdefault((match.group(1), value), name)
                break

    known_priors = {
        variable_value: action
        for action, variable_value in priors.items()
        if action in available
    }
    return parsed_priors | known_priors


def make_agent(
    name: str, seed: int | None = None
) -> RandomAgent | ActiveCausalAgent | PassiveCorrelationAgent | CausalCoreAgent:
    if name == RandomAgent.name:
        return RandomAgent(seed=seed)
    if name == ActiveCausalAgent.name:
        return ActiveCausalAgent()
    if name == PassiveCorrelationAgent.name:
        return PassiveCorrelationAgent()
    if name == RewardSeekingAgent.name:
        return RewardSeekingAgent(seed=seed)
    if name == RewardTransferAgent.name:
        return RewardTransferAgent(seed=seed)
    if name == RankingLossPredictorAgent.name:
        return RankingLossPredictorAgent(seed=seed)
    if name == CausalCoreAgent.name:
        return CausalCoreAgent()
    if name == HypothesisTestingCausalAgent.name:
        return HypothesisTestingCausalAgent()
    if name == TemporalCausalCoreAgent.name:
        return TemporalCausalCoreAgent()
    if name == PortableTemporalCausalCoreAgent.name:
        return PortableTemporalCausalCoreAgent()
    if name == DiagnosticPortableTemporalCausalCoreAgent.name:
        return DiagnosticPortableTemporalCausalCoreAgent()
    if name == UnsafeReadoutDiagnosticPortableTemporalCausalCoreAgent.name:
        return UnsafeReadoutDiagnosticPortableTemporalCausalCoreAgent()
    if name == RobustCausalCoreAgent.name:
        return RobustCausalCoreAgent()
    if name == NoiseAwareCausalCoreAgent.name:
        return NoiseAwareCausalCoreAgent()
    if name == ObservationAdapterCausalCoreAgent.name:
        return ObservationAdapterCausalCoreAgent()
    if name == LearnedObservationAdapterCausalCoreAgent.name:
        return LearnedObservationAdapterCausalCoreAgent()
    if name == CausalObservationAdapterV2Agent.name:
        return CausalObservationAdapterV2Agent()
    if name == HiddenContextObservationAdapterAgent.name:
        return HiddenContextObservationAdapterAgent()
    if name == LatentContextObservationAdapterAgent.name:
        return LatentContextObservationAdapterAgent()
    if name == StatefulLatentContextObservationAdapterAgent.name:
        return StatefulLatentContextObservationAdapterAgent()
    if name == PersistentLatentContextObservationAdapterAgent.name:
        return PersistentLatentContextObservationAdapterAgent()
    if name == ProactiveLatentContextObservationAdapterAgent.name:
        return ProactiveLatentContextObservationAdapterAgent()
    if name == ControlExperimentPlannerLatentContextAgent.name:
        return ControlExperimentPlannerLatentContextAgent()
    if name == ContextSearchControlExperimentPlannerAgent.name:
        return ContextSearchControlExperimentPlannerAgent()
    if name == UnguardedControlExperimentPlannerLatentContextAgent.name:
        return UnguardedControlExperimentPlannerLatentContextAgent()
    if name == UnguardedContextSearchControlExperimentPlannerAgent.name:
        return UnguardedContextSearchControlExperimentPlannerAgent()
    available = ", ".join(agent_names())
    raise ValueError(f"unknown agent {name!r}; available: {available}")


def agent_names() -> list[str]:
    return [
        ActiveCausalAgent.name,
        CausalCoreAgent.name,
        HypothesisTestingCausalAgent.name,
        TemporalCausalCoreAgent.name,
        PortableTemporalCausalCoreAgent.name,
        DiagnosticPortableTemporalCausalCoreAgent.name,
        UnsafeReadoutDiagnosticPortableTemporalCausalCoreAgent.name,
        RobustCausalCoreAgent.name,
        NoiseAwareCausalCoreAgent.name,
        ObservationAdapterCausalCoreAgent.name,
        LearnedObservationAdapterCausalCoreAgent.name,
        CausalObservationAdapterV2Agent.name,
        HiddenContextObservationAdapterAgent.name,
        LatentContextObservationAdapterAgent.name,
        StatefulLatentContextObservationAdapterAgent.name,
        PersistentLatentContextObservationAdapterAgent.name,
        ProactiveLatentContextObservationAdapterAgent.name,
        ControlExperimentPlannerLatentContextAgent.name,
        ContextSearchControlExperimentPlannerAgent.name,
        UnguardedControlExperimentPlannerLatentContextAgent.name,
        UnguardedContextSearchControlExperimentPlannerAgent.name,
        RewardSeekingAgent.name,
        RewardTransferAgent.name,
        RankingLossPredictorAgent.name,
        RandomAgent.name,
        PassiveCorrelationAgent.name,
    ]
