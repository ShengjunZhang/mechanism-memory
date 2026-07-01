from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .core import ActionSpec, Edge, State, Transition
from .memory import CausalMemory

StateSignature = tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class CoreDecision:
    action: str
    prediction: frozenset[str]
    reason: str


class CausalCore:
    """A small intervention-selection module for structured agent worlds.

    The core does not know the hidden rules of a world. It only sees boolean
    states, available actions, and transitions. Its first policy is model-free
    but causal in shape: try interventions under diverse contexts, predict
    action effects from prior transitions, and revisit actions whose predictions
    are still under-tested.
    """

    def __init__(
        self,
        min_effect_observations: int = 1,
        min_action_observations: int = 1,
        min_effect_lift: float = 0.0,
    ) -> None:
        self.actions: list[str] = []
        self.min_effect_observations = min_effect_observations
        self.min_action_observations = min_action_observations
        self.min_effect_lift = min_effect_lift
        self.prior_edges: set[Edge] = set()
        self.prior_effect_values: dict[Edge, bool] = {}
        self.prior_context_gates: dict[Edge, tuple[tuple[str, bool], ...]] = {}
        self.memory = CausalMemory()
        self.action_counts: Counter[str] = Counter()
        self.state_action_counts: Counter[tuple[StateSignature, str]] = Counter()
        self.state_visits: Counter[StateSignature] = Counter()
        self.action_value_counts: Counter[tuple[str, str, bool]] = Counter()
        self.successors: dict[tuple[StateSignature, str], StateSignature] = {}
        self.prediction_errors: Counter[str] = Counter()

    def reset(self, actions: list[ActionSpec]) -> None:
        self.actions = [action.name for action in actions]
        self.prior_edges = set()
        self.prior_effect_values = {}
        self.prior_context_gates = {}
        self._reset_observations()

    def start_new_environment(self, keep_priors: bool = True) -> None:
        if keep_priors:
            prior_edges = self.discovered_edges()
            prior_effect_values = {
                edge: value
                for edge in prior_edges
                if (value := self._edge_effect_value(*edge)) is not None
            }
            prior_context_gates = {
                edge: self._learned_context_gate(*edge) for edge in prior_edges
            }
        else:
            prior_edges = set()
            prior_effect_values = {}
            prior_context_gates = {}
        self.prior_edges = prior_edges
        self.prior_effect_values = prior_effect_values
        self.prior_context_gates = prior_context_gates
        self._reset_observations()

    def _reset_observations(self) -> None:
        self.memory = CausalMemory()
        self.action_counts = Counter()
        self.state_action_counts = Counter()
        self.state_visits = Counter()
        self.action_value_counts = Counter()
        self.successors = {}
        self.prediction_errors = Counter()

    def choose_action(self, state: State) -> CoreDecision:
        signature = state_signature(state)

        globally_untried = [
            action for action in self.actions if self.action_counts[action] == 0
        ]
        if globally_untried:
            action = globally_untried[0]
            return CoreDecision(
                action=action,
                prediction=self.predict(action, state),
                reason=f"global novelty: {action} has not been tested yet",
            )

        scored = [
            (self._action_score(action, signature), action)
            for action in self.actions
        ]
        score, action = max(scored, key=lambda item: (item[0], _reverse_name(item[1])))
        prediction = self.predict(action, state)
        return CoreDecision(
            action=action,
            prediction=prediction,
            reason=f"expected information gain score={score:.2f}",
        )

    def predict(self, action: str, state: State) -> frozenset[str]:
        positive_edges = self._candidate_targets_for(action)
        if not positive_edges:
            return frozenset()

        predictions = set()
        for target in positive_edges:
            if self._already_at_learned_effect_value(action, target, state):
                continue
            if self._edge_is_active_in_context(action, target, state):
                predictions.add(target)
        return frozenset(predictions)

    def predict_next_state(self, action: str, state: State) -> State:
        prediction = dict(state)
        for target in self.predict(action, state):
            effect_value = self._edge_effect_value(action, target)
            if effect_value is None:
                prediction[target] = not prediction[target]
            else:
                prediction[target] = effect_value
        return prediction

    def update(self, transition: Transition) -> None:
        before_signature = state_signature(transition.before)
        after_signature = state_signature(transition.after)
        self.memory.record(transition)
        self.action_counts[transition.action] += 1
        self.state_action_counts[(before_signature, transition.action)] += 1
        for variable, value in transition.before.items():
            self.action_value_counts[(transition.action, variable, value)] += 1
        self.state_visits[after_signature] += 1
        self.successors[(before_signature, transition.action)] = after_signature
        if transition.prediction != transition.changed:
            self.prediction_errors[transition.action] += 1

    def rebuild(self, transitions: tuple[Transition, ...]) -> None:
        self._reset_observations()
        for transition in transitions:
            self.update(transition)

    def discovered_edges(self) -> set[Edge]:
        return self._observed_edges() | self.prior_edges

    def condition_hints(self) -> dict[Edge, list[str]]:
        hints: dict[Edge, list[str]] = {}
        for action, target in sorted(self.discovered_edges()):
            edge = (action, target)
            gate = self._learned_context_gate(action, target)
            if not gate and edge in self.prior_context_gates:
                gate = self.prior_context_gates[edge]
            effect_value = self._edge_effect_value(action, target)
            edge_hints = [f"{variable}={value}" for variable, value in gate]
            if effect_value is not None:
                edge_hints.append(f"sets {target}={effect_value}")
            if edge_hints:
                hints[(action, target)] = edge_hints
        return hints

    def _action_score(self, action: str, signature: StateSignature) -> float:
        local_count = self.state_action_counts[(signature, action)]
        global_count = self.action_counts[action]
        known_successor = (signature, action) in self.successors
        prediction_error = self.prediction_errors[action]
        known_edges = [
            edge for edge in self.discovered_edges() if edge[0] == action
        ]
        context_gaps = self._current_context_gaps(action, signature)

        score = 0.0
        score += 8.0 / (1.0 + local_count)
        score += 2.5 / (1.0 + global_count)
        score += 1.75 * context_gaps
        if not known_successor:
            score += 4.0
        if not known_edges:
            score += 1.5
            if context_gaps:
                score += 3.0
        score += min(prediction_error, 3) * 0.75
        return score

    def _current_context_gaps(self, action: str, signature: StateSignature) -> int:
        gaps = 0
        for variable, value in signature:
            if self.action_value_counts[(action, variable, value)] == 0:
                gaps += 1
        return gaps

    def _candidate_targets_for(self, action: str) -> frozenset[str]:
        memory_targets = frozenset(
            target for source, target in self._observed_edges() if source == action
        )
        prior_targets = frozenset(
            target for source, target in self.prior_edges if source == action
        )
        return memory_targets | prior_targets

    def _observed_edges(self) -> set[Edge]:
        edge_counts: Counter[Edge] = Counter()
        action_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        for transition in self.memory.transitions:
            action_counts[transition.action] += 1
            for target in transition.changed:
                edge_counts[(transition.action, target)] += 1
                target_counts[target] += 1
        total_transitions = len(self.memory.transitions)
        return {
            edge
            for edge, count in edge_counts.items()
            if count >= self.min_effect_observations
            and action_counts[edge[0]] >= self.min_action_observations
            and self._effect_lift(
                edge=edge,
                edge_count=count,
                action_count=action_counts[edge[0]],
                target_count=target_counts[edge[1]],
                total_transitions=total_transitions,
            )
            >= self.min_effect_lift
        }

    @staticmethod
    def _effect_lift(
        edge: Edge,
        edge_count: int,
        action_count: int,
        target_count: int,
        total_transitions: int,
    ) -> float:
        if action_count == 0 or total_transitions == 0:
            return 0.0
        action_rate = edge_count / action_count
        background_rate = target_count / total_transitions
        return action_rate - background_rate

    def _edge_is_active_in_context(self, action: str, target: str, state: State) -> bool:
        positives, eligible_negatives = self._edge_examples(action, target)

        if not positives:
            if eligible_negatives:
                return False
            edge = (action, target)
            if edge in self.prior_context_gates:
                return all(
                    state.get(variable) == value
                    for variable, value in self.prior_context_gates[edge]
                )
            return False
        if not eligible_negatives:
            return True

        gate = self._learned_context_gate(action, target)
        if gate:
            return all(state.get(variable) == value for variable, value in gate)

        positive_distance = min(_hamming_distance(state, item) for item in positives)
        negative_distance = min(_hamming_distance(state, item) for item in eligible_negatives)
        return positive_distance <= negative_distance

    def _already_at_learned_effect_value(
        self, action: str, target: str, state: State
    ) -> bool:
        effect_value = self._edge_effect_value(action, target)
        if effect_value is None:
            return False
        return state.get(target) == effect_value

    def _edge_effect_value(self, action: str, target: str) -> bool | None:
        effect_values = {
            transition.after[target]
            for transition in self.memory.transitions
            if transition.action == action and target in transition.changed
        }
        if len(effect_values) != 1:
            return self.prior_effect_values.get((action, target))
        return next(iter(effect_values))

    def _edge_examples(
        self, action: str, target: str
    ) -> tuple[list[State], list[State]]:
        positives: list[State] = []
        eligible_negatives: list[State] = []
        effect_value = self._edge_effect_value(action, target)
        for transition in self.memory.transitions:
            if transition.action != action:
                continue
            if target in transition.changed:
                positives.append(transition.before)
            elif effect_value is None or transition.before.get(target) != effect_value:
                eligible_negatives.append(transition.before)
        return positives, eligible_negatives

    def _learned_context_gate(
        self, action: str, target: str
    ) -> tuple[tuple[str, bool], ...]:
        positives, eligible_negatives = self._edge_examples(action, target)
        if not positives or not eligible_negatives:
            return ()

        co_changed = {
            variable
            for transition in self.memory.transitions
            if transition.action == action and target in transition.changed
            for variable in transition.changed
        }
        variables = sorted(set().union(*(state.keys() for state in positives)))
        candidates: list[tuple[str, bool]] = []
        for variable in variables:
            if variable == target or variable in co_changed:
                continue
            positive_values = {state.get(variable) for state in positives}
            if len(positive_values) == 1:
                value = next(iter(positive_values))
                if value is not None:
                    candidates.append((variable, value))

        remaining = list(eligible_negatives)
        selected: list[tuple[str, bool]] = []
        while remaining:
            best: tuple[str, bool] | None = None
            best_rejected = 0
            for candidate in candidates:
                variable, value = candidate
                rejected = sum(state.get(variable) != value for state in remaining)
                if rejected > best_rejected:
                    best = candidate
                    best_rejected = rejected

            if best is None or best_rejected == 0:
                return ()

            selected.append(best)
            candidates.remove(best)
            variable, value = best
            remaining = [
                state for state in remaining if state.get(variable) == value
            ]

        return tuple(selected)


def state_signature(state: State) -> StateSignature:
    return tuple(sorted(state.items()))


def _hamming_distance(left: State, right: State) -> int:
    keys = set(left) | set(right)
    return sum(left.get(key) != right.get(key) for key in keys)


def _reverse_name(name: str) -> tuple[int, ...]:
    return tuple(-ord(char) for char in name)
