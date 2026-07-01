from __future__ import annotations

from collections import defaultdict

from .core import Edge, State, Transition


class CausalMemory:
    def __init__(self) -> None:
        self._transitions: list[Transition] = []

    @property
    def transitions(self) -> tuple[Transition, ...]:
        return tuple(self._transitions)

    def record(self, transition: Transition) -> None:
        self._transitions.append(transition)

    def discovered_edges(self) -> set[Edge]:
        edges: set[Edge] = set()
        for transition in self._transitions:
            for variable in transition.changed:
                edges.add((transition.action, variable))
        return edges

    def predicted_changes_for(self, action: str) -> frozenset[str]:
        return frozenset(
            target
            for source, target in self.discovered_edges()
            if source == action
        )

    def effect_value(self, action: str, target: str) -> bool | None:
        values = {
            transition.after[target]
            for transition in self._transitions
            if transition.action == action and target in transition.changed
        }
        if len(values) != 1:
            return None
        return next(iter(values))

    def predicted_state_for(self, action: str, state: State) -> State:
        prediction = dict(state)
        for target in self.predicted_changes_for(action):
            effect_value = self.effect_value(action, target)
            if effect_value is None:
                prediction[target] = not prediction[target]
            else:
                prediction[target] = effect_value
        return prediction

    def condition_hints(self) -> dict[Edge, list[str]]:
        hints: dict[Edge, list[str]] = {}
        grouped: dict[Edge, list[Transition]] = defaultdict(list)
        for transition in self._transitions:
            for edge in self.discovered_edges():
                if transition.action == edge[0]:
                    grouped[edge].append(transition)

        for edge, transitions in grouped.items():
            action, target = edge
            positives = [item.before for item in transitions if target in item.changed]
            negatives = [item.before for item in transitions if target not in item.changed]
            if not positives or not negatives:
                continue
            edge_hints = self._find_context_hints(target, positives, negatives)
            if edge_hints:
                hints[(action, target)] = edge_hints
        return hints

    @staticmethod
    def _find_context_hints(
        target: str, positives: list[State], negatives: list[State]
    ) -> list[str]:
        variables = sorted(set().union(*(state.keys() for state in positives + negatives)))
        hints: list[str] = []
        for variable in variables:
            values_when_changed = {state.get(variable) for state in positives}
            if len(values_when_changed) != 1:
                continue
            value = next(iter(values_when_changed))
            if any(state.get(variable) != value for state in negatives):
                prefix = "before " if variable == target else ""
                hints.append(f"{prefix}{variable}={value}")
        target_hints = [hint for hint in hints if hint.startswith("before ")]
        return target_hints or hints
