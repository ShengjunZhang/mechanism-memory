from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

State = dict[str, bool]
Edge = tuple[str, str]


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str


@dataclass(frozen=True)
class AgentDecision:
    action: str
    prediction: frozenset[str]
    hypothesis: str


@dataclass(frozen=True)
class Transition:
    step: int
    action: str
    before: State
    after: State
    prediction: frozenset[str]
    hypothesis: str

    @property
    def changed(self) -> frozenset[str]:
        keys = set(self.before) | set(self.after)
        return frozenset(
            key for key in keys if self.before.get(key) != self.after.get(key)
        )


@dataclass(frozen=True)
class EdgeScore:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    prediction_jaccard: float


class World(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> State:
        ...

    def observe(self) -> State:
        ...

    def actions(self) -> list[ActionSpec]:
        ...

    def step(self, action: str) -> State:
        ...

    def true_edges(self) -> set[Edge]:
        ...


class Agent(Protocol):
    name: str

    def reset(self, actions: list[ActionSpec]) -> None:
        ...

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        ...

    def observe_transition(self, transition: Transition) -> None:
        ...

    def discovered_edges(self) -> set[Edge]:
        ...

    def predict(self, action: str, state: State) -> frozenset[str]:
        ...

    def predict_next_state(self, action: str, state: State) -> State:
        ...
