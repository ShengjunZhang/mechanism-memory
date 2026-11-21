from __future__ import annotations

from .core import State

GOAL_WEIGHTS = {
    "lamp_on": 1.0,
    "door_open": 1.0,
    "alarm_on": 1.0,
    "fan_on": 0.5,
}

PENALTY_WEIGHTS = {
    "sound": -0.1,
}


def task_utility(state: State) -> float:
    utility = 0.0
    for variable, weight in GOAL_WEIGHTS.items():
        if state.get(variable, False):
            utility += weight
    for variable, weight in PENALTY_WEIGHTS.items():
        if state.get(variable, False):
            utility += weight
    return utility
