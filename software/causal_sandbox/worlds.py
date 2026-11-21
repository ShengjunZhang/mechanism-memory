from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .core import ActionSpec, Edge, State


class DoorLampWorld:
    """A tiny hidden-rule world with conditional action effects."""

    name = "door-lamp"

    def __init__(self) -> None:
        self._rng = Random()
        self._state: State = {}

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._state = {
            "dark": True,
            "lamp_on": False,
            "door_open": False,
            "sound": False,
        }
        return self.observe()

    def observe(self) -> State:
        return dict(self._state)

    def set_state(self, state: State) -> None:
        expected = set(self._state)
        incoming = set(state)
        if incoming != expected:
            missing = sorted(expected - incoming)
            extra = sorted(incoming - expected)
            raise ValueError(f"state keys mismatch; missing={missing}, extra={extra}")
        self._state = dict(state)

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("set_dark", "Make the room dark."),
            ActionSpec("set_bright", "Make the room bright."),
            ActionSpec("open_door", "Open the door directly."),
            ActionSpec("close_door", "Close the door directly."),
            ActionSpec("reset_lamp", "Turn the lamp off directly."),
            ActionSpec("press_a", "Press button A."),
            ActionSpec("press_b", "Press button B."),
            ActionSpec("wait", "Let transient effects fade."),
        ]

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "set_dark":
            self._state["dark"] = True
        elif action == "set_bright":
            self._state["dark"] = False
        elif action == "open_door":
            self._state["door_open"] = True
        elif action == "close_door":
            self._state["door_open"] = False
        elif action == "reset_lamp":
            self._state["lamp_on"] = False
        elif action == "press_a":
            self._state["sound"] = True
            if self._state["dark"]:
                self._state["lamp_on"] = True
        elif action == "press_b":
            self._state["sound"] = True
            if not self._state["door_open"]:
                self._state["door_open"] = True
        elif action == "wait":
            self._state["sound"] = False

        return self.observe()

    def true_edges(self) -> set[Edge]:
        return {
            ("set_dark", "dark"),
            ("set_bright", "dark"),
            ("open_door", "door_open"),
            ("close_door", "door_open"),
            ("reset_lamp", "lamp_on"),
            ("press_a", "sound"),
            ("press_a", "lamp_on"),
            ("press_b", "sound"),
            ("press_b", "door_open"),
            ("wait", "sound"),
        }


class DoorLampShiftedWorld(DoorLampWorld):
    """Same mechanisms as DoorLampWorld, but a different starting distribution."""

    name = "door-lamp-shifted"

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._state = {
            "dark": False,
            "lamp_on": True,
            "door_open": True,
            "sound": seed is not None and seed % 2 == 0,
        }
        return self.observe()


class DoorLampInvertedWorld(DoorLampWorld):
    """Same interface, but button A lights the lamp in the opposite context."""

    name = "door-lamp-inverted"

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "set_dark":
            self._state["dark"] = True
        elif action == "set_bright":
            self._state["dark"] = False
        elif action == "open_door":
            self._state["door_open"] = True
        elif action == "close_door":
            self._state["door_open"] = False
        elif action == "reset_lamp":
            self._state["lamp_on"] = False
        elif action == "press_a":
            self._state["sound"] = True
            if not self._state["dark"]:
                self._state["lamp_on"] = True
        elif action == "press_b":
            self._state["sound"] = True
            if not self._state["door_open"]:
                self._state["door_open"] = True
        elif action == "wait":
            self._state["sound"] = False

        return self.observe()


class AmbiguousGateWorld(DoorLampWorld):
    """A door-lamp variant where early evidence confounds two context gates.

    The room starts with ``dark`` and ``door_open`` both true. A single
    successful ``press_a`` trial is therefore consistent with two hypotheses:
    button A may require darkness, or it may require the door to be open. The
    correct mechanism is darkness; distinguishing it requires an intervention
    that makes the two candidate gates disagree before pressing A again.
    """

    name = "ambiguous-gate"

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._state = {
            "dark": True,
            "lamp_on": False,
            "door_open": True,
            "sound": False,
        }
        return self.observe()

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("reset_lamp", "Turn the lamp off directly."),
            ActionSpec("press_a", "Press button A."),
            ActionSpec("set_bright", "Make the room bright."),
            ActionSpec("close_door", "Close the door directly."),
            ActionSpec("set_dark", "Make the room dark."),
            ActionSpec("open_door", "Open the door directly."),
            ActionSpec("wait", "Let transient effects fade."),
        ]

    def level4_spec(self) -> dict[str, object]:
        return {
            "target_action": "press_a",
            "target_variable": "lamp_on",
            "effect_value": True,
            "candidate_gates": (("dark", True), ("door_open", True)),
        }


class PanelWorld:
    """A larger boolean control panel with several conditional mechanisms."""

    name = "panel"

    def __init__(self) -> None:
        self._rng = Random()
        self._state: State = {}

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._state = {
            "dark": True,
            "lamp_on": False,
            "door_open": False,
            "sound": False,
            "fan_on": False,
            "alarm_on": False,
        }
        return self.observe()

    def observe(self) -> State:
        return dict(self._state)

    def set_state(self, state: State) -> None:
        expected = set(self._state)
        incoming = set(state)
        if incoming != expected:
            missing = sorted(expected - incoming)
            extra = sorted(incoming - expected)
            raise ValueError(f"state keys mismatch; missing={missing}, extra={extra}")
        self._state = dict(state)

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("set_dark", "Make the room dark."),
            ActionSpec("set_bright", "Make the room bright."),
            ActionSpec("open_door", "Open the door directly."),
            ActionSpec("close_door", "Close the door directly."),
            ActionSpec("reset_lamp", "Turn the lamp off directly."),
            ActionSpec("start_fan", "Turn the fan on directly."),
            ActionSpec("stop_fan", "Turn the fan off directly."),
            ActionSpec("reset_alarm", "Turn the alarm off directly."),
            ActionSpec("press_a", "Press lamp button A."),
            ActionSpec("press_b", "Press door button B."),
            ActionSpec("press_c", "Press alarm button C."),
            ActionSpec("wait", "Let transient effects fade."),
        ]

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "set_dark":
            self._state["dark"] = True
        elif action == "set_bright":
            self._state["dark"] = False
        elif action == "open_door":
            self._state["door_open"] = True
        elif action == "close_door":
            self._state["door_open"] = False
        elif action == "reset_lamp":
            self._state["lamp_on"] = False
        elif action == "start_fan":
            self._state["fan_on"] = True
        elif action == "stop_fan":
            self._state["fan_on"] = False
        elif action == "reset_alarm":
            self._state["alarm_on"] = False
        elif action == "press_a":
            self._state["sound"] = True
            if self._state["dark"]:
                self._state["lamp_on"] = True
        elif action == "press_b":
            self._state["sound"] = True
            if not self._state["door_open"]:
                self._state["door_open"] = True
        elif action == "press_c":
            self._state["sound"] = True
            if self._press_c_alarm_condition():
                self._state["alarm_on"] = True
        elif action == "wait":
            self._state["sound"] = False

        return self.observe()

    def true_edges(self) -> set[Edge]:
        return {
            ("set_dark", "dark"),
            ("set_bright", "dark"),
            ("open_door", "door_open"),
            ("close_door", "door_open"),
            ("reset_lamp", "lamp_on"),
            ("start_fan", "fan_on"),
            ("stop_fan", "fan_on"),
            ("reset_alarm", "alarm_on"),
            ("press_a", "sound"),
            ("press_a", "lamp_on"),
            ("press_b", "sound"),
            ("press_b", "door_open"),
            ("press_c", "sound"),
            ("press_c", "alarm_on"),
            ("wait", "sound"),
        }

    def _press_c_alarm_condition(self) -> bool:
        return self._state["door_open"] and not self._state["fan_on"]


class AmbiguousPanelGateWorld(PanelWorld):
    """A panel variant requiring active discovery of a conjunctive gate.

    The first ``press_c`` trial succeeds while ``door_open`` is true and
    ``fan_on`` is false. This positive example alone is compatible with either
    single-gate hypothesis or with the true conjunctive mechanism:
    ``door_open=true`` and ``fan_on=false`` are both required.
    """

    name = "ambiguous-panel-gate"

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._state = {
            "dark": True,
            "lamp_on": False,
            "door_open": True,
            "sound": False,
            "fan_on": False,
            "alarm_on": False,
        }
        return self.observe()

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("reset_alarm", "Turn the alarm off directly."),
            ActionSpec("press_c", "Press button C."),
            ActionSpec("close_door", "Close the door directly."),
            ActionSpec("open_door", "Open the door directly."),
            ActionSpec("start_fan", "Turn the fan on directly."),
            ActionSpec("stop_fan", "Turn the fan off directly."),
            ActionSpec("wait", "Let transient effects fade."),
        ]

    def true_edges(self) -> set[Edge]:
        return {
            ("reset_alarm", "alarm_on"),
            ("press_c", "sound"),
            ("press_c", "alarm_on"),
            ("close_door", "door_open"),
            ("open_door", "door_open"),
            ("start_fan", "fan_on"),
            ("stop_fan", "fan_on"),
            ("wait", "sound"),
        }

    def level4_spec(self) -> dict[str, object]:
        return {
            "target_action": "press_c",
            "target_variable": "alarm_on",
            "effect_value": True,
            "candidate_gates": (("door_open", True), ("fan_on", False)),
        }


class DelayedLampWorld(DoorLampWorld):
    """A lamp world where the key effect is delayed behind a hidden latch."""

    name = "delayed-lamp"

    def __init__(self) -> None:
        super().__init__()
        self._pending_lamp = False

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._pending_lamp = False
        self._state = {
            "lamp_on": False,
            "sound": False,
        }
        return self.observe()

    def set_state(self, state: State) -> None:
        expected = set(self._state)
        incoming = set(state)
        if incoming != expected:
            missing = sorted(expected - incoming)
            extra = sorted(incoming - expected)
            raise ValueError(f"state keys mismatch; missing={missing}, extra={extra}")
        self._state = dict(state)
        self._pending_lamp = False

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("reset_lamp", "Turn the lamp off and clear pending effects."),
            ActionSpec("press_delay", "Press the delayed lamp switch."),
            ActionSpec("wait", "Let delayed effects resolve."),
        ]

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "reset_lamp":
            self._state["lamp_on"] = False
            self._state["sound"] = False
            self._pending_lamp = False
        elif action == "press_delay":
            self._state["sound"] = True
            self._pending_lamp = True
        elif action == "wait":
            self._state["sound"] = False
            if self._pending_lamp:
                self._state["lamp_on"] = True
                self._pending_lamp = False

        return self.observe()

    def true_edges(self) -> set[Edge]:
        return {
            ("reset_lamp", "lamp_on"),
            ("reset_lamp", "sound"),
            ("press_delay", "sound"),
            ("press_delay", "lamp_on"),
            ("wait", "sound"),
        }

    def temporal_spec(self) -> dict[str, object]:
        return {
            "target_action": "press_delay",
            "target_variable": "lamp_on",
            "effect_value": True,
            "followup_action": "wait",
            "delay_steps": 1,
        }


class DelayedLampShiftedWorld(DelayedLampWorld):
    """Same delayed mechanism as DelayedLampWorld under a shifted start state."""

    name = "delayed-lamp-shifted"

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._pending_lamp = False
        self._state = {
            "lamp_on": True,
            "sound": seed is not None and seed % 2 == 0,
        }
        return self.observe()


class DelayedLampLongDelayWorld(DelayedLampWorld):
    """Same delayed edge as DelayedLampWorld, but the temporal delay changes."""

    name = "delayed-lamp-long-delay"

    def __init__(self) -> None:
        super().__init__()
        self._pending_waits = 0

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._pending_lamp = False
        self._pending_waits = 0
        self._state = {
            "lamp_on": False,
            "sound": False,
        }
        return self.observe()

    def set_state(self, state: State) -> None:
        super().set_state(state)
        self._pending_waits = 0

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("reset_lamp", "Turn the lamp off and clear pending effects."),
            ActionSpec("press_decoy_delay", "Press a delayed-looking decoy switch."),
            ActionSpec("press_delay", "Press the delayed lamp switch."),
            ActionSpec("wait", "Let delayed effects resolve."),
        ]

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "reset_lamp":
            self._state["lamp_on"] = False
            self._state["sound"] = False
            self._pending_lamp = False
            self._pending_waits = 0
        elif action == "press_delay":
            self._state["sound"] = True
            self._pending_lamp = True
            self._pending_waits = 2
        elif action == "press_decoy_delay":
            self._state["sound"] = True
            self._pending_lamp = False
            self._pending_waits = 0
        elif action == "wait":
            self._state["sound"] = False
            if self._pending_waits > 0:
                self._pending_waits -= 1
                if self._pending_waits == 0 and self._pending_lamp:
                    self._state["lamp_on"] = True
                    self._pending_lamp = False

        return self.observe()

    def true_edges(self) -> set[Edge]:
        return super().true_edges() | {("press_decoy_delay", "sound")}

    def temporal_spec(self) -> dict[str, object]:
        return {
            "target_action": "press_delay",
            "target_variable": "lamp_on",
            "effect_value": True,
            "followup_action": "wait",
            "delay_steps": 2,
        }


class DualDelayedControlWorld(DoorLampWorld):
    """Two delayed mechanisms plus a delayed-looking decoy action."""

    name = "dual-delayed-controls"
    lamp_delay_steps = 1
    alarm_delay_steps = 1

    def __init__(self) -> None:
        super().__init__()
        self._pending_lamp_waits = 0
        self._pending_alarm_waits = 0

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._pending_lamp_waits = 0
        self._pending_alarm_waits = 0
        self._state = {
            "lamp_on": False,
            "alarm_on": False,
            "sound": False,
        }
        return self.observe()

    def set_state(self, state: State) -> None:
        expected = set(self._state)
        incoming = set(state)
        if incoming != expected:
            missing = sorted(expected - incoming)
            extra = sorted(incoming - expected)
            raise ValueError(f"state keys mismatch; missing={missing}, extra={extra}")
        self._state = dict(state)
        self._pending_lamp_waits = 0
        self._pending_alarm_waits = 0

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("reset_alarm", "Turn the alarm off and clear pending effects."),
            ActionSpec("reset_lamp", "Turn the lamp off and clear pending effects."),
            ActionSpec("press_decoy_delay", "Press a delayed-looking decoy switch."),
            ActionSpec("press_delay_alarm", "Press the delayed alarm switch."),
            ActionSpec("press_delay_lamp", "Press the delayed lamp switch."),
            ActionSpec("wait", "Let delayed effects resolve."),
        ]

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "reset_alarm":
            self._state["alarm_on"] = False
            self._state["sound"] = False
            self._pending_alarm_waits = 0
        elif action == "reset_lamp":
            self._state["lamp_on"] = False
            self._state["sound"] = False
            self._pending_lamp_waits = 0
        elif action == "press_decoy_delay":
            self._state["sound"] = True
        elif action == "press_delay_alarm":
            self._state["sound"] = True
            self._pending_alarm_waits = self.alarm_delay_steps
        elif action == "press_delay_lamp":
            self._state["sound"] = True
            self._pending_lamp_waits = self.lamp_delay_steps
        elif action == "wait":
            self._state["sound"] = False
            if self._pending_alarm_waits > 0:
                self._pending_alarm_waits -= 1
                if self._pending_alarm_waits == 0:
                    self._state["alarm_on"] = True
            if self._pending_lamp_waits > 0:
                self._pending_lamp_waits -= 1
                if self._pending_lamp_waits == 0:
                    self._state["lamp_on"] = True

        return self.observe()

    def true_edges(self) -> set[Edge]:
        return {
            ("reset_alarm", "alarm_on"),
            ("reset_alarm", "sound"),
            ("reset_lamp", "lamp_on"),
            ("reset_lamp", "sound"),
            ("press_decoy_delay", "sound"),
            ("press_delay_alarm", "alarm_on"),
            ("press_delay_alarm", "sound"),
            ("press_delay_lamp", "lamp_on"),
            ("press_delay_lamp", "sound"),
            ("wait", "sound"),
        }

    def temporal_specs(self) -> tuple[dict[str, object], ...]:
        return (
            {
                "target_action": "press_delay_alarm",
                "target_variable": "alarm_on",
                "effect_value": True,
                "followup_action": "wait",
                "delay_steps": self.alarm_delay_steps,
            },
            {
                "target_action": "press_delay_lamp",
                "target_variable": "lamp_on",
                "effect_value": True,
                "followup_action": "wait",
                "delay_steps": self.lamp_delay_steps,
            },
        )

    def temporal_spec(self) -> dict[str, object]:
        return dict(self.temporal_specs()[0])


class DualDelayedSelectiveShiftWorld(DualDelayedControlWorld):
    """Only one delayed mechanism changes while another remains stable."""

    name = "dual-delayed-controls-selective-shift"
    lamp_delay_steps = 2
    alarm_delay_steps = 1


class RenamedDualDelayedControlWorld(DoorLampWorld):
    """The dual delayed mechanisms under a renamed action and observation schema."""

    name = "renamed-dual-delayed-controls"
    glow_delay_steps = 1
    siren_delay_steps = 1

    def __init__(self) -> None:
        super().__init__()
        self._pending_glow_waits = 0
        self._pending_siren_waits = 0

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._pending_glow_waits = 0
        self._pending_siren_waits = 0
        self._state = {
            "glow_active": False,
            "siren_active": False,
            "tone_active": False,
        }
        return self.observe()

    def set_state(self, state: State) -> None:
        expected = set(self._state)
        incoming = set(state)
        if incoming != expected:
            missing = sorted(expected - incoming)
            extra = sorted(incoming - expected)
            raise ValueError(f"state keys mismatch; missing={missing}, extra={extra}")
        self._state = dict(state)
        self._pending_glow_waits = 0
        self._pending_siren_waits = 0

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("silence_siren", "Turn the alarm siren output off."),
            ActionSpec("dim_glow", "Turn the lamp glow output off."),
            ActionSpec("poke_dummy", "Press a delayed-looking decoy control."),
            ActionSpec("tap_siren", "Press the delayed alarm siren control."),
            ActionSpec("tap_glow", "Press the delayed lamp glow control."),
            ActionSpec("settle", "Let delayed effects resolve."),
        ]

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "silence_siren":
            self._state["siren_active"] = False
            self._state["tone_active"] = False
            self._pending_siren_waits = 0
        elif action == "dim_glow":
            self._state["glow_active"] = False
            self._state["tone_active"] = False
            self._pending_glow_waits = 0
        elif action == "poke_dummy":
            self._state["tone_active"] = True
        elif action == "tap_siren":
            self._state["tone_active"] = True
            self._pending_siren_waits = self.siren_delay_steps
        elif action == "tap_glow":
            self._state["tone_active"] = True
            self._pending_glow_waits = self.glow_delay_steps
        elif action == "settle":
            self._state["tone_active"] = False
            if self._pending_siren_waits > 0:
                self._pending_siren_waits -= 1
                if self._pending_siren_waits == 0:
                    self._state["siren_active"] = True
            if self._pending_glow_waits > 0:
                self._pending_glow_waits -= 1
                if self._pending_glow_waits == 0:
                    self._state["glow_active"] = True

        return self.observe()

    def true_edges(self) -> set[Edge]:
        return {
            ("silence_siren", "siren_active"),
            ("silence_siren", "tone_active"),
            ("dim_glow", "glow_active"),
            ("dim_glow", "tone_active"),
            ("poke_dummy", "tone_active"),
            ("tap_siren", "siren_active"),
            ("tap_siren", "tone_active"),
            ("tap_glow", "glow_active"),
            ("tap_glow", "tone_active"),
            ("settle", "tone_active"),
        }

    def temporal_specs(self) -> tuple[dict[str, object], ...]:
        return (
            {
                "target_action": "tap_siren",
                "target_variable": "siren_active",
                "effect_value": True,
                "followup_action": "settle",
                "delay_steps": self.siren_delay_steps,
            },
            {
                "target_action": "tap_glow",
                "target_variable": "glow_active",
                "effect_value": True,
                "followup_action": "settle",
                "delay_steps": self.glow_delay_steps,
            },
        )

    def temporal_spec(self) -> dict[str, object]:
        return dict(self.temporal_specs()[0])


class RenamedDualDelayedSelectiveShiftWorld(RenamedDualDelayedControlWorld):
    """Renamed schema where only the lamp/glow delayed mechanism changes."""

    name = "renamed-dual-delayed-controls-selective-shift"
    glow_delay_steps = 2
    siren_delay_steps = 1


class TripleDelayedControlWorld(DoorLampWorld):
    """Three delayed mechanisms plus a delayed-looking decoy action."""

    name = "triple-delayed-controls"
    lamp_delay_steps = 1
    alarm_delay_steps = 1
    fan_delay_steps = 1

    def __init__(self) -> None:
        super().__init__()
        self._pending_lamp_waits = 0
        self._pending_alarm_waits = 0
        self._pending_fan_waits = 0

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._pending_lamp_waits = 0
        self._pending_alarm_waits = 0
        self._pending_fan_waits = 0
        self._state = {
            "lamp_on": False,
            "alarm_on": False,
            "fan_on": False,
            "sound": False,
        }
        return self.observe()

    def set_state(self, state: State) -> None:
        expected = set(self._state)
        incoming = set(state)
        if incoming != expected:
            missing = sorted(expected - incoming)
            extra = sorted(incoming - expected)
            raise ValueError(f"state keys mismatch; missing={missing}, extra={extra}")
        self._state = dict(state)
        self._pending_lamp_waits = 0
        self._pending_alarm_waits = 0
        self._pending_fan_waits = 0

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("reset_alarm", "Turn the alarm off and clear pending effects."),
            ActionSpec("reset_lamp", "Turn the lamp off and clear pending effects."),
            ActionSpec("reset_fan", "Turn the fan off and clear pending effects."),
            ActionSpec("press_decoy_delay", "Press a delayed-looking decoy switch."),
            ActionSpec("press_delay_alarm", "Press the delayed alarm switch."),
            ActionSpec("press_delay_fan", "Press the delayed fan switch."),
            ActionSpec("press_delay_lamp", "Press the delayed lamp switch."),
            ActionSpec("wait", "Let delayed effects resolve."),
        ]

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "reset_alarm":
            self._state["alarm_on"] = False
            self._state["sound"] = False
            self._pending_alarm_waits = 0
        elif action == "reset_lamp":
            self._state["lamp_on"] = False
            self._state["sound"] = False
            self._pending_lamp_waits = 0
        elif action == "reset_fan":
            self._state["fan_on"] = False
            self._state["sound"] = False
            self._pending_fan_waits = 0
        elif action == "press_decoy_delay":
            self._state["sound"] = True
        elif action == "press_delay_alarm":
            self._state["sound"] = True
            self._pending_alarm_waits = self.alarm_delay_steps
        elif action == "press_delay_fan":
            self._state["sound"] = True
            self._pending_fan_waits = self.fan_delay_steps
        elif action == "press_delay_lamp":
            self._state["sound"] = True
            self._pending_lamp_waits = self.lamp_delay_steps
        elif action == "wait":
            self._state["sound"] = False
            if self._pending_alarm_waits > 0:
                self._pending_alarm_waits -= 1
                if self._pending_alarm_waits == 0:
                    self._state["alarm_on"] = True
            if self._pending_fan_waits > 0:
                self._pending_fan_waits -= 1
                if self._pending_fan_waits == 0:
                    self._state["fan_on"] = True
            if self._pending_lamp_waits > 0:
                self._pending_lamp_waits -= 1
                if self._pending_lamp_waits == 0:
                    self._state["lamp_on"] = True

        return self.observe()

    def true_edges(self) -> set[Edge]:
        return {
            ("reset_alarm", "alarm_on"),
            ("reset_alarm", "sound"),
            ("reset_lamp", "lamp_on"),
            ("reset_lamp", "sound"),
            ("reset_fan", "fan_on"),
            ("reset_fan", "sound"),
            ("press_decoy_delay", "sound"),
            ("press_delay_alarm", "alarm_on"),
            ("press_delay_alarm", "sound"),
            ("press_delay_fan", "fan_on"),
            ("press_delay_fan", "sound"),
            ("press_delay_lamp", "lamp_on"),
            ("press_delay_lamp", "sound"),
            ("wait", "sound"),
        }

    def temporal_specs(self) -> tuple[dict[str, object], ...]:
        return (
            {
                "target_action": "press_delay_alarm",
                "target_variable": "alarm_on",
                "effect_value": True,
                "followup_action": "wait",
                "delay_steps": self.alarm_delay_steps,
            },
            {
                "target_action": "press_delay_fan",
                "target_variable": "fan_on",
                "effect_value": True,
                "followup_action": "wait",
                "delay_steps": self.fan_delay_steps,
            },
            {
                "target_action": "press_delay_lamp",
                "target_variable": "lamp_on",
                "effect_value": True,
                "followup_action": "wait",
                "delay_steps": self.lamp_delay_steps,
            },
        )

    def temporal_spec(self) -> dict[str, object]:
        return dict(self.temporal_specs()[0])


class RenamedTripleDelayedControlWorld(DoorLampWorld):
    """Three delayed mechanisms under a renamed action and observation schema."""

    name = "renamed-triple-delayed-controls"
    glow_delay_steps = 1
    siren_delay_steps = 1
    rotor_delay_steps = 1

    def __init__(self) -> None:
        super().__init__()
        self._pending_glow_waits = 0
        self._pending_siren_waits = 0
        self._pending_rotor_waits = 0

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._pending_glow_waits = 0
        self._pending_siren_waits = 0
        self._pending_rotor_waits = 0
        self._state = {
            "glow_active": False,
            "siren_active": False,
            "rotor_active": False,
            "tone_active": False,
        }
        return self.observe()

    def set_state(self, state: State) -> None:
        expected = set(self._state)
        incoming = set(state)
        if incoming != expected:
            missing = sorted(expected - incoming)
            extra = sorted(incoming - expected)
            raise ValueError(f"state keys mismatch; missing={missing}, extra={extra}")
        self._state = dict(state)
        self._pending_glow_waits = 0
        self._pending_siren_waits = 0
        self._pending_rotor_waits = 0

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("silence_siren", "Turn the alarm siren output off."),
            ActionSpec("dim_glow", "Turn the lamp glow output off."),
            ActionSpec("stop_rotor", "Turn the fan rotor output off."),
            ActionSpec("poke_dummy", "Press a delayed-looking decoy control."),
            ActionSpec("spin_rotor", "Press the delayed fan rotor control."),
            ActionSpec("tap_siren", "Press the delayed alarm siren control."),
            ActionSpec("tap_glow", "Press the delayed lamp glow control."),
            ActionSpec("settle", "Let delayed effects resolve."),
        ]

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "silence_siren":
            self._state["siren_active"] = False
            self._state["tone_active"] = False
            self._pending_siren_waits = 0
        elif action == "dim_glow":
            self._state["glow_active"] = False
            self._state["tone_active"] = False
            self._pending_glow_waits = 0
        elif action == "stop_rotor":
            self._state["rotor_active"] = False
            self._state["tone_active"] = False
            self._pending_rotor_waits = 0
        elif action == "poke_dummy":
            self._state["tone_active"] = True
        elif action == "spin_rotor":
            self._state["tone_active"] = True
            self._pending_rotor_waits = self.rotor_delay_steps
        elif action == "tap_siren":
            self._state["tone_active"] = True
            self._pending_siren_waits = self.siren_delay_steps
        elif action == "tap_glow":
            self._state["tone_active"] = True
            self._pending_glow_waits = self.glow_delay_steps
        elif action == "settle":
            self._state["tone_active"] = False
            if self._pending_rotor_waits > 0:
                self._pending_rotor_waits -= 1
                if self._pending_rotor_waits == 0:
                    self._state["rotor_active"] = True
            if self._pending_siren_waits > 0:
                self._pending_siren_waits -= 1
                if self._pending_siren_waits == 0:
                    self._state["siren_active"] = True
            if self._pending_glow_waits > 0:
                self._pending_glow_waits -= 1
                if self._pending_glow_waits == 0:
                    self._state["glow_active"] = True

        return self.observe()

    def true_edges(self) -> set[Edge]:
        return {
            ("silence_siren", "siren_active"),
            ("silence_siren", "tone_active"),
            ("dim_glow", "glow_active"),
            ("dim_glow", "tone_active"),
            ("stop_rotor", "rotor_active"),
            ("stop_rotor", "tone_active"),
            ("poke_dummy", "tone_active"),
            ("spin_rotor", "rotor_active"),
            ("spin_rotor", "tone_active"),
            ("tap_siren", "siren_active"),
            ("tap_siren", "tone_active"),
            ("tap_glow", "glow_active"),
            ("tap_glow", "tone_active"),
            ("settle", "tone_active"),
        }

    def temporal_specs(self) -> tuple[dict[str, object], ...]:
        return (
            {
                "target_action": "tap_siren",
                "target_variable": "siren_active",
                "effect_value": True,
                "followup_action": "settle",
                "delay_steps": self.siren_delay_steps,
            },
            {
                "target_action": "spin_rotor",
                "target_variable": "rotor_active",
                "effect_value": True,
                "followup_action": "settle",
                "delay_steps": self.rotor_delay_steps,
            },
            {
                "target_action": "tap_glow",
                "target_variable": "glow_active",
                "effect_value": True,
                "followup_action": "settle",
                "delay_steps": self.glow_delay_steps,
            },
        )

    def temporal_spec(self) -> dict[str, object]:
        return dict(self.temporal_specs()[0])


class RenamedTripleDelayedDiagnosticShiftWorld(RenamedTripleDelayedControlWorld):
    """Renamed target where only the glow/lamp delay is shifted."""

    name = "renamed-triple-delayed-controls-diagnostic-shift"
    glow_delay_steps = 2
    siren_delay_steps = 1
    rotor_delay_steps = 1


@dataclass(frozen=True)
class ProceduralDelayedMechanismSpec:
    role: str
    action: str
    variable: str
    reset_action: str
    delay_steps: int
    action_description: str
    reset_description: str
    effect_value: bool = True


@dataclass(frozen=True)
class ProceduralReadoutSpec:
    name: str
    source_variable: str
    invert: bool = False


class ProceduralDelayedMechanismWorld(DoorLampWorld):
    """Generated delayed-effect SCM with named mechanisms and a decoy."""

    def __init__(
        self,
        name: str,
        mechanisms: tuple[ProceduralDelayedMechanismSpec, ...],
        followup_action: str,
        followup_description: str,
        decoy_action: str,
        decoy_description: str,
        transient_variable: str,
        action_order: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.name = name
        self._mechanisms = mechanisms
        self._followup_action = followup_action
        self._followup_description = followup_description
        self._decoy_action = decoy_action
        self._decoy_description = decoy_description
        self._transient_variable = transient_variable
        self._action_order = action_order
        self._pending_waits: dict[str, int] = {}

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._pending_waits = {
            mechanism.variable: 0 for mechanism in self._mechanisms
        }
        self._state = {
            mechanism.variable: not mechanism.effect_value
            for mechanism in self._mechanisms
        }
        self._state[self._transient_variable] = False
        return self.observe()

    def set_state(self, state: State) -> None:
        expected = set(self._state)
        incoming = set(state)
        if incoming != expected:
            missing = sorted(expected - incoming)
            extra = sorted(incoming - expected)
            raise ValueError(f"state keys mismatch; missing={missing}, extra={extra}")
        self._state = dict(state)
        self._pending_waits = {
            mechanism.variable: 0 for mechanism in self._mechanisms
        }

    def actions(self) -> list[ActionSpec]:
        specs = {
            mechanism.reset_action: ActionSpec(
                mechanism.reset_action,
                mechanism.reset_description,
            )
            for mechanism in self._mechanisms
        }
        specs.update(
            {
                mechanism.action: ActionSpec(
                    mechanism.action,
                    mechanism.action_description,
                )
                for mechanism in self._mechanisms
            }
        )
        specs[self._decoy_action] = ActionSpec(
            self._decoy_action,
            self._decoy_description,
        )
        specs[self._followup_action] = ActionSpec(
            self._followup_action,
            self._followup_description,
        )
        return [specs[action] for action in self._action_order]

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        mechanism_by_action = {
            mechanism.action: mechanism for mechanism in self._mechanisms
        }
        mechanism_by_reset = {
            mechanism.reset_action: mechanism for mechanism in self._mechanisms
        }

        if action in mechanism_by_reset:
            mechanism = mechanism_by_reset[action]
            self._state[mechanism.variable] = not mechanism.effect_value
            self._state[self._transient_variable] = False
            self._pending_waits[mechanism.variable] = 0
        elif action == self._decoy_action:
            self._state[self._transient_variable] = True
        elif action in mechanism_by_action:
            mechanism = mechanism_by_action[action]
            self._state[self._transient_variable] = True
            self._pending_waits[mechanism.variable] = mechanism.delay_steps
        elif action == self._followup_action:
            self._state[self._transient_variable] = False
            for mechanism in self._mechanisms:
                pending = self._pending_waits[mechanism.variable]
                if pending <= 0:
                    continue
                pending -= 1
                self._pending_waits[mechanism.variable] = pending
                if pending == 0:
                    self._state[mechanism.variable] = mechanism.effect_value

        return self.observe()

    def true_edges(self) -> set[Edge]:
        edges: set[Edge] = {
            (self._decoy_action, self._transient_variable),
            (self._followup_action, self._transient_variable),
        }
        for mechanism in self._mechanisms:
            edges.add((mechanism.reset_action, mechanism.variable))
            edges.add((mechanism.reset_action, self._transient_variable))
            edges.add((mechanism.action, mechanism.variable))
            edges.add((mechanism.action, self._transient_variable))
        return edges

    def temporal_specs(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "target_action": mechanism.action,
                "target_variable": mechanism.variable,
                "effect_value": mechanism.effect_value,
                "followup_action": self._followup_action,
                "delay_steps": mechanism.delay_steps,
            }
            for mechanism in self._mechanisms
        )

    def temporal_spec(self) -> dict[str, object]:
        return dict(self.temporal_specs()[0])


class ProceduralReadoutWorld(DoorLampWorld):
    """Observation-decorated procedural SCM with non-causal readout variables."""

    def __init__(
        self,
        base_world: ProceduralDelayedMechanismWorld,
        readouts: tuple[ProceduralReadoutSpec, ...],
        name_suffix: str,
        noise_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self._base_world = base_world
        self._readouts = readouts
        self._noise_probability = noise_probability
        self._rng = Random()
        self.name = f"{base_world.name}-{name_suffix}"

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._base_world.reset(seed=seed)
        return self.observe()

    def observe(self) -> State:
        observed = self._base_world.observe()
        for readout in self._readouts:
            value = observed[readout.source_variable]
            value = (not value) if readout.invert else value
            if self._rng.random() < self._noise_probability:
                value = not value
            observed[readout.name] = value
        return observed

    def set_state(self, state: State) -> None:
        core_state = {
            variable: value
            for variable, value in state.items()
            if variable not in self.readout_variables()
        }
        self._base_world.set_state(core_state)

    def actions(self) -> list[ActionSpec]:
        return self._base_world.actions()

    def step(self, action: str) -> State:
        self._base_world.step(action)
        return self.observe()

    def true_edges(self) -> set[Edge]:
        return self._base_world.true_edges()

    def temporal_specs(self) -> tuple[dict[str, object], ...]:
        return self._base_world.temporal_specs()

    def temporal_spec(self) -> dict[str, object]:
        return self._base_world.temporal_spec()

    def readout_variables(self) -> set[str]:
        return {readout.name for readout in self._readouts}


def make_procedural_diagnostic_world_pair(
    family_seed: int,
    mechanism_count: int = 3,
    readout_mode: str = "none",
) -> tuple[DoorLampWorld, DoorLampWorld]:
    """Generate a source/renamed-target pair with one shifted delay."""

    if mechanism_count < 2 or mechanism_count > 3:
        raise ValueError("mechanism_count must be between 2 and 3")
    if readout_mode not in {
        "none",
        "opaque",
        "semantic-confounder",
        "noisy-opaque",
        "noisy-semantic-confounder",
    }:
        raise ValueError(
            "readout_mode must be 'none', 'opaque', 'semantic-confounder', "
            "'noisy-opaque', or 'noisy-semantic-confounder'"
        )

    rng = Random(family_seed)
    roles = ["alarm", "fan", "lamp"]
    rng.shuffle(roles)
    roles = sorted(roles[:mechanism_count])
    shifted_role = rng.choice(roles)

    source_followup = "wait"
    source_transient = "sound"
    source_decoy = _unique_name("press_decoy_delay", family_seed)
    source_mechanisms = tuple(
        ProceduralDelayedMechanismSpec(
            role=role,
            action=_unique_name(f"press_delay_{role}", family_seed),
            variable=_unique_name(f"{role}_on", family_seed),
            reset_action=_unique_name(f"reset_{role}", family_seed),
            delay_steps=1,
            action_description=f"Press the delayed {role} control.",
            reset_description=(
                f"Reset the {role} output off and clear pending effects."
            ),
        )
        for role in roles
    )
    source_actions = tuple(
        sorted(
            [mechanism.reset_action for mechanism in source_mechanisms]
            + [source_decoy]
            + [mechanism.action for mechanism in source_mechanisms]
            + [source_followup]
        )
    )
    source_world = ProceduralDelayedMechanismWorld(
        name=f"procedural-delayed-source-{family_seed}",
        mechanisms=source_mechanisms,
        followup_action=source_followup,
        followup_description="Let delayed effects resolve.",
        decoy_action=source_decoy,
        decoy_description="Press a delayed-looking decoy control.",
        transient_variable=source_transient,
        action_order=source_actions,
    )

    followup_aliases = [
        ("settle", "Let delayed effects settle and resolve."),
        ("pause", "Pause so delayed effects can resolve."),
        ("resolve_step", "Resolve pending delayed effects."),
    ]
    target_followup, target_followup_description = rng.choice(followup_aliases)
    target_decoy = _unique_name(rng.choice(["poke_dummy", "tap_dummy"]), family_seed)
    target_transient = _unique_name(rng.choice(["tone_active", "chime_active"]), family_seed)
    target_mechanisms = tuple(
        _procedural_target_mechanism(
            rng=rng,
            family_seed=family_seed,
            role=role,
            delay_steps=2 if role == shifted_role else 1,
        )
        for role in roles
    )
    target_actions = [mechanism.reset_action for mechanism in target_mechanisms]
    target_actions += [target_decoy]
    target_actions += [mechanism.action for mechanism in target_mechanisms]
    target_actions += [target_followup]
    rng.shuffle(target_actions)
    target_world = ProceduralDelayedMechanismWorld(
        name=f"procedural-renamed-diagnostic-shift-{family_seed}",
        mechanisms=target_mechanisms,
        followup_action=target_followup,
        followup_description=target_followup_description,
        decoy_action=target_decoy,
        decoy_description="Press a delayed-looking decoy control.",
        transient_variable=target_transient,
        action_order=tuple(target_actions),
    )
    if readout_mode != "none":
        source_world = _decorate_procedural_readouts(
            source_world,
            family_seed=family_seed,
            namespace="source",
            mode=readout_mode,
        )
        target_world = _decorate_procedural_readouts(
            target_world,
            family_seed=family_seed,
            namespace="target",
            mode=readout_mode,
        )
    return source_world, target_world


def _decorate_procedural_readouts(
    world: ProceduralDelayedMechanismWorld,
    family_seed: int,
    namespace: str,
    mode: str,
) -> ProceduralReadoutWorld:
    semantic_mode = "semantic" in mode
    noisy_mode = mode.startswith("noisy-")
    readouts: list[ProceduralReadoutSpec] = []
    for index, mechanism in enumerate(world.temporal_specs()):
        variable = str(mechanism["target_variable"])
        role = str(
            next(
                spec.role
                for spec in world._mechanisms
                if spec.variable == variable
            )
        )
        if semantic_mode:
            readout_name = f"aaa_{role}_{namespace}_readout_{family_seed}"
        else:
            readout_name = f"aux_{namespace}_bit_{index}_{family_seed}"
        readouts.append(
            ProceduralReadoutSpec(
                name=readout_name,
                source_variable=variable,
            )
        )

    readouts.append(
        ProceduralReadoutSpec(
            name=f"aux_{namespace}_transient_{family_seed}",
            source_variable=world._transient_variable,
        )
    )
    suffix = "semantic-readouts" if semantic_mode else "opaque-readouts"
    if noisy_mode:
        suffix = f"noisy-{suffix}"
    return ProceduralReadoutWorld(
        base_world=world,
        readouts=tuple(readouts),
        name_suffix=suffix,
        noise_probability=0.10 if noisy_mode else 0.0,
    )


def _procedural_target_mechanism(
    rng: Random,
    family_seed: int,
    role: str,
    delay_steps: int,
) -> ProceduralDelayedMechanismSpec:
    aliases = {
        "alarm": [
            ("siren", "tap_siren", "silence_siren"),
            ("alert", "pulse_alert", "clear_alert"),
            ("alarm", "trigger_alarm", "mute_alarm"),
        ],
        "fan": [
            ("rotor", "spin_rotor", "stop_rotor"),
            ("vent", "open_vent", "close_vent"),
            ("blower", "start_blower", "stop_blower"),
        ],
        "lamp": [
            ("glow", "tap_glow", "dim_glow"),
            ("beacon", "pulse_beacon", "darken_beacon"),
            ("light", "flash_light", "douse_light"),
        ],
    }
    alias, action, reset_action = rng.choice(aliases[role])
    variable = f"{alias}_active"
    return ProceduralDelayedMechanismSpec(
        role=role,
        action=_unique_name(action, family_seed),
        variable=_unique_name(variable, family_seed),
        reset_action=_unique_name(reset_action, family_seed),
        delay_steps=delay_steps,
        action_description=(
            f"Press the delayed {role} {alias} control."
        ),
        reset_description=(
            f"Reset the {role} {alias} output off and clear pending effects."
        ),
    )


def _unique_name(name: str, family_seed: int) -> str:
    return f"{name}_{family_seed}"


class PanelShiftedWorld(PanelWorld):
    name = "panel-shifted"

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._state = {
            "dark": False,
            "lamp_on": True,
            "door_open": True,
            "sound": seed is not None and seed % 2 == 0,
            "fan_on": True,
            "alarm_on": False,
        }
        return self.observe()


class PanelInvertedWorld(PanelShiftedWorld):
    """Same interface as PanelWorld, but two context gates are changed."""

    name = "panel-inverted"

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "set_dark":
            self._state["dark"] = True
        elif action == "set_bright":
            self._state["dark"] = False
        elif action == "open_door":
            self._state["door_open"] = True
        elif action == "close_door":
            self._state["door_open"] = False
        elif action == "reset_lamp":
            self._state["lamp_on"] = False
        elif action == "start_fan":
            self._state["fan_on"] = True
        elif action == "stop_fan":
            self._state["fan_on"] = False
        elif action == "reset_alarm":
            self._state["alarm_on"] = False
        elif action == "press_a":
            self._state["sound"] = True
            if not self._state["dark"]:
                self._state["lamp_on"] = True
        elif action == "press_b":
            self._state["sound"] = True
            if not self._state["door_open"]:
                self._state["door_open"] = True
        elif action == "press_c":
            self._state["sound"] = True
            if self._press_c_alarm_condition():
                self._state["alarm_on"] = True
        elif action == "wait":
            self._state["sound"] = False

        return self.observe()

    def _press_c_alarm_condition(self) -> bool:
        return self._state["door_open"] and self._state["fan_on"]


class DerivedSensorPanelWorld(PanelWorld):
    """PanelWorld with deterministic sensor readouts added to observations."""

    name = "panel-derived-sensors"

    def observe(self) -> State:
        observed = dict(self._state)
        observed.update(_panel_sensor_readouts(self._state))
        return observed

    def set_state(self, state: State) -> None:
        core_keys = set(self._state)
        incoming_core = {
            key: value for key, value in state.items() if key in core_keys
        }
        if set(incoming_core) != core_keys:
            missing = sorted(core_keys - set(incoming_core))
            raise ValueError(f"state keys mismatch; missing={missing}")
        self._state = dict(incoming_core)

    def readout_variables(self) -> set[str]:
        return set(_panel_sensor_readouts(self._state))


class OpaqueReadoutPanelWorld(PanelWorld):
    """PanelWorld with deterministic readouts that do not have sensor names."""

    name = "panel-opaque-readouts"

    def observe(self) -> State:
        observed = dict(self._state)
        observed.update(_panel_opaque_readouts(self._state))
        return observed

    def set_state(self, state: State) -> None:
        core_keys = set(self._state)
        incoming_core = {
            key: value for key, value in state.items() if key in core_keys
        }
        if set(incoming_core) != core_keys:
            missing = sorted(core_keys - set(incoming_core))
            raise ValueError(f"state keys mismatch; missing={missing}")
        self._state = dict(incoming_core)

    def readout_variables(self) -> set[str]:
        return set(_panel_opaque_readouts(self._state))


class NoisyCorePanelWorld(PanelWorld):
    """PanelWorld where true causal variables are observed through noise."""

    name = "panel-noisy-core"

    def __init__(self, noise_probability: float = 0.05) -> None:
        super().__init__()
        self.noise_probability = noise_probability

    def observe(self) -> State:
        return {
            key: self._maybe_flip(value)
            for key, value in self._state.items()
        }

    def _maybe_flip(self, value: bool) -> bool:
        if self._rng.random() < self.noise_probability:
            return not value
        return value


class HiddenContextPanelWorld(PanelWorld):
    """PanelWorld with an unobserved context gate for some mechanisms."""

    name = "panel-hidden-context"

    def __init__(self, context_low_probability: float = 0.50) -> None:
        super().__init__()
        self.context_low_probability = context_low_probability
        self._power_low = False

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._state = {
            "dark": True,
            "lamp_on": False,
            "door_open": False,
            "sound": False,
            "fan_on": False,
            "alarm_on": False,
        }
        self._power_low = self._rng.random() < self.context_low_probability
        return self.observe()

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "set_dark":
            self._state["dark"] = True
        elif action == "set_bright":
            self._state["dark"] = False
        elif action == "open_door":
            self._state["door_open"] = True
        elif action == "close_door":
            self._state["door_open"] = False
        elif action == "reset_lamp":
            self._state["lamp_on"] = False
        elif action == "start_fan":
            self._state["fan_on"] = True
        elif action == "stop_fan":
            self._state["fan_on"] = False
        elif action == "reset_alarm":
            self._state["alarm_on"] = False
        elif action == "press_a":
            self._refresh_hidden_context()
            self._state["sound"] = True
            if self._state["dark"] and not self._power_low:
                self._state["lamp_on"] = True
        elif action == "press_b":
            self._state["sound"] = True
            if not self._state["door_open"]:
                self._state["door_open"] = True
        elif action == "press_c":
            self._refresh_hidden_context()
            self._state["sound"] = True
            if self._press_c_alarm_condition() and not self._power_low:
                self._state["alarm_on"] = True
        elif action == "wait":
            self._state["sound"] = False
            if self._rng.random() < 0.20:
                self._power_low = not self._power_low

        return self.observe()

    def hidden_context_edges(self) -> set[Edge]:
        return {("press_a", "lamp_on"), ("press_c", "alarm_on")}

    def _refresh_hidden_context(self) -> None:
        self._power_low = self._rng.random() < self.context_low_probability


class NoisyHiddenPanelWorld(PanelWorld):
    """PanelWorld with stochastic sensor mixing and a hidden context variable."""

    name = "panel-noisy-hidden"

    def __init__(self, noise_probability: float = 0.05) -> None:
        super().__init__()
        self.noise_probability = noise_probability
        self._power_low = False

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._state = {
            "dark": True,
            "lamp_on": False,
            "door_open": False,
            "sound": False,
            "fan_on": False,
            "alarm_on": False,
        }
        self._power_low = self._rng.random() < 0.35
        return self.observe()

    def observe(self) -> State:
        observed = {
            key: self._maybe_flip(value)
            for key, value in self._state.items()
        }
        sensor_values = _panel_sensor_readouts(self._state)
        observed.update(
            {
                key: self._maybe_flip(value)
                for key, value in sensor_values.items()
            }
        )
        return observed

    def set_state(self, state: State) -> None:
        core_keys = set(self._state)
        incoming_core = {key: value for key, value in state.items() if key in core_keys}
        if set(incoming_core) != core_keys:
            missing = sorted(core_keys - set(incoming_core))
            raise ValueError(f"state keys mismatch; missing={missing}")
        self._state = dict(incoming_core)

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "set_dark":
            self._state["dark"] = True
        elif action == "set_bright":
            self._state["dark"] = False
        elif action == "open_door":
            self._state["door_open"] = True
        elif action == "close_door":
            self._state["door_open"] = False
        elif action == "reset_lamp":
            self._state["lamp_on"] = False
        elif action == "start_fan":
            self._state["fan_on"] = True
        elif action == "stop_fan":
            self._state["fan_on"] = False
        elif action == "reset_alarm":
            self._state["alarm_on"] = False
        elif action == "press_a":
            self._state["sound"] = True
            if self._state["dark"] and not self._power_low:
                self._state["lamp_on"] = True
        elif action == "press_b":
            self._state["sound"] = True
            if not self._state["door_open"]:
                self._state["door_open"] = True
        elif action == "press_c":
            self._state["sound"] = True
            if self._press_c_alarm_condition() and not self._power_low:
                self._state["alarm_on"] = True
        elif action == "wait":
            self._state["sound"] = False
            if self._rng.random() < 0.20:
                self._power_low = not self._power_low

        return self.observe()

    def _maybe_flip(self, value: bool) -> bool:
        if self._rng.random() < self.noise_probability:
            return not value
        return value

    def readout_variables(self) -> set[str]:
        return set(_panel_sensor_readouts(self._state))

    def hidden_context_edges(self) -> set[Edge]:
        return {("press_a", "lamp_on"), ("press_c", "alarm_on")}


class ComplexNoisyHiddenPanelWorld(PanelWorld):
    """A larger noisy-hidden panel with multiple latent context gates."""

    name = "panel-complex-noisy-hidden"

    def __init__(self, noise_probability: float = 0.05) -> None:
        super().__init__()
        self.noise_probability = noise_probability
        self._power_low = False
        self._network_down = False

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._state = {
            "dark": True,
            "lamp_on": False,
            "door_open": False,
            "sound": False,
            "fan_on": False,
            "alarm_on": False,
            "coolant_on": False,
            "heater_on": False,
            "pressure_high": False,
            "valve_open": False,
            "backup_on": False,
            "locked": False,
        }
        self._power_low = self._rng.random() < 0.35
        self._network_down = self._rng.random() < 0.30
        return self.observe()

    def observe(self) -> State:
        observed = {
            key: self._maybe_flip(value)
            for key, value in self._state.items()
        }
        observed.update(
            {
                key: self._maybe_flip(value)
                for key, value in _complex_panel_sensor_readouts(self._state).items()
            }
        )
        return observed

    def set_state(self, state: State) -> None:
        core_keys = set(self._state)
        incoming_core = {key: value for key, value in state.items() if key in core_keys}
        if set(incoming_core) != core_keys:
            missing = sorted(core_keys - set(incoming_core))
            raise ValueError(f"state keys mismatch; missing={missing}")
        self._state = dict(incoming_core)

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("set_dark", "Make the room dark."),
            ActionSpec("set_bright", "Make the room bright."),
            ActionSpec("open_door", "Open the door directly."),
            ActionSpec("close_door", "Close the door directly."),
            ActionSpec("reset_lamp", "Turn the lamp off directly."),
            ActionSpec("start_fan", "Turn the fan on directly."),
            ActionSpec("stop_fan", "Turn the fan off directly."),
            ActionSpec("reset_alarm", "Turn the alarm off directly."),
            ActionSpec("start_coolant", "Turn the coolant on directly."),
            ActionSpec("stop_coolant", "Turn the coolant off directly."),
            ActionSpec("start_heater", "Turn the heater on directly."),
            ActionSpec("stop_heater", "Turn the heater off directly."),
            ActionSpec("open_valve", "Open the pressure valve directly."),
            ActionSpec("close_valve", "Close the pressure valve directly."),
            ActionSpec("reset_pressure", "Set pressure_high false directly."),
            ActionSpec("enable_backup", "Turn backup_on true directly."),
            ActionSpec("disable_backup", "Turn backup_on false directly."),
            ActionSpec("lock_panel", "Set locked true directly."),
            ActionSpec("unlock_panel", "Set locked false directly."),
            ActionSpec("press_a", "Press lamp button A."),
            ActionSpec("press_b", "Press door button B."),
            ActionSpec("press_c", "Press alarm button C."),
            ActionSpec("press_d", "Press thermal pressure button D."),
            ActionSpec("press_e", "Press backup escalation button E."),
            ActionSpec("wait", "Let transient effects fade."),
        ]

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        if action == "set_dark":
            self._state["dark"] = True
        elif action == "set_bright":
            self._state["dark"] = False
        elif action == "open_door":
            self._state["door_open"] = True
        elif action == "close_door":
            self._state["door_open"] = False
        elif action == "reset_lamp":
            self._state["lamp_on"] = False
        elif action == "start_fan":
            self._state["fan_on"] = True
        elif action == "stop_fan":
            self._state["fan_on"] = False
        elif action == "reset_alarm":
            self._state["alarm_on"] = False
        elif action == "start_coolant":
            self._state["coolant_on"] = True
        elif action == "stop_coolant":
            self._state["coolant_on"] = False
        elif action == "start_heater":
            self._state["heater_on"] = True
        elif action == "stop_heater":
            self._state["heater_on"] = False
        elif action == "open_valve":
            self._state["valve_open"] = True
        elif action == "close_valve":
            self._state["valve_open"] = False
        elif action == "reset_pressure":
            self._state["pressure_high"] = False
        elif action == "enable_backup":
            self._state["backup_on"] = True
        elif action == "disable_backup":
            self._state["backup_on"] = False
        elif action == "lock_panel":
            self._state["locked"] = True
        elif action == "unlock_panel":
            self._state["locked"] = False
        elif action == "press_a":
            self._refresh_power_context()
            self._state["sound"] = True
            if self._state["dark"] and not self._power_low:
                self._state["lamp_on"] = True
        elif action == "press_b":
            self._state["sound"] = True
            if not self._state["locked"]:
                self._state["door_open"] = True
        elif action == "press_c":
            self._refresh_power_context()
            self._state["sound"] = True
            if self._state["door_open"] and not self._state["fan_on"] and not self._power_low:
                self._state["alarm_on"] = True
        elif action == "press_d":
            self._refresh_network_context()
            self._state["sound"] = True
            if (
                self._state["heater_on"]
                and not self._state["coolant_on"]
                and not self._network_down
            ):
                self._state["pressure_high"] = True
        elif action == "press_e":
            self._refresh_network_context()
            self._state["sound"] = True
            if self._state["alarm_on"] and not self._network_down:
                self._state["backup_on"] = True
        elif action == "wait":
            self._state["sound"] = False
            if self._rng.random() < 0.20:
                self._power_low = not self._power_low
            if self._rng.random() < 0.20:
                self._network_down = not self._network_down
            if self._state["valve_open"]:
                self._state["pressure_high"] = False

        return self.observe()

    def true_edges(self) -> set[Edge]:
        return {
            ("set_dark", "dark"),
            ("set_bright", "dark"),
            ("open_door", "door_open"),
            ("close_door", "door_open"),
            ("reset_lamp", "lamp_on"),
            ("start_fan", "fan_on"),
            ("stop_fan", "fan_on"),
            ("reset_alarm", "alarm_on"),
            ("start_coolant", "coolant_on"),
            ("stop_coolant", "coolant_on"),
            ("start_heater", "heater_on"),
            ("stop_heater", "heater_on"),
            ("open_valve", "valve_open"),
            ("close_valve", "valve_open"),
            ("reset_pressure", "pressure_high"),
            ("enable_backup", "backup_on"),
            ("disable_backup", "backup_on"),
            ("lock_panel", "locked"),
            ("unlock_panel", "locked"),
            ("press_a", "sound"),
            ("press_a", "lamp_on"),
            ("press_b", "sound"),
            ("press_b", "door_open"),
            ("press_c", "sound"),
            ("press_c", "alarm_on"),
            ("press_d", "sound"),
            ("press_d", "pressure_high"),
            ("press_e", "sound"),
            ("press_e", "backup_on"),
            ("wait", "sound"),
            ("wait", "pressure_high"),
        }

    def readout_variables(self) -> set[str]:
        return set(_complex_panel_sensor_readouts(self._state))

    def hidden_context_edges(self) -> set[Edge]:
        return {
            ("press_a", "lamp_on"),
            ("press_c", "alarm_on"),
            ("press_d", "pressure_high"),
            ("press_e", "backup_on"),
        }

    def _maybe_flip(self, value: bool) -> bool:
        if self._rng.random() < self.noise_probability:
            return not value
        return value

    def _refresh_power_context(self) -> None:
        self._power_low = self._rng.random() < 0.35

    def _refresh_network_context(self) -> None:
        self._network_down = self._rng.random() < 0.30


def _panel_sensor_readouts(state: State) -> State:
    return {
        "light_sensor": state["lamp_on"] or (not state["dark"]),
        "access_sensor": state["door_open"],
        "alarm_sensor": state["alarm_on"] or (state["sound"] and state["dark"]),
    }


def _panel_opaque_readouts(state: State) -> State:
    return {
        "glow": state["lamp_on"] or (not state["dark"]),
        "alert": state["alarm_on"] or (state["sound"] and state["dark"]),
        "motion": state["sound"] or state["fan_on"],
    }


def _complex_panel_sensor_readouts(state: State) -> State:
    return {
        "light_sensor": state["lamp_on"] or (not state["dark"]),
        "access_sensor": state["door_open"] and not state["locked"],
        "alarm_sensor": state["alarm_on"] or (state["sound"] and state["dark"]),
        "pressure_sensor": state["pressure_high"] or state["valve_open"],
        "thermal_sensor": state["heater_on"] and not state["coolant_on"],
        "safety_sensor": state["backup_on"] or state["locked"],
    }


@dataclass(frozen=True)
class ProceduralHiddenMechanismSpec:
    action: str
    target: str
    visible_conditions: tuple[tuple[str, bool], ...]
    hidden_enabled_probability: float


class ProceduralComplexNoisyHiddenWorld(DoorLampWorld):
    """Procedural noisy-hidden SCM family with controllable visible contexts."""

    base_name = "procedural-complex-noisy-hidden"

    def __init__(
        self,
        family_seed: int,
        mechanism_count: int = 4,
        visible_count: int = 5,
        noise_probability: float = 0.05,
        readout_count: int = 6,
    ) -> None:
        super().__init__()
        self.family_seed = family_seed
        self.name = f"{self.base_name}-{family_seed}"
        self.mechanism_count = mechanism_count
        self.visible_count = visible_count
        self.noise_probability = noise_probability
        self.readout_count = readout_count
        self._family_rng = Random(family_seed)
        self._rng = Random()
        self._visible_variables = tuple(
            f"ctx_{family_seed}_{index}" for index in range(visible_count)
        )
        self._targets = tuple(
            f"out_{family_seed}_{index}" for index in range(mechanism_count)
        )
        self._hidden_enabled: dict[str, bool] = {}
        self._mechanisms = self._make_mechanisms()
        self._readout_sources = self._make_readouts()

    def _make_mechanisms(self) -> tuple[ProceduralHiddenMechanismSpec, ...]:
        mechanisms: list[ProceduralHiddenMechanismSpec] = []
        for index, target in enumerate(self._targets):
            condition_count = 1 + self._family_rng.randrange(2)
            variables = self._family_rng.sample(
                list(self._visible_variables),
                k=condition_count,
            )
            conditions = tuple(
                sorted(
                    (variable, bool(self._family_rng.randrange(2)))
                    for variable in variables
                )
            )
            mechanisms.append(
                ProceduralHiddenMechanismSpec(
                    action=f"probe_{self.family_seed}_{index}",
                    target=target,
                    visible_conditions=conditions,
                    hidden_enabled_probability=self._family_rng.choice(
                        [0.55, 0.65, 0.75]
                    ),
                )
            )
        return tuple(mechanisms)

    def _make_readouts(self) -> dict[str, tuple[str, ...]]:
        sources: dict[str, tuple[str, ...]] = {}
        candidates = list(self._targets + self._visible_variables)
        for index in range(self.readout_count):
            width = 1 + self._family_rng.randrange(2)
            sources[f"sensor_{self.family_seed}_{index}"] = tuple(
                sorted(self._family_rng.sample(candidates, k=width))
            )
        return sources

    def reset(self, seed: int | None = None) -> State:
        self._rng = Random(seed)
        self._state = {
            variable: False
            for variable in (
                self._visible_variables
                + self._targets
                + ("sound", "decoy_flag")
            )
        }
        self._hidden_enabled = {
            mechanism.action: (
                self._rng.random() < mechanism.hidden_enabled_probability
            )
            for mechanism in self._mechanisms
        }
        return self.observe()

    def observe(self) -> State:
        observed = {
            key: self._maybe_flip(value)
            for key, value in self._state.items()
        }
        for readout, sources in self._readout_sources.items():
            observed[readout] = self._maybe_flip(
                any(self._state[source] for source in sources)
            )
        return observed

    def set_state(self, state: State) -> None:
        core_keys = set(self._state)
        incoming_core = {
            key: value for key, value in state.items() if key in core_keys
        }
        if set(incoming_core) != core_keys:
            missing = sorted(core_keys - set(incoming_core))
            raise ValueError(f"state keys mismatch; missing={missing}")
        self._state = dict(incoming_core)

    def actions(self) -> list[ActionSpec]:
        specs: list[ActionSpec] = []
        for variable in self._visible_variables:
            specs.append(
                ActionSpec(
                    f"set_{variable}_on",
                    f"Set {variable} true directly.",
                )
            )
            specs.append(
                ActionSpec(
                    f"set_{variable}_off",
                    f"Set {variable} false directly.",
                )
            )
        for target in self._targets:
            specs.append(
                ActionSpec(
                    f"reset_{target}",
                    f"Reset {target} false directly.",
                )
            )
        for mechanism in self._mechanisms:
            condition_text = ", ".join(
                f"{variable}={value}"
                for variable, value in mechanism.visible_conditions
            )
            specs.append(
                ActionSpec(
                    mechanism.action,
                    (
                        f"Probe hidden-gated process for {mechanism.target}; "
                        f"visible setup: {condition_text}."
                    ),
                )
            )
        specs.extend(
            [
                ActionSpec(
                    f"poke_decoy_{self.family_seed}",
                    "Trigger a noisy process decoy.",
                ),
                ActionSpec(
                    f"toggle_decoy_{self.family_seed}",
                    "Toggle a decoy flag directly.",
                ),
                ActionSpec("wait", "Let transient effects fade."),
            ]
        )
        return specs

    def step(self, action: str) -> State:
        known_actions = {spec.name for spec in self.actions()}
        if action not in known_actions:
            raise ValueError(f"unknown action: {action}")

        for variable in self._visible_variables:
            if action == f"set_{variable}_on":
                self._state[variable] = True
                return self.observe()
            if action == f"set_{variable}_off":
                self._state[variable] = False
                return self.observe()
        for target in self._targets:
            if action == f"reset_{target}":
                self._state[target] = False
                return self.observe()

        mechanism_by_action = {
            mechanism.action: mechanism for mechanism in self._mechanisms
        }
        if action in mechanism_by_action:
            mechanism = mechanism_by_action[action]
            self._refresh_hidden_context(mechanism)
            self._state["sound"] = True
            visible_enabled = all(
                self._state[variable] == value
                for variable, value in mechanism.visible_conditions
            )
            if visible_enabled and self._hidden_enabled[mechanism.action]:
                self._state[mechanism.target] = True
        elif action == f"poke_decoy_{self.family_seed}":
            self._state["sound"] = True
            if self._rng.random() < 0.10:
                self._state["decoy_flag"] = not self._state["decoy_flag"]
        elif action == f"toggle_decoy_{self.family_seed}":
            self._state["decoy_flag"] = not self._state["decoy_flag"]
        elif action == "wait":
            self._state["sound"] = False
            for mechanism in self._mechanisms:
                if self._rng.random() < 0.20:
                    self._hidden_enabled[mechanism.action] = (
                        not self._hidden_enabled[mechanism.action]
                    )
        return self.observe()

    def true_edges(self) -> set[Edge]:
        edges: set[Edge] = {("wait", "sound")}
        for variable in self._visible_variables:
            edges.add((f"set_{variable}_on", variable))
            edges.add((f"set_{variable}_off", variable))
        for target in self._targets:
            edges.add((f"reset_{target}", target))
        for mechanism in self._mechanisms:
            edges.add((mechanism.action, "sound"))
            edges.add((mechanism.action, mechanism.target))
        edges.add((f"poke_decoy_{self.family_seed}", "sound"))
        edges.add((f"poke_decoy_{self.family_seed}", "decoy_flag"))
        edges.add((f"toggle_decoy_{self.family_seed}", "decoy_flag"))
        return edges

    def readout_variables(self) -> set[str]:
        return set(self._readout_sources)

    def hidden_context_edges(self) -> set[Edge]:
        return {
            (mechanism.action, mechanism.target)
            for mechanism in self._mechanisms
        }

    def procedural_spec(self) -> dict[str, object]:
        return {
            "family_seed": self.family_seed,
            "visible_variables": list(self._visible_variables),
            "targets": list(self._targets),
            "mechanisms": [
                {
                    "action": mechanism.action,
                    "target": mechanism.target,
                    "visible_conditions": list(mechanism.visible_conditions),
                    "hidden_enabled_probability": (
                        mechanism.hidden_enabled_probability
                    ),
                }
                for mechanism in self._mechanisms
            ],
            "readout_sources": {
                readout: list(sources)
                for readout, sources in self._readout_sources.items()
            },
        }

    def _refresh_hidden_context(
        self,
        mechanism: ProceduralHiddenMechanismSpec,
    ) -> None:
        self._hidden_enabled[mechanism.action] = (
            self._rng.random() < mechanism.hidden_enabled_probability
        )

    def _maybe_flip(self, value: bool) -> bool:
        if self._rng.random() < self.noise_probability:
            return not value
        return value


def make_procedural_complex_hidden_world(
    family_seed: int,
    mechanism_count: int = 4,
    visible_count: int = 5,
    noise_probability: float = 0.05,
    readout_count: int = 6,
) -> ProceduralComplexNoisyHiddenWorld:
    return ProceduralComplexNoisyHiddenWorld(
        family_seed=family_seed,
        mechanism_count=mechanism_count,
        visible_count=visible_count,
        noise_probability=noise_probability,
        readout_count=readout_count,
    )


def make_world(name: str) -> DoorLampWorld | PanelWorld:
    worlds = {
        DoorLampWorld.name: DoorLampWorld,
        DoorLampShiftedWorld.name: DoorLampShiftedWorld,
        DoorLampInvertedWorld.name: DoorLampInvertedWorld,
        AmbiguousGateWorld.name: AmbiguousGateWorld,
        PanelWorld.name: PanelWorld,
        AmbiguousPanelGateWorld.name: AmbiguousPanelGateWorld,
        DelayedLampWorld.name: DelayedLampWorld,
        DelayedLampShiftedWorld.name: DelayedLampShiftedWorld,
        DelayedLampLongDelayWorld.name: DelayedLampLongDelayWorld,
        DualDelayedControlWorld.name: DualDelayedControlWorld,
        DualDelayedSelectiveShiftWorld.name: DualDelayedSelectiveShiftWorld,
        RenamedDualDelayedControlWorld.name: RenamedDualDelayedControlWorld,
        RenamedDualDelayedSelectiveShiftWorld.name: (
            RenamedDualDelayedSelectiveShiftWorld
        ),
        TripleDelayedControlWorld.name: TripleDelayedControlWorld,
        RenamedTripleDelayedControlWorld.name: RenamedTripleDelayedControlWorld,
        RenamedTripleDelayedDiagnosticShiftWorld.name: (
            RenamedTripleDelayedDiagnosticShiftWorld
        ),
        PanelShiftedWorld.name: PanelShiftedWorld,
        PanelInvertedWorld.name: PanelInvertedWorld,
        DerivedSensorPanelWorld.name: DerivedSensorPanelWorld,
        OpaqueReadoutPanelWorld.name: OpaqueReadoutPanelWorld,
        NoisyCorePanelWorld.name: NoisyCorePanelWorld,
        HiddenContextPanelWorld.name: HiddenContextPanelWorld,
        NoisyHiddenPanelWorld.name: NoisyHiddenPanelWorld,
        ComplexNoisyHiddenPanelWorld.name: ComplexNoisyHiddenPanelWorld,
    }
    try:
        return worlds[name]()
    except KeyError as exc:
        available = ", ".join(sorted(worlds))
        raise ValueError(f"unknown world {name!r}; available: {available}") from exc


def world_names() -> list[str]:
    return [
        DoorLampWorld.name,
        DoorLampShiftedWorld.name,
        DoorLampInvertedWorld.name,
        AmbiguousGateWorld.name,
        PanelWorld.name,
        AmbiguousPanelGateWorld.name,
        DelayedLampWorld.name,
        DelayedLampShiftedWorld.name,
        DelayedLampLongDelayWorld.name,
        DualDelayedControlWorld.name,
        DualDelayedSelectiveShiftWorld.name,
        RenamedDualDelayedControlWorld.name,
        RenamedDualDelayedSelectiveShiftWorld.name,
        TripleDelayedControlWorld.name,
        RenamedTripleDelayedControlWorld.name,
        RenamedTripleDelayedDiagnosticShiftWorld.name,
        PanelShiftedWorld.name,
        PanelInvertedWorld.name,
        DerivedSensorPanelWorld.name,
        OpaqueReadoutPanelWorld.name,
        NoisyCorePanelWorld.name,
        HiddenContextPanelWorld.name,
        NoisyHiddenPanelWorld.name,
        ComplexNoisyHiddenPanelWorld.name,
    ]
