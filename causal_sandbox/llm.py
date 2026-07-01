from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
from typing import Protocol

from .agents import (
    CausalObservationAdapterV2Agent,
    ControlExperimentPlannerLatentContextAgent,
    ContextSearchControlExperimentPlannerAgent,
    PersistentLatentContextObservationAdapterAgent,
    ProactiveLatentContextObservationAdapterAgent,
)
from .core import ActionSpec, AgentDecision, Edge, State, Transition
from .memory import CausalMemory


class TextGenerator(Protocol):
    def generate(self, system: str, user: str) -> str:
        ...


class LocalHFChatModel:
    """Small wrapper around a local HuggingFace chat model.

    Imports are intentionally local so the sandbox still runs without
    transformers unless the LLM pilot is invoked.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        local_files_only: bool = True,
        torch_dtype: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            local_files_only=local_files_only,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._torch = torch

    def generate(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(text, return_tensors="pt").to(device)
        with self._torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


@dataclass(frozen=True)
class LLMActionResponse:
    action: str
    prediction: frozenset[str]
    reason: str
    raw_text: str


class LLMControllerAgent:
    name = "llm-vanilla"

    def __init__(
        self,
        generator: TextGenerator,
        seed: int | None = None,
        causal_prompt: bool = False,
    ) -> None:
        self.generator = generator
        self.seed = seed
        self.causal_prompt = causal_prompt
        self.actions: list[ActionSpec] = []
        self.memory = CausalMemory()
        self._action_counts: Counter[str] = Counter()

    def reset(self, actions: list[ActionSpec]) -> None:
        self.actions = list(actions)
        self.memory = CausalMemory()
        self._action_counts = Counter()

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        response = self._ask_model(observation, history, module_summary=None)
        prediction = response.prediction or self.memory.predicted_changes_for(
            response.action
        )
        return AgentDecision(
            action=response.action,
            prediction=prediction,
            hypothesis=(
                f"LLM chose {response.action}: {response.reason} "
                f"Raw response: {response.raw_text[:240]}"
            ),
        )

    def observe_transition(self, transition: Transition) -> None:
        self.memory.record(transition)
        self._action_counts[transition.action] += 1

    def discovered_edges(self) -> set[Edge]:
        return self.memory.discovered_edges()

    def predict(self, action: str, state: State) -> frozenset[str]:
        return self.memory.predicted_changes_for(action)

    def predict_next_state(self, action: str, state: State) -> State:
        return self.memory.predicted_state_for(action, state)

    def condition_hints(self) -> dict[Edge, list[str]]:
        return self.memory.condition_hints()

    def _ask_model(
        self,
        observation: State,
        history: tuple[Transition, ...],
        module_summary: str | None,
        required_action: str | None = None,
    ) -> LLMActionResponse:
        action_names = [action.name for action in self.actions]
        system = _system_prompt(causal_prompt=self.causal_prompt)
        user = _user_prompt(
            observation=observation,
            actions=self.actions,
            history=history,
            module_summary=module_summary,
            required_action=required_action,
        )
        raw_text = self.generator.generate(system, user)
        return _parse_llm_response(
            raw_text,
            actions=action_names,
            variables=set(observation),
            fallback_action=self._fallback_action(),
        )

    def _fallback_action(self) -> str:
        return min(
            (action.name for action in self.actions),
            key=lambda name: (self._action_counts[name], name),
        )


class LLMCausalModuleAgent(LLMControllerAgent):
    name = "llm-causal-module"

    def __init__(
        self,
        generator: TextGenerator,
        module: CausalObservationAdapterV2Agent
        | PersistentLatentContextObservationAdapterAgent
        | ProactiveLatentContextObservationAdapterAgent
        | ControlExperimentPlannerLatentContextAgent
        | ContextSearchControlExperimentPlannerAgent,
        name: str,
        seed: int | None = None,
        causal_prompt: bool = True,
    ) -> None:
        super().__init__(
            generator=generator,
            seed=seed,
            causal_prompt=causal_prompt,
        )
        self.name = name
        self.module = module

    def reset(self, actions: list[ActionSpec]) -> None:
        super().reset(actions)
        self.module.reset(actions)

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        module_decision = self.module.choose_action(observation, history)
        module_summary = _module_summary(
            module_name=self.module.name,
            recommendation=module_decision,
            discovered_edges=self.module.discovered_edges(),
            condition_hints=self.module.condition_hints(),
        )
        response = self._ask_model(observation, history, module_summary=module_summary)
        prediction = self.module.predict(response.action, observation)
        if not prediction:
            prediction = response.prediction
        return AgentDecision(
            action=response.action,
            prediction=prediction,
            hypothesis=(
                f"LLM with {self.module.name} chose {response.action}; "
                f"module recommended {module_decision.action}. "
                f"Reason: {response.reason}. Raw response: {response.raw_text[:240]}"
            ),
        )

    def observe_transition(self, transition: Transition) -> None:
        super().observe_transition(transition)
        self.module.observe_transition(transition)

    def discovered_edges(self) -> set[Edge]:
        return self.module.discovered_edges()

    def predict(self, action: str, state: State) -> frozenset[str]:
        return self.module.predict(action, state)

    def predict_next_state(self, action: str, state: State) -> State:
        return self.module.predict_next_state(action, state)

    def condition_hints(self) -> dict[Edge, list[str]]:
        return self.module.condition_hints()


class LLMAuthoritativeCausalModuleAgent(LLMCausalModuleAgent):
    """LLM stack where the causal module is the action controller.

    The LLM still receives the causal memory and can produce a rationale, but
    the environment action is the module recommendation. This tests whether an
    LLM-facing agent benefits when causal memory is integrated as a hard
    controller rather than a passive prompt hint.
    """

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        module_decision = self.module.choose_action(observation, history)
        module_summary = _module_summary(
            module_name=self.module.name,
            recommendation=module_decision,
            discovered_edges=self.module.discovered_edges(),
            condition_hints=self.module.condition_hints(),
        )
        response = self._ask_model(
            observation,
            history,
            module_summary=module_summary,
            required_action=module_decision.action,
        )
        action = module_decision.action
        prediction = module_decision.prediction or self.module.predict(
            action,
            observation,
        )
        if not prediction:
            prediction = response.prediction
        override = response.action != action
        return AgentDecision(
            action=action,
            prediction=prediction,
            hypothesis=(
                f"Module-authoritative LLM stack executed {action}; "
                f"LLM proposed {response.action}; override={override}. "
                f"Reason: {response.reason}. Raw response: {response.raw_text[:240]}"
            ),
        )


class LLMGatedCausalModuleAgent(LLMCausalModuleAgent):
    """LLM stack where a causal gate can veto risky action proposals."""

    def choose_action(
        self, observation: State, history: tuple[Transition, ...]
    ) -> AgentDecision:
        module_decision = self.module.choose_action(observation, history)
        module_summary = _module_summary(
            module_name=self.module.name,
            recommendation=module_decision,
            discovered_edges=self.module.discovered_edges(),
            condition_hints=self.module.condition_hints(),
        )
        response = self._ask_model(observation, history, module_summary=module_summary)
        action, gate_reason = self._gate_action(
            proposed_action=response.action,
            module_action=module_decision.action,
            observation=observation,
        )
        prediction = self.module.predict(action, observation)
        if not prediction and action == module_decision.action:
            prediction = module_decision.prediction
        if not prediction and action == response.action:
            prediction = response.prediction
        return AgentDecision(
            action=action,
            prediction=prediction,
            hypothesis=(
                f"Module-gated LLM stack executed {action}; "
                f"LLM proposed {response.action}; module recommended "
                f"{module_decision.action}; gate={gate_reason}. "
                f"Reason: {response.reason}. Raw response: {response.raw_text[:240]}"
            ),
        )

    def _gate_action(
        self,
        proposed_action: str,
        module_action: str,
        observation: State,
    ) -> tuple[str, str]:
        if proposed_action == module_action:
            return proposed_action, "accepted-module-match"

        action_names = [action.name for action in self.actions]
        min_count = min(self._action_counts[name] for name in action_names)
        proposed_count = self._action_counts[proposed_action]
        module_count = self._action_counts[module_action]

        if proposed_count > module_count + 1:
            return module_action, "vetoed-overused-proposal"
        if module_count == min_count and proposed_count > min_count:
            return module_action, "vetoed-less-tested-module-action"
        if proposed_count <= min_count + 1:
            return proposed_action, "accepted-balanced-exploration"
        if self.module.predict(proposed_action, observation):
            return proposed_action, "accepted-known-causal-action"
        return module_action, "vetoed-unsupported-proposal"


def make_llm_agent(
    variant: str,
    generator: TextGenerator,
    seed: int | None = None,
) -> LLMControllerAgent:
    if variant == "llm-vanilla":
        return LLMControllerAgent(generator, seed=seed, causal_prompt=False)
    if variant == "llm-causal-prompt":
        agent = LLMControllerAgent(generator, seed=seed, causal_prompt=True)
        agent.name = "llm-causal-prompt"
        return agent
    if variant == "llm-observation-adapter":
        return LLMCausalModuleAgent(
            generator,
            module=CausalObservationAdapterV2Agent(),
            name="llm-observation-adapter",
            seed=seed,
            causal_prompt=True,
        )
    if variant == "llm-persistent-latent":
        return LLMCausalModuleAgent(
            generator,
            module=PersistentLatentContextObservationAdapterAgent(),
            name="llm-persistent-latent",
            seed=seed,
            causal_prompt=True,
        )
    if variant == "llm-observation-adapter-authoritative":
        return LLMAuthoritativeCausalModuleAgent(
            generator,
            module=CausalObservationAdapterV2Agent(),
            name="llm-observation-adapter-authoritative",
            seed=seed,
            causal_prompt=True,
        )
    if variant == "llm-persistent-latent-authoritative":
        return LLMAuthoritativeCausalModuleAgent(
            generator,
            module=PersistentLatentContextObservationAdapterAgent(),
            name="llm-persistent-latent-authoritative",
            seed=seed,
            causal_prompt=True,
        )
    if variant == "llm-observation-adapter-gated":
        return LLMGatedCausalModuleAgent(
            generator,
            module=CausalObservationAdapterV2Agent(),
            name="llm-observation-adapter-gated",
            seed=seed,
            causal_prompt=True,
        )
    if variant == "llm-persistent-latent-gated":
        return LLMGatedCausalModuleAgent(
            generator,
            module=PersistentLatentContextObservationAdapterAgent(),
            name="llm-persistent-latent-gated",
            seed=seed,
            causal_prompt=True,
        )
    if variant == "llm-proactive-latent-gated":
        return LLMGatedCausalModuleAgent(
            generator,
            module=ProactiveLatentContextObservationAdapterAgent(),
            name="llm-proactive-latent-gated",
            seed=seed,
            causal_prompt=True,
        )
    if variant == "llm-control-planner-gated":
        return LLMGatedCausalModuleAgent(
            generator,
            module=ControlExperimentPlannerLatentContextAgent(),
            name="llm-control-planner-gated",
            seed=seed,
            causal_prompt=True,
        )
    if variant == "llm-context-search-gated":
        return LLMGatedCausalModuleAgent(
            generator,
            module=ContextSearchControlExperimentPlannerAgent(),
            name="llm-context-search-gated",
            seed=seed,
            causal_prompt=True,
        )
    available = ", ".join(llm_agent_names())
    raise ValueError(f"unknown LLM variant {variant!r}; available: {available}")


def llm_agent_names() -> list[str]:
    return [
        "llm-vanilla",
        "llm-causal-prompt",
        "llm-observation-adapter",
        "llm-persistent-latent",
        "llm-observation-adapter-authoritative",
        "llm-persistent-latent-authoritative",
        "llm-observation-adapter-gated",
        "llm-persistent-latent-gated",
        "llm-proactive-latent-gated",
        "llm-control-planner-gated",
        "llm-context-search-gated",
    ]


def _system_prompt(causal_prompt: bool) -> str:
    base = (
        "You control a small Boolean world. Your task is to choose one action "
        "that helps discover causal effects. Return only compact JSON with keys "
        "'action', 'prediction', and 'reason'. The action must be one of the "
        "available action names. The prediction must be a list of observed "
        "variables you expect to change after the action."
    )
    if not causal_prompt:
        return base
    return (
        base
        + " Prefer controlled interventions over reward seeking. Re-test actions "
        "under different contexts, avoid treating readout variables as causes, "
        "and suspect hidden context when the same intervention sometimes succeeds "
        "and sometimes fails under the same observed state."
    )


def _user_prompt(
    observation: State,
    actions: list[ActionSpec],
    history: tuple[Transition, ...],
    module_summary: str | None,
    required_action: str | None = None,
) -> str:
    lines = [
        "Current observation:",
        _format_state(observation),
        "",
        "Available actions:",
    ]
    lines.extend(f"- {action.name}: {action.description}" for action in actions)
    lines.extend(["", "Recent transitions:"])
    if history:
        for transition in history[-8:]:
            changed = ", ".join(sorted(transition.changed)) or "nothing"
            predicted = ", ".join(sorted(transition.prediction)) or "unknown"
            before = _format_state(transition.before)
            after = _format_state(transition.after)
            lines.append(
                f"- step {transition.step}: action={transition.action}; "
                f"before=({before}); predicted=[{predicted}]; "
                f"changed=[{changed}]; after=({after})"
            )
    else:
        lines.append("- none")
    if module_summary:
        lines.extend(["", "Causal module summary:", module_summary])
    if required_action is not None:
        lines.extend(
            [
                "",
                (
                    "The causal module is controlling the next intervention. "
                    f"Use action={required_action} in your JSON and explain why "
                    "this intervention is useful."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Return JSON only, for example:",
            '{"action":"press_a","prediction":["lamp_on"],"reason":"test button A under current context"}',
        ]
    )
    return "\n".join(lines)


def _module_summary(
    module_name: str,
    recommendation: AgentDecision,
    discovered_edges: set[Edge],
    condition_hints: dict[Edge, list[str]],
) -> str:
    edges = []
    for edge in sorted(discovered_edges):
        suffix = ""
        if edge in condition_hints:
            suffix = " when " + ", ".join(condition_hints[edge])
        edges.append(f"{edge[0]}->{edge[1]}{suffix}")
    edge_text = "; ".join(edges[:18]) or "none yet"
    return (
        f"module={module_name}; recommended_action={recommendation.action}; "
        f"recommended_prediction={sorted(recommendation.prediction)}; "
        f"known_edges={edge_text}"
    )


def _parse_llm_response(
    raw_text: str,
    actions: list[str],
    variables: set[str],
    fallback_action: str,
) -> LLMActionResponse:
    payload = _extract_json_object(raw_text)
    action = ""
    prediction: frozenset[str] = frozenset()
    reason = ""
    if isinstance(payload, dict):
        action = str(payload.get("action", ""))
        prediction = _parse_prediction(payload.get("prediction"), variables)
        reason = str(payload.get("reason", ""))
    if action not in actions:
        action = _extract_action(raw_text, actions) or fallback_action
    if not prediction:
        prediction = frozenset(
            variable
            for variable in sorted(variables)
            if re.search(rf"\b{re.escape(variable)}\b", raw_text)
        )
    if not reason:
        reason = "parsed from model response"
    return LLMActionResponse(
        action=action,
        prediction=prediction,
        reason=reason,
        raw_text=raw_text,
    )


def _extract_json_object(text: str) -> object | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _parse_prediction(value: object, variables: set[str]) -> frozenset[str]:
    if isinstance(value, str):
        values = re.split(r"[,;\s]+", value)
    elif isinstance(value, list):
        values = [str(item) for item in value]
    else:
        values = []
    return frozenset(item for item in values if item in variables)


def _extract_action(text: str, actions: list[str]) -> str | None:
    for action in actions:
        if re.search(rf"\b{re.escape(action)}\b", text):
            return action
    return None


def _format_state(state: State) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(state.items()))
