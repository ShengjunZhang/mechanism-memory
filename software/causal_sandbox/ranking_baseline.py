from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from random import Random
from typing import Callable

from .agents import make_agent
from .core import Edge, State, World
from .evaluation import run_episode, run_level6_active_diagnostic_repair_test
from .worlds import (
    make_procedural_complex_hidden_world,
    make_procedural_diagnostic_world_pair,
)


WorldFactory = Callable[[], World]


@dataclass(frozen=True)
class InterventionCandidate:
    name: str
    actions: tuple[str, ...]


@dataclass(frozen=True)
class RankingExample:
    state: State
    candidates: tuple[InterventionCandidate, ...]
    scores: dict[str, float]

    @property
    def best_candidate(self) -> str:
        return max(
            self.scores,
            key=lambda name: (self.scores[name], _reverse_name(name)),
        )


class PairwiseRankingPredictor:
    """Small ranking-loss baseline for intervention choice.

    The model optimizes top intervention ranking from supervised candidate
    scores. It deliberately does not maintain an action-target mechanism
    memory, so it can be good at choosing interventions while still failing
    mechanism-level probes.
    """

    def __init__(self, learning_rate: float = 1.0, epochs: int = 12) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: Counter[str] = Counter()

    def fit(self, examples: tuple[RankingExample, ...]) -> None:
        for _ in range(self.epochs):
            for example in examples:
                best = example.best_candidate
                predicted = self.predict(example.state, example.candidates)
                if predicted == best:
                    continue
                if example.scores[best] <= example.scores[predicted]:
                    continue
                self._update(example.state, best, +self.learning_rate)
                self._update(example.state, predicted, -self.learning_rate)

    def predict(
        self,
        state: State,
        candidates: tuple[InterventionCandidate, ...],
    ) -> str:
        return max(
            candidates,
            key=lambda candidate: (
                self.score(state, candidate.name),
                _reverse_name(candidate.name),
            ),
        ).name

    def score(self, state: State, candidate_name: str) -> float:
        return sum(
            self.weights[feature] * value
            for feature, value in self._features(state, candidate_name).items()
        )

    def _update(self, state: State, candidate_name: str, scale: float) -> None:
        for feature, value in self._features(state, candidate_name).items():
            self.weights[feature] += scale * value

    @staticmethod
    def _features(state: State, candidate_name: str) -> dict[str, float]:
        features = {f"candidate:{candidate_name}": 1.0}
        for token in candidate_name.replace("|", "_").split("_"):
            if token:
                features[f"candidate-token:{token}"] = 1.0
        for variable, value in sorted(state.items()):
            bit = "1" if value else "0"
            features[f"state:{variable}={bit}"] = 0.25
            features[f"candidate:{candidate_name}|{variable}={bit}"] = 1.0
        return features


def run_ranking_loss_baseline_experiment(
    families: int = 6,
    seeds: int = 2,
    train_states: int = 64,
    test_states: int = 64,
    hidden_steps: int = 160,
    epochs: int = 12,
) -> dict[str, object]:
    """Run the ranking-loss predictor against mechanism-level probes."""

    hidden_rows = []
    l6_rows = []
    readout_l6_rows = []

    for family_seed in range(1, families + 1):
        hidden_factory = lambda family_seed=family_seed: make_procedural_complex_hidden_world(
            family_seed=family_seed,
            mechanism_count=4,
            visible_count=5,
            noise_probability=0.02,
            readout_count=6,
        )
        hidden_candidates = _primitive_candidates(hidden_factory())
        for seed in range(1, seeds + 1):
            predictor = PairwiseRankingPredictor(epochs=epochs)
            predictor.fit(
                _ranking_examples(
                    hidden_factory,
                    hidden_candidates,
                    count=train_states,
                    seed=1000 * family_seed + seed,
                )
            )
            rank_top1 = _ranking_accuracy(
                predictor,
                hidden_factory,
                hidden_candidates,
                count=test_states,
                seed=2000 * family_seed + seed,
            )
            hidden_rows.append(
                {
                    "family_seed": family_seed,
                    "seed": seed,
                    "rank_top1": rank_top1,
                    "mechanism_f1": 0.0,
                    "hidden_context_recall": 0.0,
                    "readout_false_positive": 0.0,
                }
            )

    hidden_core_rows = []
    for family_seed in range(1, families + 1):
        world = make_procedural_complex_hidden_world(
            family_seed=family_seed,
            mechanism_count=4,
            visible_count=5,
            noise_probability=0.02,
            readout_count=6,
        )
        readouts = world.readout_variables()
        hidden_edges = world.hidden_context_edges()
        for seed in range(1, seeds + 1):
            result = run_episode(
                make_procedural_complex_hidden_world(
                    family_seed=family_seed,
                    mechanism_count=4,
                    visible_count=5,
                    noise_probability=0.02,
                    readout_count=6,
                ),
                make_agent("causal-core-context-search-planner", seed=seed),
                steps=hidden_steps,
                seed=seed,
            )
            hidden_core_rows.append(
                _mechanism_row(
                    result.discovered_edges,
                    result.true_edges,
                    readouts,
                    hidden_edges,
                )
            )

    for readout_mode, bucket in (
        ("none", l6_rows),
        ("noisy-semantic-confounder", readout_l6_rows),
    ):
        for family_seed in range(1, families + 1):
            source_world, target_world = make_procedural_diagnostic_world_pair(
                family_seed=family_seed,
                mechanism_count=3,
                readout_mode=readout_mode,
            )
            source_factory = lambda family_seed=family_seed, readout_mode=readout_mode: make_procedural_diagnostic_world_pair(
                family_seed=family_seed,
                mechanism_count=3,
                readout_mode=readout_mode,
            )[0]
            target_factory = lambda family_seed=family_seed, readout_mode=readout_mode: make_procedural_diagnostic_world_pair(
                family_seed=family_seed,
                mechanism_count=3,
                readout_mode=readout_mode,
            )[1]
            source_candidates = _temporal_candidates(source_world)
            target_candidates = _temporal_candidates(target_world)
            predictor = PairwiseRankingPredictor(epochs=epochs)
            predictor.fit(
                _ranking_examples(
                    source_factory,
                    source_candidates,
                    count=train_states,
                    seed=3000 + family_seed,
                )
            )
            source_rank_top1 = _ranking_accuracy(
                predictor,
                source_factory,
                source_candidates,
                count=test_states,
                seed=4000 + family_seed,
            )
            target_rank_top1 = _ranking_accuracy(
                predictor,
                target_factory,
                target_candidates,
                count=test_states,
                seed=5000 + family_seed,
            )
            core_result = run_level6_active_diagnostic_repair_test(
                source_world,
                target_world,
                make_agent("causal-core-temporal-diagnostic", seed=family_seed),
                explore_steps=12,
                repair_steps=1,
                seed=family_seed,
                transfer_mode="schema-prior",
            )
            bucket.append(
                {
                    "family_seed": family_seed,
                    "source_rank_top1": source_rank_top1,
                    "target_transfer_rank_top1": target_rank_top1,
                    "ranking_predictor_l6_repair": 0.0,
                    "causal_core_l6_repair": core_result.level6_repair_score,
                    "causal_core_after_score": core_result.after_score,
                    "readout_mode": readout_mode,
                }
            )

    summary = [
        {
            "task": "procedural noisy-hidden ranking",
            "ranking_predictor_rank_top1": _mean(
                [float(row["rank_top1"]) for row in hidden_rows]
            ),
            "ranking_predictor_mechanism_f1": 0.0,
            "ranking_predictor_hidden_recall": 0.0,
            "causal_core_mechanism_f1": _mean(
                [float(row["f1"]) for row in hidden_core_rows]
            ),
            "causal_core_hidden_recall": _mean(
                [float(row["hidden_context_recall"]) for row in hidden_core_rows]
            ),
            "causal_core_readout_fp": _mean(
                [float(row["readout_false_positive"]) for row in hidden_core_rows]
            ),
        },
        {
            "task": "L6 renamed diagnostic repair",
            "ranking_predictor_source_rank_top1": _mean(
                [float(row["source_rank_top1"]) for row in l6_rows]
            ),
            "ranking_predictor_target_transfer_rank_top1": _mean(
                [float(row["target_transfer_rank_top1"]) for row in l6_rows]
            ),
            "ranking_predictor_l6_repair": 0.0,
            "causal_core_l6_repair": _mean(
                [float(row["causal_core_l6_repair"]) for row in l6_rows]
            ),
            "causal_core_after_score": _mean(
                [float(row["causal_core_after_score"]) for row in l6_rows]
            ),
        },
        {
            "task": "L6 noisy semantic-readout repair",
            "ranking_predictor_source_rank_top1": _mean(
                [float(row["source_rank_top1"]) for row in readout_l6_rows]
            ),
            "ranking_predictor_target_transfer_rank_top1": _mean(
                [float(row["target_transfer_rank_top1"]) for row in readout_l6_rows]
            ),
            "ranking_predictor_l6_repair": 0.0,
            "causal_core_l6_repair": _mean(
                [float(row["causal_core_l6_repair"]) for row in readout_l6_rows]
            ),
            "causal_core_after_score": _mean(
                [float(row["causal_core_after_score"]) for row in readout_l6_rows]
            ),
        },
    ]
    return {
        "model": "pairwise-ranking-loss-predictor",
        "families": families,
        "seeds": seeds,
        "train_states": train_states,
        "test_states": test_states,
        "hidden_steps": hidden_steps,
        "epochs": epochs,
        "summary": summary,
        "runs": {
            "hidden": hidden_rows,
            "hidden_causal_core": hidden_core_rows,
            "l6": l6_rows,
            "readout_l6": readout_l6_rows,
        },
    }


def format_ranking_loss_baseline(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    lines = [
        "Ranking-loss predictor baseline",
        (
            f"Families={payload['families']}, seeds={payload['seeds']}, "
            f"train_states={payload['train_states']}, "
            f"test_states={payload['test_states']}"
        ),
        "",
        (
            "task                                rank/source  target-rank  "
            "rank-mech/L6  core-mech/L6  core-hidden/readout"
        ),
        "-" * 101,
    ]
    for row in summary:  # type: ignore[assignment]
        row = dict(row)  # type: ignore[arg-type]
        task = str(row["task"])
        if task.startswith("procedural"):
            lines.append(
                f"{task:36}"
                f"{float(row['ranking_predictor_rank_top1']):>11.2f}"
                f"{'--':>13}"
                f"{float(row['ranking_predictor_mechanism_f1']):>13.2f}"
                f"{float(row['causal_core_mechanism_f1']):>14.2f}"
                f"{float(row['causal_core_hidden_recall']):>10.2f}/"
                f"{float(row['causal_core_readout_fp']):<5.2f}"
            )
        else:
            lines.append(
                f"{task:36}"
                f"{float(row['ranking_predictor_source_rank_top1']):>11.2f}"
                f"{float(row['ranking_predictor_target_transfer_rank_top1']):>13.2f}"
                f"{float(row['ranking_predictor_l6_repair']):>13.2f}"
                f"{float(row['causal_core_l6_repair']):>14.2f}"
                f"{float(row['causal_core_after_score']):>10.2f}"
            )
    lines.extend(
        [
            "",
            "Interpretation: the ranking-loss predictor is evaluated on its own "
            "top-intervention objective, but it exposes no action-target "
            "mechanism memory. Its mechanism-level and L6 repair scores are "
            "therefore zero by construction, while the causal-core rows report "
            "the mechanism evaluator used in the paper.",
        ]
    )
    return "\n".join(lines)


def _ranking_examples(
    world_factory: WorldFactory,
    candidates: tuple[InterventionCandidate, ...],
    count: int,
    seed: int,
) -> tuple[RankingExample, ...]:
    states = _sample_states(world_factory, count=count, seed=seed)
    examples: list[RankingExample] = []
    for index, state in enumerate(states):
        scores = {
            candidate.name: _candidate_score(
                world_factory,
                state,
                candidate,
                seed=seed + index * 997,
            )
            for candidate in candidates
        }
        if len(set(scores.values())) <= 1:
            continue
        examples.append(
            RankingExample(
                state=dict(state),
                candidates=candidates,
                scores=scores,
            )
        )
    return tuple(examples)


def _ranking_accuracy(
    predictor: PairwiseRankingPredictor,
    world_factory: WorldFactory,
    candidates: tuple[InterventionCandidate, ...],
    count: int,
    seed: int,
) -> float:
    examples = _ranking_examples(world_factory, candidates, count=count, seed=seed)
    if not examples:
        return 0.0
    correct = sum(
        predictor.predict(example.state, example.candidates) == example.best_candidate
        for example in examples
    )
    return correct / len(examples)


def _candidate_score(
    world_factory: WorldFactory,
    state: State,
    candidate: InterventionCandidate,
    seed: int,
) -> float:
    scores = []
    for repeat in range(3):
        world = world_factory()
        world.reset(seed=seed + repeat)
        setter = getattr(world, "set_state", None)
        if setter is not None:
            setter(state)
        after: State = dict(state)
        for action in candidate.actions:
            after = world.step(action)
        scores.append(_state_score(world, after))
    return sum(scores) / len(scores)


def _state_score(world: World, state: State) -> float:
    readouts = _readout_variables(world)
    variables = _score_variables(world, state)
    score = sum(1.0 for variable in variables if state.get(variable, False))
    if state.get("sound", False):
        score -= 0.05
    for variable, value in state.items():
        if variable in readouts and value:
            score += 0.01
    return score


def _score_variables(world: World, state: State) -> tuple[str, ...]:
    variables: set[str] = set()
    temporal_specs = getattr(world, "temporal_specs", None)
    if temporal_specs is not None:
        for spec in temporal_specs():
            variables.add(str(spec["target_variable"]))
    hidden_edges = getattr(world, "hidden_context_edges", None)
    if hidden_edges is not None:
        variables.update(target for _, target in hidden_edges())
    if not variables:
        variables.update(
            variable
            for variable in ("lamp_on", "door_open", "alarm_on", "fan_on")
            if variable in state
        )
    if not variables:
        readouts = _readout_variables(world)
        variables.update(
            variable
            for variable in state
            if variable not in readouts
            and not variable.startswith("ctx_")
            and variable not in {"sound", "decoy_flag"}
            and "decoy" not in variable
            and "sensor" not in variable
        )
    return tuple(sorted(variables))


def _sample_states(
    world_factory: WorldFactory,
    count: int,
    seed: int,
) -> tuple[State, ...]:
    world = world_factory()
    initial = world.reset(seed=seed)
    variables = sorted(initial)
    if len(variables) <= 10:
        states = [
            dict(zip(variables, values))
            for values in product([False, True], repeat=len(variables))
        ]
        rng = Random(seed)
        rng.shuffle(states)
        return tuple(states[:count])

    rng = Random(seed)
    actions = [action.name for action in world.actions()]
    states: list[State] = [dict(initial)]
    observation = initial
    for _ in range(max(count * 4, 20)):
        action = rng.choice(actions)
        observation = world.step(action)
        states.append(dict(observation))
        if rng.random() < 0.15:
            observation = world.reset(seed=rng.randrange(1_000_000))
            states.append(dict(observation))

    unique: dict[tuple[tuple[str, bool], ...], State] = {}
    for state in states:
        unique.setdefault(tuple(sorted(state.items())), state)
    sampled = list(unique.values())
    rng.shuffle(sampled)
    return tuple(sampled[:count])


def _primitive_candidates(world: World) -> tuple[InterventionCandidate, ...]:
    return tuple(
        InterventionCandidate(name=action.name, actions=(action.name,))
        for action in world.actions()
    )


def _temporal_candidates(world: World) -> tuple[InterventionCandidate, ...]:
    temporal_specs = getattr(world, "temporal_specs", None)
    if temporal_specs is None:
        return _primitive_candidates(world)
    candidates = []
    for spec in temporal_specs():
        action = str(spec["target_action"])
        followup = str(spec["followup_action"])
        delay = int(spec["delay_steps"])
        actions = (action,) + (followup,) * max(1, delay)
        candidates.append(
            InterventionCandidate(
                name="|".join(actions),
                actions=actions,
            )
        )
    return tuple(candidates)


def _mechanism_row(
    discovered_edges: set[Edge],
    true_edges: set[Edge],
    readout_variables: set[str],
    hidden_edges: set[Edge],
) -> dict[str, float]:
    true_positive = len(discovered_edges & true_edges)
    false_positive = len(discovered_edges - true_edges)
    false_negative = len(true_edges - discovered_edges)
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "readout_false_positive": float(
            sum(1 for _, target in discovered_edges - true_edges if target in readout_variables)
        ),
        "hidden_context_recall": _safe_div(
            len(discovered_edges & hidden_edges),
            len(hidden_edges),
        ),
    }


def _readout_variables(world: World) -> set[str]:
    readout_fn = getattr(world, "readout_variables", None)
    if readout_fn is None:
        return set()
    return set(readout_fn())


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _reverse_name(name: str) -> str:
    return "".join(chr(255 - ord(char)) for char in name)
