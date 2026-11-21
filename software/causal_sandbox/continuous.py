from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
from random import Random
from statistics import mean, median


ContinuousState = dict[str, float]
ContinuousAction = dict[str, float]
MechanismAtom = tuple[str, str, str, str]


@dataclass(frozen=True)
class ContinuousScore:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    readout_false_positive: int
    context_error: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "readout_false_positive": self.readout_false_positive,
            "context_error": self.context_error,
        }


@dataclass(frozen=True)
class ContinuousRunResult:
    agent: str
    seed: int
    discovered: set[MechanismAtom]
    true_mechanisms: set[MechanismAtom]
    score: ContinuousScore

    def to_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            "seed": self.seed,
            "discovered": [_atom_to_dict(atom) for atom in sorted(self.discovered)],
            "true_mechanisms": [
                _atom_to_dict(atom) for atom in sorted(self.true_mechanisms)
            ],
            "score": self.score.to_dict(),
        }


class ContinuousControlSCM:
    """Small continuous-state SCM with a metric-gated direct mechanism."""

    readout_variables = frozenset({"pressure_sensor"})

    def __init__(self, seed: int | None = None, noise: float = 0.02) -> None:
        self._rng = Random(seed)
        self.noise = noise
        self.state: ContinuousState = {}

    def reset(self, seed: int | None = None) -> ContinuousState:
        self._rng = Random(seed)
        temp = self._rng.uniform(-1.0, 1.0)
        pressure = self._rng.uniform(-0.8, 0.8)
        self.state = {
            "temp": temp,
            "pressure": pressure,
            "pressure_sensor": pressure + self._rng.gauss(0.0, self.noise),
        }
        return dict(self.state)

    def set_state(self, state: ContinuousState) -> None:
        self.state = dict(state)

    def step(self, action: ContinuousAction) -> ContinuousState:
        heat = float(action.get("heat", 0.0))
        vent = float(action.get("vent", 0.0))
        temp0 = self.state["temp"]
        pressure0 = self.state["pressure"]
        gate = 1.0 if temp0 > 0.2 else 0.0
        temp1 = _clip(temp0 + 0.55 * heat + self._rng.gauss(0.0, self.noise))
        pressure1 = _clip(
            pressure0 - 0.50 * gate * vent + self._rng.gauss(0.0, self.noise)
        )
        self.state = {
            "temp": temp1,
            "pressure": pressure1,
            "pressure_sensor": pressure1 + self._rng.gauss(0.0, self.noise),
        }
        return dict(self.state)

    @staticmethod
    def context_bin(state: ContinuousState) -> str:
        return "high-temp" if state["temp"] > 0.2 else "low-temp"

    @staticmethod
    def true_mechanisms() -> set[MechanismAtom]:
        atoms: set[MechanismAtom] = set()
        for context in ("low-temp", "high-temp"):
            atoms.add(("heat+", context, "temp", "+"))
            atoms.add(("heat-", context, "temp", "-"))
        atoms.add(("vent+", "high-temp", "pressure", "-"))
        atoms.add(("vent-", "high-temp", "pressure", "+"))
        return atoms


def run_continuous_metric_poc(
    seeds: int = 20,
    random_trials: int = 160,
    regression_trials: int = 240,
    paired_samples: int = 8,
    noise: float = 0.02,
) -> dict[str, object]:
    agents = (
        "random-correlation",
        "global-linear-regression",
        "metric-causal-core",
    )
    runs: dict[str, list[ContinuousRunResult]] = {agent: [] for agent in agents}
    for seed in range(1, seeds + 1):
        true_atoms = ContinuousControlSCM.true_mechanisms()
        runs["random-correlation"].append(
            _run_random_correlation(seed, random_trials, true_atoms, noise=noise)
        )
        runs["global-linear-regression"].append(
            _run_global_regression(seed, regression_trials, true_atoms, noise=noise)
        )
        runs["metric-causal-core"].append(
            _run_metric_core(seed, paired_samples, true_atoms, noise=noise)
        )
    summary = [_summary_row(agent, rows) for agent, rows in runs.items()]
    return {
        "task": "continuous metric-packing proof of concept",
        "seeds": seeds,
        "random_trials": random_trials,
        "regression_trials": regression_trials,
        "paired_samples": paired_samples,
        "noise": noise,
        "agents": list(agents),
        "summary": summary,
        "runs": {
            agent: [result.to_dict() for result in results]
            for agent, results in runs.items()
        },
    }


def format_continuous_metric_poc(payload: dict[str, object]) -> str:
    lines = [
        "Continuous metric-packing proof of concept",
        (
            f"Seeds={payload['seeds']}, random_trials={payload['random_trials']}, "
            f"regression_trials={payload['regression_trials']}, "
            f"paired_samples={payload['paired_samples']}, "
            f"noise={payload.get('noise', 0.02)}"
        ),
        "",
        (
            f"{'agent':<28} {'precision':>9} {'recall':>7} {'f1':>5} "
            f"{'fp':>7} {'readout-fp':>10} {'context-err':>11}"
        ),
        "-" * 82,
    ]
    for row in payload["summary"]:  # type: ignore[index]
        lines.append(
            f"{row['agent']:<28} "
            f"{float(row['precision']):>9.2f} "
            f"{float(row['recall']):>7.2f} "
            f"{float(row['f1']):>5.2f} "
            f"{float(row['false_positive']):>7.2f} "
            f"{float(row['readout_false_positive']):>10.2f} "
            f"{float(row['context_error']):>11.2f}"
        )
    return "\n".join(lines)


def save_continuous_payload(payload: dict[str, object], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_random_correlation(
    seed: int,
    trials: int,
    true_atoms: set[MechanismAtom],
    noise: float,
) -> ContinuousRunResult:
    rng = Random(10_000 + seed)
    world = ContinuousControlSCM(seed=seed, noise=noise)
    discovered: set[MechanismAtom] = set()
    for trial in range(trials):
        before = _sample_state(
            rng,
            context=rng.choice(["low-temp", "high-temp"]),
            noise=noise,
        )
        world.set_state(before)
        action = {"heat": rng.uniform(-1.0, 1.0), "vent": rng.uniform(-1.0, 1.0)}
        after = world.step(action)
        axis = "heat" if abs(action["heat"]) >= abs(action["vent"]) else "vent"
        if abs(action[axis]) < 0.2:
            continue
        action_bin = f"{axis}{'+' if action[axis] > 0 else '-'}"
        context = world.context_bin(before)
        for target in ("temp", "pressure", "pressure_sensor"):
            delta = after[target] - before[target]
            if abs(delta) > 0.12:
                discovered.add((action_bin, context, target, "+" if delta > 0 else "-"))
    return _result("random-correlation", seed, discovered, true_atoms)


def _run_global_regression(
    seed: int,
    trials: int,
    true_atoms: set[MechanismAtom],
    noise: float,
) -> ContinuousRunResult:
    rng = Random(20_000 + seed)
    world = ContinuousControlSCM(seed=seed, noise=noise)
    rows: list[tuple[ContinuousAction, ContinuousState, ContinuousState]] = []
    for _ in range(trials):
        before = _sample_state(
            rng,
            context=rng.choice(["low-temp", "high-temp"]),
            noise=noise,
        )
        world.set_state(before)
        action = {"heat": rng.uniform(-1.0, 1.0), "vent": rng.uniform(-1.0, 1.0)}
        after = world.step(action)
        rows.append((action, before, after))

    discovered: set[MechanismAtom] = set()
    for axis in ("heat", "vent"):
        xs = [action[axis] for action, _, _ in rows]
        for target in ("temp", "pressure", "pressure_sensor"):
            ys = [after[target] - before[target] for _, before, after in rows]
            corr = _correlation(xs, ys)
            if abs(corr) <= 0.35:
                continue
            for context in ("low-temp", "high-temp"):
                discovered.add((f"{axis}+", context, target, "+" if corr > 0 else "-"))
                discovered.add((f"{axis}-", context, target, "-" if corr > 0 else "+"))
    return _result("global-linear-regression", seed, discovered, true_atoms)


def _run_metric_core(
    seed: int,
    paired_samples: int,
    true_atoms: set[MechanismAtom],
    noise: float,
) -> ContinuousRunResult:
    rng = Random(30_000 + seed)
    world = ContinuousControlSCM(seed=seed, noise=noise)
    discovered: set[MechanismAtom] = set()
    for context in ("low-temp", "high-temp"):
        for axis in ("heat", "vent"):
            for sign, value in (("+", 0.8), ("-", -0.8)):
                deltas: dict[str, list[float]] = {
                    "temp": [],
                    "pressure": [],
                    "pressure_sensor": [],
                }
                for _ in range(paired_samples):
                    before = _sample_state(rng, context=context, noise=noise)
                    world.set_state(before)
                    action = {"heat": 0.0, "vent": 0.0}
                    action[axis] = value
                    after = world.step(action)
                    for target in deltas:
                        deltas[target].append(after[target] - before[target])
                target, delta = max(
                    (
                        (target, median(values))
                        for target, values in deltas.items()
                        if target not in ContinuousControlSCM.readout_variables
                    ),
                    key=lambda item: abs(item[1]),
                )
                if abs(delta) > 0.18:
                    discovered.add((f"{axis}{sign}", context, target, "+" if delta > 0 else "-"))
    return _result("metric-causal-core", seed, discovered, true_atoms)


def _result(
    agent: str,
    seed: int,
    discovered: set[MechanismAtom],
    true_atoms: set[MechanismAtom],
) -> ContinuousRunResult:
    return ContinuousRunResult(
        agent=agent,
        seed=seed,
        discovered=discovered,
        true_mechanisms=true_atoms,
        score=_score(discovered, true_atoms),
    )


def _score(
    discovered: set[MechanismAtom],
    true_atoms: set[MechanismAtom],
) -> ContinuousScore:
    tp = len(discovered & true_atoms)
    fp = len(discovered - true_atoms)
    fn = len(true_atoms - discovered)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    readout_fp = sum(1 for _, _, target, _ in discovered - true_atoms if target.endswith("_sensor"))
    context_error = sum(
        1
        for action_bin, context, target, sign in discovered - true_atoms
        if any(
            action_bin == a and target == y and sign == s
            for a, _, y, s in true_atoms
        )
        and not target.endswith("_sensor")
    )
    return ContinuousScore(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        readout_false_positive=readout_fp,
        context_error=context_error,
    )


def _summary_row(agent: str, results: list[ContinuousRunResult]) -> dict[str, object]:
    return {
        "agent": agent,
        "precision": mean(result.score.precision for result in results),
        "recall": mean(result.score.recall for result in results),
        "f1": mean(result.score.f1 for result in results),
        "false_positive": mean(result.score.false_positive for result in results),
        "readout_false_positive": mean(
            result.score.readout_false_positive for result in results
        ),
        "context_error": mean(result.score.context_error for result in results),
    }


def _sample_state(rng: Random, context: str, noise: float = 0.02) -> ContinuousState:
    if context == "high-temp":
        temp = rng.uniform(0.35, 0.95)
    else:
        temp = rng.uniform(-0.95, -0.15)
    pressure = rng.uniform(-0.75, 0.75)
    return {
        "temp": temp,
        "pressure": pressure,
        "pressure_sensor": pressure + rng.gauss(0.0, noise),
    }


def _correlation(xs: list[float], ys: list[float]) -> float:
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sqrt(vx * vy)


def _clip(value: float, low: float = -1.5, high: float = 1.5) -> float:
    return max(low, min(high, value))


def _atom_to_dict(atom: MechanismAtom) -> dict[str, str]:
    action_bin, context, target, sign = atom
    return {
        "action_bin": action_bin,
        "context": context,
        "target": target,
        "sign": sign,
    }
