from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .worlds import make_procedural_diagnostic_world_pair


SYSTEM_PROMPT = (
    "You are given controlled intervention records from an unknown dynamical "
    "system. Judge only the stated mechanism probe. A variable that changes "
    "with a physical output may be a synchronized readout rather than a "
    "direct target. Reply yes or no."
)


@dataclass(frozen=True)
class ProbeExample:
    example_id: str
    split: str
    family: int
    condition: str
    probe_type: str
    candidate_kind: str
    mechanism_index: int
    shifted: bool
    label: int
    full_prompt: str
    query_only_prompt: str
    evidence_rule_prediction: int
    gated_state_prediction: int


def _state_text(state: dict[str, bool], variable_aliases: dict[str, str]) -> str:
    true_names = sorted(variable_aliases[name] for name, value in state.items() if value)
    return "{" + ", ".join(true_names) + "}" if true_names else "{}"


def _readout_map(world: Any) -> dict[str, list[str]]:
    readouts = getattr(world, "_readouts", ())
    mapping: dict[str, list[str]] = {}
    for readout in readouts:
        mapping.setdefault(str(readout.source_variable), []).append(str(readout.name))
    return {key: sorted(values) for key, values in mapping.items()}


def _controlled_record(
    world: Any, family: int, condition: str, repeats: int
) -> tuple[str, dict[str, Any]]:
    specs = [dict(spec) for spec in world.temporal_specs()]
    followup = str(specs[0]["followup_action"])
    reset_actions = {str(spec["target_action"]) for spec in specs}
    reset_actions = {
        action.name
        for action in world.actions()
        if action.name not in reset_actions and "reset" in action.description.lower()
    }
    candidate_actions = [
        action.name
        for action in world.actions()
        if action.name != followup and action.name not in reset_actions
    ]

    identifier_seed = int(
        hashlib.sha256(f"{family}|{condition}".encode("utf-8")).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(identifier_seed)
    action_names = sorted(action.name for action in world.actions())
    action_tokens = [f"action_{index:02d}" for index in rng.permutation(len(action_names))]
    action_aliases = dict(zip(action_names, action_tokens))
    variable_names = sorted(world.reset(seed=identifier_seed))
    variable_tokens = [f"state_{index:02d}" for index in rng.permutation(len(variable_names))]
    variable_aliases = dict(zip(variable_names, variable_tokens))

    catalog = ["ACTION CATALOG"]
    for action in world.actions():
        catalog.append(f"- {action_aliases[action.name]}: {action.description}")

    records: dict[str, list[dict[str, dict[str, bool]]]] = {}
    lines = [*catalog, "", "CONTROLLED RECORDS"]
    for action_index, action in enumerate(candidate_actions):
        records[action] = []
        for repeat in range(repeats):
            seed = family * 10_000 + action_index * 101 + repeat
            before = dict(world.reset(seed=seed))
            after_action = dict(world.step(action))
            after_one = dict(world.step(followup))
            after_two = dict(world.step(followup))
            row = {
                "before": before,
                "after_action": after_action,
                "after_one": after_one,
                "after_two": after_two,
            }
            records[action].append(row)
            lines.extend(
                [
                    f"Trial action={action}, repetition={repeat + 1}",
                    f"  before: {_state_text(before, variable_aliases)}",
                    f"  immediately after action: {_state_text(after_action, variable_aliases)}",
                    f"  after one {action_aliases[followup]}: {_state_text(after_one, variable_aliases)}",
                    f"  after two {action_aliases[followup]}: {_state_text(after_two, variable_aliases)}",
                ]
            )
    for action in candidate_actions:
        source = f"Trial action={action},"
        target = f"Trial action={action_aliases[action]},"
        lines = [line.replace(source, target) for line in lines]
    return "\n".join(lines), {
        "records": records,
        "followup": followup,
        "action_aliases": action_aliases,
        "variable_aliases": variable_aliases,
    }


def _first_effect_step(
    evidence: dict[str, Any], action: str, variable: str
) -> int | None:
    steps: list[int] = []
    for row in evidence["records"][action]:
        baseline = bool(row["before"].get(variable, False))
        sequence = [
            row["after_action"],
            row["after_one"],
            row["after_two"],
        ]
        first = None
        for index, state in enumerate(sequence):
            if bool(state.get(variable, False)) != baseline:
                first = index
                break
        if first is not None:
            steps.append(first)
    if not steps:
        return None
    values, counts = np.unique(np.asarray(steps), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _gated_target_prediction(
    evidence: dict[str, Any], action: str, variable: str
) -> int:
    first = _first_effect_step(evidence, action, variable)
    if first not in {1, 2}:
        return 0
    for other_action, rows in evidence["records"].items():
        if other_action == action:
            continue
        for row in rows:
            if any(
                bool(row[phase].get(variable, False))
                for phase in ("before", "after_action", "after_one", "after_two")
            ):
                return 0
    return 1


def _target_query(action: str, variable: str, variant: int = 0) -> str:
    templates = [
        (
            "DIRECT-TARGET PROBE\n"
            "Claim: intervention {action} directly controls {variable}, rather "
            "than {variable} being another output or a synchronized readout.\n"
            "Does the controlled record support this claim?"
        ),
        (
            "DIRECT-TARGET PROBE\n"
            "Claim: {variable} is the physical mechanism target of {action}.\n"
            "Is this claim supported by the intervention record?"
        ),
    ]
    return templates[variant % len(templates)].format(action=action, variable=variable)


def _delay_query(action: str, variable: str, delay: int) -> str:
    return (
        "DELAY PROBE\n"
        f"Claim: after intervention {action}, variable {variable} first changes "
        f"after exactly {delay} resolve step{'s' if delay != 1 else ''}.\n"
        "Does the controlled record support this claim?"
    )


def _make_example(
    *,
    split: str,
    family: int,
    condition: str,
    record: str,
    query: str,
    probe_type: str,
    candidate_kind: str,
    mechanism_index: int,
    shifted: bool,
    label: int,
    evidence_rule_prediction: int,
    gated_state_prediction: int,
) -> ProbeExample:
    example_key = "|".join(
        [
            split,
            str(family),
            condition,
            probe_type,
            candidate_kind,
            str(mechanism_index),
            query,
        ]
    )
    example_id = hashlib.sha256(example_key.encode("utf-8")).hexdigest()[:16]
    return ProbeExample(
        example_id=example_id,
        split=split,
        family=family,
        condition=condition,
        probe_type=probe_type,
        candidate_kind=candidate_kind,
        mechanism_index=mechanism_index,
        shifted=shifted,
        label=label,
        full_prompt=f"{record}\n\n{query}\nAnswer:",
        query_only_prompt=f"{query}\nAnswer:",
        evidence_rule_prediction=evidence_rule_prediction,
        gated_state_prediction=gated_state_prediction,
    )


def build_probe_examples(
    train_families: Sequence[int],
    test_families: Sequence[int],
    repeats: int = 2,
) -> list[ProbeExample]:
    examples: list[ProbeExample] = []
    conditions = (
        ("source-clean", "source", "none"),
        ("renamed-shift", "target", "none"),
        ("opaque-readout", "target", "noisy-opaque"),
        ("semantic-readout", "target", "noisy-semantic-confounder"),
    )
    for split, families in (("train", train_families), ("test", test_families)):
        for family in families:
            for condition, domain, readout_mode in conditions:
                if split == "train" and condition == "semantic-readout":
                    continue
                source, target = make_procedural_diagnostic_world_pair(
                    family_seed=family,
                    mechanism_count=3,
                    readout_mode=readout_mode,
                )
                world = source if domain == "source" else target
                record, evidence = _controlled_record(world, family, condition, repeats)
                specs = [dict(spec) for spec in world.temporal_specs()]
                targets = [str(spec["target_variable"]) for spec in specs]
                readouts = _readout_map(world)
                for index, spec in enumerate(specs):
                    action = str(spec["target_action"])
                    target_variable = str(spec["target_variable"])
                    delay = int(spec["delay_steps"])
                    shifted = domain == "target" and delay != 1
                    other_variable = targets[(index + 1) % len(targets)]
                    display_action = evidence["action_aliases"][action]
                    display_target = evidence["variable_aliases"][target_variable]
                    display_other = evidence["variable_aliases"][other_variable]

                    first_target = _first_effect_step(evidence, action, target_variable)
                    first_other = _first_effect_step(evidence, action, other_variable)
                    gated_target = _gated_target_prediction(
                        evidence, action, target_variable
                    )
                    gated_other = _gated_target_prediction(
                        evidence, action, other_variable
                    )
                    examples.append(
                        _make_example(
                            split=split,
                            family=family,
                            condition=condition,
                            record=record,
                            query=_target_query(display_action, display_target),
                            probe_type="target",
                            candidate_kind="direct",
                            mechanism_index=index,
                            shifted=shifted,
                            label=1,
                            evidence_rule_prediction=int(first_target in {1, 2}),
                            gated_state_prediction=gated_target,
                        )
                    )
                    examples.append(
                        _make_example(
                            split=split,
                            family=family,
                            condition=condition,
                            record=record,
                            query=_target_query(display_action, display_other),
                            probe_type="target",
                            candidate_kind="other-target",
                            mechanism_index=index,
                            shifted=shifted,
                            label=0,
                            evidence_rule_prediction=int(first_other in {1, 2}),
                            gated_state_prediction=gated_other,
                        )
                    )

                    wrong_delay = 2 if delay == 1 else 1
                    examples.append(
                        _make_example(
                            split=split,
                            family=family,
                            condition=condition,
                            record=record,
                            query=_delay_query(display_action, display_target, delay),
                            probe_type="delay",
                            candidate_kind="correct-delay",
                            mechanism_index=index,
                            shifted=shifted,
                            label=1,
                            evidence_rule_prediction=int(first_target == delay),
                            gated_state_prediction=int(
                                gated_target == 1 and first_target == delay
                            ),
                        )
                    )
                    examples.append(
                        _make_example(
                            split=split,
                            family=family,
                            condition=condition,
                            record=record,
                            query=_delay_query(display_action, display_target, wrong_delay),
                            probe_type="delay",
                            candidate_kind="wrong-delay",
                            mechanism_index=index,
                            shifted=shifted,
                            label=0,
                            evidence_rule_prediction=int(first_target == wrong_delay),
                            gated_state_prediction=int(
                                gated_target == 1 and first_target == wrong_delay
                            ),
                        )
                    )

                    linked_readouts = readouts.get(target_variable, [])
                    if linked_readouts:
                        readout = linked_readouts[0]
                        display_readout = evidence["variable_aliases"][readout]
                        first_readout = _first_effect_step(evidence, action, readout)
                        gated_readout = _gated_target_prediction(evidence, action, readout)
                        examples.append(
                            _make_example(
                                split=split,
                                family=family,
                                condition=condition,
                                record=record,
                                query=_target_query(display_action, display_target, variant=1),
                                probe_type="target",
                                candidate_kind="direct-repeat",
                                mechanism_index=index,
                                shifted=shifted,
                                label=1,
                                evidence_rule_prediction=int(first_target in {1, 2}),
                                gated_state_prediction=gated_target,
                            )
                        )
                        examples.append(
                            _make_example(
                                split=split,
                                family=family,
                                condition=condition,
                                record=record,
                                query=_target_query(display_action, display_readout, variant=1),
                                probe_type="target",
                                candidate_kind="readout",
                                mechanism_index=index,
                                shifted=shifted,
                                label=0,
                                evidence_rule_prediction=int(first_readout in {1, 2}),
                                gated_state_prediction=gated_readout,
                            )
                        )
    return examples


def _resolve_model_path(model_name: str) -> Path:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate.resolve()
    cache_name = "models--" + model_name.replace("/", "--")
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    available = sorted(path for path in snapshots.glob("*") if path.is_dir())
    if not available:
        raise FileNotFoundError(
            f"No local snapshot for {model_name}; provide a local --model path."
        )
    return available[-1].resolve()


def _answer_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected one token for {text!r}, found {token_ids}")
    return int(token_ids[0])


def extract_hidden_features(
    examples: Sequence[ProbeExample],
    model_name: str,
    layers: Sequence[int],
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = _resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    layer_count = int(model.config.num_hidden_layers)
    normalized_layers = [layer if layer >= 0 else layer_count + 1 + layer for layer in layers]
    if any(layer < 0 or layer > layer_count for layer in normalized_layers):
        raise ValueError(f"layers must be between 0 and {layer_count}")
    yes_id = _answer_token_id(tokenizer, "Yes")
    no_id = _answer_token_id(tokenizer, "No")

    features: dict[str, list[np.ndarray]] = {}
    for prefix in ("full", "query"):
        for layer in normalized_layers:
            features[f"{prefix}_layer_{layer}"] = []
    logit_margins: list[np.ndarray] = []
    prompt_lengths: list[int] = []

    for prefix, prompts in (
        ("full", [example.full_prompt for example in examples]),
        ("query", [example.query_only_prompt for example in examples]),
    ):
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch_prompts
            ]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.inference_mode():
                output = model(**encoded, output_hidden_states=True, use_cache=False)
            for layer in normalized_layers:
                values = output.hidden_states[layer][:, -1, :].float().cpu().numpy()
                features[f"{prefix}_layer_{layer}"].append(values)
            if prefix == "full":
                logits = output.logits[:, -1, :].float()
                margin = (logits[:, yes_id] - logits[:, no_id]).cpu().numpy()
                logit_margins.append(margin)
                prompt_lengths.extend(encoded["attention_mask"].sum(dim=1).cpu().tolist())
            del output, encoded
            torch.cuda.empty_cache()

    arrays = {
        key: np.concatenate(chunks, axis=0).astype(np.float32)
        for key, chunks in features.items()
    }
    arrays["logit_margin"] = np.concatenate(logit_margins).astype(np.float32)
    metadata = {
        "model": model_name,
        "resolved_model_path": str(model_path),
        "snapshot_revision": model_path.name,
        "layers": normalized_layers,
        "hidden_size": int(model.config.hidden_size),
        "max_prompt_tokens": max(prompt_lengths),
        "mean_prompt_tokens": float(np.mean(prompt_lengths)),
        "truncated_prompts": int(sum(length >= max_length for length in prompt_lengths)),
    }
    del model
    torch.cuda.empty_cache()
    return arrays, metadata


def _group_masks(examples: Sequence[ProbeExample]) -> dict[str, np.ndarray]:
    split = np.asarray([example.split for example in examples])
    condition = np.asarray([example.condition for example in examples])
    shifted = np.asarray([example.shifted for example in examples], dtype=bool)
    probe_type = np.asarray([example.probe_type for example in examples])
    candidate_kind = np.asarray([example.candidate_kind for example in examples])
    test = split == "test"
    return {
        "source-clean": test & (condition == "source-clean"),
        "renamed-shift": test & (condition == "renamed-shift"),
        "shifted-delay": test
        & (condition == "renamed-shift")
        & shifted
        & (probe_type == "delay"),
        "opaque-readout": test & (condition == "opaque-readout"),
        "semantic-readout": test & (condition == "semantic-readout"),
        "opaque-readout-candidates": test
        & (condition == "opaque-readout")
        & (candidate_kind == "readout"),
        "semantic-readout-candidates": test
        & (condition == "semantic-readout")
        & (candidate_kind == "readout"),
    }


def _metric_row(
    method: str,
    group: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    families: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    y = labels[mask]
    prediction = predictions[mask]
    score = scores[mask]
    row: dict[str, Any] = {
        "method": method,
        "group": group,
        "n": int(mask.sum()),
        "positive_rate": float(y.mean()) if len(y) else None,
        "accuracy": float(np.mean(prediction == y)) if len(y) else None,
        "false_positive_rate": float(prediction[y == 0].mean()) if np.any(y == 0) else None,
    }
    if len(np.unique(y)) == 2:
        row.update(
            {
                "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
                "f1": float(f1_score(y, prediction)),
                "auroc": float(roc_auc_score(y, score)),
            }
        )
    else:
        row.update({"balanced_accuracy": None, "f1": None, "auroc": None})
    family_values: list[float] = []
    for family in np.unique(families[mask]):
        family_mask = mask & (families == family)
        family_y = labels[family_mask]
        family_prediction = predictions[family_mask]
        if len(np.unique(family_y)) == 2:
            family_values.append(
                float(balanced_accuracy_score(family_y, family_prediction))
            )
        elif np.all(family_y == 0):
            family_values.append(float(family_prediction.mean()))
        else:
            family_values.append(float(np.mean(family_prediction == family_y)))
    row["family_mean"] = float(np.mean(family_values))
    row["family_se"] = (
        float(np.std(family_values, ddof=1) / np.sqrt(len(family_values)))
        if len(family_values) > 1
        else 0.0
    )
    return row


def fit_and_evaluate(
    examples: Sequence[ProbeExample],
    features: dict[str, np.ndarray],
    layers: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    families = np.asarray([example.family for example in examples], dtype=np.int64)
    train = np.asarray([example.split == "train" for example in examples])
    masks = _group_masks(examples)
    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for layer in layers:
        values = features[f"full_layer_{layer}"]
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
        )
        classifier.fit(values[train], labels[train])
        scores = classifier.predict_proba(values)[:, 1]
        methods[f"Qwen linear layer {layer}"] = ((scores >= 0.5).astype(int), scores)

    query_values = features[f"query_layer_{layers[-1]}"]
    query_classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
    )
    query_classifier.fit(query_values[train], labels[train])
    query_scores = query_classifier.predict_proba(query_values)[:, 1]
    methods["Qwen query-only linear"] = ((query_scores >= 0.5).astype(int), query_scores)

    last_values = features[f"full_layer_{layers[-1]}"]
    nonlinear = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 32),
            alpha=1e-3,
            batch_size=64,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            max_iter=300,
            random_state=0,
        ),
    )
    nonlinear.fit(last_values[train], labels[train])
    nonlinear_scores = nonlinear.predict_proba(last_values)[:, 1]
    methods["Qwen nonlinear two-layer"] = (
        (nonlinear_scores >= 0.5).astype(int),
        nonlinear_scores,
    )

    rbf = make_pipeline(
        StandardScaler(),
        SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced"),
    )
    rbf.fit(last_values[train], labels[train])
    rbf_scores = rbf.decision_function(last_values)
    methods["Qwen nonlinear RBF"] = ((rbf_scores >= 0.0).astype(int), rbf_scores)

    for name, prompts in (
        ("TF-IDF full transcript", [example.full_prompt for example in examples]),
        ("TF-IDF query only", [example.query_only_prompt for example in examples]),
    ):
        tfidf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
        )
        train_prompts = [prompt for prompt, keep in zip(prompts, train) if keep]
        tfidf.fit(train_prompts, labels[train])
        scores = tfidf.predict_proba(prompts)[:, 1]
        methods[name] = ((scores >= 0.5).astype(int), scores)

    margin = features["logit_margin"].reshape(-1, 1)
    calibration = LogisticRegression(class_weight="balanced", random_state=0)
    calibration.fit(margin[train], labels[train])
    logit_scores = calibration.predict_proba(margin)[:, 1]
    methods["Qwen calibrated Yes/No logits"] = (
        (logit_scores >= 0.5).astype(int),
        logit_scores,
    )

    evidence_predictions = np.asarray(
        [example.evidence_rule_prediction for example in examples], dtype=np.int64
    )
    methods["paired-evidence rule"] = (
        evidence_predictions,
        evidence_predictions.astype(np.float64),
    )
    gated_predictions = np.asarray(
        [example.gated_state_prediction for example in examples], dtype=np.int64
    )
    methods["gated mechanism state"] = (
        gated_predictions,
        gated_predictions.astype(np.float64),
    )

    rows: list[dict[str, Any]] = []
    for method, (predictions, scores) in methods.items():
        for group, mask in masks.items():
            rows.append(
                _metric_row(
                    method,
                    group,
                    labels,
                    predictions,
                    scores,
                    mask,
                    families,
                )
            )

    training = {
        "train_examples": int(train.sum()),
        "test_examples": int((~train).sum()),
        "train_positive_rate": float(labels[train].mean()),
        "nonlinear_iterations": int(nonlinear[-1].n_iter_),
    }
    return rows, training


def make_figure(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = {(row["method"], row["group"]): row for row in rows}
    methods = [
        "TF-IDF query only",
        "Qwen calibrated Yes/No logits",
        "Qwen linear layer 28",
        "Qwen nonlinear RBF",
        "paired-evidence rule",
        "gated mechanism state",
    ]
    labels = [
        "query only",
        "Qwen logits",
        "linear hidden",
        "nonlinear hidden",
        "paired evidence",
        "gated state",
    ]
    colors = ["#A7B0B9", "#E76F51", "#3F6FB6", "#20A37D", "#64748B", "#172033"]
    groups = ["source-clean", "renamed-shift", "semantic-readout"]
    group_labels = ["source", "renamed shift", "unseen readout"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.5, 1.0]})
    x = np.arange(len(groups))
    width = 0.13
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = [lookup[(method, group)]["balanced_accuracy"] for group in groups]
        errors = [lookup[(method, group)]["family_se"] for group in groups]
        axes[0].bar(
            x + (index - (len(methods) - 1) / 2) * width,
            values,
            width,
            label=label,
            color=color,
            yerr=errors,
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
    axes[0].axhline(0.5, color="#64748B", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(x, group_labels)
    axes[0].set_ylim(0.45, 1.02)
    axes[0].set_ylabel("balanced accuracy")
    axes[0].set_title("Frozen mechanism probes", fontsize=8.8)

    shifted = [lookup[(method, "shifted-delay")]["balanced_accuracy"] for method in methods]
    readout_fpr = [
        lookup[(method, "semantic-readout-candidates")]["false_positive_rate"]
        for method in methods
    ]
    shifted_se = [lookup[(method, "shifted-delay")]["family_se"] for method in methods]
    readout_se = [
        lookup[(method, "semantic-readout-candidates")]["family_se"]
        for method in methods
    ]
    y = np.arange(len(methods))[::-1]
    axes[1].barh(
        y + 0.16,
        shifted,
        height=0.28,
        color="#20A37D",
        label="shifted delay BA",
        xerr=shifted_se,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    axes[1].barh(
        y - 0.16,
        readout_fpr,
        height=0.28,
        color="#D9474F",
        label="readout FPR",
        xerr=readout_se,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    axes[1].set_yticks(y, labels)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_xlabel("rate")
    axes[1].set_title("Mechanism shift and readout", fontsize=8.8)
    axes[1].legend(
        frameon=False,
        fontsize=6.6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(width=0.8, labelsize=7.1)
    axes[0].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.30, wspace=0.35)
    fig.savefig(output_dir / "llm_hidden_retention.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "llm_hidden_retention.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    train_families = list(range(args.train_family_start, args.train_family_start + args.train_families))
    test_families = list(range(args.test_family_start, args.test_family_start + args.test_families))
    examples = build_probe_examples(train_families, test_families, repeats=args.repeats)
    fingerprint = hashlib.sha256(
        "|".join(example.example_id for example in examples).encode("utf-8")
    ).hexdigest()
    cache_path = Path(args.feature_cache) if args.feature_cache else None
    cache_metadata_path = (
        cache_path.with_suffix(cache_path.suffix + ".json") if cache_path else None
    )
    if cache_path and cache_path.exists() and cache_metadata_path and cache_metadata_path.exists():
        cached_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if cached_metadata.get("dataset_fingerprint") != fingerprint:
            raise ValueError("Feature cache does not match the requested probe dataset")
        with np.load(cache_path) as cached:
            features = {key: cached[key].astype(np.float32) for key in cached.files}
        model_metadata = dict(cached_metadata["model_metadata"])
    else:
        features, model_metadata = extract_hidden_features(
            examples,
            model_name=args.model,
            layers=args.layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        if cache_path and cache_metadata_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                **{
                    key: value.astype(np.float32 if key == "logit_margin" else np.float16)
                    for key, value in features.items()
                },
            )
            cache_metadata_path.write_text(
                json.dumps(
                    {
                        "dataset_fingerprint": fingerprint,
                        "model_metadata": model_metadata,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    rows, training = fit_and_evaluate(examples, features, model_metadata["layers"])
    payload = {
        "experiment": "llm-hidden-state-causal-retention",
        "model": model_metadata,
        "train_families": train_families,
        "test_families": test_families,
        "repeats": args.repeats,
        "training": training,
        "rows": rows,
        "examples": [
            {
                key: value
                for key, value in asdict(example).items()
                if key not in {"full_prompt", "query_only_prompt"}
            }
            for example in examples
        ],
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_figure(rows, Path(args.figure_dir))
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    rows = {
        (row["method"], row["group"]): row
        for row in payload["rows"]
    }
    methods = [
        "TF-IDF query only",
        "Qwen calibrated Yes/No logits",
        "Qwen linear layer 28",
        "Qwen nonlinear RBF",
        "paired-evidence rule",
        "gated mechanism state",
    ]
    print("LLM hidden-state causal-retention probes")
    for method in methods:
        source = rows[(method, "source-clean")]["balanced_accuracy"]
        renamed = rows[(method, "renamed-shift")]["balanced_accuracy"]
        shifted = rows[(method, "shifted-delay")]["balanced_accuracy"]
        readout = rows[(method, "semantic-readout-candidates")]["false_positive_rate"]
        print(
            f"{method:34s} source={source:.3f} renamed={renamed:.3f} "
            f"shifted-delay={shifted:.3f} readout-FPR={readout:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-families", type=int, default=12)
    parser.add_argument("--train-family-start", type=int, default=1)
    parser.add_argument("--test-families", type=int, default=6)
    parser.add_argument("--test-family-start", type=int, default=101)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 7, 14, 21, 28])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--feature-cache", default="")
    parser.add_argument("--output-json", default="tmp/llm_hidden_retention.json")
    parser.add_argument("--figure-dir", default="tmp/llm_hidden_retention_figures")
    return parser


def main() -> None:
    payload = run_experiment(build_parser().parse_args())
    _print_summary(payload)


if __name__ == "__main__":
    main()
