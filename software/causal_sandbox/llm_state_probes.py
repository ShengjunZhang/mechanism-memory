"""Qwen decoder reanalysis with family-disjoint target calibration.

The legacy cache binds an ordered example-ID digest, not prompt-text digests.
Fresh extraction records exact prompt and row identities; model weights stay frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np

from .llm_hidden_retention import (
    ProbeExample,
    SYSTEM_PROMPT,
    _group_masks,
    build_probe_examples,
)


TRAIN_FAMILIES = tuple(range(1, 13))
TEST_FAMILIES = tuple(range(101, 107))
TARGET_CONDITIONS = ("renamed-shift", "opaque-readout")
TRAIN_CONDITIONS = ("source-clean", *TARGET_CONDITIONS)
BUDGETS = (1, 3, 6, 12)
LAYER = 28
SEED = 20261112
DEFAULT_ROOT = Path("jmlr_revision_20261112")
SNAPSHOT = "a09a35458c702b33eeacc393d103063234e8bc28"
REQUIRED_ROW_FIELDS = {
    "example_id", "split", "family", "condition", "probe_type",
    "candidate_kind", "mechanism_index", "shifted", "label",
    "evidence_rule_prediction",
}
METRICS = ("accuracy", "balanced_accuracy", "false_positive_rate")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_id_fingerprint(examples: Sequence[ProbeExample]) -> str:
    return hashlib.sha256(
        "|".join(example.example_id for example in examples).encode("utf-8")
    ).hexdigest()


def prompt_fingerprints(example: ProbeExample) -> dict[str, str]:
    return {
        f"{field}_sha256": hashlib.sha256(
            getattr(example, field).encode("utf-8")
        ).hexdigest()
        for field in ("full_prompt", "query_only_prompt")
    }


def verify_cache_metadata(
    examples: Sequence[ProbeExample],
    cache_metadata: dict[str, Any],
    saved: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on any available row, order, split, or model mismatch."""
    fingerprint = ordered_id_fingerprint(examples)
    if cache_metadata.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Cache ordered example-ID fingerprint mismatch")
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate regenerated example IDs")
    if saved.get("train_families") != list(TRAIN_FAMILIES):
        raise ValueError("Saved train families differ from the preset split")
    if saved.get("test_families") != list(TEST_FAMILIES):
        raise ValueError("Saved test families differ from the preset split")
    if saved.get("repeats") != 2:
        raise ValueError("Saved record repeat count differs from the preset")
    model = cache_metadata.get("model_metadata", {})
    if saved.get("model") != model:
        raise ValueError("Cache and example-manifest model metadata mismatch")
    if model.get("model") != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("Unexpected model in Qwen cache")
    if LAYER not in model.get("layers", []) or not model.get("snapshot_revision"):
        raise ValueError("Cache lacks preset layer or model snapshot revision")
    rows = saved.get("examples", [])
    if len(rows) != len(examples):
        raise ValueError("Cache example metadata row count mismatch")
    checked_fields: set[str] = set()
    missing_fields: set[str] = set()
    prompt_rows = 0
    for index, (example, row) in enumerate(zip(examples, rows)):
        if not REQUIRED_ROW_FIELDS.issubset(row):
            raise ValueError(f"Missing required metadata in cached row {index}")
        expected = asdict(example)
        expected.update(prompt_fingerprints(example))
        for key, value in row.items():
            if key not in expected or expected[key] != value:
                raise ValueError(f"Cache metadata mismatch at row {index}: {key}")
            checked_fields.add(key)
        missing_fields.update(set(asdict(example)) - set(row))
        prompt_rows += int(all(
            field in row or f"{field}_sha256" in row
            for field in ("full_prompt", "query_only_prompt")
        ))
    if prompt_rows not in (0, len(examples)):
        raise ValueError("Partial cached prompt fingerprint coverage")
    has_system = "system_prompt_sha256" in cache_metadata
    system_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if has_system and cache_metadata["system_prompt_sha256"] != system_hash:
        raise ValueError("Cached system prompt fingerprint mismatch")
    return {
        "ordered_example_id_fingerprint": fingerprint,
        "matched_rows": len(examples),
        "matched_metadata_fields": sorted(checked_fields),
        "regenerated_fields_unavailable_in_saved_rows": sorted(missing_fields),
        "saved_user_prompt_fingerprint_rows": prompt_rows,
        "saved_system_prompt_fingerprint_available": has_system,
        "historical_prompt_text_match_verified": prompt_rows == len(examples) and has_system,
        "verification_scope": (
            "Exact ordered example IDs, every available saved row field, family split, "
            "repeat count, model metadata, feature shapes and finiteness. The legacy "
            "cache has no historical prompt-text/system-template hashes or per-vector "
            "row IDs. Current prompt hashes are recorded for future audits, not proof "
            "of historical extraction prompts or independent vector-row binding."
        ),
    }


def load_verified_cache(
    examples: Sequence[ProbeExample], cache_path: Path, examples_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    sidecar = cache_path.with_suffix(cache_path.suffix + ".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    saved = json.loads(examples_path.read_text(encoding="utf-8"))
    audit = verify_cache_metadata(examples, metadata, saved)
    hidden_size = int(metadata["model_metadata"]["hidden_size"])
    expected_shapes = {
        f"{prefix}_layer_{layer}": (len(examples), hidden_size)
        for prefix in ("full", "query")
        for layer in metadata["model_metadata"]["layers"]
    }
    expected_shapes["logit_margin"] = (len(examples),)
    features = {}
    with np.load(cache_path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_shapes):
            raise ValueError("Feature cache keys differ from declared model layers")
        for key, shape in expected_shapes.items():
            values = archive[key]
            if values.shape != shape or values.dtype.kind != "f":
                raise ValueError(f"Invalid feature shape or dtype for {key}")
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite cached features in {key}")
            if key in (f"full_layer_{LAYER}", f"query_layer_{LAYER}", "logit_margin"):
                features[key] = values.astype(np.float32)
    audit["feature_shapes"] = {key: list(shape) for key, shape in expected_shapes.items()}
    audit["files"] = {
        role: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for role, path in (
            ("features", cache_path), ("cache_metadata", sidecar),
            ("saved_examples", examples_path),
        )
    }
    return features, metadata["model_metadata"], audit


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def local_snapshot() -> Path:
    path = (Path.home() / ".cache" / "huggingface" / "hub"
            / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / SNAPSHOT)
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(
            f"Pinned local snapshot missing at {path}. No download attempted. "
            "Expected four weight shards, about 14.19 GiB."
        )
    shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    required = [*shards, "config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing local Qwen files: {missing}; no download attempted. "
            "Full weights require about 14.19 GiB."
        )
    return path


def prepare_prompt_records(
    examples: Sequence[ProbeExample], tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    records = []
    rendered: dict[str, list[str]] = {"full": [], "query": []}
    for index, example in enumerate(examples):
        row = {
            "row_index": index,
            **{k: v for k, v in asdict(example).items() if "prompt" not in k},
            **prompt_fingerprints(example),
        }
        for prefix, prompt in (("full", example.full_prompt), ("query", example.query_only_prompt)):
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            token_ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            rendered[prefix].append(text)
            row[f"{prefix}_rendered_chat_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            row[f"{prefix}_token_ids_sha256"] = canonical_hash(token_ids)
            row[f"{prefix}_token_count"] = len(token_ids)
        row["row_fingerprint"] = canonical_hash(row)
        records.append(row)
    return records, rendered


def validate_fresh_records(
    examples: Sequence[ProbeExample], records: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    if metadata.get("schema") != "qwen-exact-prompt-v1":
        raise ValueError("Unsupported fresh feature record schema")
    if metadata.get("snapshot_revision") != SNAPSHOT or metadata.get("layer") != LAYER:
        raise ValueError("Fresh snapshot/layer differs from preset")
    if metadata.get("system_prompt_sha256") != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest():
        raise ValueError("Fresh system prompt mismatch")
    if metadata.get("ordered_example_id_fingerprint") != ordered_id_fingerprint(examples):
        raise ValueError("Fresh ordered row ID mismatch")
    if metadata.get("ordered_row_fingerprints_sha256") != canonical_hash([r["row_fingerprint"] for r in records]):
        raise ValueError("Fresh ordered prompt/row fingerprint mismatch")
    if metadata.get("rows") != list(records):
        raise ValueError("Fresh exact prompt/row metadata mismatch")
    if metadata.get("max_length") != 1536 or metadata.get("truncated_prompts") != 0:
        raise ValueError("Fresh extraction must use max_length=1536 without truncation")


def load_fresh_vectors(
    path: Path, examples: Sequence[ProbeExample], records: Sequence[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = json.loads(path.with_suffix(".npz.json").read_text(encoding="utf-8"))
    validate_fresh_records(examples, records, metadata)
    if sha256_file(path) != metadata["feature_file_sha256"]:
        raise ValueError("Fresh feature file checksum mismatch")
    bindings = {
        "row_indices": np.arange(len(examples)),
        "example_ids": np.array([x.example_id for x in examples]),
        "row_fingerprints": np.array([x["row_fingerprint"] for x in records]),
    }
    features = {}
    with np.load(path, allow_pickle=False) as archive:
        required = {f"full_layer_{LAYER}", f"query_layer_{LAYER}", "logit_margin"}
        if set(archive.files) != required | set(bindings):
            raise ValueError("Fresh feature keys mismatch")
        for key, expected in bindings.items():
            if not np.array_equal(archive[key], expected):
                raise ValueError(f"Fresh vector row binding mismatch: {key}")
        for key in required:
            values = archive[key]
            expected_shape = (len(examples),) if key == "logit_margin" else (len(examples), 3584)
            if values.shape != expected_shape or values.dtype.kind != "f" or not np.isfinite(values).all():
                raise ValueError(f"Invalid fresh vectors: {key}")
            if hashlib.sha256(values.tobytes()).hexdigest() != metadata["array_sha256"][key]:
                raise ValueError(f"Fresh array checksum mismatch: {key}")
            features[key] = values.astype(np.float32)
    return features, metadata


def fresh_features(
    examples: Sequence[ProbeExample], args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    from transformers import AutoTokenizer

    snapshot = local_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records, rendered = prepare_prompt_records(examples, tokenizer)
    path = Path(args.output_root) / "experiments" / "llm_fresh_features.npz"
    if not path.exists() or args.regenerate_fresh:
        import gc
        import importlib.metadata
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA GPU {args.gpu} is unavailable")
        if args.batch_size not in (1, 2):
            raise ValueError("Fresh extraction batch size must be 1 or 2")
        if max(r[f"{prefix}_token_count"] for r in records for prefix in ("full", "query")) > 1536:
            raise ValueError("Current prompt exceeds 1536 tokens; refusing silent truncation")
        device = torch.device(f"cuda:{args.gpu}")
        free, total = torch.cuda.mem_get_info(device)
        if free < 18 * 1024 ** 3:
            raise RuntimeError(f"GPU {args.gpu} has only {free / 1024 ** 3:.2f} GiB free; require 18 GiB")
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        print(f"Loading pinned local Qwen snapshot on {device}; fp16, batch={args.batch_size}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.float16,
            device_map={"": str(device)}, attn_implementation="sdpa",
        )
        model.eval()
        if model.config.num_hidden_layers != LAYER or model.config.hidden_size != 3584:
            raise ValueError("Unexpected Qwen architecture")
        yes = tokenizer.encode("Yes", add_special_tokens=False)
        no = tokenizer.encode("No", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError("Yes/No answer tokenization changed")
        arrays = {
            f"{prefix}_layer_{LAYER}": np.empty((len(examples), 3584), dtype=np.float16)
            for prefix in ("full", "query")
        }
        arrays["logit_margin"] = np.empty(len(examples), dtype=np.float32)
        for prefix in ("full", "query"):
            for start in range(0, len(examples), args.batch_size):
                stop = min(start + args.batch_size, len(examples))
                encoded = tokenizer(
                    rendered[prefix][start:stop], return_tensors="pt", padding=True,
                    truncation=True, max_length=1536,
                )
                for offset in range(stop - start):
                    unpadded = encoded["input_ids"][offset][encoded["attention_mask"][offset].bool()].tolist()
                    row = records[start + offset]
                    if canonical_hash(unpadded) != row[f"{prefix}_token_ids_sha256"]:
                        raise ValueError("Actual model input token IDs differ from recorded prompt")
                encoded = encoded.to(device)
                with torch.inference_mode():
                    output = model(
                        **encoded, output_hidden_states=True, use_cache=False, logits_to_keep=1,
                    )
                arrays[f"{prefix}_layer_{LAYER}"][start:stop] = output.hidden_states[LAYER][:, -1, :].cpu().numpy()
                if prefix == "full":
                    logits = output.logits[:, -1, :].float()
                    arrays["logit_margin"][start:stop] = (logits[:, yes[0]] - logits[:, no[0]]).cpu().numpy()
                    del logits
                del output, encoded
                if stop % 32 == 0 or stop == len(examples):
                    print(f"{prefix}: {stop}/{len(examples)} rows; elapsed={time.perf_counter() - started:.1f}s; GPU peak={torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GiB", flush=True)
        metadata = {
            "schema": "qwen-exact-prompt-v1", "snapshot_revision": SNAPSHOT, "layer": LAYER,
            "extractor_path": str(Path(__file__).resolve()),
            "extractor_sha256": sha256_file(Path(__file__)),
            "legacy_prompt_generator_sha256": sha256_file(Path(__file__).with_name("llm_hidden_retention.py")),
            "model": "Qwen/Qwen2.5-7B-Instruct", "resolved_model_path": str(snapshot),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "chat_template_sha256": canonical_hash(tokenizer.chat_template),
            "ordered_example_id_fingerprint": ordered_id_fingerprint(examples),
            "ordered_row_fingerprints_sha256": canonical_hash([r["row_fingerprint"] for r in records]),
            "rows": records, "max_length": 1536, "truncated_prompts": 0,
            "max_prompt_tokens": max(r["full_token_count"] for r in records),
            "mean_prompt_tokens": float(np.mean([r["full_token_count"] for r in records])),
            "gpu": args.gpu, "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_gib": total / 1024 ** 3, "gpu_free_before_load_gib": free / 1024 ** 3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3,
            "batch_size": args.batch_size, "dtype": "float16", "padding_side": "left",
            "attn_implementation": "sdpa", "use_cache": False, "logits_to_keep": 1,
            "extraction_seconds": time.perf_counter() - started,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "accelerate")},
            "local_model_files": [
                {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in sorted(snapshot.iterdir()) if p.is_file()
            ],
            "array_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()},
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, **arrays, row_indices=np.arange(len(examples)),
            example_ids=np.array([x.example_id for x in examples]),
            row_fingerprints=np.array([r["row_fingerprint"] for r in records]),
        )
        metadata["feature_file_sha256"] = sha256_file(path)
        path.with_suffix(".npz.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    features, metadata = load_fresh_vectors(path, examples, records)
    if metadata["chat_template_sha256"] != canonical_hash(tokenizer.chat_template):
        raise ValueError("Fresh extraction chat template differs from local tokenizer")
    model_metadata = {
        "model": metadata["model"], "snapshot_revision": SNAPSHOT, "layers": [LAYER],
        "hidden_size": 3584, "max_prompt_tokens": metadata["max_prompt_tokens"],
        "mean_prompt_tokens": metadata["mean_prompt_tokens"], "truncated_prompts": 0,
    }
    record = {
        "matched_rows": len(examples), "fresh_prompt_text_match_verified": True,
        "historical_prompt_text_match_verified": False,
        "prompt_identity": "Exact user/system/rendered-chat/token-ID fingerprints recorded during this extraction and rechecked; array/file checksums and embedded ordered row bindings verified.",
        "files": {"features": {"path": str(path.resolve()), "sha256": sha256_file(path)}},
        "extraction_record_path": str(path.with_suffix(".npz.json").resolve()),
        "ordered_row_fingerprints_sha256": metadata["ordered_row_fingerprints_sha256"],
        "extraction": {k: v for k, v in metadata.items() if k not in ("rows", "local_model_files", "array_sha256")},
    }
    return features, model_metadata, record


def validate_dataset(examples: Sequence[ProbeExample]) -> None:
    train = [example for example in examples if example.split == "train"]
    test = [example for example in examples if example.split == "test"]
    if len(train) + len(test) != len(examples):
        raise ValueError("Unknown split")
    train_families = {example.family for example in train}
    test_families = {example.family for example in test}
    if not train_families.isdisjoint(test_families):
        raise ValueError("Train/test family overlap")
    if train_families != set(TRAIN_FAMILIES) or test_families != set(TEST_FAMILIES):
        raise ValueError("Dataset does not match preset 12-train/6-test families")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("Duplicate example IDs")
    expected_test_conditions = (*TRAIN_CONDITIONS, "semantic-readout")
    for split, families, conditions in (
        ("train", TRAIN_FAMILIES, TRAIN_CONDITIONS),
        ("test", TEST_FAMILIES, expected_test_conditions),
    ):
        selected = [example for example in examples if example.split == split]
        if {example.condition for example in selected} != set(conditions):
            raise ValueError(f"Unexpected conditions in {split}")
        for family in families:
            for condition in conditions:
                rows = [x for x in selected if x.family == family and x.condition == condition]
                n = 18 if "readout" in condition else 12
                if len(rows) != n or Counter(x.label for x in rows) != {0: n // 2, 1: n // 2}:
                    raise ValueError(f"Incomplete or unbalanced cell: {split}/{family}/{condition}")


def training_masks(examples: Sequence[ProbeExample]) -> dict[str, np.ndarray]:
    validate_dataset(examples)
    train = np.array([x.split == "train" for x in examples])
    conditions = np.array([x.condition for x in examples])
    families = np.array([x.family for x in examples])
    masks = {
        "source_only": train & (conditions == "source-clean"),
        "mixed_condition": train.copy(),
    }
    for budget in BUDGETS:
        masks[f"target_refit_{budget:02d}"] = (
            train & np.isin(conditions, TARGET_CONDITIONS)
            & np.isin(families, TRAIN_FAMILIES[:budget])
        )
    return masks


def count_rows(examples: Sequence[ProbeExample], mask: np.ndarray) -> dict[str, Any]:
    selected = [x for x, keep in zip(examples, mask) if keep]

    def counts(rows: Sequence[ProbeExample]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "label_counts": {str(label): sum(x.label == label for x in rows) for label in (0, 1)},
        }

    return {
        **counts(selected),
        "families": sorted({x.family for x in selected}),
        "conditions": {
            condition: counts([x for x in selected if x.condition == condition])
            for condition in sorted({x.condition for x in selected})
        },
        "family_condition_counts": [
            {"family": family, "condition": condition, **counts(rows)}
            for family in sorted({x.family for x in selected})
            for condition in sorted({x.condition for x in selected})
            if (rows := [x for x in selected if x.family == family and x.condition == condition])
        ],
    }


def make_decoder(kind: str) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if kind == "linear":
        classifier = LogisticRegression(
            C=1.0, solver="lbfgs", tol=1e-4, max_iter=2000,
            class_weight="balanced", random_state=0,
        )
    elif kind == "rbf":
        classifier = SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced")
    else:
        raise ValueError(f"Unknown decoder: {kind}")
    return make_pipeline(StandardScaler(), classifier)


def fit_decoder(
    values: np.ndarray, labels: np.ndarray, train_mask: np.ndarray,
    evaluation_mask: np.ndarray, kind: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if np.any(train_mask & evaluation_mask):
        raise ValueError("Fit/evaluation row overlap")
    if set(np.unique(labels[train_mask])) != {0, 1}:
        raise ValueError("Training must contain both labels")
    decoder = make_decoder(kind)
    # The scaler and classifier see only the selected training rows.
    decoder.fit(values[train_mask], labels[train_mask])
    scores = decoder.decision_function(values[evaluation_mask])
    predictions = decoder.predict(values[evaluation_mask]).astype(int)
    return predictions, scores, {
        "scaler_fit_rows": int(train_mask.sum()),
        "scaler_mean_sha256": hashlib.sha256(decoder[0].mean_.tobytes()).hexdigest(),
        "classifier_parameters": decoder[-1].get_params(),
        "iterations": np.asarray(decoder[-1].n_iter_).tolist(),
        "threshold": 0.0,
    }


def cluster_bootstrap_weights(
    family_count: int, replicates: int, seed: int,
) -> np.ndarray:
    if family_count < 1 or replicates < 1:
        raise ValueError("Bootstrap requires families and positive replicate count")
    draws = np.random.default_rng(seed).integers(
        0, family_count, size=(replicates, family_count),
    )
    return np.stack([(draws == index).sum(axis=1) for index in range(family_count)], axis=1)


def metrics_from_confusion(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Counts have trailing dimension (TN, FP, FN, TP)."""
    tn, fp, fn, tp = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0,
        )

    return {
        "accuracy": ratio(tn + tp, tn + fp + fn + tp),
        "balanced_accuracy": (ratio(tn, tn + fp) + ratio(tp, tp + fn)) / 2,
        "false_positive_rate": ratio(fp, tn + fp),
    }


def nullable(value: Any) -> float | None:
    return float(value) if np.isfinite(value) else None


def interval(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None


def evaluate_predictions(
    examples: Sequence[ProbeExample], predictions: np.ndarray,
    group_masks: dict[str, np.ndarray], weights: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    families = np.array([x.family for x in examples])
    labels = np.array([x.label for x in examples])
    if predictions.shape != labels.shape or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Predictions must be one binary value per heldout row")
    if set(families) != set(TEST_FAMILIES) or any(x.split != "test" for x in examples):
        raise ValueError("Evaluation requires exactly the six heldout test families")
    if weights.ndim != 2 or weights.shape[1] != len(TEST_FAMILIES):
        raise ValueError("Bootstrap family columns mismatch")
    rows = []
    draws = {}
    for group, mask in group_masks.items():
        confusion = np.zeros((len(TEST_FAMILIES), 4), dtype=int)
        family_rows = []
        for index, family in enumerate(TEST_FAMILIES):
            keep = mask & (families == family)
            codes = 2 * labels[keep] + predictions[keep]
            confusion[index] = np.bincount(codes, minlength=4)
            family_rows.append({
                "family": family, "n": int(keep.sum()),
                "confusion_tn_fp_fn_tp": confusion[index].tolist(),
                **{key: nullable(value) for key, value in metrics_from_confusion(confusion[index]).items()},
            })
        estimates = metrics_from_confusion(confusion.sum(axis=0))
        draws[group] = metrics_from_confusion(weights @ confusion)
        rows.append({
            "group": group,
            **count_rows(examples, mask),
            "confusion_tn_fp_fn_tp": confusion.sum(axis=0).tolist(),
            "metrics": {
                metric: {
                    "estimate": nullable(estimates[metric]),
                    "ci95": interval(draws[group][metric]),
                    "valid_bootstrap_replicates": int(np.isfinite(draws[group][metric]).sum()),
                }
                for metric in METRICS
            },
            "per_family": family_rows,
        })
    return rows, draws


def protocol() -> dict[str, Any]:
    return {
        "layer": LAYER,
        "layer_selection": "Historical final hidden layer 28; preset, no heldout selection.",
        "train_families": list(TRAIN_FAMILIES),
        "test_families": list(TEST_FAMILIES),
        "nested_family_order": list(TRAIN_FAMILIES),
        "budgets": list(BUDGETS),
        "source_only_conditions": ["source-clean"],
        "mixed_conditions": list(TRAIN_CONDITIONS),
        "target_refit_conditions": list(TARGET_CONDITIONS),
        "target_refit_definition": (
            "Fresh scaler and decoder trained only on renamed-shift and opaque-readout "
            "rows of the selected TRAIN families. No source rows in these fits."
        ),
        "budget_interpretation": (
            "Number of target training families used by a fresh decoder; a family has "
            "30 labeled target examples. The fixed mixed baseline already used target "
            "rows from all 12 training families. These are subset refits, not new data "
            "added to that baseline. The ascending family order is fixed, not averaged "
            "over calibration-family choices. Semantic-readout has zero training rows."
        ),
        "hyperparameters": {
            kind: make_decoder(kind)[-1].get_params() for kind in ("linear", "rbf")
        },
        "selection": "All settings preset to original linear/RBF settings; no tuning or validation.",
        "preprocessing": "StandardScaler fit separately on each fit's training rows only.",
        "inference": "Frozen cached vectors; decoder fitting is CPU-only; no GPU/model updates.",
        "interpretation": "This experiment measures probe generalization under fixed decoders and training subsets. Source directness constraints are unchanged. It does not establish that all possible decoders fail or that a mechanism is absent from the representation.",
        "bootstrap": {
            "unit": "heldout family", "family_count": 6,
            "interval": "95% percentile", "seed": SEED,
            "paired": True,
            "statistic": "Pooled row metrics after resampling six entire families with replacement.",
            "scope": "Conditional on fitted decoders and fixed training subsets; six heldout clusters.",
            "training_variability_included": False,
            "simultaneous_intervals": False,
            "degenerate_intervals": "Empirical zero variation can yield [0,0] or [1,1]; not population certainty.",
        },
    }


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    import sklearn
    from threadpoolctl import threadpool_limits

    started = time.perf_counter()
    examples = build_probe_examples(TRAIN_FAMILIES, TEST_FAMILIES, repeats=2)
    masks = training_masks(examples)
    if args.fresh:
        features, model, feature_record = fresh_features(examples, args)
    else:
        features, model, feature_record = load_verified_cache(
            examples, Path(args.feature_cache), Path(args.examples_metadata),
        )
    settings = protocol()
    settings["feature_source"] = "fresh exact-prompt extraction" if args.fresh else "legacy cached extraction"
    if args.fresh:
        settings["inference"] = "Fresh frozen Qwen extraction on GPU1; decoder fitting CPU-only; no model updates."
    settings["bootstrap"]["replicates"] = args.bootstrap_replicates
    test_mask = np.array([x.split == "test" for x in examples])
    heldout = [x for x, keep in zip(examples, test_mask) if keep]
    groups = _group_masks(heldout)
    groups["target-all"] = np.array([x.condition != "source-clean" for x in heldout])
    groups["all-test"] = np.ones(len(heldout), dtype=bool)
    labels = np.array([x.label for x in examples])
    weights = cluster_bootstrap_weights(len(TEST_FAMILIES), args.bootstrap_replicates, SEED)
    fit_audit = {}
    training = {}
    methods = {}
    values = features[f"full_layer_{LAYER}"]
    with threadpool_limits(limits=args.cpu_threads):
        for regime, mask in masks.items():
            training[regime] = {
                **count_rows(examples, mask),
                "row_indices": np.flatnonzero(mask).tolist(),
                "example_ids": [x.example_id for x, keep in zip(examples, mask) if keep],
            }
            for kind in ("linear", "rbf"):
                name = f"{regime}/{kind}"
                print(f"Fitting {name}: {int(mask.sum())} rows", flush=True)
                prediction, scores, details = fit_decoder(values, labels, mask, test_mask, kind)
                methods[name] = (prediction, scores)
                fit_audit[name] = {"training_regime": regime, **details}
        query_prediction, query_scores, details = fit_decoder(
            features[f"query_layer_{LAYER}"], labels, masks["mixed_condition"], test_mask, "linear",
        )
        methods["control/query_only_linear_mixed"] = (query_prediction, query_scores)
        fit_audit["control/query_only_linear_mixed"] = {
            "training_regime": "mixed_condition", **details,
        }
    for method, field in (
        ("control/paired_evidence", "evidence_rule_prediction"),
        ("control/gated_state", "gated_state_prediction"),
    ):
        prediction = np.array([getattr(x, field) for x in heldout], dtype=int)
        methods[method] = (prediction, prediction.astype(float))
    rows = []
    bootstrap = {}
    for name, (prediction, _) in methods.items():
        metric_rows, bootstrap[name] = evaluate_predictions(heldout, prediction, groups, weights)
        rows.extend({"method": name, **row} for row in metric_rows)
    lookup = {(r["method"], r["group"]): r for r in rows}
    comparisons = []
    for regime in masks:
        if regime == "mixed_condition":
            continue
        for kind in ("linear", "rbf"):
            name = f"{regime}/{kind}"
            baseline = f"mixed_condition/{kind}"
            for group in groups:
                for metric in METRICS:
                    current = lookup[(name, group)]["metrics"][metric]["estimate"]
                    fixed = lookup[(baseline, group)]["metrics"][metric]["estimate"]
                    comparisons.append({
                        "method": name, "baseline": baseline, "group": group,
                        "metric": metric, "direction": "method minus fixed mixed baseline",
                        "estimate": current - fixed if current is not None and fixed is not None else None,
                        "ci95": interval(bootstrap[name][group][metric] - bootstrap[baseline][group][metric]),
                    })
    payload = {
        "experiment": "jmlr-qwen-hidden-state-target-train-only-reanalysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": settings, "model": model, "feature_record": feature_record,
        "dataset_counts": count_rows(examples, np.ones(len(examples), dtype=bool)),
        "split_counts": {
            split: count_rows(examples, np.array([x.split == split for x in examples]))
            for split in ("train", "test")
        },
        "training": training, "fit_records": fit_audit,
        "rows": rows, "paired_comparisons": comparisons,
        "heldout_predictions": {
            "row_indices": np.flatnonzero(test_mask).tolist(),
            "example_ids": [x.example_id for x in heldout],
            "labels": labels[test_mask].tolist(),
            "methods": {
                name: {"predictions": prediction.tolist(), "scores": scores.tolist()}
                for name, (prediction, scores) in methods.items()
            },
        },
        "runtime": {
            "python": platform.python_version(), "executable": sys.executable,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "cpu_threads": args.cpu_threads, "gpu_used": args.fresh,
            "analysis_seconds": time.perf_counter() - started,
        },
    }
    root = Path(args.output_root)
    results = root / "results"
    experiments = root / "experiments"
    figures = root / "figures"
    for directory in (results, experiments, figures):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": settings,
        "invocation": {"executable": sys.executable, "argv": sys.argv, "arguments": vars(args)},
        "implementation_sha256": sha256_file(Path(__file__)),
        "legacy_generator_sha256": sha256_file(Path(__file__).with_name("llm_hidden_retention.py")),
        "current_system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "prompt_hash_status": "Exact fingerprints verified during fresh extraction." if args.fresh else "Regenerated now, not historical extraction fingerprints.",
        "examples": [
            {"row_index": index, **{
                key: value for key, value in asdict(example).items()
                if key not in ("full_prompt", "query_only_prompt")
            }, **prompt_fingerprints(example)}
            for index, example in enumerate(examples)
        ],
        "bootstrap_family_order": list(TEST_FAMILIES),
        "bootstrap_weights_sha256": hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest(),
    }
    reference_path = Path(args.examples_metadata)
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else {"rows": []}
    reference_rows = {(r["method"], r["group"]): r for r in reference["rows"]}
    differences = []
    compared = 0
    for row in rows:
        original_name = {"mixed_condition/linear": "Qwen linear layer 28", "mixed_condition/rbf": "Qwen nonlinear RBF"}.get(row["method"])
        if (original_name, row["group"]) not in reference_rows:
            continue
        original = reference_rows[(original_name, row["group"])]
        for metric in METRICS:
            current = row["metrics"][metric]["estimate"]
            if current is None or original[metric] is None:
                continue
            compared += 1
            if abs(current - original[metric]) > 1e-12:
                differences.append({"method": row["method"], "group": row["group"], "metric": metric, "fresh_or_current": current, "legacy": original[metric]})
    payload["mixed_baseline_reference_comparison"] = {"compared_metrics": compared, "differences_over_1e_12": differences}
    if args.fresh and Path(args.feature_cache).exists() and reference_path.exists():
        legacy_features, _, _ = load_verified_cache(examples, Path(args.feature_cache), Path(args.examples_metadata))
        payload["legacy_feature_comparison"] = {}
        for key in (f"full_layer_{LAYER}", f"query_layer_{LAYER}"):
            difference = np.abs(features[key] - legacy_features[key])
            a = features[key].astype(np.float64)
            b = legacy_features[key].astype(np.float64)
            cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            payload["legacy_feature_comparison"][key] = {
                "max_absolute_difference": float(difference.max()),
                "mean_absolute_difference": float(difference.mean()),
                "mean_row_cosine_similarity": float(cosine.mean()),
                "minimum_row_cosine_similarity": float(cosine.min()),
                "historical_prompt_identity_implied": False,
            }
    if args.fresh:
        payload["primary_result"] = {"fresh_features_validated": True, "role": "Primary probe-generalization reanalysis; legacy cached run preserved separately."}
    stem = "llm_fresh_reanalysis" if args.fresh else "llm_reanalysis"
    for path, content in (
        (results / f"{stem}.json", payload),
        (experiments / f"{stem}_record.json", manifest),
    ):
        path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_figure(payload, figures, stem)
    summary = status_text(payload)
    (results / f"{stem}_status.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    return payload


def status_text(payload: dict[str, Any]) -> str:
    lookup = {(row["method"], row["group"]): row for row in payload["rows"]}

    def display(method: str, group: str, metric: str) -> str:
        item = lookup[(method, group)]["metrics"][metric]
        low, high = item["ci95"]
        return f"{item['estimate']:.4f} [{low:.4f}, {high:.4f}]"

    fresh = payload["protocol"].get("feature_source") == "fresh exact-prompt extraction"
    lines = [
        "Qwen hidden-state reanalysis: completed; fresh exact-prompt features on GPU1, decoder fits on CPU; primary result." if fresh else "Qwen hidden-state reanalysis: completed; frozen cached vectors, CPU only.",
        "Train families 1-12; disjoint test families 101-106 (6 clusters).",
        "Source-only: source-clean, 144 rows (72/72 negative/positive).",
        "Fixed mixed baseline: source-clean 144 + renamed-shift 144 + opaque-readout 216 = 504 rows (252/252).",
        "Target-only refit budgets 1/3/6/12 families: 30/90/180/360 rows, balanced labels.",
        "Refit conditions: renamed-shift + opaque-readout; semantic-readout has zero train rows.",
        "Mixed baseline already sees all 12 target train families; curve is subset refitting, not extra data.",
        "Test: source-clean 72 + renamed-shift 72 + opaque-readout 108 + semantic-readout 108 = 360 rows (180/180).",
        f"All intervals: 95% heldout-family cluster percentile bootstrap ({payload['protocol']['bootstrap']['replicates']} resamples); fixed fits/subsets.",
        "Layer 28, linear C=1, RBF C=2 gamma=scale, balanced classes; no heldout hyperparameter selection.",
        "Fresh extraction: exact user/system/rendered-chat/token-ID fingerprints and embedded ordered row bindings verified for all 864 rows." if fresh else "Cache: exact ordered IDs/all available saved row metadata/model metadata matched (864 rows).",
        "Pinned Qwen snapshot a09a35458c702b33eeacc393d103063234e8bc28; local weights only; no download or truncation." if fresh else "Historical prompt-text fingerprints and per-vector row IDs unavailable; historical full-prompt identity is unverified.",
        "Legacy cached reanalysis is preserved separately; fresh vectors are the primary result." if fresh else "Current regenerated prompt hashes and file hashes are in the experiment record.",
        "Metric columns: renamed BA; shifted-delay BA; semantic BA; semantic candidate FPR [95% CI].",
    ]
    for regime in ("source_only", "mixed_condition", *(f"target_refit_{b:02d}" for b in BUDGETS), "control"):
        kinds = ("paired_evidence", "gated_state", "query_only_linear_mixed") if regime == "control" else ("linear", "rbf")
        for kind in kinds:
            name = f"{regime}/{kind}"
            values = [display(name, group, metric) for group, metric in (
                ("renamed-shift", "balanced_accuracy"),
                ("shifted-delay", "balanced_accuracy"),
                ("semantic-readout", "balanced_accuracy"),
                ("semantic-readout-candidates", "false_positive_rate"),
            )]
            lines.append(f"{name}: " + "; ".join(values))
    lines.extend([
        "Raw JSON includes all cell/label counts, heldout predictions/scores, per-family confusion matrices, and paired deltas versus fixed mixed baselines.",
        "Intervals describe six heldout families, not training-family choice variability or simultaneous coverage.",
        "Probe generalization under the tested decoders; source directness constraints unchanged; not proof that all decoders fail.",
        "No manuscript/response edits or public upload.",
    ])
    return "\n".join(lines) + "\n"


def make_figure(payload: dict[str, Any], directory: Path, stem: str = "llm_reanalysis") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    lookup = {(row["method"], row["group"]): row for row in payload["rows"]}
    colors = {"source_only": "#A7B0B9", "mixed_condition": "#3F6FB6", "target_refit_12": "#20A37D"}
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6))
    top_groups = ("renamed-shift", "shifted-delay", "semantic-readout")
    top_titles = ("Renamed target", "Shifted delay (12 probes)", "Unseen semantic readout")
    for axis, group, title in zip(axes[0], top_groups, top_titles):
        for offset, (regime, color) in zip((-0.22, 0, 0.22), colors.items()):
            for x, kind in enumerate(("linear", "rbf")):
                item = lookup[(f"{regime}/{kind}", group)]["metrics"]["balanced_accuracy"]
                estimate = item["estimate"]
                axis.errorbar(x + offset, estimate, yerr=np.array([
                    [estimate - item["ci95"][0]], [item["ci95"][1] - estimate],
                ]), fmt="o", ms=5, color=color, capsize=3, lw=1.1)
        for control, color, style in (
            ("paired_evidence", "#64748B", ":"), ("gated_state", "#172033", "--"),
        ):
            estimate = lookup[(f"control/{control}", group)]["metrics"]["balanced_accuracy"]["estimate"]
            axis.axhline(estimate, color=color, ls=style, lw=1.1)
        axis.axhline(0.5, color="#D9474F", ls=":", lw=0.8, alpha=0.65)
        axis.set(xticks=[0, 1], xticklabels=["Linear", "RBF"], ylim=(0, 1.06), xlim=(-0.55, 1.55), title=title)
    axes[0, 0].set_ylabel("Heldout balanced accuracy")
    for axis, kind in zip(axes[1, :2], ("linear", "rbf")):
        for group, color, marker in (
            ("renamed-shift", "#3F6FB6", "o"), ("semantic-readout", "#20A37D", "s"),
        ):
            items = [lookup[(f"target_refit_{b:02d}/{kind}", group)]["metrics"]["balanced_accuracy"] for b in BUDGETS]
            values = np.array([item["estimate"] for item in items])
            axis.plot(BUDGETS, values, marker=marker, color=color, lw=1.5, label="Renamed target" if group == "renamed-shift" else "Unseen semantic")
            axis.fill_between(BUDGETS, [v["ci95"][0] for v in items], [v["ci95"][1] for v in items], color=color, alpha=0.13, linewidth=0)
            baseline = lookup[(f"mixed_condition/{kind}", group)]["metrics"]["balanced_accuracy"]["estimate"]
            axis.axhline(baseline, color=color, ls="--", lw=1.1)
        axis.set(title=f"{kind.upper() if kind == 'rbf' else 'Linear'} target-only refit", xticks=BUDGETS, ylim=(0.4, 1.03), xlim=(0.7, 12.3), xlabel="Target train families")
    axes[1, 0].set_ylabel("Heldout balanced accuracy")
    axes[1, 0].legend(frameon=False, loc="lower right", fontsize=8)
    axis = axes[1, 2]
    for kind, color, marker in (("linear", "#D9474F", "o"), ("rbf", "#64748B", "s")):
        items = [lookup[(f"target_refit_{b:02d}/{kind}", "semantic-readout-candidates")]["metrics"]["false_positive_rate"] for b in BUDGETS]
        axis.plot(BUDGETS, [v["estimate"] for v in items], color=color, marker=marker, lw=1.5, label=kind)
        axis.fill_between(BUDGETS, [v["ci95"][0] for v in items], [v["ci95"][1] for v in items], color=color, alpha=0.12, linewidth=0)
        baseline = lookup[(f"mixed_condition/{kind}", "semantic-readout-candidates")]["metrics"]["false_positive_rate"]["estimate"]
        axis.axhline(baseline, color=color, ls="--", lw=1.1)
    axis.set(title="Semantic readout candidates (18)", xticks=BUDGETS, ylim=(-0.025, 0.8), xlim=(0.7, 12.3), xlabel="Target train families", ylabel="False-positive rate")
    axis.legend(frameon=False, loc="upper right", fontsize=8)
    for index, axis in enumerate(axes.flat):
        axis.text(-0.12, 1.06, chr(97 + index), transform=axis.transAxes, fontweight="bold", fontsize=11)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8)
    handles = [Line2D([], [], marker="o", ls="", color=color, label=label) for color, label in (
        (colors["source_only"], "Source-only (12 families)"),
        (colors["mixed_condition"], "Fixed mixed (12 families)"),
        (colors["target_refit_12"], "Target-only (12 families)"),
    )]
    handles.extend([
        Line2D([], [], color="#64748B", ls=":", label="Paired-evidence rule"),
        Line2D([], [], color="#172033", ls="--", label="Gated state"),
    ])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), frameon=False, ncol=5, fontsize=8)
    fig.suptitle("Qwen2.5-7B frozen hidden states: family-disjoint decoder reanalysis", y=0.99, fontsize=13)
    fig.text(0.06, 0.036, "Layer 28; six heldout families (101-106); 95% family-cluster bootstrap. Shading: refit CI; dashed curves: fixed mixed baseline.", fontsize=8)
    fig.text(0.06, 0.012, "Refits use renamed + opaque TRAIN examples only. Mixed baseline already uses all 12 target train families; semantic condition is unseen.", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.13, wspace=0.3, hspace=0.55)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=240)
    fig.savefig(directory / f"{stem}.pdf")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", default="tmp/llm_hidden_retention_12train_6test_features.npz")
    parser.add_argument("--examples-metadata", default="tmp/llm_hidden_retention_12train_6test_final.json")
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--fresh", action="store_true", help="Use separately stored exact-prompt fresh features")
    parser.add_argument("--regenerate-fresh", action="store_true", help="Regenerate fresh features even if they exist")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
