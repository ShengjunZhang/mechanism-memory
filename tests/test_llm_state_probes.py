from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from causal_sandbox.llm_hidden_retention import build_probe_examples
from causal_sandbox.llm_state_probes import (
    BUDGETS, LAYER, SNAPSHOT, TEST_FAMILIES, TRAIN_FAMILIES, canonical_hash, cluster_bootstrap_weights,
    count_rows, evaluate_predictions, fit_decoder, load_verified_cache,
    load_fresh_vectors, metrics_from_confusion, ordered_id_fingerprint,
    prepare_prompt_records, prompt_fingerprints, sha256_file,
    training_masks, validate_dataset, validate_fresh_records, verify_cache_metadata,
)


class RevisionLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = build_probe_examples(TRAIN_FAMILIES, TEST_FAMILIES, repeats=2)
        cls.model = {
            "model": "Qwen/Qwen2.5-7B-Instruct", "snapshot_revision": "test-snapshot",
            "layers": [LAYER], "hidden_size": 2,
        }
        cls.sidecar = {
            "dataset_fingerprint": ordered_id_fingerprint(cls.examples),
            "model_metadata": cls.model,
        }
        cls.saved = {
            "train_families": list(TRAIN_FAMILIES), "test_families": list(TEST_FAMILIES),
            "repeats": 2, "model": cls.model,
            "examples": [{k: v for k, v in asdict(x).items() if "prompt" not in k} for x in cls.examples],
        }

    def test_nested_target_masks_source_definition_and_exact_counts(self) -> None:
        masks = training_masks(self.examples)
        self.assertEqual(int(masks["source_only"].sum()), 144)
        self.assertEqual(int(masks["mixed_condition"].sum()), 504)
        previous = np.zeros(len(self.examples), dtype=bool)
        for budget in BUDGETS:
            mask = masks[f"target_refit_{budget:02d}"]
            self.assertTrue(np.all(~previous | mask))
            rows = [x for x, keep in zip(self.examples, mask) if keep]
            self.assertEqual(len(rows), 30 * budget)
            self.assertEqual({x.family for x in rows}, set(TRAIN_FAMILIES[:budget]))
            self.assertEqual({x.condition for x in rows}, {"renamed-shift", "opaque-readout"})
            self.assertTrue(all(x.split == "train" for x in rows))
            self.assertEqual(sum(x.label for x in rows), 15 * budget)
            self.assertTrue({x.family for x in rows}.isdisjoint(TEST_FAMILIES))
            previous = mask
        source = [x for x, keep in zip(self.examples, masks["source_only"]) if keep]
        self.assertEqual({x.condition for x in source}, {"source-clean"})
        mixed_counts = count_rows(self.examples, masks["mixed_condition"])
        self.assertEqual(mixed_counts["conditions"]["opaque-readout"]["n"], 216)
        self.assertEqual(mixed_counts["label_counts"], {"0": 252, "1": 252})

    def test_split_overlap_missing_cells_and_forbidden_conditions_rejected(self) -> None:
        changed = list(self.examples)
        test_index = next(i for i, x in enumerate(changed) if x.split == "test")
        changed[test_index] = replace(changed[test_index], family=1)
        with self.assertRaisesRegex(ValueError, "family overlap"):
            validate_dataset(changed)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_dataset(self.examples[1:])
        changed = list(self.examples)
        changed[0] = replace(changed[0], condition="semantic-readout")
        with self.assertRaisesRegex(ValueError, "Unexpected conditions"):
            validate_dataset(changed)

    def test_legacy_cache_reports_only_available_verification(self) -> None:
        audit = verify_cache_metadata(self.examples, self.sidecar, self.saved)
        self.assertEqual(audit["matched_rows"], 864)
        self.assertFalse(audit["historical_prompt_text_match_verified"])
        self.assertEqual(audit["saved_user_prompt_fingerprint_rows"], 0)
        self.assertIn("full_prompt", audit["regenerated_fields_unavailable_in_saved_rows"])

    def test_reordered_rows_changed_labels_and_model_rejected(self) -> None:
        for mutation in ("order", "label", "model", "missing", "duplicate"):
            saved = deepcopy(self.saved)
            if mutation == "order":
                saved["examples"][0], saved["examples"][1] = saved["examples"][1], saved["examples"][0]
            elif mutation == "label":
                saved["examples"][0]["label"] = 0
            elif mutation == "model":
                saved["model"]["snapshot_revision"] = "wrong"
            elif mutation == "missing":
                del saved["examples"][0]["candidate_kind"]
            else:
                saved["examples"][1] = deepcopy(saved["examples"][0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_cache_metadata(self.examples, self.sidecar, saved)
        reversed_examples = list(reversed(self.examples))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            verify_cache_metadata(reversed_examples, self.sidecar, self.saved)

    def test_prompt_changes_not_overclaimed_and_hashes_checked_when_present(self) -> None:
        changed = list(self.examples)
        changed[0] = replace(changed[0], full_prompt=changed[0].full_prompt + " changed record")
        # Legacy IDs omit the controlled-record text. Report this gap explicitly.
        audit = verify_cache_metadata(changed, self.sidecar, self.saved)
        self.assertFalse(audit["historical_prompt_text_match_verified"])
        saved = deepcopy(self.saved)
        for row, example in zip(saved["examples"], self.examples):
            row.update(prompt_fingerprints(example))
        with self.assertRaisesRegex(ValueError, "full_prompt_sha256"):
            verify_cache_metadata(changed, self.sidecar, saved)
        from causal_sandbox.llm_hidden_retention import SYSTEM_PROMPT
        sidecar = {**self.sidecar, "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}
        self.assertTrue(verify_cache_metadata(self.examples, sidecar, saved)["historical_prompt_text_match_verified"])

    def test_cache_shape_count_nonfinite_and_keys_rejected(self) -> None:
        n = len(self.examples)
        features = {
            f"full_layer_{LAYER}": np.zeros((n, 2), dtype=np.float16),
            f"query_layer_{LAYER}": np.zeros((n, 2), dtype=np.float16),
            "logit_margin": np.zeros(n, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "vectors.npz"
            metadata = cache.with_suffix(".npz.json")
            rows = Path(directory) / "examples.json"
            metadata.write_text(json.dumps(self.sidecar), encoding="utf-8")
            rows.write_text(json.dumps(self.saved), encoding="utf-8")
            np.savez(cache, **features)
            loaded, _, audit = load_verified_cache(self.examples, cache, rows)
            self.assertEqual(loaded[f"full_layer_{LAYER}"].shape, (864, 2))
            self.assertEqual(len(audit["files"]["features"]["sha256"]), 64)
            for mutation in ("rows", "nan", "keys"):
                altered = {k: v.copy() for k, v in features.items()}
                if mutation == "rows":
                    altered[f"full_layer_{LAYER}"] = altered[f"full_layer_{LAYER}"][:-1]
                elif mutation == "nan":
                    altered[f"full_layer_{LAYER}"][0, 0] = np.nan
                else:
                    altered["extra"] = np.zeros(n)
                np.savez(cache, **altered)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_verified_cache(self.examples, cache, rows)

    def test_fit_scaler_and_parameters_do_not_use_heldout_rows_or_labels(self) -> None:
        rng = np.random.default_rng(9)
        values = rng.normal(size=(24, 3))
        labels = np.tile([0, 1], 12)
        train = np.arange(24) < 16
        test = ~train
        for kind in ("linear", "rbf"):
            pred, scores, audit = fit_decoder(values, labels, train, test, kind)
            altered = values.copy()
            altered[test] += 1e6
            altered_labels = labels.copy()
            altered_labels[test] = 1 - altered_labels[test]
            unchanged_pred, unchanged_scores, _ = fit_decoder(values, altered_labels, train, test, kind)
            np.testing.assert_array_equal(pred, unchanged_pred)
            np.testing.assert_array_equal(scores, unchanged_scores)
            _, _, changed = fit_decoder(altered, altered_labels, train, test, kind)
            self.assertEqual(audit["scaler_mean_sha256"], changed["scaler_mean_sha256"])
            expected = hashlib.sha256(values[train].mean(axis=0).tobytes()).hexdigest()
            self.assertEqual(audit["scaler_mean_sha256"], expected)
            self.assertEqual(audit["classifier_parameters"], changed["classifier_parameters"])
            self.assertEqual(pred.shape, (8,))
            self.assertEqual(scores.shape, (8,))
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_decoder(values, labels, train, np.ones(24, dtype=bool), "linear")

    def test_cluster_resampling_preserves_families_and_single_class_fpr(self) -> None:
        weights = cluster_bootstrap_weights(6, 2000, 7)
        np.testing.assert_array_equal(weights.sum(axis=1), np.full(2000, 6))
        np.testing.assert_array_equal(weights, cluster_bootstrap_weights(6, 2000, 7))
        test = [x for x in self.examples if x.split == "test"]
        labels = np.array([x.label for x in test])
        prediction = labels.copy()
        prediction[np.array([x.family == 101 for x in test])] = 1 - prediction[np.array([x.family == 101 for x in test])]
        masks = {
            "all": np.ones(len(test), dtype=bool),
            "negative-only": np.array([x.candidate_kind == "readout" for x in test]),
        }
        rows, draws = evaluate_predictions(test, prediction, masks, weights)
        self.assertAlmostEqual(rows[0]["metrics"]["accuracy"]["estimate"], 5 / 6)
        np.testing.assert_allclose(draws["all"]["accuracy"], 1 - weights[:, 0] / 6)
        self.assertIsNone(rows[1]["metrics"]["balanced_accuracy"]["estimate"])
        self.assertIsNone(rows[1]["metrics"]["balanced_accuracy"]["ci95"])
        self.assertAlmostEqual(rows[1]["metrics"]["false_positive_rate"]["estimate"], 1 / 6)
        np.testing.assert_allclose(draws["negative-only"]["false_positive_rate"], weights[:, 0] / 6)
        self.assertEqual(len(rows[0]["per_family"]), 6)
        self.assertTrue(np.all(draws["all"]["accuracy"] - draws["all"]["accuracy"] == 0))

    def test_confusion_metrics_use_pooled_counts_not_wrong_family_average(self) -> None:
        counts = np.array([[1, 0, 0, 1], [0, 9, 9, 0]])
        metrics = metrics_from_confusion(counts.sum(axis=0))
        self.assertAlmostEqual(float(metrics["accuracy"]), 0.1)
        self.assertAlmostEqual(float(metrics["balanced_accuracy"]), 0.1)
        self.assertAlmostEqual(float(metrics["false_positive_rate"]), 0.9)

    def test_fresh_prompt_rendered_token_and_order_fingerprints(self) -> None:
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages) + "assistant:"

            def __call__(self, text, **kwargs):
                return {"input_ids": [len(text), 123]}

        examples = self.examples[:2]
        rows, texts = prepare_prompt_records(examples, Tokenizer())
        from causal_sandbox.llm_hidden_retention import SYSTEM_PROMPT
        metadata = {
            "schema": "qwen-exact-prompt-v1", "snapshot_revision": SNAPSHOT,
            "layer": LAYER, "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "ordered_example_id_fingerprint": ordered_id_fingerprint(examples),
            "ordered_row_fingerprints_sha256": canonical_hash([r["row_fingerprint"] for r in rows]),
            "rows": rows, "max_length": 1536, "truncated_prompts": 0,
        }
        validate_fresh_records(examples, rows, metadata)
        self.assertEqual(len(texts["full"]), 2)
        for key in ("full_prompt_sha256", "query_only_prompt_sha256", "full_rendered_chat_sha256", "query_token_ids_sha256"):
            self.assertEqual(len(rows[0][key]), 64)
        changed = [replace(examples[0], full_prompt=examples[0].full_prompt + "changed"), examples[1]]
        changed_rows, _ = prepare_prompt_records(changed, Tokenizer())
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_fresh_records(changed, changed_rows, metadata)
        with self.assertRaisesRegex(ValueError, "ordered"):
            validate_fresh_records(examples, list(reversed(rows)), metadata)
        features = {
            f"full_layer_{LAYER}": np.zeros((2, 3584), dtype=np.float16),
            f"query_layer_{LAYER}": np.zeros((2, 3584), dtype=np.float16),
            "logit_margin": np.zeros(2, dtype=np.float32),
        }
        bindings = {
            "row_indices": np.arange(2),
            "example_ids": np.array([x.example_id for x in examples]),
            "row_fingerprints": np.array([r["row_fingerprint"] for r in rows]),
        }
        metadata["array_sha256"] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in features.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm_fresh_features.npz"
            np.savez(path, **features, **bindings)
            metadata["feature_file_sha256"] = sha256_file(path)
            sidecar = path.with_suffix(".npz.json")
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            loaded, _ = load_fresh_vectors(path, examples, rows)
            self.assertEqual(loaded[f"full_layer_{LAYER}"].shape, (2, 3584))
            bindings["row_fingerprints"] = bindings["row_fingerprints"][::-1]
            np.savez(path, **features, **bindings)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_fresh_vectors(path, examples, rows)
            metadata["feature_file_sha256"] = sha256_file(path)
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row binding"):
                load_fresh_vectors(path, examples, rows)


if __name__ == "__main__":
    unittest.main()
