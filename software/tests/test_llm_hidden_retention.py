from __future__ import annotations

import unittest

from causal_sandbox.llm_hidden_retention import build_probe_examples


class LLMHiddenRetentionDataTests(unittest.TestCase):
    def test_probe_family_is_balanced_and_disjoint(self) -> None:
        examples = build_probe_examples([1], [101], repeats=2)
        train = [example for example in examples if example.split == "train"]
        test = [example for example in examples if example.split == "test"]

        self.assertEqual(len(train), 42)
        self.assertEqual(len(test), 60)
        self.assertEqual(sum(example.label for example in train), len(train) // 2)
        self.assertEqual(sum(example.label for example in test), len(test) // 2)
        self.assertTrue({example.family for example in train}.isdisjoint(
            {example.family for example in test}
        ))

    def test_paired_evidence_recovers_clean_targets_and_delays(self) -> None:
        examples = build_probe_examples([1], [101], repeats=2)
        clean = [
            example
            for example in examples
            if example.condition in {"source-clean", "renamed-shift"}
        ]

        self.assertTrue(clean)
        self.assertTrue(
            all(example.evidence_rule_prediction == example.label for example in clean)
        )
        self.assertTrue(
            all(example.gated_state_prediction == example.label for example in clean)
        )


if __name__ == "__main__":
    unittest.main()
