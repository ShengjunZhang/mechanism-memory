"""Lightweight reproduction checks; all generated files live in temporary directories."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

SPEC = importlib.util.spec_from_file_location("historical_reproduction", Path(__file__).with_name("historical_reproduction.py"))
reproduction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduction)


class HistoricalReproductionTests(unittest.TestCase):
    def test_saved_data_and_inspected_source_hashes(self):
        data = reproduction.historical_bytes(reproduction.ROOT)
        report = reproduction.validate(reproduction.ROOT, data, "all")
        self.assertEqual(report["historical_members_checked"], 21)
        self.assertIn("match the packaged raw results", report["reported_claim_check"])

    def test_archive_only_loading(self):
        data = reproduction.historical_bytes(reproduction.ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source_data.zip"
            with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
                for name, content in data.items():
                    handle.writestr(name, content)
            record = reproduction.manifest()
            record["historical_archive"]["sha256"] = reproduction.digest(archive.read_bytes())
            with patch.object(reproduction, "manifest", return_value=record):
                self.assertEqual(data, reproduction.historical_bytes(Path(temporary), source_zip=archive))

    def test_tampered_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reproduction.stage(reproduction.historical_bytes(reproduction.ROOT), directory)
            (directory / "README_SourceData.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                reproduction.historical_bytes(reproduction.ROOT, source_data=directory)

    def test_every_data_table_and_figure_has_a_map_entry(self):
        labels = {
            "fig:main-results-v10", "fig:tdmpc2-retention", "fig:llm-hidden-retention",
            "fig:tdmpc2-independent", "fig:tdmpc2-sensitivity", "fig:llm-revision",
            "tab:matched-baselines", "tab:ranking-adaptation", "tab:continuous-results",
            "tab:tdmpc2-responses", "tab:llm-hidden-results", "tab:llm-proposal-results",
            "tab:tdmpc2-independent", "tab:tdmpc2-decoders", "tab:finite-tuples",
            "tab:finite-budget", "tab:llm-revision",
        }
        mapped = {row[0] for row in reproduction.RESULT_MAP}
        self.assertEqual(labels - mapped, set())
        used = set()
        for _, group, _ in reproduction.RESULT_MAP:
            if not group.startswith("added"):
                used.update(group.split(","))
        self.assertEqual(used, set(reproduction.RAW))

    def test_reject_output_inside_source_tree_or_existing_directory(self):
        data = reproduction.historical_bytes(reproduction.ROOT)
        with self.assertRaises(ValueError):
            reproduction.replot(reproduction.ROOT, data, reproduction.ROOT / "not-created-by-test", "historical")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                reproduction.replot(reproduction.ROOT, data, Path(temporary), "historical")

    def test_all_six_figures_replot_without_training(self):
        try:
            import numpy as np
            import matplotlib.image as image
        except ImportError:
            self.skipTest("Replot requires numpy and matplotlib")
        with tempfile.TemporaryDirectory(prefix="causal-reproduction-test-") as temporary:
            output = Path(temporary) / "figures"
            data = reproduction.historical_bytes(reproduction.ROOT)
            report = reproduction.replot(reproduction.ROOT, data, output, "all")
            self.assertFalse(report["model_training"])
            self.assertEqual(len(report["figures"]), 6)
            for name in report["figures"]:
                self.assertTrue((output / name).read_bytes().startswith(b"%PDF"))
                pixels = image.imread(output / Path(name).with_suffix(".png"))
                self.assertGreater(float(np.ptp(pixels[..., :3])), 0.5)
            record = json.loads((output / "replot_record.json").read_text())
            self.assertEqual(record["scope"], "all")
            self.assertTrue((output / "table_inputs/matched.json").is_file())


if __name__ == "__main__":
    unittest.main()
