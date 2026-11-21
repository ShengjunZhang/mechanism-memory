import json
from pathlib import Path
import re

from analysis.tools.historical_reproduction import source_path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_filenames_are_domain_neutral():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name in manifest["files"]:
        assert not re.search(r"(?i)revision|jmlr|icml|nature|ncomms|journal|nmi", name)


def test_frozen_snapshot_paths_resolve_to_repository_files():
    snapshot = json.loads((ROOT / "data/controlled/experiments/reproduction_snapshot.json").read_text())
    for name in snapshot["inspected_files"]:
        assert source_path(ROOT, name).is_file(), name


def test_data_files_fit_regular_git_objects():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name in manifest["files"]:
        assert (ROOT / name).stat().st_size < 100 * 2**20, name
