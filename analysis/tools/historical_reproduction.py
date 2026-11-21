"""Validate saved sources and replot results without acquiring data or fitting models."""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from causal_sandbox.source_compat import recorded_source_matches
HERE = Path(__file__).resolve().parent
REVISION = "data/controlled"
HISTORICAL = "data/historical"
RAW = {
    "coarse": "finite_agent_comparison_complex_100x5.json",
    "matched": "matched_mechanism_baselines_8families_2seeds.json",
    "ranking": "ranking_and_adaptation_baselines_8families_2seeds.json",
    "continuous": "continuous_metric_comparison_20seeds.json",
    "noise": "continuous_noise_sweep_20seeds.json",
    "tdmpc2": "tdmpc2_distributional_retention_3tasks_3seeds.json",
    "hidden": "llm_hidden_retention_12train_6test.json",
    "proposal": "llm_proposal_policy_comparison_3seeds.json",
    "budget": "candidate_budget_sensitivity_4families.json",
}
# Labels are stable identifiers, not page or table-number assertions.
RESULT_MAP = [
    ("fig:main-results-v10", "coarse,matched,ranking,proposal", "SourceData_Fig2_fixed_probe_outcomes.csv; replot --scope historical"),
    ("tab:matched-baselines", "matched", "matched_mechanism_baselines.summary; command H2"),
    ("tab:ranking-adaptation", "ranking", "summary (saved adaptation schema); command H3"),
    ("tab:continuous-results", "continuous", "summary; command H4"),
    ("tab:tdmpc2-responses", "tdmpc2", "SourceData_Table4_tdmpc2_by_task.csv and runs; command H6"),
    ("tab:llm-hidden-results", "hidden", "rows / SourceData_Fig4_llm_hidden_retention.csv; command H7"),
    ("tab:llm-proposal-results", "proposal", "worlds.panel-complex-noisy-hidden.*.summary; command H8"),
    ("fig:tdmpc2-retention", "tdmpc2", "summary; replot --scope historical"),
    ("fig:llm-hidden-retention", "hidden", "SourceData_Fig4_llm_hidden_retention.csv; replot --scope historical"),
    ("coarse-interface text", "coarse", "results.panel-complex-noisy-hidden.100.summary; command H1"),
    ("continuous-noise text", "noise", "continuous_noise_sweep.summary and payloads; command H5"),
    ("candidate-budget / hidden-recall text", "budget", "candidate_budget_sensitivity.runs and summary; H9 has missing metadata"),
    ("tab:tdmpc2-independent", "added TD", "results/tdmpc2/main_results.csv and combined_summary.json; A1"),
    ("tab:tdmpc2-decoders", "added TD", "results/tdmpc2/default/tdmpc2_*.json rows, fixed-decoder selection in REPRODUCTION.txt; A1"),
    ("fig:tdmpc2-independent", "added TD", "results/tdmpc2/combined_summary.json settings.default.primary; replot --scope added"),
    ("fig:tdmpc2-sensitivity", "added TD", "results/tdmpc2/combined_summary.json settings.*.primary; replot --scope added"),
    ("tab:finite-tuples", "added finite", "results/finite_results.json controlled_aggregate plus finite_answer_risk.json primary; A2/A4"),
    ("tab:finite-budget", "added finite", "results/finite_budget_curve.json budgets; A2"),
    ("description-redacted comparison text", "added finite", "results/finite_results.json historical_aggregate / historical_provenance; A2"),
    ("tab:llm-revision", "added Qwen", "results/llm_fresh_reanalysis.json rows; A3"),
    ("fig:llm-revision", "added Qwen", "results/llm_fresh_reanalysis.json rows; replot --scope added"),
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest() -> dict:
    return json.loads((ROOT / REVISION / "experiments/reproduction_snapshot.json").read_text(encoding="utf-8"))


def source_path(root: Path, name: str) -> Path:
    locations = json.loads((root / "manifest.json").read_text())["source_locations"]
    name = locations.get(name, name)
    if name.startswith(f"{REVISION}/experiments/") and name.endswith(".py"):
        return root / "analysis/tools" / Path(name).name
    if name.startswith(HISTORICAL + "/") and name.endswith(".py"):
        return root / "analysis/historical" / Path(name).name
    return root / name


def historical_bytes(root: Path, source_data: Path | None = None,
                     source_zip: Path | None = None) -> dict[str, bytes]:
    record = manifest()["historical_archive"]
    directory = source_data or root / HISTORICAL
    if source_zip is not None:
        archive = source_zip.read_bytes()
        if digest(archive) != record["sha256"]:
            raise ValueError(f"Historical archive hash mismatch: {path}")
        with ZipFile(io.BytesIO(archive)) as handle:
            data = {name: handle.read(name) for name in record["members"]}
    elif directory.is_dir():
        data = {name: ((directory / name) if (directory / name).is_file()
                       else source_path(root, HISTORICAL + "/" + name)).read_bytes()
                for name in record["members"]}
    else:
        raise FileNotFoundError(f"Missing historical data directory: {directory}")
    for name, expected in record["members"].items():
        if digest(data[name]) != expected:
            raise ValueError(f"Historical member hash mismatch: {name}")
    return data


def stage(data: dict[str, bytes], directory: Path) -> None:
    for name, content in data.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Unsafe archive member: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def verify_snapshot(root: Path, scope: str) -> int:
    count = 0
    for relative, expected in manifest()["inspected_files"].items():
        path = source_path(root, relative)
        if scope == "historical" and path.is_relative_to(root / REVISION):
            continue
        if not recorded_source_matches(path, expected):
            raise ValueError(f"Inspected source/result hash mismatch: {relative}")
        count += 1
    return count


def validate(root: Path, data: dict[str, bytes], scope: str) -> dict:
    checked = verify_snapshot(root, scope)
    for name, content in data.items():
        if name.endswith(".json"):
            json.loads(content)
        elif name.endswith(".csv"):
            if not list(csv.DictReader(io.StringIO(content.decode("utf-8-sig")))):
                raise ValueError(f"Empty historical CSV: {name}")
    with tempfile.TemporaryDirectory(prefix="causal-validate-") as temporary:
        stage(data, Path(temporary))
        with redirect_stdout(io.StringIO()) as capture:
            runpy.run_path(str(Path(temporary) / "verify_reported_results.py"), run_name="__main__")
    return {"historical_members_checked": len(data), "inspected_files_checked": checked,
            "reported_claim_check": capture.getvalue().strip(),
            "scope": scope, "rerun_or_model_training": False,
            "limit": "Saved-byte and reported-claim consistency, not independent experimental reproduction."}


def load_plot(path: Path, function: str, globals_: dict):
    """Compile only the rendering function, excluding acquisition/training entry points."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    namespace = dict(globals_)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function]


def plot_added_td(root: Path, output: Path, np) -> None:
    payload = json.loads((root / REVISION / "results/tdmpc2/combined_summary.json").read_text())
    summaries = {key: value["primary"] for key, value in payload["settings"].items()}
    source = root / "analysis/tools/summarize_tdmpc2.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main.body)
                 if isinstance(n, ast.Import) and any(a.name == "matplotlib" for a in n.names))
    # Reuse only plotting statements; do not rerun bootstrap or rewrite saved summaries.
    body = main.body[start:]
    if isinstance(body[-1], ast.Expr) and isinstance(body[-1].value, ast.Call) and isinstance(body[-1].value.func, ast.Name) and body[-1].value.func.id == "print":
        body = body[:-1]
    namespace = {"np": np, "figure_dir": output, "all_summaries": summaries,
                 "summaries": summaries["default"],
                 "default": [r for r in summaries["default"] if r["budget"] == 5
                             and r["ratio"] in (None, 0.5) and r["alpha"] in (None, 5.0)]}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), namespace)


def export_table_inputs(root: Path, data: dict[str, bytes], output: Path, scope: str) -> None:
    tables = output / "table_inputs"
    tables.mkdir()
    if scope != "added":
        for name in ("SourceData_Table4_tdmpc2_by_task.csv", "SourceData_Table5_llm_proposal_policies.csv",
                     "SourceData_Fig4_llm_hidden_retention.csv"):
            (tables / name).write_bytes(data[name])
        selectors = {"matched": ("matched_mechanism_baselines", "summary"),
                     "ranking": ("summary",), "continuous": ("summary",),
                     "noise": ("continuous_noise_sweep", "summary"),
                     "budget": ("candidate_budget_sensitivity", "summary")}
        for key, selector in selectors.items():
            value = json.loads(data[f"raw_json/{RAW[key]}"])
            for part in selector:
                value = value[part]
            (tables / f"{key}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    if scope != "historical":
        for name in ("finite_results.json", "finite_budget_curve.json", "finite_answer_risk.json", "llm_fresh_reanalysis.json",
                     "tdmpc2/main_results.csv", "tdmpc2/combined_summary.json"):
            shutil.copyfile(root / REVISION / "results" / name, tables / Path(name).name)


def replot(root: Path, data: dict[str, bytes], output: Path, scope: str) -> dict:
    output = output.resolve()
    # A fresh destination prevents accidental replacement of manuscript/data files.
    if output.exists():
        raise ValueError(f"Output must be a new directory: {output}")
    if output == root or output.is_relative_to(root):
        raise ValueError("Use a new output directory outside the source checkout.")
    verify_snapshot(root, scope)
    import numpy as np
    with tempfile.TemporaryDirectory(prefix="causal-replot-") as temporary:
        os.environ["MPLCONFIGDIR"] = str(Path(temporary) / "matplotlib")
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib
        matplotlib.use("Agg")
        output.mkdir(parents=True)
        if scope != "added":
            inputs = Path(temporary) / "source_data"
            stage(data, inputs)
            overview = runpy.run_path(str(inputs / "make_probe_figures.py"), run_name="reproduction_plot")
            overview["make_overview"]()
            hidden = runpy.run_path(str(inputs / "make_llm_hidden_figure.py"), run_name="reproduction_plot")
            hidden["main"]()
            for file in (inputs / "generated_figures").iterdir():
                shutil.copyfile(file, output / file.name)
            payload = json.loads(data[f"raw_json/{RAW['tdmpc2']}"])
            load_plot(inputs / "tdmpc2_retention.py", "_make_figure", {"np": np, "Path": Path})(payload, output)
        if scope != "historical":
            plot_added_td(root, output, np)
            payload = json.loads((root / REVISION / "results/llm_fresh_reanalysis.json").read_text())
            load_plot(root / "causal_sandbox/llm_state_probes.py", "make_figure",
                      {"np": np, "Path": Path, "BUDGETS": (1, 3, 6, 12)})(payload, output, "llm_fresh_reanalysis")
        export_table_inputs(root, data, output, scope)
    report = {"scope": scope, "output": str(output),
              "figures": sorted(p.name for p in output.glob("*.pdf")),
              "model_training": False, "inputs": "hash-checked saved CSV/JSON",
              "limit": "Numerical replot; fonts, PDF metadata, and rendering can differ from original PDFs."}
    (output / "replot_record.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "replot", "matrix"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--scope", choices=("historical", "added", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.source_data and args.source_zip:
        parser.error("Choose --source-data or --source-zip, not both.")
    if args.command == "matrix":
        print("Result | Source group | Saved-data selector / command (see REPRODUCTION.txt)")
        for label, group, instruction in RESULT_MAP:
            sources = ", ".join(f"raw_json/{RAW[k]}" for k in group.split(",")) if not group.startswith("added") else group
            print(f"{label} | {sources} | {instruction}")
        return 0
    try:
        root = args.root.resolve()
        data = historical_bytes(root, args.source_data, args.source_zip) if args.scope != "added" else {}
        if args.command == "replot":
            if args.output is None:
                parser.error("replot requires --output NEW_DIRECTORY_OUTSIDE_CHECKOUT")
            result = replot(root, data, args.output, args.scope)
        else:
            result = validate(root, data, args.scope) if args.scope != "added" else {
                "inspected_files_checked": verify_snapshot(root, args.scope), "scope": "added",
                "limit": "Saved-byte validation only; no numerical reanalysis or model training."}
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, AssertionError, StopIteration) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
