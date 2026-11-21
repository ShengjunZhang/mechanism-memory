"""Run mechanism-memory experiments and analyze saved records."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = "analysis/tools/"
JOBS = {
    "validate": [EXPERIMENTS + "historical_reproduction.py", "validate"],
    "map": [EXPERIMENTS + "historical_reproduction.py", "matrix"],
    "figures": [EXPERIMENTS + "historical_reproduction.py", "replot"],
    "finite": ["-m", "causal_sandbox.finite_interventions", "--protocol", "configs/finite_protocol.json"],
    "answer-risk": [EXPERIMENTS + "derive_finite_answer_risk.py"],
    "world-model": ["-m", "causal_sandbox.world_model_probes"],
    "llm": ["-m", "causal_sandbox.llm_state_probes"],
    "readout": ["-m", "causal_sandbox.readout_conflict_followup"],
    "readout-analysis": ["experiments/readout_conflict_20261114/analyze.py"],
    "sequential": ["-m", "causal_sandbox.tdmpc2_sequential"],
    "sequential-analysis": ["-m", "causal_sandbox.analyze_tdmpc2_sequential"],
    "sequential-figure": ["-m", "causal_sandbox.plot_tdmpc2_sequential"],
    "tests": ["-m", "pytest", "tests", EXPERIMENTS + "test_historical_reproduction.py", "-q"],
}


def verify_files():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Invalid member: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing repository file: {name}; use a complete clone.")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"Changed repository file: {name}")
    print(f"Verified {len(manifest['files'])} repository files.", flush=True)
    from causal_sandbox.source_compat import verify_source_transforms
    print(f"Verified {verify_source_transforms()} source transformations.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Additional arguments are passed to the selected driver. See README.md and REPRODUCTION.txt.")
    parser.add_argument("command", choices=["verify", *JOBS])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.arguments
    if args.command == "verify":
        if extra:
            parser.error("verify takes no additional arguments")
        verify_files()
        return
    if args.command == "validate":
        verify_files()
    result = subprocess.run([sys.executable, "-B", *JOBS[args.command], *extra], cwd=ROOT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
