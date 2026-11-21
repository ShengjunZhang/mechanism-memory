"""Verify the preserved failed development candidate without running confirmation."""
from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from causal_sandbox.source_compat import sources_match
from causal_sandbox.readout_increment_followup import settings, source_hashes, sha, select_settings

SPEC = importlib.util.spec_from_file_location("increment_checks", ROOT / "analyze.py")
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


def main():
    p = settings()
    lock = json.loads((ROOT / "development_lock.json").read_text())
    assert sources_match(source_hashes(), lock["source_hashes"])
    assert sha(ROOT / "protocol.json") == lock["protocol_sha256"]
    assert sha(Path(p["parent_protocol"])) == lock["parent_protocol_sha256"]
    rows, checks = [], []
    for seed in p["development_families"]:
        run, check = CHECKS.PRIOR.verify(ROOT / "results/development" / f"family_{seed}.json", p, lock)
        check["readout_rule_checks"] = CHECKS.THRESHOLD.verify_rules(run, p)
        check["increment_checks"], check["copy_checks"] = CHECKS.verify_increments(run, p, lock)
        checks.append(check)
        rows.extend(run["rows"])
    summary = json.loads((ROOT / "results/development/summary.json").read_text())
    CHECKS.PRIOR.verify_summary(summary, rows, p | {"confirmation_families": p["development_families"]})
    selected = select_settings(summary, p)
    assert selected == json.loads((ROOT / "development_selection.json").read_text())
    report = {"status": "development_retention_guard_failed", "confirmation_run": False,
              "verification_scope": "Independent rescoring of saved predictions and formula increments, not full retraining.",
              "rows_checked": len(rows), "checks": checks, "selection": selected}
    (ROOT / "development_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("checks", "selection")}, indent=2))


if __name__ == "__main__":
    main()
