"""Run the fixed confirmation environments in independent CPU processes."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from causal_sandbox.source_compat import sources_match
from causal_sandbox.noisy_readout_filter_followup import run_family, summarize, sha, source_hashes


def worker(seed, protocol, lock):
    result = run_family(seed, "confirmation", protocol, lock)
    return seed, result["seconds"]


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    lock = json.loads((ROOT / "confirmation_lock.json").read_text())
    assert sources_match(source_hashes(), lock["source_hashes"])
    assert sha(ROOT / "protocol.json") == lock["protocol_sha256"]
    assert sha(Path(__file__)) == lock["launcher_sha256"]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, seed, protocol, lock) for seed in protocol["confirmation_families"]]
        for future in as_completed(futures):
            seed, seconds = future.result()
            print(f"CONFIRMED family={seed} elapsed={seconds:.1f}s", flush=True)
    folder = ROOT / "results/confirmation"
    runs = [json.loads((folder / f"family_{seed}.json").read_text()) for seed in protocol["confirmation_families"]]
    summary = summarize(runs, protocol)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "families"} for r in summary if r["condition"] == "noise10"], indent=2))


if __name__ == "__main__":
    main()
