# Mechanism Memory

Code and data for **Causal Retention in Interactive Agents**.

This repository contains mechanism-memory agents, matched baselines, experiment
drivers, protocols, tests and saved experiment records. Experiments cover finite
interventions, continuous simulators, TD-MPC2 response retention, readout
filtering, sequential adaptation and Qwen hidden-state probes.

## Setup

Clone the repository and install the dependencies with Python 3.12:

```text
git clone --depth 1 https://github.com/ShengjunZhang/mechanism-memory.git
cd mechanism-memory
python -m pip install -r requirements.txt
```

Code and experiment records are stored directly in the repository. GitHub's
**Code > Download ZIP** is an alternative to cloning. The checked-out records
occupy about 1.5 GiB. [DATA_GUIDE.md](DATA_GUIDE.md) lists the data directories,
result mappings and commands for saved-result analysis.

## Check and Reproduce Results

Run these commands from the repository root:

```text
python reproduce.py verify
python reproduce.py validate
python reproduce.py tests
```

Verification checks file hashes and recorded source identities. Validation
checks saved numerical results. Tests cover algorithms and saved-record checks.

The following commands use the saved records in this repository:

```text
python reproduce.py map
python reproduce.py validate
python reproduce.py figures --output ../reproduced_figures
python reproduce.py answer-risk
python reproduce.py readout-analysis
python reproduce.py sequential-analysis
python reproduce.py sequential-figure
```

The figure output directory must be new and outside the checkout.
Analysis commands can rewrite derived summaries. Run `verify` before analysis.
The full protocols, model versions, parameters and result-to-source mapping are
in [REPRODUCTION.txt](REPRODUCTION.txt).

## Run Experiments

```text
python reproduce.py finite --output outputs/finite
python reproduce.py world-model --output outputs/world_model --device cuda:0 --pulse 0.25
python reproduce.py llm --fresh --gpu 0 --output-root outputs/qwen
python reproduce.py readout --phase development
python reproduce.py readout --phase confirmation
python reproduce.py sequential --phase development --device cuda:0
python reproduce.py sequential --phase confirmation --device cuda:0
```

Use `--help` for driver options. Readout and sequential commands reuse saved
records when present. New acquisition follows the development and confirmation
protocols in `REPRODUCTION.txt`. TD-MPC2 and Qwen require the specified checkpoints.

## Repository Layout

| Directory | Contents |
| --- | --- |
| `causal_sandbox/` | Environments, agents, baselines and experiment drivers |
| `analysis/` | Saved-result checks, analyses and plotting |
| `configs/` | Finite-intervention configuration |
| `data/historical/` | Historical JSON results and figure/table CSV inputs |
| `data/controlled/` | Finite, Qwen and independent-agent results and metadata |
| `experiments/` | Readout and sequential protocols, records, parameter locks and analyses |
| `tests/` | Automated tests |
| `external/tdmpc2/` | Upstream TD-MPC2 source and license |

`manifest.json` records file hashes and source-path mappings. The detailed
result-to-source map, acquisition settings and recorded provenance are in
[REPRODUCTION.txt](REPRODUCTION.txt). Model weights, hidden-feature arrays and
local environments are not redistributed; fresh neural runs use the pinned
public checkpoints specified in that guide.

The bundled TD-MPC2 source retains its upstream [license](external/tdmpc2/LICENSE).
