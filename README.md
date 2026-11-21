# Mechanism Memory

Code and data for **Causal Retention in Interactive Agents**.

This repository contains mechanism-memory agents, matched baselines, experiment
drivers, saved observations and evaluation results. Experiments cover finite
interventions, continuous simulators, TD-MPC2 response retention, readout
filtering, sequential adaptation and Qwen hidden-state probes.

## Setup

Clone the repository and install the dependencies with Python 3.12:

```text
git clone https://github.com/ShengjunZhang/mechanism-memory.git
cd mechanism-memory
python -m pip install -r requirements.txt
```

Code and data are included directly in the repository. No separate download is
required for saved-result analysis. The checked-out data occupy about 1.5 GiB;
model weights and local Python environments are not included.

## Check and Reproduce Results

Run these commands from the repository root:

```text
python reproduce.py verify
python reproduce.py validate
python reproduce.py tests
python reproduce.py figures --output ../reproduced_figures
```

The figure output directory must be new and outside the checkout. Verification
checks file hashes and recorded source identities. Validation checks saved
numerical results; figure generation reads those results without fitting models.

Additional analyses:

```text
python reproduce.py map
python reproduce.py answer-risk
python reproduce.py readout-analysis
python reproduce.py sequential-analysis
python reproduce.py sequential-figure
```

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
| `data/historical/` | Original finite, continuous, TD-MPC2 and Qwen results |
| `data/controlled/` | Controlled experiments and independent-checkpoint results |
| `experiments/` | Readout and sequential protocols, records and analyses |
| `tests/` | Automated tests |
| `external/tdmpc2/` | Upstream TD-MPC2 source and license |

Raw numerical records are unchanged. `manifest.json` records their hashes and
the path changes needed to check older experiment records. Historical names
inside those records remain as provenance; executable paths use the layout above.

The bundled TD-MPC2 source retains its upstream [license](external/tdmpc2/LICENSE).
