# Mechanism Memory

Code for **Causal Retention in Interactive Agents**.

This repository contains mechanism-memory agents, matched baselines, experiment
drivers, protocols and tests. Experiments cover finite
interventions, continuous simulators, TD-MPC2 response retention, readout
filtering, sequential adaptation and Qwen hidden-state probes.

## Setup

Clone the repository and install the dependencies with Python 3.12:

```text
git clone --depth 1 https://github.com/ShengjunZhang/mechanism-memory.git
cd mechanism-memory
python -m pip install -r requirements.txt
```

The current tree contains code and configuration only. Raw observations,
saved experiment results, model weights and local Python environments are not
included. Use the shallow clone above or GitHub's **Download ZIP** to obtain
the current code without downloading data retained in earlier Git history.

## Check and Reproduce Results

Run these commands from the repository root:

```text
python reproduce.py verify
python reproduce.py tests
```

Verification checks code/configuration hashes and recorded source identities.
Tests that require saved experiment records are explicitly skipped when those
records are absent; algorithm tests generate their own inputs.

With the corresponding saved records restored at the paths in
[REPRODUCTION.txt](REPRODUCTION.txt), the following analyses are available:

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
| `experiments/` | Readout and sequential protocols, parameter locks and analysis scripts |
| `tests/` | Automated tests |
| `external/tdmpc2/` | Upstream TD-MPC2 source and license |

`manifest.json` records code/configuration hashes and source-path mappings.
Saved-result checks additionally require `data/historical/`, `data/controlled/`
and the relevant `experiments/*/results/` records, stored separately from the
current repository tree. The data-dependent commands do not substitute synthetic
results when those records are absent.

The bundled TD-MPC2 source retains its upstream [license](external/tdmpc2/LICENSE).
