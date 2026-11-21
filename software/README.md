# Mechanism Memory

Algorithms, experiment drivers, analysis scripts and tests.

## Setup

Python 3.12 was used for the experiments.

```text
python -m pip install -r requirements.txt
```

For saved-result analysis, extract the companion `data.zip` into the same
directory as `code.zip`. The data bundle supplies the numerical records,
source checksums and detailed `REPRODUCTION.txt` instructions.

```text
python reproduce.py verify
python reproduce.py validate
python reproduce.py tests
```

## Experiments

```text
python reproduce.py finite --output outputs/finite
python reproduce.py world-model --output outputs/world_model --device cuda:0 --pulse 0.25
python reproduce.py llm --fresh --gpu 0 --output-root outputs/qwen
python reproduce.py readout --phase development
python reproduce.py readout --phase confirmation
python reproduce.py sequential --phase development --device cuda:0
python reproduce.py sequential --phase confirmation --device cuda:0
```

Use a separate directory for fresh acquisition. The finite, world-model and
Qwen commands support new output paths. Readout commands also require the
protocol locks and source manifest from `data.zip`; the listed readout and
sequential commands reuse saved records when present. TD-MPC2 and Qwen require
their documented checkpoints. Weights and local environments are not included.
Run a command with `--help` for its options.

## Saved Results

```text
python reproduce.py map
python reproduce.py figures --output ../reproduced_figures
python reproduce.py answer-risk
python reproduce.py readout-analysis
python reproduce.py sequential-analysis
python reproduce.py sequential-figure
```

The figure output directory must be new and outside the extraction directory.
Analysis commands recompute statistics from saved records without model
training. They can rewrite derived summaries; run `verify` before analysis.

## Contents

- `causal_sandbox/`: algorithms, environments and experiment drivers.
- `experiments/`: experiment configurations and analysis entry points.
- `analysis/`: saved-result analysis and plotting scripts.
- `configs/`: finite-intervention configuration.
- `tests/`: automated tests.
- `external/tdmpc2/`: upstream code, configuration and license.
- `reproduce.py`: command-line entry point.

Experiment modules are `finite_interventions`, `llm_state_probes`,
`mechanism_memory` and `world_model_probes`. The data manifest records their
old and current source hashes and the exact naming changes. Saved-record
checks verify this correspondence; further source edits are rejected.
The included TD-MPC2 files retain their original license and notices.
