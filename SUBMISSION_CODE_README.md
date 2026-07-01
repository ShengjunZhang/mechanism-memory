# Supplementary software

This archive contains the code and result files accompanying the manuscript
`Successful agents can fail to retain causal mechanisms`.

## Contents

- `causal_sandbox/`: symbolic SCM environments, mechanism-memory agents, baselines, continuous probes, MuJoCo probe utilities and LLM proposal-policy wrappers.
- `tests/`: unit tests for the core environments and evaluation utilities.
- `scripts/make_v3_figures.py`: script used to generate the main statistical figures for the Nature Machine Intelligence version.
- `results/`: JSON outputs used by the figure script and supplementary tables.
- `docs/`: experimental notes with representative commands.

## Basic setup

```powershell
python -m pip install -e .
python -m pip install -r requirements.txt
```

The symbolic and metric-cell experiments require only the lightweight Python dependencies listed in `requirements.txt`. MuJoCo and local LLM experiments require optional local installations of MuJoCo/Gymnasium and a local Hugging Face model, respectively.

## Rebuild figures

From this directory:

```powershell
python scripts/make_v3_figures.py
```

The script reads JSON files from `results/` and writes figures to `figures/`.

## Run tests

```powershell
python -m unittest discover -s tests
```

## Representative experiment commands

The full manuscript experiments are deterministic over the reported seeds. Representative commands are listed in `docs/experiment_package_v2.md` and `docs/llm_pilot_notes.md`.
