# Data and Reproduction

Code and saved experiment records are included in the repository. Clone it
with `git clone --depth 1 https://github.com/ShengjunZhang/mechanism-memory.git`
or select **Code > Download ZIP** on GitHub, then enter the project directory.
The checked-out records occupy about 1.5 GiB.

## Data Locations

| Location | Contents |
| --- | --- |
| `data/historical/raw_json/` | Finite baselines, continuous probes, TD-MPC2 and Qwen results |
| `data/historical/SourceData_*.csv` | Historical figure and table inputs |
| `data/controlled/results/` | Finite tuple/budget results, fresh Qwen decoder results and TD-MPC2 comparisons |
| `data/controlled/experiments/` | Source snapshots, prompt bindings and analysis metadata |
| `experiments/*/protocol.json` | Experimental conditions, seeds and budgets |
| `experiments/*/*_lock.json` | Fixed source identities and development-selected settings |
| `experiments/*/results/development/` | Development records and candidate comparisons |
| `experiments/*/results/confirmation/` | Held-out confirmation records and prediction arrays |

The readout studies include all recorded candidates, conditions and ablations.
Sequential TD-MPC2 records cover the five tasks, development and confirmation
checkpoints, and the schedule/noise/budget grid. Historical result files retain
their original metadata. `manifest.json` maps those identities to current paths
and records file hashes. Section 7 of `REPRODUCTION.txt` describes the available
historical provenance and missing original-run artifacts.

## Verify and Analyze Saved Results

Install the dependencies using Python 3.12, then verify the original files
before running analyses that write derived summaries:

```text
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py validate
python reproduce.py tests
python reproduce.py map
```

`map` lists the result identifiers, source groups and selectors. Sections 2--5
of [REPRODUCTION.txt](REPRODUCTION.txt) give the complete mapping and parameters.

| Result group | Command |
| --- | --- |
| Six overview, world-model and Qwen data figures | `python reproduce.py figures --output ../reproduced_figures` |
| Complete-answer and candidate-risk summaries | `python reproduce.py answer-risk` |
| Readout-filter and memory comparisons | `python reproduce.py readout-analysis` |
| Sequential TD-MPC2 decisions and metrics | `python reproduce.py sequential-analysis` |
| Sequential TD-MPC2 figure | `python reproduce.py sequential-figure` |

The figure output directory must be new and outside the checkout. Readout and
sequential analyses write summaries into their study directories. These commands
read saved records and do not acquire interventions or train neural models.

## Generate New Records

The `finite`, `world-model`, `llm`, `readout` and `sequential` commands run the
experiment drivers. Use `python reproduce.py <command> --help` for options.
Sections H1--H9 and A1--A5 of `REPRODUCTION.txt` specify acquisition protocols,
checkpoints and seeds. Run new acquisition in a separate checkout so that the
original result records remain available for comparison.

Fresh neural runs require the specified TD-MPC2 or Qwen checkpoints and simulator
dependencies. Model weights and hidden-feature arrays are not bundled.
