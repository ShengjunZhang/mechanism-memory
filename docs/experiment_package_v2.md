# Experiment Package v2

Date: 2026-06-24

This note consolidates the current experiment package after adding noisy true
causal variables, hidden context, proactive latent-context probing, and LLM
control integration.

## Reproducible Commands

Main observation sweep:

```powershell
python -m causal_sandbox observation-sweep --worlds panel-noisy-core panel-hidden-context panel-noisy-hidden --agents causal-core causal-core-observation-adapter-v2 causal-core-persistent-latent-context-adapter random --steps-list 60 120 200 --seeds 10 --save-json tmp/observation_sweep_main.json
```

Hidden-context 300-step diagnostic:

```powershell
python -m causal_sandbox observation-sweep --worlds panel-hidden-context --agents causal-core causal-core-hidden-context-adapter causal-core-latent-context-adapter causal-core-persistent-latent-context-adapter random --steps-list 300 --seeds 10 --save-json tmp/hidden_context_300.json
```

Proactive latent-context diagnostic:

```powershell
python -m causal_sandbox observation-sweep --worlds panel-hidden-context --agents causal-core-persistent-latent-context-adapter causal-core-proactive-latent-context-adapter --steps-list 120 200 --seeds 10 --save-json tmp/proactive_hidden_key_10seed.json
```

Noisy-hidden ablation:

```powershell
python -m causal_sandbox observation-sweep --worlds panel-noisy-hidden --agents causal-core causal-core-noise-aware causal-core-observation-adapter-v2 causal-core-hidden-context-adapter causal-core-latent-context-adapter causal-core-stateful-latent-context-adapter causal-core-persistent-latent-context-adapter random passive-correlation --steps-list 200 --seeds 10 --save-json tmp/noisy_hidden_ablation_200.json
```

Proactive noisy-hidden check:

```powershell
python -m causal_sandbox observation-sweep --worlds panel-noisy-hidden --agents causal-core-observation-adapter-v2 causal-core-persistent-latent-context-adapter causal-core-proactive-latent-context-adapter --steps-list 200 --seeds 10 --save-json tmp/proactive_noisy_hidden_200.json
```

Focused real-LLM check:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-noisy-hidden --variants llm-vanilla llm-persistent-latent llm-persistent-latent-gated causal-core-persistent-latent-context-adapter --steps 30 --seeds 2 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 40 --local-files-only --save-json tmp/llm_noisy_hidden_focus_30x2.json
```

Complex noisy-hidden scenario:

```powershell
python -m causal_sandbox observation-sweep --worlds panel-complex-noisy-hidden --agents causal-core causal-core-observation-adapter-v2 causal-core-stateful-latent-context-adapter causal-core-persistent-latent-context-adapter random passive-correlation --steps-list 80 160 --seeds 5 --save-json tmp/complex_noisy_hidden_sweep.json
```

Complex proactive and control-inspired checks:

```powershell
python -m causal_sandbox observation-sweep --worlds panel-complex-noisy-hidden --agents causal-core-persistent-latent-context-adapter causal-core-proactive-latent-context-adapter causal-core-control-experiment-planner --steps-list 160 --seeds 5 --save-json tmp/control_complex_160_5seed.json
python -m causal_sandbox observation-sweep --worlds panel-complex-noisy-hidden --agents causal-core-control-experiment-planner causal-core-context-search-planner --steps-list 240 --seeds 5 --save-json tmp/control_vs_context_lift_priority_complex_240_5seed_hidden_guard_v5_sweep.json
```

Procedural noisy-hidden generalization checks:

```powershell
python -m causal_sandbox procedural-hidden-sweep --families 10 --steps 100 --seeds 2 --mechanisms 2 --visible 2 --readouts 2 --noise 0.02 --agents causal-core-persistent-latent-context-adapter causal-core-proactive-latent-context-adapter causal-core-control-experiment-planner causal-core-context-search-planner random passive-correlation --save-json tmp/procedural_hidden_micro_10fam_2seed.json
python -m causal_sandbox procedural-hidden-sweep --families 5 --steps 160 --seeds 1 --mechanisms 3 --visible 3 --readouts 3 --noise 0.02 --agents causal-core-control-experiment-planner causal-core-context-search-planner --save-json tmp/procedural_hidden_small_5fam_160_1seed.json
python -m causal_sandbox procedural-hidden-sweep --families 2 --steps 160 --seeds 1 --mechanisms 4 --visible 5 --readouts 6 --noise 0.02 --agents causal-core-control-experiment-planner causal-core-context-search-planner --save-json tmp/procedural_hidden_wide_160_budgeted.json
python -m causal_sandbox procedural-hidden-sweep --families 4 --steps 160 --seeds 2 --mechanisms 4 --visible 5 --readouts 6 --noise 0.02 --agents causal-core-control-experiment-planner causal-core-context-search-planner --save-json tmp/procedural_hidden_wide_4fam_2seed_160_hidden_guard_v5.json
python -m causal_sandbox procedural-hidden-sweep --families 6 --steps 160 --seeds 1 --mechanisms 4 --visible 5 --readouts 6 --noise 0.02 --agents causal-core-control-experiment-planner causal-core-context-search-planner --save-json tmp/procedural_hidden_wide_6fam_1seed_160_hidden_guard_v5.json
python -m causal_sandbox procedural-hidden-sweep --families 8 --steps 160 --seeds 2 --mechanisms 4 --visible 5 --readouts 6 --noise 0.02 --agents causal-core-control-experiment-planner causal-core-context-search-planner --save-json tmp/procedural_hidden_wide_8fam_2seed_160_hidden_guard_v6.json
python -m causal_sandbox procedural-hidden-sweep --family-start 3 --families 1 --steps 160 --seeds 1 --mechanisms 4 --visible 5 --readouts 6 --noise 0.02 --agents causal-core-context-search-planner-unguarded causal-core-context-search-planner --save-json tmp/procedural_hidden_guard_ablation_family3_160_v2.json
python -m causal_sandbox procedural-hidden-sweep --family-start 6 --families 1 --steps 160 --seeds 1 --mechanisms 4 --visible 5 --readouts 6 --noise 0.02 --agents causal-core-context-search-planner-unguarded causal-core-context-search-planner --save-json tmp/procedural_hidden_guard_ablation_family6_160_v2.json
```

Complex real-LLM stress checks:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-complex-noisy-hidden --variants llm-vanilla llm-persistent-latent-gated causal-core-persistent-latent-context-adapter --steps 20 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 40 --local-files-only --save-json tmp/llm_complex_focus_20x1.json
python -m causal_sandbox llm-pilot --worlds panel-complex-noisy-hidden --variants llm-persistent-latent-gated llm-persistent-latent-authoritative causal-core-persistent-latent-context-adapter --steps 50 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 40 --local-files-only --save-json tmp/llm_complex_module_50x1.json
```

Control-planner LLM checks:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-complex-noisy-hidden --variants llm-vanilla llm-persistent-latent-gated llm-proactive-latent-gated llm-control-planner-gated causal-core-control-experiment-planner --steps 20 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 32 --local-files-only --save-json tmp/llm_control_planner_complex_20x1.json
python -m causal_sandbox llm-pilot --worlds panel-complex-noisy-hidden --variants llm-persistent-latent-gated llm-proactive-latent-gated llm-control-planner-gated causal-core-control-experiment-planner --steps 50 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 32 --local-files-only --save-json tmp/llm_control_planner_complex_50x1.json
python -m causal_sandbox llm-pilot --worlds panel-complex-noisy-hidden --variants llm-control-planner-gated llm-context-search-gated causal-core-context-search-planner --steps 100 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 24 --local-files-only --save-json tmp/llm_context_search_complex_100x1.json
```

## Main Observation Results

All rows below use 200 steps and 10 seeds.

| world | agent | precision | recall | F1 | FP | readout FP | non-readout FP | hidden recall | hidden FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `panel-noisy-core` | `causal-core` | 0.62 | 0.94 | 0.75 | 8.90 | 0.00 | 8.90 | 0.00 | 0.00 |
| `panel-noisy-core` | `causal-core-observation-adapter-v2` | 0.90 | 0.83 | 0.87 | 1.40 | 0.00 | 1.40 | 0.00 | 0.00 |
| `panel-noisy-core` | `causal-core-persistent-latent-context-adapter` | 0.96 | 0.75 | 0.84 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 |
| `panel-noisy-core` | `random` | 0.25 | 0.99 | 0.39 | 45.70 | 0.00 | 45.70 | 0.00 | 0.00 |
| `panel-hidden-context` | `causal-core` | 1.00 | 0.94 | 0.97 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `panel-hidden-context` | `causal-core-observation-adapter-v2` | 1.00 | 0.80 | 0.89 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `panel-hidden-context` | `causal-core-persistent-latent-context-adapter` | 1.00 | 0.79 | 0.88 | 0.00 | 0.00 | 0.00 | 0.40 | 0.00 |
| `panel-hidden-context` | `causal-core-proactive-latent-context-adapter` | 1.00 | 0.81 | 0.90 | 0.00 | 0.00 | 0.00 | 0.70 | 0.00 |
| `panel-hidden-context` | `random` | 1.00 | 0.96 | 0.98 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `panel-noisy-hidden` | `causal-core` | 0.37 | 0.93 | 0.53 | 24.30 | 12.60 | 11.70 | 0.00 | 0.00 |
| `panel-noisy-hidden` | `causal-core-observation-adapter-v2` | 0.80 | 0.78 | 0.79 | 3.10 | 0.00 | 3.10 | 0.00 | 0.00 |
| `panel-noisy-hidden` | `causal-core-persistent-latent-context-adapter` | 0.91 | 0.68 | 0.77 | 1.10 | 0.00 | 1.10 | 0.25 | 0.70 |
| `panel-noisy-hidden` | `causal-core-proactive-latent-context-adapter` | 0.93 | 0.71 | 0.80 | 0.80 | 0.00 | 0.80 | 0.40 | 0.60 |
| `panel-noisy-hidden` | `random` | 0.16 | 0.97 | 0.28 | 76.10 | 30.20 | 45.90 | 0.00 | 0.00 |

## Hidden-Context Curve

Persistent latent-context adapter on `panel-hidden-context`, 10 seeds:

| steps | F1 | hidden recall | hidden FP |
|---:|---:|---:|---:|
| 60 | 0.88 | 0.00 | 0.00 |
| 120 | 0.87 | 0.15 | 0.00 |
| 200 | 0.88 | 0.40 | 0.00 |
| 300 | 0.86 | 0.50 | 0.00 |

Proactive latent-context adapter on `panel-hidden-context`, 10 seeds:

| steps | F1 | hidden recall | hidden FP |
|---:|---:|---:|---:|
| 120 | 0.89 | 0.55 | 0.00 |
| 200 | 0.90 | 0.70 | 0.00 |

This is the cleanest evidence that graph F1 is insufficient. Graph F1 stays
high for agents that never mark hidden context. The hidden-recall metric is the
actual diagnostic for latent insufficiency. The proactive adapter improves
hidden-context recognition by actively controlling visible preconditions before
testing a suspected hidden-gated edge.

## Noisy-Hidden Ablation

`panel-noisy-hidden`, 200 steps, 10 seeds:

| agent | precision | recall | F1 | FP | readout FP | non-readout FP | hidden recall | hidden FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `causal-core` | 0.37 | 0.93 | 0.53 | 24.30 | 12.60 | 11.70 | 0.00 | 0.00 |
| `causal-core-noise-aware` | 0.56 | 0.66 | 0.60 | 8.10 | 7.20 | 0.90 | 0.00 | 0.00 |
| `causal-core-observation-adapter-v2` | 0.80 | 0.78 | 0.79 | 3.10 | 0.00 | 3.10 | 0.00 | 0.00 |
| `causal-core-hidden-context-adapter` | 0.87 | 0.74 | 0.79 | 1.80 | 0.00 | 1.80 | 0.30 | 12.00 |
| `causal-core-latent-context-adapter` | 0.87 | 0.67 | 0.75 | 1.50 | 0.00 | 1.50 | 0.20 | 1.30 |
| `causal-core-stateful-latent-context-adapter` | 0.93 | 0.65 | 0.76 | 0.80 | 0.00 | 0.80 | 0.20 | 0.50 |
| `causal-core-persistent-latent-context-adapter` | 0.91 | 0.68 | 0.77 | 1.10 | 0.00 | 1.10 | 0.25 | 0.70 |
| `causal-core-proactive-latent-context-adapter` | 0.93 | 0.71 | 0.80 | 0.80 | 0.00 | 0.80 | 0.40 | 0.60 |

Interpretation:

- Readout filtering is essential: raw causal-core learns many sensor/readout
  false edges.
- Noise-aware filtering reduces non-readout false positives, but still keeps
  readout false positives.
- Observation adapter v2 eliminates readout false positives and gives the best
  overall F1 in this budget.
- Hidden-context hinting is too aggressive: it detects some true hidden gates
  but creates many hidden false positives.
- Latent/stateful/persistent adapters are more conservative. They keep hidden
  false positives low but still do not fully solve hidden recall in the mixed
  noisy-hidden world.
- Proactive latent probing improves hidden recall and F1 without reintroducing
  readout false positives. It is still not solved: noisy hidden-context
  false positives remain nonzero.

## Focused LLM Check

`panel-noisy-hidden`, Qwen2.5-7B-Instruct, 30 steps, 2 seeds:

| agent | precision | recall | F1 | FP | readout FP | non-readout FP |
|---|---:|---:|---:|---:|---:|---:|
| `llm-vanilla` | 0.31 | 0.70 | 0.43 | 23.00 | 12.00 | 11.00 |
| `llm-persistent-latent` | 0.62 | 0.27 | 0.37 | 2.50 | 0.00 | 2.50 |
| `llm-persistent-latent-gated` | 0.52 | 0.37 | 0.43 | 5.00 | 0.00 | 5.00 |
| pure persistent module | 0.53 | 0.40 | 0.45 | 5.50 | 0.00 | 5.50 |

The short LLM check confirms the qualitative pilot: vanilla LLM exploration
has broad recall but learns many corrupted observation edges. Adding explicit
causal memory removes readout false positives. The gated/controller direction
is promising, but real LLM multi-seed sweeps are expensive and should remain a
focused stress test until the generation loop is batched or replaced with a
tool/replay protocol.

## Complex Noisy-Hidden Scenario

We added `panel-complex-noisy-hidden`, a larger symbolic SCM with 12 core
variables, 25 actions, 6 noisy readout sensors, noisy true causal variables,
and four hidden-gated mechanisms. It is deliberately still controlled, but it
has a much larger action-effect surface than the original panel.

`panel-complex-noisy-hidden`, 5 seeds:

| steps | agent | precision | recall | F1 | FP | readout FP | non-readout FP | hidden recall | hidden FP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | `causal-core` | 0.17 | 0.85 | 0.28 | 128.40 | 53.80 | 74.60 | 0.00 | 0.00 |
| 80 | `causal-core-observation-adapter-v2` | 0.31 | 0.76 | 0.44 | 53.40 | 0.00 | 53.40 | 0.00 | 0.00 |
| 80 | `causal-core-persistent-latent-context-adapter` | 0.31 | 0.74 | 0.44 | 51.40 | 0.00 | 51.40 | 0.00 | 0.00 |
| 160 | `causal-core` | 0.14 | 0.90 | 0.25 | 167.00 | 62.40 | 104.60 | 0.00 | 0.00 |
| 160 | `causal-core-observation-adapter-v2` | 0.40 | 0.74 | 0.52 | 34.80 | 0.00 | 34.80 | 0.00 | 0.00 |
| 160 | `causal-core-stateful-latent-context-adapter` | 0.60 | 0.62 | 0.61 | 13.20 | 0.00 | 13.20 | 0.00 | 0.80 |
| 160 | `causal-core-persistent-latent-context-adapter` | 0.60 | 0.63 | 0.61 | 13.40 | 0.00 | 13.40 | 0.00 | 0.80 |
| 160 | `causal-core-proactive-latent-context-adapter` | 0.62 | 0.68 | 0.64 | 13.20 | 0.00 | 13.20 | 0.20 | 1.00 |
| 160 | `causal-core-control-experiment-planner` | 0.62 | 0.67 | 0.64 | 13.80 | 0.00 | 13.80 | 0.45 | 0.80 |
| 240 | `causal-core-control-experiment-planner` + hidden-label guard | 0.87 | 0.65 | 0.74 | 3.40 | 0.00 | 3.40 | 0.40 | 0.20 |
| 240 | `causal-core-context-search-planner` + hidden-label guard | 0.89 | 0.68 | 0.77 | 2.60 | 0.00 | 2.60 | 0.35 | 0.00 |

The complex setting is a sharper stress test. Raw causal memory accumulates
large numbers of readout and non-readout false positives. Observation adapter
v2 still removes readout false positives. Stateful and persistent latent
variants become useful at 160 steps, reducing total false positives from
167.00 to about 13 while keeping readout-FP at 0.00. The proactive variant
raises hidden recall to 0.20. The control-inspired planner, which separates
direct actuator actions from process interventions and uses recursive setup
planning, raises explicit hidden recall further to 0.45. Its graph F1 remains
roughly tied with proactive, so the gain is specifically in hidden-gate
recognition rather than ordinary edge recovery. This marks the next technical
boundary.

The guarded 240-step check separates graph recovery from hidden-label
precision. The direct-actuator and hidden-label guards use action affordance
metadata already needed by the planner: direct setters are allowed to report
only their directly controlled variable, readouts and decoys cannot be marked
as hidden gates, and semantically conflicting action-target pairs are filtered.
In the fixed complex panel this makes context search the cleaner graph learner
(F1=0.77 versus 0.74, FP=2.60 versus 3.40) and removes hidden false positives
(0.00 versus 0.20), although its hidden recall is slightly lower than the
control planner (0.35 versus 0.40). The stronger hidden-recall evidence is now
the wider procedural setting below, where context search improves hidden
recall while preserving hidden-FP=0.00.

The procedural noisy-hidden checks ask whether this is merely panel tuning. In
the 10-family micro sweep (2 hidden mechanisms, 2 visible context variables,
2 readouts, 100 steps, 2 seeds), the control planner reaches F1=0.79 and
hidden recall=0.62 with hidden-FP=0.00, while context-search reaches F1=0.78
and hidden recall=0.60 with hidden-FP=0.00. Both beat proactive latent probing
on hidden recall (0.48). In the harder 5-family sweep (3 hidden mechanisms,
3 visible context variables, 3 readouts, 160 steps, 1 seed), context-search
overtakes the control planner: F1=0.78 versus 0.76, hidden recall=0.53 versus
0.47, and FP=0.00 versus 0.20. This is a first positive generalization signal,
not yet a full scaling result.

The current implementation now includes budgeted readout/formula search and
bounded causal candidate scheduling. The readout learner ranks candidate
sources before enumerating boolean formulas, which keeps opaque readout
learning while avoiding the previous all-variable combinatorial search. On a
wider procedural check (2 families, 4 hidden mechanisms, 5 visible context
variables, 6 readouts, 1 seed), a run that previously exposed a runtime
boundary now completes in about 29 seconds at 80 steps and about 136 seconds
at 160 steps on the current machine. At 160 steps, the control planner reaches
F1=0.74 and hidden recall=0.375, while context-search reaches F1=0.71 and
hidden recall=0.50; both keep readout-FP=0.00 and hidden-FP=0.00. We then
expanded the same wide setting. With 4 families and 2 episode seeds, the
control planner reaches F1=0.760, FP=1.25, hidden recall=0.469, and
hidden-FP=0.00, while context-search reaches F1=0.770, FP=0.875, hidden
recall=0.531, and hidden-FP=0.00; readout-FP remains 0.00 for both. With 6
families and 1 seed, control reaches F1=0.748, FP=1.50, hidden recall=0.417,
and hidden-FP=0.00, while context-search reaches F1=0.768, FP=1.17, hidden
recall=0.583, and hidden-FP=0.00. In the 6-family split, context-search has
higher hidden recall on 3 families, equal hidden recall on 2 families, and
lower hidden recall on 1 family. This makes the hidden-recall signal less
likely to be a single-family accident, but it is still not a formal
paper-scale scaling result.

With eight families and two episode seeds, the hidden-recall advantage remains:
control reaches F1=0.777, FP=1.31, hidden recall=0.438, and hidden-FP=0.00,
while context-search reaches F1=0.764, FP=1.38, hidden recall=0.531, and
hidden-FP=0.00. Context-search improves hidden recall on 6 families, ties on
1, and falls below control on 1. This is the clearest current evidence that
context-search is a hidden-mechanism specialist rather than merely a graph-F1
optimizer: it trades a small graph-F1 cost for better hidden-gate recovery
while keeping readout-FP and hidden-FP at 0.00.

The hidden-label guard ablation isolates the precision gain. On procedural
family 3, guarded and unguarded context-search have identical graph F1=0.783
and hidden recall=0.75, but hidden-FP falls from 2 to 0. On family 6, both
versions have graph F1=0.727 and hidden recall=0.50, but hidden-FP again
falls from 2 to 0. The guard therefore improves the mechanism witness
without reducing the recovered hidden mechanisms in these diagnostic cases.

The complex LLM stress test is consistent with the smaller one. At 20 steps,
vanilla Qwen learns 44 false positives, including 22 readout false positives.
The gated/persistent module and the pure module have not yet reached the
evidence threshold at 20 steps, so they record no causal edges. At 50 steps,
the gated LLM stack begins to form edges through balanced exploration
(`F1=0.14`, `readout-FP=0.00`), while authoritative and pure-module variants
are still too conservative under the same short horizon. This suggests a useful
future path: LLM proposals may help exploration in larger action spaces, but
they need causal gating and longer horizons to avoid corrupting memory.

The newer control-planner LLM check sharpens this result. On
`panel-complex-noisy-hidden` at 20 steps and 1 seed, vanilla Qwen again records
44 false positives, 22 of them readout false positives. The persistent,
proactive, and control-planner gated stacks record no readout false positives,
but are too conservative to form edges at 20 steps. At 50 steps, the gated
stacks begin forming filtered edges: persistent gated F1=0.11 with 5
non-readout false positives, proactive gated F1=0.06 with 3 non-readout false
positives, and control-planner gated F1=0.06 with 2 non-readout false
positives. Hidden recall remains 0.00. The immediate conclusion is therefore
integration/safety, not LLM causal competence: explicit causal modules prevent
LLM-driven readout contamination, but longer horizons or a batched LLM loop are
needed to test hidden-gate recovery.

At 100 steps and 1 seed, the context-search gated LLM gives the cleanest
integration result so far: `llm-context-search-gated` reaches F1=0.72 with
1.00 false positive, readout-FP=0.00, hidden recall=0.25, and hidden-FP=0.00.
`llm-control-planner-gated` reaches higher F1=0.76 but with 8.00 false
positives and hidden-FP=1.00. The pure context-search module is more
conservative (F1=0.62, FP=0.00). This remains a one-seed pilot; it supports
LLM-module action-loop safety, not yet robust LLM hidden-causal learning.

## Current Status

The experiment package now supports three claims:

1. A causal module can reduce false causal structure under noisy true causal
   variables and readout corruption.
2. Hidden context requires a separate metric; graph F1 alone is misleading.
3. Proactive latent probing can move hidden-gate recognition earlier by
   controlling visible preconditions before testing suspected hidden edges.
4. Control-inspired planning gives a concrete path forward: distinguish
   actuators from process interventions, reason about controllable setup
   states, and use recursive causal setup to test deeper hidden gates.
5. Context search is now cleanest on the fixed complex panel: at 240 steps it
   improves graph F1 from 0.74 to 0.77, lowers FP from 3.40 to 2.60, keeps
   readout-FP at 0.00, and removes hidden-FP, though hidden recall is 0.35
   versus 0.40 for the control planner.
6. Procedural noisy-hidden families provide the first generalization check:
   context-search remains competitive in micro families, overtakes control in
   the harder 3-mechanism setting, and keeps a hidden-recall advantage in
   expanded wide checks with 4 mechanisms, 5 visible context variables, and 6
   readouts.
7. LLM causal language is not enough. Causal memory must be integrated into
   the action loop to affect behavior.

## Guarantee Status

We do not yet have an unconditional guarantee that the agent has learned
causality in arbitrary environments. The current guarantee is conditional and
mechanism-level:

- **Hidden-label soundness:** if readouts, visible setup variables, decoys,
  transient carriers, and semantically conflicting action-target pairs are
  identifiable from metadata or learned readout filters, and if generated
  action-target schema indices are consistent, the hidden-label guard prevents
  those classes from being counted as hidden causal gates.
- **Active identifiability:** if a hidden-gated mechanism has a bounded
  controllable setup, the agent can reset the target and repeat the same
  process action under matched observed conditions. Residual success/failure
  variation after the best observed separator is then evidence for hidden
  context rather than an ordinary visible gate.
- **Finite-sample hidden-gate test:** if eligible trials have margin
  `gamma` away from the success-rate and visible-separator thresholds, then
  after `m` eligible trials the test accepts a true hidden gate with
  probability at least
  `1 - 2 exp(-2 m gamma^2) - S exp(-2 m gamma^2)`, where `S` is the number of
  observed separators considered.
- **LLM composition guarantee:** if the LLM is only an action proposer and the
  causal module is the only writer of causal memory, then proposal completeness
  plus the finite-sample theorem gives a system-level guarantee for the
  LLM-module agent. The guarantee applies to the agent's mechanism memory and
  behavior, not to the LLM parameters alone.

The experimental contribution is the finite-sample version of these
guarantees: the guard ablations show that hidden-FP falls from 2 to 0 without
reducing hidden recall in diagnostic procedural families, and the expanded
wide procedural checks show a hidden-recall advantage for context search while
keeping hidden-FP and readout-FP at 0.00 through the 8-family, 2-seed run.

The main unresolved boundary is robust hidden-context recovery in mixed and
larger noisy-hidden settings without the current runtime blow-up. The current
best trade-off is strong but still early: hidden-label guarding gives clean
mechanism witnesses, and context search gives the best wide-procedural
hidden recall, but a final module needs broader procedural scaling and a
faster candidate scheduler.
