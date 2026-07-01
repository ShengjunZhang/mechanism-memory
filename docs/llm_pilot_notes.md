# LLM Pilot Notes

Date: 2026-06-23

## Setup

Local model:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-noisy-core panel-hidden-context panel-noisy-hidden --variants llm-vanilla llm-causal-prompt llm-observation-adapter llm-persistent-latent causal-core causal-core-observation-adapter-v2 causal-core-persistent-latent-context-adapter random --steps 60 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 48 --local-files-only --save-json tmp/llm_pilot_results_60.json
```

This is a pilot, not a final paper-scale run. It uses one seed and a short
60-step horizon so that we can inspect the qualitative failure modes.

## Main Observations

1. Vanilla LLM exploration has high recall but many false positives under
   noisy observations.

2. A causal prompt alone does not fix the problem. It sometimes changes the
   action distribution, but it does not reliably suppress noisy or readout
   false structure.

3. LLM plus causal module as an advisor improves precision in noisy settings,
   especially by avoiding readout false positives, but it also reduces recall.
   The LLM frequently overrides the module recommendation.

4. The strongest non-LLM causal-module controller still beats the current
   LLM-advisor variants on the noisy-hidden world. This suggests that the
   causal module should become an action-authoritative tool or trained
   controller, not merely a text hint.

## Result Snapshot

### `panel-noisy-core`, 60 steps, 1 seed

| agent | precision | recall | F1 | false positives |
|---|---:|---:|---:|---:|
| `llm-vanilla` | 0.38 | 0.87 | 0.53 | 21 |
| `llm-causal-prompt` | 0.31 | 0.67 | 0.43 | 22 |
| `llm-observation-adapter` | 0.60 | 0.40 | 0.48 | 4 |
| `llm-persistent-latent` | 1.00 | 0.40 | 0.57 | 0 |
| `causal-core-persistent-latent-context-adapter` | 0.85 | 0.73 | 0.79 | 2 |

The LLM variants show the core precision-recall tradeoff. The persistent
latent module as an advisor removes false positives, but the module-controlled
baseline still discovers more true edges.

### `panel-hidden-context`, 60 steps, 1 seed

| agent | precision | recall | F1 | hidden recall |
|---|---:|---:|---:|---:|
| `llm-vanilla` | 1.00 | 0.73 | 0.85 | 0.00 |
| `llm-causal-prompt` | 1.00 | 0.73 | 0.85 | 0.00 |
| `llm-persistent-latent` | 1.00 | 0.27 | 0.42 | 0.00 |
| `causal-core` | 1.00 | 1.00 | 1.00 | 0.00 |
| `causal-core-persistent-latent-context-adapter` | 1.00 | 0.80 | 0.89 | 0.00 |

At 60 steps, none of the LLM variants detects the hidden context. This is
consistent with our earlier result that hidden-gate metrics require longer,
more targeted evidence than graph F1.

### `panel-noisy-hidden`, 60 steps, 1 seed

| agent | precision | recall | F1 | readout FP | non-readout FP |
|---|---:|---:|---:|---:|---:|
| `llm-vanilla` | 0.26 | 0.73 | 0.38 | 16 | 16 |
| `llm-causal-prompt` | 0.24 | 0.67 | 0.35 | 14 | 18 |
| `llm-observation-adapter` | 0.75 | 0.20 | 0.32 | 0 | 1 |
| `llm-persistent-latent` | 0.62 | 0.33 | 0.43 | 0 | 3 |
| `causal-core-observation-adapter-v2` | 0.80 | 0.53 | 0.64 | 0 | 2 |
| `causal-core-persistent-latent-context-adapter` | 0.88 | 0.47 | 0.61 | 0 | 1 |

The LLM alone is easily attracted to noisy/readout structure. The causal
module as an advisor suppresses readout false positives, but the current LLM
controller does not exploit it as well as the module-controlled baseline.

## Action Override Diagnostic

When the causal module is presented as text advice, Qwen frequently chooses a
different action from the module recommendation:

| world | `llm-observation-adapter` override rate | `llm-persistent-latent` override rate |
|---|---:|---:|
| `panel-noisy-core` | 0.82 | 0.82 |
| `panel-hidden-context` | 0.82 | 0.82 |
| `panel-noisy-hidden` | 0.82 | 0.62 |

This is the most important pilot finding. Prompting an LLM to "use causal
reasoning" is not enough; the causal module likely needs to be integrated as a
controller, tool, memory bottleneck, or trainable adapter.

## Interpretation

The pilot gives a useful contrast:

- LLMs can produce plausible causal language and diverse interventions.
- They do not automatically protect causal memory from observation corruption.
- Textual causal instructions are weaker than explicit intervention memory.
- A causal module can clean up false structure, but only if the controller
  actually follows or learns to use it.

For the paper, this supports a strong framing: causal competence is not
equivalent to verbal causal explanation. It requires intervention-stable
beliefs, explicit memory, and disciplined active experimentation.

## Next LLM Experiments

1. Add a tool-calling interface where the LLM must query `predict`,
   `known_edges`, and `recommended_probe` before acting.

2. Run longer hidden-context experiments with targeted probe budgets so that
   hidden recall can be measured fairly.

3. Train a lightweight policy or adapter to choose when to follow the causal
   module versus when to explore outside it.

## Control-Integrated Update

We next added two integration variants:

- `llm-persistent-latent-gated`: the LLM proposes an action, but a causal gate
  can veto overused, unsupported, or coverage-breaking proposals.
- `llm-persistent-latent-authoritative`: the causal module selects the action;
  the LLM supplies the surrounding agent rationale, but the module action is
  enforced.

Command:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-noisy-core panel-hidden-context panel-noisy-hidden --variants llm-persistent-latent-gated llm-persistent-latent-authoritative causal-core-persistent-latent-context-adapter --steps 60 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 48 --local-files-only --save-json tmp/llm_pilot_control_results_60.json
```

### Control Result Snapshot

| world | agent | precision | recall | F1 | false positives | readout FP |
|---|---|---:|---:|---:|---:|---:|
| `panel-noisy-core` | `llm-persistent-latent` advisor | 1.00 | 0.40 | 0.57 | 0 | 0 |
| `panel-noisy-core` | `llm-persistent-latent-gated` | 0.83 | 0.67 | 0.74 | 2 | 0 |
| `panel-noisy-core` | `llm-persistent-latent-authoritative` | 0.79 | 0.73 | 0.76 | 3 | 0 |
| `panel-noisy-core` | pure persistent module | 0.85 | 0.73 | 0.79 | 2 | 0 |
| `panel-hidden-context` | `llm-persistent-latent` advisor | 1.00 | 0.27 | 0.42 | 0 | 0 |
| `panel-hidden-context` | `llm-persistent-latent-gated` | 1.00 | 0.80 | 0.89 | 0 | 0 |
| `panel-hidden-context` | `llm-persistent-latent-authoritative` | 1.00 | 0.80 | 0.89 | 0 | 0 |
| `panel-hidden-context` | pure persistent module | 1.00 | 0.80 | 0.89 | 0 | 0 |
| `panel-noisy-hidden` | `llm-persistent-latent` advisor | 0.62 | 0.33 | 0.43 | 3 | 0 |
| `panel-noisy-hidden` | `llm-persistent-latent-gated` | 0.70 | 0.47 | 0.56 | 3 | 0 |
| `panel-noisy-hidden` | `llm-persistent-latent-authoritative` | 0.67 | 0.53 | 0.59 | 4 | 0 |
| `panel-noisy-hidden` | pure persistent module | 0.88 | 0.47 | 0.61 | 1 | 0 |

The important movement is recall. The advisor version is very conservative
because the LLM often ignores the module's information-gain probes. Gating and
authoritative control recover much of the pure module behavior while keeping
the LLM in the agent stack.

### Gate Diagnostics

For `llm-persistent-latent-gated`, the causal gate made the following decisions
over 60 steps:

| world | module match | balanced exploration | known causal action | veto less-tested module action | veto overused proposal | veto unsupported proposal |
|---|---:|---:|---:|---:|---:|---:|
| `panel-noisy-core` | 39 | 11 | 0 | 7 | 0 | 3 |
| `panel-hidden-context` | 42 | 7 | 1 | 7 | 3 | 0 |
| `panel-noisy-hidden` | 36 | 11 | 3 | 4 | 3 | 3 |

For `llm-persistent-latent-authoritative`, Qwen almost always complied with
the enforced module action: override attempts were 0/60, 0/60, and 1/60 across
the three worlds.

This strengthens the core claim: the problem is not that LLMs cannot verbalize
causal reasoning; it is that causal competence needs an explicit intervention
memory and a control interface that makes the memory behaviorally binding.

## Focused Robustness Check

A larger real-LLM multi-seed run at 60 steps is currently expensive with the
unbatched local generation loop. We therefore added a smaller real-Qwen check
on the hardest world:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-noisy-hidden --variants llm-vanilla llm-persistent-latent llm-persistent-latent-gated causal-core-persistent-latent-context-adapter --steps 30 --seeds 2 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 40 --local-files-only --save-json tmp/llm_noisy_hidden_focus_30x2.json
```

| agent | precision | recall | F1 | FP | readout FP | non-readout FP |
|---|---:|---:|---:|---:|---:|---:|
| `llm-vanilla` | 0.31 | 0.70 | 0.43 | 23.00 | 12.00 | 11.00 |
| `llm-persistent-latent` | 0.62 | 0.27 | 0.37 | 2.50 | 0.00 | 2.50 |
| `llm-persistent-latent-gated` | 0.52 | 0.37 | 0.43 | 5.00 | 0.00 | 5.00 |
| pure persistent module | 0.53 | 0.40 | 0.45 | 5.50 | 0.00 | 5.50 |

The smaller robustness check repeats the key pattern: vanilla LLM exploration
has broad recall but learns many corrupted observation edges. Explicit causal
memory removes readout false positives. Gating improves recall over the advisor
variant in the 60-step single-seed control run, but at 30 steps the main effect
is still readout protection rather than hidden-context recovery.

## Control-Planner LLM Check

After adding proactive and control-inspired experiment planners, we ran a
focused real-Qwen check on the larger `panel-complex-noisy-hidden` world. This
is intentionally a small sanity check, not a paper-scale LLM result.

Short-horizon command:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-complex-noisy-hidden --variants llm-vanilla llm-persistent-latent-gated llm-proactive-latent-gated llm-control-planner-gated causal-core-control-experiment-planner --steps 20 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 32 --local-files-only --save-json tmp/llm_control_planner_complex_20x1.json
```

Longer module-focused command:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-complex-noisy-hidden --variants llm-persistent-latent-gated llm-proactive-latent-gated llm-control-planner-gated causal-core-control-experiment-planner --steps 50 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 32 --local-files-only --save-json tmp/llm_control_planner_complex_50x1.json
```

Context-search gated command:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-complex-noisy-hidden --variants llm-control-planner-gated llm-context-search-gated causal-core-context-search-planner --steps 100 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 24 --local-files-only --save-json tmp/llm_context_search_complex_100x1.json
```

### `panel-complex-noisy-hidden`, 20 steps, 1 seed

| agent | precision | recall | F1 | FP | readout FP | non-readout FP |
|---|---:|---:|---:|---:|---:|---:|
| `llm-vanilla` | 0.19 | 0.32 | 0.24 | 44.00 | 22.00 | 22.00 |
| `llm-persistent-latent-gated` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `llm-proactive-latent-gated` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `llm-control-planner-gated` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pure control planner | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

The 20-step run separates contamination from conservative evidence thresholds.
Vanilla LLM exploration immediately writes many false causal edges, including
22 readout false positives. All module-gated variants keep readout-FP at zero,
but the horizon is too short for filtered modules to pass their evidence
thresholds.

### `panel-complex-noisy-hidden`, 50 steps, 1 seed

| agent | precision | recall | F1 | FP | readout FP | non-readout FP |
|---|---:|---:|---:|---:|---:|---:|
| `llm-persistent-latent-gated` | 0.29 | 0.06 | 0.11 | 5.00 | 0.00 | 5.00 |
| `llm-proactive-latent-gated` | 0.25 | 0.03 | 0.06 | 3.00 | 0.00 | 3.00 |
| `llm-control-planner-gated` | 0.33 | 0.03 | 0.06 | 2.00 | 0.00 | 2.00 |
| pure control planner | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

At 50 steps, gated LLM stacks start to form a small number of filtered causal
edges. The control-planner gate has the fewest non-readout false positives
among the LLM variants in this run, but hidden-gate recall is still zero. The
gate diagnostics show that the module is behaviorally active: for
`llm-control-planner-gated`, 43/50 actions matched the module recommendation,
5 proposals were vetoed in favor of a less-tested module action, and 2 were
accepted as balanced exploration.

Interpretation: this is not yet a positive LLM-hidden-causality result. It is
a useful LLM integration result: vanilla LLM exploration corrupts causal memory
quickly, while explicit causal modules prevent readout contamination and can
control the LLM action loop. Longer horizons or a lower-cost batched LLM loop
are needed before testing whether LLM proposals improve hidden-gate recovery.

### `panel-complex-noisy-hidden`, 100 steps, 1 seed

| agent | precision | recall | F1 | FP | readout FP | non-readout FP | hidden recall | hidden FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llm-control-planner-gated` | 0.75 | 0.77 | 0.76 | 8.00 | 0.00 | 8.00 | 0.25 | 1.00 |
| `llm-context-search-gated` | 0.95 | 0.58 | 0.72 | 1.00 | 0.00 | 1.00 | 0.25 | 0.00 |
| pure context-search planner | 1.00 | 0.45 | 0.62 | 0.00 | 0.00 | 0.00 | 0.25 | 0.00 |

The 100-step run is the first useful LLM/context-search integration signal.
LLM proposals improve ordinary recall over the pure module, while the
context-search gate keeps readout-FP and hidden-FP at zero in this seed.
The control-planner gate gets higher F1, but with more false positives and a
hidden false positive. This is still a one-seed pilot, so the claim is safety
and integration, not robust LLM hidden-causal learning.
