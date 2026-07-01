# Causal Sandbox

This is a tiny first prototype for testing whether an agent can learn causal
action effects through an explicit loop:

```text
observe -> hypothesize -> intervene -> compare -> update
```

The first world is `door-lamp`.

- Button A makes a sound, and turns the lamp on only when the room is dark.
- Button B makes a sound, and opens the door only when the door is closed.
- Some setup actions directly change context variables, such as darkness,
  door state, and lamp state.

The goal is not to answer a causal question from text. The goal is to recover
which interventions actually change which state variables, using as few useful
experiments as possible.

## Run

```powershell
python -m causal_sandbox run --agent active --steps 20
```

Compare with a random experiment policy:

```powershell
python -m causal_sandbox run --agent random --steps 20 --seed 7
```

Compare several policies over multiple seeds:

```powershell
python -m causal_sandbox compare --steps 20 --seeds 20
```

Run the first non-hardcoded causal module:

```powershell
python -m causal_sandbox run --agent causal-core --steps 20 --seed 1
```

Freeze after exploration and test held-out interventions:

```powershell
python -m causal_sandbox intervention-test --agent causal-core --explore-steps 20 --seed 1
python -m causal_sandbox intervention-compare --explore-steps 20 --seeds 20
```

Freeze after exploration and test counterfactual edits:

```powershell
python -m causal_sandbox counterfactual-test --agent causal-core --explore-steps 20 --seed 1
python -m causal_sandbox counterfactual-compare --explore-steps 20 --seeds 20
python -m causal_sandbox counterfactual-compare --explore-steps 20 --seeds 20 --case-source all-states
```

Explore one world, freeze, and test transfer in another world:

```powershell
python -m causal_sandbox transfer-test --source-world door-lamp --target-world door-lamp-shifted --agent causal-core --explore-steps 20 --seed 1
python -m causal_sandbox transfer-compare --source-world door-lamp --target-world door-lamp-shifted --explore-steps 20 --seeds 20
python -m causal_sandbox transfer-adapt-compare --source-world door-lamp --target-world door-lamp-inverted --explore-steps 20 --adapt-steps 12 --seeds 20
```

`door-lamp-shifted` keeps the same mechanisms but changes the starting
distribution. `door-lamp-inverted` keeps the same interface but changes the
button A context rule, so it is useful for testing transfer boundaries and
target-environment adaptation.

Run the Level 3 ladder: counterfactual prediction, stable transfer, and
mechanism repair in one comparison:

```powershell
python -m causal_sandbox level3-compare --source-world door-lamp --stable-target-world door-lamp-shifted --repair-target-world door-lamp-inverted --agents causal-core passive-correlation random --explore-steps 20 --adapt-steps 12 --seeds 5 --case-source all-states
python -m causal_sandbox level3-compare --source-world panel --stable-target-world panel-shifted --repair-target-world panel-inverted --agents causal-core passive-correlation random --explore-steps 60 --adapt-steps 40 --seeds 5 --case-source history
python -m causal_sandbox repair-efficiency-compare --source-world panel --target-world panel-inverted --agents causal-core passive-correlation random --explore-steps 60 --adapt-steps 12 --seeds 10
python -m causal_sandbox reward-compare --world panel --agents causal-core reward-rl reward-transfer random passive-correlation --steps 60 --seeds 10
python -m causal_sandbox level3-compare --source-world panel --stable-target-world panel-shifted --repair-target-world panel-inverted --agents causal-core reward-rl reward-transfer passive-correlation random --explore-steps 60 --adapt-steps 12 --seeds 5 --case-source history
python -m causal_sandbox repair-efficiency-compare --source-world panel --target-world panel-inverted --agents causal-core reward-rl reward-transfer passive-correlation random --explore-steps 60 --adapt-steps 12 --seeds 5
python -m causal_sandbox level4-compare --world ambiguous-gate --agents causal-core-active causal-core reward-rl random passive-correlation active --steps 8 --seeds 20
python -m causal_sandbox level4-compare --world ambiguous-panel-gate --agents causal-core-active causal-core reward-rl random passive-correlation active --steps 12 --seeds 20
python -m causal_sandbox temporal-credit-compare --world delayed-lamp --agents causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --steps 8 --seeds 20
python -m causal_sandbox temporal-transfer-compare --source-world delayed-lamp --target-world delayed-lamp-shifted --agents causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --explore-steps 8 --seeds 20
python -m causal_sandbox temporal-adapt-compare --source-world delayed-lamp --target-world delayed-lamp-long-delay --agents causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --explore-steps 8 --adapt-steps 4 --seeds 20
python -m causal_sandbox temporal-adapt-compare --source-world delayed-lamp --target-world delayed-lamp-long-delay --agents causal-core-temporal --explore-steps 8 --adapt-steps 4 --seeds 20 --adaptation-mode fresh
python -m causal_sandbox temporal-selective-repair-compare --source-world dual-delayed-controls --target-world dual-delayed-controls-selective-shift --agents causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --explore-steps 9 --adapt-steps 6 --seeds 20
python -m causal_sandbox temporal-selective-repair-compare --source-world dual-delayed-controls --target-world dual-delayed-controls-selective-shift --agents causal-core-temporal --explore-steps 9 --adapt-steps 6 --seeds 20 --adaptation-mode fresh
python -m causal_sandbox level6-schema-compare --source-world dual-delayed-controls --target-world renamed-dual-delayed-controls --agents causal-core-temporal-portable causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --explore-steps 9 --target-steps 6 --seeds 20
python -m causal_sandbox level6-schema-compare --source-world dual-delayed-controls --target-world renamed-dual-delayed-controls --agents causal-core-temporal-portable --explore-steps 9 --target-steps 6 --seeds 20 --transfer-mode fresh
python -m causal_sandbox level6-schema-repair-compare --source-world dual-delayed-controls --target-world renamed-dual-delayed-controls-selective-shift --agents causal-core-temporal-portable causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --explore-steps 9 --target-steps 6 --seeds 20
python -m causal_sandbox level6-schema-repair-compare --source-world dual-delayed-controls --target-world renamed-dual-delayed-controls-selective-shift --agents causal-core-temporal-portable --explore-steps 9 --target-steps 6 --seeds 20 --transfer-mode fresh
python -m causal_sandbox level6-schema-repair-compare --source-world dual-delayed-controls --target-world renamed-dual-delayed-controls-selective-shift --agents causal-core-temporal-portable --explore-steps 9 --target-steps 9 --seeds 20 --transfer-mode fresh
python -m causal_sandbox level6-active-diagnostic-compare --source-world triple-delayed-controls --target-world renamed-triple-delayed-controls-diagnostic-shift --agents causal-core-temporal-diagnostic causal-core-temporal-portable causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --explore-steps 12 --repair-steps 1 --seeds 20
python -m causal_sandbox level6-active-diagnostic-compare --source-world triple-delayed-controls --target-world renamed-triple-delayed-controls-diagnostic-shift --agents causal-core-temporal-diagnostic --explore-steps 12 --repair-steps 1 --seeds 20 --transfer-mode fresh
python -m causal_sandbox level6-active-diagnostic-compare --source-world triple-delayed-controls --target-world renamed-triple-delayed-controls-diagnostic-shift --agents causal-core-temporal-portable --explore-steps 12 --repair-steps 7 --seeds 20
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-diagnostic causal-core-temporal-portable causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 1
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-diagnostic --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 1 --transfer-mode fresh
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-portable --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 10
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-diagnostic causal-core-temporal-portable causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 1 --readout-mode opaque
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-diagnostic causal-core-temporal-portable causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 1 --readout-mode semantic-confounder
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-portable --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 10 --readout-mode semantic-confounder
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-diagnostic causal-core-temporal-portable causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 1 --readout-mode noisy-opaque
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-diagnostic causal-core-temporal-portable causal-core-temporal causal-core-active causal-core reward-rl random passive-correlation --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 1 --readout-mode noisy-semantic-confounder
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-diagnostic causal-core-temporal-diagnostic-unsafe-readout causal-core-temporal-portable --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 1 --readout-mode semantic-confounder
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-diagnostic causal-core-temporal-diagnostic-unsafe-readout causal-core-temporal-portable --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 1 --readout-mode noisy-semantic-confounder
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-diagnostic --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 1 --readout-mode noisy-semantic-confounder --transfer-mode fresh
python -m causal_sandbox level6-procedural-diagnostic-compare --agents causal-core-temporal-portable --families 30 --mechanisms 3 --explore-steps 12 --repair-steps 10 --readout-mode noisy-semantic-confounder
```

Current Level 3 ladder results:

```text
door-lamp, 5 seeds:
  causal-core: cf-delta-F1=1.00, transfer-F1=1.00, repair 0.44 -> 0.83, level3=0.94
  passive-correlation: cf-delta-F1=0.32, transfer-F1=0.20, repair 0.37 -> 0.17, level3=0.23
  random: cf-delta-F1=0.96, transfer-F1=0.60, repair 0.60 -> 0.35, level3=0.64

panel, 5 seeds:
  causal-core: cf-delta-F1=1.00, transfer-F1=1.00, repair 0.53 -> 1.00, level3=1.00
  passive-correlation: cf-delta-F1=0.25, transfer-F1=0.19, repair 0.36 -> 0.42, level3=0.28
  random: cf-delta-F1=0.96, transfer-F1=0.61, repair 0.62 -> 0.57, level3=0.71

panel repair efficiency, 12 target steps, 10 seeds:
  causal-core: prior-F1=0.97, fresh-F1=0.71, F1-gain=0.27, shift-gain=0.26
  passive-correlation: prior-F1=0.19, fresh-F1=0.19, F1-gain=0.00
  random: prior-F1=0.42, fresh-F1=0.42, F1-gain=0.00

panel reward comparison, 60 steps, 10 seeds:
  reward-rl: return=90.99, graph-F1=0.90, pred-jaccard=0.17
  reward-transfer: return=90.99, graph-F1=0.90, pred-jaccard=0.17
  causal-core: return=77.90, graph-F1=1.00, pred-jaccard=0.71

panel Level 3 with reward/transfer baselines, 12 repair steps, 5 seeds:
  causal-core: level3=0.92, repair 0.53 -> 0.77
  reward-rl: level3=0.67, repair 0.58 -> 0.49
  reward-transfer: level3=0.57, repair 0.58 -> 0.19

panel repair efficiency with reward/transfer baselines, 12 target steps, 5 seeds:
  causal-core: F1-gain=0.26, shift-gain=0.23
  reward-rl: F1-gain=0.00, shift-gain=0.00
  reward-transfer: F1-gain=-0.34, shift-gain=-0.30

ambiguous-gate Level 4 active disambiguation, 8 steps, 20 seeds:
  causal-core-active: first-correct=5.00, success=1.00, level4=0.50
  causal-core: first-correct=9.00, success=0.00, level4=0.00
  reward-rl: first-correct=9.00, success=0.00, level4=0.00
  random: first-correct=9.00, success=0.00, level4=0.00
  active: first-correct=9.00, success=0.00, level4=0.00

ambiguous-panel-gate Level 4 conjunctive disambiguation, 12 steps, 20 seeds:
  causal-core-active: first-correct=8.00, success=1.00, level4=0.42
  causal-core: first-correct=13.00, success=0.00, level4=0.00
  reward-rl: first-correct=13.00, success=0.00, level4=0.00
  random: first-correct=13.00, success=0.00, level4=0.00
  active: first-correct=13.00, success=0.00, level4=0.00

delayed-lamp temporal causal credit, 8 steps, 20 seeds:
  causal-core-temporal: first-correct=2.00, success=1.00, delayed-edge=1.00, wait-misattr=0.00, temporal=0.88
  causal-core-active: first-correct=9.00, success=0.00, delayed-edge=0.00, wait-misattr=1.00, temporal=0.00
  causal-core: first-correct=9.00, success=0.00, delayed-edge=0.00, wait-misattr=1.00, temporal=0.00
  reward-rl: first-correct=9.00, success=0.00, delayed-edge=0.00, wait-misattr=1.00, temporal=0.00
  random: first-correct=9.00, success=0.00, delayed-edge=0.00, wait-misattr=0.50, temporal=0.00

delayed-lamp -> delayed-lamp-shifted temporal transfer, 8 source steps, 20 seeds:
  causal-core-temporal: target-success=1.00, delayed-edge=1.00, wait-misattr=0.00, mech-shift=0.00, transfer=1.00
  causal-core-active: target-success=0.00, delayed-edge=0.00, wait-misattr=1.00, transfer=0.00
  causal-core: target-success=0.00, delayed-edge=0.00, wait-misattr=1.00, transfer=0.00
  reward-rl: target-success=0.00, delayed-edge=0.00, wait-misattr=1.00, transfer=0.00
  random: target-success=0.00, delayed-edge=0.00, wait-misattr=0.50, transfer=0.00

delayed-lamp -> delayed-lamp-long-delay temporal repair, 4 target steps, 20 seeds:
  causal-core-temporal structural-prior: before=0.00, after=1.00, gain=1.00, delayed-edge=1.00, wait-misattr=0.00
  causal-core-temporal fresh: before=0.00, after=0.00, gain=0.00
  causal-core-active: before=0.00, after=0.00, wait-misattr=1.00
  causal-core: before=0.00, after=0.00, wait-misattr=1.00
  reward-rl: before=0.00, after=0.00

dual-delayed-controls -> selective-shift temporal selective repair, 6 target steps, 20 seeds:
  causal-core-temporal structural-prior: before=0.50, after=1.00, shifted 0.00 -> 1.00, stable-after=1.00, selective=1.00
  causal-core-temporal fresh: before=0.50, after=0.50, shifted 0.00 -> 0.00, stable-after=1.00, selective=0.00
  causal-core-active: before=0.00, after=0.00, wait-misattr=1.00
  causal-core: before=0.00, after=0.00, wait-misattr=1.00
  reward-rl: before=0.00, after=0.00

dual-delayed-controls -> renamed-dual-delayed-controls Level 6 schema transfer, 6 target steps, 20 seeds:
  causal-core-temporal-portable schema-prior: target=1.00, all-success=1.00, delayed-edge=1.00, followup-misattr=0.00, level6=1.00
  causal-core-temporal-portable fresh: target=0.50, all-success=0.00, delayed-edge=0.50, level6=0.00
  causal-core-temporal: target=0.00, all-success=0.00, followup-misattr=1.00, level6=0.00
  causal-core-active: target=0.00, all-success=0.00, followup-misattr=1.00, level6=0.00
  causal-core: target=0.00, all-success=0.00, followup-misattr=1.00, level6=0.00

dual-delayed-controls -> renamed-dual-delayed-controls-selective-shift Level 6 schema repair, 6 target steps, 20 seeds:
  causal-core-temporal-portable schema-prior: before=0.50, after=1.00, shifted 0.00 -> 1.00, stable 1.00 -> 1.00, delayed-edge=1.00, followup-misattr=0.00, l6-repair=1.00
  causal-core-temporal-portable fresh: before=0.00, after=0.50, shifted 0.00 -> 1.00, stable 0.00 -> 0.00, delayed-edge=0.50, l6-repair=0.00
  causal-core-temporal-portable fresh, 9 target steps: before=0.00, after=1.00, delayed-edge=1.00, l6-repair=1.00
  causal-core-temporal: before=0.00, after=0.00, followup-misattr=0.50, l6-repair=0.00
  causal-core-active: before=0.00, after=0.00, followup-misattr=0.50, l6-repair=0.00
  causal-core: before=0.00, after=0.00, followup-misattr=0.50, l6-repair=0.00

triple-delayed-controls -> renamed-triple-delayed-controls-diagnostic-shift Level 6 active diagnostic repair, 1 free repair step after diagnostic exposure, 20 seeds:
  causal-core-temporal-diagnostic schema-prior: before=0.67, after=1.00, shifted 0.00 -> 1.00, stable 1.00 -> 1.00, delayed-edge=1.00, followup-misattr=0.00, l6-repair=1.00
  causal-core-temporal-portable schema-prior: before=0.67, after=0.67, shifted 0.00 -> 0.00, stable 1.00 -> 1.00, l6-repair=0.00
  causal-core-temporal-diagnostic fresh: before=0.00, after=0.00, l6-repair=0.00
  causal-core-temporal-portable schema-prior, 7 free repair steps: before=0.67, after=1.00, l6-repair=1.00

procedural delayed SCM families, 30 generated families, 3 mechanisms each, 1 free repair step after diagnostic exposure:
  causal-core-temporal-diagnostic schema-prior: before=0.67, after=1.00, shifted 0.00 -> 1.00, stable 1.00 -> 1.00, delayed-edge=1.00, followup-misattr=0.00, l6-repair=1.00
  causal-core-temporal-portable schema-prior: before=0.67, after=0.67, shifted 0.00 -> 0.00, stable 1.00 -> 1.00, l6-repair=0.00
  causal-core-temporal-diagnostic fresh: before=0.00, after=0.00, l6-repair=0.00
  causal-core-temporal-portable schema-prior, 10 free repair steps: before=0.67, after=1.00, l6-repair=1.00

procedural delayed SCM families with deterministic readout observations, 30 generated families, 3 mechanisms each:
  opaque readouts, diagnostic schema-prior: before=0.67, after=1.00, shifted 0.00 -> 1.00, stable 1.00 -> 1.00, l6-repair=1.00
  opaque readouts, ordinary portable schema-prior: before=0.67, after=0.67, l6-repair=0.00; 10 free repair steps catch up to after=1.00
  semantic-role readouts, diagnostic schema-prior: before=0.67, after=1.00, shifted 0.00 -> 1.00, stable 1.00 -> 1.00, l6-repair=1.00
  semantic-role readouts, ordinary portable schema-prior: before=0.67, after=0.67, l6-repair=0.00; 10 free repair steps catch up to after=1.00

procedural delayed SCM families with noisy readout observations, 30 generated families, 3 mechanisms each:
  noisy opaque readouts, diagnostic schema-prior: before=0.67, after=1.00, shifted 0.00 -> 1.00, stable 1.00 -> 1.00, l6-repair=1.00
  noisy semantic-role readouts, diagnostic schema-prior: before=0.67, after=1.00, shifted 0.00 -> 1.00, stable 1.00 -> 1.00, l6-repair=1.00
  noisy semantic-role readouts, diagnostic fresh: before=0.00, after=0.00, l6-repair=0.00
  noisy semantic-role readouts, ordinary portable schema-prior: before=0.67, after=0.67, l6-repair=0.00; 10 free repair steps catch up to after=1.00

readout-safe schema alignment ablation, 30 generated families, 3 mechanisms each:
  opaque readouts, diagnostic unsafe-readout ablation: before=0.67, after=1.00, l6-repair=1.00
  noisy opaque readouts, diagnostic unsafe-readout ablation: before=0.67, after=1.00, l6-repair=1.00
  semantic-role readouts, diagnostic unsafe-readout ablation: before=0.00, after=0.33, l6-repair=0.00
  noisy semantic-role readouts, diagnostic unsafe-readout ablation: before=0.00, after=0.23, l6-repair=0.00
```

This ladder is the current operational definition of Level 3: the agent must
predict counterfactual deltas, reuse unchanged mechanisms under a shifted
starting distribution, and repair changed context gates after a small number of
target-environment interventions. The repair-efficiency comparison adds the
stronger sample-efficiency check: structural repair should outperform fresh
relearning under the same target-interaction budget. The reward comparison is
the current separation from RL: reward-driven agents can achieve higher task
return, but they do not produce the same counterfactual, transfer, and repair
witness.

Level 4 adds an active disambiguation requirement. In `ambiguous-gate`, the
first successful button-A trial is consistent with two hypotheses:
`dark=True` or `door_open=True` may be the context gate. `causal-core-active`
detects this ambiguity, creates a state where the two candidate gates disagree,
and presses A again. The pass condition is not graph F1 alone; it is whether
the agent reaches the correct intervention prediction on the two discriminating
challenge states within a small interaction budget. `ambiguous-panel-gate`
raises the bar from a single necessary context to a conjunctive context:
`press_c -> alarm_on` requires both `door_open=True` and `fan_on=False`.
The active agent must run two discriminating interventions, one breaking each
candidate condition, before it can predict the full challenge set.

Temporal causal credit adds a different hard case. In `delayed-lamp`,
`press_delay` does not immediately change `lamp_on`; it creates a hidden
pending effect that only becomes visible after `wait`. The pass condition is
whether the agent learns `press_delay -> lamp_on after wait`, not whether it
learns the surface correlation `wait -> lamp_on`. `causal-core-temporal`
solves this in every 20-seed run, while ordinary causal-core, active causal
core, and reward-driven baselines fail the temporal attribution check.
The shifted transfer version freezes the source-learned temporal mechanism and
tests it in `delayed-lamp-shifted`, where the target starts with
`lamp_on=True` but the delayed mechanism is unchanged. `causal-core-temporal`
keeps the correct delayed edge and reaches target-success=1.00; the baselines
again fail because their learned structure either lacks the delayed edge or
keeps the `wait -> lamp_on` misattribution.
The shifted-delay repair version changes the mechanism itself:
`press_delay -> lamp_on after wait x1` becomes `after wait x2`, and the target
also includes a decoy delayed-looking action. A stale source model fails before
adaptation. With the structural prior, `causal-core-temporal` retests the
known edge and repairs to the new temporal mechanism in 4 target steps; fresh
temporal learning does not pass within the same 4-step budget because it first
has to rule out the decoy action.
The selective-repair version raises the bar again. In `dual-delayed-controls`,
both `press_delay_alarm -> alarm_on` and `press_delay_lamp -> lamp_on` are
delayed by one wait, with a decoy delayed-looking action also available. In
`dual-delayed-controls-selective-shift`, only the lamp mechanism changes to two
waits; the alarm mechanism remains stable. The structural-prior temporal agent
keeps the stable alarm mechanism at score 1.00 and repairs the shifted lamp
mechanism from 0.00 to 1.00 within 6 target steps. Fresh temporal learning
keeps the stable mechanism but cannot repair the shifted one in the same
budget, and needs 9 target steps to catch up.
The Level 6 schema-transfer version changes the action and observation names:
`alarm_on/lamp_on/sound/wait` become `siren_active/glow_active/tone_active/settle`.
`causal-core-temporal-portable` uses a small schema-alignment module over action
descriptions and variable tokens to map the source causal mechanisms into the
target schema. With that schema prior it verifies both delayed mechanisms in 6
target steps; fresh portable temporal learning only recovers one mechanism in
the same budget and needs 9 target steps to catch up.
The Level 6 schema-repair version combines the two hard cases. The target uses
the renamed schema and also changes only the glow/lamp delay from one `settle`
to two. With a schema prior, `causal-core-temporal-portable` starts at 0.50
because the stable siren/alarm mechanism transfers but the shifted glow/lamp
mechanism is stale. After 6 target steps it reaches 1.00 while preserving the
stable mechanism, repairing the shifted one, and avoiding `settle -> glow_active`
misattribution. Fresh portable temporal learning reaches only 0.50 in the same
budget and needs 9 target steps to catch up.
The active diagnostic version adds a third delayed mechanism and makes the
target budget too small for full retesting. After a scripted exposure reveals
that the glow/lamp mechanism failed at the old one-`settle` delay,
`causal-core-temporal-diagnostic` uses one additional free target step to test
the suspected mechanism, update it to two `settle` actions, and keep the two
stable mechanisms correct. The ordinary portable temporal core has the same
schema prior but does not prioritize the failed mechanism, so it stays at 0.67
under the same one-step repair budget and needs 7 free repair steps to catch up.
The procedural SCM version repeats this structure over 30 generated source/target
families with randomized role aliases, target action names, followup names, and
shifted mechanism identity. The diagnostic agent keeps the same 1-step repair
advantage across the generated family; the ordinary portable core needs a much
larger retesting budget to close the gap.
The readout-augmented procedural version adds deterministic non-causal readout
variables to every generated family. Opaque readouts have no useful role names;
semantic-role readouts deliberately share role tokens with the true causal
variables. A readout-safe schema alignment rule now ignores source edges whose
targets have no causal role and prefers non-readout-like target variables during
role alignment. With this guard, the diagnostic result remains at Level 6 repair
score 1.00 in deterministic and noisy readout modes, including noisy semantic
readouts that share role tokens with true causal variables. The unsafe-readout
ablation shows why the guard matters: it still succeeds on opaque and noisy
opaque readouts, but drops to Level 6 repair score 0.00 on semantic-role and
noisy semantic-role readouts because schema alignment can map mechanisms onto
proxy variables rather than true causal state.

Run a noisy, mixed-observation, hidden-context robustness check:

```powershell
python -m causal_sandbox compare --world panel-noisy-hidden --agents causal-core causal-core-robust causal-core-noise-aware random passive-correlation --steps 60 --seeds 20
```

`causal-core-robust` requires repeated evidence before accepting an edge. It is
not a final solution, but it exposes the precision/recall tradeoff under noisy
observations. `causal-core-noise-aware` also compares action-conditioned change
rates against background change rates, which helps prevent false edges from
accumulating during longer noisy runs.

`causal-core` now learns both effect values and simple context gates, for example:

```text
press_a -> lamp_on when dark=True, sets lamp_on=True
```

Run an observation-focused comparison that splits false positives into
sensor/readout variables and non-sensor variables:

```powershell
python -m causal_sandbox observation-compare --world panel-derived-sensors --agents causal-core causal-core-observation-adapter causal-core-learned-observation-adapter causal-core-observation-adapter-v2 causal-core-persistent-latent-context-adapter --steps 60 --seeds 10
python -m causal_sandbox observation-compare --world panel-opaque-readouts --agents causal-core causal-core-observation-adapter causal-core-learned-observation-adapter causal-core-observation-adapter-v2 causal-core-persistent-latent-context-adapter causal-core-noise-aware random passive-correlation --steps 80 --seeds 10
python -m causal_sandbox observation-compare --world panel-noisy-core --agents causal-core causal-core-noise-aware causal-core-observation-adapter-v2 causal-core-latent-context-adapter causal-core-persistent-latent-context-adapter random passive-correlation --steps 200 --seeds 10
python -m causal_sandbox observation-compare --world panel-hidden-context --agents causal-core causal-core-hidden-context-adapter causal-core-latent-context-adapter causal-core-persistent-latent-context-adapter random passive-correlation --steps 300 --seeds 10
python -m causal_sandbox observation-compare --world panel-noisy-hidden --agents causal-core causal-core-observation-adapter-v2 causal-core-latent-context-adapter causal-core-persistent-latent-context-adapter random passive-correlation --steps 200 --seeds 5
```

`panel-derived-sensors` keeps the clean panel mechanisms but adds deterministic
sensor readouts. It isolates whether the agent can avoid treating derived
observations as causal state. `causal-core-observation-adapter` is a first
Observation Adapter v0: it filters sensor-like readouts before updating the
causal core. This is an upper-bound style sanity check for the observation
layer, not a complete solution for hidden confounding.

`panel-opaque-readouts` removes the naming hint by using variables such as
`glow`, `alert`, and `motion` instead of `*_sensor`. It tests whether the agent
can learn readout structure from observations. `causal-core-learned-observation-adapter`
learns simple composite readouts from trajectories and protects variables
referenced by action affordances. It is a first learned adapter, not a full
solution for noisy or identity-copy observations.

`causal-core-observation-adapter-v2` is the next observation prototype. It
combines a sensor prior, learned readout rules, and action-vs-background lift
filtering before updating Causal Core. `causal-core-hidden-context-adapter`
adds effect-value stability and hidden-context hypotheses for intermittent
action success/failure patterns. `causal-core-latent-context-adapter` turns
that hint into an explicit eligible-attempt latent gate: it distinguishes true
failures from normal no-ops where the target was already at the learned effect
value. `causal-core-stateful-latent-context-adapter` adds a recent-window
latent belief, uncertainty score, and cooldown-limited active probes.
`causal-core-persistent-latent-context-adapter` adds a full-history posterior,
confidence score, and recent-drift estimate.
`causal-core-proactive-latent-context-adapter` adds pre-gate active probing:
it controls visible preconditions with setter actions before retesting a
suspected hidden-gated edge. `causal-core-control-experiment-planner` adds a
control-inspired layer: direct actuator actions set state, while non-actuator
process interventions are used for mechanism identification and recursive
setup planning. It also applies a direct-actuator evidence guard so noisy
collateral observations from setter actions are not reported as new mechanisms.
`causal-core-context-search-planner` adds a targeted factorial search over
controllable setup states when an action has rare eligible successes but not
enough evidence for a latent gate. `panel-noisy-core` isolates
observation noise on true causal variables. `panel-hidden-context` isolates
unobserved context gates for `press_a -> lamp_on` and `press_c -> alarm_on`.
`panel-noisy-hidden` combines noisy core variables, named sensor readouts, and
hidden context. In the current comparison runs:

```text
panel-derived-sensors, 60 steps, 10 seeds:
  v2 F1=0.89, readout-FP=0.00
  persistent latent-context adapter F1=0.89, readout-FP=0.00

panel-opaque-readouts, 80 steps, 10 seeds:
  v2 F1=0.89, readout-FP=0.00
  persistent latent-context adapter F1=0.89, readout-FP=0.00

panel-noisy-core, 200 steps, 10 seeds:
  causal-core F1=0.75, non-readout-FP=8.90
  noise-aware F1=0.80, non-readout-FP=0.40
  v2 F1=0.87, non-readout-FP=1.40
  latent-context adapter F1=0.86, non-readout-FP=1.30
  persistent latent-context adapter F1=0.84, non-readout-FP=0.50

panel-hidden-context, 300 steps, 10 seeds:
  causal-core F1=0.97, hidden-recall=0.00, hidden-FP=0.00
  hidden-context adapter F1=0.85, hidden-recall=0.30, hidden-FP=10.60
  latent-context adapter F1=0.87, hidden-recall=0.50, hidden-FP=0.00
  persistent latent-context adapter F1=0.86, hidden-recall=0.50, hidden-FP=0.00
  random F1=0.99, hidden-recall=0.00, hidden-FP=0.00

panel-noisy-hidden, 200 steps, 5 seeds:
  causal-core F1=0.52, readout-FP=12.40, non-readout-FP=11.60, hidden-recall=0.00
  v2 F1=0.73, readout-FP=0.00, non-readout-FP=4.20, hidden-recall=0.00
  latent-context adapter F1=0.75, readout-FP=0.00, non-readout-FP=1.60, hidden-recall=0.10
  persistent latent-context adapter F1=0.77, readout-FP=0.00, non-readout-FP=1.40, hidden-recall=0.20

new proactive checks:
  panel-hidden-context, 200 steps, 10 seeds: F1=0.90, hidden-recall=0.70, hidden-FP=0.00
  panel-noisy-hidden, 200 steps, 10 seeds: F1=0.80, readout-FP=0.00, non-readout-FP=0.80, hidden-recall=0.40

complex guarded control checks:
  panel-complex-noisy-hidden, control planner, 240 steps, 5 seeds: F1=0.74, FP=3.40, readout-FP=0.00, hidden-recall=0.40, hidden-FP=0.20
  panel-complex-noisy-hidden, context-search planner, 240 steps, 5 seeds: F1=0.77, FP=2.60, readout-FP=0.00, hidden-recall=0.35, hidden-FP=0.00

procedural noisy-hidden families:
  10 families, 2 mechanisms, 2 visible contexts, 100 steps, 2 seeds:
    proactive hidden-recall=0.48, hidden-FP=0.00
    control hidden-recall=0.62, hidden-FP=0.00, F1=0.79
    context-search hidden-recall=0.60, hidden-FP=0.00, F1=0.78
  5 families, 3 mechanisms, 3 visible contexts, 160 steps, 1 seed:
    control F1=0.76, hidden-recall=0.47, FP=0.20
    context-search F1=0.78, hidden-recall=0.53, FP=0.00
  wider procedural family after budgeted readout/formula search, 2 families, 4 mechanisms, 5 visible contexts, 6 readouts, 1 seed:
    80 steps: control/context-search F1=0.70, hidden-recall=0.125, readout-FP=0.00
    120 steps: control F1=0.67, hidden-recall=0.125; context-search F1=0.68, hidden-recall=0.25; readout-FP=0.00
    160 steps: control F1=0.74, hidden-recall=0.375; context-search F1=0.71, hidden-recall=0.50; readout-FP=0.00
  expanded wide procedural checks, 4 mechanisms, 5 visible contexts, 6 readouts, 160 steps:
    4 families x 2 seeds: control F1=0.76, hidden-recall=0.469, hidden-FP=0.00; context-search F1=0.77, hidden-recall=0.531, hidden-FP=0.00; readout-FP=0.00
    6 families x 1 seed: control F1=0.75, hidden-recall=0.417, hidden-FP=0.00; context-search F1=0.77, hidden-recall=0.583, hidden-FP=0.00; readout-FP=0.00
    8 families x 2 seeds: control F1=0.78, hidden-recall=0.438, hidden-FP=0.00; context-search F1=0.76, hidden-recall=0.531, hidden-FP=0.00; readout-FP=0.00
    family-level split in the 8-family run: context-search hidden recall is higher on 6 families, equal on 1, lower on 1
    hidden-label guard ablation: on family 3, guarded and unguarded context-search both reach F1=0.783 and hidden-recall=0.75, but hidden-FP falls from 2 to 0; on family 6, both reach F1=0.727 and hidden-recall=0.50, but hidden-FP again falls from 2 to 0
```

The formal v2 experiment package adds step sweeps, a hidden-context curve, a
noisy-hidden ablation, and a focused LLM check. Reproduce the main non-LLM
sweep with:

```powershell
python -m causal_sandbox observation-sweep --worlds panel-noisy-core panel-hidden-context panel-noisy-hidden --agents causal-core causal-core-observation-adapter-v2 causal-core-persistent-latent-context-adapter random --steps-list 60 120 200 --seeds 10 --save-json tmp/observation_sweep_main.json
```

The 200-step, 10-seed main result strengthens the earlier picture:

```text
panel-noisy-core:
  causal-core F1=0.75, FP=8.90
  v2 F1=0.87, FP=1.40
  persistent latent F1=0.84, FP=0.50

panel-hidden-context:
  causal-core F1=0.97, hidden-recall=0.00
  persistent latent F1=0.88, hidden-recall=0.40, hidden-FP=0.00
  random F1=0.98, hidden-recall=0.00

panel-noisy-hidden:
  causal-core F1=0.53, readout-FP=12.60, non-readout-FP=11.70
  v2 F1=0.79, readout-FP=0.00, non-readout-FP=3.10
  persistent latent F1=0.77, readout-FP=0.00, non-readout-FP=1.10, hidden-recall=0.25
```

The hidden-context curve shows why hidden metrics are necessary:

```text
persistent latent on panel-hidden-context, 10 seeds:
  60 steps: hidden-recall=0.00, hidden-FP=0.00
 120 steps: hidden-recall=0.15, hidden-FP=0.00
 200 steps: hidden-recall=0.40, hidden-FP=0.00
 300 steps: hidden-recall=0.50, hidden-FP=0.00

proactive latent on panel-hidden-context, 10 seeds:
 120 steps: hidden-recall=0.55, hidden-FP=0.00
 200 steps: hidden-recall=0.70, hidden-FP=0.00
```

The noisy-hidden ablation at 200 steps shows the mechanism contributions:
noise-aware filtering reduces non-readout false positives but leaves readout
false positives; v2 removes readout false positives; hidden-context hints are
too aggressive; latent/stateful/persistent variants keep hidden false positives
low but do not yet solve mixed noisy-hidden hidden recall. Full commands and
tables are in `docs/experiment_package_v2.md`.

The newer `causal-core-proactive-latent-context-adapter` adds active pre-gate
probing: when an edge looks intermittently causal, it first controls visible
preconditions with available setter actions and then retests the suspected
edge. On `panel-noisy-hidden` at 200 steps and 10 seeds it improves over the
persistent latent adapter from F1=0.77 to F1=0.80 and from hidden-recall=0.25
to hidden-recall=0.40 while keeping readout-FP=0.00.

We also added a larger `panel-complex-noisy-hidden` world with 12 core
variables, 25 actions, 6 noisy readout sensors, noisy true variables, and four
hidden-gated mechanisms:

```powershell
python -m causal_sandbox observation-sweep --worlds panel-complex-noisy-hidden --agents causal-core causal-core-observation-adapter-v2 causal-core-stateful-latent-context-adapter causal-core-persistent-latent-context-adapter random passive-correlation --steps-list 80 160 --seeds 5 --save-json tmp/complex_noisy_hidden_sweep.json
```

```text
panel-complex-noisy-hidden, 160 steps, 5 seeds:
  causal-core F1=0.25, readout-FP=62.40, non-readout-FP=104.60
  v2 F1=0.52, readout-FP=0.00, non-readout-FP=34.80
  stateful latent F1=0.61, readout-FP=0.00, non-readout-FP=13.20
  persistent latent F1=0.61, readout-FP=0.00, non-readout-FP=13.40
  proactive latent F1=0.64, readout-FP=0.00, hidden-recall=0.20
  control experiment planner F1=0.64, readout-FP=0.00, hidden-recall=0.45
```

The complex result strengthens the non-mystical contribution: the module does
not "understand causality" in a broad sense, but it substantially reduces
spurious causal memory under larger corrupted action-observation surfaces.
Hidden-context recovery remains the hard unsolved part, although proactive
probing and the control-inspired planner start to move the needle. The control
planner's gain is specifically in hidden-gate recognition, not ordinary graph
F1.

This means v2 now handles the first observation boundary: readouts and named
sensors no longer enter causal memory as action targets. Persistent latent
inference preserves that behavior in derived and opaque readout settings. The
new `panel-noisy-core` result shows that noise on true causal variables is a
different failure mode: raw causal memory accumulates 8.90 non-readout false
positives, while v2 raises F1 to 0.87 and persistent latent inference reduces
non-readout false positives to 0.50. The new `panel-hidden-context` result also
shows why graph F1 is not enough: causal-core and random can have high graph F1
without identifying hidden gates. The latent adapters recover half of the true
hidden-gated edges with no hidden false positives, while the simpler
hidden-context hint adapter overgeneralizes to many ordinary edges. The mixed
`panel-noisy-hidden` setting remains the hardest boundary: persistent latent
inference removes sensor/readout false positives and improves F1 to 0.77, but
hidden-recall remains only 0.25. Proactive latent probing raises that to 0.40
and F1 to 0.80, at the cost of a small remaining hidden-FP rate.

## LLM pilot

The first local LLM-as-agent pilot uses the cached
`Qwen/Qwen2.5-7B-Instruct` model from a local Hugging Face cache. It is a
short qualitative stress test, not a final paper-scale result:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-noisy-core panel-hidden-context panel-noisy-hidden --variants llm-vanilla llm-causal-prompt llm-observation-adapter llm-persistent-latent causal-core causal-core-observation-adapter-v2 causal-core-persistent-latent-context-adapter random --steps 60 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 48 --local-files-only --save-json tmp/llm_pilot_results_60.json
```

The pilot result is informative: vanilla LLM exploration gets broad recall but
many false positives under noisy observations. A causal prompt alone does not
solve this. Giving the LLM a causal module as text advice suppresses readout
false positives and improves precision, but the LLM often ignores the module
recommendation. In the 60-step run, the module-advisor override rate is about
62--82%, depending on world and module.

```text
panel-noisy-core, 60 steps, 1 seed:
  llm-vanilla F1=0.53, FP=21
  llm-causal-prompt F1=0.43, FP=22
  llm-persistent-latent F1=0.57, FP=0
  causal-core-persistent-latent-context-adapter F1=0.79, FP=2

panel-hidden-context, 60 steps, 1 seed:
  llm-vanilla F1=0.85, hidden-recall=0.00
  llm-causal-prompt F1=0.85, hidden-recall=0.00
  llm-persistent-latent F1=0.42, hidden-recall=0.00
  causal-core-persistent-latent-context-adapter F1=0.89, hidden-recall=0.00

panel-noisy-hidden, 60 steps, 1 seed:
  llm-vanilla F1=0.38, readout-FP=16, non-readout-FP=16
  llm-causal-prompt F1=0.35, readout-FP=14, non-readout-FP=18
  llm-persistent-latent F1=0.43, readout-FP=0, non-readout-FP=3
  causal-core-observation-adapter-v2 F1=0.64, readout-FP=0, non-readout-FP=2
```

The current lesson is that causal language is not causal competence. The LLM
can describe interventions, but explicit causal memory still does the hard
work of suppressing corrupted observations. The next LLM step is a
module-authoritative or tool-calling controller, rather than treating the
module as a passive text hint. Detailed notes are in
`docs/llm_pilot_notes.md`.

The control-planner LLM check on `panel-complex-noisy-hidden` repeats the
same safety pattern. At 20 steps, vanilla Qwen records 44 false positives,
including 22 readout false positives. The persistent/proactive/control-planner
gated stacks record no readout false positives but are conservative. At 50
steps, the gated stacks begin forming filtered edges; `llm-control-planner-gated`
has the fewest non-readout false positives in that run, but hidden-recall is
still 0.00. This is an integration result, not yet an LLM hidden-causality
result.

The 100-step context-search LLM pilot is cleaner but still preliminary. On one
`panel-complex-noisy-hidden` seed, `llm-context-search-gated` reaches F1=0.72
with FP=1.00, readout-FP=0.00, hidden-recall=0.25, and hidden-FP=0.00. The
control-planner gated LLM reaches F1=0.76 but with FP=8.00 and hidden-FP=1.00.
This supports the action-loop safety story: LLM proposals can raise ordinary
recall, but the causal gate determines whether memory stays clean.

The next control-integrated pilot adds `llm-persistent-latent-gated`, where a
causal gate can veto unsafe LLM proposals, and
`llm-persistent-latent-authoritative`, where the causal module controls the
action and the LLM supplies only the surrounding rationale:

```powershell
python -m causal_sandbox llm-pilot --worlds panel-noisy-core panel-hidden-context panel-noisy-hidden --variants llm-persistent-latent-gated llm-persistent-latent-authoritative causal-core-persistent-latent-context-adapter --steps 60 --seeds 1 --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 48 --local-files-only --save-json tmp/llm_pilot_control_results_60.json
```

```text
panel-noisy-core, 60 steps, 1 seed:
  llm-persistent-latent advisor F1=0.57, recall=0.40, FP=0
  llm-persistent-latent-gated F1=0.74, recall=0.67, FP=2
  llm-persistent-latent-authoritative F1=0.76, recall=0.73, FP=3
  pure persistent module F1=0.79, recall=0.73, FP=2

panel-hidden-context, 60 steps, 1 seed:
  llm-persistent-latent advisor F1=0.42, recall=0.27
  llm-persistent-latent-gated F1=0.89, recall=0.80
  llm-persistent-latent-authoritative F1=0.89, recall=0.80
  pure persistent module F1=0.89, recall=0.80

panel-noisy-hidden, 60 steps, 1 seed:
  llm-persistent-latent advisor F1=0.43, recall=0.33, readout-FP=0
  llm-persistent-latent-gated F1=0.56, recall=0.47, readout-FP=0
  llm-persistent-latent-authoritative F1=0.59, recall=0.53, readout-FP=0
  pure persistent module F1=0.61, recall=0.47, readout-FP=0
```

This turns the first LLM pilot into a cleaner experimental point: an LLM can be
part of the agent, but the causal module must be behaviorally binding. Prompt
advice improves language, while gating/control improves causal action.

Machine-readable output:

```powershell
python -m causal_sandbox run --agent active --steps 20 --json
```

## Test

```powershell
python -m unittest discover -s tests
```

## Next steps

- Learn noisy sensor grouping and background rates instead of relying on
  suffix-based or deterministic readout rules.
- Calibrate the persistent latent posterior and probe policy so the agent can
  choose between precision-first and recall-balanced causal hypotheses.
- Extend active causal experiment design from clean symbolic gate and
  repaired delayed-effect settings to noisy-hidden, multi-step temporal, and
  visual settings.
- Make the active policy learn intervention affordances instead of relying on
  simple action-name priors.
- Add an LLM agent only after the symbolic active causal module is stronger.
