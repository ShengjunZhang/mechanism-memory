from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import agent_names, make_agent
from .continuous import (
    format_continuous_metric_poc,
    run_continuous_metric_poc,
    save_continuous_payload,
)
from .core import EdgeScore
from .evaluation import (
    CounterfactualTestResult,
    InterventionTestResult,
    Level6SchemaRepairResult,
    Level3TestResult,
    Level4DisambiguationResult,
    Level6SchemaTransferResult,
    RewardEpisodeResult,
    TemporalAdaptationResult,
    TemporalCreditResult,
    TemporalSelectiveAdaptationResult,
    TemporalTransferResult,
    TransferAdaptationTestResult,
    TransferInterventionTestResult,
    format_counterfactual_test,
    format_episode,
    format_intervention_test,
    format_transfer_intervention_test,
    run_counterfactual_test,
    run_episode,
    run_intervention_test,
    run_level6_active_diagnostic_repair_test,
    run_level6_schema_repair_test,
    run_level6_schema_transfer_test,
    run_level3_test,
    run_level4_disambiguation_test,
    run_reward_episode,
    run_temporal_adaptation_test,
    run_temporal_credit_test,
    run_temporal_selective_adaptation_test,
    run_temporal_transfer_test,
    run_transfer_adaptation_test,
    run_transfer_intervention_test,
)
from .llm import LocalHFChatModel, llm_agent_names, make_llm_agent
from .mujoco_probe import (
    format_mujoco_causal_checks,
    format_mujoco_probe,
    run_mujoco_causal_checks,
    run_mujoco_probe_experiment,
)
from .ranking_baseline import (
    format_ranking_loss_baseline,
    run_ranking_loss_baseline_experiment,
)
from .stress_suite import (
    format_icml_stress_suite,
    format_matched_mechanism_baseline_suite,
    run_icml_stress_suite,
    run_matched_mechanism_baseline_suite,
    save_icml_stress_suite,
)
from .worlds import (
    make_procedural_complex_hidden_world,
    make_procedural_diagnostic_world_pair,
    make_world,
    world_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-sandbox",
        description="Run small causal-learning sandbox episodes.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="run one sandbox episode")
    run_parser.add_argument("--world", choices=world_names(), default="door-lamp")
    run_parser.add_argument("--agent", choices=agent_names(), default="active")
    run_parser.add_argument("--steps", type=int, default=20)
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--json", action="store_true")

    compare_parser = subparsers.add_parser(
        "compare", help="compare agents over several seeds"
    )
    compare_parser.add_argument("--world", choices=world_names(), default="door-lamp")
    compare_parser.add_argument(
        "--agents", nargs="+", choices=agent_names(), default=agent_names()
    )
    compare_parser.add_argument("--steps", type=int, default=20)
    compare_parser.add_argument("--seeds", type=int, default=20)

    reward_compare_parser = subparsers.add_parser(
        "reward-compare",
        help="compare reward return against mechanism-learning metrics",
    )
    reward_compare_parser.add_argument(
        "--world", choices=world_names(), default="panel"
    )
    reward_compare_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=["causal-core", "reward-rl", "reward-transfer", "random"],
    )
    reward_compare_parser.add_argument("--steps", type=int, default=60)
    reward_compare_parser.add_argument("--seeds", type=int, default=20)

    observation_compare_parser = subparsers.add_parser(
        "observation-compare",
        help="compare agents and split false positives by observation readouts",
    )
    observation_compare_parser.add_argument(
        "--world", choices=world_names(), default="panel-derived-sensors"
    )
    observation_compare_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core",
            "causal-core-noise-aware",
            "causal-core-observation-adapter",
            "random",
        ],
    )
    observation_compare_parser.add_argument("--steps", type=int, default=60)
    observation_compare_parser.add_argument("--seeds", type=int, default=20)
    observation_compare_parser.add_argument("--sensor-suffix", default="_sensor")

    test_parser = subparsers.add_parser(
        "intervention-test",
        help="freeze after exploration and test held-out interventions",
    )
    test_parser.add_argument("--world", choices=world_names(), default="door-lamp")
    test_parser.add_argument("--agent", choices=agent_names(), default="causal-core")
    test_parser.add_argument("--explore-steps", type=int, default=20)
    test_parser.add_argument("--seed", type=int, default=7)
    test_parser.add_argument("--include-seen", action="store_true")
    test_parser.add_argument("--json", action="store_true")

    test_compare_parser = subparsers.add_parser(
        "intervention-compare",
        help="compare held-out intervention prediction over several seeds",
    )
    test_compare_parser.add_argument(
        "--world", choices=world_names(), default="door-lamp"
    )
    test_compare_parser.add_argument(
        "--agents", nargs="+", choices=agent_names(), default=agent_names()
    )
    test_compare_parser.add_argument("--explore-steps", type=int, default=20)
    test_compare_parser.add_argument("--seeds", type=int, default=20)
    test_compare_parser.add_argument("--include-seen", action="store_true")

    counterfactual_parser = subparsers.add_parser(
        "counterfactual-test",
        help="freeze after exploration and test counterfactual edits",
    )
    counterfactual_parser.add_argument(
        "--world", choices=world_names(), default="door-lamp"
    )
    counterfactual_parser.add_argument(
        "--agent", choices=agent_names(), default="causal-core"
    )
    counterfactual_parser.add_argument("--explore-steps", type=int, default=20)
    counterfactual_parser.add_argument("--seed", type=int, default=7)
    counterfactual_parser.add_argument(
        "--case-source", choices=["history", "all-states"], default="history"
    )
    counterfactual_parser.add_argument("--json", action="store_true")

    counterfactual_compare_parser = subparsers.add_parser(
        "counterfactual-compare",
        help="compare counterfactual prediction over several seeds",
    )
    counterfactual_compare_parser.add_argument(
        "--world", choices=world_names(), default="door-lamp"
    )
    counterfactual_compare_parser.add_argument(
        "--agents", nargs="+", choices=agent_names(), default=agent_names()
    )
    counterfactual_compare_parser.add_argument("--explore-steps", type=int, default=20)
    counterfactual_compare_parser.add_argument("--seeds", type=int, default=20)
    counterfactual_compare_parser.add_argument(
        "--case-source", choices=["history", "all-states"], default="history"
    )

    transfer_parser = subparsers.add_parser(
        "transfer-test",
        help="explore one world, freeze, and test interventions in another world",
    )
    transfer_parser.add_argument(
        "--source-world", choices=world_names(), default="door-lamp"
    )
    transfer_parser.add_argument(
        "--target-world", choices=world_names(), default="door-lamp-shifted"
    )
    transfer_parser.add_argument("--agent", choices=agent_names(), default="causal-core")
    transfer_parser.add_argument("--explore-steps", type=int, default=20)
    transfer_parser.add_argument("--seed", type=int, default=7)
    transfer_parser.add_argument("--include-seen", action="store_true")
    transfer_parser.add_argument("--json", action="store_true")

    transfer_compare_parser = subparsers.add_parser(
        "transfer-compare",
        help="compare source-to-target intervention transfer over several seeds",
    )
    transfer_compare_parser.add_argument(
        "--source-world", choices=world_names(), default="door-lamp"
    )
    transfer_compare_parser.add_argument(
        "--target-world", choices=world_names(), default="door-lamp-shifted"
    )
    transfer_compare_parser.add_argument(
        "--agents", nargs="+", choices=agent_names(), default=agent_names()
    )
    transfer_compare_parser.add_argument("--explore-steps", type=int, default=20)
    transfer_compare_parser.add_argument("--seeds", type=int, default=20)
    transfer_compare_parser.add_argument("--include-seen", action="store_true")

    transfer_adapt_compare_parser = subparsers.add_parser(
        "transfer-adapt-compare",
        help="compare target adaptation after source-to-target transfer",
    )
    transfer_adapt_compare_parser.add_argument(
        "--source-world", choices=world_names(), default="door-lamp"
    )
    transfer_adapt_compare_parser.add_argument(
        "--target-world", choices=world_names(), default="door-lamp-inverted"
    )
    transfer_adapt_compare_parser.add_argument(
        "--agents", nargs="+", choices=agent_names(), default=agent_names()
    )
    transfer_adapt_compare_parser.add_argument("--explore-steps", type=int, default=20)
    transfer_adapt_compare_parser.add_argument("--adapt-steps", type=int, default=8)
    transfer_adapt_compare_parser.add_argument("--seeds", type=int, default=20)
    transfer_adapt_compare_parser.add_argument(
        "--adaptation-mode",
        choices=["continue", "fresh", "structural-prior"],
        default="structural-prior",
    )

    level3_compare_parser = subparsers.add_parser(
        "level3-compare",
        help=(
            "compare counterfactual prediction, stable transfer, and mechanism "
            "repair over several seeds"
        ),
    )
    level3_compare_parser.add_argument(
        "--source-world", choices=world_names(), default="door-lamp"
    )
    level3_compare_parser.add_argument(
        "--stable-target-world", choices=world_names(), default="door-lamp-shifted"
    )
    level3_compare_parser.add_argument(
        "--repair-target-world", choices=world_names(), default="door-lamp-inverted"
    )
    level3_compare_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=["causal-core", "passive-correlation", "random"],
    )
    level3_compare_parser.add_argument("--explore-steps", type=int, default=20)
    level3_compare_parser.add_argument("--adapt-steps", type=int, default=12)
    level3_compare_parser.add_argument("--seeds", type=int, default=20)
    level3_compare_parser.add_argument(
        "--case-source", choices=["history", "all-states"], default="all-states"
    )
    level3_compare_parser.add_argument(
        "--adaptation-mode",
        choices=["continue", "fresh", "structural-prior"],
        default="structural-prior",
    )

    level4_compare_parser = subparsers.add_parser(
        "level4-compare",
        help="compare active disambiguating interventions over several seeds",
    )
    level4_compare_parser.add_argument(
        "--world", choices=world_names(), default="ambiguous-gate"
    )
    level4_compare_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core-active",
            "causal-core",
            "reward-rl",
            "random",
            "passive-correlation",
        ],
    )
    level4_compare_parser.add_argument("--steps", type=int, default=8)
    level4_compare_parser.add_argument("--seeds", type=int, default=20)

    temporal_compare_parser = subparsers.add_parser(
        "temporal-credit-compare",
        help="compare temporal credit assignment for delayed effects",
    )
    temporal_compare_parser.add_argument(
        "--world", choices=world_names(), default="delayed-lamp"
    )
    temporal_compare_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core-temporal",
            "causal-core-active",
            "causal-core",
            "reward-rl",
            "random",
            "passive-correlation",
        ],
    )
    temporal_compare_parser.add_argument("--steps", type=int, default=8)
    temporal_compare_parser.add_argument("--seeds", type=int, default=20)

    temporal_transfer_parser = subparsers.add_parser(
        "temporal-transfer-compare",
        help="compare source-to-target transfer of delayed causal credit",
    )
    temporal_transfer_parser.add_argument(
        "--source-world", choices=world_names(), default="delayed-lamp"
    )
    temporal_transfer_parser.add_argument(
        "--target-world", choices=world_names(), default="delayed-lamp-shifted"
    )
    temporal_transfer_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core-temporal",
            "causal-core-active",
            "causal-core",
            "reward-rl",
            "random",
            "passive-correlation",
        ],
    )
    temporal_transfer_parser.add_argument("--explore-steps", type=int, default=8)
    temporal_transfer_parser.add_argument("--seeds", type=int, default=20)

    temporal_adapt_parser = subparsers.add_parser(
        "temporal-adapt-compare",
        help="compare repair of shifted delayed causal mechanisms",
    )
    temporal_adapt_parser.add_argument(
        "--source-world", choices=world_names(), default="delayed-lamp"
    )
    temporal_adapt_parser.add_argument(
        "--target-world", choices=world_names(), default="delayed-lamp-long-delay"
    )
    temporal_adapt_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core-temporal",
            "causal-core-active",
            "causal-core",
            "reward-rl",
            "random",
            "passive-correlation",
        ],
    )
    temporal_adapt_parser.add_argument("--explore-steps", type=int, default=8)
    temporal_adapt_parser.add_argument("--adapt-steps", type=int, default=4)
    temporal_adapt_parser.add_argument("--seeds", type=int, default=20)
    temporal_adapt_parser.add_argument(
        "--adaptation-mode",
        choices=["continue", "fresh", "structural-prior"],
        default="structural-prior",
    )

    temporal_selective_parser = subparsers.add_parser(
        "temporal-selective-repair-compare",
        help="compare selective repair when only some delayed mechanisms change",
    )
    temporal_selective_parser.add_argument(
        "--source-world", choices=world_names(), default="dual-delayed-controls"
    )
    temporal_selective_parser.add_argument(
        "--target-world",
        choices=world_names(),
        default="dual-delayed-controls-selective-shift",
    )
    temporal_selective_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core-temporal",
            "causal-core-active",
            "causal-core",
            "reward-rl",
            "random",
            "passive-correlation",
        ],
    )
    temporal_selective_parser.add_argument("--explore-steps", type=int, default=9)
    temporal_selective_parser.add_argument("--adapt-steps", type=int, default=6)
    temporal_selective_parser.add_argument("--seeds", type=int, default=20)
    temporal_selective_parser.add_argument(
        "--adaptation-mode",
        choices=["continue", "fresh", "structural-prior"],
        default="structural-prior",
    )

    level6_schema_parser = subparsers.add_parser(
        "level6-schema-compare",
        help="compare few-shot causal transfer under a renamed schema",
    )
    level6_schema_parser.add_argument(
        "--source-world", choices=world_names(), default="dual-delayed-controls"
    )
    level6_schema_parser.add_argument(
        "--target-world",
        choices=world_names(),
        default="renamed-dual-delayed-controls",
    )
    level6_schema_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core-temporal-portable",
            "causal-core-temporal",
            "causal-core-active",
            "causal-core",
            "reward-rl",
            "random",
            "passive-correlation",
        ],
    )
    level6_schema_parser.add_argument("--explore-steps", type=int, default=9)
    level6_schema_parser.add_argument("--target-steps", type=int, default=6)
    level6_schema_parser.add_argument("--seeds", type=int, default=20)
    level6_schema_parser.add_argument(
        "--transfer-mode",
        choices=["schema-prior", "fresh", "continue"],
        default="schema-prior",
    )

    level6_schema_repair_parser = subparsers.add_parser(
        "level6-schema-repair-compare",
        help="compare schema transfer plus selective repair of a shifted mechanism",
    )
    level6_schema_repair_parser.add_argument(
        "--source-world", choices=world_names(), default="dual-delayed-controls"
    )
    level6_schema_repair_parser.add_argument(
        "--target-world",
        choices=world_names(),
        default="renamed-dual-delayed-controls-selective-shift",
    )
    level6_schema_repair_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core-temporal-portable",
            "causal-core-temporal",
            "causal-core-active",
            "causal-core",
            "reward-rl",
            "random",
            "passive-correlation",
        ],
    )
    level6_schema_repair_parser.add_argument("--explore-steps", type=int, default=9)
    level6_schema_repair_parser.add_argument("--target-steps", type=int, default=6)
    level6_schema_repair_parser.add_argument("--seeds", type=int, default=20)
    level6_schema_repair_parser.add_argument(
        "--transfer-mode",
        choices=["schema-prior", "fresh", "continue"],
        default="schema-prior",
    )

    level6_active_diagnostic_parser = subparsers.add_parser(
        "level6-active-diagnostic-compare",
        help="compare prediction-error-guided schema repair under a tight budget",
    )
    level6_active_diagnostic_parser.add_argument(
        "--source-world", choices=world_names(), default="triple-delayed-controls"
    )
    level6_active_diagnostic_parser.add_argument(
        "--target-world",
        choices=world_names(),
        default="renamed-triple-delayed-controls-diagnostic-shift",
    )
    level6_active_diagnostic_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core-temporal-diagnostic",
            "causal-core-temporal-portable",
            "causal-core-temporal",
            "causal-core-active",
            "causal-core",
            "reward-rl",
            "random",
            "passive-correlation",
        ],
    )
    level6_active_diagnostic_parser.add_argument("--explore-steps", type=int, default=12)
    level6_active_diagnostic_parser.add_argument(
        "--adaptation-steps",
        "--repair-steps",
        dest="repair_steps",
        metavar="ADAPTATION_STEPS",
        type=int,
        default=1,
    )
    level6_active_diagnostic_parser.add_argument("--seeds", type=int, default=20)
    level6_active_diagnostic_parser.add_argument(
        "--transfer-mode",
        choices=["schema-prior", "fresh", "continue"],
        default="schema-prior",
    )

    level6_procedural_diagnostic_parser = subparsers.add_parser(
        "level6-procedural-diagnostic-compare",
        aliases=["fewshot-procedural-diagnostic-compare"],
        help="compare active diagnostic adaptation over generated SCM families",
    )
    level6_procedural_diagnostic_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core-temporal-diagnostic",
            "causal-core-temporal-portable",
            "causal-core-temporal",
            "causal-core-active",
            "causal-core",
            "reward-rl",
            "random",
            "passive-correlation",
        ],
    )
    level6_procedural_diagnostic_parser.add_argument("--families", type=int, default=30)
    level6_procedural_diagnostic_parser.add_argument(
        "--mechanisms",
        type=int,
        choices=[2, 3],
        default=3,
    )
    level6_procedural_diagnostic_parser.add_argument(
        "--explore-steps", type=int, default=12
    )
    level6_procedural_diagnostic_parser.add_argument(
        "--adaptation-steps",
        "--repair-steps",
        dest="repair_steps",
        metavar="ADAPTATION_STEPS",
        type=int,
        default=1,
    )
    level6_procedural_diagnostic_parser.add_argument(
        "--transfer-mode",
        choices=["schema-prior", "fresh", "continue"],
        default="schema-prior",
    )
    level6_procedural_diagnostic_parser.add_argument(
        "--readout-mode",
        choices=[
            "none",
            "opaque",
            "semantic-confounder",
            "noisy-opaque",
            "noisy-semantic-confounder",
        ],
        default="none",
    )

    repair_efficiency_parser = subparsers.add_parser(
        "repair-efficiency-compare",
        help="compare structural-prior repair against fresh relearning",
    )
    repair_efficiency_parser.add_argument(
        "--source-world", choices=world_names(), default="panel"
    )
    repair_efficiency_parser.add_argument(
        "--target-world", choices=world_names(), default="panel-inverted"
    )
    repair_efficiency_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=["causal-core", "passive-correlation", "random"],
    )
    repair_efficiency_parser.add_argument("--explore-steps", type=int, default=60)
    repair_efficiency_parser.add_argument("--adapt-steps", type=int, default=12)
    repair_efficiency_parser.add_argument("--seeds", type=int, default=20)

    llm_pilot_parser = subparsers.add_parser(
        "llm-pilot",
        help="run a local LLM-as-agent pilot against causal baselines",
    )
    llm_pilot_parser.add_argument(
        "--worlds",
        nargs="+",
        choices=world_names(),
        default=["panel-noisy-core", "panel-hidden-context", "panel-noisy-hidden"],
    )
    llm_pilot_parser.add_argument(
        "--variants",
        nargs="+",
        choices=llm_agent_names() + agent_names(),
        default=[
            "llm-vanilla",
            "llm-causal-prompt",
            "llm-observation-adapter",
            "llm-persistent-latent",
            "causal-core",
            "causal-core-observation-adapter-v2",
            "causal-core-persistent-latent-context-adapter",
            "random",
        ],
    )
    llm_pilot_parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace chat model name or local model path",
    )
    llm_pilot_parser.add_argument("--steps", type=int, default=40)
    llm_pilot_parser.add_argument("--seeds", type=int, default=2)
    llm_pilot_parser.add_argument("--max-new-tokens", type=int, default=96)
    llm_pilot_parser.add_argument("--temperature", type=float, default=0.0)
    llm_pilot_parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="load the HuggingFace model from local cache unless disabled",
    )
    llm_pilot_parser.add_argument(
        "--save-json",
        default="tmp/llm_pilot_results.json",
        help="path for detailed pilot results; empty string disables saving",
    )

    observation_sweep_parser = subparsers.add_parser(
        "observation-sweep",
        help="sweep observation worlds over agents, steps, and seeds",
    )
    observation_sweep_parser.add_argument(
        "--worlds",
        nargs="+",
        choices=world_names(),
        default=["panel-noisy-core", "panel-hidden-context", "panel-noisy-hidden"],
    )
    observation_sweep_parser.add_argument(
        "--agents",
        nargs="+",
        choices=llm_agent_names() + agent_names(),
        default=[
            "causal-core",
            "causal-core-noise-aware",
            "causal-core-observation-adapter-v2",
            "causal-core-hidden-context-adapter",
            "causal-core-latent-context-adapter",
            "causal-core-stateful-latent-context-adapter",
            "causal-core-persistent-latent-context-adapter",
            "random",
            "passive-correlation",
        ],
    )
    observation_sweep_parser.add_argument(
        "--steps-list",
        nargs="+",
        type=int,
        default=[60, 120, 200],
    )
    observation_sweep_parser.add_argument("--seeds", type=int, default=10)
    observation_sweep_parser.add_argument("--sensor-suffix", default="_sensor")
    observation_sweep_parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace chat model for LLM agents, if any are requested",
    )
    observation_sweep_parser.add_argument("--max-new-tokens", type=int, default=64)
    observation_sweep_parser.add_argument("--temperature", type=float, default=0.0)
    observation_sweep_parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    observation_sweep_parser.add_argument(
        "--save-json",
        default="tmp/observation_sweep_results.json",
    )

    procedural_hidden_parser = subparsers.add_parser(
        "procedural-hidden-sweep",
        help="sweep generated noisy-hidden SCM families over causal agents",
    )
    procedural_hidden_parser.add_argument("--families", type=int, default=10)
    procedural_hidden_parser.add_argument("--family-start", type=int, default=1)
    procedural_hidden_parser.add_argument("--steps", type=int, default=100)
    procedural_hidden_parser.add_argument("--seeds", type=int, default=2)
    procedural_hidden_parser.add_argument("--mechanisms", type=int, default=2)
    procedural_hidden_parser.add_argument("--visible", type=int, default=2)
    procedural_hidden_parser.add_argument("--readouts", type=int, default=2)
    procedural_hidden_parser.add_argument("--noise", type=float, default=0.02)
    procedural_hidden_parser.add_argument(
        "--agents",
        nargs="+",
        choices=agent_names(),
        default=[
            "causal-core-persistent-latent-context-adapter",
            "causal-core-proactive-latent-context-adapter",
            "causal-core-control-experiment-planner",
            "causal-core-context-search-planner",
            "random",
            "passive-correlation",
        ],
    )
    procedural_hidden_parser.add_argument(
        "--save-json",
        default="tmp/procedural_hidden_sweep.json",
    )

    ranking_loss_parser = subparsers.add_parser(
        "ranking-loss-baseline",
        help=(
            "evaluate a ranking-supervised predictor against mechanism-level "
            "hidden-context and L6 repair probes"
        ),
    )
    ranking_loss_parser.add_argument("--families", type=int, default=6)
    ranking_loss_parser.add_argument("--seeds", type=int, default=2)
    ranking_loss_parser.add_argument("--train-states", type=int, default=64)
    ranking_loss_parser.add_argument("--test-states", type=int, default=64)
    ranking_loss_parser.add_argument("--hidden-steps", type=int, default=160)
    ranking_loss_parser.add_argument("--epochs", type=int, default=12)
    ranking_loss_parser.add_argument(
        "--save-json",
        default="tmp/ranking_loss_baseline.json",
    )

    continuous_parser = subparsers.add_parser(
        "continuous-poc",
        help="run a continuous-state/action metric-packing proof of concept",
    )
    continuous_parser.add_argument("--seeds", type=int, default=20)
    continuous_parser.add_argument("--random-trials", type=int, default=160)
    continuous_parser.add_argument("--regression-trials", type=int, default=240)
    continuous_parser.add_argument("--paired-samples", type=int, default=8)
    continuous_parser.add_argument("--noise", type=float, default=0.02)
    continuous_parser.add_argument(
        "--save-json",
        default="tmp/continuous_metric_poc.json",
    )

    stress_parser = subparsers.add_parser(
        "icml-stress-suite",
        help="run matched mechanism baselines, budget sensitivity, and continuous sweeps",
    )
    stress_parser.add_argument("--families", type=int, default=6)
    stress_parser.add_argument("--seeds", type=int, default=2)
    stress_parser.add_argument("--steps", type=int, default=160)
    stress_parser.add_argument("--mechanisms", type=int, default=4)
    stress_parser.add_argument("--visible", type=int, default=5)
    stress_parser.add_argument("--readouts", type=int, default=6)
    stress_parser.add_argument("--noise", type=float, default=0.02)
    stress_parser.add_argument("--neural-epochs", type=int, default=24)
    stress_parser.add_argument(
        "--save-json",
        default="tmp/icml_stress_suite.json",
    )

    matched_parser = subparsers.add_parser(
        "matched-mechanism-baselines",
        help="run trajectory-matched discovery, tuple, and neural mechanism baselines",
    )
    matched_parser.add_argument("--families", type=int, default=8)
    matched_parser.add_argument("--seeds", type=int, default=3)
    matched_parser.add_argument("--steps", type=int, default=200)
    matched_parser.add_argument("--mechanisms", type=int, default=4)
    matched_parser.add_argument("--visible", type=int, default=5)
    matched_parser.add_argument("--readouts", type=int, default=6)
    matched_parser.add_argument("--noise", type=float, default=0.02)
    matched_parser.add_argument("--neural-epochs", type=int, default=24)
    matched_parser.add_argument(
        "--save-json",
        default="tmp/matched_mechanism_baselines.json",
    )

    mujoco_parser = subparsers.add_parser(
        "mujoco-probe",
        help="train a dynamics model and run frozen mechanism probes in MuJoCo",
    )
    mujoco_parser.add_argument("--env-id", default="Reacher-v5")
    mujoco_parser.add_argument("--seeds", type=int, default=5)
    mujoco_parser.add_argument("--train-states", type=int, default=240)
    mujoco_parser.add_argument("--test-states", type=int, default=120)
    mujoco_parser.add_argument("--random-transitions", type=int, default=3000)
    mujoco_parser.add_argument("--val-transitions", type=int, default=800)
    mujoco_parser.add_argument("--epochs", type=int, default=60)
    mujoco_parser.add_argument("--pulse", type=float, default=0.8)
    mujoco_parser.add_argument("--context-dims", nargs="+", type=int, default=[0, 1])
    mujoco_parser.add_argument(
        "--save-json",
        default="tmp/mujoco_probe_reacher.json",
    )
    mujoco_parser.add_argument(
        "--figure-dir",
        default="docs/figures",
    )

    mujoco_causal_parser = subparsers.add_parser(
        "mujoco-causal-checks",
        help="run MuJoCo context-shift and readout checks for causal mechanisms",
    )
    mujoco_causal_parser.add_argument("--env-id", default="HalfCheetah-v5")
    mujoco_causal_parser.add_argument("--seeds", type=int, default=5)
    mujoco_causal_parser.add_argument("--train-states", type=int, default=240)
    mujoco_causal_parser.add_argument("--test-states", type=int, default=120)
    mujoco_causal_parser.add_argument("--adapt-states", type=int, default=30)
    mujoco_causal_parser.add_argument("--random-transitions", type=int, default=3000)
    mujoco_causal_parser.add_argument("--val-transitions", type=int, default=800)
    mujoco_causal_parser.add_argument("--epochs", type=int, default=60)
    mujoco_causal_parser.add_argument("--pulse", type=float, default=0.8)
    mujoco_causal_parser.add_argument(
        "--context-dims",
        nargs="+",
        type=int,
        default=[0, 1],
    )
    mujoco_causal_parser.add_argument("--readout-scale", type=float, default=4.0)
    mujoco_causal_parser.add_argument(
        "--save-json",
        default="tmp/mujoco_causal_checks.json",
    )
    mujoco_causal_parser.add_argument(
        "--figure-dir",
        default="docs/figures",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        world = make_world(args.world)
        agent = make_agent(args.agent, seed=args.seed)
        result = run_episode(world, agent, steps=args.steps, seed=args.seed)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_episode(result))
        return 0

    if args.command == "compare":
        rows = []
        for agent_name in args.agents:
            scores: list[EdgeScore] = []
            edge_counts: list[int] = []
            for seed in range(1, args.seeds + 1):
                result = run_episode(
                    make_world(args.world),
                    make_agent(agent_name, seed=seed),
                    steps=args.steps,
                    seed=seed,
                )
                scores.append(result.score)
                edge_counts.append(len(result.discovered_edges))
            rows.append(_comparison_row(agent_name, scores, edge_counts))
        print(_format_comparison(args.world, args.steps, args.seeds, rows))
        return 0

    if args.command == "reward-compare":
        rows = []
        for agent_name in args.agents:
            results: list[RewardEpisodeResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_reward_episode(
                        make_world(args.world),
                        make_agent(agent_name, seed=seed),
                        steps=args.steps,
                        seed=seed,
                    )
                )
            rows.append(_reward_comparison_row(agent_name, results))
        print(_format_reward_comparison(args.world, args.steps, args.seeds, rows))
        return 0

    if args.command == "observation-compare":
        rows = []
        readout_variables = _readout_variables(args.world, args.sensor_suffix)
        hidden_context_edges = _hidden_context_edges(args.world)
        for agent_name in args.agents:
            results = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_episode(
                        make_world(args.world),
                        make_agent(agent_name, seed=seed),
                        steps=args.steps,
                        seed=seed,
                    )
                )
            rows.append(
                _observation_comparison_row(
                    agent_name,
                    results,
                    readout_variables=readout_variables,
                    hidden_context_edges=hidden_context_edges,
                )
            )
        print(
            _format_observation_comparison(
                args.world,
                args.steps,
                args.seeds,
                args.sensor_suffix,
                readout_variables,
                hidden_context_edges,
                rows,
            )
        )
        return 0

    if args.command == "llm-pilot":
        print(
            _run_llm_pilot(
                worlds=args.worlds,
                variants=args.variants,
                model=args.model,
                steps=args.steps,
                seeds=args.seeds,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                local_files_only=args.local_files_only,
                save_json=args.save_json,
            )
        )
        return 0

    if args.command == "observation-sweep":
        print(
            _run_observation_sweep(
                worlds=args.worlds,
                agents=args.agents,
                steps_list=args.steps_list,
                seeds=args.seeds,
                sensor_suffix=args.sensor_suffix,
                model=args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                local_files_only=args.local_files_only,
                save_json=args.save_json,
            )
        )
        return 0

    if args.command == "procedural-hidden-sweep":
        print(
            _run_procedural_hidden_sweep(
                families=args.families,
                family_start=args.family_start,
                agents=args.agents,
                steps=args.steps,
                seeds=args.seeds,
                mechanisms=args.mechanisms,
                visible=args.visible,
                readouts=args.readouts,
                noise=args.noise,
                save_json=args.save_json,
            )
        )
        return 0

    if args.command == "ranking-loss-baseline":
        payload = run_ranking_loss_baseline_experiment(
            families=args.families,
            seeds=args.seeds,
            train_states=args.train_states,
            test_states=args.test_states,
            hidden_steps=args.hidden_steps,
            epochs=args.epochs,
        )
        if args.save_json:
            output_path = Path(args.save_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(format_ranking_loss_baseline(payload))
        if args.save_json:
            print(f"\nSaved detailed JSON: {args.save_json}")
        return 0

    if args.command == "continuous-poc":
        payload = run_continuous_metric_poc(
            seeds=args.seeds,
            random_trials=args.random_trials,
            regression_trials=args.regression_trials,
            paired_samples=args.paired_samples,
            noise=args.noise,
        )
        if args.save_json:
            save_continuous_payload(payload, args.save_json)
        print(format_continuous_metric_poc(payload))
        if args.save_json:
            print(f"\nSaved detailed JSON: {args.save_json}")
        return 0

    if args.command == "icml-stress-suite":
        payload = run_icml_stress_suite(
            families=args.families,
            seeds=args.seeds,
            steps=args.steps,
            mechanisms=args.mechanisms,
            visible=args.visible,
            readouts=args.readouts,
            noise=args.noise,
            neural_epochs=args.neural_epochs,
        )
        if args.save_json:
            save_icml_stress_suite(payload, args.save_json)
        print(format_icml_stress_suite(payload))
        if args.save_json:
            print(f"\nSaved detailed JSON: {args.save_json}")
        return 0

    if args.command == "matched-mechanism-baselines":
        payload = run_matched_mechanism_baseline_suite(
            families=args.families,
            seeds=args.seeds,
            steps=args.steps,
            mechanisms=args.mechanisms,
            visible=args.visible,
            readouts=args.readouts,
            noise=args.noise,
            neural_epochs=args.neural_epochs,
        )
        if args.save_json:
            save_icml_stress_suite(payload, args.save_json)
        print(format_matched_mechanism_baseline_suite(payload))
        if args.save_json:
            print(f"\nSaved detailed JSON: {args.save_json}")
        return 0

    if args.command == "mujoco-probe":
        payload = run_mujoco_probe_experiment(
            env_id=args.env_id,
            seeds=args.seeds,
            train_states=args.train_states,
            test_states=args.test_states,
            random_transitions=args.random_transitions,
            val_transitions=args.val_transitions,
            epochs=args.epochs,
            pulse=args.pulse,
            context_dims=tuple(args.context_dims),
            save_json=args.save_json,
            figure_dir=args.figure_dir,
        )
        print(format_mujoco_probe(payload))
        if args.save_json:
            print(f"\nSaved detailed JSON: {args.save_json}")
        if args.figure_dir:
            print(f"Saved figures under: {args.figure_dir}")
        return 0

    if args.command == "mujoco-causal-checks":
        payload = run_mujoco_causal_checks(
            env_id=args.env_id,
            seeds=args.seeds,
            train_states=args.train_states,
            test_states=args.test_states,
            adapt_states=args.adapt_states,
            random_transitions=args.random_transitions,
            val_transitions=args.val_transitions,
            epochs=args.epochs,
            pulse=args.pulse,
            context_dims=tuple(args.context_dims),
            readout_scale=args.readout_scale,
            save_json=args.save_json,
            figure_dir=args.figure_dir,
        )
        print(format_mujoco_causal_checks(payload))
        if args.save_json:
            print(f"\nSaved detailed JSON: {args.save_json}")
        if args.figure_dir:
            print(f"Saved figures under: {args.figure_dir}")
        return 0

    if args.command == "intervention-test":
        result = run_intervention_test(
            make_world(args.world),
            make_agent(args.agent, seed=args.seed),
            explore_steps=args.explore_steps,
            seed=args.seed,
            include_seen=args.include_seen,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_intervention_test(result))
        return 0

    if args.command == "intervention-compare":
        rows = []
        for agent_name in args.agents:
            results: list[InterventionTestResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_intervention_test(
                        make_world(args.world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        seed=seed,
                        include_seen=args.include_seen,
                    )
                )
            rows.append(_intervention_comparison_row(agent_name, results))
        print(
            _format_intervention_comparison(
                args.world, args.explore_steps, args.seeds, rows
            )
        )
        return 0

    if args.command == "counterfactual-test":
        result = run_counterfactual_test(
            make_world(args.world),
            make_agent(args.agent, seed=args.seed),
            explore_steps=args.explore_steps,
            seed=args.seed,
            case_source=args.case_source,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_counterfactual_test(result))
        return 0

    if args.command == "counterfactual-compare":
        rows = []
        for agent_name in args.agents:
            results: list[CounterfactualTestResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_counterfactual_test(
                        make_world(args.world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        seed=seed,
                        case_source=args.case_source,
                    )
                )
            rows.append(_counterfactual_comparison_row(agent_name, results))
        print(
            _format_counterfactual_comparison(
                args.world, args.explore_steps, args.seeds, rows
            )
        )
        return 0

    if args.command == "transfer-test":
        result = run_transfer_intervention_test(
            make_world(args.source_world),
            make_world(args.target_world),
            make_agent(args.agent, seed=args.seed),
            explore_steps=args.explore_steps,
            seed=args.seed,
            include_seen=args.include_seen,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_transfer_intervention_test(result))
        return 0

    if args.command == "transfer-compare":
        rows = []
        for agent_name in args.agents:
            results: list[TransferInterventionTestResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_transfer_intervention_test(
                        make_world(args.source_world),
                        make_world(args.target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        seed=seed,
                        include_seen=args.include_seen,
                    )
                )
            rows.append(_transfer_comparison_row(agent_name, results))
        print(
            _format_transfer_comparison(
                args.source_world,
                args.target_world,
                args.explore_steps,
                args.seeds,
                rows,
            )
        )
        return 0

    if args.command == "transfer-adapt-compare":
        rows = []
        for agent_name in args.agents:
            results: list[TransferAdaptationTestResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_transfer_adaptation_test(
                        make_world(args.source_world),
                        make_world(args.target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        adapt_steps=args.adapt_steps,
                        seed=seed,
                        adaptation_mode=args.adaptation_mode,
                    )
                )
            rows.append(_transfer_adaptation_row(agent_name, results))
        print(
            _format_transfer_adaptation_comparison(
                args.source_world,
                args.target_world,
                args.explore_steps,
                args.adapt_steps,
                args.seeds,
                args.adaptation_mode,
                rows,
            )
        )
        return 0

    if args.command == "level3-compare":
        rows = []
        for agent_name in args.agents:
            results: list[Level3TestResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_level3_test(
                        make_world(args.source_world),
                        make_world(args.stable_target_world),
                        make_world(args.repair_target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        adapt_steps=args.adapt_steps,
                        seed=seed,
                        case_source=args.case_source,
                        adaptation_mode=args.adaptation_mode,
                    )
                )
            rows.append(_level3_comparison_row(agent_name, results))
        print(
            _format_level3_comparison(
                args.source_world,
                args.stable_target_world,
                args.repair_target_world,
                args.explore_steps,
                args.adapt_steps,
                args.seeds,
                args.case_source,
                args.adaptation_mode,
                rows,
            )
        )
        return 0

    if args.command == "level4-compare":
        rows = []
        for agent_name in args.agents:
            results: list[Level4DisambiguationResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_level4_disambiguation_test(
                        make_world(args.world),
                        make_agent(agent_name, seed=seed),
                        steps=args.steps,
                        seed=seed,
                    )
                )
            rows.append(_level4_comparison_row(agent_name, results, args.steps))
        print(_format_level4_comparison(args.world, args.steps, args.seeds, rows))
        return 0

    if args.command == "temporal-credit-compare":
        rows = []
        for agent_name in args.agents:
            results: list[TemporalCreditResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_temporal_credit_test(
                        make_world(args.world),
                        make_agent(agent_name, seed=seed),
                        steps=args.steps,
                        seed=seed,
                    )
                )
            rows.append(_temporal_credit_row(agent_name, results, args.steps))
        print(
            _format_temporal_credit_comparison(
                args.world, args.steps, args.seeds, rows
            )
        )
        return 0

    if args.command == "temporal-transfer-compare":
        rows = []
        for agent_name in args.agents:
            results: list[TemporalTransferResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_temporal_transfer_test(
                        make_world(args.source_world),
                        make_world(args.target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        seed=seed,
                    )
                )
            rows.append(_temporal_transfer_row(agent_name, results))
        print(
            _format_temporal_transfer_comparison(
                args.source_world,
                args.target_world,
                args.explore_steps,
                args.seeds,
                rows,
            )
        )
        return 0

    if args.command == "temporal-adapt-compare":
        rows = []
        for agent_name in args.agents:
            results: list[TemporalAdaptationResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_temporal_adaptation_test(
                        make_world(args.source_world),
                        make_world(args.target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        adapt_steps=args.adapt_steps,
                        seed=seed,
                        adaptation_mode=args.adaptation_mode,
                    )
                )
            rows.append(_temporal_adaptation_row(agent_name, results))
        print(
            _format_temporal_adaptation_comparison(
                args.source_world,
                args.target_world,
                args.explore_steps,
                args.adapt_steps,
                args.seeds,
                args.adaptation_mode,
                rows,
            )
        )
        return 0

    if args.command == "temporal-selective-repair-compare":
        rows = []
        for agent_name in args.agents:
            results: list[TemporalSelectiveAdaptationResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_temporal_selective_adaptation_test(
                        make_world(args.source_world),
                        make_world(args.target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        adapt_steps=args.adapt_steps,
                        seed=seed,
                        adaptation_mode=args.adaptation_mode,
                    )
                )
            rows.append(_temporal_selective_adaptation_row(agent_name, results))
        print(
            _format_temporal_selective_adaptation_comparison(
                args.source_world,
                args.target_world,
                args.explore_steps,
                args.adapt_steps,
                args.seeds,
                args.adaptation_mode,
                rows,
            )
        )
        return 0

    if args.command == "level6-schema-compare":
        rows = []
        for agent_name in args.agents:
            results: list[Level6SchemaTransferResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_level6_schema_transfer_test(
                        make_world(args.source_world),
                        make_world(args.target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        target_steps=args.target_steps,
                        seed=seed,
                        transfer_mode=args.transfer_mode,
                    )
                )
            rows.append(_level6_schema_row(agent_name, results))
        print(
            _format_level6_schema_comparison(
                args.source_world,
                args.target_world,
                args.explore_steps,
                args.target_steps,
                args.seeds,
                args.transfer_mode,
                rows,
            )
        )
        return 0

    if args.command == "level6-schema-repair-compare":
        rows = []
        for agent_name in args.agents:
            results: list[Level6SchemaRepairResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_level6_schema_repair_test(
                        make_world(args.source_world),
                        make_world(args.target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        target_steps=args.target_steps,
                        seed=seed,
                        transfer_mode=args.transfer_mode,
                    )
                )
            rows.append(_level6_schema_repair_row(agent_name, results))
        print(
            _format_level6_schema_repair_comparison(
                args.source_world,
                args.target_world,
                args.explore_steps,
                args.target_steps,
                args.seeds,
                args.transfer_mode,
                rows,
            )
        )
        return 0

    if args.command == "level6-active-diagnostic-compare":
        rows = []
        for agent_name in args.agents:
            results: list[Level6SchemaRepairResult] = []
            for seed in range(1, args.seeds + 1):
                results.append(
                    run_level6_active_diagnostic_repair_test(
                        make_world(args.source_world),
                        make_world(args.target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        repair_steps=args.repair_steps,
                        seed=seed,
                        transfer_mode=args.transfer_mode,
                    )
                )
            rows.append(_level6_schema_repair_row(agent_name, results))
        print(
            _format_level6_active_diagnostic_comparison(
                args.source_world,
                args.target_world,
                args.explore_steps,
                args.repair_steps,
                args.seeds,
                args.transfer_mode,
                rows,
            )
        )
        return 0

    if args.command in {
        "level6-procedural-diagnostic-compare",
        "fewshot-procedural-diagnostic-compare",
    }:
        rows = []
        for agent_name in args.agents:
            results: list[Level6SchemaRepairResult] = []
            for family_seed in range(1, args.families + 1):
                source_world, target_world = make_procedural_diagnostic_world_pair(
                    family_seed=family_seed,
                    mechanism_count=args.mechanisms,
                    readout_mode=args.readout_mode,
                )
                results.append(
                    run_level6_active_diagnostic_repair_test(
                        source_world,
                        target_world,
                        make_agent(agent_name, seed=family_seed),
                        explore_steps=args.explore_steps,
                        repair_steps=args.repair_steps,
                        seed=family_seed,
                        transfer_mode=args.transfer_mode,
                    )
                )
            rows.append(_level6_schema_repair_row(agent_name, results))
        print(
            _format_level6_procedural_diagnostic_comparison(
                args.families,
                args.mechanisms,
                args.explore_steps,
                args.repair_steps,
                args.transfer_mode,
                args.readout_mode,
                rows,
            )
        )
        return 0

    if args.command == "repair-efficiency-compare":
        rows = []
        for agent_name in args.agents:
            structural_results: list[TransferAdaptationTestResult] = []
            fresh_results: list[TransferAdaptationTestResult] = []
            for seed in range(1, args.seeds + 1):
                structural_results.append(
                    run_transfer_adaptation_test(
                        make_world(args.source_world),
                        make_world(args.target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        adapt_steps=args.adapt_steps,
                        seed=seed,
                        adaptation_mode="structural-prior",
                    )
                )
                fresh_results.append(
                    run_transfer_adaptation_test(
                        make_world(args.source_world),
                        make_world(args.target_world),
                        make_agent(agent_name, seed=seed),
                        explore_steps=args.explore_steps,
                        adapt_steps=args.adapt_steps,
                        seed=seed,
                        adaptation_mode="fresh",
                    )
                )
            rows.append(
                _repair_efficiency_row(
                    agent_name, structural_results, fresh_results
                )
            )
        print(
            _format_repair_efficiency_comparison(
                args.source_world,
                args.target_world,
                args.explore_steps,
                args.adapt_steps,
                args.seeds,
                rows,
            )
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _comparison_row(
    agent_name: str, scores: list[EdgeScore], edge_counts: list[int]
) -> dict[str, object]:
    return {
        "agent": agent_name,
        "precision": _mean([score.precision for score in scores]),
        "recall": _mean([score.recall for score in scores]),
        "f1": _mean([score.f1 for score in scores]),
        "prediction": _mean([score.prediction_jaccard for score in scores]),
        "false_positive": _mean([score.false_positive for score in scores]),
        "false_negative": _mean([score.false_negative for score in scores]),
        "edges": _mean(edge_counts),
    }


def _format_comparison(
    world: str, steps: int, seeds: int, rows: list[dict[str, object]]
) -> str:
    lines = [
        f"World: {world}",
        f"Steps per run: {steps}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 precision  recall  f1    pred-jaccard  "
            "avg-fp  avg-fn  avg-edges"
        ),
        "-" * 78,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['precision']):>9.2f}"
            f"{float(row['recall']):>8.2f}"
            f"{float(row['f1']):>6.2f}"
            f"{float(row['prediction']):>14.2f}"
            f"{float(row['false_positive']):>8.2f}"
            f"{float(row['false_negative']):>8.2f}"
            f"{float(row['edges']):>11.2f}"
        )
    return "\n".join(lines)


def _reward_comparison_row(
    agent_name: str, results: list[RewardEpisodeResult]
) -> dict[str, object]:
    return {
        "agent": agent_name,
        "total_return": _mean([result.total_return for result in results]),
        "mean_return": _mean([result.mean_return for result in results]),
        "final_utility": _mean([result.final_utility for result in results]),
        "graph_f1": _mean([result.exploration.score.f1 for result in results]),
        "graph_precision": _mean(
            [result.exploration.score.precision for result in results]
        ),
        "graph_recall": _mean([result.exploration.score.recall for result in results]),
        "prediction": _mean(
            [result.exploration.score.prediction_jaccard for result in results]
        ),
    }


def _format_reward_comparison(
    world: str, steps: int, seeds: int, rows: list[dict[str, object]]
) -> str:
    lines = [
        f"World: {world}",
        f"Steps per run: {steps}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 return  mean-r  final-u  graph-p  "
            "graph-r  graph-f1  pred-j"
        ),
        "-" * 82,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['total_return']):>7.2f}"
            f"{float(row['mean_return']):>8.2f}"
            f"{float(row['final_utility']):>9.2f}"
            f"{float(row['graph_precision']):>9.2f}"
            f"{float(row['graph_recall']):>9.2f}"
            f"{float(row['graph_f1']):>10.2f}"
            f"{float(row['prediction']):>8.2f}"
        )
    return "\n".join(lines)


def _observation_comparison_row(
    agent_name: str,
    results: list[object],
    readout_variables: set[str],
    hidden_context_edges: set[tuple[str, str]],
) -> dict[str, object]:
    readout_false_positives = []
    non_readout_false_positives = []
    false_positives = []
    hidden_context_recalls = []
    hidden_context_false_positives = []
    scores = []
    edge_counts = []
    for result in results:
        scores.append(result.score)
        edge_counts.append(len(result.discovered_edges))
        fp_edges = result.discovered_edges - result.true_edges
        false_positives.append(len(fp_edges))
        readout_false_positives.append(
            sum(target in readout_variables for _, target in fp_edges)
        )
        non_readout_false_positives.append(
            sum(target not in readout_variables for _, target in fp_edges)
        )
        hidden_hint_edges = _hidden_context_hint_edges(result.condition_hints)
        hidden_context_recalls.append(
            _safe_div(
                len(hidden_hint_edges & hidden_context_edges),
                len(hidden_context_edges),
            )
        )
        hidden_context_false_positives.append(
            len(hidden_hint_edges - hidden_context_edges)
        )
    return {
        "agent": agent_name,
        "precision": _mean([score.precision for score in scores]),
        "recall": _mean([score.recall for score in scores]),
        "f1": _mean([score.f1 for score in scores]),
        "false_positive": _mean(false_positives),
        "readout_false_positive": _mean(readout_false_positives),
        "non_readout_false_positive": _mean(non_readout_false_positives),
        "hidden_context_recall": _mean(hidden_context_recalls),
        "hidden_context_false_positive": _mean(hidden_context_false_positives),
        "edges": _mean(edge_counts),
    }


def _procedural_hidden_comparison_row(
    agent_name: str,
    runs: list[tuple[object, set[str], set[tuple[str, str]]]],
) -> dict[str, object]:
    readout_false_positives = []
    non_readout_false_positives = []
    false_positives = []
    hidden_context_recalls = []
    hidden_context_false_positives = []
    scores = []
    edge_counts = []
    for result, readout_variables, hidden_context_edges in runs:
        scores.append(result.score)
        edge_counts.append(len(result.discovered_edges))
        fp_edges = result.discovered_edges - result.true_edges
        false_positives.append(len(fp_edges))
        readout_false_positives.append(
            sum(target in readout_variables for _, target in fp_edges)
        )
        non_readout_false_positives.append(
            sum(target not in readout_variables for _, target in fp_edges)
        )
        hidden_hint_edges = _hidden_context_hint_edges(result.condition_hints)
        hidden_context_recalls.append(
            _safe_div(
                len(hidden_hint_edges & hidden_context_edges),
                len(hidden_context_edges),
            )
        )
        hidden_context_false_positives.append(
            len(hidden_hint_edges - hidden_context_edges)
        )
    return {
        "agent": agent_name,
        "precision": _mean([score.precision for score in scores]),
        "recall": _mean([score.recall for score in scores]),
        "f1": _mean([score.f1 for score in scores]),
        "false_positive": _mean(false_positives),
        "readout_false_positive": _mean(readout_false_positives),
        "non_readout_false_positive": _mean(non_readout_false_positives),
        "hidden_context_recall": _mean(hidden_context_recalls),
        "hidden_context_false_positive": _mean(hidden_context_false_positives),
        "edges": _mean(edge_counts),
    }


def _procedural_hidden_detail_row(
    agent_name: str,
    family_seed: int,
    seed: int,
    result: object,
    readout_variables: set[str],
    hidden_context_edges: set[tuple[str, str]],
) -> dict[str, object]:
    fp_edges = result.discovered_edges - result.true_edges
    hidden_hint_edges = _hidden_context_hint_edges(result.condition_hints)
    return {
        "agent": agent_name,
        "family_seed": family_seed,
        "seed": seed,
        "precision": result.score.precision,
        "recall": result.score.recall,
        "f1": result.score.f1,
        "false_positive": len(fp_edges),
        "readout_false_positive": sum(
            target in readout_variables for _, target in fp_edges
        ),
        "non_readout_false_positive": sum(
            target not in readout_variables for _, target in fp_edges
        ),
        "hidden_context_recall": _safe_div(
            len(hidden_hint_edges & hidden_context_edges),
            len(hidden_context_edges),
        ),
        "hidden_context_false_positive": len(
            hidden_hint_edges - hidden_context_edges
        ),
        "edges": len(result.discovered_edges),
    }


def _run_procedural_hidden_sweep(
    families: int,
    family_start: int,
    agents: list[str],
    steps: int,
    seeds: int,
    mechanisms: int,
    visible: int,
    readouts: int,
    noise: float,
    save_json: str,
) -> str:
    family_seeds = list(range(family_start, family_start + families))
    rows: list[dict[str, object]] = []
    agent_runs: dict[str, list[tuple[object, set[str], set[tuple[str, str]]]]] = {
        agent_name: [] for agent_name in agents
    }
    family_specs = []
    detailed_runs: list[dict[str, object]] = []
    for family_seed in family_seeds:
        spec_world = make_procedural_complex_hidden_world(
            family_seed=family_seed,
            mechanism_count=mechanisms,
            visible_count=visible,
            noise_probability=noise,
            readout_count=readouts,
        )
        family_specs.append(spec_world.procedural_spec())
        readout_variables = spec_world.readout_variables()
        hidden_context_edges = spec_world.hidden_context_edges()
        for agent_name in agents:
            for seed in range(1, seeds + 1):
                result = run_episode(
                    make_procedural_complex_hidden_world(
                        family_seed=family_seed,
                        mechanism_count=mechanisms,
                        visible_count=visible,
                        noise_probability=noise,
                        readout_count=readouts,
                    ),
                    make_agent(agent_name, seed=seed),
                    steps=steps,
                    seed=seed,
                )
                agent_runs[agent_name].append(
                    (result, readout_variables, hidden_context_edges)
                )
                detailed_runs.append(
                    _procedural_hidden_detail_row(
                        agent_name=agent_name,
                        family_seed=family_seed,
                        seed=seed,
                        result=result,
                        readout_variables=readout_variables,
                        hidden_context_edges=hidden_context_edges,
                    )
                )

    for agent_name in agents:
        rows.append(
            _procedural_hidden_comparison_row(
                agent_name,
                agent_runs[agent_name],
            )
        )

    payload: dict[str, object] = {
        "families": families,
        "family_start": family_start,
        "family_seeds": family_seeds,
        "steps": steps,
        "seeds": seeds,
        "mechanisms": mechanisms,
        "visible": visible,
        "readouts": readouts,
        "noise": noise,
        "agents": agents,
        "summary": rows,
        "runs": detailed_runs,
        "family_specs": family_specs,
    }
    if save_json:
        output_path = Path(save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    lines = [
        "Procedural noisy-hidden sweep",
        f"Families: {families} ({family_start}..{family_start + families - 1})",
        f"Steps per run: {steps}",
        f"Seeds per family: {seeds}",
        f"Mechanisms per family: {mechanisms}",
        f"Visible context variables: {visible}",
        f"Readout variables: {readouts}",
        f"Observation noise: {noise:.3f}",
        "",
        (
            "agent                           precision  recall  f1    "
            "avg-fp  readout-fp  non-readout-fp  hidden-r  hidden-fp  avg-edges"
        ),
        "-" * 116,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):31}"
            f"{float(row['precision']):>9.2f}"
            f"{float(row['recall']):>8.2f}"
            f"{float(row['f1']):>6.2f}"
            f"{float(row['false_positive']):>8.2f}"
            f"{float(row['readout_false_positive']):>12.2f}"
            f"{float(row['non_readout_false_positive']):>16.2f}"
            f"{float(row['hidden_context_recall']):>10.2f}"
            f"{float(row['hidden_context_false_positive']):>11.2f}"
            f"{float(row['edges']):>11.2f}"
        )
    if save_json:
        lines.append(f"\nSaved detailed JSON: {save_json}")
    return "\n".join(lines)


def _run_llm_pilot(
    worlds: list[str],
    variants: list[str],
    model: str,
    steps: int,
    seeds: int,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
    save_json: str,
) -> str:
    llm_variants = set(llm_agent_names())
    needs_llm = any(variant in llm_variants for variant in variants)
    generator = None
    if needs_llm:
        generator = LocalHFChatModel(
            model_name=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            local_files_only=local_files_only,
        )

    report_sections = [
        (
            "LLM pilot\n"
            f"Model: {model}\n"
            f"Steps per run: {steps}\n"
            f"Seeds: {seeds}\n"
            f"Variants: {', '.join(variants)}"
        )
    ]
    payload: dict[str, object] = {
        "model": model,
        "steps": steps,
        "seeds": seeds,
        "variants": variants,
        "worlds": {},
    }
    world_payloads: dict[str, object] = {}

    for world_name in worlds:
        readout_variables = _readout_variables(world_name, "_sensor")
        hidden_context_edges = _hidden_context_edges(world_name)
        rows = []
        variant_payloads: dict[str, object] = {}
        for variant in variants:
            results = []
            detail = []
            for seed in range(1, seeds + 1):
                if variant in llm_variants:
                    if generator is None:
                        raise RuntimeError("LLM generator was not initialized")
                    agent = make_llm_agent(variant, generator, seed=seed)
                else:
                    agent = make_agent(variant, seed=seed)
                result = run_episode(
                    make_world(world_name),
                    agent,
                    steps=steps,
                    seed=seed,
                )
                results.append(result)
                detail.append(result.to_dict())
            rows.append(
                _observation_comparison_row(
                    variant,
                    results,
                    readout_variables=readout_variables,
                    hidden_context_edges=hidden_context_edges,
                )
            )
            variant_payloads[variant] = {
                "summary": rows[-1],
                "runs": detail,
            }
        world_payloads[world_name] = variant_payloads
        report_sections.append(
            _format_observation_comparison(
                world_name,
                steps,
                seeds,
                "_sensor",
                readout_variables,
                hidden_context_edges,
                rows,
            )
        )

    payload["worlds"] = world_payloads
    if save_json:
        output_path = Path(save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        report_sections.append(f"Saved detailed JSON: {output_path}")

    return "\n\n".join(report_sections)


def _run_observation_sweep(
    worlds: list[str],
    agents: list[str],
    steps_list: list[int],
    seeds: int,
    sensor_suffix: str,
    model: str,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
    save_json: str,
) -> str:
    llm_variants = set(llm_agent_names())
    needs_llm = any(agent in llm_variants for agent in agents)
    generator = None
    if needs_llm:
        generator = LocalHFChatModel(
            model_name=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            local_files_only=local_files_only,
        )

    payload: dict[str, object] = {
        "worlds": worlds,
        "agents": agents,
        "steps_list": steps_list,
        "seeds": seeds,
        "sensor_suffix": sensor_suffix,
        "model": model if needs_llm else None,
        "results": {},
    }
    all_results: dict[str, object] = {}
    sections = [
        (
            "Observation sweep\n"
            f"Worlds: {', '.join(worlds)}\n"
            f"Agents: {', '.join(agents)}\n"
            f"Steps: {', '.join(str(step) for step in steps_list)}\n"
            f"Seeds: {seeds}"
        )
    ]

    for world_name in worlds:
        readout_variables = _readout_variables(world_name, sensor_suffix)
        hidden_context_edges = _hidden_context_edges(world_name)
        world_results: dict[str, object] = {}
        for steps in steps_list:
            rows = []
            agent_results: dict[str, object] = {}
            for agent_name in agents:
                results = []
                for seed in range(1, seeds + 1):
                    if agent_name in llm_variants:
                        if generator is None:
                            raise RuntimeError("LLM generator was not initialized")
                        agent = make_llm_agent(agent_name, generator, seed=seed)
                    else:
                        agent = make_agent(agent_name, seed=seed)
                    results.append(
                        run_episode(
                            make_world(world_name),
                            agent,
                            steps=steps,
                            seed=seed,
                        )
                    )
                row = _observation_comparison_row(
                    agent_name,
                    results,
                    readout_variables=readout_variables,
                    hidden_context_edges=hidden_context_edges,
                )
                rows.append(row)
                agent_results[agent_name] = row
            world_results[str(steps)] = {
                "summary": rows,
                "readout_variables": sorted(readout_variables),
                "hidden_context_edges": [
                    {"action": action, "target": target}
                    for action, target in sorted(hidden_context_edges)
                ],
            }
            sections.append(
                _format_observation_comparison(
                    world_name,
                    steps,
                    seeds,
                    sensor_suffix,
                    readout_variables,
                    hidden_context_edges,
                    rows,
                )
            )
        all_results[world_name] = world_results

    payload["results"] = all_results
    if save_json:
        output_path = Path(save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        sections.append(f"Saved detailed JSON: {output_path}")
    return "\n\n".join(sections)


def _format_observation_comparison(
    world: str,
    steps: int,
    seeds: int,
    sensor_suffix: str,
    readout_variables: set[str],
    hidden_context_edges: set[tuple[str, str]],
    rows: list[dict[str, object]],
) -> str:
    readout_label = ", ".join(sorted(readout_variables)) or f"*{sensor_suffix}"
    hidden_label = (
        ", ".join(f"{action}->{target}" for action, target in sorted(hidden_context_edges))
        or "none"
    )
    lines = [
        f"World: {world}",
        f"Steps per run: {steps}",
        f"Seeds: {seeds}",
        f"Readout variables: {readout_label}",
        f"Hidden-context edges: {hidden_label}",
        "",
        (
            "agent                           precision  recall  f1    "
            "avg-fp  readout-fp  non-readout-fp  hidden-r  hidden-fp  avg-edges"
        ),
        "-" * 116,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):31}"
            f"{float(row['precision']):>9.2f}"
            f"{float(row['recall']):>8.2f}"
            f"{float(row['f1']):>6.2f}"
            f"{float(row['false_positive']):>8.2f}"
            f"{float(row['readout_false_positive']):>12.2f}"
            f"{float(row['non_readout_false_positive']):>16.2f}"
            f"{float(row['hidden_context_recall']):>10.2f}"
            f"{float(row['hidden_context_false_positive']):>11.2f}"
            f"{float(row['edges']):>11.2f}"
        )
    return "\n".join(lines)


def _readout_variables(world_name: str, sensor_suffix: str) -> set[str]:
    world = make_world(world_name)
    world.reset(seed=0)
    reader = getattr(world, "readout_variables", None)
    if reader is not None:
        return set(reader())
    return {
        variable
        for variable in world.observe()
        if variable.endswith(sensor_suffix)
    }


def _hidden_context_edges(world_name: str) -> set[tuple[str, str]]:
    world = make_world(world_name)
    reader = getattr(world, "hidden_context_edges", None)
    if reader is None:
        return set()
    return set(reader())


def _hidden_context_hint_edges(
    condition_hints: dict[tuple[str, str], list[str]]
) -> set[tuple[str, str]]:
    hidden_edges: set[tuple[str, str]] = set()
    for edge, hints in condition_hints.items():
        hint_text = " ".join(hints).lower()
        if "hidden" in hint_text or "latent" in hint_text:
            hidden_edges.add(edge)
    return hidden_edges


def _intervention_comparison_row(
    agent_name: str, results: list[InterventionTestResult]
) -> dict[str, object]:
    return {
        "agent": agent_name,
        "graph_f1": _mean([result.exploration.score.f1 for result in results]),
        "exact": _mean([result.exact_match_rate for result in results]),
        "jaccard": _mean([result.mean_jaccard for result in results]),
        "precision": _mean([result.prediction_precision for result in results]),
        "recall": _mean([result.prediction_recall for result in results]),
        "f1": _mean([result.prediction_f1 for result in results]),
        "cases": _mean([len(result.cases) for result in results]),
    }


def _format_intervention_comparison(
    world: str, explore_steps: int, seeds: int, rows: list[dict[str, object]]
) -> str:
    lines = [
        f"World: {world}",
        f"Explore steps: {explore_steps}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 graph-f1  exact  jaccard  pred-p  "
            "pred-r  pred-f1  cases"
        ),
        "-" * 78,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['graph_f1']):>8.2f}"
            f"{float(row['exact']):>8.2f}"
            f"{float(row['jaccard']):>9.2f}"
            f"{float(row['precision']):>8.2f}"
            f"{float(row['recall']):>8.2f}"
            f"{float(row['f1']):>9.2f}"
            f"{float(row['cases']):>7.1f}"
        )
    return "\n".join(lines)


def _counterfactual_comparison_row(
    agent_name: str, results: list[CounterfactualTestResult]
) -> dict[str, object]:
    return {
        "agent": agent_name,
        "graph_f1": _mean([result.exploration.score.f1 for result in results]),
        "exact": _mean([result.exact_match_rate for result in results]),
        "state_accuracy": _mean([result.mean_state_accuracy for result in results]),
        "delta_jaccard": _mean([result.mean_delta_jaccard for result in results]),
        "delta_f1": _mean([result.delta_f1 for result in results]),
        "cases": _mean([len(result.cases) for result in results]),
    }


def _format_counterfactual_comparison(
    world: str, explore_steps: int, seeds: int, rows: list[dict[str, object]]
) -> str:
    lines = [
        f"World: {world}",
        f"Explore steps: {explore_steps}",
        f"Seeds: {seeds}",
        "",
        "agent                 graph-f1  exact  state-acc  delta-j  delta-f1    cases",
        "-" * 80,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['graph_f1']):>8.2f}"
            f"{float(row['exact']):>8.2f}"
            f"{float(row['state_accuracy']):>11.2f}"
            f"{float(row['delta_jaccard']):>9.2f}"
            f"{float(row['delta_f1']):>10.2f}"
            f"{float(row['cases']):>9.1f}"
        )
    return "\n".join(lines)


def _transfer_comparison_row(
    agent_name: str, results: list[TransferInterventionTestResult]
) -> dict[str, object]:
    return {
        "agent": agent_name,
        "source_graph_f1": _mean([result.exploration.score.f1 for result in results]),
        "exact": _mean([result.exact_match_rate for result in results]),
        "jaccard": _mean([result.mean_jaccard for result in results]),
        "precision": _mean([result.prediction_precision for result in results]),
        "recall": _mean([result.prediction_recall for result in results]),
        "f1": _mean([result.prediction_f1 for result in results]),
        "shifted_f1": _mean([result.shifted_prediction_f1 for result in results]),
        "cases": _mean([len(result.cases) for result in results]),
    }


def _format_transfer_comparison(
    source_world: str,
    target_world: str,
    explore_steps: int,
    seeds: int,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        f"Source world: {source_world}",
        f"Target world: {target_world}",
        f"Explore steps: {explore_steps}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 graph-f1  exact  jaccard  pred-p  "
            "pred-r  pred-f1  shift-f1  cases"
        ),
        "-" * 78,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['source_graph_f1']):>8.2f}"
            f"{float(row['exact']):>8.2f}"
            f"{float(row['jaccard']):>9.2f}"
            f"{float(row['precision']):>8.2f}"
            f"{float(row['recall']):>8.2f}"
            f"{float(row['f1']):>9.2f}"
            f"{float(row['shifted_f1']):>10.2f}"
            f"{float(row['cases']):>7.1f}"
        )
    return "\n".join(lines)


def _transfer_adaptation_row(
    agent_name: str, results: list[TransferAdaptationTestResult]
) -> dict[str, object]:
    return {
        "agent": agent_name,
        "before_f1": _mean(
            [result.before_adaptation.prediction_f1 for result in results]
        ),
        "after_f1": _mean(
            [result.after_adaptation.prediction_f1 for result in results]
        ),
        "before_shift_f1": _mean(
            [result.before_adaptation.shifted_prediction_f1 for result in results]
        ),
        "after_shift_f1": _mean(
            [result.after_adaptation.shifted_prediction_f1 for result in results]
        ),
        "after_exact": _mean(
            [result.after_adaptation.exact_match_rate for result in results]
        ),
    }


def _format_transfer_adaptation_comparison(
    source_world: str,
    target_world: str,
    explore_steps: int,
    adapt_steps: int,
    seeds: int,
    adaptation_mode: str,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        f"Source world: {source_world}",
        f"Target world: {target_world}",
        f"Explore steps: {explore_steps}",
        f"Adapt steps: {adapt_steps}",
        f"Adaptation mode: {adaptation_mode}",
        f"Seeds: {seeds}",
        "",
        "agent                 before-f1  after-f1  shift-before  shift-after  exact-after",
        "-" * 82,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['before_f1']):>9.2f}"
            f"{float(row['after_f1']):>10.2f}"
            f"{float(row['before_shift_f1']):>14.2f}"
            f"{float(row['after_shift_f1']):>13.2f}"
            f"{float(row['after_exact']):>13.2f}"
        )
    return "\n".join(lines)


def _level3_comparison_row(
    agent_name: str, results: list[Level3TestResult]
) -> dict[str, object]:
    return {
        "agent": agent_name,
        "counterfactual_delta_f1": _mean(
            [result.counterfactual_delta_f1 for result in results]
        ),
        "stable_transfer_f1": _mean(
            [result.stable_transfer_f1 for result in results]
        ),
        "repair_before_shift_f1": _mean(
            [result.repair_before_shift_f1 for result in results]
        ),
        "repair_after_shift_f1": _mean(
            [result.repair_after_shift_f1 for result in results]
        ),
        "repair_shift_gain": _mean([result.repair_shift_gain for result in results]),
        "level3_score": _mean([result.level3_score for result in results]),
    }


def _format_level3_comparison(
    source_world: str,
    stable_target_world: str,
    repair_target_world: str,
    explore_steps: int,
    adapt_steps: int,
    seeds: int,
    case_source: str,
    adaptation_mode: str,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Level 3 ladder",
        f"Source world: {source_world}",
        f"Stable transfer target: {stable_target_world}",
        f"Repair target: {repair_target_world}",
        f"Explore steps: {explore_steps}",
        f"Adapt steps: {adapt_steps}",
        f"Counterfactual cases: {case_source}",
        f"Adaptation mode: {adaptation_mode}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 cf-delta-f1  transfer-f1  repair-before  "
            "repair-after  repair-gain  level3"
        ),
        "-" * 94,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['counterfactual_delta_f1']):>12.2f}"
            f"{float(row['stable_transfer_f1']):>13.2f}"
            f"{float(row['repair_before_shift_f1']):>15.2f}"
            f"{float(row['repair_after_shift_f1']):>14.2f}"
            f"{float(row['repair_shift_gain']):>13.2f}"
            f"{float(row['level3_score']):>8.2f}"
        )
    return "\n".join(lines)


def _level4_comparison_row(
    agent_name: str,
    results: list[Level4DisambiguationResult],
    max_steps: int,
) -> dict[str, object]:
    no_success_step = max_steps + 1
    return {
        "agent": agent_name,
        "first_disambiguating_step": _mean(
            [
                result.first_disambiguating_step or no_success_step
                for result in results
            ]
        ),
        "first_correct_step": _mean(
            [result.first_correct_step or no_success_step for result in results]
        ),
        "success_rate": _mean([1 if result.final_correct else 0 for result in results]),
        "disambiguating_tests": _mean(
            [result.disambiguating_tests for result in results]
        ),
        "graph_f1": _mean([result.exploration.score.f1 for result in results]),
        "level4_score": _mean([result.level4_score for result in results]),
    }


def _format_level4_comparison(
    world: str,
    steps: int,
    seeds: int,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Level 4 active disambiguation",
        f"World: {world}",
        f"Steps per run: {steps}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 first-test  first-correct  success  "
            "tests  graph-f1  level4"
        ),
        "-" * 82,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['first_disambiguating_step']):>11.2f}"
            f"{float(row['first_correct_step']):>15.2f}"
            f"{float(row['success_rate']):>9.2f}"
            f"{float(row['disambiguating_tests']):>7.2f}"
            f"{float(row['graph_f1']):>10.2f}"
            f"{float(row['level4_score']):>8.2f}"
        )
    return "\n".join(lines)


def _temporal_credit_row(
    agent_name: str,
    results: list[TemporalCreditResult],
    max_steps: int,
) -> dict[str, object]:
    no_success_step = max_steps + 1
    return {
        "agent": agent_name,
        "first_correct_step": _mean(
            [result.first_correct_step or no_success_step for result in results]
        ),
        "success_rate": _mean([1 if result.final_correct else 0 for result in results]),
        "delayed_edge_rate": _mean(
            [1 if result.delayed_edge_learned else 0 for result in results]
        ),
        "followup_misattribution_rate": _mean(
            [1 if result.followup_misattribution else 0 for result in results]
        ),
        "graph_f1": _mean([result.exploration.score.f1 for result in results]),
        "temporal_score": _mean([result.temporal_score for result in results]),
    }


def _format_temporal_credit_comparison(
    world: str,
    steps: int,
    seeds: int,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Temporal causal credit",
        f"World: {world}",
        f"Steps per run: {steps}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 first-correct  success  delayed-edge  "
            "wait-misattr  graph-f1  temporal"
        ),
        "-" * 94,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['first_correct_step']):>15.2f}"
            f"{float(row['success_rate']):>9.2f}"
            f"{float(row['delayed_edge_rate']):>14.2f}"
            f"{float(row['followup_misattribution_rate']):>14.2f}"
            f"{float(row['graph_f1']):>10.2f}"
            f"{float(row['temporal_score']):>10.2f}"
        )
    return "\n".join(lines)


def _temporal_transfer_row(
    agent_name: str,
    results: list[TemporalTransferResult],
) -> dict[str, object]:
    return {
        "agent": agent_name,
        "target_success": _mean([1 if result.final_correct else 0 for result in results]),
        "delayed_edge_rate": _mean(
            [1 if result.delayed_edge_learned else 0 for result in results]
        ),
        "followup_misattribution_rate": _mean(
            [1 if result.followup_misattribution else 0 for result in results]
        ),
        "mechanism_shift_rate": _mean(
            [1 if result.mechanism_shifted else 0 for result in results]
        ),
        "source_graph_f1": _mean([result.exploration.score.f1 for result in results]),
        "transfer_score": _mean(
            [result.temporal_transfer_score for result in results]
        ),
    }


def _format_temporal_transfer_comparison(
    source_world: str,
    target_world: str,
    explore_steps: int,
    seeds: int,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Temporal causal transfer",
        f"Source world: {source_world}",
        f"Target world: {target_world}",
        f"Explore steps: {explore_steps}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 target-success  delayed-edge  wait-misattr  "
            "mech-shift  graph-f1  transfer"
        ),
        "-" * 98,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['target_success']):>15.2f}"
            f"{float(row['delayed_edge_rate']):>14.2f}"
            f"{float(row['followup_misattribution_rate']):>14.2f}"
            f"{float(row['mechanism_shift_rate']):>12.2f}"
            f"{float(row['source_graph_f1']):>10.2f}"
            f"{float(row['transfer_score']):>10.2f}"
        )
    return "\n".join(lines)


def _temporal_adaptation_row(
    agent_name: str,
    results: list[TemporalAdaptationResult],
) -> dict[str, object]:
    return {
        "agent": agent_name,
        "before_score": _mean(
            [result.before_adaptation.temporal_transfer_score for result in results]
        ),
        "after_score": _mean(
            [result.after_adaptation.temporal_transfer_score for result in results]
        ),
        "repair_gain": _mean([result.repair_gain for result in results]),
        "after_success": _mean(
            [1 if result.after_adaptation.final_correct else 0 for result in results]
        ),
        "after_delayed_edge": _mean(
            [
                1 if result.after_adaptation.delayed_edge_learned else 0
                for result in results
            ]
        ),
        "after_wait_misattribution": _mean(
            [
                1 if result.after_adaptation.followup_misattribution else 0
                for result in results
            ]
        ),
        "mechanism_shift": _mean(
            [1 if result.after_adaptation.mechanism_shifted else 0 for result in results]
        ),
    }


def _format_temporal_adaptation_comparison(
    source_world: str,
    target_world: str,
    explore_steps: int,
    adapt_steps: int,
    seeds: int,
    adaptation_mode: str,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Temporal mechanism repair",
        f"Source world: {source_world}",
        f"Target world: {target_world}",
        f"Explore steps: {explore_steps}",
        f"Adapt steps: {adapt_steps}",
        f"Adaptation mode: {adaptation_mode}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 before  after  gain  after-success  "
            "delayed-edge  wait-misattr  mech-shift"
        ),
        "-" * 100,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['before_score']):>7.2f}"
            f"{float(row['after_score']):>7.2f}"
            f"{float(row['repair_gain']):>6.2f}"
            f"{float(row['after_success']):>15.2f}"
            f"{float(row['after_delayed_edge']):>14.2f}"
            f"{float(row['after_wait_misattribution']):>14.2f}"
            f"{float(row['mechanism_shift']):>12.2f}"
        )
    return "\n".join(lines)


def _temporal_selective_adaptation_row(
    agent_name: str,
    results: list[TemporalSelectiveAdaptationResult],
) -> dict[str, object]:
    after_targets = [
        target for result in results for target in result.after_targets
    ]
    return {
        "agent": agent_name,
        "before_score": _mean([result.before_score for result in results]),
        "after_score": _mean([result.after_score for result in results]),
        "repair_gain": _mean([result.repair_gain for result in results]),
        "all_success": _mean(
            [1 if result.after_all_success else 0 for result in results]
        ),
        "shifted_before": _mean([result.shifted_before_score for result in results]),
        "shifted_after": _mean([result.shifted_after_score for result in results]),
        "stable_after": _mean([result.stable_after_score for result in results]),
        "selective_score": _mean([result.selective_score for result in results]),
        "wait_misattribution": _mean(
            [1 if target.followup_misattribution else 0 for target in after_targets]
        ),
    }


def _format_temporal_selective_adaptation_comparison(
    source_world: str,
    target_world: str,
    explore_steps: int,
    adapt_steps: int,
    seeds: int,
    adaptation_mode: str,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Temporal selective mechanism repair",
        f"Source world: {source_world}",
        f"Target world: {target_world}",
        f"Explore steps: {explore_steps}",
        f"Adapt steps: {adapt_steps}",
        f"Adaptation mode: {adaptation_mode}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 before  after  gain  all-success  "
            "shift-before  shift-after  stable-after  selective  wait-misattr"
        ),
        "-" * 117,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['before_score']):>7.2f}"
            f"{float(row['after_score']):>7.2f}"
            f"{float(row['repair_gain']):>6.2f}"
            f"{float(row['all_success']):>13.2f}"
            f"{float(row['shifted_before']):>14.2f}"
            f"{float(row['shifted_after']):>13.2f}"
            f"{float(row['stable_after']):>14.2f}"
            f"{float(row['selective_score']):>11.2f}"
            f"{float(row['wait_misattribution']):>14.2f}"
        )
    return "\n".join(lines)


def _level6_schema_row(
    agent_name: str,
    results: list[Level6SchemaTransferResult],
) -> dict[str, object]:
    targets = [target for result in results for target in result.targets]
    return {
        "agent": agent_name,
        "target_score": _mean([result.target_score for result in results]),
        "all_success": _mean([1 if result.all_success else 0 for result in results]),
        "delayed_edge_rate": _mean(
            [result.delayed_edge_rate for result in results]
        ),
        "followup_misattribution": _mean(
            [result.followup_misattribution_rate for result in results]
        ),
        "level6_score": _mean([result.level6_score for result in results]),
        "target_count": _mean([len(result.targets) for result in results]),
        "target_final_correct": _mean(
            [1 if target.final_correct else 0 for target in targets]
        ),
    }


def _format_level6_schema_comparison(
    source_world: str,
    target_world: str,
    explore_steps: int,
    target_steps: int,
    seeds: int,
    transfer_mode: str,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Level 6 schema transfer",
        f"Source world: {source_world}",
        f"Target world: {target_world}",
        f"Explore steps: {explore_steps}",
        f"Target steps: {target_steps}",
        f"Transfer mode: {transfer_mode}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                            target  all-success  "
            "delayed-edge  followup-misattr  final-correct  level6"
        ),
        "-" * 111,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):32}"
            f"{float(row['target_score']):>8.2f}"
            f"{float(row['all_success']):>13.2f}"
            f"{float(row['delayed_edge_rate']):>14.2f}"
            f"{float(row['followup_misattribution']):>18.2f}"
            f"{float(row['target_final_correct']):>15.2f}"
            f"{float(row['level6_score']):>9.2f}"
        )
    return "\n".join(lines)


def _level6_schema_repair_row(
    agent_name: str,
    results: list[Level6SchemaRepairResult],
) -> dict[str, object]:
    after_targets = [target for result in results for target in result.after_targets]
    return {
        "agent": agent_name,
        "before_score": _mean([result.before_score for result in results]),
        "after_score": _mean([result.after_score for result in results]),
        "repair_gain": _mean([result.repair_gain for result in results]),
        "all_success": _mean([1 if result.all_success else 0 for result in results]),
        "shifted_before": _mean([result.shifted_before_score for result in results]),
        "shifted_after": _mean([result.shifted_after_score for result in results]),
        "stable_before": _mean([result.stable_before_score for result in results]),
        "stable_after": _mean([result.stable_after_score for result in results]),
        "delayed_edge_rate": _mean([result.delayed_edge_rate for result in results]),
        "followup_misattribution": _mean(
            [result.followup_misattribution_rate for result in results]
        ),
        "level6_repair": _mean(
            [result.level6_repair_score for result in results]
        ),
        "target_final_correct": _mean(
            [1 if target.final_correct else 0 for target in after_targets]
        ),
    }


def _format_level6_schema_repair_comparison(
    source_world: str,
    target_world: str,
    explore_steps: int,
    target_steps: int,
    seeds: int,
    transfer_mode: str,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Level 6 schema transfer with selective repair",
        f"Source world: {source_world}",
        f"Target world: {target_world}",
        f"Explore steps: {explore_steps}",
        f"Target steps: {target_steps}",
        f"Transfer mode: {transfer_mode}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                            before  after  gain  all-success  "
            "shift-before  shift-after  stable-before  stable-after  "
            "delayed-edge  followup-misattr  l6-repair"
        ),
        "-" * 157,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):32}"
            f"{float(row['before_score']):>8.2f}"
            f"{float(row['after_score']):>7.2f}"
            f"{float(row['repair_gain']):>6.2f}"
            f"{float(row['all_success']):>13.2f}"
            f"{float(row['shifted_before']):>14.2f}"
            f"{float(row['shifted_after']):>13.2f}"
            f"{float(row['stable_before']):>15.2f}"
            f"{float(row['stable_after']):>14.2f}"
            f"{float(row['delayed_edge_rate']):>14.2f}"
            f"{float(row['followup_misattribution']):>18.2f}"
            f"{float(row['level6_repair']):>11.2f}"
        )
    return "\n".join(lines)


def _format_level6_active_diagnostic_comparison(
    source_world: str,
    target_world: str,
    explore_steps: int,
    repair_steps: int,
    seeds: int,
    transfer_mode: str,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Level 6 active diagnostic schema repair",
        f"Source world: {source_world}",
        f"Target world: {target_world}",
        f"Explore steps: {explore_steps}",
        f"Scripted diagnostic exposure: shifted action + learned followup",
        f"Free repair steps after exposure: {repair_steps}",
        f"Transfer mode: {transfer_mode}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                            before  after  gain  all-success  "
            "shift-before  shift-after  stable-before  stable-after  "
            "delayed-edge  followup-misattr  l6-repair"
        ),
        "-" * 157,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):32}"
            f"{float(row['before_score']):>8.2f}"
            f"{float(row['after_score']):>7.2f}"
            f"{float(row['repair_gain']):>6.2f}"
            f"{float(row['all_success']):>13.2f}"
            f"{float(row['shifted_before']):>14.2f}"
            f"{float(row['shifted_after']):>13.2f}"
            f"{float(row['stable_before']):>15.2f}"
            f"{float(row['stable_after']):>14.2f}"
            f"{float(row['delayed_edge_rate']):>14.2f}"
            f"{float(row['followup_misattribution']):>18.2f}"
            f"{float(row['level6_repair']):>11.2f}"
        )
    return "\n".join(lines)


def _format_level6_procedural_diagnostic_comparison(
    families: int,
    mechanisms: int,
    explore_steps: int,
    repair_steps: int,
    transfer_mode: str,
    readout_mode: str,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Level 6 procedural active diagnostic schema repair",
        f"Generated SCM families: {families}",
        f"Mechanisms per family: {mechanisms}",
        f"Explore steps: {explore_steps}",
        f"Scripted diagnostic exposure: shifted action + learned followup",
        f"Free repair steps after exposure: {repair_steps}",
        f"Transfer mode: {transfer_mode}",
        f"Readout mode: {readout_mode}",
        "",
        (
            "agent                            before  after  gain  all-success  "
            "shift-before  shift-after  stable-before  stable-after  "
            "delayed-edge  followup-misattr  l6-repair"
        ),
        "-" * 157,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):32}"
            f"{float(row['before_score']):>8.2f}"
            f"{float(row['after_score']):>7.2f}"
            f"{float(row['repair_gain']):>6.2f}"
            f"{float(row['all_success']):>13.2f}"
            f"{float(row['shifted_before']):>14.2f}"
            f"{float(row['shifted_after']):>13.2f}"
            f"{float(row['stable_before']):>15.2f}"
            f"{float(row['stable_after']):>14.2f}"
            f"{float(row['delayed_edge_rate']):>14.2f}"
            f"{float(row['followup_misattribution']):>18.2f}"
            f"{float(row['level6_repair']):>11.2f}"
        )
    return "\n".join(lines)


def _repair_efficiency_row(
    agent_name: str,
    structural_results: list[TransferAdaptationTestResult],
    fresh_results: list[TransferAdaptationTestResult],
) -> dict[str, object]:
    structural_after_f1 = _mean(
        [result.after_adaptation.prediction_f1 for result in structural_results]
    )
    fresh_after_f1 = _mean(
        [result.after_adaptation.prediction_f1 for result in fresh_results]
    )
    structural_shift_f1 = _mean(
        [
            result.after_adaptation.shifted_prediction_f1
            for result in structural_results
        ]
    )
    fresh_shift_f1 = _mean(
        [result.after_adaptation.shifted_prediction_f1 for result in fresh_results]
    )
    structural_exact = _mean(
        [result.after_adaptation.exact_match_rate for result in structural_results]
    )
    fresh_exact = _mean(
        [result.after_adaptation.exact_match_rate for result in fresh_results]
    )
    return {
        "agent": agent_name,
        "structural_after_f1": structural_after_f1,
        "fresh_after_f1": fresh_after_f1,
        "after_f1_gain": structural_after_f1 - fresh_after_f1,
        "structural_shift_f1": structural_shift_f1,
        "fresh_shift_f1": fresh_shift_f1,
        "shift_f1_gain": structural_shift_f1 - fresh_shift_f1,
        "structural_exact": structural_exact,
        "fresh_exact": fresh_exact,
        "exact_gain": structural_exact - fresh_exact,
    }


def _format_repair_efficiency_comparison(
    source_world: str,
    target_world: str,
    explore_steps: int,
    adapt_steps: int,
    seeds: int,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Repair efficiency",
        f"Source world: {source_world}",
        f"Target world: {target_world}",
        f"Explore steps: {explore_steps}",
        f"Adapt steps: {adapt_steps}",
        f"Seeds: {seeds}",
        "",
        (
            "agent                 prior-f1  fresh-f1  f1-gain  "
            "prior-shift  fresh-shift  shift-gain  exact-gain"
        ),
        "-" * 96,
    ]
    for row in rows:
        lines.append(
            f"{str(row['agent']):21}"
            f"{float(row['structural_after_f1']):>9.2f}"
            f"{float(row['fresh_after_f1']):>10.2f}"
            f"{float(row['after_f1_gain']):>9.2f}"
            f"{float(row['structural_shift_f1']):>13.2f}"
            f"{float(row['fresh_shift_f1']):>13.2f}"
            f"{float(row['shift_f1_gain']):>12.2f}"
            f"{float(row['exact_gain']):>12.2f}"
        )
    return "\n".join(lines)


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
