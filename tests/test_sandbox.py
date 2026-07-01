import unittest

from causal_sandbox.agents import (
    ActiveCausalAgent,
    CausalObservationAdapterV2Agent,
    CausalCoreAgent,
    DiagnosticPortableTemporalCausalCoreAgent,
    HiddenContextObservationAdapterAgent,
    HypothesisTestingCausalAgent,
    LatentContextObservationAdapterAgent,
    LearnedObservationAdapterCausalCoreAgent,
    ObservationAdapterCausalCoreAgent,
    PassiveCorrelationAgent,
    PersistentLatentContextObservationAdapterAgent,
    PortableTemporalCausalCoreAgent,
    ProactiveLatentContextObservationAdapterAgent,
    RankingLossPredictorAgent,
    ControlExperimentPlannerLatentContextAgent,
    ContextSearchControlExperimentPlannerAgent,
    UnguardedContextSearchControlExperimentPlannerAgent,
    RewardSeekingAgent,
    RewardTransferAgent,
    StatefulLatentContextObservationAdapterAgent,
    TemporalCausalCoreAgent,
    UnsafeReadoutDiagnosticPortableTemporalCausalCoreAgent,
)
from causal_sandbox.core import Transition
from causal_sandbox.continuous import run_continuous_metric_poc
from causal_sandbox.evaluation import (
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
    score_edges,
)
from causal_sandbox.llm import (
    LLMAuthoritativeCausalModuleAgent,
    LLMControllerAgent,
    LLMGatedCausalModuleAgent,
    LLMCausalModuleAgent,
)
from causal_sandbox.worlds import (
    AmbiguousGateWorld,
    AmbiguousPanelGateWorld,
    DelayedLampLongDelayWorld,
    DelayedLampWorld,
    DelayedLampShiftedWorld,
    DoorLampInvertedWorld,
    DoorLampShiftedWorld,
    DoorLampWorld,
    DualDelayedControlWorld,
    DualDelayedSelectiveShiftWorld,
    DerivedSensorPanelWorld,
    OpaqueReadoutPanelWorld,
    PanelInvertedWorld,
    NoisyHiddenPanelWorld,
    NoisyCorePanelWorld,
    HiddenContextPanelWorld,
    ComplexNoisyHiddenPanelWorld,
    PanelWorld,
    make_procedural_complex_hidden_world,
    RenamedDualDelayedControlWorld,
    RenamedDualDelayedSelectiveShiftWorld,
    RenamedTripleDelayedDiagnosticShiftWorld,
    TripleDelayedControlWorld,
    make_procedural_diagnostic_world_pair,
)
from causal_sandbox.observation import (
    LatentContextObservationAdapter,
    PersistentLatentContextObservationAdapter,
    StatefulLatentContextObservationAdapter,
)
from causal_sandbox.ranking_baseline import run_ranking_loss_baseline_experiment
from causal_sandbox.stress_suite import run_matched_mechanism_baseline_suite


class ScriptedTextGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, system, user):
        self.prompts.append((system, user))
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[index]


class CausalSandboxTests(unittest.TestCase):
    def test_door_lamp_conditional_effects(self):
        world = DoorLampWorld()
        world.reset(seed=1)

        world.step("set_bright")
        bright_after = world.step("press_a")
        self.assertIs(bright_after["sound"], True)
        self.assertIs(bright_after["lamp_on"], False)

        world.step("wait")
        world.step("set_dark")
        dark_after = world.step("press_a")
        self.assertIs(dark_after["sound"], True)
        self.assertIs(dark_after["lamp_on"], True)

    def test_active_agent_recovers_core_edges(self):
        result = run_episode(DoorLampWorld(), ActiveCausalAgent(), steps=20, seed=1)

        self.assertIn(("press_a", "lamp_on"), result.discovered_edges)
        self.assertIn(("press_a", "sound"), result.discovered_edges)
        self.assertIn(("press_b", "door_open"), result.discovered_edges)
        self.assertIn(("press_b", "sound"), result.discovered_edges)
        self.assertIn(("open_door", "door_open"), result.discovered_edges)
        self.assertGreaterEqual(result.score.recall, 0.9)

    def test_score_edges_counts_false_positive(self):
        score = score_edges(
            discovered_edges={("a", "x"), ("a", "noise")},
            true_edges={("a", "x"), ("b", "y")},
            transitions=(),
        )

        self.assertEqual(score.true_positive, 1)
        self.assertEqual(score.false_positive, 1)
        self.assertEqual(score.false_negative, 1)
        self.assertEqual(score.precision, 0.5)
        self.assertEqual(score.recall, 0.5)

    def test_ranking_loss_predictor_has_no_mechanism_edges(self):
        result = run_episode(DoorLampWorld(), RankingLossPredictorAgent(seed=1), steps=12, seed=1)

        self.assertEqual(result.discovered_edges, set())
        self.assertEqual(result.score.recall, 0.0)

    def test_ranking_loss_baseline_smoke(self):
        payload = run_ranking_loss_baseline_experiment(
            families=1,
            seeds=1,
            train_states=8,
            test_states=8,
            hidden_steps=20,
            epochs=2,
        )

        self.assertEqual(payload["model"], "pairwise-ranking-loss-predictor")
        self.assertEqual(len(payload["summary"]), 3)

    def test_matched_baseline_suite_includes_conditional_discovery(self):
        payload = run_matched_mechanism_baseline_suite(
            families=1,
            seeds=1,
            steps=40,
            mechanisms=2,
            visible=2,
            readouts=2,
            neural_epochs=2,
        )

        methods = {
            row["method"]
            for row in payload["matched_mechanism_baselines"]["summary"]
        }
        self.assertIn("conditional-discovery", methods)
        self.assertIn("conditional-discovery-readout-oracle", methods)
        self.assertIn("latent-world-model", methods)
        self.assertIn("latent-world-model-readout-oracle", methods)

    def test_continuous_metric_poc_smoke(self):
        payload = run_continuous_metric_poc(
            seeds=2,
            random_trials=30,
            regression_trials=40,
            paired_samples=4,
        )

        self.assertEqual(len(payload["summary"]), 3)
        rows = {row["agent"]: row for row in payload["summary"]}
        self.assertGreaterEqual(
            rows["metric-causal-core"]["f1"],
            rows["random-correlation"]["f1"],
        )
        self.assertEqual(rows["metric-causal-core"]["readout_false_positive"], 0)

    def test_passive_correlation_baseline_learns_spurious_edges(self):
        result = run_episode(DoorLampWorld(), PassiveCorrelationAgent(), steps=20, seed=1)

        self.assertIn(("dark", "lamp_on"), result.discovered_edges)
        self.assertGreater(result.score.false_positive, 0)
        self.assertEqual(result.score.true_positive, 0)

    def test_causal_core_agent_selects_its_own_interventions(self):
        result = run_episode(DoorLampWorld(), CausalCoreAgent(), steps=20, seed=1)

        self.assertIn(("press_a", "lamp_on"), result.discovered_edges)
        self.assertIn(("press_b", "door_open"), result.discovered_edges)
        self.assertIn(("reset_lamp", "lamp_on"), result.discovered_edges)
        self.assertEqual(result.score.recall, 1.0)

    def test_intervention_test_freezes_after_exploration(self):
        result = run_intervention_test(
            DoorLampWorld(), CausalCoreAgent(), explore_steps=20, seed=1
        )

        self.assertGreater(len(result.cases), 0)
        self.assertEqual(result.exact_match_rate, 1.0)
        self.assertEqual(result.prediction_f1, 1.0)

    def test_causal_core_learns_context_gate(self):
        result = run_episode(DoorLampWorld(), CausalCoreAgent(), steps=20, seed=1)

        self.assertIn(
            "dark=True", result.condition_hints[("press_a", "lamp_on")]
        )

    def test_counterfactual_test_after_exploration(self):
        result = run_counterfactual_test(
            DoorLampWorld(), CausalCoreAgent(), explore_steps=20, seed=1
        )

        self.assertGreater(len(result.cases), 0)
        self.assertGreater(result.exact_match_rate, 0.95)
        self.assertGreater(result.delta_f1, 0.95)

    def test_counterfactual_test_all_states_source(self):
        result = run_counterfactual_test(
            DoorLampWorld(),
            CausalCoreAgent(),
            explore_steps=20,
            seed=1,
            case_source="all-states",
        )

        self.assertEqual(result.exact_match_rate, 1.0)
        self.assertEqual(result.delta_f1, 1.0)

    def test_transfer_intervention_test_shifted_world(self):
        result = run_transfer_intervention_test(
            DoorLampWorld(),
            DoorLampShiftedWorld(),
            CausalCoreAgent(),
            explore_steps=20,
            seed=1,
        )

        self.assertEqual(result.exact_match_rate, 1.0)
        self.assertEqual(result.prediction_f1, 1.0)

    def test_inverted_world_changes_press_a_context(self):
        world = DoorLampInvertedWorld()
        world.reset(seed=1)

        world.step("set_dark")
        dark_after = world.step("press_a")
        self.assertIs(dark_after["lamp_on"], False)

        world.step("wait")
        world.step("set_bright")
        bright_after = world.step("press_a")
        self.assertIs(bright_after["lamp_on"], True)

    def test_ambiguous_gate_requires_disambiguating_context(self):
        world = AmbiguousGateWorld()
        world.reset(seed=1)

        first_press = world.step("press_a")
        self.assertIs(first_press["lamp_on"], True)

        world.step("set_bright")
        world.step("reset_lamp")
        second_press = world.step("press_a")
        self.assertIs(second_press["door_open"], True)
        self.assertIs(second_press["dark"], False)
        self.assertIs(second_press["lamp_on"], False)

    def test_transfer_adaptation_with_structural_prior(self):
        result = run_transfer_adaptation_test(
            DoorLampWorld(),
            DoorLampInvertedWorld(),
            CausalCoreAgent(),
            explore_steps=20,
            adapt_steps=12,
            seed=1,
            adaptation_mode="structural-prior",
        )

        self.assertGreaterEqual(
            result.after_adaptation.shifted_prediction_f1,
            result.before_adaptation.shifted_prediction_f1,
        )

    def test_panel_world_press_c_gate(self):
        world = PanelWorld()
        world.reset(seed=1)

        world.step("open_door")
        world.step("stop_fan")
        after_source_condition = world.step("press_c")
        self.assertIs(after_source_condition["alarm_on"], True)

        inverted = PanelInvertedWorld()
        inverted.reset(seed=1)
        inverted.step("open_door")
        inverted.step("stop_fan")
        after_inverted_condition = inverted.step("press_c")
        self.assertIs(after_inverted_condition["alarm_on"], False)

    def test_ambiguous_panel_gate_requires_conjunctive_context(self):
        world = AmbiguousPanelGateWorld()
        world.reset(seed=1)

        first_press = world.step("press_c")
        self.assertIs(first_press["alarm_on"], True)

        world.step("reset_alarm")
        world.step("close_door")
        missing_door = world.step("press_c")
        self.assertIs(missing_door["door_open"], False)
        self.assertIs(missing_door["fan_on"], False)
        self.assertIs(missing_door["alarm_on"], False)

        world.step("open_door")
        world.step("start_fan")
        fan_blocks = world.step("press_c")
        self.assertIs(fan_blocks["door_open"], True)
        self.assertIs(fan_blocks["fan_on"], True)
        self.assertIs(fan_blocks["alarm_on"], False)

    def test_panel_transfer_adaptation_smoke(self):
        result = run_transfer_adaptation_test(
            PanelWorld(),
            PanelInvertedWorld(),
            CausalCoreAgent(),
            explore_steps=60,
            adapt_steps=40,
            seed=1,
            adaptation_mode="structural-prior",
        )

        self.assertGreater(
            result.after_adaptation.shifted_prediction_f1,
            result.before_adaptation.shifted_prediction_f1,
        )

    def test_level3_ladder_separates_mechanisms_from_correlation(self):
        causal_result = run_level3_test(
            DoorLampWorld(),
            DoorLampShiftedWorld(),
            DoorLampInvertedWorld(),
            CausalCoreAgent(),
            explore_steps=20,
            adapt_steps=12,
            seed=1,
            case_source="all-states",
        )
        passive_result = run_level3_test(
            DoorLampWorld(),
            DoorLampShiftedWorld(),
            DoorLampInvertedWorld(),
            PassiveCorrelationAgent(),
            explore_steps=20,
            adapt_steps=12,
            seed=1,
            case_source="all-states",
        )

        self.assertEqual(causal_result.counterfactual_delta_f1, 1.0)
        self.assertEqual(causal_result.stable_transfer_f1, 1.0)
        self.assertGreaterEqual(
            causal_result.repair_after_shift_f1,
            causal_result.repair_before_shift_f1,
        )
        self.assertGreater(causal_result.level3_score, passive_result.level3_score)

    def test_structural_prior_repairs_faster_than_fresh_relearning(self):
        structural = run_transfer_adaptation_test(
            PanelWorld(),
            PanelInvertedWorld(),
            CausalCoreAgent(),
            explore_steps=60,
            adapt_steps=12,
            seed=1,
            adaptation_mode="structural-prior",
        )
        fresh = run_transfer_adaptation_test(
            PanelWorld(),
            PanelInvertedWorld(),
            CausalCoreAgent(),
            explore_steps=60,
            adapt_steps=12,
            seed=1,
            adaptation_mode="fresh",
        )

        self.assertGreater(
            structural.after_adaptation.prediction_f1,
            fresh.after_adaptation.prediction_f1,
        )
        self.assertGreater(
            structural.after_adaptation.shifted_prediction_f1,
            fresh.after_adaptation.shifted_prediction_f1,
        )

    def test_reward_rl_optimizes_return_without_matching_causal_core(self):
        causal = run_reward_episode(
            PanelWorld(), CausalCoreAgent(), steps=60, seed=1
        )
        reward_rl = run_reward_episode(
            PanelWorld(), RewardSeekingAgent(seed=1), steps=60, seed=1
        )

        self.assertGreater(reward_rl.total_return, causal.total_return)
        self.assertLess(
            reward_rl.exploration.score.f1,
            causal.exploration.score.f1,
        )

    def test_reward_transfer_does_not_replace_mechanism_repair(self):
        causal = run_level3_test(
            PanelWorld(),
            PanelWorld(),
            PanelInvertedWorld(),
            CausalCoreAgent(),
            explore_steps=60,
            adapt_steps=12,
            seed=1,
            case_source="history",
        )
        reward_transfer = run_level3_test(
            PanelWorld(),
            PanelWorld(),
            PanelInvertedWorld(),
            RewardTransferAgent(seed=1),
            explore_steps=60,
            adapt_steps=12,
            seed=1,
            case_source="history",
        )

        self.assertGreater(causal.level3_score, reward_transfer.level3_score)
        self.assertGreater(
            causal.repair_after_shift_f1,
            reward_transfer.repair_after_shift_f1,
        )

    def test_level4_active_agent_disambiguates_competing_gates(self):
        active = run_level4_disambiguation_test(
            AmbiguousGateWorld(),
            HypothesisTestingCausalAgent(),
            steps=8,
            seed=1,
        )
        passive_core = run_level4_disambiguation_test(
            AmbiguousGateWorld(),
            CausalCoreAgent(),
            steps=8,
            seed=1,
        )

        self.assertTrue(active.final_correct)
        self.assertEqual(active.first_correct_step, 5)
        self.assertEqual(active.first_disambiguating_step, 5)
        self.assertFalse(passive_core.final_correct)
        self.assertGreater(active.level4_score, passive_core.level4_score)
        self.assertIn(
            "dark=True",
            active.exploration.condition_hints[("press_a", "lamp_on")],
        )

    def test_level4_active_agent_learns_panel_conjunctive_gate(self):
        active = run_level4_disambiguation_test(
            AmbiguousPanelGateWorld(),
            HypothesisTestingCausalAgent(),
            steps=12,
            seed=1,
        )
        passive_core = run_level4_disambiguation_test(
            AmbiguousPanelGateWorld(),
            CausalCoreAgent(),
            steps=12,
            seed=1,
        )

        self.assertTrue(active.final_correct)
        self.assertEqual(active.first_correct_step, 8)
        self.assertEqual(active.first_disambiguating_step, 5)
        self.assertFalse(passive_core.final_correct)
        self.assertGreater(active.level4_score, passive_core.level4_score)
        hints = active.exploration.condition_hints[("press_c", "alarm_on")]
        self.assertIn("door_open=True", hints)
        self.assertIn("fan_on=False", hints)

    def test_delayed_lamp_requires_temporal_credit(self):
        world = DelayedLampWorld()
        world.reset(seed=1)

        immediate = world.step("press_delay")
        self.assertIs(immediate["sound"], True)
        self.assertIs(immediate["lamp_on"], False)

        delayed = world.step("wait")
        self.assertIs(delayed["sound"], False)
        self.assertIs(delayed["lamp_on"], True)
        self.assertIn(("press_delay", "lamp_on"), world.true_edges())

    def test_temporal_agent_assigns_delayed_effect_to_trigger(self):
        temporal = run_temporal_credit_test(
            DelayedLampWorld(),
            TemporalCausalCoreAgent(),
            steps=8,
            seed=1,
        )
        passive_core = run_temporal_credit_test(
            DelayedLampWorld(),
            CausalCoreAgent(),
            steps=8,
            seed=1,
        )

        self.assertTrue(temporal.final_correct)
        self.assertEqual(temporal.first_correct_step, 2)
        self.assertTrue(temporal.delayed_edge_learned)
        self.assertFalse(temporal.followup_misattribution)
        self.assertIn(
            "after wait x1",
            temporal.exploration.condition_hints[("press_delay", "lamp_on")],
        )
        self.assertFalse(passive_core.final_correct)
        self.assertTrue(passive_core.followup_misattribution)
        self.assertGreater(temporal.temporal_score, passive_core.temporal_score)

    def test_delayed_lamp_shifted_preserves_temporal_mechanism(self):
        world = DelayedLampShiftedWorld()
        observation = world.reset(seed=2)
        self.assertIs(observation["lamp_on"], True)
        self.assertIs(observation["sound"], True)

        world.step("reset_lamp")
        immediate = world.step("press_delay")
        self.assertIs(immediate["lamp_on"], False)
        delayed = world.step("wait")
        self.assertIs(delayed["lamp_on"], True)
        self.assertEqual(world.true_edges(), DelayedLampWorld().true_edges())

    def test_temporal_agent_transfers_delayed_credit_to_shifted_world(self):
        temporal = run_temporal_transfer_test(
            DelayedLampWorld(),
            DelayedLampShiftedWorld(),
            TemporalCausalCoreAgent(),
            explore_steps=8,
            seed=1,
        )
        passive_core = run_temporal_transfer_test(
            DelayedLampWorld(),
            DelayedLampShiftedWorld(),
            CausalCoreAgent(),
            explore_steps=8,
            seed=1,
        )

        self.assertTrue(temporal.final_correct)
        self.assertTrue(temporal.delayed_edge_learned)
        self.assertFalse(temporal.followup_misattribution)
        self.assertFalse(temporal.mechanism_shifted)
        self.assertEqual(temporal.temporal_transfer_score, 1.0)
        self.assertFalse(passive_core.final_correct)
        self.assertTrue(passive_core.followup_misattribution)
        self.assertGreater(
            temporal.temporal_transfer_score,
            passive_core.temporal_transfer_score,
        )

    def test_delayed_lamp_long_delay_changes_temporal_mechanism(self):
        world = DelayedLampLongDelayWorld()
        world.reset(seed=1)

        immediate = world.step("press_delay")
        self.assertIs(immediate["lamp_on"], False)
        first_wait = world.step("wait")
        self.assertIs(first_wait["lamp_on"], False)
        second_wait = world.step("wait")
        self.assertIs(second_wait["lamp_on"], True)
        self.assertEqual(world.temporal_spec()["delay_steps"], 2)

    def test_temporal_agent_repairs_shifted_delay_mechanism(self):
        temporal = run_temporal_adaptation_test(
            DelayedLampWorld(),
            DelayedLampLongDelayWorld(),
            TemporalCausalCoreAgent(),
            explore_steps=8,
            adapt_steps=4,
            seed=1,
            adaptation_mode="structural-prior",
        )
        fresh_temporal = run_temporal_adaptation_test(
            DelayedLampWorld(),
            DelayedLampLongDelayWorld(),
            TemporalCausalCoreAgent(),
            explore_steps=8,
            adapt_steps=4,
            seed=1,
            adaptation_mode="fresh",
        )
        passive_core = run_temporal_adaptation_test(
            DelayedLampWorld(),
            DelayedLampLongDelayWorld(),
            CausalCoreAgent(),
            explore_steps=8,
            adapt_steps=4,
            seed=1,
            adaptation_mode="structural-prior",
        )

        self.assertFalse(temporal.before_adaptation.final_correct)
        self.assertTrue(temporal.before_adaptation.mechanism_shifted)
        self.assertTrue(temporal.after_adaptation.final_correct)
        self.assertEqual(temporal.after_adaptation.temporal_transfer_score, 1.0)
        self.assertGreater(temporal.repair_gain, 0.0)
        self.assertFalse(fresh_temporal.after_adaptation.final_correct)
        self.assertFalse(passive_core.after_adaptation.final_correct)
        self.assertGreater(
            temporal.after_adaptation.temporal_transfer_score,
            passive_core.after_adaptation.temporal_transfer_score,
        )

    def test_dual_delayed_world_has_selective_mechanism_shift(self):
        source = DualDelayedControlWorld()
        source.reset(seed=1)
        source.step("press_delay_alarm")
        source_alarm = source.step("wait")
        self.assertIs(source_alarm["alarm_on"], True)

        source.reset(seed=1)
        source.step("press_delay_lamp")
        source_lamp = source.step("wait")
        self.assertIs(source_lamp["lamp_on"], True)

        target = DualDelayedSelectiveShiftWorld()
        target.reset(seed=1)
        target.step("press_delay_alarm")
        target_alarm = target.step("wait")
        self.assertIs(target_alarm["alarm_on"], True)

        target.reset(seed=1)
        target.step("press_delay_lamp")
        first_wait = target.step("wait")
        second_wait = target.step("wait")
        self.assertIs(first_wait["lamp_on"], False)
        self.assertIs(second_wait["lamp_on"], True)

        specs = {
            spec["target_variable"]: spec["delay_steps"]
            for spec in target.temporal_specs()
        }
        self.assertEqual(specs["alarm_on"], 1)
        self.assertEqual(specs["lamp_on"], 2)

    def test_temporal_agent_selectively_repairs_one_shifted_mechanism(self):
        temporal = run_temporal_selective_adaptation_test(
            DualDelayedControlWorld(),
            DualDelayedSelectiveShiftWorld(),
            TemporalCausalCoreAgent(),
            explore_steps=9,
            adapt_steps=6,
            seed=1,
            adaptation_mode="structural-prior",
        )
        fresh_temporal = run_temporal_selective_adaptation_test(
            DualDelayedControlWorld(),
            DualDelayedSelectiveShiftWorld(),
            TemporalCausalCoreAgent(),
            explore_steps=9,
            adapt_steps=6,
            seed=1,
            adaptation_mode="fresh",
        )
        passive_core = run_temporal_selective_adaptation_test(
            DualDelayedControlWorld(),
            DualDelayedSelectiveShiftWorld(),
            CausalCoreAgent(),
            explore_steps=9,
            adapt_steps=6,
            seed=1,
            adaptation_mode="structural-prior",
        )

        self.assertEqual(len(temporal.after_targets), 2)
        self.assertEqual(temporal.shifted_count, 1)
        self.assertEqual(temporal.stable_count, 1)
        self.assertEqual(temporal.shifted_before_score, 0.0)
        self.assertEqual(temporal.stable_before_score, 1.0)
        self.assertEqual(temporal.shifted_after_score, 1.0)
        self.assertEqual(temporal.stable_after_score, 1.0)
        self.assertTrue(temporal.after_all_success)
        self.assertEqual(temporal.selective_score, 1.0)
        self.assertEqual(fresh_temporal.shifted_after_score, 0.0)
        self.assertFalse(fresh_temporal.after_all_success)
        self.assertGreater(temporal.after_score, fresh_temporal.after_score)
        self.assertGreater(temporal.after_score, passive_core.after_score)

    def test_renamed_dual_delayed_world_changes_schema_only(self):
        world = RenamedDualDelayedControlWorld()
        world.reset(seed=1)
        siren_trigger = world.step("tap_siren")
        self.assertIs(siren_trigger["siren_active"], False)
        siren_after = world.step("settle")
        self.assertIs(siren_after["siren_active"], True)

        world.reset(seed=1)
        glow_trigger = world.step("tap_glow")
        self.assertIs(glow_trigger["glow_active"], False)
        glow_after = world.step("settle")
        self.assertIs(glow_after["glow_active"], True)

        self.assertEqual(
            {spec["target_variable"] for spec in world.temporal_specs()},
            {"siren_active", "glow_active"},
        )
        self.assertEqual(
            {spec["followup_action"] for spec in world.temporal_specs()},
            {"settle"},
        )

    def test_renamed_dual_delayed_selective_shift_changes_glow_delay_only(self):
        world = RenamedDualDelayedSelectiveShiftWorld()
        world.reset(seed=1)
        siren_trigger = world.step("tap_siren")
        self.assertIs(siren_trigger["siren_active"], False)
        siren_after = world.step("settle")
        self.assertIs(siren_after["siren_active"], True)

        world.reset(seed=1)
        glow_trigger = world.step("tap_glow")
        self.assertIs(glow_trigger["glow_active"], False)
        glow_after_one = world.step("settle")
        self.assertIs(glow_after_one["glow_active"], False)
        glow_after_two = world.step("settle")
        self.assertIs(glow_after_two["glow_active"], True)

        delays = {
            spec["target_variable"]: spec["delay_steps"]
            for spec in world.temporal_specs()
        }
        self.assertEqual(delays["siren_active"], 1)
        self.assertEqual(delays["glow_active"], 2)

    def test_portable_temporal_agent_transfers_to_renamed_schema(self):
        portable = run_level6_schema_transfer_test(
            DualDelayedControlWorld(),
            RenamedDualDelayedControlWorld(),
            PortableTemporalCausalCoreAgent(),
            explore_steps=9,
            target_steps=6,
            seed=1,
            transfer_mode="schema-prior",
        )
        fresh_portable = run_level6_schema_transfer_test(
            DualDelayedControlWorld(),
            RenamedDualDelayedControlWorld(),
            PortableTemporalCausalCoreAgent(),
            explore_steps=9,
            target_steps=6,
            seed=1,
            transfer_mode="fresh",
        )
        temporal = run_level6_schema_transfer_test(
            DualDelayedControlWorld(),
            RenamedDualDelayedControlWorld(),
            TemporalCausalCoreAgent(),
            explore_steps=9,
            target_steps=6,
            seed=1,
            transfer_mode="schema-prior",
        )

        self.assertTrue(portable.all_success)
        self.assertEqual(portable.target_score, 1.0)
        self.assertEqual(portable.delayed_edge_rate, 1.0)
        self.assertEqual(portable.followup_misattribution_rate, 0.0)
        self.assertEqual(portable.level6_score, 1.0)
        self.assertFalse(fresh_portable.all_success)
        self.assertEqual(fresh_portable.level6_score, 0.0)
        self.assertFalse(temporal.all_success)
        self.assertEqual(temporal.level6_score, 0.0)

    def test_portable_temporal_agent_repairs_renamed_selective_shift(self):
        portable = run_level6_schema_repair_test(
            DualDelayedControlWorld(),
            RenamedDualDelayedSelectiveShiftWorld(),
            PortableTemporalCausalCoreAgent(),
            explore_steps=9,
            target_steps=6,
            seed=1,
            transfer_mode="schema-prior",
        )
        fresh_portable = run_level6_schema_repair_test(
            DualDelayedControlWorld(),
            RenamedDualDelayedSelectiveShiftWorld(),
            PortableTemporalCausalCoreAgent(),
            explore_steps=9,
            target_steps=6,
            seed=1,
            transfer_mode="fresh",
        )
        temporal = run_level6_schema_repair_test(
            DualDelayedControlWorld(),
            RenamedDualDelayedSelectiveShiftWorld(),
            TemporalCausalCoreAgent(),
            explore_steps=9,
            target_steps=6,
            seed=1,
            transfer_mode="schema-prior",
        )

        self.assertEqual(portable.shifted_count, 1)
        self.assertEqual(portable.stable_count, 1)
        self.assertEqual(portable.before_score, 0.5)
        self.assertEqual(portable.shifted_before_score, 0.0)
        self.assertEqual(portable.stable_before_score, 1.0)
        self.assertEqual(portable.after_score, 1.0)
        self.assertEqual(portable.shifted_after_score, 1.0)
        self.assertEqual(portable.stable_after_score, 1.0)
        self.assertEqual(portable.followup_misattribution_rate, 0.0)
        self.assertEqual(portable.level6_repair_score, 1.0)
        self.assertFalse(fresh_portable.all_success)
        self.assertEqual(fresh_portable.level6_repair_score, 0.0)
        self.assertFalse(temporal.all_success)
        self.assertEqual(temporal.level6_repair_score, 0.0)

    def test_renamed_triple_delayed_diagnostic_shift_changes_glow_only(self):
        world = RenamedTripleDelayedDiagnosticShiftWorld()
        world.reset(seed=1)
        rotor_trigger = world.step("spin_rotor")
        self.assertIs(rotor_trigger["rotor_active"], False)
        rotor_after = world.step("settle")
        self.assertIs(rotor_after["rotor_active"], True)

        world.reset(seed=1)
        siren_trigger = world.step("tap_siren")
        self.assertIs(siren_trigger["siren_active"], False)
        siren_after = world.step("settle")
        self.assertIs(siren_after["siren_active"], True)

        world.reset(seed=1)
        glow_trigger = world.step("tap_glow")
        self.assertIs(glow_trigger["glow_active"], False)
        glow_after_one = world.step("settle")
        self.assertIs(glow_after_one["glow_active"], False)
        glow_after_two = world.step("settle")
        self.assertIs(glow_after_two["glow_active"], True)

        delays = {
            spec["target_variable"]: spec["delay_steps"]
            for spec in world.temporal_specs()
        }
        self.assertEqual(delays["rotor_active"], 1)
        self.assertEqual(delays["siren_active"], 1)
        self.assertEqual(delays["glow_active"], 2)

    def test_diagnostic_portable_agent_repairs_only_suspect_renamed_shift(self):
        diagnostic = run_level6_active_diagnostic_repair_test(
            TripleDelayedControlWorld(),
            RenamedTripleDelayedDiagnosticShiftWorld(),
            DiagnosticPortableTemporalCausalCoreAgent(),
            explore_steps=12,
            repair_steps=1,
            seed=1,
            transfer_mode="schema-prior",
        )
        portable = run_level6_active_diagnostic_repair_test(
            TripleDelayedControlWorld(),
            RenamedTripleDelayedDiagnosticShiftWorld(),
            PortableTemporalCausalCoreAgent(),
            explore_steps=12,
            repair_steps=1,
            seed=1,
            transfer_mode="schema-prior",
        )
        fresh_diagnostic = run_level6_active_diagnostic_repair_test(
            TripleDelayedControlWorld(),
            RenamedTripleDelayedDiagnosticShiftWorld(),
            DiagnosticPortableTemporalCausalCoreAgent(),
            explore_steps=12,
            repair_steps=1,
            seed=1,
            transfer_mode="fresh",
        )

        self.assertEqual(diagnostic.shifted_count, 1)
        self.assertEqual(diagnostic.stable_count, 2)
        self.assertAlmostEqual(diagnostic.before_score, 2 / 3)
        self.assertEqual(diagnostic.shifted_before_score, 0.0)
        self.assertEqual(diagnostic.stable_before_score, 1.0)
        self.assertEqual(diagnostic.after_score, 1.0)
        self.assertEqual(diagnostic.shifted_after_score, 1.0)
        self.assertEqual(diagnostic.stable_after_score, 1.0)
        self.assertEqual(diagnostic.level6_repair_score, 1.0)
        self.assertEqual(diagnostic.target_steps, 3)
        self.assertLess(portable.after_score, diagnostic.after_score)
        self.assertEqual(portable.level6_repair_score, 0.0)
        self.assertEqual(fresh_diagnostic.level6_repair_score, 0.0)

    def test_procedural_diagnostic_family_has_one_shifted_mechanism(self):
        source, target = make_procedural_diagnostic_world_pair(family_seed=3)
        source_delays = {
            spec["target_variable"]: spec["delay_steps"]
            for spec in source.temporal_specs()
        }
        target_delays = {
            spec["target_variable"]: spec["delay_steps"]
            for spec in target.temporal_specs()
        }

        self.assertEqual(len(source_delays), 3)
        self.assertEqual(len(target_delays), 3)
        self.assertEqual(set(source_delays.values()), {1})
        self.assertEqual(sorted(target_delays.values()), [1, 1, 2])

    def test_procedural_readout_family_preserves_temporal_specs(self):
        source, target = make_procedural_diagnostic_world_pair(
            family_seed=3,
            readout_mode="noisy-semantic-confounder",
        )
        source_observation = source.reset(seed=3)
        target_observation = target.reset(seed=3)

        self.assertEqual(len(source.temporal_specs()), 3)
        self.assertEqual(len(target.temporal_specs()), 3)
        self.assertEqual(len(source.readout_variables()), 4)
        self.assertEqual(len(target.readout_variables()), 4)
        self.assertTrue(source.readout_variables() <= set(source_observation))
        self.assertTrue(target.readout_variables() <= set(target_observation))

    def test_diagnostic_agent_generalizes_over_procedural_families(self):
        diagnostic_scores = []
        portable_scores = []
        for family_seed in range(1, 6):
            source, target = make_procedural_diagnostic_world_pair(family_seed)
            diagnostic = run_level6_active_diagnostic_repair_test(
                source,
                target,
                DiagnosticPortableTemporalCausalCoreAgent(),
                explore_steps=12,
                repair_steps=1,
                seed=family_seed,
                transfer_mode="schema-prior",
            )
            source, target = make_procedural_diagnostic_world_pair(family_seed)
            portable = run_level6_active_diagnostic_repair_test(
                source,
                target,
                PortableTemporalCausalCoreAgent(),
                explore_steps=12,
                repair_steps=1,
                seed=family_seed,
                transfer_mode="schema-prior",
            )
            diagnostic_scores.append(diagnostic.level6_repair_score)
            portable_scores.append(portable.level6_repair_score)
            self.assertEqual(diagnostic.after_score, 1.0)
            self.assertAlmostEqual(portable.after_score, 2 / 3)

        self.assertEqual(diagnostic_scores, [1.0] * 5)
        self.assertEqual(portable_scores, [0.0] * 5)

    def test_diagnostic_agent_handles_procedural_readout_families(self):
        for readout_mode in (
            "opaque",
            "semantic-confounder",
            "noisy-opaque",
            "noisy-semantic-confounder",
        ):
            diagnostic_scores = []
            portable_scores = []
            for family_seed in range(1, 6):
                source, target = make_procedural_diagnostic_world_pair(
                    family_seed,
                    readout_mode=readout_mode,
                )
                diagnostic = run_level6_active_diagnostic_repair_test(
                    source,
                    target,
                    DiagnosticPortableTemporalCausalCoreAgent(),
                    explore_steps=12,
                    repair_steps=1,
                    seed=family_seed,
                    transfer_mode="schema-prior",
                )
                source, target = make_procedural_diagnostic_world_pair(
                    family_seed,
                    readout_mode=readout_mode,
                )
                portable = run_level6_active_diagnostic_repair_test(
                    source,
                    target,
                    PortableTemporalCausalCoreAgent(),
                    explore_steps=12,
                    repair_steps=1,
                    seed=family_seed,
                    transfer_mode="schema-prior",
                )
                diagnostic_scores.append(diagnostic.level6_repair_score)
                portable_scores.append(portable.level6_repair_score)
                self.assertEqual(diagnostic.after_score, 1.0)
                self.assertAlmostEqual(portable.after_score, 2 / 3)

            self.assertEqual(diagnostic_scores, [1.0] * 5)
            self.assertEqual(portable_scores, [0.0] * 5)

    def test_readout_safe_schema_alignment_is_needed_for_semantic_readouts(self):
        safe_scores = []
        unsafe_scores = []
        for family_seed in range(1, 6):
            source, target = make_procedural_diagnostic_world_pair(
                family_seed,
                readout_mode="semantic-confounder",
            )
            safe = run_level6_active_diagnostic_repair_test(
                source,
                target,
                DiagnosticPortableTemporalCausalCoreAgent(),
                explore_steps=12,
                repair_steps=1,
                seed=family_seed,
                transfer_mode="schema-prior",
            )
            source, target = make_procedural_diagnostic_world_pair(
                family_seed,
                readout_mode="semantic-confounder",
            )
            unsafe = run_level6_active_diagnostic_repair_test(
                source,
                target,
                UnsafeReadoutDiagnosticPortableTemporalCausalCoreAgent(),
                explore_steps=12,
                repair_steps=1,
                seed=family_seed,
                transfer_mode="schema-prior",
            )
            safe_scores.append(safe.level6_repair_score)
            unsafe_scores.append(unsafe.level6_repair_score)

        self.assertEqual(safe_scores, [1.0] * 5)
        self.assertEqual(unsafe_scores, [0.0] * 5)

    def test_noisy_hidden_panel_observation_surface(self):
        world = NoisyHiddenPanelWorld(noise_probability=0.0)
        observation = world.reset(seed=1)

        self.assertIn("light_sensor", observation)
        self.assertIn("access_sensor", observation)
        self.assertIn("alarm_sensor", observation)
        self.assertNotIn("power_low", observation)

    def test_complex_noisy_hidden_panel_observation_surface(self):
        world = ComplexNoisyHiddenPanelWorld(noise_probability=0.0)
        observation = world.reset(seed=1)

        self.assertIn("pressure_sensor", observation)
        self.assertIn("thermal_sensor", observation)
        self.assertIn("safety_sensor", observation)
        self.assertNotIn("power_low", observation)
        self.assertNotIn("network_down", observation)
        self.assertEqual(
            world.hidden_context_edges(),
            {
                ("press_a", "lamp_on"),
                ("press_c", "alarm_on"),
                ("press_d", "pressure_high"),
                ("press_e", "backup_on"),
            },
        )
        self.assertEqual(
            world.readout_variables(),
            {
                "light_sensor",
                "access_sensor",
                "alarm_sensor",
                "pressure_sensor",
                "thermal_sensor",
                "safety_sensor",
            },
        )

    def test_procedural_complex_hidden_world_surface(self):
        world = make_procedural_complex_hidden_world(
            family_seed=7,
            mechanism_count=4,
            visible_count=5,
            noise_probability=0.0,
        )
        observation = world.reset(seed=1)
        action_names = {action.name for action in world.actions()}

        self.assertEqual(len(world.hidden_context_edges()), 4)
        self.assertEqual(len(world.readout_variables()), 6)
        self.assertTrue(world.readout_variables() <= set(observation))
        for action, target in world.hidden_context_edges():
            self.assertIn(action, action_names)
            self.assertIn(target, observation)
            self.assertIn((action, target), world.true_edges())
            self.assertIn((action, "sound"), world.true_edges())

    def test_noisy_core_panel_observation_surface(self):
        world = NoisyCorePanelWorld(noise_probability=0.0)
        observation = world.reset(seed=1)

        self.assertEqual(set(observation), set(PanelWorld().reset(seed=1)))
        self.assertEqual(world.true_edges(), PanelWorld().true_edges())

    def test_llm_controller_agent_runs_with_scripted_json_response(self):
        generator = ScriptedTextGenerator(
            [
                '{"action":"set_bright","prediction":["dark"],"reason":"flip dark"}',
                '{"action":"set_dark","prediction":["dark"],"reason":"flip dark back"}',
            ]
        )
        result = run_episode(
            PanelWorld(),
            LLMControllerAgent(generator),
            steps=2,
            seed=1,
        )

        self.assertEqual(result.transitions[0].action, "set_bright")
        self.assertEqual(result.transitions[1].action, "set_dark")
        self.assertIn(("set_bright", "dark"), result.discovered_edges)
        self.assertIn(("set_dark", "dark"), result.discovered_edges)

    def test_llm_causal_module_agent_exposes_module_summary(self):
        generator = ScriptedTextGenerator(
            [
                '{"action":"set_bright","prediction":["dark"],"reason":"probe"}',
                '{"action":"set_dark","prediction":["dark"],"reason":"probe"}',
            ]
        )
        agent = LLMCausalModuleAgent(
            generator,
            module=CausalObservationAdapterV2Agent(),
            name="llm-test-module",
        )

        run_episode(PanelWorld(), agent, steps=2, seed=1)

        self.assertTrue(
            any("Causal module summary:" in user for _, user in generator.prompts)
        )

    def test_llm_authoritative_module_executes_module_action(self):
        generator = ScriptedTextGenerator(
            ['{"action":"press_a","prediction":["lamp_on"],"reason":"ignore module"}']
        )
        agent = LLMAuthoritativeCausalModuleAgent(
            generator,
            module=CausalObservationAdapterV2Agent(),
            name="llm-test-authoritative",
        )

        result = run_episode(PanelWorld(), agent, steps=1, seed=1)

        self.assertEqual(result.transitions[0].action, "set_dark")
        self.assertIn("override=True", result.transitions[0].hypothesis)
        self.assertTrue(
            any("action=set_dark" in user for _, user in generator.prompts)
        )

    def test_llm_gated_module_vetoes_overused_proposal(self):
        generator = ScriptedTextGenerator(
            [
                '{"action":"press_a","prediction":["lamp_on"],"reason":"repeat"}',
                '{"action":"press_a","prediction":["lamp_on"],"reason":"repeat"}',
                '{"action":"press_a","prediction":["lamp_on"],"reason":"repeat"}',
            ]
        )
        agent = LLMGatedCausalModuleAgent(
            generator,
            module=CausalObservationAdapterV2Agent(),
            name="llm-test-gated",
        )

        result = run_episode(PanelWorld(), agent, steps=3, seed=1)

        self.assertEqual(result.transitions[0].action, "press_a")
        self.assertEqual(result.transitions[1].action, "set_dark")
        self.assertEqual(result.transitions[2].action, "set_bright")
        self.assertIn("vetoed-less-tested-module-action", result.transitions[1].hypothesis)

    def test_hidden_context_panel_hides_context_variable(self):
        world = HiddenContextPanelWorld()
        observation = world.reset(seed=1)

        self.assertEqual(set(observation), set(PanelWorld().reset(seed=1)))
        self.assertNotIn("power_low", observation)
        self.assertEqual(world.true_edges(), PanelWorld().true_edges())

    def test_derived_sensor_panel_observation_surface(self):
        world = DerivedSensorPanelWorld()
        observation = world.reset(seed=1)

        self.assertIn("light_sensor", observation)
        self.assertIn("access_sensor", observation)
        self.assertIn("alarm_sensor", observation)
        self.assertEqual(world.true_edges(), PanelWorld().true_edges())

    def test_opaque_readout_panel_observation_surface(self):
        world = OpaqueReadoutPanelWorld()
        observation = world.reset(seed=1)

        self.assertIn("glow", observation)
        self.assertIn("alert", observation)
        self.assertIn("motion", observation)
        self.assertEqual(world.readout_variables(), {"glow", "alert", "motion"})
        self.assertEqual(world.true_edges(), PanelWorld().true_edges())

    def test_observation_adapter_suppresses_sensor_false_positives(self):
        raw_result = run_episode(
            DerivedSensorPanelWorld(),
            CausalCoreAgent(),
            steps=60,
            seed=1,
        )
        adapted_result = run_episode(
            DerivedSensorPanelWorld(),
            ObservationAdapterCausalCoreAgent(),
            steps=60,
            seed=1,
        )

        raw_sensor_fp = sum(
            target.endswith("_sensor")
            for _, target in raw_result.discovered_edges - raw_result.true_edges
        )
        adapted_sensor_fp = sum(
            target.endswith("_sensor")
            for _, target in adapted_result.discovered_edges
            - adapted_result.true_edges
        )
        self.assertGreater(raw_sensor_fp, 0)
        self.assertEqual(adapted_sensor_fp, 0)
        self.assertGreaterEqual(adapted_result.score.f1, raw_result.score.f1)

    def test_learned_observation_adapter_suppresses_opaque_readouts(self):
        raw_result = run_episode(
            OpaqueReadoutPanelWorld(),
            CausalCoreAgent(),
            steps=80,
            seed=1,
        )
        suffix_result = run_episode(
            OpaqueReadoutPanelWorld(),
            ObservationAdapterCausalCoreAgent(),
            steps=80,
            seed=1,
        )
        learned_result = run_episode(
            OpaqueReadoutPanelWorld(),
            LearnedObservationAdapterCausalCoreAgent(),
            steps=80,
            seed=1,
        )
        readouts = {"glow", "alert", "motion"}

        raw_readout_fp = sum(
            target in readouts
            for _, target in raw_result.discovered_edges - raw_result.true_edges
        )
        suffix_readout_fp = sum(
            target in readouts
            for _, target in suffix_result.discovered_edges - suffix_result.true_edges
        )
        learned_readout_fp = sum(
            target in readouts
            for _, target in learned_result.discovered_edges
            - learned_result.true_edges
        )
        self.assertGreater(raw_readout_fp, 0)
        self.assertGreater(suffix_readout_fp, learned_readout_fp)
        self.assertLess(learned_readout_fp, raw_readout_fp)
        self.assertGreater(learned_result.score.f1, raw_result.score.f1)

    def test_noisy_hidden_panel_episode_runs(self):
        result = run_episode(
            NoisyHiddenPanelWorld(noise_probability=0.05),
            CausalCoreAgent(),
            steps=20,
            seed=1,
        )

        self.assertEqual(result.world_name, "panel-noisy-hidden")
        self.assertGreater(len(result.transitions), 0)

    def test_complex_noisy_hidden_panel_episode_runs(self):
        result = run_episode(
            ComplexNoisyHiddenPanelWorld(noise_probability=0.05),
            CausalObservationAdapterV2Agent(),
            steps=30,
            seed=1,
        )

        self.assertEqual(result.world_name, "panel-complex-noisy-hidden")
        self.assertGreater(len(result.transitions), 0)

    def test_observation_adapter_v2_filters_named_sensors(self):
        result = run_episode(
            DerivedSensorPanelWorld(),
            CausalObservationAdapterV2Agent(),
            steps=60,
            seed=1,
        )

        sensor_false_positives = [
            edge
            for edge in result.discovered_edges - result.true_edges
            if edge[1].endswith("_sensor")
        ]
        self.assertEqual(sensor_false_positives, [])
        self.assertNotIn(("observation", "sound"), result.condition_hints)
        self.assertGreaterEqual(result.score.f1, 0.85)

    def test_observation_adapter_v2_filters_opaque_readouts(self):
        result = run_episode(
            OpaqueReadoutPanelWorld(),
            CausalObservationAdapterV2Agent(),
            steps=80,
            seed=1,
        )
        readouts = {"glow", "alert", "motion"}

        readout_false_positives = [
            edge
            for edge in result.discovered_edges - result.true_edges
            if edge[1] in readouts
        ]
        self.assertEqual(readout_false_positives, [])
        self.assertGreaterEqual(result.score.f1, 0.85)

    def test_hidden_context_adapter_reduces_noisy_core_false_positives(self):
        v2_result = run_episode(
            NoisyHiddenPanelWorld(noise_probability=0.05),
            CausalObservationAdapterV2Agent(),
            steps=200,
            seed=1,
        )
        hidden_result = run_episode(
            NoisyHiddenPanelWorld(noise_probability=0.05),
            HiddenContextObservationAdapterAgent(),
            steps=200,
            seed=1,
        )
        readouts = {"access_sensor", "alarm_sensor", "light_sensor"}

        v2_false_positives = v2_result.discovered_edges - v2_result.true_edges
        hidden_false_positives = (
            hidden_result.discovered_edges - hidden_result.true_edges
        )
        v2_non_readout_false_positives = [
            edge for edge in v2_false_positives if edge[1] not in readouts
        ]
        hidden_non_readout_false_positives = [
            edge for edge in hidden_false_positives if edge[1] not in readouts
        ]
        hidden_readout_false_positives = [
            edge for edge in hidden_false_positives if edge[1] in readouts
        ]

        self.assertEqual(hidden_readout_false_positives, [])
        self.assertLess(
            len(hidden_non_readout_false_positives),
            len(v2_non_readout_false_positives),
        )
        self.assertGreaterEqual(hidden_result.score.f1, 0.80)

    def test_latent_context_adapter_forms_gate_when_observed_state_cannot_separate(self):
        adapter = LatentContextObservationAdapter()
        adapter.min_total_transitions = 1
        adapter.min_action_observations = 1
        adapter.min_effect_observations = 1
        adapter.min_latent_eligible_attempts = 6
        adapter.min_latent_successes = 3
        adapter.min_latent_failures = 3

        for step in range(1, 13):
            success = step % 2 == 0
            before = {"dark": True, "lamp_on": False, "door_open": False}
            after = dict(before)
            if success:
                after["lamp_on"] = True
            adapter.observe_transition(
                Transition(
                    step=step,
                    action="press_a",
                    before=before,
                    after=after,
                    prediction=frozenset(),
                    hypothesis="synthetic latent evidence",
                )
            )

        gates = adapter.latent_context_gates()
        gate = gates[("press_a", "lamp_on")]
        self.assertEqual(gate.latent_name, "latent:press_a->lamp_on:enabled")
        self.assertEqual(gate.eligible_count, 12)
        self.assertEqual(gate.success_count, 6)
        self.assertEqual(gate.failure_count, 6)
        self.assertLess(gate.best_observed_gate_accuracy, 0.88)

    def test_latent_context_adapter_does_not_invent_gate_for_observed_context(self):
        adapter = LatentContextObservationAdapter()
        adapter.min_total_transitions = 1
        adapter.min_action_observations = 1
        adapter.min_effect_observations = 1
        adapter.min_latent_eligible_attempts = 6
        adapter.min_latent_successes = 3
        adapter.min_latent_failures = 3

        for step in range(1, 13):
            success = step % 2 == 0
            before = {
                "dark": success,
                "lamp_on": False,
                "door_open": False,
            }
            after = dict(before)
            if success:
                after["lamp_on"] = True
            adapter.observe_transition(
                Transition(
                    step=step,
                    action="press_a",
                    before=before,
                    after=after,
                    prediction=frozenset(),
                    hypothesis="synthetic observed context evidence",
                )
            )

        self.assertNotIn(("press_a", "lamp_on"), adapter.latent_context_gates())

    def test_latent_context_agent_runs_on_noisy_hidden_panel(self):
        result = run_episode(
            NoisyHiddenPanelWorld(noise_probability=0.05),
            LatentContextObservationAdapterAgent(),
            steps=200,
            seed=1,
        )
        readouts = {"access_sensor", "alarm_sensor", "light_sensor"}
        readout_false_positives = [
            edge
            for edge in result.discovered_edges - result.true_edges
            if edge[1] in readouts
        ]

        self.assertEqual(readout_false_positives, [])
        self.assertGreaterEqual(result.score.f1, 0.75)

    def test_latent_context_agent_marks_true_hidden_context_edges(self):
        world = HiddenContextPanelWorld()
        result = run_episode(
            world,
            LatentContextObservationAdapterAgent(),
            steps=250,
            seed=1,
        )
        hidden_hint_edges = {
            edge
            for edge, hints in result.condition_hints.items()
            if any("hidden" in hint or "latent" in hint for hint in hints)
        }

        self.assertEqual(hidden_hint_edges, world.hidden_context_edges())

    def test_stateful_latent_adapter_tracks_recent_latent_belief(self):
        adapter = StatefulLatentContextObservationAdapter()
        adapter.min_total_transitions = 1
        adapter.min_action_observations = 1
        adapter.min_effect_observations = 1
        adapter.min_latent_eligible_attempts = 6
        adapter.min_latent_successes = 3
        adapter.min_latent_failures = 3
        adapter.latent_belief_window = 4
        outcomes = [
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            True,
        ]

        for step, success in enumerate(outcomes, start=1):
            before = {"dark": True, "lamp_on": False, "door_open": False}
            after = dict(before)
            if success:
                after["lamp_on"] = True
            adapter.observe_transition(
                Transition(
                    step=step,
                    action="press_a",
                    before=before,
                    after=after,
                    prediction=frozenset(),
                    hypothesis="synthetic stateful latent evidence",
                )
            )

        belief = adapter.latent_state_beliefs()[("press_a", "lamp_on")]
        self.assertGreater(belief.enabled_probability, 0.80)
        self.assertLess(belief.uncertainty, 0.40)
        self.assertEqual(belief.recent_successes, 4)
        self.assertEqual(belief.recent_failures, 0)

    def test_stateful_latent_adapter_selects_probe_when_eligible(self):
        adapter = StatefulLatentContextObservationAdapter()
        adapter.min_total_transitions = 1
        adapter.min_action_observations = 1
        adapter.min_effect_observations = 1
        adapter.min_latent_eligible_attempts = 6
        adapter.min_latent_successes = 3
        adapter.min_latent_failures = 3
        adapter.min_probe_uncertainty = 0.10

        for step in range(1, 13):
            success = step % 2 == 0
            before = {"dark": True, "lamp_on": False, "door_open": False}
            after = dict(before)
            if success:
                after["lamp_on"] = True
            adapter.observe_transition(
                Transition(
                    step=step,
                    action="press_a",
                    before=before,
                    after=after,
                    prediction=frozenset(),
                    hypothesis="synthetic probe evidence",
                )
            )

        probe = adapter.probe_candidate(
            {"dark": True, "lamp_on": False, "door_open": False}
        )
        self.assertIsNotNone(probe)
        self.assertEqual(probe.gate.action, "press_a")
        self.assertEqual(probe.gate.target, "lamp_on")

    def test_stateful_latent_context_agent_runs_on_noisy_hidden_panel(self):
        result = run_episode(
            NoisyHiddenPanelWorld(noise_probability=0.05),
            StatefulLatentContextObservationAdapterAgent(),
            steps=200,
            seed=1,
        )
        readouts = {"access_sensor", "alarm_sensor", "light_sensor"}
        readout_false_positives = [
            edge
            for edge in result.discovered_edges - result.true_edges
            if edge[1] in readouts
        ]

        self.assertEqual(readout_false_positives, [])
        self.assertGreaterEqual(result.score.f1, 0.70)

    def test_persistent_latent_adapter_accumulates_total_evidence(self):
        adapter = PersistentLatentContextObservationAdapter()
        adapter.min_total_transitions = 1
        adapter.min_action_observations = 1
        adapter.min_effect_observations = 1
        adapter.min_latent_eligible_attempts = 8
        adapter.min_latent_successes = 2
        adapter.min_latent_failures = 2
        adapter.max_latent_success_rate = 0.90
        adapter.latent_belief_window = 4
        outcomes = [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
        ]

        for step, success in enumerate(outcomes, start=1):
            before = {"dark": True, "lamp_on": False, "door_open": False}
            after = dict(before)
            if success:
                after["lamp_on"] = True
            adapter.observe_transition(
                Transition(
                    step=step,
                    action="press_a",
                    before=before,
                    after=after,
                    prediction=frozenset(),
                    hypothesis="synthetic persistent latent evidence",
                )
            )

        belief = adapter.latent_state_beliefs()[("press_a", "lamp_on")]
        self.assertEqual(belief.belief_source, "persistent-posterior")
        self.assertEqual(belief.total_successes, 10)
        self.assertEqual(belief.total_failures, 2)
        self.assertGreater(belief.enabled_probability, 0.75)
        self.assertGreater(belief.confidence, 0.50)

    def test_persistent_latent_adapter_reports_recent_drift(self):
        adapter = PersistentLatentContextObservationAdapter()
        adapter.min_total_transitions = 1
        adapter.min_action_observations = 1
        adapter.min_effect_observations = 1
        adapter.min_latent_eligible_attempts = 8
        adapter.min_latent_successes = 2
        adapter.min_latent_failures = 2
        adapter.latent_belief_window = 4
        outcomes = [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ]

        for step, success in enumerate(outcomes, start=1):
            before = {"dark": True, "lamp_on": False, "door_open": False}
            after = dict(before)
            if success:
                after["lamp_on"] = True
            adapter.observe_transition(
                Transition(
                    step=step,
                    action="press_a",
                    before=before,
                    after=after,
                    prediction=frozenset(),
                    hypothesis="synthetic persistent drift evidence",
                )
            )

        belief = adapter.latent_state_beliefs()[("press_a", "lamp_on")]
        self.assertGreater(belief.recent_drift, 0.25)
        self.assertGreater(belief.uncertainty, 0.50)

    def test_persistent_latent_context_agent_runs_on_noisy_hidden_panel(self):
        result = run_episode(
            NoisyHiddenPanelWorld(noise_probability=0.05),
            PersistentLatentContextObservationAdapterAgent(),
            steps=200,
            seed=1,
        )
        readouts = {"access_sensor", "alarm_sensor", "light_sensor"}
        readout_false_positives = [
            edge
            for edge in result.discovered_edges - result.true_edges
            if edge[1] in readouts
        ]

        self.assertEqual(readout_false_positives, [])
        self.assertGreaterEqual(result.score.f1, 0.70)

    def test_proactive_latent_context_agent_runs_on_noisy_hidden_panel(self):
        result = run_episode(
            NoisyHiddenPanelWorld(noise_probability=0.05),
            ProactiveLatentContextObservationAdapterAgent(),
            steps=200,
            seed=1,
        )
        readouts = {"access_sensor", "alarm_sensor", "light_sensor"}
        readout_false_positives = [
            edge
            for edge in result.discovered_edges - result.true_edges
            if edge[1] in readouts
        ]

        self.assertEqual(readout_false_positives, [])
        self.assertGreaterEqual(result.score.f1, 0.70)

    def test_proactive_latent_context_agent_marks_true_hidden_context_edges(self):
        world = HiddenContextPanelWorld()
        result = run_episode(
            world,
            ProactiveLatentContextObservationAdapterAgent(),
            steps=200,
            seed=1,
        )
        hidden_hint_edges = {
            edge
            for edge, hints in result.condition_hints.items()
            if any("latent" in hint for hint in hints)
        }

        self.assertEqual(hidden_hint_edges, world.hidden_context_edges())

    def test_control_experiment_planner_runs_on_complex_noisy_hidden_panel(self):
        result = run_episode(
            ComplexNoisyHiddenPanelWorld(noise_probability=0.05),
            ControlExperimentPlannerLatentContextAgent(),
            steps=180,
            seed=1,
        )
        readouts = {
            "access_sensor",
            "alarm_sensor",
            "light_sensor",
            "pressure_sensor",
            "safety_sensor",
            "thermal_sensor",
        }
        readout_false_positives = [
            edge
            for edge in result.discovered_edges - result.true_edges
            if edge[1] in readouts
        ]

        self.assertEqual(readout_false_positives, [])
        self.assertGreaterEqual(result.score.f1, 0.55)

    def test_context_search_planner_runs_on_complex_noisy_hidden_panel(self):
        result = run_episode(
            ComplexNoisyHiddenPanelWorld(noise_probability=0.05),
            ContextSearchControlExperimentPlannerAgent(),
            steps=180,
            seed=1,
        )
        readouts = {
            "access_sensor",
            "alarm_sensor",
            "light_sensor",
            "pressure_sensor",
            "safety_sensor",
            "thermal_sensor",
        }
        readout_false_positives = [
            edge
            for edge in result.discovered_edges - result.true_edges
            if edge[1] in readouts
        ]

        self.assertEqual(readout_false_positives, [])
        self.assertGreaterEqual(result.score.f1, 0.50)

    def test_context_search_hidden_label_guard_filters_non_hidden_targets(self):
        complex_world = ComplexNoisyHiddenPanelWorld(noise_probability=0.05)
        guarded = ContextSearchControlExperimentPlannerAgent()
        guarded.reset(complex_world.actions())

        for edge in complex_world.hidden_context_edges():
            self.assertTrue(guarded._hidden_context_hint_allowed(*edge))
        self.assertFalse(
            guarded._hidden_context_hint_allowed("press_b", "alarm_on")
        )
        self.assertFalse(
            guarded._hidden_context_hint_allowed("press_d", "coolant_on")
        )

        procedural_world = make_procedural_complex_hidden_world(
            family_seed=6,
            mechanism_count=4,
            visible_count=5,
            noise_probability=0.02,
            readout_count=6,
        )
        guarded.reset(procedural_world.actions())
        unguarded = UnguardedContextSearchControlExperimentPlannerAgent()
        unguarded.reset(procedural_world.actions())

        self.assertTrue(
            guarded._hidden_context_hint_allowed("probe_6_2", "out_6_2")
        )
        self.assertFalse(
            guarded._hidden_context_hint_allowed("probe_6_2", "ctx_6_2")
        )
        self.assertFalse(
            guarded._hidden_context_hint_allowed("probe_6_3", "decoy_flag")
        )
        self.assertFalse(
            guarded._hidden_context_hint_allowed("probe_6_0", "sound")
        )
        self.assertFalse(
            guarded._hidden_context_hint_allowed("probe_6_2", "out_6_3")
        )
        self.assertTrue(
            unguarded._hidden_context_hint_allowed("probe_6_2", "ctx_6_2")
        )


if __name__ == "__main__":
    unittest.main()
