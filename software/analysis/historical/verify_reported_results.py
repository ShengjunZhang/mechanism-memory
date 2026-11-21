from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_json"
if not RAW.exists():
    RAW = HERE.parent / "source_data" / "raw_json"


def load(name: str):
    with (RAW / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def rows_by_method(rows: list[dict]) -> dict[str, dict]:
    return {
        str(row["method"] if "method" in row else row["agent"]): row
        for row in rows
    }


def check(actual: float, expected: float, label: str, tol: float = 5e-4) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def main() -> None:
    coarse = load("finite_agent_comparison_complex_100x5.json")
    coarse_rows = rows_by_method(coarse["results"]["panel-complex-noisy-hidden"]["100"]["summary"])
    check(coarse_rows["passive-correlation"]["f1"], 0.0, "passive target F1")
    check(coarse_rows["reward-rl"]["f1"], 0.3262996948, "reward target F1")

    matched = load("matched_mechanism_baselines_8families_2seeds.json")
    matched_rows = rows_by_method(matched["matched_mechanism_baselines"]["summary"])
    expected_matched = {
        "causal-core-context-search": (0.7931627625, 1.0625, 0.0),
        "conditional-discovery": (0.6396028865, 8.4375, 6.8125),
        "conditional-discovery-readout-oracle": (0.7630485562, 1.9375, 0.0),
        "tuple-lift-readout-oracle": (0.7892389632, 1.3125, 0.0),
        "latent-world-model": (0.2389294497, 48.6875, 16.25),
        "latent-world-model-readout-oracle": (0.3188283382, 43.5625, 0.0),
        "neural-transition": (0.3957005638, 38.625, 13.25),
        "neural-transition-readout-oracle": (0.4789424736, 25.375, 0.0),
    }
    for method, (f1, fp, readout_fp) in expected_matched.items():
        check(matched_rows[method]["f1"], f1, f"{method} F1")
        check(matched_rows[method]["false_positive"], fp, f"{method} FP")
        check(matched_rows[method]["readout_false_positive"], readout_fp, f"{method} readout FP")
    check(matched_rows["latent-world-model"]["model_next_bit_accuracy"], 0.9046875, "world-model accuracy")

    ranking = load("ranking_and_adaptation_baselines_8families_2seeds.json")
    ranking_rows = {row["task"]: row for row in ranking["summary"]}
    hidden = ranking_rows["procedural noisy-hidden ranking"]
    check(hidden["ranking_predictor_rank_top1"], 0.5208333333, "ranking top-1")
    check(hidden["ranking_predictor_mechanism_f1"], 0.0, "ranking target F1")
    check(hidden["causal_core_mechanism_f1"], 0.7641356918, "ranking core target F1")
    for task in ("renamed diagnostic adaptation", "noisy semantic-readout adaptation"):
        check(ranking_rows[task]["ranking_predictor_adaptation"], 0.0, f"{task} ranking AdaptScore")
        check(ranking_rows[task]["causal_core_adaptation"], 1.0, f"{task} core AdaptScore")

    continuous = load("continuous_metric_comparison_20seeds.json")
    continuous_rows = rows_by_method(continuous["summary"])
    check(continuous_rows["global-linear-regression"]["f1"], 2.0 / 3.0, "continuous linear F1")
    check(continuous_rows["metric-causal-core"]["f1"], 1.0, "continuous core F1")
    check(continuous_rows["metric-causal-core"]["readout_false_positive"], 0.0, "continuous core readout FP")

    noise = load("continuous_noise_sweep_20seeds.json")["continuous_noise_sweep"]["summary"]
    for row, random_f1, linear_f1 in zip(
        noise,
        (0.4021182266, 0.3210287972, 0.2650955286, 0.2522507284),
        (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.6952380952),
    ):
        check(row["metric_causal_core_f1"], 1.0, f"noise={row['noise']} core F1")
        check(row["random_correlation_f1"], random_f1, f"noise={row['noise']} random F1")
        check(row["global_linear_f1"], linear_f1, f"noise={row['noise']} linear F1")

    tdmpc2 = load("tdmpc2_distributional_retention_3tasks_3seeds.json")
    tdmpc2_rows = {
        (str(row["condition"]), str(row["method"])): row
        for row in tdmpc2["summary"]
    }
    expected_tdmpc2 = {
        ("source", "TD-MPC2 nonlinear probe"): (0.6285206974, 0.9185185185),
        ("target-random", "TD-MPC2 frozen"): (1.1381745272, 0.6245370370),
        ("target-random", "global probe-head update"): (1.5476475689, 0.7597222222),
        ("target-random", "selective mechanism update"): (0.6297227558, 0.9203703704),
        ("target-shifted", "TD-MPC2 frozen"): (1.7008613083, 0.0574074074),
        ("target-shifted", "global probe-head update"): (1.2874301672, 0.6935185185),
        ("target-shifted", "selective mechanism update"): (0.5513722855, 0.9481481481),
        ("target-stable", "TD-MPC2 frozen"): (0.6725335684, 0.9148148148),
        ("target-stable", "global probe-head update"): (1.4596690337, 0.8564814815),
        ("target-stable", "selective mechanism update"): (0.6725335684, 0.9148148148),
    }
    for key, (nrmse, sign_accuracy) in expected_tdmpc2.items():
        row = tdmpc2_rows[key]
        check(row["nrmse"], nrmse, f"{key} NRMSE")
        check(row["sign_accuracy"], sign_accuracy, f"{key} sign accuracy")
    check(tdmpc2["localization_rate"], 1.0, "TD-MPC2 actuator localization")
    for row in tdmpc2["diagnostic_margin_sensitivity"]:
        check(
            row["localization_rate"],
            1.0,
            f"TD-MPC2 threshold={row['loss_ratio_threshold']} localization",
        )
    if tdmpc2["checkpoint_sha256"].lower() != "f3c6909e3282f115ce57b2ab29a88e654b4728d819d2c93a4ca079bad63b1455":
        raise AssertionError("TD-MPC2 checkpoint SHA-256 does not match the reported checkpoint")

    llm = load("llm_proposal_policy_comparison_3seeds.json")["worlds"]["panel-complex-noisy-hidden"]
    expected_llm = {
        "llm-causal-prompt": (0.1630681184, 82.3333333333, 32.0),
        "llm-context-search-gated": (0.6783882784, 6.0, 0.0),
        "llm-control-planner-gated": (0.7272441844, 2.0, 0.0),
        "causal-core-context-search-planner": (0.6798910302, 4.0, 0.0),
        "causal-core-control-experiment-planner": (0.6968621792, 5.6666666667, 0.0),
    }
    for method, (f1, fp, readout_fp) in expected_llm.items():
        row = llm[method]["summary"]
        check(row["f1"], f1, f"{method} target F1")
        check(row["false_positive"], fp, f"{method} FP")
        check(row["readout_false_positive"], readout_fp, f"{method} readout FP")

    hidden = load("llm_hidden_retention_12train_6test.json")
    hidden_rows = {
        (str(row["method"]), str(row["group"])): row
        for row in hidden["rows"]
    }
    expected_hidden = {
        "TF-IDF query only": (0.7361111111, 0.6111111111, 0.0, 0.6111111111),
        "Qwen calibrated Yes/No logits": (0.6527777778, 0.5833333333, 0.4166666667, 0.0),
        "Qwen linear layer 28": (0.9583333333, 0.8472222222, 0.5833333333, 0.0),
        "Qwen nonlinear RBF": (0.9444444444, 0.875, 0.25, 0.1111111111),
        "paired-evidence rule": (1.0, 1.0, 1.0, 0.9444444444),
        "gated mechanism state": (1.0, 1.0, 1.0, 0.0555555556),
    }
    for method, (source, renamed, shifted, readout_fpr) in expected_hidden.items():
        check(
            hidden_rows[(method, "source-clean")]["balanced_accuracy"],
            source,
            f"{method} source BA",
        )
        check(
            hidden_rows[(method, "renamed-shift")]["balanced_accuracy"],
            renamed,
            f"{method} renamed BA",
        )
        check(
            hidden_rows[(method, "shifted-delay")]["balanced_accuracy"],
            shifted,
            f"{method} shifted-delay BA",
        )
        check(
            hidden_rows[(method, "semantic-readout-candidates")]["false_positive_rate"],
            readout_fpr,
            f"{method} semantic-readout FPR",
        )
    if hidden["training"]["train_examples"] != 504:
        raise AssertionError("LLM hidden-state training-example count does not match")
    if hidden["training"]["test_examples"] != 360:
        raise AssertionError("LLM hidden-state test-example count does not match")
    if hidden["model"]["layers"] != [0, 7, 14, 21, 28]:
        raise AssertionError("LLM hidden-state layer list does not match")
    if hidden["model"]["truncated_prompts"] != 0:
        raise AssertionError("LLM hidden-state run contains truncated prompts")
    if hidden["model"]["snapshot_revision"] != "a09a35458c702b33eeacc393d103063234e8bc28":
        raise AssertionError("Qwen snapshot revision does not match")

    budget = load("candidate_budget_sensitivity_4families.json")["candidate_budget_sensitivity"]["summary"]
    for row in budget:
        expected_f1 = 0.6980980013 if row["steps"] == 80 else 0.7629742890
        expected_hidden = 0.125 if row["steps"] == 80 else 0.625
        check(row["f1"], expected_f1, f"{row['config']} {row['steps']} F1")
        check(row["hidden_context_recall"], expected_hidden, f"{row['config']} {row['steps']} hidden recall")
        check(row["readout_false_positive"], 0.0, f"{row['config']} {row['steps']} readout FP")

    print("All reported numerical claims match the packaged raw results.")


if __name__ == "__main__":
    main()
