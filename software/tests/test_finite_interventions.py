"""Finite reviewer counterexamples and new-writer/scoring contracts."""

from dataclasses import replace
from fractions import Fraction
from itertools import combinations, product
import json
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys

import pytest

from causal_sandbox.finite_interventions import (
    Candidate, DirectEvidence, Evidence, FiniteSCM, GateConfig, Mechanism, PairStats,
    collect_evidence, contrast_peak_emulation, coverage_counterexample,
    REQUIRED_SOURCES, controlled_strata, exact_population_examples, paired_writer,
    run_historical, source_provenance, tuple_scores,
)


@pytest.mark.parametrize("reviewer,static", [("R1", Fraction(2, 3)), ("R2", Fraction(1, 2))])
def test_exact_static_and_retest_informed_adaptive_coverage(reviewer, static):
    result = coverage_counterexample(reviewer)
    assert Fraction(result["static_optimum"]) == static
    assert Fraction(result["adaptive_coverage"]) == 1
    paths = result["paths"]
    # Independently enumerate every admissible static list, including shorter ones.
    values = []
    for n in range(4):
        for fixed in combinations(range(1, 5), n):
            values.append(Fraction(sum(set(p["shifted"]) <= set(fixed) for p in paths), len(paths)))
    assert max(values) == static
    for path in paths:
        assert len(path["tested"]) == len(set(path["tested"])) <= 3
        assert path["tested"][0] == 1
        assert path["retest_outcomes"] == [j in path["shifted"] for j in path["tested"]]
        assert set(path["shifted"]) <= set(path["tested"])
    if reviewer == "R1":
        assert [p["tested"] for p in paths] == [[1, 2], [1, 2, 3], [1, 2, 4]]
    else:
        assert [p["tested"] for p in paths] == [[1, 2], [1, 3, 4]]


def test_r1_modal_and_positive_contrast_disagree_exactly():
    scm = FiniteSCM(7, "modal", varied=False)
    one = scm.population(1, scm.active_context, 1)[scm.root]
    zero = scm.population(0, scm.active_context, 1)[scm.root]
    assert (1 - one, 1 - zero) == (Fraction(3, 5), Fraction(9, 10))
    assert one - zero == Fraction(3, 10)
    assert (1 - one) - (1 - zero) == Fraction(-3, 10)
    assert next(iter(scm.truth())).value == 0
    evidence = collect_evidence(scm, 17, 4096)
    assert paired_writer(evidence, scm.candidates, GateConfig()).mechanisms == scm.truth()
    emulation = contrast_peak_emulation(evidence, scm.candidates, GateConfig())
    assert all(m.value == 1 for m in emulation.mechanisms)
    assert emulation.mechanisms.isdisjoint(scm.truth())


def test_r2_first_response_differs_from_peak_exactly():
    scm = FiniteSCM(7, "late_peak", varied=False)
    contrasts = [scm.population(1, scm.active_context, l)[scm.root]
                 - scm.population(0, scm.active_context, l)[scm.root] for l in scm.lags]
    assert contrasts == [Fraction(3, 5), Fraction(17, 20)]
    assert next(iter(scm.truth())).delay == 1
    evidence = collect_evidence(scm, 29, 4096)
    assert paired_writer(evidence, scm.candidates, GateConfig()).mechanisms == scm.truth()
    old = contrast_peak_emulation(evidence, scm.candidates, GateConfig())
    assert {m.delay for m in old.mechanisms} == {2}


def test_r2_noisy_descendant_is_responsive_but_not_direct_exactly():
    scm = FiniteSCM(7, "noisy_descendant", varied=False)
    c, y, r = scm.active_context, scm.root, scm.descendant
    one, zero = scm.population(1, c, 1), scm.population(0, c, 1)
    assert one[y] - zero[y] == 1
    assert one[r] - zero[r] == Fraction(3, 5) > Fraction(1, 5)
    assert one["copy_agreement"] == Fraction(4, 5) < Fraction(9, 10)
    for clamp in (0, 1):
        assert scm.population(1, c, 1, {y: clamp})[r] == scm.population(0, c, 1, {y: clamp})[r]
        assert scm.population(1, c, 1, {r: clamp})[y] - scm.population(0, c, 1, {r: clamp})[y] == 1
    evidence = collect_evidence(scm, 43, 4096)
    corrected = paired_writer(evidence, scm.candidates, GateConfig())
    assert corrected.mechanisms == scm.truth()
    indirect = next(d for d in corrected.decisions if d["candidate"]["target"] == r
                    and d["candidate"]["context"] == ((scm.context_variable, c),))
    assert indirect["reason"] == "direct_effect_not_supported_under_clamping"
    assert {m.target for m in contrast_peak_emulation(evidence, scm.candidates, GateConfig()).mechanisms} == {y, r}


def test_perfect_formula_copy_is_symmetric_not_directional():
    scm = FiniteSCM(7, "noisy_descendant", varied=False)
    scm.noise = Fraction(0)
    for arm in (0, 1):
        assert scm.population(arm, scm.active_context, 1)["copy_agreement"] == 1
    evidence = collect_evidence(scm, 17, 4096)
    assert contrast_peak_emulation(evidence, scm.candidates, GateConfig()).mechanisms == set()
    assert paired_writer(evidence, scm.candidates, GateConfig()).mechanisms == scm.truth()


def test_natural_contrast_without_direct_evidence_abstains():
    scm = FiniteSCM(7, "noisy_descendant", varied=False)
    evidence = collect_evidence(scm, 17, 4096, direct=False)
    writer = paired_writer(evidence, scm.candidates, GateConfig())
    assert not writer.mechanisms
    assert all(d["reason"] == "incomplete_direct_separation" for d in writer.decisions)
    ablation = paired_writer(evidence, scm.candidates, GateConfig(), require_direct=False)
    assert not ablation.mechanisms
    assert "missing_direct_separation" in {d["reason"] for d in ablation.decisions}


def test_incomplete_precontext_clamp_support_is_ineligible():
    scm = FiniteSCM(7, "noisy_descendant", varied=False)
    evidence = collect_evidence(scm, 17, 4096)
    root = next(iter(scm.truth())).candidate
    cert = evidence.direct[root]
    cells = dict(cert.cells)
    cells.pop((1, (1,)))
    evidence.direct[root] = DirectEvidence(cert.clamped_variables, cells)
    result = paired_writer(evidence, [root], GateConfig())
    assert not result.mechanisms
    assert result.decisions[0]["eligible"] is False
    assert result.decisions[0]["reason"] == "incomplete_direct_separation"


def test_onset_comes_from_natural_law_after_blocking_certifies_directness():
    # Eligible synthetic measurements: the natural response is mediated at lag 1;
    # a detectable direct response exists only at lag 2 under every clamp.
    candidate = Candidate("a", (("c", 0),), "x")
    strong = PairStats(1_000_000, 0.9, 0.1, 0.8, 0.8)
    null = PairStats(1_000_000, 0.1, 0.1, 0.0, 0.8)
    certificate = DirectEvidence(("z",), {(l, (a,)): null if l == 1 else strong
                                          for l, a in product((1, 2), (0, 1))})
    evidence = Evidence(("x", "z"), (1, 2), {(candidate, 1): strong, (candidate, 2): strong},
                        {candidate: certificate}, {})
    result = paired_writer(evidence, [candidate], GateConfig())
    assert result.mechanisms == {Mechanism("a", (("c", 0),), "x", 1, 1)}
    assert result.decisions[0]["directness_certified"]
    assert {x["lag"] for x in result.decisions[0]["directness_support"]} == {2}


def test_unresolved_earlier_natural_tv_prevents_later_selection():
    candidate = Candidate("a", (("c", 0),), "x")
    boundary = PairStats(4096, 0.7, 0.5, 0.2, 0.8)
    strong = PairStats(4096, 1.0, 0.0, 1.0, 0.8)
    certificate = DirectEvidence(("z",), {(l, (a,)): boundary if l == 1 else strong
                                          for l, a in product((1, 2), (0, 1))})
    evidence = Evidence(("x", "z"), (1, 2), {(candidate, 1): boundary, (candidate, 2): strong},
                        {candidate: certificate}, {})
    result = paired_writer(evidence, [candidate], GateConfig())
    assert not result.mechanisms
    assert result.decisions[0]["reason"] == "unresolved_earlier_natural_lag"


def test_natural_modal_p1_can_disagree_with_blocked_modes():
    candidate = Candidate("a", (("c", 0),), "x")
    natural = PairStats(1_000_000, 0.4, 0.1, 0.3, 0.8)
    blocked_one = PairStats(1_000_000, 0.9, 0.1, 0.8, 0.8)
    blocked_zero = PairStats(1_000_000, 0.1, 0.9, -0.8, 0.8)
    cert = DirectEvidence(("z",), {(l, (a,)): blocked_one if a else blocked_zero
                                   for l, a in product((1, 2), (0, 1))})
    evidence = Evidence(("x", "z"), (1, 2), {(candidate, l): natural for l in (1, 2)},
                        {candidate: cert}, {})
    result = paired_writer(evidence, [candidate], GateConfig())
    assert result.mechanisms == {Mechanism("a", (("c", 0),), "x", 0, 1)}
    assert result.decisions[0]["modal_value_law"] == "natural"


def test_blocked_modal_tie_does_not_redefine_natural_value():
    candidate = Candidate("a", (("c", 0),), "x")
    natural = PairStats(1_000_000, 0.9, 0.1, 0.8, 0.8)
    blocked = PairStats(1_000_000, 0.5, 0.0, 0.5, 0.8)
    cert = DirectEvidence(("z",), {(l, (a,)): blocked for l, a in product((1, 2), (0, 1))})
    evidence = Evidence(("x", "z"), (1, 2), {(candidate, l): natural for l in (1, 2)},
                        {candidate: cert}, {})
    assert paired_writer(evidence, [candidate], GateConfig()).mechanisms == {
        Mechanism("a", (("c", 0),), "x", 1, 1)}


def test_unresolved_natural_mode_abstains_even_if_blocked_mode_clear():
    candidate = Candidate("a", (("c", 0),), "x")
    natural = PairStats(4096, 0.51, 0.0, 0.51, 0.8)
    blocked = PairStats(4096, 0.9, 0.1, 0.8, 0.8)
    cert = DirectEvidence(("z",), {(l, (a,)): blocked for l, a in product((1, 2), (0, 1))})
    evidence = Evidence(("x", "z"), (1, 2), {(candidate, l): natural for l in (1, 2)},
                        {candidate: cert}, {})
    result = paired_writer(evidence, [candidate], GateConfig())
    assert not result.mechanisms
    assert result.decisions[0]["reason"] == "unresolved_natural_interventional_mode"


def test_modal_p1_not_sign_of_tv_or_contrast():
    scm = FiniteSCM(7, "modal", varied=False)
    # Negative signed contrast, while the most probable p1 value is still zero.
    scm.q0 = (Fraction(4, 5),) * 2
    scm.q1 = (Fraction(2, 5),) * 2
    evidence = collect_evidence(scm, 17, 4096)
    assert paired_writer(evidence, scm.candidates, GateConfig()).mechanisms == scm.truth()
    assert next(iter(scm.truth())).value == 0


def test_weak_structural_early_effect_is_counted_as_wrong_full_tuple():
    scm = FiniteSCM(7, "weak_early", varied=False)
    assert scm.q1[0] - scm.q0[0] == Fraction(1, 25)
    evidence = collect_evidence(scm, 17, 4096)
    result = paired_writer(evidence, scm.candidates, GateConfig())
    assert {m.delay for m in result.mechanisms} == {2}
    assert {m.value for m in result.mechanisms} == {1}
    metrics = tuple_scores(result.mechanisms, scm.truth(), scm.candidates, scm.candidates)
    assert metrics["full_tuple"] == {"true_positive": 0, "false_positive": 1, "false_negative": 1,
                                     "precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert metrics["coverage"]["answer_coverage"] == 1
    assert metrics["all_query_accuracy"] == 0.75


def test_omitted_true_candidate_remains_in_risk_and_recall_denominators():
    scm = FiniteSCM(7, "noisy_descendant", varied=False)
    evidence = collect_evidence(scm, 17, 4096)
    true_candidate = next(iter(scm.truth())).candidate
    generated = [c for c in scm.candidates if c != true_candidate]
    result = paired_writer(evidence, generated, GateConfig())
    metrics = tuple_scores(result.mechanisms, scm.truth(), scm.candidates, generated)
    assert metrics["full_tuple"]["false_negative"] == 1
    assert metrics["full_tuple"]["precision"] is None
    assert metrics["full_tuple"]["recall"] == 0
    assert metrics["coverage"]["candidate_omission_errors"] == 1
    assert metrics["coverage"]["true_candidate_abstentions"] == 0
    assert metrics["coverage"]["generation_coverage"] == 0
    assert metrics["all_query_count"] == 4
    assert metrics["all_query_accuracy"] == 0.75


def test_null_and_inactive_queries_penalize_spurious_writes():
    scm = FiniteSCM(7, "noisy_descendant", varied=False)
    truth = scm.truth()
    inactive = next(c for c in scm.candidates if c.context[0][1] != scm.active_context and c.target == scm.root)
    spurious = Mechanism(inactive.action, inactive.context, inactive.target, 1, 1)
    metrics = tuple_scores(truth | {spurious}, truth, scm.candidates, scm.candidates)
    assert metrics["full_tuple"]["precision"] == 0.5
    assert metrics["all_query_accuracy"] == 0.75
    assert metrics["coverage"]["spurious_candidate_writes"] == 1


def test_joint_score_is_not_product_of_coordinate_scores():
    c = (("c", 1),)
    truth = {Mechanism("a", c, "x", 0, 1), Mechanism("a", c, "z", 1, 2)}
    predicted = {Mechanism("a", c, "x", 1, 2), Mechanism("a", c, "z", 0, 1)}
    candidates = {m.candidate for m in truth}
    metrics = tuple_scores(predicted, truth, candidates, candidates)
    assert all(s["projected_set_exact"] for s in metrics["coordinate_projections"].values())
    assert metrics["positive_query_accuracy"] == 0
    assert metrics["all_query_accuracy"] == 0


def test_cost_ledger_counts_every_arm_context_lag_and_clamp():
    scm = FiniteSCM(7, "noisy_descendant")
    n = 31
    ledger = collect_evidence(scm, 17, n).ledger
    # Natural: 2 contexts * 2 lags * 2 arms * n.
    assert ledger["natural_joint_do_calls"] == 8 * n
    # Separation: 2 contexts * 2 candidates * 2 lags * 2 clamps * 2 arms * n.
    assert ledger["separation_joint_do_calls"] == 32 * n
    assert ledger["total_joint_do_calls"] == 40 * n
    assert ledger["total_atomic_assignments"] == 8 * n * 2 + 32 * n * 3
    assert ledger["fresh_endpoint_instances"] == 40 * n
    assert ledger["extra_reset_calls"] == ledger["extra_setup_calls"] == 0


def test_candidate_labels_are_anonymous_and_fixed_without_truth():
    scm = FiniteSCM(7, "modal")
    candidates = scm.candidates
    scm.root, scm.descendant = scm.descendant, scm.root
    assert scm.candidates == candidates
    assert {c.target for c in candidates} == set(scm.variables)
    assert {c.context[0][1] for c in candidates} == {0, 1}
    assert all("root" not in c.target and "descendant" not in c.target for c in candidates)


def test_collection_is_seed_reproducible():
    scm = FiniteSCM(7, "modal")
    assert collect_evidence(scm, 17, 61) == collect_evidence(scm, 17, 61)


def test_historical_matched_replay_has_honest_schema_gap_and_redacted_labels():
    protocol = json.loads((Path(__file__).resolve().parents[1] / "jmlr_revision_20261112/experiments/finite_protocol.json").read_text())
    protocol["historical_steps"] = 9
    result = run_historical(4100, 17, protocol)
    assert len(result["trajectory"]) == result["ledger"]["total_step_intervention_calls"] == 9
    assert {a["description"] for a in result["actions"]} == {"Available intervention."}
    assert len(result["methods"]) == 2
    for method in result["methods"].values():
        assert method["full_tuple"] is None
        assert method["full_tuple_accuracy"] is None
        assert method["tuple_coverage"] is None
        assert method["coordinate_status"]["delay"].startswith("absent")
        assert method["matched_training_records"] == 9
    assert sum(m["intervention_calls_attributable_to_method"] for m in result["methods"].values()) == 9


def test_population_artifact_records_exact_review_values():
    result = exact_population_examples()
    assert result["R1_modal"]["delta_y0"] == "-3/10"
    assert result["R2_delay"]["contrasts"] == ["3/5", "17/20"]
    assert result["R2_descendant"]["descendant_contrast_under_do_y0"] == "0"
    assert result["R2_descendant"]["descendant_contrast_under_do_y1"] == "0"


def test_invalid_scoring_universe_cannot_silently_drop_truth():
    m = Mechanism("a", (("c", 0),), "x", 1, 1)
    with pytest.raises(ValueError, match="include all true"):
        tuple_scores([], [m], [], [])
    with pytest.raises(ValueError, match="generated"):
        tuple_scores([m], [m], [m.candidate], [])


def test_zero_samples_rejected():
    with pytest.raises(ValueError, match="positive"):
        collect_evidence(FiniteSCM(7, "modal"), 1, 0)


@pytest.fixture
def public_sources(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for source in REQUIRED_SOURCES:
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / source, target)
    return tmp_path


def test_missing_confidential_reviews_do_not_block_required_source_hashes(public_sources):
    assert not (public_sources / "review").exists()
    provenance = source_provenance(public_sources)
    assert set(provenance["source_sha256"]) == set(REQUIRED_SOURCES)
    assert provenance["optional_review_sha256"] == {}


def test_existing_optional_review_provenance_is_recorded(public_sources):
    review = public_sources / "review/review-1.txt"
    review.parent.mkdir()
    review.write_text("Optional local provenance fixture.", encoding="utf-8")
    provenance = source_provenance(public_sources)
    assert provenance["optional_review_sha256"] == {
        "review/review-1.txt": hashlib.sha256(review.read_bytes()).hexdigest()}
    assert set(provenance["source_sha256"]) == set(REQUIRED_SOURCES)


def test_missing_required_scientific_source_still_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        source_provenance(tmp_path)


def test_clean_public_cli_reproduces_without_review_folder(public_sources):
    root = Path(__file__).resolve().parents[1]
    for source in (root / "causal_sandbox").glob("*.py"):
        shutil.copy2(source, public_sources / "causal_sandbox" / source.name)
    protocol = json.loads((root / "jmlr_revision_20261112/experiments/finite_protocol.json").read_text())
    protocol.update(family_seeds=[4100], replicate_seeds=[17], historical_steps=1,
                    pairs_per_cell=64, sample_budgets=[64], bootstrap_replicates=20)
    protocol_path = public_sources / "finite_protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    output = public_sources / "finite_results"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-m", "causal_sandbox.finite_interventions", "--protocol", str(protocol_path),
         "--output", str(output)], cwd=public_sources, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "finite_results.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["optional_review_sha256"] == {}
    assert set(result["source_sha256"]) == set(REQUIRED_SOURCES)
    assert result["historical_provenance"]["reused"] is False
    assert not (public_sources / "review").exists()
