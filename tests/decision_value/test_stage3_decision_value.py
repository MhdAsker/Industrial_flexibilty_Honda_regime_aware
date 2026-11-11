from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.honda.decision_value.data import DecisionDay, SettlementDay, load_spark_stage3_days
from src.honda.decision_value.evaluation import evaluate_policy, optimize_policy, settle_commitment, tariff_for_day
from src.honda.decision_value.pipeline import adversarial_audit, flex, grid, negative_controls
from src.honda.decision_value.statistics import bootstrap_mean, moving_block_bootstrap, paired_comparison
from src.honda.optimization.contracts import DecisionInput
from src.honda.optimization.optimizer import settle_schedule
from src.honda.scenarios.adapters import spark_root


ROOT = Path(__file__).resolve().parents[2]
SPARK = spark_root()
START = "aff0c9335f5e00985269e2a1aa6c27a112c6888d"
FROZEN_HASH = "e8903167aa9288b50f883142130e84732c37082a735073d01c4bab346dc20d02"
PROTECTED = {
    "config/honda_forecasting_protocol_frozen.json": "281f49331bc3b3c31aec0d33b50f6a83a149261db93ce04a29d67479cb62676e",
    "results/ml/test_predictions_2023.parquet": "301cfe03fd824cb741f192179b5f6bcc14100c3120729653eaf49339da3e9a67",
    "config/scenarios/scenario_method_frozen.json": "790d3574d468f10fbc565bef1ed0eb492d45ff9db00436f2700a74b7069171ba",
    "results/scenarios/spark_2024_diagnostic_scenarios.parquet": "8914336a0cc1dcd905ed5e2ed1587d0b58703856294be3631cfd9a087ab8dee0",
    "config/optimization/flexibility_archetypes.json": "ef1ad47b19d063ccdd2d73aaa08b54587a4d1fcf1d9f474ad5b144ab68a1c761",
    "config/optimization/tariff_scenarios.json": "a623c94f789b1bbdb7f7a4dd3dcb49659532561aada56ba109da3dbcfbfe2609",
}


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def one_day():
    decisions, settlements = load_spark_stage3_days(limit=1)
    key = next(iter(decisions))
    return decisions[key], settlements[key]


def test_starting_checkpoint_is_ancestor():
    subprocess.run(["git", "merge-base", "--is-ancestor", START, "HEAD"], cwd=ROOT, check=True)


@pytest.mark.parametrize("relative,expected", PROTECTED.items())
def test_protected_artifact_hashes(relative, expected):
    assert _sha(ROOT / relative) == expected


@pytest.mark.external_integrity
@pytest.mark.skipif(os.environ.get("STRICT_EXTERNAL_REPO_CHECKS") != "1", reason="set STRICT_EXTERNAL_REPO_CHECKS=1 to enforce companion-repository cleanliness")
def test_spark_repository_remains_clean_and_read_only():
    assert subprocess.check_output(["git", "-C", str(SPARK), "status", "--porcelain"], text=True) == ""
    assert subprocess.check_output(["git", "-C", str(SPARK), "rev-parse", "HEAD"], text=True).strip() == "32cff65f7f5384d08095d8d0325c45e8ce002ec3"


def test_frozen_experiment_is_immutable_and_pre_outcome():
    path = ROOT / "config/decision_value/stage3_experiment_frozen.json"
    assert _sha(path) == FROZEN_HASH
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["status"] == "FROZEN_BEFORE_HISTORICAL_OUTCOME_ANALYSIS"
    assert cfg["statistics"]["seed"] == 20261029


def test_documentation_amendment_does_not_rewrite_frozen_config():
    amendment = json.loads((ROOT / "config/decision_value/stage3_experiment_amendment_001.json").read_text(encoding="utf-8"))
    assert amendment["original_frozen_configuration_sha256"] == FROZEN_HASH
    assert amendment["actual_common_day_count"] == 273
    assert amendment["scientific_impact"].startswith("Documentation-only")


def test_decision_contract_excludes_realized_target():
    assert "actual_kw" not in DecisionDay.__dataclass_fields__
    assert "actual_kw" not in DecisionInput.__dataclass_fields__
    assert "actual_kw" in SettlementDay.__dataclass_fields__


def test_day_contract_and_context_are_aligned(one_day):
    day, settlement = one_day
    n = len(day.timestamp_utc)
    assert n in {92, 96, 100}
    assert len(day.point_forecast_kw) == len(settlement.actual_kw) == n
    assert np.asarray(day.s1_scenarios_kw).shape == (50, n)
    assert np.asarray(day.s2_scenarios_kw).shape == (50, n)
    assert np.isclose(sum(day.scenario_weights), 1)
    assert pd.Timestamp(day.forecast_issue_time_utc) < pd.Timestamp(day.timestamp_utc[0])


def test_no_future_observed_components_in_decision_contract():
    fields = DecisionDay.__dataclass_fields__
    assert not {"pv", "chp", "weather", "thermal", "actual_kw"} & set(fields)


@pytest.mark.parametrize("policy", ["P0_NO_FLEX", "P2_LGBM_DETERMINISTIC", "P3_S1_INDEPENDENT", "P4_S2_COHERENT", "P6_S2_CVAR"])
def test_implementable_policies_solve_without_actual(policy, one_day):
    day, _ = one_day
    commitment, _ = optimize_policy(day, policy, flex(), grid(), risk_lambda=.25)
    assert commitment.solver_status in {"optimal", "not_required"}
    assert len(commitment.schedule_kw) == len(day.timestamp_utc)


def test_oracle_requires_explicit_actual(one_day):
    day, _ = one_day
    with pytest.raises(ValueError, match="Oracle"):
        optimize_policy(day, "P7_ORACLE_UPPER_BOUND", flex(), grid())


def test_settlement_does_not_reoptimize(one_day):
    day, settlement = one_day
    commitment, _ = optimize_policy(day, "P2_LGBM_DETERMINISTIC", flex(), grid())
    before = np.asarray(commitment.schedule_kw).copy()
    metrics = settle_commitment(commitment, settlement, day, flex(), grid())
    np.testing.assert_array_equal(before, np.asarray(commitment.schedule_kw))
    assert metrics["savings_eur"] == pytest.approx(metrics["no_flex_cost_eur"] - metrics["realized_cost_eur"])


def test_realized_grid_violation_is_reported_not_clipped(one_day):
    day, settlement = one_day
    commitment, metrics = evaluate_policy(day, settlement, "P2_LGBM_DETERMINISTIC", flex(), grid())
    assert isinstance(metrics["realized_grid_cap_compliant"], bool)
    assert metrics["maximum_export_kw"] >= metrics["export_violation_max_kw"]
    assert commitment.schedule_kw == tuple(commitment.schedule_kw)


def test_nonanticipativity_is_one_shared_schedule(one_day):
    day, _ = one_day
    commitment, result = optimize_policy(day, "P4_S2_COHERENT", flex(), grid())
    assert commitment.scenario_count == 50
    assert result is not None and result.schedule_kw.ndim == 1
    assert result.scenario_import_kw.shape[0] == 50


def test_scenario_permutation_invariance(one_day):
    day, _ = one_day
    order = np.arange(50)[::-1]
    matrix = np.asarray(day.s2_scenarios_kw)
    weights = np.asarray(day.scenario_weights)
    permuted = replace(day, s2_scenarios_kw=tuple(map(tuple, matrix[order])), scenario_weights=tuple(weights[order]))
    a, _ = optimize_policy(day, "P4_S2_COHERENT", flex(), grid())
    b, _ = optimize_policy(permuted, "P4_S2_COHERENT", flex(), grid())
    np.testing.assert_allclose(a.schedule_kw, b.schedule_kw, atol=1e-7)


def test_identical_scenarios_reduce_to_deterministic(one_day):
    day, _ = one_day
    point = tuple(day.point_forecast_kw)
    collapsed = replace(day, s2_scenarios_kw=(point, point), scenario_weights=(.5, .5))
    det, _ = optimize_policy(day, "P2_LGBM_DETERMINISTIC", flex(), grid())
    sto, _ = optimize_policy(collapsed, "P4_S2_COHERENT", flex(), grid())
    np.testing.assert_allclose(det.schedule_kw, sto.schedule_kw, atol=1e-7)


def test_cost_consistency_between_optimizer_and_settlement(one_day):
    day, _ = one_day
    pseudo = SettlementDay(day.dataset_id, day.target_day_local, day.timestamp_utc, day.point_forecast_kw, False, False, 1)
    commitment, metrics = evaluate_policy(day, pseudo, "P2_LGBM_DETERMINISTIC", flex(), grid())
    assert commitment.predicted_objective_eur == pytest.approx(metrics["realized_cost_eur"], abs=1e-7)


def test_negative_prices_and_signed_exchange_settle_correctly():
    from src.honda.optimization.contracts import TariffConfig
    tariff = TariffConfig.from_eur_per_mwh([-100, 100], [-150, 50])
    result = settle_schedule(np.zeros(2), np.array([5., -5.]), np.full(2, .25), tariff)
    assert result["import_kw"][0] == 5 and result["export_kw"][1] == 5
    assert np.isfinite(result["cost_eur"])


def test_all_negative_controls_pass(one_day):
    day, settlement = one_day
    controls = negative_controls({day.target_day_local: day}, {day.target_day_local: settlement})
    assert len(controls) == 10
    assert controls.passed.astype(bool).all()


def test_bootstraps_are_deterministic_and_block_aware():
    values = np.arange(20, dtype=float)
    np.testing.assert_array_equal(bootstrap_mean(values), bootstrap_mean(values))
    np.testing.assert_array_equal(moving_block_bootstrap(values), moving_block_bootstrap(values))
    assert not np.array_equal(bootstrap_mean(values), moving_block_bootstrap(values))


def test_paired_comparison_uses_common_days_only():
    frame = pd.DataFrame({
        "target_day_local": ["a", "b", "a", "c"],
        "policy": ["A", "A", "B", "B"],
        "realized_cost_eur": [2., 3., 1., 9.],
    })
    result = paired_comparison(frame, "A", "B", label="A_minus_B", resamples=20)
    assert result["common_days"] == 1
    assert result["mean_cost_A_minus_B_eur"] == 1


def test_primary_artifacts_have_complete_common_coverage():
    daily = pd.read_parquet(ROOT / "results/decision_value/policy_daily_metrics.parquet")
    assert daily.target_day_local.nunique() == 273
    assert set(daily.groupby("policy").size()) == {273}
    assert daily.realized_cost_eur.notna().all()
    assert {92, 96, 100}.issuperset(set(daily.groupby("target_day_local").first().index.map(lambda _: 96)))


def test_operational_feasibility_is_not_hidden():
    daily = pd.read_parquet(ROOT / "results/decision_value/policy_daily_metrics.parquet")
    violations = daily.groupby("policy").realized_grid_cap_compliant.apply(lambda x: int((~x).sum()))
    assert violations["P2_LGBM_DETERMINISTIC"] == 82
    assert violations["P3_S1_INDEPENDENT"] == 16
    assert violations["P4_S2_COHERENT"] == 11
    assert (~daily.loc[~daily.realized_grid_cap_compliant, "operational_regret_interpretable"]).all()


def test_honda_is_not_fabricated_into_economic_results():
    eligibility = pd.read_csv(ROOT / "results/decision_value/evaluation_day_eligibility.csv")
    honda = eligibility[eligibility.dataset.eq("HONDA")].iloc[0]
    assert honda.common_primary_days == 0
    assert honda.status == "NOT_EVALUABLE_NO_AUTHORITATIVE_PRICE"


def test_adversarial_audit_has_no_unacknowledged_failure():
    audit = adversarial_audit()
    assert set(audit.status) <= {"PASS", "LIMITATION", "NOT_EVALUABLE"}
    assert len(audit) == 20


def test_manifest_covers_exact_dates_and_frozen_hash():
    manifest = json.loads((ROOT / "results/decision_value/stage3_manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen_experiment_sha256"] == FROZEN_HASH
    assert len(manifest["evaluation_dates"]) == 273
    assert manifest["market_timing"] == "EX_POST_PRICE_AWARE"

