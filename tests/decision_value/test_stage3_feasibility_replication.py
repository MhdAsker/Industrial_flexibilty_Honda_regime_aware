from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.honda.decision_value_replication.replication import (
    CONFIG,
    EXPECTED_CONFIG_HASH,
    explicit_zero_export_settlement_correction,
)
from src.honda.scenarios.adapters import spark_root


ROOT = Path(__file__).resolve().parents[2]
SPARK = spark_root()
OUT = ROOT / "results/decision_value_replication"
PROTECTED = {
    "config/decision_value/stage3_experiment_frozen.json": "e8903167aa9288b50f883142130e84732c37082a735073d01c4bab346dc20d02",
    "results/decision_value/policy_daily_metrics.parquet": "9adbf3efa9c88dd20f496533328f7f521965195bfb9451825e15ed07ca3e088c",
    "results/decision_value/policy_decisions.parquet": "f309b856149d8b4e4be0b2856b9feb22ade58810f2b36b767fd4ea79b7351e20",
    "config/scenarios/scenario_method_frozen.json": "790d3574d468f10fbc565bef1ed0eb492d45ff9db00436f2700a74b7069171ba",
}


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_replication_configuration_is_frozen_before_outcomes():
    assert _sha(CONFIG) == EXPECTED_CONFIG_HASH
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["status"] == "FROZEN_BEFORE_REPLICATION_OUTCOME_ANALYSIS"
    assert cfg["physical_asset"]["p_up_kw"] == 5
    assert cfg["physical_asset"]["e_advance_kwh"] == 10
    assert cfg["information_timing"]["classification"] == "POST_PUBLICATION_TIMING_ASSUMED"
    assert len(cfg["primary_comparisons"]) == 3


@pytest.mark.parametrize("relative,expected", PROTECTED.items())
def test_original_stage3_and_scenario_artifacts_unchanged(relative, expected):
    assert _sha(ROOT / relative) == expected


@pytest.mark.external_integrity
@pytest.mark.skipif(os.environ.get("STRICT_EXTERNAL_REPO_CHECKS") != "1", reason="set STRICT_EXTERNAL_REPO_CHECKS=1 to enforce companion-repository cleanliness")
def test_spark_repository_is_still_clean():
    assert subprocess.check_output(["git", "-C", str(SPARK), "status", "--porcelain"], text=True) == ""
    assert subprocess.check_output(["git", "-C", str(SPARK), "rev-parse", "HEAD"], text=True).strip() == "32cff65f7f5384d08095d8d0325c45e8ce002ec3"


def test_explicit_correction_is_minimum_and_does_not_mutate_schedule():
    actual = np.array([2.0, 2.0, 2.0])
    schedule = np.array([-3.0, -1.0, 1.0])
    before = schedule.copy()
    result = explicit_zero_export_settlement_correction(actual, schedule, np.array([100.0, 100.0, 100.0]))
    np.testing.assert_array_equal(schedule, before)
    np.testing.assert_allclose(result["nominal_grid_kw"], [-1, 1, 3])
    np.testing.assert_allclose(result["correction_kw"], [1, 0, 0])
    np.testing.assert_allclose(result["corrected_grid_kw"], [0, 1, 3])
    assert result["violation_intervals"] == 1
    assert result["correction_energy_kwh"] == pytest.approx(.25)


def test_negative_price_does_not_create_extra_correction_arbitrage():
    result = explicit_zero_export_settlement_correction(
        np.array([1.0, 1.0]), np.array([-2.0, 0.0]), np.array([-100.0, -100.0])
    )
    np.testing.assert_allclose(result["correction_kw"], [1.0, 0.0])
    np.testing.assert_allclose(result["corrected_grid_kw"], [0.0, 1.0])


def test_correction_rejects_misaligned_or_nonfinite_inputs():
    with pytest.raises(ValueError):
        explicit_zero_export_settlement_correction(np.ones(2), np.ones(3), np.ones(2))
    with pytest.raises(ValueError):
        explicit_zero_export_settlement_correction(np.array([np.nan]), np.zeros(1), np.zeros(1))


def test_replication_does_not_import_scenario_generation():
    source = (ROOT / "src/honda/decision_value_replication/replication.py").read_text(encoding="utf-8")
    assert "generate_scenarios" not in source
    assert "load_spark_stage3_days" not in source


def test_day_level_outputs_cover_only_the_three_frozen_policies():
    daily = pd.read_parquet(OUT / "day_level_replication_metrics.parquet")
    assert daily.target_day_local.nunique() == 273
    assert set(daily.policy) == {"P2_LGBM_DETERMINISTIC", "P3_S1_INDEPENDENT", "P4_S2_COHERENT"}
    assert set(daily.groupby("policy").size()) == {273}
    assert daily.common_feasible_day.groupby(daily.target_day_local).first().sum() == 191
    assert (daily.timing_classification == "POST_PUBLICATION_TIMING_ASSUMED").all()


def test_common_feasible_population_is_identical_for_all_policies():
    daily = pd.read_parquet(OUT / "day_level_replication_metrics.parquet")
    dates = daily.loc[daily.common_feasible_day].groupby("policy").target_day_local.apply(set)
    assert len(dates.iloc[0]) == 191
    assert all(value == dates.iloc[0] for value in dates.iloc[1:])
    assert daily.loc[daily.common_feasible_day, "cap_compliant"].all()


def test_updated_violation_counts_reproduce_frozen_stage3():
    summary = pd.read_csv(OUT / "replication_policy_summary.csv").set_index("policy")
    assert summary.loc["P2_LGBM_DETERMINISTIC", "cap_violation_days"] == 82
    assert summary.loc["P3_S1_INDEPENDENT", "cap_violation_days"] == 16
    assert summary.loc["P4_S2_COHERENT", "cap_violation_days"] == 11


def test_primary_pairwise_stats_use_common_feasible_days():
    frame = pd.read_csv(OUT / "primary_common_feasible_comparisons.csv").set_index("comparison")
    assert set(frame.common_days) == {191}
    assert frame.loc["P2_MINUS_P4", "mean_cost_A_minus_B_eur_day"] == pytest.approx(-0.1911909478)
    assert frame.loc["P3_MINUS_P4", "mean_cost_A_minus_B_eur_day"] == pytest.approx(-0.04723961278)
    assert frame.loc["P2_MINUS_P3", "mean_cost_A_minus_B_eur_day"] == pytest.approx(-0.1439513350)
    assert (frame.result_classification == "PRE_REGISTERED_PRIMARY").all()


def test_corrected_stats_use_all_days_and_explicit_labels():
    frame = pd.read_csv(OUT / "secondary_corrected_comparisons.csv")
    assert set(frame.common_days) == {273}
    assert (frame.result_classification == "PRE_SPECIFIED_SECONDARY").all()
    daily = pd.read_parquet(OUT / "day_level_replication_metrics.parquet")
    np.testing.assert_allclose(
        daily.corrected_cost_eur - daily.nominal_cost_eur,
        daily.correction_cost_eur,
        atol=1e-12,
    )
    assert (daily.customer_value_label == "NOT_CUSTOMER_SAVINGS").all()
    assert (daily.profit_label == "NOT_TRADING_PROFIT").all()
    assert (daily.physical_label == "NOT_PLANT_VALIDATION").all()


def test_no_silent_clipping_or_schedule_rewrite():
    daily = pd.read_parquet(OUT / "day_level_replication_metrics.parquet")
    violating = daily[~daily.cap_compliant]
    assert len(violating) == 109
    assert (violating.correction_energy_kwh > 0).all()
    assert (daily.schedule_energy_residual_kwh.abs() < 1e-10).all()
    assert (daily.terminal_state_residual_kwh < 1e-10).all()


def test_binding_diagnostics_are_explicitly_exploratory():
    binding = pd.read_csv(OUT / "binding_constraint_diagnostics.csv")
    assert (binding.classification == "EXPLORATORY_DIAGNOSTIC").all()
    assert binding.maximum_terminal_state_residual_kwh.max() < 1e-10


def test_bootstrap_seed_and_block_length_are_frozen():
    primary = pd.read_csv(OUT / "primary_common_feasible_comparisons.csv")
    assert set(primary.seed) == {20261029}
    assert set(primary.bootstrap_resamples) == {2000}
    assert set(primary.block_length_days) == {7}
    assert primary.holm_adjusted_p.notna().all()


def test_reproducibility_check_passes():
    result = json.loads((OUT / "reproducibility_check.json").read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["mismatches"] == []


def test_manifest_records_frozen_hashes_and_no_honda_economics():
    manifest = json.loads((OUT / "replication_manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen_config_sha256"] == EXPECTED_CONFIG_HASH
    assert manifest["candidate_days"] == 273
    assert manifest["policies"] == list(("P2_LGBM_DETERMINISTIC", "P3_S1_INDEPENDENT", "P4_S2_COHERENT"))
    assert "HONDA" not in json.dumps(manifest)
