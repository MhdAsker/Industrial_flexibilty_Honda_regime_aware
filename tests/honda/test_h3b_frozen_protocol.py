from __future__ import annotations

import hashlib
import json
import subprocess

import pandas as pd
import pytest

from src.honda import h3b_freeze_protocol as h3b


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_h3a_and_h2_checkpoints_and_artifacts_unchanged():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=h3b.ROOT, text=True).strip()
    assert subprocess.run(["git", "merge-base", "--is-ancestor", h3b.H3A_COMMIT, head], cwd=h3b.ROOT).returncode == 0
    h3a_protected = [
        "config/honda_h3a_protocol_analysis.json", "reports/h3a_measurement_availability.md",
        "reports/honda_forecasting_protocol_h3a.md", "results/protocol_h3a/honda_h3a_manifest.json",
        "src/honda/h3a_protocol_analysis.py",
    ]
    h2_protected = [
        "config/honda_h2.json", "reports/honda_provenance_and_silver.md", "results/provenance/honda_h2_manifest.json",
        "src/honda/h2_provenance.py", "tests/honda/test_h2_provenance.py",
    ]
    assert subprocess.run(["git", "diff", "--quiet", h3b.H3A_COMMIT, "--", *h3a_protected], cwd=h3b.ROOT).returncode == 0
    assert subprocess.run(["git", "diff", "--quiet", h3b.H2_COMMIT, "--", *h2_protected], cwd=h3b.ROOT).returncode == 0


def test_primary_target_sign_resolution_and_issuance_exact():
    c = h3b.CONFIG
    assert c["primary_target"] == "grid_exchange_kw_signed"
    assert c["units"] == "kW" and c["resolution_minutes"] == 15
    assert c["sign_convention"] == {"positive": "grid import", "negative": "grid export", "prediction_clipping": "FORBIDDEN"}
    assert c["forecast_issue_local_time"] == "13:00" and c["timezone"] == "Europe/Berlin"
    silver = pd.read_parquet(h3b.SILVER_PATH, columns=["grid_exchange_kw_signed"])
    assert silver.grid_exchange_kw_signed.min() < 0
    index = pd.read_parquet(h3b.INDEX_PATH, columns=["forecast_issue_time_local"])
    assert set(index.forecast_issue_time_local.dt.hour) == {13}


def test_measurement_completion_cutoff_boundary():
    issue = pd.Timestamp("2023-07-01T11:00:00Z")
    assert h3b.CONFIG["historical_measurement_cutoff_rule"] == "interval_end_utc <= forecast_issue_time_utc"
    assert h3b.historical_interval_available(pd.Timestamp("2023-07-01T10:45:00Z"), issue)
    assert not h3b.historical_interval_available(pd.Timestamp("2023-07-01T11:00:00Z"), issue)


def test_dst_physical_day_counts_and_unique_utc_keys():
    index = pd.read_parquet(h3b.INDEX_PATH, columns=["timestamp_utc", "target_day_local"])
    sizes = index.groupby("target_day_local").size()
    assert sizes.value_counts().to_dict() == {96: 1453, 92: 4, 100: 4}
    assert index.timestamp_utc.is_unique


def test_target_and_predictor_provenance_rules():
    c = h3b.CONFIG
    assert c["target_primary_provenance"] == ["OBSERVED_ORIGINAL"]
    assert c["target_sensitivity_provenance"] == ["OBSERVED_ORIGINAL", "CORRECTED_NONIMPUTED"]
    assert set(c["target_forbidden_provenance"]) == {"IMPUTED_OR_FILLED", "UNKNOWN_PROVENANCE", "MISSING_UNRESOLVED"}
    predictor = c["historical_predictor_provenance_policy"]
    assert predictor["OBSERVED_ORIGINAL"] == "PRIMARY_ALLOWED"
    assert predictor["CORRECTED_NONIMPUTED"] == "SENSITIVITY_SAFE_WITH_EXPLICIT_FLAG"
    assert all(predictor[state] == "UNAVAILABLE" for state in ["IMPUTED_OR_FILLED", "UNKNOWN_PROVENANCE", "MISSING_UNRESOLVED"])


def test_missing_history_never_changes_canonical_target_set():
    policy = h3b.CONFIG["missing_history_policy"]
    assert policy["canonical_target_set_independent_of_context_completeness"] is True
    assert policy["drop_valid_target_for_missing_predictor"] is False
    assert policy["global_context_threshold"] is None
    assert policy["protocol_generation_imputation"] == "FORBIDDEN"


def test_p1_and_p2_chronology_and_no_2023_development_access():
    p1 = h3b.CONFIG["primary_development_folds"]
    assert p1 == [
        {"fold": "P1_FOLD_A", "train_years": [2020], "validation_years": [2021]},
        {"fold": "P1_FOLD_B", "train_years": [2020, 2021], "validation_years": [2022]},
    ]
    assert h3b.CONFIG["final_development_refit_years"] == [2020, 2021, 2022]
    assert h3b.CONFIG["final_test_period"]["start_local"] == "2023-01-01"
    p2 = h3b.CONFIG["structural_sensitivity_folds"]
    assert p2["development_train_years"] == [2021] and p2["development_validation_years"] == [2022]
    assert 2023 not in h3b.CONFIG["final_development_refit_years"] and 2023 not in p2["final_refit_years"]


def test_2023_test_access_guard():
    for purpose in h3b.CONFIG["test_access_guard"]["forbidden_uses"]:
        with pytest.raises(PermissionError):
            h3b.assert_development_test_guard(pd.Timestamp("2023-06-01T00:00:00Z"), purpose)
    h3b.assert_development_test_guard(pd.Timestamp("2022-06-01T00:00:00Z"), "model_choice")


def test_lead_bands_exhaustive_nonoverlapping_and_dst_safe():
    index = pd.read_parquet(h3b.INDEX_PATH, columns=["lead_time_minutes", "lead_band", "target_day_local"])
    assert not index.lead_band.isna().any()
    assert set(index.lead_band) == {"H1", "H2", "H3", "H4"}
    assert (index.lead_band == index.lead_time_minutes.div(60).map(h3b.lead_band)).all()
    assert index.lead_band.value_counts().to_dict() == {"H1": 35064, "H2": 35064, "H3": 35064, "H4": 35064}
    with pytest.raises(ValueError): h3b.lead_band(10.999)
    with pytest.raises(ValueError): h3b.lead_band(36.001)


def test_hessen_holiday_and_bridge_policy():
    assert h3b.CONFIG["holiday_state"] == "Hessen" and h3b.CONFIG["holiday_region_code"] == "DE-HE"
    assert h3b.CONFIG["bridge_day_rule"]["use"] == "DESCRIPTIVE_DIAGNOSTIC_ONLY"
    index = pd.read_parquet(h3b.INDEX_PATH, columns=["target_day_local", "holiday_context"])
    assert index[index.holiday_context.str.contains("BRIDGE")].target_day_local.nunique() == 9


def test_feature_contract_forbids_realized_future_and_accepts_safe_history():
    f = pd.read_csv(h3b.OUT / "frozen_feature_contract.csv")
    future_forbidden = f.classification.eq("FORBIDDEN_REALIZED_FUTURE")
    assert {"Ta ambient temperature", "Igm irradiance", "PV generation", "CHP generation/state", "heating", "cooling"}.issubset(set(f.loc[future_forbidden, "variable_or_domain"]))
    historical = f[f.time_relation.eq("historical completed")]
    assert set(historical.classification) == {"HISTORICAL_ONLY"}
    mode = f[f.variable_or_domain.eq("realized target grid mode")].iloc[0]
    assert mode.classification == "DIAGNOSTIC_ONLY"


def test_grid_mode_is_diagnostic_and_epsilon_zero():
    assert h3b.CONFIG["near_zero_tolerance"]["epsilon_kw"] == 0.0
    assert h3b.grid_mode(1.0) == "IMPORT" and h3b.grid_mode(-1.0) == "EXPORT" and h3b.grid_mode(0.0) == "NEUTRAL"
    index = pd.read_parquet(h3b.INDEX_PATH, columns=["grid_mode_diagnostic_only"])
    assert set(index.grid_mode_diagnostic_only).issubset({"IMPORT", "EXPORT", "NEUTRAL"})


def test_metrics_use_absolute_scale_and_no_headline_mape():
    metrics = h3b.CONFIG["primary_metrics"]
    assert metrics["NMAE_absolute_target_scale"] == "mean(abs(y_pred - y_true)) / mean(abs(y_true))"
    assert "MAPE" not in metrics
    assert h3b.CONFIG["secondary_metrics"]["ordinary_MAPE"] == "FORBIDDEN_HEADLINE_METRIC"
    assert h3b.CONFIG["secondary_metrics"]["absolute_target_scaled_WAPE"] == "sum(abs(y_pred - y_true)) / sum(abs(y_true))"


def test_bootstrap_pairwise_and_prediction_coverage_rules():
    assert h3b.CONFIG["bootstrap_unit"] == "local_calendar_day"
    assert h3b.CONFIG["bootstrap_resamples"] == 2000 and h3b.CONFIG["bootstrap_seed"] == 20261029
    assert "common eligible targets" in h3b.CONFIG["pairwise_comparison_rule"]
    required = {"eligible_target_count", "predicted_target_count", "prediction_coverage_fraction", "complete_day_forecast_coverage", "fallback_usage"}
    assert set(h3b.CONFIG["prediction_coverage_reporting"]) == required


def test_foundation_model_fairness_context_and_zero_shot_rules():
    rules = h3b.CONFIG["foundation_model_rules"]
    assert rules["same_issuance_target_test_quality_metrics_and_DST"] is True
    assert rules["canonical_target_set_independent_of_model_context"] is True
    assert rules["common_context_subset_report_if_needed"] is True
    assert rules["zero_shot_before_fine_tuning"] is True
    assert rules["zero_shot_and_fine_tuned_results_separate"] is True
    assert rules["candidate_context_days"] == [7, 14, 28, 56]
    assert rules["hidden_author_imputation_forbidden"] is True


def test_probabilistic_and_coherent_scenario_rules():
    assert "2023 calibration is forbidden" in h3b.CONFIG["probabilistic_test_rule"]
    assert "temporally coherent daily trajectories" in h3b.CONFIG["coherent_scenario_requirement"]
    assert "marginal quantiles alone are insufficient" in h3b.CONFIG["coherent_scenario_requirement"]


def test_no_model_or_prediction_artifacts_created():
    assert h3b.CONFIG["model_training_allowed_in_h3b"] is False
    assert not [p for p in (h3b.ROOT / "models").rglob("*") if p.is_file()]
    assert not [p for p in (h3b.ROOT / "data/gold").rglob("*") if p.is_file()]
    index = pd.read_parquet(h3b.INDEX_PATH)
    assert not any("prediction" in column.lower() for column in index.columns)
    source = (h3b.ROOT / "src/honda/h3b_freeze_protocol.py").read_text(encoding="utf-8").lower()
    assert not any(token in source for token in ["lightgbm", "xgboost", "chronos", "timesfm", "timegpt", ".fit(", "predict(", "kmeans"])


def test_frozen_config_and_forecasting_index_match_manifest_hashes():
    manifest = json.loads((h3b.OUT / "honda_h3_manifest.json").read_text())
    assert sha256(h3b.CONFIG_PATH) == manifest["frozen_protocol_sha256"]
    assert sha256(h3b.INDEX_PATH) == manifest["forecast_index_sha256"]
    assert len(pd.read_parquet(h3b.INDEX_PATH)) == 140256
    assert manifest["model_training_performed"] is False and manifest["predictions_created"] is False
