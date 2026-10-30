from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.honda import h4_baseline_benchmark as h4
from src.honda import h5_deterministic_ml as h5


def _config() -> dict:
    return json.loads(h5.CONFIG.read_text(encoding="utf-8"))


def test_h4_checkpoint_and_frozen_protocol_are_unchanged():
    h5.verify_checkpoint()
    assert h5.digest(h4.PROTOCOL) == h5.H3_PROTOCOL_HASH
    assert h5.digest(h4.SELECTION) == h5.H4_SELECTION_HASH


def test_target_sign_and_negative_predictions_are_preserved():
    p = pd.read_parquet(h5.OUT / "test_predictions_2023.parquet")
    assert p.actual_kw.min() < 0
    assert p.PRIMARY_ML.min() < 0
    assert not np.all(p.PRIMARY_ML >= 0)


def test_window_features_stop_at_the_issuance_boundary():
    issue = pd.DatetimeIndex([pd.Timestamp("2023-01-02 12:00", tz="UTC")])
    safe = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2023-01-02 11:30Z", "2023-01-02 11:45Z", "2023-01-02 12:00Z"]
            ),
            "value": [1.0, 2.0, 999.0],
        }
    )
    result = h5._window_stats(safe, issue, "x", {"1h": 60}).iloc[0]
    assert result.x_1h_last == 2.0
    assert result.x_1h_max == 2.0


def test_training_rows_simulate_their_own_issuance_and_lineage_is_clean():
    audit = pd.read_csv(h5.OUT / "feature_lineage_audit.csv")
    issue = pd.to_datetime(audit.forecast_issue_time_utc, utc=True)
    latest = pd.to_datetime(audit.latest_permitted_interval_start_utc, utc=True)
    assert (latest + pd.Timedelta(minutes=15) <= issue).all()
    assert not audit.future_realized_used.any()
    assert not audit.imputed_or_unknown_used.any()
    assert not audit.leakage_violation.any()


def test_m0_m1_contracts_and_forbidden_future_inputs():
    cfg = _config()
    m0 = set(cfg["exact_features"]["LGBM_M0"])
    m1 = set(cfg["exact_features"]["LGBM_M1"])
    assert m0 < m1
    assert any(x.startswith("chp_") for x in m1 - m0)
    forbidden = {"future_Ta", "future_Igm", "future_PV", "future_CHP", "future_heating", "future_cooling"}
    assert forbidden.isdisjoint(m1)
    catalog = pd.read_csv(h5.OUT / "feature_catalog.csv")
    assert set(catalog.loc[catalog.feature_type == "NOT_EVALUABLE", "feature_name"]) == {"weather_history", "thermal_history"}


def test_imputed_and_unknown_history_are_never_consumed():
    audit = pd.read_csv(h5.OUT / "leakage_audit.csv").set_index("check")
    assert audit.loc["imputed_or_unknown_history", "violations"] == 0


def test_fold_chronology_and_test_selection_guard():
    cfg = _config()
    rules = cfg["training_period_rules"]
    assert rules["Fold_A"] == {"train": [2020], "validation": [2021]}
    assert rules["Fold_B"] == {"train": [2020, 2021], "validation": [2022]}
    assert rules["test"] == [2023]
    assert cfg["selection_status"] == "FROZEN_BEFORE_TEST"
    assert cfg["test_used_for_selection"] is False
    assert "2023" not in cfg["selection_metric"]


def test_test_scoring_requires_frozen_config(monkeypatch, tmp_path):
    frame = pd.DataFrame()
    monkeypatch.setattr(h5, "CONFIG", tmp_path / "missing.json")
    with pytest.raises(PermissionError):
        h5.final_models(frame, {}, {})


def test_b5_predictions_are_byte_for_byte_numerically_unchanged():
    ml = pd.read_parquet(h5.OUT / "test_predictions_2023.parquet").sort_values("timestamp_utc")
    h4p = pd.read_parquet(h4.OUT / "test_predictions_2023.parquet")
    h4p = h4p[(h4p.protocol == "P1") & h4p.baseline_id.eq("B5") & h4p.target_primary_eligible].sort_values("timestamp_utc")
    assert np.array_equal(ml.B5.to_numpy(), h4p.prediction_kw.to_numpy())
    assert pd.read_csv(h5.OUT / "leakage_audit.csv").violations.sum() == 0


def test_development_only_selection_and_deterministic_seed():
    cfg = _config()
    assert cfg["random_seed"] == h5.SEED == 20261029
    hp = pd.read_csv(h5.OUT / "lgbm_hyperparameter_selection.csv")
    assert set(hp.validation_year.astype(str)) <= {"2021", "2022", "2021+2022"}
    assert 2023 not in set(pd.to_numeric(hp.validation_year, errors="coerce").dropna().astype(int))


def test_feature_order_matches_serialized_models():
    cfg = _config()
    importance = pd.read_csv(h5.OUT / "feature_importance.csv")
    for model, key in [("LGBM_M0", "LGBM_M0"), ("LGBM_M1", "LGBM_M1")]:
        assert set(importance.loc[importance.model == model, "feature_name"]) == set(cfg["exact_features"][key])


def test_full_prediction_coverage_and_common_target_comparison():
    summary = pd.read_csv(h5.OUT / "test_model_summary.csv")
    primary = summary[summary.model == "PRIMARY_ML"].iloc[0]
    assert primary.coverage == 1.0
    p = pd.read_parquet(h5.OUT / "test_predictions_2023.parquet")
    assert len(p) == 34161 and p.PRIMARY_ML.notna().all() and p.B5.notna().all()


def test_complete_day_bootstrap_is_deterministic_and_separated_from_p2():
    for name in ["bootstrap_vs_b5.csv", "bootstrap_vs_b3.csv"]:
        b = pd.read_csv(h5.OUT / name).iloc[0]
        assert b.seed == h5.SEED
        assert b.resamples == h5.BOOTSTRAPS
        assert b.common_complete_days == 347
    p = pd.read_parquet(h5.OUT / "test_predictions_2023.parquet")
    assert {"PRIMARY_ML", "PRIMARY_ML_P2"}.issubset(p.columns)
    assert not np.array_equal(p.PRIMARY_ML.to_numpy(), p.PRIMARY_ML_P2.to_numpy())


def test_mode_diagnostics_and_complete_day_metrics_are_present():
    mode = pd.read_csv(h5.OUT / "test_grid_mode_summary.csv")
    row = mode[mode.model == "PRIMARY_ML"].iloc[0]
    assert 0 <= row.sign_accuracy <= 1
    assert 0 <= row.import_recall <= 1
    assert 0 <= row.export_recall <= 1
    daily = pd.read_csv(h5.OUT / "test_daily_summary.csv")
    assert {"daily_mae_mean", "daily_mae_median", "daily_mae_p90", "daily_mae_max"}.issubset(daily.columns)


def test_manifest_hashes_and_scope_guards():
    manifest = json.loads(h5.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_code_base_commit"] == h5.H4_COMMIT
    assert manifest["generator_source_sha256"] == h5.digest(Path(h5.__file__))
    for rel, meta in manifest["deterministic_artifacts"].items():
        assert h5.digest(h5.ROOT / rel) == meta["sha256"]
    assert manifest["foundation_models_run"] is False
    assert manifest["probabilistic_models_run"] is False
    assert manifest["optimization_run"] is False
    names = {p.name.lower() for p in Path(h5.OUT).rglob("*")}
    assert not any(any(token in name for token in ["chronos", "timesfm", "timegpt", "quantile", "optimization"]) for name in names)
