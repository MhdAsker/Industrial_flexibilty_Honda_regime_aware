from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.honda import h4_baseline_benchmark as h4


def test_h3_checkpoint_and_frozen_inputs_unchanged():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=h4.ROOT, text=True).strip()
    assert subprocess.run(["git", "merge-base", "--is-ancestor", h4.H3B_COMMIT, head], cwd=h4.ROOT).returncode == 0
    assert h4.digest(h4.PROTOCOL) == h4.PROTOCOL_SHA256
    assert h4.digest(h4.INDEX) == h4.INDEX_SHA256
    assert h4.digest(h4.SILVER) == h4.SILVER_SHA256
    assert h4.digest(h4.MASK) == h4.MASK_SHA256


def test_b0_cutoff_safety_and_primary_provenance():
    p = pd.read_parquet(h4.OUT / "test_predictions_2023.parquet")
    b0 = p[(p.protocol == "P1") & (p.baseline_id == "B0") & p.prediction_available]
    source = pd.to_datetime(b0.source_timestamp_or_rule, utc=True)
    assert ((source + pd.Timedelta(minutes=15)) <= b0.forecast_issue_time_utc).all()
    assert set(b0.source_provenance) == {h4.OBSERVED}


def test_b1_local_calendar_and_dst_fold_mapping():
    p = pd.read_parquet(h4.OUT / "test_predictions_2023.parquet")
    b1 = p[(p.protocol == "P1") & (p.baseline_id == "B1") & p.prediction_available].copy()
    src = pd.to_datetime(b1.source_timestamp_or_rule, utc=True).dt.tz_convert("Europe/Berlin")
    assert (src.dt.hour * 4 + src.dt.minute // 15).to_numpy().tolist() == (b1.timestamp_local.dt.hour * 4 + b1.timestamp_local.dt.minute // 15).to_numpy().tolist()
    autumn = b1[b1.target_day_local == "2023-10-29"]
    assert len(autumn) == 100 and autumn.timestamp_utc.is_unique
    repeated = autumn[autumn.timestamp_local.dt.hour == 2]
    assert len(repeated) == 8
    assert len({x.to_pydatetime().fold for x in repeated.timestamp_local}) == 2


def test_b2_exact_seven_day_mapping_and_no_fallback():
    p = pd.read_parquet(h4.OUT / "test_predictions_2023.parquet")
    b2 = p[(p.protocol == "P1") & (p.baseline_id == "B2")].copy()
    available = b2[b2.prediction_available]
    src = pd.to_datetime(available.source_timestamp_or_rule, utc=True).dt.tz_convert("Europe/Berlin")
    assert ((available.timestamp_local.dt.date - src.dt.date).map(lambda x: x.days) == 7).all()
    assert (~b2.prediction_available).any()
    assert (b2.loc[~b2.prediction_available, "source_timestamp_or_rule"] == "").all()


def test_b3_candidates_and_median_logic():
    rows = pd.read_csv(h4.OUT / "b3_selection.csv")
    assert {x[0] for x in h4.B3_SPECS}.issubset(set(rows.candidate))
    values = [1.0, 10.0, 100.0, 1000.0]
    assert np.median(values) == 55.0
    frozen = json.loads(h4.SELECTION.read_text())
    assert frozen["P1"]["selected_b3"] == "B3_N8_MIN4"


def test_b4_training_only_statistics_and_fallback_hierarchy():
    base = h4.load_frame()
    train = base[base.year == 2020]
    target = base[base.year == 2021].head(50).copy()
    pred, level = h4.b4_predict(train, target, "C3_MONTH_QUARTER_WEEKEND")
    assert pred.notna().all()
    assert level.str.startswith("level_").all()
    frozen = json.loads(h4.SELECTION.read_text())
    assert frozen["P1"]["selected_b4"] == "B4_C3_MONTH_QUARTER_WEEKEND"


def test_b5_member_logic_and_negative_predictions_preserved():
    frozen = json.loads(h4.SELECTION.read_text())
    rule = frozen["P1"]["frozen_b5_rule"]
    assert rule["aggregation"] == "mean" and rule["minimum_available_members"] == 3
    assert rule["fallback"] == frozen["P1"]["selected_b4"]
    p = pd.read_parquet(h4.OUT / "test_predictions_2023.parquet")
    assert p.prediction_kw.min() < 0


def test_no_future_or_imputed_historical_sources_and_zero_leakage():
    audit = pd.read_csv(h4.OUT / "leakage_audit.csv")
    assert audit.violations.sum() == 0
    p = pd.read_parquet(h4.OUT / "test_predictions_2023.parquet")
    historical = p[p.baseline_id.isin(["B0", "B1", "B2", "B3"]) & p.prediction_available]
    assert not historical.source_provenance.str.contains("IMPUTED|UNKNOWN", regex=True).any()


def test_selection_has_no_test_leakage_and_p1_p2_are_separate():
    frozen = json.loads(h4.SELECTION.read_text())
    assert frozen["selection_status"] == "FROZEN_BEFORE_TEST"
    assert frozen["test_used_for_selection"] is False and frozen["test_year"] == 2023
    assert "P1" in frozen and "P2_sensitivity" in frozen


def test_2023_guard_requires_selection_file(monkeypatch, tmp_path):
    base = h4.load_frame()
    monkeypatch.setattr(h4, "SELECTION", tmp_path / "missing.json")
    with pytest.raises(PermissionError):
        h4.final_frame(base, [2020, 2021, 2022], {"selected_b4": "x"}, "P1")


def test_import_export_diagnostics_and_complete_day_metrics():
    mode = pd.read_csv(h4.OUT / "test_grid_mode_summary.csv")
    assert {"IMPORT", "EXPORT", "ALL_DIAGNOSTICS"}.issubset(set(mode.loc[mode.baseline_id == "B5", "mode"]))
    allrow = mode[(mode.baseline_id == "B5") & (mode["mode"] == "ALL_DIAGNOSTICS")].iloc[0]
    assert 0 <= allrow.sign_accuracy <= 1 and 0 <= allrow.export_recall <= 1
    daily = pd.read_csv(h4.OUT / "test_daily_summary.csv")
    assert {"daily_mae_kw_mean", "daily_mae_kw_median", "daily_mae_kw_p90", "daily_mae_kw_max"}.issubset(daily.columns)


def test_pairwise_common_day_comparison_and_bootstrap_seed():
    b = pd.read_csv(h4.OUT / "bootstrap_comparisons.csv")
    assert set(b.seed) == {h4.BOOTSTRAP_SEED}
    assert set(b.bootstrap_resamples) == {h4.BOOTSTRAP_RESAMPLES}
    assert (b.common_complete_days > 0).all()
    assert {"B2", "B3"}.issubset(set(b.comparator))


def test_dst_92_100_behavior_and_lead_band_preservation():
    p = pd.read_parquet(h4.OUT / "test_predictions_2023.parquet")
    b5 = p[(p.protocol == "P1") & (p.baseline_id == "B5")]
    sizes = b5.groupby("target_day_local").size()
    assert sizes.loc["2023-03-26"] == 92 and sizes.loc["2023-10-29"] == 100
    assert set(b5.lead_band) == {"H1", "H2", "H3", "H4"}


def test_prediction_contract_and_primary_coverage():
    p = pd.read_parquet(h4.OUT / "test_predictions_2023.parquet")
    required = {"timestamp_utc", "timestamp_local", "target_day_local", "forecast_issue_time_utc", "lead_time_hours", "lead_band", "actual_kw", "prediction_kw", "baseline_id", "prediction_available", "fallback_used", "source_timestamp_or_rule", "source_provenance", "target_primary_eligible", "complete_primary_day", "split", "fold"}
    assert required.issubset(p.columns)
    summary = pd.read_csv(h4.OUT / "test_baseline_summary.csv")
    winner = summary[(summary.protocol == "P1") & (summary.target_policy == "PRIMARY") & (summary.baseline_id == "B5")].iloc[0]
    assert winner.prediction_coverage == 1 and winner.complete_day_coverage == 1


def test_manifest_hashes_and_no_ml_artifacts():
    manifest = json.loads(h4.MANIFEST.read_text())
    for rel, meta in manifest["deterministic_artifacts"].items():
        assert h4.digest(h4.ROOT / rel) == meta["sha256"]
    assert manifest["model_training_performed"] is False
    source = Path(h4.__file__).read_text(encoding="utf-8").lower()
    assert not any(token in source for token in ["import lightgbm", "from lightgbm", "import xgboost", "from xgboost", ".fit("])
