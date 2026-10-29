from __future__ import annotations

import json
import subprocess
from datetime import date

import pandas as pd

from src.honda import h3a_protocol_analysis as h3a


def test_h2_checkpoint_and_finalized_artifacts_unchanged():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=h3a.ROOT, text=True).strip()
    assert subprocess.run(["git", "merge-base", "--is-ancestor", h3a.H2_COMMIT, head], cwd=h3a.ROOT).returncode == 0
    protected = [
        "config/honda_h2.json", "reports/honda_provenance_and_silver.md", "reports/honda_source_acquisition.md",
        "results/provenance/honda_h2_manifest.json", "src/honda/h2_provenance.py", "tests/honda/test_h2_provenance.py",
    ]
    assert subprocess.run(["git", "diff", "--quiet", h3a.H2_COMMIT, "--", *protected], cwd=h3a.ROOT).returncode == 0


def test_target_sign_preserved_and_exports_retained():
    silver = pd.read_parquet(h3a.SILVER_PATH, columns=["grid_exchange_kw_signed"])
    assert h3a.CONFIG["target_semantics"] == "signed net grid exchange; positive import, negative export"
    assert silver.grid_exchange_kw_signed.min() < 0
    summary = pd.read_csv(h3a.OUT / "grid_target_summary.csv")
    assert (summary.negative_export_fraction > 0).all()


def test_primary_and_sensitivity_provenance_rules():
    assert h3a.CONFIG["primary_target_provenance"] == [h3a.OBSERVED]
    assert h3a.CORRECTED not in h3a.PRIMARY
    assert "IMPUTED_OR_FILLED" not in h3a.PRIMARY
    assert "UNKNOWN_PROVENANCE" not in h3a.PRIMARY
    assert h3a.SENSITIVITY == {h3a.OBSERVED, h3a.CORRECTED}


def test_physical_days_have_92_96_100_unique_utc_intervals():
    spring = h3a.physical_day_grid(date(2023, 3, 26))
    ordinary = h3a.physical_day_grid(date(2023, 3, 27))
    autumn = h3a.physical_day_grid(date(2023, 10, 29))
    assert [len(spring), len(ordinary), len(autumn)] == [92, 96, 100]
    assert spring.target_timestamp_utc.is_unique and autumn.target_timestamp_utc.is_unique
    assert autumn.fold.sum() == 4


def test_issuance_local_utc_conversion():
    winter = h3a.issue_time(date(2023, 1, 2))
    summer = h3a.issue_time(date(2023, 7, 2))
    assert winter.hour == summer.hour == 13
    assert winter.tz_convert("UTC").hour == 12
    assert summer.tz_convert("UTC").hour == 11


def test_interval_completion_cutoff_excludes_unfinished_interval():
    issue = pd.Timestamp("2023-07-01T11:00:00Z")
    assert h3a.historical_interval_available(pd.Timestamp("2023-07-01T10:45:00Z"), issue)
    assert not h3a.historical_interval_available(pd.Timestamp("2023-07-01T11:00:00Z"), issue)


def test_future_observed_weather_pv_chp_rejected_and_history_accepted():
    policy = pd.read_csv(h3a.OUT / "feature_availability_matrix.csv")
    for feature in ["Ta ambient temperature", "Igm irradiance"]:
        assert policy[(policy.feature == feature) & (policy.time_relation == "future_observed")].iloc[0].h3a_classification == "FORBIDDEN"
    for feature in ["PV generation", "CHP generation/state"]:
        assert policy[(policy.feature == feature) & policy.time_relation.str.startswith("future")].iloc[0].h3a_classification == "FORBIDDEN"
        assert policy[(policy.feature == feature) & (policy.time_relation == "historical_completed")].iloc[0].h3a_classification == "SAFE_CANDIDATE"


def test_context_windows_end_at_or_before_issuance():
    context = pd.read_csv(h3a.OUT / "context_availability.csv", parse_dates=["forecast_issue_time_utc", "latest_context_interval_end_utc"])
    assert (context.latest_context_interval_end_utc <= context.forecast_issue_time_utc).all()
    assert set(context.expected_intervals) == {96, 192, 672, 1344, 2688, 5376}


def test_candidate_split_chronology_and_2023_test_is_untouched():
    protocols = h3a.CONFIG["candidate_protocols"]
    for name, rows in protocols.items():
        development_years = [y for row in rows if row["role"] != "test" for y in row["years"]]
        test_rows = [row for row in rows if row["role"] == "test"]
        assert 2023 not in development_years
        assert test_rows == [{"stage": "final", "role": "test", "years": [2023]}]
        for row in rows:
            if row["role"] in {"validation", "test"}:
                earlier = [r for r in rows if r["stage"] == row["stage"] and r["role"] in {"train", "development_refit"}]
                if earlier:
                    assert max(earlier[0]["years"]) < min(row["years"])


def test_no_model_training_or_frozen_h3_configuration():
    source = (h3a.ROOT / "src/honda/h3a_protocol_analysis.py").read_text(encoding="utf-8").lower()
    assert not any(token in source for token in ["lightgbm", "xgboost", "chronos", "timesfm", "timegpt", ".fit(", "predict("])
    assert not (h3a.ROOT / "config/honda_forecasting_protocol_frozen.json").exists()
    assert not [p for p in (h3a.ROOT / "data/gold").rglob("*") if p.is_file()]
    assert h3a.CONFIG["final_protocol_frozen"] is False
    assert h3a.CONFIG["model_training_allowed"] is False


def test_manifest_declares_no_training_and_no_freeze():
    manifest = json.loads((h3a.OUT / "honda_h3a_manifest.json").read_text())
    assert manifest["h2_commit"] == h3a.H2_COMMIT
    assert manifest["model_training_performed"] is False
    assert manifest["final_protocol_frozen"] is False
