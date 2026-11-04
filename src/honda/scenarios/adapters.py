from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


HONDA_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPARK_ROOT = HONDA_ROOT.parent / "Industrial_energy_forecasting"
TZ = "Europe/Berlin"


def spark_root() -> Path:
    return Path(os.environ.get("SPARK_RESEARCH_ROOT", DEFAULT_SPARK_ROOT)).resolve()


def season(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def expected_intervals(local_date: str | pd.Timestamp) -> int:
    day = pd.Timestamp(local_date).date()
    start = pd.Timestamp(day, tz=TZ)
    end = pd.Timestamp(day + pd.Timedelta(days=1), tz=TZ)
    return int((end.tz_convert("UTC") - start.tz_convert("UTC")) / pd.Timedelta(minutes=15))


def _calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    local = pd.DatetimeIndex(frame.timestamp_local)
    frame["local_hour"] = local.hour
    frame["local_quarter_hour"] = local.minute // 15
    frame["month"] = local.month
    frame["season"] = [season(int(x)) for x in local.month]
    weekend = local.weekday >= 5
    frame["day_type"] = np.where(weekend, "weekend", "weekday")
    return frame


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame["timestamp_local"] = pd.to_datetime(frame.timestamp_local, utc=True).dt.tz_convert(TZ)
    frame["forecast_issue_time_utc"] = pd.to_datetime(frame.forecast_issue_time_utc, utc=True)
    frame["forecast_issue_time_local"] = pd.to_datetime(frame.forecast_issue_time_local, utc=True).dt.tz_convert(TZ)
    frame["target_day_local"] = pd.to_datetime(frame.target_day_local).dt.strftime("%Y-%m-%d")
    frame = _calendar_columns(frame)
    ordered = [
        "dataset_id", "model_id", "split", "target_day_local", "timestamp_utc", "timestamp_local",
        "forecast_issue_time_utc", "forecast_issue_time_local", "lead_time_minutes", "actual_kw",
        "point_forecast_kw", "complete_day", "local_hour", "local_quarter_hour", "month", "season",
        "day_type", "holiday", "quality_policy",
    ]
    return frame[ordered].sort_values(["target_day_local", "timestamp_utc"]).reset_index(drop=True)


def load_honda_development() -> pd.DataFrame:
    pred = pd.read_parquet(HONDA_ROOT / "results/ml/development_predictions.parquet")
    index = pd.read_parquet(
        HONDA_ROOT / "data/silver/honda/honda_forecasting_index_15min.parquet",
        columns=[
            "timestamp_utc", "forecast_issue_time_utc", "forecast_issue_time_local", "lead_time_minutes",
        ],
    )
    frame = pred.merge(index, on="timestamp_utc", validate="one_to_one")
    frame = frame.rename(
        columns={
            "actual_kw": "actual_kw",
            "LGBM_M1": "point_forecast_kw",
            "complete_primary_day": "complete_day",
            "fold": "split",
        }
    )
    frame["dataset_id"] = "HONDA"
    frame["model_id"] = "LGBM_M1_CHP"
    frame["holiday"] = frame.holiday_context.str.contains("PUBLIC_HOLIDAY", na=False)
    frame["quality_policy"] = "OBSERVED_ORIGINAL"
    return _finalize(frame)


def load_honda_test() -> pd.DataFrame:
    pred = pd.read_parquet(HONDA_ROOT / "results/ml/test_predictions_2023.parquet")
    index = pd.read_parquet(
        HONDA_ROOT / "data/silver/honda/honda_forecasting_index_15min.parquet",
        columns=[
            "timestamp_utc", "forecast_issue_time_utc", "forecast_issue_time_local", "lead_time_minutes",
        ],
    )
    frame = pred.merge(index, on="timestamp_utc", validate="one_to_one")
    frame = frame.rename(columns={"PRIMARY_ML": "point_forecast_kw", "complete_primary_day": "complete_day"})
    frame["split"] = "TEST_2023_PREVIOUSLY_EXPOSED"
    frame["dataset_id"] = "HONDA"
    frame["model_id"] = "LGBM_M1_CHP"
    frame["holiday"] = frame.holiday_context.str.contains("PUBLIC_HOLIDAY", na=False)
    frame["quality_policy"] = "OBSERVED_ORIGINAL"
    return _finalize(frame)


def _load_spark(split: str) -> pd.DataFrame:
    path = spark_root() / "results/lightgbm/lightgbm_predictions.parquet"
    raw = pd.read_parquet(path)
    raw = raw[(raw.model_name == "LGBM_DIRECT") & (raw.split == split)].copy()
    raw["target_time_utc"] = pd.to_datetime(raw.target_time_utc, utc=True)
    raw["target_time_local"] = pd.to_datetime(raw.target_time_local, utc=True).dt.tz_convert(TZ)
    raw["forecast_date_local"] = pd.to_datetime(raw.forecast_date_local).dt.strftime("%Y-%m-%d")
    completeness: dict[str, bool] = {}
    for day, group in raw.groupby("forecast_date_local"):
        expected = expected_intervals(day)
        completeness[day] = bool(
            len(group) == expected
            and group.target_quality_primary.all()
            and group.y_true_kw.notna().all()
            and group.y_pred_kw.notna().all()
        )
    frame = raw[raw.target_quality_primary].copy()
    frame = frame.rename(
        columns={
            "forecast_date_local": "target_day_local",
            "target_time_utc": "timestamp_utc",
            "target_time_local": "timestamp_local",
            "y_true_kw": "actual_kw",
            "y_pred_kw": "point_forecast_kw",
        }
    )
    frame["complete_day"] = frame.target_day_local.map(completeness).fillna(False)
    frame["dataset_id"] = "SPARK"
    frame["model_id"] = "LGBM_DIRECT"
    frame["holiday"] = frame.is_public_holiday.astype(bool)
    frame["quality_policy"] = "SPARK_PRIMARY_TARGET_QUALITY"
    frame["split"] = "VALIDATION_2023_OOS" if split == "validation" else "TEST_2024_PREVIOUSLY_EXPOSED"
    return _finalize(frame)


def load_spark_development() -> pd.DataFrame:
    return _load_spark("validation")


def load_spark_test() -> pd.DataFrame:
    return _load_spark("test")
