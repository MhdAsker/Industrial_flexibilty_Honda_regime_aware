from __future__ import annotations

import numpy as np
import pandas as pd

from src.honda.h6_foundation_data import INDEX, ROOT, TZ
from src.honda.scenarios.adapters import expected_intervals


def point_forecast_adapter(predictions: pd.DataFrame, *, phase: str, require_complete_days: bool = True) -> pd.DataFrame:
    """Adapt a selected Kumo point forecast to the existing Honda scenario contract.

    This function deliberately creates no quantiles or trajectories. Any uncertainty
    model must be independently calibrated on permitted development residuals.
    """

    if phase not in {"development", "test"}:
        raise ValueError("phase must be development or test")
    required = {
        "timestamp_utc",
        "target_day_local",
        "forecast_issue_time_utc",
        "prediction_kw",
        "prediction_available",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Kumo predictions are missing {sorted(missing)}")
    if predictions.timestamp_utc.duplicated().any():
        raise ValueError("Kumo point forecasts contain duplicate timestamps")
    if not predictions.prediction_available.all() or not np.isfinite(predictions.prediction_kw).all():
        raise ValueError("Incomplete Kumo forecasts cannot enter downstream scheduling")

    reference_path = (
        ROOT / "results/ml/development_predictions.parquet"
        if phase == "development"
        else ROOT / "results/ml/test_predictions_2023.parquet"
    )
    reference = pd.read_parquet(reference_path, columns=["timestamp_utc", "actual_kw"])
    protocol = pd.read_parquet(
        INDEX,
        columns=[
            "timestamp_utc",
            "timestamp_local",
            "forecast_issue_time_utc",
            "forecast_issue_time_local",
            "lead_time_minutes",
            "holiday_context",
        ],
    )
    frame = predictions.merge(reference, on="timestamp_utc", validate="one_to_one")
    frame = frame.merge(
        protocol,
        on=["timestamp_utc", "forecast_issue_time_utc"],
        validate="one_to_one",
    )
    completeness = {}
    for day, group in frame.groupby("target_day_local", sort=True):
        completeness[str(day)] = bool(
            len(group) == expected_intervals(day)
            and group.timestamp_utc.nunique() == len(group)
            and group.prediction_available.all()
        )
    frame["complete_day"] = frame.target_day_local.astype(str).map(completeness)
    if require_complete_days and not frame.complete_day.all():
        incomplete = sorted(frame.loc[~frame.complete_day, "target_day_local"].astype(str).unique())
        raise ValueError(f"Incomplete Kumo days cannot enter downstream scheduling: {incomplete[:5]}")

    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame["timestamp_local"] = pd.to_datetime(frame.timestamp_local, utc=True).dt.tz_convert(TZ)
    frame["forecast_issue_time_local"] = pd.to_datetime(
        frame.forecast_issue_time_local, utc=True
    ).dt.tz_convert(TZ)
    frame["dataset_id"] = "HONDA"
    frame["model_id"] = "KUMO_FORECAST"
    frame["split"] = "DEVELOPMENT_2021_2022" if phase == "development" else "TEST_2023_PREVIOUSLY_EXPOSED"
    frame["point_forecast_kw"] = frame.prediction_kw.astype(float)
    frame["holiday"] = frame.holiday_context.str.contains("PUBLIC_HOLIDAY", na=False)
    frame["quality_policy"] = "OBSERVED_ORIGINAL"
    output = [
        "dataset_id",
        "model_id",
        "split",
        "target_day_local",
        "timestamp_utc",
        "timestamp_local",
        "forecast_issue_time_utc",
        "forecast_issue_time_local",
        "lead_time_minutes",
        "actual_kw",
        "point_forecast_kw",
        "complete_day",
        "holiday",
        "quality_policy",
    ]
    return frame[output].sort_values(["target_day_local", "timestamp_utc"]).reset_index(drop=True)
