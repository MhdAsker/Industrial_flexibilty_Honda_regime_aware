from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import DecisionInput


def load_h7_day(
    path: Path,
    target_day_local: str,
    *,
    decision_time_utc: str,
    price_publication_time_utc: str | None,
    timing_classification: str,
) -> DecisionInput:
    columns = [
        "dataset_id", "forecast_issue_time_utc", "target_timestamp_utc", "target_day_local",
        "scenario_id", "scenario_weight", "forecast_value_kw", "scenario_value_kw", "configuration_sha256",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame.target_day_local.astype(str).eq(str(target_day_local))].copy()
    if frame.empty:
        raise ValueError("Requested H7 scenario day is unavailable")
    if frame.duplicated(["scenario_id", "target_timestamp_utc"]).any():
        raise ValueError("Duplicate scenario/timestamp rows")
    timestamps = sorted(frame.target_timestamp_utc.astype(str).unique())
    pivot = frame.pivot(index="scenario_id", columns="target_timestamp_utc", values="scenario_value_kw").reindex(columns=timestamps)
    if pivot.isna().any().any():
        raise ValueError("Incomplete H7 scenario matrix")
    point = frame.drop_duplicates("target_timestamp_utc").set_index("target_timestamp_utc").reindex(timestamps).forecast_value_kw
    weights = frame.groupby("scenario_id").scenario_weight.first().reindex(pivot.index)
    return DecisionInput(
        dataset_id=str(frame.dataset_id.iloc[0]), target_day_local=str(target_day_local), timestamp_utc=tuple(timestamps),
        interval_hours=tuple(np.full(len(timestamps), .25)), deterministic_forecast_kw=tuple(point.astype(float)),
        scenario_matrix_kw=tuple(map(tuple, pivot.to_numpy(float))), scenario_weights=tuple(weights.astype(float)),
        forecast_issue_time_utc=str(frame.forecast_issue_time_utc.iloc[0]), decision_time_utc=decision_time_utc,
        price_publication_time_utc=price_publication_time_utc, timing_classification=timing_classification,
        provenance_hashes={"scenario_configuration": str(frame.configuration_sha256.iloc[0])},
    )
