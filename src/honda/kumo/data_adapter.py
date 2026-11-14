from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.honda.h6_foundation_data import INDEX, OBSERVED, SILVER, TZ


FREQ = pd.Timedelta(minutes=15)
SUPPORTED_SEQUENCE_LENGTHS = (256, 512, 1024, 2048)


@dataclass(frozen=True)
class HondaKumoOrigin:
    target_day_local: str
    issue_utc: pd.Timestamp
    input_frame: pd.DataFrame
    forecast_path_timestamps: pd.DatetimeIndex
    target_timestamps: pd.DatetimeIndex
    target_positions: np.ndarray
    observed_fraction: float
    imputed_history_count: int

    @property
    def forecast_horizon(self) -> int:
        return len(self.forecast_path_timestamps)

    def select_target(self, full_prediction: np.ndarray) -> np.ndarray:
        values = np.asarray(full_prediction, dtype=float)
        if values.shape != (self.forecast_horizon,):
            raise ValueError(
                f"Expected the complete issue-to-delivery path of length {self.forecast_horizon}, "
                f"received {values.shape}"
            )
        return values[self.target_positions]


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Honda timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")


def causal_history_fill(values: pd.Series) -> tuple[pd.Series, int]:
    """Fill only a pre-issuance context, never labels or retrieval continuations.

    All values in ``values`` are already available at issuance. Linear interpolation
    and endpoint extension are therefore information-time safe, but imputed points
    remain explicitly counted and are never reclassified as observations.
    """

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    missing = int(numeric.isna().sum())
    if numeric.notna().sum() == 0:
        raise ValueError("Kumo context contains no OBSERVED_ORIGINAL history")
    filled = numeric.interpolate(method="linear", limit_direction="both")
    if filled.isna().any():
        raise ValueError("Historical-only interpolation did not resolve the Kumo context")
    return filled, missing


def build_origin(
    target: pd.DataFrame,
    observed_series: pd.Series,
    *,
    seq_len: int,
) -> HondaKumoOrigin:
    if seq_len not in SUPPORTED_SEQUENCE_LENGTHS:
        raise ValueError(f"seq_len must be one of {SUPPORTED_SEQUENCE_LENGTHS}")
    target = target.sort_values("timestamp_utc")
    if target.empty:
        raise ValueError("Target-day frame is empty")
    issue = _utc_timestamp(target.forecast_issue_time_utc.iloc[0])
    if target.forecast_issue_time_utc.map(_utc_timestamp).nunique() != 1:
        raise ValueError("A target day must have exactly one forecast issue time")

    context_index = pd.date_range(end=issue - FREQ, periods=seq_len, freq=FREQ, tz="UTC")
    raw_context = observed_series.reindex(context_index)
    filled_context, missing = causal_history_fill(raw_context)
    input_frame = pd.DataFrame(
        {
            "timestamp": context_index.tz_localize(None),
            "grid_exchange_kw_signed": filled_context.to_numpy(float),
        }
    )

    target_timestamps = pd.DatetimeIndex(pd.to_datetime(target.timestamp_utc, utc=True))
    if target_timestamps.has_duplicates or not target_timestamps.is_monotonic_increasing:
        raise ValueError("Target timestamps must be unique and increasing in physical UTC")
    if target_timestamps[0] < issue:
        raise ValueError("Target starts before the forecast issue time")

    full_path = pd.date_range(issue, target_timestamps[-1], freq=FREQ, tz="UTC")
    positions = full_path.get_indexer(target_timestamps)
    if (positions < 0).any():
        raise ValueError("Target timestamps do not align to the complete UTC forecast path")
    expected = {92, 96, 100}
    if len(target_timestamps) not in expected:
        raise ValueError(f"Honda delivery day must contain 92, 96, or 100 intervals, got {len(target_timestamps)}")
    if len(full_path) > 512:
        raise ValueError("Kumo's verified public API limits forecast_horizon to 512")

    return HondaKumoOrigin(
        target_day_local=str(target.target_day_local.iloc[0]),
        issue_utc=issue,
        input_frame=input_frame,
        forecast_path_timestamps=full_path,
        target_timestamps=target_timestamps,
        target_positions=positions.astype(int),
        observed_fraction=float(raw_context.notna().mean()),
        imputed_history_count=missing,
    )


def _observed_grid(path: Path = SILVER) -> pd.Series:
    frame = pd.read_parquet(
        path,
        columns=["timestamp_utc", "grid_exchange_kw_signed", "grid_exchange_provenance"],
    )
    timestamps = pd.to_datetime(frame.timestamp_utc, utc=True)
    values = frame.grid_exchange_kw_signed.where(frame.grid_exchange_provenance.eq(OBSERVED))
    return pd.Series(values.to_numpy(float), index=pd.DatetimeIndex(timestamps), name="grid_exchange_kw_signed")


def load_honda_origins(
    years: tuple[int, ...] | list[int],
    *,
    seq_len: int,
    max_origins: int | None = None,
    index_path: Path = INDEX,
    silver_path: Path = SILVER,
) -> list[HondaKumoOrigin]:
    protocol_index = pd.read_parquet(index_path)
    local_year = pd.to_datetime(protocol_index.target_day_local).dt.year
    protocol_index = protocol_index[local_year.isin(years)].copy()
    groups = list(protocol_index.groupby("target_day_local", sort=True))
    if max_origins is not None:
        groups = groups[:max_origins]
    observed = _observed_grid(silver_path)
    return [build_origin(day, observed, seq_len=seq_len) for _, day in groups]


def result_frame(
    origins: list[HondaKumoOrigin],
    predictions: list[np.ndarray],
    *,
    configuration_id: str,
    method: str,
) -> pd.DataFrame:
    if len(origins) != len(predictions):
        raise ValueError("Every origin must have exactly one prediction")
    rows: list[dict[str, object]] = []
    for origin, complete_path in zip(origins, predictions, strict=True):
        target_prediction = origin.select_target(complete_path)
        for timestamp, value in zip(origin.target_timestamps, target_prediction, strict=True):
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "target_day_local": origin.target_day_local,
                    "forecast_issue_time_utc": origin.issue_utc,
                    "prediction_kw": float(value),
                    "prediction_available": bool(np.isfinite(value)),
                    "model": "KUMO_FORECAST",
                    "method": method,
                    "configuration_id": configuration_id,
                    "context_observed_fraction": origin.observed_fraction,
                    "context_imputed_count": origin.imputed_history_count,
                    "full_forecast_horizon_steps": origin.forecast_horizon,
                }
            )
    return pd.DataFrame(rows)
