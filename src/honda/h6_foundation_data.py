from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "data/silver/honda/honda_forecasting_index_15min.parquet"
SILVER = ROOT / "data/silver/honda/honda_energy_15min.parquet"
HOLIDAYS = ROOT / "results/protocol_h3a/holiday_calendar.csv"
OBSERVED = "OBSERVED_ORIGINAL"
TZ = "Europe/Berlin"
FOUNDATION_CALENDAR_KEYS = ["daily_sin", "daily_cos", "annual_sin", "annual_cos", "weekend", "holiday"]


@dataclass
class ForecastOrigin:
    target_day_local: str
    issue_utc: pd.Timestamp
    target_timestamps: pd.DatetimeIndex
    target_horizon: int
    context: np.ndarray
    chp_context: np.ndarray
    calendar_past: dict[str, np.ndarray]
    calendar_future: dict[str, np.ndarray]
    observed_fraction: float
    context_valid: bool


@lru_cache(maxsize=1)
def _holiday_sets() -> tuple[set, set, set]:
    frame = pd.read_csv(HOLIDAYS)
    date_col = next(c for c in frame.columns if "date" in c.lower())
    holidays = set(pd.to_datetime(frame[date_col]).dt.date)
    before = {d - pd.Timedelta(days=1) for d in holidays}
    after = {d + pd.Timedelta(days=1) for d in holidays}
    return holidays, before, after


def calendar_features(timestamps: pd.DatetimeIndex, issue: pd.Timestamp) -> dict[str, np.ndarray]:
    local = timestamps.tz_convert(TZ)
    quarter = local.hour * 4 + local.minute // 15
    day = local.dayofyear
    holidays, before, after = _holiday_sets()
    dates = np.array(local.date)
    return {
        "daily_sin": np.sin(2 * np.pi * quarter / 96).astype(np.float32),
        "daily_cos": np.cos(2 * np.pi * quarter / 96).astype(np.float32),
        "annual_sin": np.sin(2 * np.pi * day / 365.25).astype(np.float32),
        "annual_cos": np.cos(2 * np.pi * day / 365.25).astype(np.float32),
        "weekday": local.weekday.to_numpy(dtype=np.float32) / 6.0,
        "weekend": np.asarray(local.weekday >= 5, dtype=np.float32),
        "month": local.month.to_numpy(dtype=np.float32) / 12.0,
        "dst": np.array([float(x.dst().total_seconds() != 0) for x in local], dtype=np.float32),
        "holiday": np.array([float(x in holidays) for x in dates], dtype=np.float32),
        "day_before_holiday": np.array([float(x in before) for x in dates], dtype=np.float32),
        "day_after_holiday": np.array([float(x in after) for x in dates], dtype=np.float32),
        "lead_days": ((timestamps.asi8 - issue.value) / 86_400_000_000_000).astype(np.float32),
    }


def _series() -> tuple[pd.Series, pd.Series]:
    frame = pd.read_parquet(
        SILVER,
        columns=[
            "timestamp_utc",
            "grid_exchange_kw_signed",
            "grid_exchange_provenance",
            "chp_generation_kw",
            "chp_provenance",
        ],
    ).set_index("timestamp_utc")
    grid = frame.grid_exchange_kw_signed.where(frame.grid_exchange_provenance.eq(OBSERVED))
    chp = frame.chp_generation_kw.where(frame.chp_provenance.eq(OBSERVED))
    return grid, chp


def load_origins(
    years: list[int], context_days: int, max_origins: int | None = None
) -> list[ForecastOrigin]:
    index = pd.read_parquet(INDEX)
    index = index[pd.to_datetime(index.target_day_local).dt.year.isin(years)].copy()
    grid, chp = _series()
    origins: list[ForecastOrigin] = []
    groups = list(index.groupby("target_day_local", sort=True))
    if max_origins is not None:
        groups = groups[:max_origins]
    for day, target in groups:
        target = target.sort_values("timestamp_utc")
        issue = pd.Timestamp(target.forecast_issue_time_utc.iloc[0])
        context_index = pd.date_range(
            issue - pd.Timedelta(days=context_days),
            issue - pd.Timedelta(minutes=15),
            freq="15min",
            tz="UTC",
        )
        context = grid.reindex(context_index).to_numpy(dtype=np.float32)
        chp_context = chp.reindex(context_index).to_numpy(dtype=np.float32)
        target_index = pd.DatetimeIndex(target.timestamp_utc)
        all_times = context_index.append(target_index)
        all_calendar = calendar_features(all_times, issue)
        n_context = len(context_index)
        past = {k: v[:n_context] for k, v in all_calendar.items()}
        future = {k: v[n_context:] for k, v in all_calendar.items()}
        observed = int(np.isfinite(context).sum())
        origins.append(
            ForecastOrigin(
                target_day_local=str(day),
                issue_utc=issue,
                target_timestamps=target_index,
                target_horizon=len(target_index),
                context=context,
                chp_context=chp_context,
                calendar_past=past,
                calendar_future=future,
                observed_fraction=observed / len(context),
                context_valid=observed > 0,
            )
        )
    return origins


def result_frame(
    origins: list[ForecastOrigin], predictions: list[np.ndarray], model: str, mode: str, context_days: int
) -> pd.DataFrame:
    rows = []
    for origin, pred in zip(origins, predictions):
        if len(pred) != origin.target_horizon:
            raise ValueError(f"Prediction length mismatch for {origin.target_day_local}")
        for timestamp, value in zip(origin.target_timestamps, pred):
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "target_day_local": origin.target_day_local,
                    "forecast_issue_time_utc": origin.issue_utc,
                    "prediction_kw": float(value),
                    "prediction_available": bool(np.isfinite(value)),
                    "context_valid": origin.context_valid,
                    "context_observed_fraction": origin.observed_fraction,
                    "model": model,
                    "mode": mode,
                    "context_days": context_days,
                }
            )
    return pd.DataFrame(rows)


def chronos_inputs(origins: list[ForecastOrigin], mode: str):
    if mode == "U":
        return [x.context for x in origins]
    result = []
    for x in origins:
        past = {k: x.calendar_past[k] for k in FOUNDATION_CALENDAR_KEYS}
        future = {k: x.calendar_future[k] for k in FOUNDATION_CALENDAR_KEYS}
        if mode == "C1":
            past["chp_history"] = x.chp_context
        result.append({"target": x.context, "past_covariates": past, "future_covariates": future})
    return result


def timesfm_inputs(origins: list[ForecastOrigin], mode: str):
    contexts = [x.context for x in origins]
    if mode == "U":
        return contexts, None, None
    keys = FOUNDATION_CALENDAR_KEYS
    past_future = [
        np.stack([np.concatenate([x.calendar_past[k], x.calendar_future[k]]) for k in keys]).astype(np.float32)
        for x in origins
    ]
    past_only = None
    if mode == "C1":
        past_only = [x.chp_context[None, :].astype(np.float32) for x in origins]
    return contexts, past_only, past_future


def write_runtime(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
