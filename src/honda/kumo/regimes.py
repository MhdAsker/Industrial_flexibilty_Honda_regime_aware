from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CALENDAR_FAMILIES = {
    "NORMAL_WEEKDAY": "WORKDAY",
    "DAY_BEFORE_PUBLIC_HOLIDAY": "WORKDAY_ADJACENT_HOLIDAY",
    "DAY_AFTER_PUBLIC_HOLIDAY": "WORKDAY_ADJACENT_HOLIDAY",
    "WEEKEND": "NONWORKDAY",
    "PUBLIC_HOLIDAY": "NONWORKDAY",
}


@dataclass(frozen=True)
class PreissueThresholds:
    low_abs_level: float
    high_abs_level: float
    high_volatility: float

    def to_dict(self) -> dict[str, float]:
        return {
            "low_abs_level": self.low_abs_level,
            "high_abs_level": self.high_abs_level,
            "high_volatility": self.high_volatility,
        }


def preissue_features(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError("Pre-issuance regime features require a finite one-dimensional context")
    recent = x[-min(len(x), 96) :]
    return {
        "recent_abs_level": float(np.median(np.abs(recent))),
        "recent_volatility": float(np.median(np.abs(np.diff(recent)))),
        "recent_export_fraction": float(np.mean(recent < 0)),
        "recent_mean_kw": float(np.mean(recent)),
    }


def fit_preissue_thresholds(feature_rows: pd.DataFrame) -> PreissueThresholds:
    required = {"recent_abs_level", "recent_volatility"}
    if not required.issubset(feature_rows.columns) or feature_rows.empty:
        raise ValueError(f"Threshold fitting requires {sorted(required)}")
    return PreissueThresholds(
        low_abs_level=float(feature_rows.recent_abs_level.quantile(0.25)),
        high_abs_level=float(feature_rows.recent_abs_level.quantile(0.75)),
        high_volatility=float(feature_rows.recent_volatility.quantile(0.75)),
    )


def preissue_regime(features: dict[str, float], thresholds: PreissueThresholds) -> str:
    if features["recent_volatility"] >= thresholds.high_volatility:
        return "UNUSUAL_HIGH_VOLATILITY"
    if features["recent_abs_level"] <= thresholds.low_abs_level:
        return "LOW_LOAD"
    if features["recent_abs_level"] >= thresholds.high_abs_level:
        return "HIGH_LOAD"
    return "TYPICAL_LOAD"


def calendar_family(calendar_context: str) -> str:
    labels = {value.strip() for value in calendar_context.split(";") if value.strip()}
    if labels & {"PUBLIC_HOLIDAY", "WEEKEND"}:
        return "NONWORKDAY"
    if labels & {"DAY_BEFORE_PUBLIC_HOLIDAY", "DAY_AFTER_PUBLIC_HOLIDAY"}:
        return "WORKDAY_ADJACENT_HOLIDAY"
    if "NORMAL_WEEKDAY" in labels:
        return "WORKDAY"
    return CALENDAR_FAMILIES.get(calendar_context, "OTHER_KNOWN_CALENDAR")
