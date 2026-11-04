from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


METHODS = ("S0", "S1", "S2", "S3")


@dataclass(frozen=True)
class ScenarioConfig:
    method: str
    scenario_count: int
    seed: int = 20261029
    minimum_global_donors: int = 60
    minimum_conditional_donors: int = 20
    minimum_exact_length_donors: int = 5
    conditional_keys: tuple[str, ...] = ("season", "day_type")

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unsupported scenario method: {self.method}")
        if self.scenario_count < 1:
            raise ValueError("scenario_count must be positive")


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join([str(seed), *(str(x) for x in parts)]).encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)


def residual_vector(day: pd.DataFrame) -> np.ndarray:
    """Canonical H7 sign: actual minus point forecast."""
    return day.actual_kw.to_numpy(float) - day.point_forecast_kw.to_numpy(float)


def complete_donor_days(frame: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    donors: list[pd.DataFrame] = []
    audit: list[dict] = []
    for date, group in frame.groupby("target_day_local", sort=True):
        group = group.sort_values("timestamp_utc").reset_index(drop=True)
        reasons: list[str] = []
        if not bool(group.complete_day.all()):
            reasons.append("incomplete_physical_or_quality_day")
        if group.timestamp_utc.duplicated().any():
            reasons.append("duplicate_timestamp")
        if not np.isfinite(group[["actual_kw", "point_forecast_kw"]].to_numpy(float)).all():
            reasons.append("nonfinite_actual_or_forecast")
        if len(group) not in {92, 96, 100}:
            reasons.append("invalid_physical_interval_count")
        eligible = not reasons
        if eligible:
            donors.append(group)
        audit.append(
            {
                "dataset_id": str(group.dataset_id.iloc[0]),
                "target_day_local": str(date),
                "year": int(pd.Timestamp(date).year),
                "physical_intervals": len(group),
                "season": str(group.season.iloc[0]),
                "day_type": str(group.day_type.iloc[0]),
                "eligible_donor": eligible,
                "exclusion_reason": ";".join(reasons) if reasons else "",
            }
        )
    return donors, pd.DataFrame(audit)


def _eligible_donors(
    target: pd.DataFrame,
    donors: list[pd.DataFrame],
    config: ScenarioConfig,
) -> tuple[list[pd.DataFrame], str, bool]:
    target_date = pd.Timestamp(target.target_day_local.iloc[0])
    causal = [d for d in donors if pd.Timestamp(d.target_day_local.iloc[0]) < target_date]
    if len(causal) < config.minimum_global_donors:
        return [], "insufficient_global_history", False
    if config.method == "S1":
        return causal, "marginal_quarter_hour", False
    same_length = [d for d in causal if len(d) == len(target)]
    if len(same_length) < config.minimum_exact_length_donors:
        return [], "unsupported_physical_day_length", False
    if config.method == "S2":
        return same_length, "global_exact_length", False
    selected = same_length
    for key in config.conditional_keys:
        value = target[key].iloc[0]
        selected = [d for d in selected if d[key].iloc[0] == value]
    if len(selected) >= config.minimum_conditional_donors:
        return selected, "+".join(config.conditional_keys), False
    return same_length, "global_exact_length_fallback", True


def _marginal_residuals(
    target: pd.DataFrame,
    donors: list[pd.DataFrame],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pools: dict[tuple[int, int], list[float]] = {}
    global_pool: list[float] = []
    for donor in donors:
        values = residual_vector(donor)
        for row, value in zip(donor.itertuples(), values):
            key = (int(row.local_hour), int(row.local_quarter_hour))
            pools.setdefault(key, []).append(float(value))
            global_pool.append(float(value))
    result = np.empty((count, len(target)), dtype=np.float64)
    for position, row in enumerate(target.itertuples()):
        pool = np.asarray(pools.get((int(row.local_hour), int(row.local_quarter_hour)), global_pool), dtype=float)
        result[:, position] = rng.choice(pool, size=count, replace=True)
    return result


def generate_scenarios(
    target_day: pd.DataFrame,
    donor_days: list[pd.DataFrame],
    config: ScenarioConfig,
) -> tuple[np.ndarray | None, dict]:
    target = target_day.sort_values("timestamp_utc").reset_index(drop=True)
    point = target.point_forecast_kw.to_numpy(float)
    day = str(target.target_day_local.iloc[0])
    dataset = str(target.dataset_id.iloc[0])
    if config.method == "S0":
        matrix = point[None, :]
        return matrix, {
            "supported": True,
            "donor_count": 0,
            "conditioning": "point_forecast_only",
            "fallback_used": False,
        }
    selected, conditioning, fallback = _eligible_donors(target, donor_days, config)
    if not selected:
        return None, {
            "supported": False,
            "donor_count": 0,
            "conditioning": conditioning,
            "fallback_used": fallback,
        }
    rng = stable_rng(config.seed, dataset, day, config.method, config.scenario_count)
    if config.method == "S1":
        residuals = _marginal_residuals(target, selected, config.scenario_count, rng)
        donor_ids = "marginal"
    else:
        choices = rng.integers(0, len(selected), size=config.scenario_count)
        residuals = np.stack([residual_vector(selected[i]) for i in choices])
        donor_ids = [str(selected[i].target_day_local.iloc[0]) for i in choices]
    matrix = point[None, :] + residuals
    validate_scenario_matrix(matrix, len(target), allow_negative=True)
    return matrix, {
        "supported": True,
        "donor_count": len(selected),
        "conditioning": conditioning,
        "fallback_used": fallback,
        "donor_ids": donor_ids,
    }


def validate_scenario_matrix(
    matrix: np.ndarray,
    expected_intervals: int,
    *,
    allow_negative: bool,
) -> None:
    if matrix.ndim != 2 or matrix.shape[1] != expected_intervals:
        raise ValueError("Scenario matrix does not match the physical target horizon")
    if matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise ValueError("Scenario matrix contains no scenarios or nonfinite values")
    if not allow_negative and np.any(matrix < 0):
        raise ValueError("Negative scenario violates nonnegative target policy")


def scenario_long_frame(
    target_day: pd.DataFrame,
    matrix: np.ndarray,
    config: ScenarioConfig,
    config_hash: str,
    metadata: dict,
) -> pd.DataFrame:
    target = target_day.sort_values("timestamp_utc").reset_index(drop=True)
    count, horizon = matrix.shape
    point = np.tile(target.point_forecast_kw.to_numpy(float), count)
    values = matrix.reshape(-1)
    scenario_ids = np.repeat(np.arange(count, dtype=int), horizon)
    repeated = pd.concat([target] * count, ignore_index=True)
    return pd.DataFrame(
        {
            "dataset_id": repeated.dataset_id,
            "model_id": repeated.model_id,
            "forecast_issue_time_utc": repeated.forecast_issue_time_utc,
            "forecast_issue_time_local": repeated.forecast_issue_time_local,
            "target_timestamp_utc": repeated.timestamp_utc,
            "target_timestamp_local": repeated.timestamp_local,
            "target_day_local": repeated.target_day_local,
            "lead_time_minutes": repeated.lead_time_minutes,
            "scenario_id": scenario_ids,
            "scenario_weight": 1.0 / count,
            "forecast_value_kw": point,
            "scenario_value_kw": values,
            "scenario_residual_kw": values - point,
            "construction_method": config.method,
            "conditioning": metadata.get("conditioning", ""),
            "fallback_used": bool(metadata.get("fallback_used", False)),
            "configuration_sha256": config_hash,
        }
    )
