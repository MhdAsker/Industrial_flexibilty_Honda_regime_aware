from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.honda.scenarios.adapters import load_spark_development, load_spark_test, spark_root
from src.honda.scenarios.core import ScenarioConfig, complete_donor_days, generate_scenarios


ROOT = Path(__file__).resolve().parents[3]
SPARK = spark_root()


@dataclass(frozen=True)
class DecisionDay:
    dataset_id: str
    target_day_local: str
    timestamp_utc: tuple[str, ...]
    forecast_issue_time_utc: str
    decision_time_utc: str
    point_forecast_kw: tuple[float, ...]
    s1_scenarios_kw: tuple[tuple[float, ...], ...]
    s2_scenarios_kw: tuple[tuple[float, ...], ...]
    scenario_weights: tuple[float, ...]
    price_eur_per_mwh: tuple[float, ...]
    scenario_config_sha256: str
    price_source_sha256: str


@dataclass(frozen=True)
class SettlementDay:
    dataset_id: str
    target_day_local: str
    timestamp_utc: tuple[str, ...]
    actual_kw: tuple[float, ...]
    is_weekend: bool
    is_public_holiday: bool
    month: int


def _load_s2(path: Path) -> dict[str, dict]:
    columns = [
        "target_day_local", "target_timestamp_utc", "scenario_id", "scenario_weight",
        "forecast_value_kw", "scenario_value_kw", "configuration_sha256",
    ]
    raw = pd.read_parquet(path, columns=columns)
    result: dict[str, dict] = {}
    for day, group in raw.groupby("target_day_local", sort=True):
        group = group.copy()
        group["target_timestamp_utc"] = pd.to_datetime(group.target_timestamp_utc, utc=True)
        pivot = group.pivot(index="scenario_id", columns="target_timestamp_utc", values="scenario_value_kw").sort_index(axis=1)
        point = group.drop_duplicates("target_timestamp_utc").set_index("target_timestamp_utc").sort_index().forecast_value_kw
        weights = group.groupby("scenario_id").scenario_weight.first().reindex(pivot.index)
        if pivot.isna().any().any() or not np.isclose(weights.sum(), 1):
            raise ValueError(f"Invalid frozen S2 matrix for {day}")
        result[str(day)] = {
            "timestamps": tuple(pivot.columns.astype(str)), "point": tuple(point.astype(float)),
            "matrix": tuple(map(tuple, pivot.to_numpy(float))), "weights": tuple(weights.astype(float)),
            "config_hash": str(group.configuration_sha256.iloc[0]),
        }
    return result


def _load_prices() -> tuple[pd.Series, str]:
    path = SPARK / "data/silver/exogenous/canonical/electricity_prices_15min_de_2021_2024.parquet"
    prices = pd.read_parquet(path)
    prices.utc_datetime = pd.to_datetime(prices.utc_datetime, utc=True)
    if prices.utc_datetime.duplicated().any() or prices.price_eur_per_mwh.isna().any():
        raise ValueError("Invalid canonical SPARK price series")
    import hashlib
    return prices.set_index("utc_datetime").price_eur_per_mwh, hashlib.sha256(path.read_bytes()).hexdigest()


def load_spark_stage3_days(*, development: bool = False, limit: int | None = None) -> tuple[dict[str, DecisionDay], dict[str, SettlementDay]]:
    target = load_spark_development() if development else load_spark_test()
    scenario_path = ROOT / "results/scenarios" / ("spark_development_scenarios.parquet" if development else "spark_2024_diagnostic_scenarios.parquet")
    s2 = _load_s2(scenario_path)
    prices, price_hash = _load_prices()
    donors, _ = complete_donor_days(load_spark_development())
    config = ScenarioConfig(method="S1", scenario_count=50, seed=20261029)
    decisions: dict[str, DecisionDay] = {}; settlements: dict[str, SettlementDay] = {}
    dates = sorted(set(s2) & set(target.target_day_local.astype(str)))
    if limit is not None: dates = dates[:limit]
    for day in dates:
        group = target[target.target_day_local.eq(day)].sort_values("timestamp_utc").reset_index(drop=True)
        if not bool(group.complete_day.all()) or len(group) not in {92, 96, 100}:
            continue
        frozen = s2[day]
        timestamps = tuple(group.timestamp_utc.astype(str))
        if timestamps != frozen["timestamps"]:
            raise ValueError(f"Frozen S2 timestamp mismatch on {day}")
        price_index = pd.DatetimeIndex(group.timestamp_utc)
        price = prices.reindex(price_index)
        if price.isna().any():
            continue
        # S1 generation uses only point/calendar columns from the target day;
        # actuals are deliberately erased before calling the frozen H7 sampler.
        safe_target = group.copy(); safe_target["actual_kw"] = np.nan
        s1, metadata = generate_scenarios(safe_target, donors, config)
        if s1 is None or not metadata["supported"]:
            continue
        s1 = np.maximum(s1, 0.0)  # Frozen SPARK nonnegative physical policy.
        issue = pd.Timestamp(group.forecast_issue_time_utc.iloc[0])
        decisions[day] = DecisionDay(
            dataset_id="SPARK", target_day_local=day, timestamp_utc=timestamps,
            forecast_issue_time_utc=str(issue), decision_time_utc=str(issue + pd.Timedelta(hours=2)),
            point_forecast_kw=tuple(group.point_forecast_kw.astype(float)),
            s1_scenarios_kw=tuple(map(tuple, s1)), s2_scenarios_kw=frozen["matrix"],
            scenario_weights=frozen["weights"], price_eur_per_mwh=tuple(price.astype(float)),
            scenario_config_sha256=frozen["config_hash"], price_source_sha256=price_hash,
        )
        settlements[day] = SettlementDay(
            dataset_id="SPARK", target_day_local=day, timestamp_utc=timestamps,
            actual_kw=tuple(group.actual_kw.astype(float)), is_weekend=bool(group.day_type.eq("weekend").all()),
            is_public_holiday=bool(group.holiday.any()), month=int(group.month.iloc[0]),
        )
    return decisions, settlements
