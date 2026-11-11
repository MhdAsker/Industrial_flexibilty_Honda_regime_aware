from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from honda.scenarios.adapters import load_honda_test, load_spark_test

from .schemas import SiteConfig


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SiteData:
    dataset_id: str
    target_day_local: str
    timestamps_utc: tuple[str, ...]
    actual_kw: np.ndarray
    point_forecast_kw: np.ndarray
    independent_scenarios_kw: np.ndarray
    coherent_scenarios_kw: np.ndarray
    scenario_weights: np.ndarray
    provenance: dict[str, str]

    def validate(self, site: SiteConfig) -> None:
        horizon = len(self.timestamps_utc)
        stamps = pd.DatetimeIndex(pd.to_datetime(self.timestamps_utc, utc=True))
        if stamps.has_duplicates:
            raise ValueError("Duplicate UTC timestamps")
        if horizon not in {92, 96, 100}:
            raise ValueError(f"Unsupported physical day length: {horizon}")
        if any(len(x) != horizon for x in (self.actual_kw, self.point_forecast_kw)):
            raise ValueError("Observation or forecast horizon mismatch")
        for name, matrix in (("independent", self.independent_scenarios_kw), ("coherent", self.coherent_scenarios_kw)):
            if matrix.ndim != 2 or matrix.shape[1] != horizon:
                raise ValueError(f"{name} scenario-horizon mismatch")
            if matrix.shape[0] != len(self.scenario_weights):
                raise ValueError(f"{name} scenario-weight mismatch")
        if not np.isclose(self.scenario_weights.sum(), 1) or np.any(self.scenario_weights < 0):
            raise ValueError("Scenario weights must be nonnegative and sum to one")
        arrays = (self.actual_kw, self.point_forecast_kw, self.independent_scenarios_kw, self.coherent_scenarios_kw)
        if not all(np.isfinite(x).all() for x in arrays):
            raise ValueError("Site data contain nonfinite values")
        if site.target_semantics == "demand_nonnegative" and min(np.min(x) for x in arrays) < -1e-9:
            raise ValueError("Nonnegative demand adapter returned negative values")


class SiteAdapter(Protocol):
    def load(self, site: SiteConfig) -> SiteData: ...


def _expected_utc(day: str, timezone: str) -> pd.DatetimeIndex:
    date = pd.Timestamp(day).date()
    start = pd.Timestamp(date, tz=timezone)
    end = pd.Timestamp(date + pd.Timedelta(days=1), tz=timezone)
    return pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="15min", inclusive="left")


class SyntheticSiteAdapter:
    def load(self, site: SiteConfig) -> SiteData:
        stamps = _expected_utc(site.target_day_local, site.timezone)
        n = len(stamps); t = np.arange(n); rng = np.random.default_rng(site.seed)
        base = 55 + 12 * np.sin(2 * np.pi * (t - n * .22) / n) + 6 * np.sin(4 * np.pi * t / n)
        actual = base + 2.5 * np.cos(6 * np.pi * t / n)
        point = base
        scenario_count = 50
        coherent = np.vstack([
            point + rng.normal(0, 2.0) + rng.normal(0, 3.5) * np.sin(2 * np.pi * t / n + rng.uniform(-.7, .7))
            for _ in range(scenario_count)
        ])
        independent = point + rng.normal(0, np.std(coherent - point), size=(scenario_count, n))
        if site.target_semantics == "demand_nonnegative":
            actual = np.maximum(actual, 0); point = np.maximum(point, 0)
            coherent = np.maximum(coherent, 0); independent = np.maximum(independent, 0)
        data = SiteData(site.dataset_id, site.target_day_local, tuple(stamps.astype(str)), actual, point, independent, coherent, np.full(scenario_count, 1 / scenario_count), {"source": "DETERMINISTIC_SYNTHETIC_FIXTURE", "seed": str(site.seed)})
        data.validate(site)
        return data


def _frozen_site_data(site: SiteConfig, target: pd.DataFrame, scenario_path: Path) -> SiteData:
    target = target[target.target_day_local.eq(site.target_day_local)].sort_values("timestamp_utc").copy()
    if target.empty:
        raise ValueError(f"No target data for {site.dataset_id} on {site.target_day_local}")
    raw = pd.read_parquet(scenario_path)
    raw = raw[raw.target_day_local.eq(site.target_day_local)].copy()
    if raw.empty:
        raise ValueError(f"No scenarios for {site.dataset_id} on {site.target_day_local}")
    raw["target_timestamp_utc"] = pd.to_datetime(raw.target_timestamp_utc, utc=True)
    coherent = raw.pivot(index="scenario_id", columns="target_timestamp_utc", values="scenario_value_kw").sort_index(axis=1)
    point = raw.drop_duplicates("target_timestamp_utc").set_index("target_timestamp_utc").sort_index().forecast_value_kw
    weights = raw.groupby("scenario_id").scenario_weight.first().reindex(coherent.index).to_numpy(float)
    timestamps = pd.DatetimeIndex(coherent.columns)
    actual = target.set_index(pd.to_datetime(target.timestamp_utc, utc=True)).actual_kw.reindex(timestamps)
    if actual.isna().any() or not np.allclose(point.to_numpy(), target.set_index(pd.to_datetime(target.timestamp_utc, utc=True)).point_forecast_kw.reindex(timestamps).to_numpy()):
        raise ValueError("Frozen target/scenario alignment failed")
    residual_pool = (coherent.to_numpy() - point.to_numpy()[None, :]).reshape(-1)
    rng = np.random.default_rng(site.seed)
    independent = point.to_numpy()[None, :] + rng.choice(residual_pool, size=coherent.shape, replace=True)
    if site.target_semantics == "demand_nonnegative":
        independent = np.maximum(independent, 0)
    data = SiteData(site.dataset_id, site.target_day_local, tuple(timestamps.astype(str)), actual.to_numpy(float), point.to_numpy(float), independent, coherent.to_numpy(float), weights, {"source": str(scenario_path.relative_to(ROOT)), "configuration_sha256": str(raw.configuration_sha256.iloc[0])})
    data.validate(site)
    return data


class HondaSiteAdapter:
    def load(self, site: SiteConfig) -> SiteData:
        return _frozen_site_data(site, load_honda_test(), ROOT / "results/scenarios/honda_2023_diagnostic_scenarios.parquet")


class SparkSiteAdapter:
    def load(self, site: SiteConfig) -> SiteData:
        return _frozen_site_data(site, load_spark_test(), ROOT / "results/scenarios/spark_2024_diagnostic_scenarios.parquet")


def adapter_for(name: str) -> SiteAdapter:
    adapters: dict[str, SiteAdapter] = {"synthetic": SyntheticSiteAdapter(), "honda": HondaSiteAdapter(), "spark": SparkSiteAdapter()}
    return adapters[name]
