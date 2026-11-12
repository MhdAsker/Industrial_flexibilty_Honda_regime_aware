from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from honda.scenarios.adapters import load_honda_development
from honda.scenarios.metrics import day_scores

from .evaluation import write_json
from .scenario_experiment import (
    QUANTILE_COLUMNS,
    _descriptor,
    _expand_weighted,
    _marginal_samples,
    _recent_by_day,
)
from .scenarios import conditional_coherent_scenarios, reduce_scenarios

ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS = ROOT / "results/prescriptive_ml/development_predictions.parquet"
BASELINE = ROOT / "results/scenarios/honda_development_scenarios.parquet"
OUTPUT = ROOT / "results/prescriptive_ml"
METRICS = ("wis_kw", "energy_score_kw", "variogram_score", "coverage_90", "width_90_kw", "acf_rmse")


def _baseline_matrix(frame: pd.DataFrame, target_id: str) -> np.ndarray:
    day = frame[frame.target_day_local.astype(str).eq(target_id)]
    return np.stack([
        day[day.scenario_id.eq(identifier)].sort_values("target_timestamp_utc").scenario_value_kw.to_numpy(float)
        for identifier in sorted(day.scenario_id.unique())
    ])


def _generate(
    target: pd.DataFrame,
    donor_days: list[pd.DataFrame],
    recent: dict[str, dict[str, float]],
    bandwidth: float,
    raw_count: int,
):
    issue = pd.Timestamp(target.forecast_issue_time_utc.iloc[0])
    candidates = [
        day for day in donor_days
        if len(day) == len(target) and pd.Timestamp(day.timestamp_utc.iloc[-1]) + pd.Timedelta(minutes=15) <= issue
    ]
    if len(candidates) < 10:
        return None
    donor_features = np.stack([_descriptor(day, recent.get(str(day.target_day_local.iloc[0]), {})) for day in candidates])
    donor_residuals = np.stack([day.actual_kw.to_numpy(float) - day.point_forecast_kw.to_numpy(float) for day in candidates])
    availability = tuple((pd.Timestamp(day.timestamp_utc.iloc[-1]) + pd.Timedelta(minutes=15)).isoformat() for day in candidates)
    marginal = _marginal_samples(target[list(QUANTILE_COLUMNS)].to_numpy(float), raw_count)
    return conditional_coherent_scenarios(
        target.point_forecast_kw.to_numpy(float), _descriptor(target, recent.get(str(target.target_day_local.iloc[0]), {})),
        donor_features, donor_residuals, availability, issue.isoformat(), raw_count,
        marginal_samples_kw=marginal, target_id=str(target.target_day_local.iloc[0]),
        bandwidth=bandwidth, minimum_effective_donors=20,
    )


def run(
    output: str | Path = OUTPUT,
    raw_count: int = 200,
    bandwidths: tuple[float, ...] = (.5, 1.0, 2.0),
    reduction_counts: tuple[int, ...] = (25, 50),
) -> dict[str, object]:
    started = time.perf_counter(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for path in (PREDICTIONS, BASELINE):
        if not path.exists():
            raise ValueError(f"Missing development-selection input: {path}")
    development = load_honda_development()
    prediction = pd.read_parquet(PREDICTIONS, columns=["timestamp_utc", *QUANTILE_COLUMNS])
    development = development.merge(prediction, on="timestamp_utc", validate="one_to_one")
    targets = development[pd.to_datetime(development.target_day_local).dt.year.eq(2022)]
    donors = [group.sort_values("timestamp_utc").reset_index(drop=True) for _, group in development.groupby("target_day_local") if bool(group.complete_day.all())]
    baseline = pd.read_parquet(BASELINE); recent = _recent_by_day()
    bandwidth_rows: list[dict[str, object]] = []
    for bandwidth in bandwidths:
        for target_id, group in targets.groupby("target_day_local", sort=True):
            target = group.sort_values("timestamp_utc").reset_index(drop=True)
            generated = _generate(target, donors, recent, bandwidth, raw_count)
            if generated is None:
                continue
            scores = day_scores(target.actual_kw.to_numpy(float), target.point_forecast_kw.to_numpy(float), generated.scenarios)
            bandwidth_rows.append({"target_day_local": str(target_id), "bandwidth": bandwidth, **scores})
    bandwidth_daily = pd.DataFrame(bandwidth_rows)
    bandwidth_summary = bandwidth_daily.groupby("bandwidth")[list(METRICS)].mean().reset_index()
    selected_bandwidth = float(bandwidth_summary.sort_values(["wis_kw", "energy_score_kw", "variogram_score", "bandwidth"]).iloc[0].bandwidth)

    reduction_rows: list[dict[str, object]] = []
    for target_id, group in targets.groupby("target_day_local", sort=True):
        target = group.sort_values("timestamp_utc").reset_index(drop=True)
        generated = _generate(target, donors, recent, selected_bandwidth, raw_count)
        if generated is None:
            continue
        actual = target.actual_kw.to_numpy(float); point = target.point_forecast_kw.to_numpy(float)
        frozen = _baseline_matrix(baseline, str(target_id))
        for method, matrix in (("S2_FROZEN", frozen), (f"RAW_{raw_count}", generated.scenarios)):
            reduction_rows.append({"target_day_local": str(target_id), "method": method, **day_scores(actual, point, matrix)})
        for count in reduction_counts:
            reduced = reduce_scenarios(generated.scenarios, count)
            expanded = _expand_weighted(reduced.scenarios, reduced.weights, 50)
            reduction_rows.append({"target_day_local": str(target_id), "method": f"REDUCED_{count}", **day_scores(actual, point, expanded)})
    reduction_daily = pd.DataFrame(reduction_rows)
    reduction_summary = reduction_daily.groupby("method")[list(METRICS)].mean().reset_index()
    lookup = reduction_summary.set_index("method")
    raw_key = f"RAW_{raw_count}"; raw = lookup.loc[raw_key]
    feasible_counts: list[int] = []
    for count in reduction_counts:
        candidate = lookup.loc[f"REDUCED_{count}"]
        if (
            candidate.wis_kw <= raw.wis_kw * 1.01
            and candidate.energy_score_kw <= raw.energy_score_kw * 1.01
            and candidate.variogram_score <= raw.variogram_score * 1.01
            and abs(candidate.coverage_90 - raw.coverage_90) <= .03
        ):
            feasible_counts.append(count)
    selected_count = min(feasible_counts) if feasible_counts else raw_count
    bandwidth_daily.to_csv(output / "scenario_bandwidth_development_daily.csv", index=False, float_format="%.10g", lineterminator="\n")
    bandwidth_summary.to_csv(output / "scenario_bandwidth_development_summary.csv", index=False, float_format="%.10g", lineterminator="\n")
    reduction_daily.to_csv(output / "scenario_reduction_development_daily.csv", index=False, float_format="%.10g", lineterminator="\n")
    reduction_summary.to_csv(output / "scenario_reduction_development_summary.csv", index=False, float_format="%.10g", lineterminator="\n")
    result: dict[str, object] = {
        "schema_version": "prescriptive-ml.scenario-selection.v1",
        "development_year": 2022,
        "selection_primary_metric": "mean_daily_wis_kw",
        "bandwidth_candidates": bandwidths,
        "selected_bandwidth": selected_bandwidth,
        "raw_scenarios": raw_count,
        "reduction_candidates": reduction_counts,
        "selected_scenario_count": selected_count,
        "reduction_selected": selected_count != raw_count,
        "supported_days": int(reduction_daily.target_day_local.nunique()),
        "runtime_seconds": time.perf_counter() - started,
        "test_period_used_for_selection": False,
        "claim_boundary": "DEVELOPMENT_SELECTION_ONLY",
    }
    write_json(output / "scenario_development_selection.json", result)
    return result
