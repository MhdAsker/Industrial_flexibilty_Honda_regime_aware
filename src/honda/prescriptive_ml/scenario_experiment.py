from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from honda.scenarios.adapters import load_honda_development, load_honda_test
from honda.scenarios.metrics import day_scores

from .evaluation import write_json
from .scenarios import conditional_coherent_scenarios, reduce_scenarios

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ROOT / "results/ml/h5_feature_matrix.parquet"
PREDICTIONS = ROOT / "results/prescriptive_ml/test_predictions_2023_exploratory.parquet"
BASELINE = ROOT / "results/scenarios/honda_2023_diagnostic_scenarios.parquet"
OUTPUT = ROOT / "results/prescriptive_ml"
QUANTILE_COLUMNS = ("Q05", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95")
QUANTILE_LEVELS = np.asarray((.05, .10, .25, .50, .75, .90, .95))


def _descriptor(day: pd.DataFrame, recent: dict[str, float]) -> np.ndarray:
    point = day.point_forecast_kw.to_numpy(float)
    season = str(day.season.iloc[0]); day_type = str(day.day_type.iloc[0])
    return np.asarray([
        np.sum(point) * .25, np.max(point), np.min(point), np.std(point),
        np.mean(np.abs(np.diff(point))), np.mean(point < 0),
        season == "winter", season == "spring", season == "summer", season == "autumn",
        day_type == "weekend",
        recent.get("grid_24h_mean", np.nan), recent.get("grid_7d_mean", np.nan),
        recent.get("grid_24h_export_fraction", np.nan), recent.get("grid_7d_export_fraction", np.nan),
        recent.get("chp_24h_on_fraction", np.nan),
    ], dtype=float)


def _recent_by_day() -> dict[str, dict[str, float]]:
    columns = [
        "target_day_local", "grid_24h_mean", "grid_7d_mean", "grid_24h_export_fraction",
        "grid_7d_export_fraction", "chp_24h_on_fraction",
    ]
    frame = pd.read_parquet(FEATURES, columns=columns)
    return frame.groupby("target_day_local", sort=False).first().to_dict("index")


def _marginal_samples(quantiles: np.ndarray, count: int) -> np.ndarray:
    probabilities = (np.arange(count) + .5) / count
    return np.column_stack([
        np.interp(probabilities, QUANTILE_LEVELS, quantiles[row], left=quantiles[row, 0], right=quantiles[row, -1])
        for row in range(len(quantiles))
    ])


def _expand_weighted(scenarios: np.ndarray, weights: np.ndarray, count: int = 50) -> np.ndarray:
    expected = weights * count
    repeats = np.floor(expected).astype(int)
    remainder = count - repeats.sum()
    if remainder:
        order = np.argsort(-(expected - repeats), kind="mergesort")
        repeats[order[:remainder]] += 1
    return np.repeat(scenarios, repeats, axis=0)


def run(
    output: str | Path = OUTPUT,
    raw_count: int = 200,
    reduced_count: int = 25,
    bandwidth: float = 2.0,
) -> dict[str, object]:
    started = time.perf_counter()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for path in (FEATURES, PREDICTIONS, BASELINE):
        if not path.exists():
            raise ValueError(f"Missing scenario experiment input: {path}")
    development = load_honda_development()
    test = load_honda_test()
    predictions = pd.read_parquet(PREDICTIONS, columns=["timestamp_utc", *QUANTILE_COLUMNS])
    test = test.merge(predictions, on="timestamp_utc", validate="one_to_one")
    baseline = pd.read_parquet(BASELINE)
    recent = _recent_by_day()
    donor_days = [group.sort_values("timestamp_utc").reset_index(drop=True) for _, group in development.groupby("target_day_local") if bool(group.complete_day.all())]
    rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for target_id, target_group in test.groupby("target_day_local", sort=True):
        target = target_group.sort_values("timestamp_utc").reset_index(drop=True)
        issue_time = pd.Timestamp(target.forecast_issue_time_utc.iloc[0])
        candidates = [
            day for day in donor_days
            if len(day) == len(target) and pd.Timestamp(day.timestamp_utc.iloc[-1]) + pd.Timedelta(minutes=15) <= issue_time
        ]
        if len(candidates) < 10:
            continue
        donor_features = np.stack([_descriptor(day, recent.get(str(day.target_day_local.iloc[0]), {})) for day in candidates])
        donor_residuals = np.stack([day.actual_kw.to_numpy(float) - day.point_forecast_kw.to_numpy(float) for day in candidates])
        availability = tuple((pd.Timestamp(day.timestamp_utc.iloc[-1]) + pd.Timedelta(minutes=15)).isoformat() for day in candidates)
        target_features = _descriptor(target, recent.get(str(target_id), {}))
        quantiles = target[list(QUANTILE_COLUMNS)].to_numpy(float)
        marginal_samples = _marginal_samples(quantiles, raw_count)
        generated = conditional_coherent_scenarios(
            target.point_forecast_kw.to_numpy(float), target_features, donor_features, donor_residuals,
            availability, issue_time.isoformat(), raw_count,
            marginal_samples_kw=marginal_samples, target_id=str(target_id), minimum_effective_donors=20,
            bandwidth=bandwidth,
        )
        reduced = reduce_scenarios(generated.scenarios, reduced_count)
        expanded = _expand_weighted(reduced.scenarios, reduced.weights, 50)
        actual = target.actual_kw.to_numpy(float); point = target.point_forecast_kw.to_numpy(float)
        raw_scores = day_scores(actual, point, generated.scenarios)
        reduced_scores = day_scores(actual, point, expanded)
        baseline_day = baseline[baseline.target_day_local.astype(str).eq(str(target_id))]
        scenario_ids = sorted(baseline_day.scenario_id.unique())
        baseline_matrix = np.stack([
            baseline_day[baseline_day.scenario_id.eq(identifier)].sort_values("target_timestamp_utc").scenario_value_kw.to_numpy(float)
            for identifier in scenario_ids
        ])
        old_scores = day_scores(actual, point, baseline_matrix)
        for method, scores in (
            ("S2_FROZEN", old_scores),
            ("CONDITIONAL_ECC_RAW", raw_scores),
            ("CONDITIONAL_ECC_REDUCED", reduced_scores),
        ):
            rows.append({"target_day_local": str(target_id), "method": method, **scores})
        selected_rows.append({
            "target_day_local": str(target_id), "raw_scenarios": raw_count, "reduced_scenarios": reduced_count,
            "effective_donor_count": generated.effective_donor_count,
            "reduction_mean_assignment_distance": reduced.mean_assignment_distance,
        })
    daily = pd.DataFrame(rows)
    daily.to_csv(output / "scenario_daily_scores_exploratory.csv", index=False, float_format="%.10g", lineterminator="\n")
    pd.DataFrame(selected_rows).to_csv(output / "scenario_reduction_diagnostics.csv", index=False, float_format="%.10g", lineterminator="\n")
    metrics = ["wis_kw", "energy_score_kw", "variogram_score", "coverage_90", "width_90_kw", "acf_rmse"]
    summary = daily.groupby("method")[metrics].mean().reset_index()
    summary.to_csv(output / "scenario_summary_exploratory.csv", index=False, float_format="%.10g", lineterminator="\n")
    pivot = summary.set_index("method")
    comparison = {
        method: {
            metric: {
                "value": float(pivot.loc[method, metric]),
                "frozen_s2": float(pivot.loc["S2_FROZEN", metric]),
                "relative_change": float((pivot.loc[method, metric] - pivot.loc["S2_FROZEN", metric]) / abs(pivot.loc["S2_FROZEN", metric])),
            }
            for metric in metrics
        }
        for method in ("CONDITIONAL_ECC_RAW", "CONDITIONAL_ECC_REDUCED")
    }
    result: dict[str, object] = {
        "schema_version": "prescriptive-ml.scenario-experiment.v1",
        "days": int(daily.target_day_local.nunique()),
        "raw_scenarios": raw_count,
        "reduced_scenarios": reduced_count,
        "selected_bandwidth": bandwidth,
        "selected_scenario_count": raw_count,
        "reduction_selected": False,
        "comparison": comparison,
        "runtime_seconds": time.perf_counter() - started,
        "status": "DEVELOPMENT_SELECTED_EXPLORATORY_DO_NOT_PROMOTE",
        "claim_boundary": "EXPOSED_2023_DIAGNOSTIC",
    }
    write_json(output / "scenario_experiment_summary.json", result)
    return result
