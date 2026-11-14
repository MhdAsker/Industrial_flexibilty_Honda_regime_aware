from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.honda.h6_foundation_data import INDEX, ROOT


DEVELOPMENT_REFERENCE = ROOT / "results/ml/development_predictions.parquet"
TEST_REFERENCE = ROOT / "results/ml/test_predictions_2023.parquet"


def point_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape:
        raise ValueError("Actuals and predictions must have identical shape")
    eligible = np.isfinite(actual)
    predicted = eligible & np.isfinite(prediction)
    if not predicted.any():
        return {
            "eligible_targets": int(eligible.sum()),
            "predicted_targets": 0,
            "coverage": 0.0,
            "mae_kw": np.nan,
            "rmse_kw": np.nan,
            "bias_kw": np.nan,
            "median_ae_kw": np.nan,
            "nmae_absolute_target_scale": np.nan,
        }
    y = actual[predicted]
    p = prediction[predicted]
    error = p - y
    scale = float(np.mean(np.abs(y)))
    return {
        "eligible_targets": int(eligible.sum()),
        "predicted_targets": int(predicted.sum()),
        "coverage": float(predicted.sum() / eligible.sum()),
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(error**2))),
        "bias_kw": float(np.mean(error)),
        "median_ae_kw": float(np.median(np.abs(error))),
        "nmae_absolute_target_scale": float(np.mean(np.abs(error)) / scale) if scale else np.nan,
    }


def load_comparison_frame(predictions: pd.DataFrame, *, phase: str) -> pd.DataFrame:
    if phase not in {"development", "test"}:
        raise ValueError("phase must be development or test")
    reference_path = DEVELOPMENT_REFERENCE if phase == "development" else TEST_REFERENCE
    reference = pd.read_parquet(reference_path)
    reference_columns = ["timestamp_utc", "target_day_local", "actual_kw", "B5", "LGBM_M1"]
    if phase == "development":
        reference_columns.append("fold")
    reference = reference[reference_columns].rename(columns={"LGBM_M1": "LIGHTGBM"})
    pred = predictions[["timestamp_utc", "prediction_kw"]].copy()
    pred["timestamp_utc"] = pd.to_datetime(pred.timestamp_utc, utc=True)
    if pred.timestamp_utc.duplicated().any():
        raise ValueError("Kumo prediction timestamps are not unique")
    frame = reference.merge(pred, on="timestamp_utc", how="left", validate="one_to_one")
    protocol = pd.read_parquet(INDEX)[
        ["timestamp_utc", "lead_band", "holiday_context", "structural_era_metadata"]
    ]
    frame = frame.merge(protocol, on="timestamp_utc", how="left", validate="one_to_one")
    frame["grid_mode"] = np.select(
        [frame.actual_kw > 0, frame.actual_kw < 0], ["IMPORT", "EXPORT"], default="NEUTRAL"
    )
    frame["error_kw"] = frame.prediction_kw - frame.actual_kw
    return frame


def evaluation_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = pd.DataFrame([{"model": "KUMO_FORECAST", **point_metrics(frame.actual_kw, frame.prediction_kw)}])
    tables = {"summary": summary}
    for key in ["lead_band", "holiday_context", "structural_era_metadata", "grid_mode"]:
        rows = []
        for value, group in frame.groupby(key, dropna=False, sort=True):
            rows.append({key: value, **point_metrics(group.actual_kw, group.prediction_kw)})
        tables[key] = pd.DataFrame(rows)
    return tables


def paired_day_bootstrap(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    first_name: str,
    second_name: str,
    resamples: int = 2000,
    seed: int = 20261029,
) -> dict[str, float | int | str]:
    left = first[["timestamp_utc", "target_day_local", "actual_kw", "prediction_kw"]].rename(
        columns={"prediction_kw": "first"}
    )
    right = second[["timestamp_utc", "prediction_kw"]].rename(columns={"prediction_kw": "second"})
    common = left.merge(right, on="timestamp_utc", validate="one_to_one")
    common = common[np.isfinite(common[["actual_kw", "first", "second"]]).all(axis=1)]
    daily = common.groupby("target_day_local").apply(
        lambda x: pd.Series(
            {
                "first_mae": np.mean(np.abs(x["first"] - x.actual_kw)),
                "second_mae": np.mean(np.abs(x["second"] - x.actual_kw)),
            }
        ),
        include_groups=False,
    )
    if daily.empty:
        raise ValueError("No common complete comparison days")
    difference = (daily.first_mae - daily.second_mae).to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = np.empty(resamples)
    for index in range(resamples):
        sampled[index] = difference[rng.integers(0, len(difference), len(difference))].mean()
    return {
        "first": first_name,
        "second": second_name,
        "common_days": int(len(difference)),
        "mean_daily_mae_difference_kw": float(difference.mean()),
        "ci95_lower_kw": float(np.quantile(sampled, 0.025)),
        "ci95_upper_kw": float(np.quantile(sampled, 0.975)),
        "first_daily_win_rate": float(np.mean(difference < 0)),
        "bootstrap_resamples": resamples,
        "seed": seed,
    }


def write_evaluation(prediction_path: Path, output_dir: Path, *, phase: str) -> dict[str, Path]:
    predictions = pd.read_parquet(prediction_path)
    frame = load_comparison_frame(predictions, phase=phase)
    tables = evaluation_tables(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{phase}_{name}.csv"
        table.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
        paths[name] = path
    return paths
