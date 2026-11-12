from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def pinball_loss(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = np.asarray(actual, dtype=float) - np.asarray(prediction, dtype=float)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def weighted_interval_score(actual: np.ndarray, quantile_predictions: np.ndarray, quantiles: tuple[float, ...]) -> float:
    """Normalized WIS for symmetric central intervals plus an optional median."""
    actual = np.asarray(actual, dtype=float)
    predictions = np.asarray(quantile_predictions, dtype=float)
    if predictions.shape != (len(actual), len(quantiles)):
        raise ValueError("Quantile prediction shape mismatch")
    lookup = {round(value, 10): index for index, value in enumerate(quantiles)}
    interval_scores: list[np.ndarray] = []
    weights: list[float] = []
    for lower in quantiles:
        upper = round(1 - lower, 10)
        if lower >= 0.5 or upper not in lookup:
            continue
        alpha = 2 * lower
        low = predictions[:, lookup[round(lower, 10)]]
        high = predictions[:, lookup[upper]]
        score = high - low + 2 / alpha * (low - actual) * (actual < low) + 2 / alpha * (actual - high) * (actual > high)
        interval_scores.append(score); weights.append(alpha / 2)
    if not interval_scores:
        raise ValueError("WIS requires at least one symmetric quantile pair")
    numerator = sum(weight * score for weight, score in zip(weights, interval_scores))
    denominator = sum(weights)
    if 0.5 in lookup:
        numerator = numerator + 0.5 * np.abs(actual - predictions[:, lookup[0.5]])
        denominator += 0.5
    return float(np.mean(numerator / denominator))


def evaluate_point_forecast(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float); prediction = np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 1 or not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Actual and prediction arrays must be aligned finite vectors")
    error = prediction - actual
    actual_export = actual < 0; predicted_export = prediction < 0
    export_rows = int(actual_export.sum())
    return {
        "rows": float(len(actual)),
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(error**2))),
        "bias_kw": float(np.mean(error)),
        "coverage": 1.0,
        "export_recall": float(np.sum(actual_export & predicted_export) / export_rows) if export_rows else float("nan"),
        "false_export_rate": float(np.sum(~actual_export & predicted_export) / max(int((~actual_export).sum()), 1)),
        "sign_accuracy": float(np.mean(actual_export == predicted_export)),
    }


def interval_reliability(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float) -> dict[str, float]:
    actual = np.asarray(actual); lower = np.asarray(lower); upper = np.asarray(upper)
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("Reliability arrays must align")
    coverage = float(np.mean((actual >= lower) & (actual <= upper)))
    return {
        "nominal_coverage": nominal,
        "empirical_coverage": coverage,
        "absolute_coverage_error": abs(coverage - nominal),
        "mean_width_kw": float(np.mean(upper - lower)),
    }


def paired_day_bootstrap(
    frame: pd.DataFrame,
    challenger_column: str,
    champion_column: str,
    *,
    day_column: str = "target_day_local",
    actual_column: str = "actual_kw",
    samples: int = 2000,
    seed: int = 20261029,
) -> dict[str, float]:
    required = {challenger_column, champion_column, day_column, actual_column}
    if not required.issubset(frame):
        raise ValueError(f"Missing bootstrap columns: {sorted(required - set(frame))}")
    daily = frame.groupby(day_column).apply(
        lambda group: pd.Series({
            "challenger": np.mean(np.abs(group[challenger_column] - group[actual_column])),
            "champion": np.mean(np.abs(group[champion_column] - group[actual_column])),
        }),
        include_groups=False,
    )
    difference = (daily.challenger - daily.champion).to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    draws = difference[indices].mean(axis=1)
    return {
        "days": float(len(difference)),
        "mean_mae_difference_kw": float(difference.mean()),
        "ci_lower_kw": float(np.quantile(draws, 0.025)),
        "ci_upper_kw": float(np.quantile(draws, 0.975)),
        "challenger_win_rate": float(np.mean(difference < 0)),
    }


@dataclass(frozen=True)
class PromotionGates:
    maximum_mae_kw: float = 69.4
    minimum_export_recall: float = 0.40
    minimum_coverage: float = 1.0
    bootstrap_upper_below_kw: float = 0.0
    maximum_slice_degradation: float = 0.05
    minimum_wis_improvement: float = 0.05
    minimum_interval_coverage: float = 0.87
    maximum_interval_coverage: float = 0.93
    maximum_absolute_coverage_error: float = 0.03


def promotion_report(
    point_metrics: dict[str, float],
    bootstrap: dict[str, float],
    *,
    gates: PromotionGates | None = None,
    interval_metrics: dict[str, float] | None = None,
    wis_improvement: float | None = None,
    maximum_slice_degradation: float | None = None,
) -> dict[str, Any]:
    gates = gates or PromotionGates()
    checks: dict[str, bool | None] = {
        "mae": point_metrics["mae_kw"] <= gates.maximum_mae_kw,
        "export_recall": point_metrics["export_recall"] >= gates.minimum_export_recall,
        "prediction_coverage": point_metrics["coverage"] >= gates.minimum_coverage,
        "paired_bootstrap": bootstrap["ci_upper_kw"] < gates.bootstrap_upper_below_kw,
        "slice_stability": None if maximum_slice_degradation is None else maximum_slice_degradation <= gates.maximum_slice_degradation,
        "wis": None if wis_improvement is None else wis_improvement >= gates.minimum_wis_improvement,
        "interval_coverage": None if interval_metrics is None else gates.minimum_interval_coverage <= interval_metrics["empirical_coverage"] <= gates.maximum_interval_coverage,
        "coverage_error": None if interval_metrics is None else interval_metrics["absolute_coverage_error"] <= gates.maximum_absolute_coverage_error,
    }
    required_present = all(value is not None for value in checks.values())
    return {
        "status": "PROMOTE" if required_present and all(checks.values()) else "DO_NOT_PROMOTE",
        "checks": checks,
        "required_metrics_complete": required_present,
        "gates": asdict(gates),
        "point_metrics": point_metrics,
        "paired_day_bootstrap": bootstrap,
        "interval_metrics": interval_metrics,
        "wis_improvement": wis_improvement,
        "maximum_slice_degradation": maximum_slice_degradation,
        "claim_boundary": "EXPLORATORY_EXPOSED_HOLDOUT" if point_metrics.get("evaluation_exposed", True) else "FROZEN_FORWARD_HOLDOUT",
    }


def artifact_digest(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
