from __future__ import annotations

import numpy as np
import pandas as pd


QUANTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def pinball_loss(y: np.ndarray, q: np.ndarray, level: float) -> float:
    error = y - q
    return float(np.mean(np.maximum(level * error, (level - 1.0) * error)))


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    return (
        upper
        - lower
        + (2.0 / alpha) * (lower - y) * (y < lower)
        + (2.0 / alpha) * (y - upper) * (y > upper)
    )


def ensemble_quantiles(matrix: np.ndarray) -> np.ndarray:
    return np.quantile(matrix, QUANTILES, axis=0, method="linear")


def marginal_scores(actual: np.ndarray, matrix: np.ndarray) -> dict:
    q = ensemble_quantiles(matrix)
    levels = {level: q[i] for i, level in enumerate(QUANTILES)}
    result: dict[str, float] = {
        "mean_pinball_loss": float(np.mean([pinball_loss(actual, levels[x], x) for x in QUANTILES])),
        "quantile_crossings": 0,
    }
    weighted = 0.5 * np.abs(actual - levels[0.50])
    denominator = 0.5
    for nominal, alpha, lo, hi in [
        (50, 0.50, 0.25, 0.75),
        (80, 0.20, 0.10, 0.90),
        (90, 0.10, 0.05, 0.95),
    ]:
        lower, upper = levels[lo], levels[hi]
        result[f"coverage_{nominal}"] = float(np.mean((actual >= lower) & (actual <= upper)))
        result[f"width_{nominal}_kw"] = float(np.mean(upper - lower))
        weighted = weighted + (alpha / 2.0) * interval_score(actual, lower, upper, alpha)
        denominator += alpha / 2.0
    result["wis_kw"] = float(np.mean(weighted / denominator))
    return result


def energy_score(actual: np.ndarray, matrix: np.ndarray) -> float:
    scale = np.sqrt(len(actual))
    first = np.linalg.norm(matrix - actual[None, :], axis=1).mean() / scale
    pairwise = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2).mean() / scale
    return float(first - 0.5 * pairwise)


def variogram_score(
    actual: np.ndarray,
    matrix: np.ndarray,
    *,
    lags: tuple[int, ...] = (1, 4, 8, 16),
    p: float = 0.5,
) -> float:
    total = 0.0
    weight_sum = 0.0
    for lag in lags:
        if lag >= len(actual):
            continue
        observed = np.abs(actual[lag:] - actual[:-lag]) ** p
        simulated = np.mean(np.abs(matrix[:, lag:] - matrix[:, :-lag]) ** p, axis=0)
        weight = 1.0 / lag
        total += weight * float(np.mean((observed - simulated) ** 2))
        weight_sum += weight
    return float(total / weight_sum)


def lag_correlation(values: np.ndarray, lag: int) -> float:
    if lag >= len(values):
        return np.nan
    x, y = values[:-lag], values[lag:]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def dependence_scores(actual_residual: np.ndarray, scenario_residuals: np.ndarray) -> dict:
    result: dict[str, float] = {}
    squared = []
    for lag in [1, 4, 8, 16]:
        observed = lag_correlation(actual_residual, lag)
        simulated_values = np.asarray(
            [lag_correlation(row, lag) for row in scenario_residuals], dtype=float
        )
        simulated = (
            float(np.mean(simulated_values[np.isfinite(simulated_values)]))
            if np.isfinite(simulated_values).any()
            else np.nan
        )
        result[f"observed_acf_lag_{lag}"] = observed
        result[f"scenario_acf_lag_{lag}"] = float(simulated)
        if np.isfinite(observed) and np.isfinite(simulated):
            squared.append((observed - simulated) ** 2)
    result["acf_rmse"] = float(np.sqrt(np.mean(squared))) if squared else np.nan
    result["observed_mean_abs_ramp_kw"] = float(np.mean(np.abs(np.diff(actual_residual))))
    result["scenario_mean_abs_ramp_kw"] = float(np.mean(np.abs(np.diff(scenario_residuals, axis=1))))
    observed_energy = float(np.sum(actual_residual) * 0.25)
    scenario_energy = np.sum(scenario_residuals, axis=1) * 0.25
    result["observed_daily_residual_energy_kwh"] = observed_energy
    result["scenario_daily_energy_std_kwh"] = float(np.std(scenario_energy))
    result["scenario_daily_energy_mean_kwh"] = float(np.mean(scenario_energy))
    return result


def export_scores(actual: np.ndarray, matrix: np.ndarray) -> dict:
    event = (actual < 0).astype(float)
    probability = np.mean(matrix < 0, axis=0)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "export_prevalence": float(event.mean()),
        "mean_export_probability": float(probability.mean()),
        "brier_score": float(np.mean((probability - event) ** 2)),
        "log_loss": float(-np.mean(event * np.log(clipped) + (1 - event) * np.log(1 - clipped))),
        "observed_export_intervals": int(event.sum()),
        "scenario_export_intervals_mean": float(np.mean(np.sum(matrix < 0, axis=1))),
        "observed_any_export": bool(event.any()),
        "probability_any_export": float(np.mean(np.any(matrix < 0, axis=1))),
    }


def calibration_bins(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    index = np.minimum(np.digitize(probability, edges, right=False) - 1, bins - 1)
    rows = []
    for i in range(bins):
        mask = index == i
        if mask.any():
            rows.append(
                {
                    "bin": i,
                    "lower": edges[i],
                    "upper": edges[i + 1],
                    "count": int(mask.sum()),
                    "mean_probability": float(probability[mask].mean()),
                    "observed_frequency": float(actual[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def day_scores(actual: np.ndarray, point: np.ndarray, matrix: np.ndarray) -> dict:
    result = marginal_scores(actual, matrix)
    result["energy_score_kw"] = energy_score(actual, matrix)
    result["variogram_score"] = variogram_score(actual, matrix)
    result.update(dependence_scores(actual - point, matrix - point[None, :]))
    return result
