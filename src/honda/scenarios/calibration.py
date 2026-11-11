from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryScaleCalibration:
    scale: float
    nominal_levels: tuple[float, ...]
    mean_absolute_coverage_error_before: float
    mean_absolute_coverage_error_after: float
    fit_observations: int
    method: str = "DEVELOPMENT_ONLY_GLOBAL_TRAJECTORY_SCALE"

    def apply(self, point: np.ndarray, scenarios: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float); scenarios = np.asarray(scenarios, dtype=float)
        if scenarios.ndim != 2 or scenarios.shape[1] != point.shape[-1]:
            raise ValueError("Calibration scenario horizon mismatch")
        return point[None, :] + self.scale * (scenarios - point[None, :])


def _coverage_error(actual: np.ndarray, point: np.ndarray, scenarios: np.ndarray, levels: tuple[float, ...]) -> float:
    errors = []
    for level in levels:
        tail = (1 - level) / 2
        lower = np.quantile(scenarios, tail, axis=1)
        upper = np.quantile(scenarios, 1 - tail, axis=1)
        errors.append(abs(float(np.mean((actual >= lower) & (actual <= upper))) - level))
    return float(np.mean(errors))


def fit_global_trajectory_scale(
    actual: np.ndarray, point: np.ndarray, scenarios: np.ndarray,
    *, nominal_levels: tuple[float, ...] = (.5, .8, .9), candidate_scales: np.ndarray | None = None,
) -> TrajectoryScaleCalibration:
    """Fit one transparent dispersion scale using development observations only.

    Shapes are days×intervals for actual/point and days×scenarios×intervals for scenarios.
    A single global parameter preserves within-trajectory dependence and avoids
    optimization-outcome tuning.
    """
    actual = np.asarray(actual, dtype=float); point = np.asarray(point, dtype=float); scenarios = np.asarray(scenarios, dtype=float)
    if actual.shape != point.shape or scenarios.ndim != 3 or scenarios.shape[0] != actual.shape[0] or scenarios.shape[2] != actual.shape[1]:
        raise ValueError("Development calibration arrays are not aligned")
    if not all(np.isfinite(x).all() for x in (actual, point, scenarios)):
        raise ValueError("Calibration arrays must be finite")
    scales = np.linspace(.5, 3.0, 101) if candidate_scales is None else np.asarray(candidate_scales, dtype=float)
    if np.any(scales <= 0):
        raise ValueError("Candidate scales must be positive")
    flat_actual = actual.reshape(-1); flat_point = point.reshape(-1)
    raw = scenarios.transpose(0, 2, 1).reshape(-1, scenarios.shape[1])
    before = _coverage_error(flat_actual, flat_point, raw, nominal_levels)
    best_scale = 1.0; best_error = float("inf")
    for scale in scales:
        calibrated = flat_point[:, None] + scale * (raw - flat_point[:, None])
        error = _coverage_error(flat_actual, flat_point, calibrated, nominal_levels)
        if error < best_error - 1e-12 or (np.isclose(error, best_error) and abs(scale - 1) < abs(best_scale - 1)):
            best_scale, best_error = float(scale), error
    return TrajectoryScaleCalibration(best_scale, nominal_levels, before, best_error, int(flat_actual.size))


def joint_reliability(actual: np.ndarray, scenarios: np.ndarray, *, level: float = .9) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float); scenarios = np.asarray(scenarios, dtype=float)
    if actual.ndim != 2 or scenarios.ndim != 3 or scenarios.shape[0] != actual.shape[0] or scenarios.shape[2] != actual.shape[1]:
        raise ValueError("Joint reliability arrays are not aligned")
    tail = (1 - level) / 2
    lower = np.quantile(scenarios, tail, axis=1); upper = np.quantile(scenarios, 1 - tail, axis=1)
    covered = (actual >= lower) & (actual <= upper)
    dt = .25
    scenario_energy = scenarios.sum(axis=2) * dt; actual_energy = actual.sum(axis=1) * dt
    energy_lower = np.quantile(scenario_energy, tail, axis=1); energy_upper = np.quantile(scenario_energy, 1 - tail, axis=1)
    scenario_ramp = np.max(np.abs(np.diff(scenarios, axis=2)), axis=2); actual_ramp = np.max(np.abs(np.diff(actual, axis=1)), axis=1)
    ramp_upper = np.quantile(scenario_ramp, level, axis=1)
    return {
        "nominal_level": level,
        "marginal_interval_coverage": float(covered.mean()),
        "simultaneous_whole_day_coverage": float(covered.all(axis=1).mean()),
        "daily_energy_coverage": float(((actual_energy >= energy_lower) & (actual_energy <= energy_upper)).mean()),
        "maximum_ramp_upper_coverage": float((actual_ramp <= ramp_upper).mean()),
    }
