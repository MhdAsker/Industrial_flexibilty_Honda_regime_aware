from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MonitoringThresholds:
    maximum_data_age_seconds: float
    maximum_missing_fraction: float
    maximum_clock_drift_seconds: float
    maximum_absolute_bias_kw: float
    minimum_scenario_coverage: float
    maximum_solver_runtime_seconds: float
    maximum_schedule_deviation_kw: float
    maximum_value_shortfall_eur: float
    maximum_forecast_mae_kw: float = float("inf")
    maximum_dependence_score: float = float("inf")
    maximum_distribution_shift_score: float = float("inf")

    def validate(self) -> None:
        if min(asdict(self).values()) < 0:
            raise ValueError("Monitoring thresholds must be nonnegative")
        if not 0 <= self.maximum_missing_fraction <= 1 or not 0 <= self.minimum_scenario_coverage <= 1:
            raise ValueError("Missingness and coverage thresholds must lie in [0,1]")


@dataclass(frozen=True)
class OperationalMetrics:
    data_age_seconds: float
    missing_fraction: float
    clock_drift_seconds: float
    forecast_bias_kw: float
    scenario_coverage: float
    dependence_score: float | None
    distribution_shift_score: float | None
    solver_feasible: bool
    solver_runtime_seconds: float
    hard_constraint_violations: int
    schedule_deviation_kw: float
    expected_value_eur: float
    realized_value_eur: float
    tariff_version_changed: bool = False
    forecast_mae_kw: float | None = None

    def validate(self) -> None:
        numeric = [value for value in asdict(self).values() if isinstance(value, (float, int)) and not isinstance(value, bool)]
        if not np.isfinite(numeric).all():
            raise ValueError("Operational metrics contain nonfinite values")


def evaluate_health(metrics: OperationalMetrics, thresholds: MonitoringThresholds) -> dict[str, Any]:
    metrics.validate(); thresholds.validate()
    checks = {
        "data_freshness": metrics.data_age_seconds <= thresholds.maximum_data_age_seconds,
        "missingness": metrics.missing_fraction <= thresholds.maximum_missing_fraction,
        "clock_drift": abs(metrics.clock_drift_seconds) <= thresholds.maximum_clock_drift_seconds,
        "forecast_bias": abs(metrics.forecast_bias_kw) <= thresholds.maximum_absolute_bias_kw,
        "forecast_accuracy": None if metrics.forecast_mae_kw is None else metrics.forecast_mae_kw <= thresholds.maximum_forecast_mae_kw,
        "scenario_coverage": metrics.scenario_coverage >= thresholds.minimum_scenario_coverage,
        "scenario_dependence": None if metrics.dependence_score is None else metrics.dependence_score <= thresholds.maximum_dependence_score,
        "distribution_shift": None if metrics.distribution_shift_score is None else metrics.distribution_shift_score <= thresholds.maximum_distribution_shift_score,
        "solver_feasibility": metrics.solver_feasible,
        "solver_runtime": metrics.solver_runtime_seconds <= thresholds.maximum_solver_runtime_seconds,
        "hard_constraints": metrics.hard_constraint_violations == 0,
        "schedule_deviation": metrics.schedule_deviation_kw <= thresholds.maximum_schedule_deviation_kw,
        "value_shortfall": metrics.expected_value_eur - metrics.realized_value_eur <= thresholds.maximum_value_shortfall_eur,
        "tariff_rule_stability": not metrics.tariff_version_changed,
    }
    critical_names = {"data_freshness", "clock_drift", "solver_feasibility", "hard_constraints"}
    unknown = sorted(name for name, passed in checks.items() if passed is None)
    failed = sorted(name for name, passed in checks.items() if passed is False)
    severity = "HEALTHY"
    if unknown:
        severity = "UNKNOWN"
    elif failed:
        severity = "CRITICAL" if critical_names.intersection(failed) else "WARNING"
    return {
        "status": severity,
        "checks": checks,
        "failed_checks": failed,
        "unknown_checks": unknown,
        "automatic_action_allowed": severity == "HEALTHY",
        "metrics": asdict(metrics),
        "thresholds": asdict(thresholds),
    }
