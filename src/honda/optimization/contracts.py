from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pandas as pd


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class F1Config:
    """SIMULATED_DEFERRABLE_PROCESS; capacities are assumptions, not site claims."""

    p_up_kw: float
    p_down_kw: float
    e_advance_kwh: float
    e_backlog_kwh: float
    x_initial_kwh: float = 0.0
    x_terminal_kwh: float = 0.0
    operating_mode: str = "advance_and_deferral"
    recovery_mode: str = "daily_terminal"
    recovery_horizon_intervals: int | None = None
    ramp_limit_kw_per_interval: float | None = None
    label: str = "SIMULATED_DEFERRABLE_PROCESS"

    def validate(self) -> None:
        if min(self.p_up_kw, self.p_down_kw, self.e_advance_kwh, self.e_backlog_kwh) < 0:
            raise ValueError("F1 capacities must be nonnegative")
        if self.operating_mode not in {"deferral_only", "advance_and_deferral"}:
            raise ValueError("Unsupported F1 operating mode")
        if self.recovery_mode not in {"daily_terminal", "exact_flow_deadline"}:
            raise ValueError("Unsupported F1 recovery mode")
        if self.recovery_mode == "exact_flow_deadline":
            if not self.recovery_horizon_intervals or self.recovery_horizon_intervals < 1:
                raise ValueError("Exact-flow recovery requires a positive horizon")
        if not (-self.e_backlog_kwh <= self.x_initial_kwh <= self.e_advance_kwh):
            raise ValueError("Initial F1 state is outside its energy bounds")
        if not (-self.e_backlog_kwh <= self.x_terminal_kwh <= self.e_advance_kwh):
            raise ValueError("Terminal F1 state is outside its energy bounds")
        if self.operating_mode == "deferral_only" and (
            self.x_initial_kwh > 0 or self.x_terminal_kwh > 0
        ):
            raise ValueError("Deferral-only mode cannot start or end advanced")


@dataclass(frozen=True)
class F2Config:
    """SIMULATED_THERMAL_BUFFER in electrical/thermal engineering units."""

    p_electric_max_kw: float
    thermal_capacity_kwh_th: float
    cop: float
    retention_per_interval: float = 1.0
    e_initial_kwh_th: float = 0.0
    e_terminal_kwh_th: float = 0.0
    label: str = "SIMULATED_THERMAL_BUFFER"

    def validate(self) -> None:
        if self.p_electric_max_kw < 0 or self.thermal_capacity_kwh_th < 0:
            raise ValueError("F2 capacities must be nonnegative")
        if self.cop <= 0:
            raise ValueError("COP must be positive")
        if not 0 < self.retention_per_interval <= 1:
            raise ValueError("Thermal retention must be in (0, 1]")
        if not 0 <= self.e_initial_kwh_th <= self.thermal_capacity_kwh_th:
            raise ValueError("Initial thermal state outside capacity")
        if not 0 <= self.e_terminal_kwh_th <= self.thermal_capacity_kwh_th:
            raise ValueError("Terminal thermal state outside capacity")


@dataclass(frozen=True)
class F3Config:
    import_max_kw: float | None = None
    export_max_kw: float | None = None
    peak_penalty_eur_per_kw_per_day: float = 0.0
    grid_limit_mode: str = "scenario_wise"
    chance_quantile: float = 0.95
    soft_violation_penalty_eur_per_kw_interval: float = 0.0

    def validate(self) -> None:
        if self.import_max_kw is not None and self.import_max_kw < 0:
            raise ValueError("Import capacity must be nonnegative")
        if self.export_max_kw is not None and self.export_max_kw < 0:
            raise ValueError("Export capacity must be nonnegative")
        if self.grid_limit_mode not in {"nominal", "scenario_wise", "chance_approximation", "soft"}:
            raise ValueError("Unsupported grid-limit mode")
        if not 0 < self.chance_quantile < 1:
            raise ValueError("Chance approximation quantile must lie in (0,1)")
        if self.peak_penalty_eur_per_kw_per_day < 0 or self.soft_violation_penalty_eur_per_kw_interval < 0:
            raise ValueError("Penalty coefficients must be nonnegative")


@dataclass(frozen=True)
class TariffConfig:
    import_price_eur_per_kwh: tuple[float, ...]
    export_price_eur_per_kwh: tuple[float, ...]
    label: str = "HYPOTHETICAL_WHOLESALE_PASS_THROUGH"

    @classmethod
    def from_eur_per_mwh(
        cls,
        import_price: np.ndarray | list[float],
        export_price: np.ndarray | list[float] | None = None,
        label: str = "HYPOTHETICAL_WHOLESALE_PASS_THROUGH",
    ) -> "TariffConfig":
        buy = np.asarray(import_price, dtype=float) / 1000.0
        sell = buy if export_price is None else np.asarray(export_price, dtype=float) / 1000.0
        return cls(tuple(buy), tuple(sell), label)

    def arrays(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        buy = np.asarray(self.import_price_eur_per_kwh, dtype=float)
        sell = np.asarray(self.export_price_eur_per_kwh, dtype=float)
        if len(buy) != horizon or len(sell) != horizon:
            raise ValueError("Tariff vector length does not match physical horizon")
        if not np.isfinite(buy).all() or not np.isfinite(sell).all():
            raise ValueError("Tariff contains nonfinite values")
        # With this dominance, increasing import and export together cannot reduce cost.
        if np.any(buy < sell - 1e-12):
            raise ValueError("Import price must dominate export remuneration to prevent artificial arbitrage")
        return buy, sell


@dataclass(frozen=True)
class DecisionInput:
    dataset_id: str
    target_day_local: str
    timestamp_utc: tuple[str, ...]
    interval_hours: tuple[float, ...]
    deterministic_forecast_kw: tuple[float, ...]
    scenario_matrix_kw: tuple[tuple[float, ...], ...]
    scenario_weights: tuple[float, ...]
    forecast_issue_time_utc: str
    decision_time_utc: str
    price_publication_time_utc: str | None
    timing_classification: str
    provenance_hashes: dict[str, str] = field(default_factory=dict)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        point = np.asarray(self.deterministic_forecast_kw, dtype=float)
        scenarios = np.asarray(self.scenario_matrix_kw, dtype=float)
        weights = np.asarray(self.scenario_weights, dtype=float)
        dt = np.asarray(self.interval_hours, dtype=float)
        if scenarios.ndim != 2 or scenarios.shape[1] != len(point):
            raise ValueError("Scenario matrix does not match deterministic horizon")
        if len(dt) != len(point) or len(self.timestamp_utc) != len(point):
            raise ValueError("Timestamp/duration horizon mismatch")
        if len(set(self.timestamp_utc)) != len(self.timestamp_utc):
            raise ValueError("Target UTC timestamps must be unique")
        if len(point) not in {2, 92, 96, 100}:
            raise ValueError("Unsupported physical horizon")
        if not np.isfinite(point).all() or not np.isfinite(scenarios).all() or not np.isfinite(dt).all():
            raise ValueError("Decision input contains nonfinite values")
        if np.any(dt <= 0):
            raise ValueError("Interval durations must be positive")
        if len(weights) != len(scenarios) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Scenario weights must be nonnegative and sum to one")
        issue = pd.Timestamp(self.forecast_issue_time_utc)
        decision = pd.Timestamp(self.decision_time_utc)
        if issue.tzinfo is None or decision.tzinfo is None or decision < issue:
            raise ValueError("Decision cannot precede the timezone-aware forecast issue")
        if self.timing_classification == "OPERATIONAL_TIMING_VERIFIED":
            if self.price_publication_time_utc is None or decision < pd.Timestamp(self.price_publication_time_utc):
                raise ValueError("Operational price-aware decision precedes verified publication")
        elif self.timing_classification not in {"EX_POST_PRICE_AWARE", "UNSUPPORTED_TIMING"}:
            raise ValueError("Unknown timing classification")
        return point, scenarios, weights, dt

    def configuration_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass
class OptimizationResult:
    solver_status: str
    solver_status_code: int
    objective_eur: float | None
    schedule_kw: np.ndarray
    state_kwh: np.ndarray
    scenario_import_kw: np.ndarray
    scenario_export_kw: np.ndarray
    scenario_cost_eur: np.ndarray
    peak_kw: np.ndarray
    cvar_eur: float | None
    feasibility: dict[str, float]
    runtime_seconds: float
    configuration_hash: str
    active_constraints: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
