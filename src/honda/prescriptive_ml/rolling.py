from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingVintage:
    issue_time_utc: str
    timestamps_utc: tuple[str, ...]
    interval_hours: tuple[float, ...]
    information: dict[str, Any]

    def validate(self) -> None:
        issue = pd.Timestamp(self.issue_time_utc)
        stamps = pd.DatetimeIndex(pd.to_datetime(self.timestamps_utc, utc=True))
        dt = np.asarray(self.interval_hours, dtype=float)
        if issue.tzinfo is None or len(stamps) != len(dt) or len(stamps) == 0:
            raise ValueError("Rolling vintage timestamps and durations are invalid")
        if stamps[0] < issue:
            raise ValueError("Rolling vintage starts before its information issue time")
        if not stamps.is_monotonic_increasing or len(set(stamps)) != len(stamps) or np.any(dt <= 0):
            raise ValueError("Rolling physical timestamps must be increasing and unique")


@dataclass(frozen=True)
class RollingPlan:
    schedule_kw: tuple[float, ...]
    expected_cost_eur: float
    worst_case_cost_eur: float
    solver_runtime_seconds: float
    feasible: bool
    metadata: dict[str, Any]


class RollingPolicy(Protocol):
    def __call__(self, vintage: RollingVintage, initial_state_kwh: float) -> RollingPlan:
        ...


@dataclass
class RollingSimulationResult:
    executed_schedule_kw: np.ndarray
    state_kwh: np.ndarray
    steps: list[dict[str, Any]]
    infeasible_steps: int
    missed_intervals: int


@dataclass(frozen=True)
class RollingPolicyComparison:
    policy: str
    result: RollingSimulationResult
    settlement: dict[str, float]


def simulate_execute_first(
    vintages: tuple[RollingVintage, ...],
    realized_timestamps_utc: tuple[str, ...],
    interval_hours: tuple[float, ...],
    policy: RollingPolicy,
    *,
    initial_state_kwh: float,
    execute_intervals: int,
    state_min_kwh: float,
    state_max_kwh: float,
) -> RollingSimulationResult:
    """Execute only each vintage's commitment block, then carry state forward."""
    if execute_intervals < 1 or state_min_kwh > initial_state_kwh or state_max_kwh < initial_state_kwh:
        raise ValueError("Rolling execution configuration is invalid")
    realized_stamps = pd.DatetimeIndex(pd.to_datetime(realized_timestamps_utc, utc=True))
    realized_dt = np.asarray(interval_hours, dtype=float)
    if len(realized_stamps) != len(realized_dt) or len(set(realized_stamps)) != len(realized_stamps):
        raise ValueError("Realized rolling timeline is invalid")
    schedule_by_time: dict[pd.Timestamp, float] = {}
    state = float(initial_state_kwh)
    state_trace = [state]
    steps: list[dict[str, Any]] = []
    infeasible = 0
    for vintage in sorted(vintages, key=lambda value: pd.Timestamp(value.issue_time_utc)):
        vintage.validate()
        plan = policy(vintage, state)
        if len(plan.schedule_kw) != len(vintage.timestamps_utc):
            raise ValueError("Rolling policy schedule does not match its vintage horizon")
        if not plan.feasible:
            infeasible += 1
            steps.append({"issue_time_utc": vintage.issue_time_utc, "feasible": False, "executed_intervals": 0})
            continue
        executed = 0
        for stamp_raw, duration, adjustment in zip(vintage.timestamps_utc[:execute_intervals], vintage.interval_hours[:execute_intervals], plan.schedule_kw[:execute_intervals]):
            stamp = pd.Timestamp(stamp_raw)
            if stamp in schedule_by_time:
                continue
            candidate_state = state + float(adjustment) * float(duration)
            if not state_min_kwh - 1e-9 <= candidate_state <= state_max_kwh + 1e-9:
                raise ValueError("Rolling policy violates the carried state bounds")
            schedule_by_time[stamp] = float(adjustment)
            state = candidate_state; state_trace.append(state); executed += 1
        steps.append({
            "issue_time_utc": vintage.issue_time_utc,
            "feasible": True,
            "executed_intervals": executed,
            "state_after_kwh": state,
            "expected_cost_eur": plan.expected_cost_eur,
            "worst_case_cost_eur": plan.worst_case_cost_eur,
            "solver_runtime_seconds": plan.solver_runtime_seconds,
            "metadata": plan.metadata,
        })
    executed_schedule = np.asarray([schedule_by_time.get(stamp, np.nan) for stamp in realized_stamps])
    return RollingSimulationResult(
        executed_schedule_kw=executed_schedule,
        state_kwh=np.asarray(state_trace),
        steps=steps,
        infeasible_steps=infeasible,
        missed_intervals=int(np.isnan(executed_schedule).sum()),
    )


def compare_rolling_policies(
    policies: dict[str, RollingPolicy],
    vintages: tuple[RollingVintage, ...],
    realized_timestamps_utc: tuple[str, ...],
    interval_hours: tuple[float, ...],
    settlement: Callable[[RollingSimulationResult], dict[str, float]],
    *,
    initial_state_kwh: float,
    execute_intervals: int,
    state_min_kwh: float,
    state_max_kwh: float,
) -> tuple[RollingPolicyComparison, ...]:
    """Evaluate named policies on exactly the same vintages and settlement contract."""
    if not policies:
        raise ValueError("At least one rolling policy is required")
    comparisons = []
    for name, policy in policies.items():
        result = simulate_execute_first(
            vintages,
            realized_timestamps_utc,
            interval_hours,
            policy,
            initial_state_kwh=initial_state_kwh,
            execute_intervals=execute_intervals,
            state_min_kwh=state_min_kwh,
            state_max_kwh=state_max_kwh,
        )
        metrics = {key: float(value) for key, value in settlement(result).items()}
        if not metrics or not np.isfinite(list(metrics.values())).all():
            raise ValueError(f"Settlement metrics are invalid for policy {name}")
        comparisons.append(RollingPolicyComparison(name, result, metrics))
    return tuple(comparisons)


def cvar(costs: np.ndarray, weights: np.ndarray, alpha: float = 0.95) -> float:
    costs = np.asarray(costs, dtype=float); weights = np.asarray(weights, dtype=float)
    if len(costs) != len(weights) or np.any(weights < 0) or not np.isclose(weights.sum(), 1) or not 0 < alpha < 1:
        raise ValueError("CVaR inputs are invalid")
    order = np.argsort(costs); costs = costs[order]; weights = weights[order]
    cumulative = np.cumsum(weights)
    var = costs[min(np.searchsorted(cumulative, alpha, side="left"), len(costs) - 1)]
    return float(var + np.dot(weights, np.maximum(costs - var, 0)) / (1 - alpha))


def information_value_metrics(
    stochastic_cost: float,
    expected_value_policy_cost: float,
    wait_and_see_cost: float,
    no_recourse_cost: float,
    oracle_cost: float,
) -> dict[str, float]:
    values = np.asarray([stochastic_cost, expected_value_policy_cost, wait_and_see_cost, no_recourse_cost, oracle_cost], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Information-value costs must be finite")
    return {
        "value_of_stochastic_solution_eur": expected_value_policy_cost - stochastic_cost,
        "expected_value_of_perfect_information_eur": stochastic_cost - wait_and_see_cost,
        "recourse_value_eur": no_recourse_cost - stochastic_cost,
        "oracle_regret_eur": stochastic_cost - oracle_cost,
    }
