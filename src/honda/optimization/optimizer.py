from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .contracts import DecisionInput, F1Config, F3Config, OptimizationResult, TariffConfig, stable_hash


class _Rows:
    def __init__(self, n_variables: int) -> None:
        self.n = n_variables
        self.rows: list[dict[int, float]] = []
        self.rhs: list[float] = []

    def add(self, coefficients: dict[int, float], rhs: float) -> None:
        self.rows.append({k: v for k, v in coefficients.items() if v != 0})
        self.rhs.append(float(rhs))

    def matrix(self):
        rr, cc, vv = [], [], []
        for r, row in enumerate(self.rows):
            for c, value in row.items():
                rr.append(r); cc.append(c); vv.append(value)
        return coo_matrix((vv, (rr, cc)), shape=(len(self.rows), self.n)).tocsr(), np.asarray(self.rhs)


def _deadline_pairs(horizon: int, maximum_lag: int, mode: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for source in range(horizon):
        for destination in range(horizon):
            if source == destination or abs(destination - source) > maximum_lag:
                continue
            if mode == "deferral_only" and destination < source:
                continue
            pairs.append((source, destination))
    return pairs


def _empirical_cvar(costs: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    order = np.argsort(costs)
    costs, weights = costs[order], weights[order]
    cumulative = np.cumsum(weights)
    z = costs[np.searchsorted(cumulative, alpha, side="left")]
    return float(z + np.sum(weights * np.maximum(costs - z, 0)) / (1 - alpha))


def solve_f1(
    decision: DecisionInput,
    flexibility: F1Config,
    grid: F3Config,
    tariff: TariffConfig,
    *,
    risk_lambda: float = 0.0,
    cvar_alpha: float = 0.95,
    highs_options: dict | None = None,
) -> OptimizationResult:
    """Solve F1 with one schedule shared by every uncertainty scenario."""
    started = time.perf_counter()
    flexibility.validate(); grid.validate()
    if risk_lambda < 0 or not 0 < cvar_alpha < 1:
        raise ValueError("Invalid CVaR settings")
    point, scenarios, weights, dt = decision.arrays()
    scenario_count, horizon = scenarios.shape
    buy, sell = tariff.arrays(horizon)

    cursor = 0
    u = np.arange(cursor, cursor + horizon); cursor += horizon
    x = np.arange(cursor, cursor + horizon + 1); cursor += horizon + 1
    imp = np.arange(cursor, cursor + scenario_count * horizon).reshape(scenario_count, horizon); cursor += scenario_count * horizon
    exp = np.arange(cursor, cursor + scenario_count * horizon).reshape(scenario_count, horizon); cursor += scenario_count * horizon
    peak = np.arange(cursor, cursor + scenario_count); cursor += scenario_count
    pairs: list[tuple[int, int]] = []
    q = np.asarray([], dtype=int)
    if flexibility.recovery_mode == "exact_flow_deadline":
        pairs = _deadline_pairs(horizon, int(flexibility.recovery_horizon_intervals), flexibility.operating_mode)
        q = np.arange(cursor, cursor + len(pairs)); cursor += len(pairs)
    soft_import = soft_export = np.empty((scenario_count, horizon), dtype=int)
    if grid.grid_limit_mode == "soft":
        soft_import = np.arange(cursor, cursor + scenario_count * horizon).reshape(scenario_count, horizon); cursor += scenario_count * horizon
        soft_export = np.arange(cursor, cursor + scenario_count * horizon).reshape(scenario_count, horizon); cursor += scenario_count * horizon
    z_idx = None; xi = np.asarray([], dtype=int)
    if risk_lambda > 0:
        z_idx = cursor; cursor += 1
        xi = np.arange(cursor, cursor + scenario_count); cursor += scenario_count

    c = np.zeros(cursor)
    for s in range(scenario_count):
        c[imp[s]] = weights[s] * dt * buy
        c[exp[s]] = -weights[s] * dt * sell
        c[peak[s]] = weights[s] * grid.peak_penalty_eur_per_kw_per_day
    if grid.grid_limit_mode == "soft":
        for s in range(scenario_count):
            c[soft_import[s]] = weights[s] * grid.soft_violation_penalty_eur_per_kw_interval
            c[soft_export[s]] = weights[s] * grid.soft_violation_penalty_eur_per_kw_interval
    if risk_lambda > 0:
        c[z_idx] = risk_lambda
        c[xi] = risk_lambda * weights / (1 - cvar_alpha)

    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend([(-flexibility.p_down_kw, flexibility.p_up_kw)] * horizon)
    advance = 0.0 if flexibility.operating_mode == "deferral_only" else flexibility.e_advance_kwh
    bounds.extend([(-flexibility.e_backlog_kwh, advance)] * (horizon + 1))
    for _ in range(scenario_count * horizon):
        upper = grid.import_max_kw if grid.grid_limit_mode == "scenario_wise" else None
        bounds.append((0.0, upper))
    for _ in range(scenario_count * horizon):
        upper = grid.export_max_kw if grid.grid_limit_mode == "scenario_wise" else None
        bounds.append((0.0, upper))
    bounds.extend([(0.0, None)] * scenario_count)
    bounds.extend([(0.0, None)] * len(pairs))
    if grid.grid_limit_mode == "soft":
        bounds.extend([(0.0, None)] * (2 * scenario_count * horizon))
    if risk_lambda > 0:
        bounds.append((None, None)); bounds.extend([(0.0, None)] * scenario_count)
    bounds[x[0]] = (flexibility.x_initial_kwh, flexibility.x_initial_kwh)
    bounds[x[-1]] = (flexibility.x_terminal_kwh, flexibility.x_terminal_kwh)

    equality = _Rows(cursor)
    inequality = _Rows(cursor)
    for t in range(horizon):
        equality.add({int(x[t + 1]): 1, int(x[t]): -1, int(u[t]): -dt[t]}, 0)
    for s in range(scenario_count):
        for t in range(horizon):
            equality.add({int(imp[s, t]): 1, int(exp[s, t]): -1, int(u[t]): -1}, scenarios[s, t])
            inequality.add({int(imp[s, t]): 1, int(peak[s]): -1}, 0)

    if flexibility.ramp_limit_kw_per_interval is not None:
        limit = flexibility.ramp_limit_kw_per_interval
        for t in range(1, horizon):
            inequality.add({int(u[t]): 1, int(u[t - 1]): -1}, limit)
            inequality.add({int(u[t]): -1, int(u[t - 1]): 1}, limit)

    if pairs:
        for t in range(horizon):
            row = {int(u[t]): dt[t]}
            for k, (source, destination) in enumerate(pairs):
                if destination == t: row[int(q[k])] = row.get(int(q[k]), 0) - 1
                if source == t: row[int(q[k])] = row.get(int(q[k]), 0) + 1
            equality.add(row, 0)
            outgoing = {int(q[k]): 1 for k, pair in enumerate(pairs) if pair[0] == t}
            incoming = {int(q[k]): 1 for k, pair in enumerate(pairs) if pair[1] == t}
            inequality.add(outgoing, flexibility.p_down_kw * dt[t])
            inequality.add(incoming, flexibility.p_up_kw * dt[t])

    if grid.grid_limit_mode in {"nominal", "chance_approximation"}:
        reference_upper = point if grid.grid_limit_mode == "nominal" else np.quantile(scenarios, grid.chance_quantile, axis=0)
        reference_lower = point if grid.grid_limit_mode == "nominal" else np.quantile(scenarios, 1 - grid.chance_quantile, axis=0)
        for t in range(horizon):
            if grid.import_max_kw is not None:
                inequality.add({int(u[t]): 1}, grid.import_max_kw - reference_upper[t])
            if grid.export_max_kw is not None:
                inequality.add({int(u[t]): -1}, grid.export_max_kw + reference_lower[t])
    elif grid.grid_limit_mode == "soft":
        for s in range(scenario_count):
            for t in range(horizon):
                if grid.import_max_kw is not None:
                    inequality.add({int(u[t]): 1, int(soft_import[s, t]): -1}, grid.import_max_kw - scenarios[s, t])
                if grid.export_max_kw is not None:
                    inequality.add({int(u[t]): -1, int(soft_export[s, t]): -1}, grid.export_max_kw + scenarios[s, t])

    if risk_lambda > 0:
        for s in range(scenario_count):
            row: dict[int, float] = {int(z_idx): -1, int(xi[s]): -1}
            for t in range(horizon):
                row[int(imp[s, t])] = dt[t] * buy[t]
                row[int(exp[s, t])] = -dt[t] * sell[t]
            row[int(peak[s])] = grid.peak_penalty_eur_per_kw_per_day
            inequality.add(row, 0)

    aeq, beq = equality.matrix(); aub, bub = inequality.matrix()
    solved = linprog(c, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq, bounds=bounds, method="highs", options=highs_options)
    cfg_hash = stable_hash({
        "decision": decision.configuration_hash(), "F1": asdict(flexibility), "F3": asdict(grid),
        "tariff": asdict(tariff), "risk_lambda": risk_lambda, "cvar_alpha": cvar_alpha,
    })
    if not solved.success:
        return OptimizationResult(
            solved.message, int(solved.status), None, np.full(horizon, np.nan), np.full(horizon + 1, np.nan),
            np.empty((0, horizon)), np.empty((0, horizon)), np.asarray([]), np.asarray([]), None,
            {"infeasible": 1.0}, time.perf_counter() - started, cfg_hash,
        )
    values = solved.x
    schedule = values[u]; state = values[x]
    # Canonicalize the split from physical net exchange. Under the validated
    # tariff dominance condition this is an equivalent optimum and removes
    # degenerate simultaneous import/export when buy and sell prices are equal.
    physical_grid = scenarios + schedule[None, :]
    imports = np.maximum(physical_grid, 0)
    exports = np.maximum(-physical_grid, 0)
    peaks = np.max(imports, axis=1)
    costs = np.sum((imports * buy - exports * sell) * dt, axis=1) + grid.peak_penalty_eur_per_kw_per_day * peaks
    if grid.grid_limit_mode == "soft":
        costs = costs + grid.soft_violation_penalty_eur_per_kw_interval * (
            np.sum(values[soft_import], axis=1) + np.sum(values[soft_export], axis=1)
        )
    grid_balance = scenarios + schedule[None, :] - imports + exports
    state_balance = state[1:] - state[:-1] - schedule * dt
    simultaneous = np.minimum(imports, exports)
    feasibility = {
        "max_grid_balance_residual_kw": float(np.max(np.abs(grid_balance))),
        "max_state_balance_residual_kwh": float(np.max(np.abs(state_balance))),
        "terminal_residual_kwh": float(abs(state[-1] - flexibility.x_terminal_kwh)),
        "max_power_violation_kw": float(max(np.max(schedule - flexibility.p_up_kw), np.max(-flexibility.p_down_kw - schedule), 0)),
        "max_state_violation_kwh": float(max(np.max(state - advance), np.max(-flexibility.e_backlog_kwh - state), 0)),
        "max_simultaneous_import_export_kw": float(np.max(simultaneous)),
    }
    if grid.import_max_kw is not None and grid.grid_limit_mode == "scenario_wise":
        feasibility["max_import_cap_violation_kw"] = float(max(np.max(imports - grid.import_max_kw), 0))
    if grid.export_max_kw is not None and grid.grid_limit_mode == "scenario_wise":
        feasibility["max_export_cap_violation_kw"] = float(max(np.max(exports - grid.export_max_kw), 0))
    return OptimizationResult(
        "optimal", int(solved.status), float(np.dot(weights, costs) + (risk_lambda * _empirical_cvar(costs, weights, cvar_alpha) if risk_lambda > 0 else 0)), schedule, state, imports, exports, costs, peaks,
        _empirical_cvar(costs, weights, cvar_alpha) if risk_lambda > 0 else None,
        feasibility, time.perf_counter() - started, cfg_hash,
        active_constraints={
            "power": int(np.sum(np.isclose(schedule, flexibility.p_up_kw) | np.isclose(schedule, -flexibility.p_down_kw))),
            "state": int(np.sum(np.isclose(state, advance) | np.isclose(state, -flexibility.e_backlog_kwh))),
        },
        metadata={"scenario_count": scenario_count, "horizon": horizon, "shared_first_stage_schedule": True, "deadline_flow_variables": len(pairs)},
    )


def settle_schedule(
    schedule_kw: np.ndarray,
    realized_baseline_kw: np.ndarray,
    interval_hours: np.ndarray,
    tariff: TariffConfig,
) -> dict[str, np.ndarray | float]:
    """Settle a committed schedule without reoptimization."""
    schedule = np.asarray(schedule_kw, dtype=float)
    baseline = np.asarray(realized_baseline_kw, dtype=float)
    dt = np.asarray(interval_hours, dtype=float)
    if not (len(schedule) == len(baseline) == len(dt)):
        raise ValueError("Settlement horizon mismatch")
    buy, sell = tariff.arrays(len(schedule))
    grid = baseline + schedule
    imports = np.maximum(grid, 0)
    exports = np.maximum(-grid, 0)
    cost = float(np.sum((imports * buy - exports * sell) * dt))
    return {"grid_kw": grid, "import_kw": imports, "export_kw": exports, "cost_eur": cost, "reoptimized": False}
