from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
from scipy.optimize import linprog

from .contracts import F2Config, F3Config, OptimizationResult, TariffConfig, stable_hash


def solve_f2(
    baseline_grid_kw: np.ndarray,
    baseline_thermal_input_kw: np.ndarray,
    thermal_demand_kw_th: np.ndarray,
    interval_hours: np.ndarray,
    flexibility: F2Config,
    grid: F3Config,
    tariff: TariffConfig,
) -> OptimizationResult:
    """Solve a deterministic SIMULATED_THERMAL_BUFFER engineering fixture."""
    started = time.perf_counter()
    flexibility.validate(); grid.validate()
    baseline = np.asarray(baseline_grid_kw, dtype=float)
    baseline_input = np.asarray(baseline_thermal_input_kw, dtype=float)
    demand = np.asarray(thermal_demand_kw_th, dtype=float)
    dt = np.asarray(interval_hours, dtype=float)
    horizon = len(baseline)
    if not (len(baseline_input) == len(demand) == len(dt) == horizon):
        raise ValueError("F2 horizon mismatch")
    buy, sell = tariff.arrays(horizon)
    p = np.arange(horizon)
    e = np.arange(horizon, horizon + horizon + 1)
    imp = np.arange(2 * horizon + 1, 3 * horizon + 1)
    exp = np.arange(3 * horizon + 1, 4 * horizon + 1)
    peak = 4 * horizon + 1
    n = peak + 1
    c = np.zeros(n)
    c[imp] = dt * buy; c[exp] = -dt * sell; c[peak] = grid.peak_penalty_eur_per_kw_per_day
    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend([(0, flexibility.p_electric_max_kw)] * horizon)
    bounds.extend([(0, flexibility.thermal_capacity_kwh_th)] * (horizon + 1))
    bounds.extend([(0, grid.import_max_kw if grid.grid_limit_mode == "scenario_wise" else None)] * horizon)
    bounds.extend([(0, grid.export_max_kw if grid.grid_limit_mode == "scenario_wise" else None)] * horizon)
    bounds.append((0, None))
    bounds[e[0]] = (flexibility.e_initial_kwh_th, flexibility.e_initial_kwh_th)
    bounds[e[-1]] = (flexibility.e_terminal_kwh_th, flexibility.e_terminal_kwh_th)
    aeq = []; beq = []
    for t in range(horizon):
        row = np.zeros(n); row[e[t + 1]] = 1; row[e[t]] = -flexibility.retention_per_interval; row[p[t]] = -flexibility.cop * dt[t]
        aeq.append(row); beq.append(-demand[t] * dt[t])
        row = np.zeros(n); row[imp[t]] = 1; row[exp[t]] = -1; row[p[t]] = -1
        aeq.append(row); beq.append(baseline[t] - baseline_input[t])
    aub = []; bub = []
    for t in range(horizon):
        row = np.zeros(n); row[imp[t]] = 1; row[peak] = -1; aub.append(row); bub.append(0)
    solved = linprog(c, A_ub=np.asarray(aub), b_ub=np.asarray(bub), A_eq=np.asarray(aeq), b_eq=np.asarray(beq), bounds=bounds, method="highs")
    cfg_hash = stable_hash({"F2": asdict(flexibility), "F3": asdict(grid), "tariff": asdict(tariff)})
    if not solved.success:
        return OptimizationResult(
            solved.message, int(solved.status), None, np.full(horizon, np.nan), np.full(horizon + 1, np.nan),
            np.empty((0, horizon)), np.empty((0, horizon)), np.asarray([]), np.asarray([]), None,
            {"infeasible": 1.0}, time.perf_counter() - started, cfg_hash,
        )
    values = solved.x
    schedule = values[p] - baseline_input
    state = values[e]
    imports = values[imp][None, :]; exports = values[exp][None, :]
    cost = np.asarray([np.sum((imports[0] * buy - exports[0] * sell) * dt) + grid.peak_penalty_eur_per_kw_per_day * values[peak]])
    thermal_residual = state[1:] - flexibility.retention_per_interval * state[:-1] - (flexibility.cop * values[p] - demand) * dt
    grid_residual = baseline + schedule - imports[0] + exports[0]
    return OptimizationResult(
        "optimal", int(solved.status), float(solved.fun), schedule, state, imports, exports, cost,
        np.asarray([values[peak]]), None,
        {
            "max_thermal_balance_residual_kwh_th": float(np.max(np.abs(thermal_residual))),
            "max_grid_balance_residual_kw": float(np.max(np.abs(grid_residual))),
            "terminal_residual_kwh_th": float(abs(state[-1] - flexibility.e_terminal_kwh_th)),
            "max_simultaneous_import_export_kw": float(np.max(np.minimum(imports, exports))),
        },
        time.perf_counter() - started, cfg_hash,
        metadata={"archetype": flexibility.label, "thermal_input_kw": values[p]},
    )
