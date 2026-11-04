from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linprog

from .contracts import OptimizationResult, TariffConfig, stable_hash


def solve_f0(
    baseline_demand_kw: np.ndarray,
    interval_hours: np.ndarray,
    tariff: TariffConfig,
    *,
    flexible_fraction: float = 0.2,
    shift_window_intervals: int = 8,
    import_cap_kw: float | None = None,
) -> OptimizationResult:
    """SPARK-only legacy divisible percentage-shifting comparator."""
    started = time.perf_counter()
    baseline = np.asarray(baseline_demand_kw, dtype=float)
    dt = np.asarray(interval_hours, dtype=float)
    if np.any(baseline < 0):
        raise ValueError("F0 is disabled for signed/negative grid exchange")
    if not 0 <= flexible_fraction <= 1 or shift_window_intervals < 0 or len(dt) != len(baseline):
        raise ValueError("Invalid F0 configuration")
    if not np.allclose(dt, dt[0]):
        raise ValueError("F0 legacy routing requires equal physical interval durations")
    buy, _ = tariff.arrays(len(baseline))
    horizon = len(baseline)
    pairs = [(i, j) for i in range(horizon) for j in range(horizon) if abs(j - i) <= shift_window_intervals]
    n = len(pairs)
    c = np.asarray([buy[j] for i, j in pairs])
    aeq = np.zeros((horizon, n)); beq = flexible_fraction * baseline * dt
    for k, (source, destination) in enumerate(pairs): aeq[source, k] = 1
    aub = []; bub = []
    if import_cap_kw is not None:
        fixed = (1 - flexible_fraction) * baseline
        for destination in range(horizon):
            row = np.zeros(n)
            for k, pair in enumerate(pairs):
                if pair[1] == destination: row[k] = 1 / dt[destination]
            aub.append(row); bub.append(import_cap_kw - fixed[destination])
    solved = linprog(c, A_ub=np.asarray(aub) if aub else None, b_ub=np.asarray(bub) if bub else None, A_eq=aeq, b_eq=beq, bounds=[(0, None)] * n, method="highs")
    cfg_hash = stable_hash({"archetype": "F0_SPARK_LEGACY", "fraction": flexible_fraction, "window": shift_window_intervals, "cap": import_cap_kw})
    if not solved.success:
        return OptimizationResult(solved.message, int(solved.status), None, np.full(horizon, np.nan), np.full(horizon + 1, np.nan), np.empty((0, horizon)), np.empty((0, horizon)), np.asarray([]), np.asarray([]), None, {"infeasible": 1.0}, time.perf_counter() - started, cfg_hash)
    routed = np.zeros(horizon)
    for value, (_, destination) in zip(solved.x, pairs): routed[destination] += value / dt[destination]
    optimized = (1 - flexible_fraction) * baseline + routed
    schedule = optimized - baseline
    state = np.r_[0, np.cumsum(schedule * dt)]
    cost = float(np.sum(optimized * buy * dt))
    return OptimizationResult(
        "optimal", int(solved.status), cost, schedule, state, optimized[None, :], np.zeros((1, horizon)),
        np.asarray([cost]), np.asarray([optimized.max()]), None,
        {"daily_energy_residual_kwh": float(abs(np.sum(schedule * dt))), "terminal_residual_kwh": float(abs(state[-1]))},
        time.perf_counter() - started, cfg_hash, metadata={"archetype": "F0_SPARK_LEGACY_ONLY"},
    )
