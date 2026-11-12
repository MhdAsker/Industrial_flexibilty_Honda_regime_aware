from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .contracts import (
    CommercialTerms,
    RollingDecisionInput,
    ShadowAsset,
    TwoStageConfig,
    stable_hash,
)


class _Rows:
    def __init__(self, variables: int) -> None:
        self.variables = variables
        self.rows: list[dict[int, float]] = []
        self.rhs: list[float] = []

    def add(self, row: dict[int, float], rhs: float) -> None:
        self.rows.append({key: value for key, value in row.items() if value})
        self.rhs.append(float(rhs))

    def matrix(self):
        rr: list[int] = []
        cc: list[int] = []
        vv: list[float] = []
        for row_index, row in enumerate(self.rows):
            for column, value in row.items():
                rr.append(row_index); cc.append(column); vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(self.rows), self.variables)).tocsr()
        return matrix, np.asarray(self.rhs)


@dataclass
class TwoStageResult:
    solver_status: str
    objective_eur: float | None
    day_ahead_schedule_kw: np.ndarray
    scenario_recourse_kw: np.ndarray
    scenario_state_kwh: np.ndarray
    scenario_grid_kw: np.ndarray
    scenario_cost_eur: np.ndarray
    expected_cost_eur: float | None
    worst_case_cost_eur: float | None
    cvar_eur: float | None
    active_constraints: dict[str, int]
    nonanticipativity_max_residual_kw: float
    runtime_seconds: float
    configuration_hash: str
    metadata: dict[str, Any]


def _nonanticipativity_residual(recourse: np.ndarray, nodes: np.ndarray) -> float:
    residual = 0.0
    for t in range(nodes.shape[1]):
        for node in np.unique(nodes[:, t]):
            values = recourse[nodes[:, t] == node, t]
            if len(values):
                residual = max(residual, float(np.ptp(values)))
    return residual


def _weighted_cvar(costs: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    order = np.argsort(costs)
    ordered_costs = costs[order]; ordered_weights = weights[order]
    threshold = ordered_costs[min(np.searchsorted(np.cumsum(ordered_weights), alpha, side="left"), len(costs) - 1)]
    return float(threshold + np.dot(weights, np.maximum(costs - threshold, 0)) / (1 - alpha))


def solve_two_stage(
    decision: RollingDecisionInput,
    asset: ShadowAsset,
    commercial: CommercialTerms,
    config: TwoStageConfig,
) -> TwoStageResult:
    """Solve a two-stage rolling schedule with scenario-tree nonanticipativity.

    The day-ahead schedule is common to all scenarios. Intraday corrections may
    differ only after their scenario-tree node IDs diverge.
    """
    started = time.perf_counter()
    asset.validate()
    _point, scenarios, weights, dt, available = decision.arrays()
    scenario_count, horizon = scenarios.shape
    nodes = np.asarray(decision.scenario_node_ids, dtype=str)
    config.validate(horizon)
    buy, sell, network = commercial.arrays(horizon)

    cursor = 0
    day_ahead = np.arange(cursor, cursor + horizon); cursor += horizon
    recourse = np.arange(cursor, cursor + scenario_count * horizon).reshape(scenario_count, horizon); cursor += scenario_count * horizon
    state = np.arange(cursor, cursor + scenario_count * (horizon + 1)).reshape(scenario_count, horizon + 1); cursor += scenario_count * (horizon + 1)
    imports = np.arange(cursor, cursor + scenario_count * horizon).reshape(scenario_count, horizon); cursor += scenario_count * horizon
    exports = np.arange(cursor, cursor + scenario_count * horizon).reshape(scenario_count, horizon); cursor += scenario_count * horizon
    peak = np.arange(cursor, cursor + scenario_count); cursor += scenario_count
    recourse_abs = np.arange(cursor, cursor + scenario_count * horizon).reshape(scenario_count, horizon); cursor += scenario_count * horizon
    state_abs = np.arange(cursor, cursor + scenario_count * (horizon + 1)).reshape(scenario_count, horizon + 1); cursor += scenario_count * (horizon + 1)
    import_violation = export_violation = np.empty((scenario_count, horizon), dtype=int)
    if not config.hard_connection_limits:
        import_violation = np.arange(cursor, cursor + scenario_count * horizon).reshape(scenario_count, horizon); cursor += scenario_count * horizon
        export_violation = np.arange(cursor, cursor + scenario_count * horizon).reshape(scenario_count, horizon); cursor += scenario_count * horizon
    cvar_threshold = None
    cvar_excess = np.asarray([], dtype=int)
    if config.risk_lambda > 0:
        cvar_threshold = cursor; cursor += 1
        cvar_excess = np.arange(cursor, cursor + scenario_count); cursor += scenario_count

    objective = np.zeros(cursor)
    for s in range(scenario_count):
        objective[imports[s]] = weights[s] * dt * (buy + network)
        objective[exports[s]] = -weights[s] * dt * sell
        objective[peak[s]] = weights[s] * commercial.demand_charge_eur_per_kw
        objective[recourse_abs[s]] = weights[s] * dt * (commercial.recourse_eur_per_kwh + commercial.imbalance_eur_per_kwh)
        objective[state_abs[s]] = weights[s] * commercial.service_state_eur_per_kwh_interval
        if not config.hard_connection_limits:
            objective[import_violation[s]] = weights[s] * commercial.connection_violation_eur_per_kw_interval
            objective[export_violation[s]] = weights[s] * commercial.connection_violation_eur_per_kw_interval
    if config.risk_lambda > 0:
        objective[cvar_threshold] = config.risk_lambda
        objective[cvar_excess] = config.risk_lambda * weights / (1 - config.cvar_alpha)

    bounds: list[tuple[float | None, float | None]] = []
    for is_available in available:
        bounds.append((-asset.p_down_kw, asset.p_up_kw) if is_available else (0.0, 0.0))
    for _s in range(scenario_count):
        for t, is_available in enumerate(available):
            limit = config.adjustment_limit_kw
            bounds.append((-limit, limit) if is_available and t >= config.recourse_start_interval else (0.0, 0.0))
    bounds.extend([(asset.state_min_kwh, asset.state_max_kwh)] * (scenario_count * (horizon + 1)))
    import_upper = config.import_limit_kw if config.hard_connection_limits else None
    export_upper = config.export_limit_kw if config.hard_connection_limits else None
    bounds.extend([(0.0, import_upper)] * (scenario_count * horizon))
    bounds.extend([(0.0, export_upper)] * (scenario_count * horizon))
    bounds.extend([(0.0, None)] * scenario_count)
    bounds.extend([(0.0, None)] * (scenario_count * horizon))
    bounds.extend([(0.0, None)] * (scenario_count * (horizon + 1)))
    if not config.hard_connection_limits:
        bounds.extend([(0.0, None)] * (2 * scenario_count * horizon))
    if config.risk_lambda > 0:
        bounds.append((None, None)); bounds.extend([(0.0, None)] * scenario_count)
    for s in range(scenario_count):
        bounds[int(state[s, 0])] = (asset.initial_state_kwh, asset.initial_state_kwh)
        bounds[int(state[s, -1])] = (asset.terminal_state_kwh, asset.terminal_state_kwh)

    equality = _Rows(cursor)
    inequality = _Rows(cursor)
    for s in range(scenario_count):
        for t in range(horizon):
            equality.add(
                {int(state[s, t + 1]): 1, int(state[s, t]): -1, int(day_ahead[t]): -dt[t], int(recourse[s, t]): -dt[t]},
                0,
            )
            equality.add(
                {int(imports[s, t]): 1, int(exports[s, t]): -1, int(day_ahead[t]): -1, int(recourse[s, t]): -1},
                scenarios[s, t],
            )
            inequality.add({int(day_ahead[t]): 1, int(recourse[s, t]): 1}, asset.p_up_kw)
            inequality.add({int(day_ahead[t]): -1, int(recourse[s, t]): -1}, asset.p_down_kw)
            inequality.add({int(recourse[s, t]): 1, int(recourse_abs[s, t]): -1}, 0)
            inequality.add({int(recourse[s, t]): -1, int(recourse_abs[s, t]): -1}, 0)
            inequality.add({int(imports[s, t]): 1, int(peak[s]): -1}, 0)
            if not config.hard_connection_limits:
                if config.import_limit_kw is not None:
                    inequality.add({int(imports[s, t]): 1, int(import_violation[s, t]): -1}, config.import_limit_kw)
                if config.export_limit_kw is not None:
                    inequality.add({int(exports[s, t]): 1, int(export_violation[s, t]): -1}, config.export_limit_kw)
        for t in range(horizon + 1):
            inequality.add({int(state[s, t]): 1, int(state_abs[s, t]): -1}, 0)
            inequality.add({int(state[s, t]): -1, int(state_abs[s, t]): -1}, 0)
        if asset.ramp_limit_kw_per_interval is not None:
            for t in range(1, horizon):
                limit = asset.ramp_limit_kw_per_interval
                inequality.add({int(day_ahead[t]): 1, int(recourse[s, t]): 1, int(day_ahead[t - 1]): -1, int(recourse[s, t - 1]): -1}, limit)
                inequality.add({int(day_ahead[t]): -1, int(recourse[s, t]): -1, int(day_ahead[t - 1]): 1, int(recourse[s, t - 1]): 1}, limit)

    # Same information node => same correction. This is the explicit
    # nonanticipativity condition absent from the offline ex-post correction.
    for t in range(horizon):
        for node in np.unique(nodes[:, t]):
            members = np.flatnonzero(nodes[:, t] == node)
            anchor = int(members[0])
            for member in members[1:]:
                equality.add({int(recourse[int(member), t]): 1, int(recourse[anchor, t]): -1}, 0)

    if config.risk_lambda > 0:
        for s in range(scenario_count):
            row: dict[int, float] = {int(cvar_threshold): -1, int(cvar_excess[s]): -1}
            for t in range(horizon):
                row[int(imports[s, t])] = dt[t] * (buy[t] + network[t])
                row[int(exports[s, t])] = -dt[t] * sell[t]
                row[int(recourse_abs[s, t])] = dt[t] * (commercial.recourse_eur_per_kwh + commercial.imbalance_eur_per_kwh)
            for t in range(horizon + 1):
                row[int(state_abs[s, t])] = commercial.service_state_eur_per_kwh_interval
            row[int(peak[s])] = commercial.demand_charge_eur_per_kw
            if not config.hard_connection_limits:
                for t in range(horizon):
                    row[int(import_violation[s, t])] = commercial.connection_violation_eur_per_kw_interval
                    row[int(export_violation[s, t])] = commercial.connection_violation_eur_per_kw_interval
            inequality.add(row, 0)

    aeq, beq = equality.matrix()
    aub, bub = inequality.matrix()
    solved = linprog(objective, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq, bounds=bounds, method="highs")
    configuration_hash = stable_hash({
        "decision": decision.hash(), "asset": asdict(asset), "commercial": asdict(commercial), "config": asdict(config)
    })
    if not solved.success:
        return TwoStageResult(
            solver_status=solved.message,
            objective_eur=None,
            day_ahead_schedule_kw=np.full(horizon, np.nan),
            scenario_recourse_kw=np.full((scenario_count, horizon), np.nan),
            scenario_state_kwh=np.full((scenario_count, horizon + 1), np.nan),
            scenario_grid_kw=np.full((scenario_count, horizon), np.nan),
            scenario_cost_eur=np.full(scenario_count, np.nan),
            expected_cost_eur=None,
            worst_case_cost_eur=None,
            cvar_eur=None,
            active_constraints={},
            nonanticipativity_max_residual_kw=float("nan"),
            runtime_seconds=time.perf_counter() - started,
            configuration_hash=configuration_hash,
            metadata={"scenario_count": scenario_count, "horizon": horizon, "genuine_recourse": True},
        )

    values = solved.x
    u = values[day_ahead]
    r = values[recourse]
    x = values[state]
    grid = scenarios + u[None, :] + r
    imp = np.maximum(grid, 0)
    exp = np.maximum(-grid, 0)
    scenario_cost = np.sum((imp * (buy + network) - exp * sell) * dt, axis=1)
    scenario_cost += commercial.demand_charge_eur_per_kw * np.max(imp, axis=1)
    scenario_cost += (commercial.recourse_eur_per_kwh + commercial.imbalance_eur_per_kwh) * np.sum(np.abs(r) * dt, axis=1)
    scenario_cost += commercial.service_state_eur_per_kwh_interval * np.sum(np.abs(x), axis=1)
    if not config.hard_connection_limits:
        scenario_cost += commercial.connection_violation_eur_per_kw_interval * (
            np.maximum(imp - (config.import_limit_kw if config.import_limit_kw is not None else np.inf), 0).sum(axis=1)
            + np.maximum(exp - (config.export_limit_kw if config.export_limit_kw is not None else np.inf), 0).sum(axis=1)
        )
    physical = u[None, :] + r
    cvar_value = _weighted_cvar(scenario_cost, weights, config.cvar_alpha)
    return TwoStageResult(
        solver_status="optimal",
        objective_eur=float(solved.fun),
        day_ahead_schedule_kw=u,
        scenario_recourse_kw=r,
        scenario_state_kwh=x,
        scenario_grid_kw=grid,
        scenario_cost_eur=scenario_cost,
        expected_cost_eur=float(np.dot(weights, scenario_cost)),
        worst_case_cost_eur=float(np.max(scenario_cost)),
        cvar_eur=cvar_value,
        active_constraints={
            "power": int(np.sum(np.isclose(physical, asset.p_up_kw) | np.isclose(physical, -asset.p_down_kw))),
            "state": int(np.sum(np.isclose(x, asset.state_min_kwh) | np.isclose(x, asset.state_max_kwh))),
            "grid": int(np.sum(np.isclose(imp, config.import_limit_kw))) if config.import_limit_kw is not None else 0,
        },
        nonanticipativity_max_residual_kw=_nonanticipativity_residual(r, nodes),
        runtime_seconds=time.perf_counter() - started,
        configuration_hash=configuration_hash,
        metadata={
            "scenario_count": scenario_count,
            "horizon": horizon,
            "physical_hours": float(dt.sum()),
            "commitment_intervals": config.commitment_intervals,
            "recourse_start_interval": config.recourse_start_interval,
            "genuine_recourse": True,
            "first_stage_shared": True,
            "scenario_tree_nonanticipativity": True,
            "risk_lambda": config.risk_lambda,
            "cvar_alpha": config.cvar_alpha,
        },
    )


def solve_deterministic_fallback(
    decision: RollingDecisionInput,
    asset: ShadowAsset,
    commercial: CommercialTerms,
    config: TwoStageConfig,
) -> TwoStageResult:
    horizon = len(decision.timestamps_utc)
    fallback_decision = replace(
        decision,
        scenario_matrix_kw=(tuple(decision.point_forecast_kw),),
        scenario_weights=(1.0,),
        scenario_node_ids=(tuple("deterministic" for _ in range(horizon)),),
    )
    return solve_two_stage(fallback_decision, asset, commercial, replace(config, adjustment_limit_kw=0.0))
