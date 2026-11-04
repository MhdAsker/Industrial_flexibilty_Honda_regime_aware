from __future__ import annotations

import numpy as np

from .contracts import DecisionInput, F1Config, F3Config, OptimizationResult


def audit_f1(
    decision: DecisionInput,
    flexibility: F1Config,
    grid: F3Config,
    result: OptimizationResult,
    tolerance: float = 1e-7,
) -> dict[str, float | bool]:
    point, scenarios, weights, dt = decision.arrays()
    if result.solver_status != "optimal":
        return {"passed": False, "solver_optimal": False}
    schedule = result.schedule_kw; state = result.state_kwh
    grid_values = scenarios + schedule[None, :]
    import_expected = np.maximum(grid_values, 0)
    export_expected = np.maximum(-grid_values, 0)
    metrics: dict[str, float | bool] = {
        "solver_optimal": True,
        "scenario_weights_residual": float(abs(weights.sum() - 1)),
        "state_transition_residual": float(np.max(np.abs(state[1:] - state[:-1] - schedule * dt))),
        "terminal_residual": float(abs(state[-1] - flexibility.x_terminal_kwh)),
        "power_upper_violation": float(max(np.max(schedule - flexibility.p_up_kw), 0)),
        "power_lower_violation": float(max(np.max(-flexibility.p_down_kw - schedule), 0)),
        "state_upper_violation": float(max(np.max(state - (0 if flexibility.operating_mode == "deferral_only" else flexibility.e_advance_kwh)), 0)),
        "state_lower_violation": float(max(np.max(-flexibility.e_backlog_kwh - state), 0)),
        "import_definition_residual": float(np.max(np.abs(result.scenario_import_kw - import_expected))),
        "export_definition_residual": float(np.max(np.abs(result.scenario_export_kw - export_expected))),
        "shared_first_stage_schedule": bool(result.metadata.get("shared_first_stage_schedule")),
    }
    if grid.grid_limit_mode == "scenario_wise" and grid.import_max_kw is not None:
        metrics["import_cap_violation"] = float(max(np.max(grid_values - grid.import_max_kw), 0))
    if grid.grid_limit_mode == "scenario_wise" and grid.export_max_kw is not None:
        metrics["export_cap_violation"] = float(max(np.max(-grid.export_max_kw - grid_values), 0))
    numeric = [float(v) for k, v in metrics.items() if k not in {"solver_optimal", "shared_first_stage_schedule"}]
    metrics["passed"] = bool(max(numeric, default=0) <= tolerance and metrics["shared_first_stage_schedule"])
    return metrics


def schedules_are_nonanticipative(result: OptimizationResult, scenario_count: int) -> bool:
    return bool(
        result.metadata.get("shared_first_stage_schedule") is True
        and result.scenario_import_kw.shape[0] == scenario_count
        and result.schedule_kw.ndim == 1
    )
