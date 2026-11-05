from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from src.honda.optimization.contracts import DecisionInput, F1Config, F3Config, OptimizationResult, TariffConfig
from src.honda.optimization.optimizer import settle_schedule, solve_f1

from .data import DecisionDay, SettlementDay


@dataclass(frozen=True)
class Commitment:
    dataset_id: str
    target_day_local: str
    policy: str
    schedule_kw: tuple[float, ...]
    state_kwh: tuple[float, ...]
    predicted_objective_eur: float | None
    solver_status: str
    solver_status_code: int
    runtime_seconds: float
    configuration_hash: str
    scenario_hash: str
    price_hash: str
    scenario_count: int
    archetype: str


def export_price_from_multiplier(import_price_eur_per_kwh: np.ndarray, multiplier: float) -> np.ndarray:
    if not 0 <= multiplier <= 1:
        raise ValueError("Export multiplier must be in [0,1]")
    # Remuneration is reduced for positive prices and made no more favorable
    # than import for negative prices, preserving convex no-arbitrage dominance.
    return import_price_eur_per_kwh - (1 - multiplier) * np.abs(import_price_eur_per_kwh)


def tariff_for_day(day: DecisionDay, export_multiplier: float = 1.0) -> TariffConfig:
    buy = np.asarray(day.price_eur_per_mwh, dtype=float) / 1000.0
    sell = export_price_from_multiplier(buy, export_multiplier)
    return TariffConfig(tuple(buy), tuple(sell), f"HYPOTHETICAL_MKT_EXPORT_MULTIPLIER_{export_multiplier:g}")


def _input(day: DecisionDay, scenarios: np.ndarray, weights: np.ndarray) -> DecisionInput:
    return DecisionInput(
        dataset_id=day.dataset_id, target_day_local=day.target_day_local, timestamp_utc=day.timestamp_utc,
        interval_hours=tuple(np.full(len(day.timestamp_utc), .25)), deterministic_forecast_kw=day.point_forecast_kw,
        scenario_matrix_kw=tuple(map(tuple, scenarios)), scenario_weights=tuple(weights),
        forecast_issue_time_utc=day.forecast_issue_time_utc, decision_time_utc=day.decision_time_utc,
        price_publication_time_utc=None, timing_classification="EX_POST_PRICE_AWARE",
        provenance_hashes={"scenario": day.scenario_config_sha256, "price": day.price_source_sha256},
    )


def optimize_policy(
    day: DecisionDay,
    policy: str,
    flexibility: F1Config,
    grid: F3Config,
    *,
    export_multiplier: float = 1.0,
    risk_lambda: float = 0.0,
    cvar_alpha: float = 0.9,
    oracle_actual_kw: np.ndarray | None = None,
) -> tuple[Commitment, OptimizationResult | None]:
    started = time.perf_counter()
    horizon = len(day.timestamp_utc)
    if policy == "P0_NO_FLEX":
        commitment = Commitment(day.dataset_id, day.target_day_local, policy, tuple(np.zeros(horizon)), tuple(np.zeros(horizon + 1)), None, "not_required", 0, time.perf_counter() - started, "NO_FLEX", day.scenario_config_sha256, day.price_source_sha256, 0, "F1")
        return commitment, None
    point = np.asarray(day.point_forecast_kw, dtype=float)
    if policy == "P2_LGBM_DETERMINISTIC":
        scenarios = point[None, :]; weights = np.ones(1)
    elif policy == "P3_S1_INDEPENDENT":
        scenarios = np.asarray(day.s1_scenarios_kw, dtype=float); weights = np.asarray(day.scenario_weights, dtype=float)
    elif policy in {"P4_S2_COHERENT", "P6_S2_CVAR"}:
        scenarios = np.asarray(day.s2_scenarios_kw, dtype=float); weights = np.asarray(day.scenario_weights, dtype=float)
    elif policy == "P7_ORACLE_UPPER_BOUND":
        if oracle_actual_kw is None:
            raise ValueError("Oracle policy requires explicitly labelled settlement-only actuals")
        scenarios = np.asarray(oracle_actual_kw, dtype=float)[None, :]; weights = np.ones(1)
    else:
        raise ValueError(f"Unsupported policy {policy}")
    result = solve_f1(
        _input(day, scenarios, weights), flexibility, grid, tariff_for_day(day, export_multiplier),
        risk_lambda=risk_lambda if policy == "P6_S2_CVAR" else 0.0, cvar_alpha=cvar_alpha,
    )
    commitment = Commitment(
        day.dataset_id, day.target_day_local, policy, tuple(result.schedule_kw), tuple(result.state_kwh),
        result.objective_eur, result.solver_status, result.solver_status_code, result.runtime_seconds,
        result.configuration_hash, day.scenario_config_sha256, day.price_source_sha256,
        len(scenarios), flexibility.label,
    )
    return commitment, result


def settle_commitment(
    commitment: Commitment,
    settlement: SettlementDay,
    day: DecisionDay,
    flexibility: F1Config,
    grid: F3Config,
    *,
    export_multiplier: float = 1.0,
) -> dict:
    if commitment.target_day_local != settlement.target_day_local:
        raise ValueError("Commitment/settlement mismatch")
    schedule = np.asarray(commitment.schedule_kw, dtype=float)
    actual = np.asarray(settlement.actual_kw, dtype=float)
    tariff = tariff_for_day(day, export_multiplier)
    settled = settle_schedule(schedule, actual, np.full(len(actual), .25), tariff)
    import_kw = np.asarray(settled["import_kw"]); export_kw = np.asarray(settled["export_kw"])
    peak_cost = grid.peak_penalty_eur_per_kw_per_day * float(import_kw.max())
    realized_cost = float(settled["cost_eur"] + peak_cost)
    no_flex = settle_schedule(np.zeros(len(actual)), actual, np.full(len(actual), .25), tariff)
    no_flex_cost = float(no_flex["cost_eur"] + grid.peak_penalty_eur_per_kw_per_day * np.max(no_flex["import_kw"]))
    import_violation = np.maximum(import_kw - grid.import_max_kw, 0) if grid.import_max_kw is not None else np.zeros_like(import_kw)
    export_violation = np.maximum(export_kw - grid.export_max_kw, 0) if grid.export_max_kw is not None else np.zeros_like(export_kw)
    state = np.asarray(commitment.state_kwh, dtype=float)
    shifted = float(.5 * np.sum(np.abs(schedule)) * .25)
    capacity = max(flexibility.e_advance_kwh, flexibility.e_backlog_kwh, 1e-12)
    return {
        "dataset_id": settlement.dataset_id, "target_day_local": settlement.target_day_local,
        "policy": commitment.policy, "solver_status": commitment.solver_status,
        "realized_cost_eur": realized_cost, "no_flex_cost_eur": no_flex_cost,
        "savings_eur": no_flex_cost - realized_cost,
        "peak_import_kw": float(import_kw.max()), "maximum_export_kw": float(export_kw.max()),
        "shifted_energy_kwh": shifted, "backlog_utilization": float(np.max(np.abs(state)) / capacity),
        "terminal_residual_kwh": float(abs(state[-1] - flexibility.x_terminal_kwh)),
        "import_violation_intervals": int(np.sum(import_violation > 1e-7)),
        "import_violation_max_kw": float(import_violation.max()), "import_violation_energy_kwh": float(import_violation.sum() * .25),
        "export_violation_intervals": int(np.sum(export_violation > 1e-7)),
        "export_violation_max_kw": float(export_violation.max()), "export_violation_energy_kwh": float(export_violation.sum() * .25),
        "realized_grid_cap_compliant": bool(np.max(import_violation) <= 1e-7 and np.max(export_violation) <= 1e-7),
        "runtime_seconds": commitment.runtime_seconds, "scenario_count": commitment.scenario_count,
        "predicted_objective_eur": commitment.predicted_objective_eur,
        "is_weekend": settlement.is_weekend, "is_public_holiday": settlement.is_public_holiday,
        "month": settlement.month, "actual_mean_kw": float(actual.mean()), "actual_peak_kw": float(actual.max()),
        "price_std_eur_per_mwh": float(np.std(day.price_eur_per_mwh)),
        "price_range_eur_per_mwh": float(np.ptp(day.price_eur_per_mwh)),
        "timing_classification": "EX_POST_PRICE_AWARE", "evidence_class": "PREVIOUSLY_EXPOSED_HOLDOUT_DIAGNOSTIC",
    }


def evaluate_policy(
    day: DecisionDay,
    settlement: SettlementDay,
    policy: str,
    flexibility: F1Config,
    grid: F3Config,
    **kwargs,
) -> tuple[Commitment, dict]:
    oracle_actual = np.asarray(settlement.actual_kw) if policy == "P7_ORACLE_UPPER_BOUND" else None
    commitment, _ = optimize_policy(day, policy, flexibility, grid, oracle_actual_kw=oracle_actual, **kwargs)
    if commitment.solver_status not in {"optimal", "not_required"}:
        return commitment, {
            "dataset_id": settlement.dataset_id, "target_day_local": settlement.target_day_local,
            "policy": policy, "solver_status": commitment.solver_status,
            "realized_cost_eur": np.nan, "no_flex_cost_eur": np.nan, "savings_eur": np.nan,
            "runtime_seconds": commitment.runtime_seconds, "scenario_count": commitment.scenario_count,
            "timing_classification": "EX_POST_PRICE_AWARE", "evidence_class": "PREVIOUSLY_EXPOSED_HOLDOUT_DIAGNOSTIC",
        }
    metrics = settle_commitment(commitment, settlement, day, flexibility, grid, export_multiplier=kwargs.get("export_multiplier", 1.0))
    return commitment, metrics
