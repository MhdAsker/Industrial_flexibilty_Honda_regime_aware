from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.honda.optimization.audit import audit_f1, schedules_are_nonanticipative
from src.honda.optimization.contracts import DecisionInput, F1Config, F2Config, F3Config, TariffConfig
from src.honda.optimization.optimizer import settle_schedule, solve_f1
from src.honda.optimization.legacy import solve_f0
from src.honda.optimization.scenario_adapter import load_h7_day
from src.honda.optimization.thermal import solve_f2
from src.honda.optimization.timing import observation_available, validate_price_timing


ROOT = Path(__file__).resolve().parents[2]
SPARK = Path(r"C:\Users\asker\Documents\GenAIProjects\Industrial_energy_forecasting")
H7_COMMIT = "bb0536df77f709a20452ed3c86ab16e66a5e0555"
H7_CONFIG_HASH = "790d3574d468f10fbc565bef1ed0eb492d45ff9db00436f2700a74b7069171ba"


def _decision(
    point=(10.0, 10.0), scenarios=None, weights=None, horizon=None,
    classification="EX_POST_PRICE_AWARE",
) -> DecisionInput:
    if horizon is not None:
        point = tuple(np.full(horizon, 10.0))
    point = tuple(point)
    scenarios = (point,) if scenarios is None else tuple(tuple(x) for x in scenarios)
    weights = tuple(np.full(len(scenarios), 1 / len(scenarios))) if weights is None else tuple(weights)
    stamps = tuple(pd.date_range("2025-01-01", periods=len(point), freq="15min", tz="UTC").astype(str))
    return DecisionInput(
        dataset_id="SYNTHETIC", target_day_local="2025-01-01", timestamp_utc=stamps,
        interval_hours=tuple(np.full(len(point), .25)), deterministic_forecast_kw=point,
        scenario_matrix_kw=scenarios, scenario_weights=weights,
        forecast_issue_time_utc="2024-12-31T12:00:00+00:00", decision_time_utc="2024-12-31T13:15:00+00:00",
        price_publication_time_utc=None, timing_classification=classification,
    )


def _f1(**kwargs) -> F1Config:
    base = dict(p_up_kw=5, p_down_kw=5, e_advance_kwh=2, e_backlog_kwh=2)
    base.update(kwargs)
    return F1Config(**base)


def _grid(**kwargs) -> F3Config:
    base = dict(grid_limit_mode="scenario_wise")
    base.update(kwargs)
    return F3Config(**base)


def _tariff(import_prices=(100, 100), export_prices=None) -> TariffConfig:
    return TariffConfig.from_eur_per_mwh(import_prices, export_prices)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_h7_checkpoint_is_ancestor_and_config_preserved():
    subprocess.run(["git", "merge-base", "--is-ancestor", H7_COMMIT, "HEAD"], cwd=ROOT, check=True)
    assert _sha(ROOT / "config/scenarios/scenario_method_frozen.json") == H7_CONFIG_HASH


def test_h1_h6_protected_artifacts_unchanged():
    changed = subprocess.check_output(["git", "diff", "--name-only", H7_COMMIT, "--", "reports", "config", "results"], cwd=ROOT, text=True)
    # H7 remains protected.  Later committed work is deliberately isolated in
    # Stage 2, Stage 3, and the separately named Stage 3 replication namespaces.
    allowed = (
        "reports/optimization/", "config/optimization/", "results/optimization/",
        "reports/decision_value/", "config/decision_value/", "results/decision_value/",
        "config/decision_value_replication/", "results/decision_value_replication/",
        "reports/paper_overleaf/", "reports/progress_overleaf/",
        "reports/progress_overleaf_upload.zip", "reports/research_checkpoint/",
    )
    assert all(x.startswith(allowed) for x in changed.splitlines())


def test_spark_repository_is_read_only_and_clean():
    assert subprocess.check_output(["git", "-C", str(SPARK), "status", "--porcelain"], text=True) == ""
    assert subprocess.check_output(["git", "-C", str(SPARK), "rev-parse", "HEAD"], text=True).strip() == "32cff65f7f5384d08095d8d0325c45e8ce002ec3"


def test_f1_state_transition_and_terminal_recovery():
    result = solve_f1(_decision(), _f1(), _grid(), _tariff((200, 50)))
    assert result.solver_status == "optimal"
    np.testing.assert_allclose(result.state_kwh[1:] - result.state_kwh[:-1], .25 * result.schedule_kw, atol=1e-8)
    assert result.state_kwh[-1] == pytest.approx(0, abs=1e-8)


def test_f0_legacy_conserves_positive_spark_energy():
    result = solve_f0(np.array([10., 10.]), np.full(2, .25), _tariff((200, 0)), flexible_fraction=.2, shift_window_intervals=1)
    assert result.solver_status == "optimal"
    assert np.sum(result.schedule_kw * .25) == pytest.approx(0, abs=1e-8)
    assert result.schedule_kw[0] < 0 < result.schedule_kw[1]


def test_f0_rejects_signed_honda_exchange():
    with pytest.raises(ValueError, match="disabled"):
        solve_f0(np.array([10., -1.]), np.full(2, .25), _tariff((100, 100)))


def test_f1_power_and_energy_limits():
    result = solve_f1(_decision(), _f1(p_up_kw=2, p_down_kw=3, e_advance_kwh=.25, e_backlog_kwh=.25), _grid(), _tariff((200, 0)))
    assert np.max(result.schedule_kw) <= 2 + 1e-8
    assert np.min(result.schedule_kw) >= -3 - 1e-8
    assert np.max(result.state_kwh) <= .25 + 1e-8
    assert np.min(result.state_kwh) >= -.25 - 1e-8


def test_f1_deferral_only_never_advances_state():
    result = solve_f1(_decision(), _f1(operating_mode="deferral_only"), _grid(), _tariff((200, 50)))
    assert np.max(result.state_kwh) <= 1e-8
    assert result.schedule_kw[0] < 0 < result.schedule_kw[1]


def test_f1_advancement_mode_can_advance():
    result = solve_f1(_decision(), _f1(), _grid(), _tariff((50, 200)))
    assert np.max(result.state_kwh) > 0
    assert result.schedule_kw[0] > 0 > result.schedule_kw[1]


def test_exact_flow_deadline_enforced():
    result = solve_f1(_decision(), _f1(recovery_mode="exact_flow_deadline", recovery_horizon_intervals=1), _grid(), _tariff((200, 0)))
    assert result.metadata["deadline_flow_variables"] > 0
    assert result.schedule_kw.sum() == pytest.approx(0, abs=1e-8)


def test_invalid_recovery_horizon_rejected():
    with pytest.raises(ValueError):
        _f1(recovery_mode="exact_flow_deadline").validate()


def test_f2_thermal_balance_and_cop_conversion():
    cfg = F2Config(10, 10, cop=2, e_initial_kwh_th=1, e_terminal_kwh_th=1)
    result = solve_f2(np.array([10., 10.]), np.array([2., 2.]), np.array([4., 4.]), np.array([.25, .25]), cfg, _grid(), _tariff())
    assert result.solver_status == "optimal"
    assert result.feasibility["max_thermal_balance_residual_kwh_th"] < 1e-8
    assert result.metadata["thermal_input_kw"].sum() == pytest.approx(4, abs=1e-8)


def test_f2_state_bounds_and_terminal_requirement():
    cfg = F2Config(5, 1, cop=2, e_initial_kwh_th=.5, e_terminal_kwh_th=.5)
    result = solve_f2(np.array([10., 10.]), np.array([2., 2.]), np.array([4., 4.]), np.array([.25, .25]), cfg, _grid(), _tariff())
    assert np.all((result.state_kwh >= -1e-8) & (result.state_kwh <= 1 + 1e-8))
    assert result.state_kwh[-1] == pytest.approx(.5, abs=1e-8)


def test_f2_infeasible_service_is_reported_not_filled():
    cfg = F2Config(0, 0, cop=2)
    result = solve_f2(np.array([10., 10.]), np.zeros(2), np.ones(2), np.full(2, .25), cfg, _grid(), _tariff())
    assert result.solver_status != "optimal"
    assert result.feasibility["infeasible"] == 1


def test_f3_import_cap():
    result = solve_f1(_decision(point=(10, 0)), _f1(), _grid(import_max_kw=8), _tariff())
    assert result.solver_status == "optimal"
    assert np.max(result.scenario_import_kw) <= 8 + 1e-8


def test_f3_export_cap_and_signed_grid_exchange():
    result = solve_f1(_decision(point=(-10, 0)), _f1(), _grid(export_max_kw=8), _tariff())
    assert result.solver_status == "optimal"
    assert np.max(result.scenario_export_kw) <= 8 + 1e-8
    assert np.any(result.scenario_export_kw > 0)


def test_peak_penalty_reduces_peak_when_feasible():
    no_peak = solve_f1(_decision(point=(10, 0)), _f1(), _grid(), _tariff())
    peak = solve_f1(_decision(point=(10, 0)), _f1(), _grid(peak_penalty_eur_per_kw_per_day=1), _tariff())
    assert peak.peak_kw[0] <= no_peak.peak_kw[0] + 1e-8
    assert peak.peak_kw[0] == pytest.approx(5, abs=1e-7)


def test_nominal_grid_cap_mode():
    result = solve_f1(_decision(point=(10, 0)), _f1(), _grid(import_max_kw=8, grid_limit_mode="nominal"), _tariff())
    assert result.solver_status == "optimal"
    assert np.max(np.array([10., 0.]) + result.schedule_kw) <= 8 + 1e-8


def test_chance_approximation_grid_cap_mode():
    decision = _decision(point=(10, 0), scenarios=((10, 0), (12, 0)))
    result = solve_f1(decision, _f1(), _grid(import_max_kw=10, grid_limit_mode="chance_approximation", chance_quantile=.9), _tariff())
    assert result.solver_status == "optimal"
    q = np.quantile(np.array([[10., 0.], [12., 0.]]), .9, axis=0)
    assert np.max(q + result.schedule_kw) <= 10 + 1e-8


def test_soft_grid_cap_records_violation_without_infeasibility():
    result = solve_f1(_decision(point=(20, 20)), _f1(p_up_kw=0, p_down_kw=0), _grid(import_max_kw=10, grid_limit_mode="soft", soft_violation_penalty_eur_per_kw_interval=1), _tariff())
    assert result.solver_status == "optimal"
    assert result.objective_eur > 1.0


def test_ramp_limit_is_enforced():
    result = solve_f1(_decision(), _f1(ramp_limit_kw_per_interval=1), _grid(), _tariff((200, 0)))
    assert abs(np.diff(result.schedule_kw)[0]) <= 1 + 1e-8


def test_no_honda_pv_chp_double_counting_interface():
    fields = DecisionInput.__dataclass_fields__
    assert "pv" not in fields and "chp" not in fields


def test_import_export_cost_accounting_and_no_simultaneous_trade():
    result = solve_f1(_decision(point=(10, -10)), _f1(p_up_kw=0, p_down_kw=0), _grid(), _tariff((100, 100), (50, 50)))
    assert result.scenario_cost_eur[0] == pytest.approx(.25 * (10 * .1 - 10 * .05))
    assert np.max(np.minimum(result.scenario_import_kw, result.scenario_export_kw)) < 1e-8


def test_arbitrage_tariff_rejected():
    with pytest.raises(ValueError, match="artificial arbitrage"):
        solve_f1(_decision(), _f1(), _grid(), _tariff((0, 0), (100, 100)))


def test_negative_price_handling_moves_energy_to_negative_interval():
    result = solve_f1(_decision(), _f1(), _grid(), _tariff((100, -100), (100, -100)))
    assert result.schedule_kw[0] < 0 < result.schedule_kw[1]


def test_eur_mwh_conversion():
    buy, sell = TariffConfig.from_eur_per_mwh([100], [50]).arrays(1)
    assert buy[0] == pytest.approx(.1) and sell[0] == pytest.approx(.05)


def test_flat_price_cost_conservation():
    decision = _decision()
    result = solve_f1(decision, _f1(), _grid(), _tariff((100, 100)))
    settled = settle_schedule(result.schedule_kw, np.array([10., 10.]), np.full(2, .25), _tariff((100, 100)))
    assert settled["cost_eur"] == pytest.approx(.5, abs=1e-8)


def test_two_interval_analytic_optimum():
    result = solve_f1(_decision(), _f1(e_backlog_kwh=1.25, e_advance_kwh=1.25), _grid(), _tariff((200, 0)))
    np.testing.assert_allclose(result.schedule_kw, [-5, 5], atol=1e-7)
    assert result.objective_eur == pytest.approx(.25 * 5 * .2, abs=1e-8)


def test_observation_completion_cutoff():
    assert observation_available("2025-01-01T12:00:00+00:00", "2025-01-01T12:00:00+00:00")
    assert not observation_available("2025-01-01T12:15:00+00:00", "2025-01-01T12:00:00+00:00")


def test_price_publication_timing_guard():
    with pytest.raises(ValueError):
        validate_price_timing("2025-01-01T11:00:00+00:00", "2025-01-01T12:00:00+00:00", "2025-01-01T12:30:00+00:00", "OPERATIONAL_TIMING_VERIFIED")


def test_ex_post_timing_allows_unknown_publication_but_not_early_decision():
    validate_price_timing("2025-01-01T11:00:00+00:00", "2025-01-01T12:00:00+00:00", None, "EX_POST_PRICE_AWARE")
    with pytest.raises(ValueError):
        validate_price_timing("2025-01-01T12:00:00+00:00", "2025-01-01T11:00:00+00:00", None, "EX_POST_PRICE_AWARE")


@pytest.mark.parametrize("horizon", [92, 96, 100])
def test_physical_dst_horizons(horizon):
    result = solve_f1(_decision(horizon=horizon), _f1(p_up_kw=0, p_down_kw=0), _grid(), TariffConfig.from_eur_per_mwh(np.full(horizon, 100)))
    assert result.solver_status == "optimal" and len(result.schedule_kw) == horizon


def test_scenario_weight_normalization_rejected():
    with pytest.raises(ValueError):
        _decision(scenarios=((10, 10), (10, 10)), weights=(.8, .3)).arrays()


def test_shared_first_stage_schedule_and_nonanticipativity():
    decision = _decision(scenarios=((20, 0), (0, 20)))
    result = solve_f1(decision, _f1(), _grid(), _tariff((200, 0)))
    assert schedules_are_nonanticipative(result, 2)
    assert result.schedule_kw.shape == (2,)


def test_h7_scenario_interface_loads_complete_physical_day():
    path = ROOT / "results/scenarios/honda_2023_diagnostic_scenarios.parquet"
    first = pd.read_parquet(path, columns=["target_day_local", "forecast_issue_time_utc"]).iloc[0]
    issue = pd.Timestamp(first.forecast_issue_time_utc)
    loaded = load_h7_day(
        path, str(first.target_day_local), decision_time_utc=str(issue + pd.Timedelta(hours=1)),
        price_publication_time_utc=None, timing_classification="EX_POST_PRICE_AWARE",
    )
    point, scenarios, weights, dt = loaded.arrays()
    assert len(point) in {92, 96, 100}
    assert scenarios.shape[1] == len(point) and np.isclose(weights.sum(), 1)


def test_implementable_decision_input_has_no_realized_target_field():
    assert "realized_target" not in DecisionInput.__dataclass_fields__


def test_deterministic_stochastic_equivalence_identical_scenarios():
    one = solve_f1(_decision(), _f1(), _grid(), _tariff((200, 0)))
    two = solve_f1(_decision(scenarios=((10, 10), (10, 10))), _f1(), _grid(), _tariff((200, 0)))
    np.testing.assert_allclose(one.schedule_kw, two.schedule_kw, atol=1e-8)
    assert one.objective_eur == pytest.approx(two.objective_eur, abs=1e-8)


def test_cvar_fixture_matches_empirical_distribution():
    decision = _decision(point=(10, 10), scenarios=((10, 10), (30, 30)), weights=(.9, .1))
    result = solve_f1(decision, _f1(p_up_kw=0, p_down_kw=0), _grid(), _tariff(), risk_lambda=1, cvar_alpha=.9)
    assert result.cvar_eur == pytest.approx(1.5, abs=1e-8)
    assert result.objective_eur == pytest.approx(.6 + 1.5, abs=1e-8)


def test_infeasible_cap_is_reported():
    result = solve_f1(_decision(point=(20, 20)), _f1(p_up_kw=0, p_down_kw=0), _grid(import_max_kw=10), _tariff())
    assert result.solver_status != "optimal" and result.objective_eur is None


def test_independent_feasibility_audit():
    decision = _decision(scenarios=((10, 10), (12, 8)))
    flex = _f1(); grid = _grid()
    result = solve_f1(decision, flex, grid, _tariff((200, 0)))
    assert audit_f1(decision, flex, grid, result)["passed"]


def test_settlement_never_reoptimizes():
    result = settle_schedule(np.array([-1., 1.]), np.array([10., 10.]), np.full(2, .25), _tariff())
    assert result["reoptimized"] is False
    np.testing.assert_allclose(result["grid_kw"], [9, 11])


def test_fixed_solver_is_reproducible():
    args = (_decision(scenarios=((10, 10), (12, 8))), _f1(), _grid(), _tariff((200, 0)))
    first = solve_f1(*args); second = solve_f1(*args)
    np.testing.assert_array_equal(first.schedule_kw, second.schedule_kw)
    assert first.objective_eur == second.objective_eur


def test_no_full_historical_economic_outputs_or_oracle_rankings():
    prohibited = ["honda_2023_economic_ranking", "spark_2024_economic_ranking", "economic_winner"]
    for name in prohibited:
        assert not any((ROOT / "results/optimization").glob(f"*{name}*"))


def test_required_stage2_configs_are_valid_json():
    for name in ["flexibility_archetypes", "market_timing", "tariff_scenarios", "solver_settings", "stage3_experiment_design"]:
        path = ROOT / f"config/optimization/{name}.json"
        assert path.exists()
        json.loads(path.read_text(encoding="utf-8"))
