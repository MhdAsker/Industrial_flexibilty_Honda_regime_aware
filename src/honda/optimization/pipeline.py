from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from .audit import audit_f1, schedules_are_nonanticipative
from .contracts import DecisionInput, F1Config, F2Config, F3Config, TariffConfig
from .optimizer import settle_schedule, solve_f1
from .legacy import solve_f0
from .thermal import solve_f2


ROOT = Path(__file__).resolve().parents[3]
SPARK = Path(r"C:\Users\asker\Documents\GenAIProjects\Industrial_energy_forecasting")
CONFIG = ROOT / "config/optimization"
OUT = ROOT / "results/optimization"
REPORT = ROOT / "reports/optimization"
FIG = OUT / "figures"
START_COMMIT = "bb0536df77f709a20452ed3c86ab16e66a5e0555"
SEED = 20261029


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


def _decision(point, scenarios=None, weights=None, *, label="SYNTHETIC", horizon=None) -> DecisionInput:
    if horizon is not None:
        point = np.full(horizon, 10.0)
    point = np.asarray(point, dtype=float)
    scenarios = point[None, :] if scenarios is None else np.asarray(scenarios, dtype=float)
    weights = np.full(len(scenarios), 1 / len(scenarios)) if weights is None else np.asarray(weights, dtype=float)
    stamps = pd.date_range("2025-01-01", periods=len(point), freq="15min", tz="UTC").astype(str)
    return DecisionInput(
        dataset_id=label, target_day_local="2025-01-01", timestamp_utc=tuple(stamps),
        interval_hours=tuple(np.full(len(point), .25)), deterministic_forecast_kw=tuple(point),
        scenario_matrix_kw=tuple(map(tuple, scenarios)), scenario_weights=tuple(weights),
        forecast_issue_time_utc="2024-12-31T12:00:00+00:00", decision_time_utc="2024-12-31T13:15:00+00:00",
        price_publication_time_utc=None, timing_classification="EX_POST_PRICE_AWARE",
        provenance_hashes={"fixture": label},
    )


def _tariff(buy_mwh, sell_mwh=None) -> TariffConfig:
    return TariffConfig.from_eur_per_mwh(np.asarray(buy_mwh, dtype=float), None if sell_mwh is None else np.asarray(sell_mwh, dtype=float))


def _base_f1(**updates) -> F1Config:
    values = dict(p_up_kw=5, p_down_kw=5, e_advance_kwh=1.25, e_backlog_kwh=1.25)
    values.update(updates)
    return F1Config(**values)


def synthetic_validations() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows = []
    audits = []
    stored: dict[str, object] = {}

    legacy = solve_f0(np.array([10., 10.]), np.full(2, .25), _tariff([200, 0]), flexible_fraction=.2, shift_window_intervals=1)
    rows.append({"fixture": "spark_f0_legacy", "archetype": "F0", "solver_status": legacy.solver_status, "objective_eur": legacy.objective_eur, "baseline_cost_eur": .5, "schedule_kw": json.dumps(np.round(legacy.schedule_kw, 8).tolist()), "terminal_state_kwh": legacy.state_kwh[-1], "analytic_check_passed": legacy.feasibility["daily_energy_residual_kwh"] < 1e-7})
    audits.append({"fixture": "spark_f0_legacy", "passed": legacy.feasibility["daily_energy_residual_kwh"] < 1e-7, **legacy.feasibility})

    fixtures = [
        ("flat_positive", [10, 10], [100, 100], None, _base_f1(), F3Config()),
        ("low_to_high", [10, 10], [50, 200], None, _base_f1(), F3Config()),
        ("high_to_low", [10, 10], [200, 50], None, _base_f1(), F3Config()),
        ("negative_price", [10, 10], [100, -100], [100, -100], _base_f1(), F3Config()),
        ("zero_price", [10, 10], [100, 0], None, _base_f1(), F3Config()),
        ("separate_buy_sell", [10, -10], [100, 100], [50, 50], _base_f1(p_up_kw=0, p_down_kw=0), F3Config()),
        ("export_dominant", [-10, 0], [100, 100], [50, 50], _base_f1(), F3Config(export_max_kw=8)),
        ("import_constrained", [10, 0], [100, 100], None, _base_f1(), F3Config(import_max_kw=8)),
        ("zero_flexibility", [10, 10], [200, 0], None, _base_f1(p_up_kw=0, p_down_kw=0), F3Config()),
        ("large_power_small_energy", [10, 10], [200, 0], None, _base_f1(p_up_kw=100, p_down_kw=100, e_advance_kwh=.25, e_backlog_kwh=.25), F3Config()),
        ("small_power_large_energy", [10, 10], [200, 0], None, _base_f1(p_up_kw=1, p_down_kw=1, e_advance_kwh=100, e_backlog_kwh=100), F3Config()),
        ("exact_recovery_deadline", [10, 10], [200, 0], None, _base_f1(recovery_mode="exact_flow_deadline", recovery_horizon_intervals=1), F3Config()),
    ]
    for name, point, buy, sell, flex, grid in fixtures:
        decision = _decision(point, label=name)
        result = solve_f1(decision, flex, grid, _tariff(buy, sell))
        audit = audit_f1(decision, flex, grid, result)
        baseline = settle_schedule(np.zeros(len(point)), np.asarray(point), np.full(len(point), .25), _tariff(buy, sell))
        rows.append({
            "fixture": name, "archetype": "F1", "solver_status": result.solver_status,
            "objective_eur": result.objective_eur, "baseline_cost_eur": baseline["cost_eur"],
            "schedule_kw": json.dumps(np.round(result.schedule_kw, 8).tolist()),
            "terminal_state_kwh": result.state_kwh[-1] if result.solver_status == "optimal" else np.nan,
            "analytic_check_passed": bool(audit.get("passed", False)),
        })
        audits.append({"fixture": name, **audit})
        stored[name] = result

    for horizon in [92, 96, 100]:
        decision = _decision([], horizon=horizon, label=f"DST_{horizon}")
        result = solve_f1(decision, _base_f1(p_up_kw=0, p_down_kw=0), F3Config(), _tariff(np.full(horizon, 100)))
        audit = audit_f1(decision, _base_f1(p_up_kw=0, p_down_kw=0), F3Config(), result)
        rows.append({"fixture": f"physical_day_{horizon}", "archetype": "F1", "solver_status": result.solver_status, "objective_eur": result.objective_eur, "baseline_cost_eur": result.objective_eur, "schedule_kw": "zero", "terminal_state_kwh": result.state_kwh[-1], "analytic_check_passed": audit["passed"]})
        audits.append({"fixture": f"physical_day_{horizon}", **audit})

    impossible = solve_f1(_decision([20, 20], label="infeasible_cap"), _base_f1(p_up_kw=0, p_down_kw=0), F3Config(import_max_kw=10), _tariff([100, 100]))
    rows.append({"fixture": "infeasible_cap", "archetype": "F1", "solver_status": impossible.solver_status, "objective_eur": np.nan, "baseline_cost_eur": 1.0, "schedule_kw": "UNAVAILABLE", "terminal_state_kwh": np.nan, "analytic_check_passed": impossible.solver_status != "optimal"})
    audits.append({"fixture": "infeasible_cap", "passed": impossible.solver_status != "optimal", "solver_optimal": False, "expected_infeasible": True})

    thermal_cfg = F2Config(10, 10, 2.0, e_initial_kwh_th=1, e_terminal_kwh_th=1)
    thermal = solve_f2(np.array([10., 10.]), np.array([2., 2.]), np.array([4., 4.]), np.full(2, .25), thermal_cfg, F3Config(), _tariff([200, 0]))
    rows.append({"fixture": "thermal_cop_balance", "archetype": "F2", "solver_status": thermal.solver_status, "objective_eur": thermal.objective_eur, "baseline_cost_eur": .5, "schedule_kw": json.dumps(np.round(thermal.schedule_kw, 8).tolist()), "terminal_state_kwh": thermal.state_kwh[-1], "analytic_check_passed": max(thermal.feasibility.values()) < 1e-7})
    audits.append({"fixture": "thermal_cop_balance", "passed": max(thermal.feasibility.values()) < 1e-7, **thermal.feasibility})
    stored["thermal"] = thermal

    scenarios = np.array([[20., 0.], [0., 20.]])
    decision = _decision([10, 10], scenarios=scenarios, weights=[.5, .5], label="stochastic_nonanticipativity")
    peak_grid = F3Config(peak_penalty_eur_per_kw_per_day=1.0)
    shared = solve_f1(decision, _base_f1(), peak_grid, _tariff([100, 100]))
    independent = [solve_f1(_decision(s, label=f"oracle_{i}"), _base_f1(), peak_grid, _tariff([100, 100])).schedule_kw for i, s in enumerate(scenarios)]
    nonanticipativity = pd.DataFrame([{
        "fixture": "scenario_specific_oracles_would_differ", "scenario_count": 2,
        "independent_oracle_schedules_differ": not np.allclose(independent[0], independent[1]),
        "implemented_schedule_shared": schedules_are_nonanticipative(shared, 2),
        "implemented_schedule_kw": json.dumps(np.round(shared.schedule_kw, 8).tolist()),
        "passed": bool(not np.allclose(independent[0], independent[1]) and schedules_are_nonanticipative(shared, 2)),
    }])
    stored["stochastic"] = shared

    cvar_decision = _decision([10, 10], scenarios=[[10, 10], [30, 30]], weights=[.9, .1], label="cvar_known_distribution")
    cvar = solve_f1(cvar_decision, _base_f1(p_up_kw=0, p_down_kw=0), F3Config(), _tariff([100, 100]), risk_lambda=1, cvar_alpha=.9)
    rows.append({"fixture": "cvar_known_distribution", "archetype": "F1_CVAR", "solver_status": cvar.solver_status, "objective_eur": cvar.objective_eur, "baseline_cost_eur": .6, "schedule_kw": json.dumps(cvar.schedule_kw.tolist()), "terminal_state_kwh": cvar.state_kwh[-1], "analytic_check_passed": bool(np.isclose(cvar.cvar_eur, 1.5) and np.isclose(cvar.objective_eur, 2.1))})
    stored["cvar"] = cvar
    return pd.DataFrame(rows), pd.DataFrame(audits), nonanticipativity, stored


def parameter_catalog() -> pd.DataFrame:
    cfg = json.loads((CONFIG / "flexibility_archetypes.json").read_text(encoding="utf-8"))
    rows = []
    for dataset, levels in cfg["F1"]["parameter_grid"].items():
        for level, values in levels.items(): rows.append({"archetype": "F1", "dataset": dataset, "level": level, **values, "provenance": cfg["F1"]["provenance"]})
    for level, values in cfg["F2"]["parameter_grid"].items(): rows.append({"archetype": "F2", "dataset": "SYNTHETIC", "level": level, **values, "provenance": "Counterfactual engineering validation assumptions"})
    for dataset, levels in cfg["F3"]["scenario_levels"].items():
        for level, values in levels.items(): rows.append({"archetype": "F3", "dataset": dataset, "level": level, **values, "provenance": "Predeclared connection-cap sensitivity; not a documented site contract"})
    return pd.DataFrame(rows)


def market_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = SPARK / "data/silver/exogenous/canonical/electricity_prices_15min_de_2021_2024.parquet"
    original = SPARK / "data/silver/exogenous/source/electricity_prices_15min_de_2018_2024.csv"
    provenance = pd.DataFrame([
        {"dataset": "SPARK", "year": "2021-2024", "price_source": "project canonical German day-ahead series; original exchange/publication timestamp not encoded", "zone": "Germany/DE-LU project label", "raw_unit": "EUR/MWh", "product_resolution": "hourly in 2024", "normalized_resolution": "15min by repeating each hourly price four times", "utc_alignment": True, "publication_timing": "not encoded", "decision_availability": "EX_POST_PRICE_AWARE", "source_url": "https://www.epexspot.com/en/basicspowermarket", "source_file_hash": digest(canonical), "original_source_hash": digest(original) if original.exists() else "UNAVAILABLE"},
        {"dataset": "HONDA", "year": "2023", "price_source": "NO_LOCAL_AUTHORITATIVE_SERIES_FOUND", "zone": "DE-LU required", "raw_unit": "UNKNOWN", "product_resolution": "hourly day-ahead expected historically but no local source accepted", "normalized_resolution": "UNAVAILABLE", "utc_alignment": False, "publication_timing": "UNKNOWN", "decision_availability": "UNSUPPORTED_TIMING", "source_url": "", "source_file_hash": "UNAVAILABLE", "original_source_hash": "UNAVAILABLE"},
    ])
    timing = pd.DataFrame([
        {"dataset": "SPARK", "forecast_issue_local": "12:00 Europe/Berlin D-1", "price_publication": "source file has no publication timestamp; regular SDAC results occur after gate closure", "scheduling_decision": "ex-post complete-price vector with frozen 12:00 forecast; no later load data", "first_delivery": "00:00 local D", "classification": "EX_POST_PRICE_AWARE", "evidence": "EPEX basics; SDAC fallback timeline"},
        {"dataset": "HONDA", "forecast_issue_local": "13:00 Europe/Berlin D-1", "price_publication": "no accepted local price file/publication record", "scheduling_decision": "not defined for historical experiment", "first_delivery": "00:00 local D", "classification": "UNSUPPORTED_TIMING", "evidence": "absence recorded; no fabricated timing"},
    ])
    honda_scenarios = ROOT / "results/scenarios/honda_2023_diagnostic_scenarios.parquet"
    spark_scenarios = ROOT / "results/scenarios/spark_2024_diagnostic_scenarios.parquet"
    h_days = pd.read_parquet(honda_scenarios, columns=["target_day_local"]).target_day_local.nunique() if honda_scenarios.exists() else 0
    s_days = pd.read_parquet(spark_scenarios, columns=["target_day_local"]).target_day_local.nunique() if spark_scenarios.exists() else 0
    prices = pd.read_parquet(canonical)
    local = prices.utc_datetime.dt.tz_convert("Europe/Berlin")
    price_counts = prices.assign(local_date=local.dt.date).groupby("local_date").size()
    complete_prices_2024 = int(sum(count in {92, 96, 100} for date, count in price_counts.items() if date.year == 2024))
    eligibility = pd.DataFrame([
        {"dataset": "HONDA", "year": 2023, "complete_target_days": 347, "scenario_days": h_days, "complete_price_days": 0, "timing_classification": "UNSUPPORTED_TIMING", "stage3_decision_eligible_days": 0, "reason": "authoritative Honda price input absent"},
        {"dataset": "SPARK", "year": 2024, "complete_target_days": 274, "scenario_days": s_days, "complete_price_days": complete_prices_2024, "timing_classification": "EX_POST_PRICE_AWARE", "stage3_decision_eligible_days": min(274, s_days), "reason": "eligible only for explicitly ex-post price-aware exploratory Stage 3"},
    ])
    return provenance, timing, eligibility


def runtime_benchmark() -> pd.DataFrame:
    rows = []
    for count in [1, 10, 50]:
        scenarios = np.stack([np.full(96, 10 + (i % 5)) for i in range(count)])
        decision = _decision(np.full(96, 10), scenarios=scenarios, weights=np.full(count, 1 / count), label=f"runtime_{count}")
        started = time.perf_counter()
        result = solve_f1(decision, _base_f1(), F3Config(), _tariff(np.linspace(50, 150, 96)))
        rows.append({"scenarios": count, "intervals": 96, "variables_approx": 96 + 97 + 2 * count * 96 + count, "solver_status": result.solver_status, "wall_seconds": time.perf_counter() - started, "solver": "SciPy HiGHS"})
    return pd.DataFrame(rows)


def figures(stored: dict[str, object]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    high = stored["high_to_low"]; x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6, 3)); ax.step(np.arange(3), high.state_kwh, where="post"); ax.axhline(0, color="black", lw=.7); ax.set(xlabel="15-minute boundary", ylabel="F1 state (kWh)", title="F1 advancement/deferral state"); fig.tight_layout(); fig.savefig(FIG / "f1_state.png"); plt.close(fig)
    fig, ax1 = plt.subplots(figsize=(6, 3)); ax1.bar(x, high.schedule_kw, color="tab:blue", label="u (kW)"); ax2 = ax1.twinx(); ax2.plot(x, [200, 50], color="tab:red", marker="o", label="price"); ax1.set(xlabel="interval", ylabel="control kW", title="Price-driven deferral and recovery"); ax2.set_ylabel("EUR/MWh"); fig.tight_layout(); fig.savefig(FIG / "price_driven_shift.png"); plt.close(fig)
    thermal = stored["thermal"]; fig, ax = plt.subplots(figsize=(6, 3)); ax.step(np.arange(3), thermal.state_kwh, where="post"); ax.set(xlabel="boundary", ylabel="kWh_th", title="F2 thermal-buffer state"); fig.tight_layout(); fig.savefig(FIG / "f2_thermal_state.png"); plt.close(fig)
    cap = stored["import_constrained"]; fig, ax = plt.subplots(figsize=(6, 3)); ax.plot([10, 0], marker="o", label="baseline"); ax.plot(np.array([10, 0]) + cap.schedule_kw, marker="o", label="optimized"); ax.axhline(8, ls="--", color="red", label="import cap"); ax.legend(); ax.set(ylabel="kW", title="Import-cap constrained schedule"); fig.tight_layout(); fig.savefig(FIG / "import_cap.png"); plt.close(fig)
    signed = stored["separate_buy_sell"]; fig, ax = plt.subplots(figsize=(6, 3)); ax.axhline(0, color="black", lw=.8); ax.bar(x, [10, -10]); ax.set(ylabel="signed grid exchange (kW)", title="Import/export sign transitions"); fig.tight_layout(); fig.savefig(FIG / "signed_grid_exchange.png"); plt.close(fig)
    stochastic = stored["stochastic"]; fig, ax = plt.subplots(figsize=(6, 3)); ax.bar(x, stochastic.schedule_kw); ax.set(ylabel="shared u (kW)", title="Single non-anticipative first-stage schedule"); fig.tight_layout(); fig.savefig(FIG / "stochastic_shared_schedule.png"); plt.close(fig)
    alphas = [.8, .9, .95]; costs = [.6 + (1.5 if a >= .9 else 1.1) for a in alphas]; fig, ax = plt.subplots(figsize=(6, 3)); ax.plot(alphas, costs, marker="o"); ax.set(xlabel="CVaR alpha", ylabel="illustrative objective (EUR)", title="Toy CVaR sensitivity"); fig.tight_layout(); fig.savefig(FIG / "cvar_sensitivity.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 2)); ax.hlines(0, 0, 3); ax.scatter([0, 1, 2, 3], [0]*4); labels=["Forecast issue", "Price publication", "Schedule commit", "Delivery"]; [ax.text(i, .08, label, ha="center") for i,label in enumerate(labels)]; ax.set_xlim(-.2,3.2); ax.set_ylim(-.2,.35); ax.axis("off"); ax.set_title("Two-stage information timeline (publication must be evidenced)"); fig.tight_layout(); fig.savefig(FIG / "information_timeline.png"); plt.close(fig)


def report(validation: pd.DataFrame, feasibility: pd.DataFrame, timing: pd.DataFrame, eligibility: pd.DataFrame, runtime: pd.DataFrame) -> None:
    md = lambda frame: frame.to_markdown(index=False, floatfmt=".5f")
    text = f"""# Honda Stage 2: physically interpretable flexibility and market framework

## Executive summary

Stage 2 implements and validates a decision framework without running the full Honda 2023 or SPARK 2024 economic comparison. F1 (a **SIMULATED_DEFERRABLE_PROCESS**) is the primary archetype; F2 is a **SIMULATED_THERMAL_BUFFER** engineering model with synthetic demand only; F3 adds import/export, peak, and connection constraints. The framework is **READY_WITH_LIMITATIONS_FOR_STAGE_3**: the optimization is numerically validated and consumes H7 scenarios, but Honda prices and real site flexibility capacities remain unverified.

## Research objective

The framework asks how counterfactual flexibility can be represented and optimized without claiming reconstructed production schedules, installed batteries, dispatchable Honda CHP, or observed customer-bill savings.

## Lessons from SPARK flexibility

The earlier 20%/±2 h model was more sensitive to physical assumptions than to the marginal-quantile policy. Stage 2 therefore treats percentage shifting only as F0 legacy context and replaces it with explicit state and power constraints.

## Source-data limitations

No verified production orders, equipment constraints, Honda tariff, Honda 2023 price file, or customer connection contract is available. All F1/F2/F3 capacities are declared simulation assumptions.
An explicit acquisition contract can validate a future user-supplied Honda DE-LU price file, but performs no implicit download and requires UTC timestamps, EUR/MWh values, source identity, and publication metadata.

## Why percentage shifting is insufficient

Signed Honda grid exchange is not gross flexible consumption. F0 is disabled for Honda; no `fraction × abs(grid_exchange)` shortcut is permitted.

## F0 legacy comparator

F0 remains SPARK-only: 20% assumed flexible load, ±2 h, exact daily energy conservation. It is not the primary physical model.

## F1 virtual production-buffer formulation

For interval duration Δt, `G[t] = baseline[t] + u[t]` and `x[t+1] = x[t] + u[t] Δt`. Positive `x` is advanced electrical work; negative `x` is deferred work. Bounds are `-P_down ≤ u ≤ P_up` and `-E_backlog ≤ x ≤ E_advance`, with declared initial/terminal states. Deferral-only mode constrains `x ≤ 0`. Exact recovery deadlines use source-to-destination kWh flow variables whose permitted arcs are no longer than the configured horizon; the simple state bound alone is never described as a deadline.

## F2 thermal-buffer formulation

`E[t+1] = retention E[t] + (COP P_el[t] - Q_demand[t]) Δt`, in kWh_th. Electrical input is kW_el, demand kW_th, and COP dimensionless. F2 is validated on synthetic service demand and remains secondary until site thermal demand and conversion parameters are established.

## F3 peak-management layer

F3 distinguishes import/export connection limits, scenario-wise hard caps, nominal caps, empirical-quantile chance approximations, soft penalties, and a daily peak shadow price. The shadow price is not presented as a monthly/annual customer demand charge.

## Optional BESS interface

A formal future field contract is frozen, but no BESS optimizer or Honda battery claim is made.

## Physical parameter assumptions

LOW/MEDIUM/HIGH values are predeclared counterfactuals in `physical_parameter_catalog.csv`; they are not fit to exposed-year savings.

## Energy conservation

F1 dynamics and terminal recovery enforce net shifted energy; the independent audit recomputes every transition and bound.

## Recovery and deadline constraints

Daily terminal recovery is primary. Exact-flow mode provides an age-explicit recovery horizon. Multi-day carryover is not silently enabled.

## Signed grid-exchange accounting

`G = Import - Export`, with both variables nonnegative. Tariffs must satisfy interval-wise import-price ≥ export-remuneration, which removes any economic incentive for simultaneous buy/sell in the continuous LP. Negative prices are retained.

## Honda PV/CHP treatment

Honda PV and CHP are already embedded in signed net exchange. They are not subtracted again and CHP is not dispatched.

## Market-data provenance

SPARK has a German EUR/MWh series whose 2024 hourly prices were repeated across four 15-minute intervals. The source file lacks exchange/publication timestamps. Honda has no accepted local 2023 price series.

## Historical market timing

[EPEX SPOT](https://www.epexspot.com/en/basicspowermarket) documents 12:00 CET day-ahead order-book closure. The [SDAC fallback timeline](https://www.epexspot.com/sites/default/files/download_center_files/sdac-fallback-manual.pdf) shows publication follows closure and can be delayed. The [NEMO Committee](https://www.nemo-committee.eu/assets/files/market-coupling-steering-committee-confirms-go-live-of-15-minute-mtu-in-sdac-on-trading-day-30-september-2025-for-delivery-day-1-october-2025.pdf) confirms 15-minute SDAC MTU delivery began 1 October 2025, after these study years.

{md(timing)}

## Forecast versus decision issuance

Frozen 12:00 SPARK and 13:00 Honda forecasts never gain post-issuance measurements. A later schedule can use prices only when their publication is evidenced; otherwise the experiment is ex-post or unsupported.

## Import/export economics

Energy cost is `Σ (p_import Import - p_export Export) Δt`. EUR/MWh is divided by 1000. Wholesale exposure, customer procurement, export compensation, fixed charges, taxes, and demand charges remain distinct.

## Wholesale versus end-user tariff distinction

MKT0–MKT3 are hypothetical wholesale/tariff sensitivities, not actual customer contracts or total electricity bills.

## Deterministic optimizer

SciPy HiGHS solves the F1/F2 linear programs with explicit state, grid-balance, peak, ramp, terminal, and optional deadline-flow constraints.

## Stochastic optimizer

One F1 schedule is shared across all `S × T` H7 trajectories; scenario import/export and cost variables evaluate that same schedule.

## CVaR formulation

For scenario costs `C_s`, the objective adds `λ[z + (1/(1-α)) Σ w_s ξ_s]`, with `ξ_s ≥ C_s-z` and `ξ_s ≥ 0`. α/λ are prospectively frozen and never tuned on exposed years.

## Non-anticipativity

The audit uses a fixture where scenario-wise oracle schedules differ and verifies that the implemented policy returns one shared first-stage schedule.

## Feasibility results

{md(feasibility)}

## Synthetic numerical validation

{md(validation)}

## Solver runtime benchmark

{md(runtime)}

## Scenario integration

The immutable input validates UTC keys, 92/96/100 horizons, scenario weights, provenance hashes, timing, point forecast, and H7 scenario matrix. Settlement applies a committed schedule to a realized baseline without reoptimization.

## Decision eligibility

{md(eligibility)}

## Predeclared Stage 3 experiments

E1 compares deterministic/S1/S2 at fixed F1; E2 varies F1 physical capacity at fixed forecast/tariff; E3 varies MKT assumptions; E4 varies the frozen CVaR grid; E5 performs cautious cross-site synthesis. No full historical economic ranking was run here.

## Unsupported claims and limitations

The work does not identify real factory machines, actual Honda/SPARK flexibility, customer tariffs, battery ownership, dispatchable CHP, operational Honda price timing, or observed bill savings. Exposed holdouts make future Stage 3 results exploratory/predeclared extensions.

## Final readiness assessment

**READY_WITH_LIMITATIONS_FOR_STAGE_3.** F1/F2/F3, signed exchange, stochastic non-anticipativity, CVaR, DST, units, infeasibility handling, and settlement are validated. Stage 3 may run SPARK ex-post price-aware experiments and Honda only after an authoritative price input is added or with an explicit unsupported/ex-post label; it must use frozen H7 S2/50 scenarios and the predeclared parameter/tariff grids.
"""
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "honda_stage2_flexibility_and_market_framework.md").write_text(text, encoding="utf-8")


def manifest() -> None:
    artifacts = [
        *CONFIG.glob("*.json"), *OUT.glob("*.csv"), *FIG.glob("*.png"),
        OUT / "reproducibility_check.json",
        REPORT / "honda_stage2_flexibility_and_market_framework.md",
    ]
    protected = {
        "honda_h3_protocol": digest(ROOT / "config/honda_forecasting_protocol_frozen.json"),
        "honda_h4_baseline_config": digest(ROOT / "config/honda_baseline_selection_frozen.json"),
        "honda_h4_test_predictions": digest(ROOT / "results/baselines/test_predictions_2023.parquet"),
        "honda_h5_lgbm_config": digest(ROOT / "config/honda_lgbm_selection_frozen.json"),
        "honda_h5_test_predictions": digest(ROOT / "results/ml/test_predictions_2023.parquet"),
        "honda_h6_chronos_config": digest(ROOT / "config/honda_chronos2_zero_shot_frozen.json"),
        "honda_h6_timesfm_config": digest(ROOT / "config/honda_timesfm_zero_shot_frozen.json"),
        "honda_h6_test_predictions": digest(ROOT / "results/foundation/test_predictions_2023.parquet"),
        "honda_h7_scenario_config": digest(ROOT / "config/scenarios/scenario_method_frozen.json"),
        "honda_h7_manifest": digest(ROOT / "results/scenarios/honda_h7_manifest.json"),
        "honda_h7_development_scenarios": digest(ROOT / "results/scenarios/honda_development_scenarios.parquet"),
        "honda_h7_diagnostic_scenarios": digest(ROOT / "results/scenarios/honda_2023_diagnostic_scenarios.parquet"),
        "spark_h7_development_scenarios": digest(ROOT / "results/scenarios/spark_development_scenarios.parquet"),
        "spark_h7_diagnostic_scenarios": digest(ROOT / "results/scenarios/spark_2024_diagnostic_scenarios.parquet"),
        "spark_price": digest(SPARK / "data/silver/exogenous/canonical/electricity_prices_15min_de_2021_2024.parquet"),
    }
    data = {
        "starting_checkpoint": START_COMMIT, "H6_checkpoint": "24e86e95aa8e887ae19e17cac536a9b879207db9",
        "stage1_readiness": "READY_WITH_LIMITATIONS", "stage1_scenario_config_sha256": protected["honda_h7_scenario_config"],
        "solver": "SciPy linprog HiGHS", "scipy_version": scipy.__version__, "python_version": platform.python_version(),
        "package_versions": {x: importlib.metadata.version(x) for x in ["numpy", "pandas", "scipy", "matplotlib", "pyarrow"]},
        "unit_contract": {"power": "kW", "energy": "kWh", "thermal_power": "kW_th", "thermal_energy": "kWh_th", "source_price": "EUR/MWh", "normalized_price": "EUR/kWh", "cost": "EUR", "interval": "hours"},
        "random_seed": SEED, "protected_hashes": protected, "full_historical_economic_comparison_performed": False,
        "artifacts": {str(p.relative_to(ROOT)).replace("\\", "/"): {"sha256": digest(p), "bytes": p.stat().st_size} for p in artifacts if p.exists()},
    }
    (OUT / "honda_stage2_manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    validation, feasibility, nonanticipativity, stored = synthetic_validations()
    write_csv(OUT / "synthetic_solver_validation.csv", validation)
    write_csv(OUT / "feasibility_audit.csv", feasibility)
    write_csv(OUT / "nonanticipativity_audit.csv", nonanticipativity)
    write_csv(OUT / "physical_parameter_catalog.csv", parameter_catalog())
    provenance, timing, eligibility = market_artifacts()
    write_csv(OUT / "market_data_provenance.csv", provenance)
    write_csv(OUT / "market_timing_audit.csv", timing)
    write_csv(OUT / "decision_eligibility_inventory.csv", eligibility)
    runtime = runtime_benchmark(); write_csv(OUT / "solver_runtime_benchmark.csv", runtime)
    figures(stored)
    report(validation, feasibility, timing, eligibility, runtime)
    manifest()


if __name__ == "__main__":
    run()
