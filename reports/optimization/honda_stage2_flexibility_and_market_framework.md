# Honda Stage 2: physically interpretable flexibility and market framework

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

| dataset   | forecast_issue_local    | price_publication                                                                       | scheduling_decision                                                          | first_delivery   | classification      | evidence                               |
|:----------|:------------------------|:----------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------|:-----------------|:--------------------|:---------------------------------------|
| SPARK     | 12:00 Europe/Berlin D-1 | source file has no publication timestamp; regular SDAC results occur after gate closure | ex-post complete-price vector with frozen 12:00 forecast; no later load data | 00:00 local D    | EX_POST_PRICE_AWARE | EPEX basics; SDAC fallback timeline    |
| HONDA     | 13:00 Europe/Berlin D-1 | no accepted local price file/publication record                                         | not defined for historical experiment                                        | 00:00 local D    | UNSUPPORTED_TIMING  | absence recorded; no fabricated timing |

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

| fixture                  | passed   |   daily_energy_residual_kwh |   terminal_residual_kwh |   solver_optimal |   scenario_weights_residual |   state_transition_residual |   terminal_residual |   power_upper_violation |   power_lower_violation |   state_upper_violation |   state_lower_violation |   import_definition_residual |   export_definition_residual |   shared_first_stage_schedule |   export_cap_violation |   import_cap_violation |   expected_infeasible |   max_thermal_balance_residual_kwh_th |   max_grid_balance_residual_kw |   terminal_residual_kwh_th |   max_simultaneous_import_export_kw |
|:-------------------------|:---------|----------------------------:|------------------------:|-----------------:|----------------------------:|----------------------------:|--------------------:|------------------------:|------------------------:|------------------------:|------------------------:|-----------------------------:|-----------------------------:|------------------------------:|-----------------------:|-----------------------:|----------------------:|--------------------------------------:|-------------------------------:|---------------------------:|------------------------------------:|
| spark_f0_legacy          | True     |                     0.00000 |                 0.00000 |        nan       |                   nan       |                   nan       |           nan       |               nan       |               nan       |               nan       |               nan       |                    nan       |                    nan       |                     nan       |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| flat_positive            | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| low_to_high              | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| high_to_low              | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| negative_price           | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| zero_price               | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| separate_buy_sell        | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| export_dominant          | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |                0.00000 |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| import_constrained       | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |                0.00000 |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| zero_flexibility         | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| large_power_small_energy | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| small_power_large_energy | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| exact_recovery_deadline  | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| physical_day_92          | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| physical_day_96          | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| physical_day_100         | True     |                   nan       |               nan       |          1.00000 |                     0.00000 |                     0.00000 |             0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                 0.00000 |                      0.00000 |                      0.00000 |                       1.00000 |              nan       |              nan       |             nan       |                             nan       |                      nan       |                  nan       |                           nan       |
| infeasible_cap           | True     |                   nan       |               nan       |          0.00000 |                   nan       |                   nan       |           nan       |               nan       |               nan       |               nan       |               nan       |                    nan       |                    nan       |                     nan       |              nan       |              nan       |               1.00000 |                             nan       |                      nan       |                  nan       |                           nan       |
| thermal_cop_balance      | True     |                   nan       |               nan       |        nan       |                   nan       |                   nan       |           nan       |               nan       |               nan       |               nan       |               nan       |                    nan       |                    nan       |                     nan       |              nan       |              nan       |             nan       |                               0.00000 |                        0.00000 |                    0.00000 |                             0.00000 |

## Synthetic numerical validation

| fixture                  | archetype   | solver_status                                                                                  |   objective_eur |   baseline_cost_eur | schedule_kw   |   terminal_state_kwh | analytic_check_passed   |
|:-------------------------|:------------|:-----------------------------------------------------------------------------------------------|----------------:|--------------------:|:--------------|---------------------:|:------------------------|
| spark_f0_legacy          | F0          | optimal                                                                                        |         0.40000 |             0.50000 | [-2.0, 2.0]   |              0.00000 | True                    |
| flat_positive            | F1          | optimal                                                                                        |         0.50000 |             0.50000 | [5.0, -5.0]   |              0.00000 | True                    |
| low_to_high              | F1          | optimal                                                                                        |         0.43750 |             0.62500 | [5.0, -5.0]   |              0.00000 | True                    |
| high_to_low              | F1          | optimal                                                                                        |         0.43750 |             0.62500 | [-5.0, 5.0]   |              0.00000 | True                    |
| negative_price           | F1          | optimal                                                                                        |        -0.25000 |             0.00000 | [-5.0, 5.0]   |              0.00000 | True                    |
| zero_price               | F1          | optimal                                                                                        |         0.12500 |             0.25000 | [-5.0, 5.0]   |              0.00000 | True                    |
| separate_buy_sell        | F1          | optimal                                                                                        |         0.12500 |             0.12500 | [0.0, 0.0]    |              0.00000 | True                    |
| export_dominant          | F1          | optimal                                                                                        |        -0.12500 |            -0.12500 | [2.0, -2.0]   |              0.00000 | True                    |
| import_constrained       | F1          | optimal                                                                                        |         0.25000 |             0.25000 | [-2.0, 2.0]   |              0.00000 | True                    |
| zero_flexibility         | F1          | optimal                                                                                        |         0.50000 |             0.50000 | [0.0, 0.0]    |              0.00000 | True                    |
| large_power_small_energy | F1          | optimal                                                                                        |         0.45000 |             0.50000 | [-1.0, 1.0]   |              0.00000 | True                    |
| small_power_large_energy | F1          | optimal                                                                                        |         0.45000 |             0.50000 | [-1.0, 1.0]   |              0.00000 | True                    |
| exact_recovery_deadline  | F1          | optimal                                                                                        |         0.25000 |             0.50000 | [-5.0, 5.0]   |              0.00000 | True                    |
| physical_day_92          | F1          | optimal                                                                                        |        23.00000 |            23.00000 | zero          |              0.00000 | True                    |
| physical_day_96          | F1          | optimal                                                                                        |        24.00000 |            24.00000 | zero          |              0.00000 | True                    |
| physical_day_100         | F1          | optimal                                                                                        |        25.00000 |            25.00000 | zero          |              0.00000 | True                    |
| infeasible_cap           | F1          | The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None) |       nan       |             1.00000 | UNAVAILABLE   |            nan       | True                    |
| thermal_cop_balance      | F2          | optimal                                                                                        |         0.40000 |             0.50000 | [-2.0, 2.0]   |              1.00000 | True                    |
| cvar_known_distribution  | F1_CVAR     | optimal                                                                                        |         2.10000 |             0.60000 | [0.0, 0.0]    |              0.00000 | True                    |

## Solver runtime benchmark

|   scenarios |   intervals |   variables_approx | solver_status   |   wall_seconds | solver      |
|------------:|------------:|-------------------:|:----------------|---------------:|:------------|
|           1 |          96 |                386 | optimal         |        0.00639 | SciPy HiGHS |
|          10 |          96 |               2123 | optimal         |        0.01907 | SciPy HiGHS |
|          50 |          96 |               9843 | optimal         |        0.16242 | SciPy HiGHS |

## Scenario integration

The immutable input validates UTC keys, 92/96/100 horizons, scenario weights, provenance hashes, timing, point forecast, and H7 scenario matrix. Settlement applies a committed schedule to a realized baseline without reoptimization.

## Decision eligibility

| dataset   |   year |   complete_target_days |   scenario_days |   complete_price_days | timing_classification   |   stage3_decision_eligible_days | reason                                                               |
|:----------|-------:|-----------------------:|----------------:|----------------------:|:------------------------|--------------------------------:|:---------------------------------------------------------------------|
| HONDA     |   2023 |                    347 |             346 |                     0 | UNSUPPORTED_TIMING      |                               0 | authoritative Honda price input absent                               |
| SPARK     |   2024 |                    274 |             273 |                   366 | EX_POST_PRICE_AWARE     |                             273 | eligible only for explicitly ex-post price-aware exploratory Stage 3 |

## Predeclared Stage 3 experiments

E1 compares deterministic/S1/S2 at fixed F1; E2 varies F1 physical capacity at fixed forecast/tariff; E3 varies MKT assumptions; E4 varies the frozen CVaR grid; E5 performs cautious cross-site synthesis. No full historical economic ranking was run here.

## Unsupported claims and limitations

The work does not identify real factory machines, actual Honda/SPARK flexibility, customer tariffs, battery ownership, dispatchable CHP, operational Honda price timing, or observed bill savings. Exposed holdouts make future Stage 3 results exploratory/predeclared extensions.

## Final readiness assessment

**READY_WITH_LIMITATIONS_FOR_STAGE_3.** F1/F2/F3, signed exchange, stochastic non-anticipativity, CVaR, DST, units, infeasibility handling, and settlement are validated. Stage 3 may run SPARK ex-post price-aware experiments and Honda only after an authoritative price input is added or with an explicit unsupported/ex-post label; it must use frozen H7 S2/50 scenarios and the predeclared parameter/tariff grids.
