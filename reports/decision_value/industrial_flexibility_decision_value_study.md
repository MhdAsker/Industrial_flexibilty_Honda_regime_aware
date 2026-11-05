# Decision-Relevant Forecast Uncertainty and Physically Constrained Flexibility Valuation for Industrial Energy Systems

## Executive summary

On 273 common SPARK 2024 days, under an explicitly hypothetical F1 MEDIUM process and **ex-post-known hourly German day-ahead prices**, deterministic LightGBM saved 1.999617 EUR/day, independent S1 scenarios saved 1.800671, and coherent S2 scenarios saved 1.747536. The paired P3−P4 cost effect was -0.053135 EUR/day with 95% CI [-0.109641, -0.011827]. These are counterfactual wholesale-cost simulations, not actual customer savings or trading P&L. Honda is not economically evaluated because no authoritative 2023 price series passed Stage 2 provenance requirements.

The frozen zero-export cap was violated after realization on 82 deterministic, 16 independent-scenario and 11 coherent-scenario days. Therefore the all-day settlement comparison is the predeclared forecast-risk result, but it is **not fully operationally feasible**. Negative deterministic regret on all days is not interpreted as beating the oracle: it is caused by settlement of cap-violating export. A secondary, explicitly qualified common-feasible-day comparison is reported below.

## Research questions and hypotheses

The frozen hypotheses test scenario coherence, probabilistic value, physical-capacity dominance, market dependence, CVaR, cross-site heterogeneity and Honda export value. Verdicts are reported without forcing a positive result.

| hypothesis                        | verdict             | evidence                                                         | limitation                                   |
|:----------------------------------|:--------------------|:-----------------------------------------------------------------|:---------------------------------------------|
| H1_COHERENT_VS_INDEPENDENT        | NOT_SUPPORTED       | P3-P4 mean -0.053135 EUR/day; CI [-0.109641,-0.011827]           | ex-post exposed 2024                         |
| H2_PROBABILISTIC_VALUE            | NOT_SUPPORTED       | P2-P4 mean -0.252081 EUR/day; CI [-0.315669,-0.199605]           | ex-post exposed 2024                         |
| H3_PHYSICAL_FLEXIBILITY_DOMINATES | SUPPORTED           | physical savings range 2.316880 vs P2-P4 effect 0.252081 EUR/day | counterfactual capacities                    |
| H4_MARKET_STRUCTURE_DEPENDENCE    | PARTIALLY_SUPPORTED | peak/grid savings range 0.418404; tariff range 0.001066          | hypothetical tariffs                         |
| H5_CVAR_RISK_AWARE                | NOT_SUPPORTED       | P4-P6 mean realized cost difference 0.000000 EUR/day             | 273 exposed days; tail inference descriptive |
| H6_CROSS_DATASET_HETEROGENEITY    | NOT_EVALUABLE       | Honda authoritative price input absent                           | single economically evaluated dataset        |
| H7_HONDA_EXPORT_STATE_VALUE       | NOT_EVALUABLE       | Honda economic settlement blocked                                | no price/timing series                       |

## Prior SPARK/Honda experimental evidence

Frozen forecasts, H7 residual scenarios and Stage 2 optimization configurations are reused without retraining or recalibration. Earlier SPARK marginal-quantile flexibility remains a legacy comparator only.

## Literature context and novelty boundary

The contribution is integration and falsification: proper coherent scenarios are connected to one shared physically constrained schedule and ex-post settlement. It does not introduce a new forecasting architecture, claim plant identification, or validate an actual tariff.

## Data provenance and limitations

SPARK is positive facility demand; Honda is signed net exchange. SPARK prices are German hourly day-ahead values repeated across four 15-minute physical intervals. Price publication is not encoded, so every historical economic result is `EX_POST_PRICE_AWARE` and previously exposed.

The exact artifact-derived common range is 6 January through 1 December 2024. Amendment 001 corrects a descriptive end-date typo in the original frozen JSON without changing the 273-day intersection or any outcome. All evaluated days contain 96 intervals; no DST transition day survives the Stage 1 scenario interface, so 92/100-interval economics are not claimed here.

## Frozen forecast policies

P0 no flexibility; P2 frozen LightGBM; P3 frozen-algorithm S1; P4 frozen S2/50; P6 S2/50 with α=0.9, λ=0.25; P7 comparable oracle. B5 is not a frozen SPARK Stage 3 policy and S3 was rejected by H7 development evidence.

## Probabilistic/scenario representations

S1 independently samples interval residuals from causal 2023 development donors. S2 samples whole residual days. Both use 50 equally weighted trajectories and the same point forecast. No 2024 residual enters either library.

## Physical flexibility archetypes

Primary F1 is the frozen SPARK MEDIUM counterfactual: ±5 kW, ±10 kWh state, Δt=0.25 h, zero initial/terminal state and no export. LOW/HIGH are sensitivity assumptions, not measured machines.

## Market timing and price availability

The forecast remains issued at 12:00 D−1. Historical prices are treated as ex-post-known at scheduling, without post-12:00 load measurements. Honda is blocked rather than assigned synthetic market history.

## Economic objective and tariff assumptions

Primary cost is symmetric wholesale pass-through `Σ(price × import − price × export)Δt`, with SPARK export prohibited. Peak shadow prices and asymmetric remuneration are hypothetical sensitivities, not complete bills.

## Decision and settlement separation

Optimization contracts exclude actual demand. Schedules are sealed first; actuals enter only the standalone settlement engine. Oracle actuals require explicit P7 labelling.

Realized cap violations are never clipped or reoptimized. Regret is marked operationally interpretable only when the realized trajectory respects the frozen grid caps.

## Chronological evaluation protocol

The common set contains 273 complete S2/target/price days. Honda 2023 and SPARK 2024 are disclosed as previously exposed diagnostics. The experiment file was frozen after a five-day 2023 development dry run and before 2024 cost calculation.

## Statistical inference

Primary CIs use 2,000 whole-day bootstrap resamples, seed 20261029, with Holm adjustment for two primary comparisons. Seven-day moving-block CIs, lag-1 dependence and top-1% outlier sensitivity are also reported.

| comparison   | policy_A              | policy_B       | interpretation                                             |   common_days |   mean_cost_A_minus_B_eur |   median_cost_A_minus_B_eur |   ci95_lower_eur |   ci95_upper_eur |   block_ci95_lower_eur |   block_ci95_upper_eur |   policy_B_daily_win_fraction |   lag1_daily_difference |   mean_without_top1pct_abs_outliers_eur |   two_sided_bootstrap_p |   bootstrap_resamples |   block_length |     seed |   holm_adjusted_p |
|:-------------|:----------------------|:---------------|:-----------------------------------------------------------|--------------:|--------------------------:|----------------------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|------------------------------:|------------------------:|----------------------------------------:|------------------------:|----------------------:|---------------:|---------:|------------------:|
| P2_minus_P4  | P2_LGBM_DETERMINISTIC | P4_S2_COHERENT | positive difference means policy B has lower realized cost |           273 |                 -0.252081 |                   -0.110937 |        -0.315669 |        -0.199605 |              -0.338466 |              -0.184494 |                      0.003663 |                0.155560 |                               -0.216736 |                0.000000 |                  2000 |              7 | 20261029 |          0.000000 |
| P3_minus_P4  | P3_S1_INDEPENDENT     | P4_S2_COHERENT | positive difference means policy B has lower realized cost |           273 |                 -0.053135 |                    0.021040 |        -0.109641 |        -0.011827 |              -0.113310 |              -0.011385 |                      0.615385 |               -0.061425 |                               -0.022541 |                0.009000 |                  2000 |              7 | 20261029 |          0.009000 |

## Primary deterministic versus probabilistic comparison

| policy                |   available_days |   valid_days |   infeasible_or_failed_days |   mean_realized_cost_eur_day |   mean_savings_eur_day |   median_savings_eur_day |   p10_savings_eur_day |   p90_savings_eur_day |   negative_savings_fraction |   p90_realized_cost_eur |   p95_realized_cost_eur |   realized_cvar90_eur |   realized_cvar95_eur |   mean_peak_import_kw |   mean_shifted_energy_kwh |   mean_backlog_utilization |   realized_import_violation_days |   realized_export_violation_days |   median_runtime_seconds |   p95_runtime_seconds |   mean_regret_eur |   p90_regret_eur |   worst_regret_eur |
|:----------------------|-----------------:|-------------:|----------------------------:|-----------------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|----------------------------:|------------------------:|------------------------:|----------------------:|----------------------:|----------------------:|--------------------------:|---------------------------:|---------------------------------:|---------------------------------:|-------------------------:|----------------------:|------------------:|-----------------:|-------------------:|
| P0_NO_FLEX            |              273 |          273 |                           0 |                    28.447594 |               0.000000 |                 0.000000 |              0.000000 |              0.000000 |                           0 |               49.553120 |               55.769643 |             58.955770 |             65.956331 |             25.918068 |                  0.000000 |                   0.000000 |                                0 |                                0 |                 0.000025 |              0.000030 |          1.998817 |         3.098010 |          12.633250 |
| P2_LGBM_DETERMINISTIC |              273 |          273 |                           0 |                    26.447976 |               1.999617 |                 1.842800 |              0.905317 |              3.098010 |                           0 |               47.696816 |               53.183035 |             55.804232 |             62.017802 |             29.784567 |                 56.424439 |                   1.000000 |                                0 |                               82 |                 0.019856 |              0.021378 |         -0.000800 |         0.000000 |           0.024714 |
| P3_S1_INDEPENDENT     |              273 |          273 |                           0 |                    26.646923 |               1.800671 |                 1.692989 |              0.635573 |              2.912118 |                           0 |               47.785660 |               53.249728 |             55.963367 |             62.257839 |             29.508777 |                 43.133410 |                   0.958470 |                                0 |                               16 |                 0.248488 |              0.266442 |          0.198146 |         0.549675 |           1.254330 |
| P4_S2_COHERENT        |              273 |          273 |                           0 |                    26.700058 |               1.747536 |                 1.693100 |              0.551133 |              2.934018 |                           0 |               47.698721 |               53.183196 |             56.211531 |             62.645896 |             29.443515 |                 42.609141 |                   0.947019 |                                0 |                               11 |                 0.247547 |              0.267706 |          0.251281 |         0.626715 |           6.542013 |
| P6_S2_CVAR            |              273 |          273 |                           0 |                    26.700058 |               1.747536 |                 1.693100 |              0.551133 |              2.934018 |                           0 |               47.698721 |               53.183196 |             56.211531 |             62.645896 |             29.195815 |                 40.190930 |                   0.945893 |                                0 |                               11 |                 0.303166 |              0.338510 |          0.251281 |         0.626715 |           6.542013 |
| P7_ORACLE_UPPER_BOUND |              273 |          273 |                           0 |                    26.448777 |               1.998817 |                 1.842800 |              0.904932 |              3.098010 |                           0 |               47.696816 |               53.183035 |             55.804232 |             62.017802 |             29.781805 |                 56.342677 |                   1.000000 |                                0 |                                0 |                 0.020559 |              0.023725 |          0.000000 |         0.000000 |           0.000000 |

### Realized-grid-cap qualification

| policy                |   available_days |   valid_days |   infeasible_or_failed_days |   mean_realized_cost_eur_day |   mean_savings_eur_day |   median_savings_eur_day |   p10_savings_eur_day |   p90_savings_eur_day |   negative_savings_fraction |   p90_realized_cost_eur |   p95_realized_cost_eur |   realized_cvar90_eur |   realized_cvar95_eur |   mean_peak_import_kw |   mean_shifted_energy_kwh |   mean_backlog_utilization |   realized_import_violation_days |   realized_export_violation_days |   median_runtime_seconds |   p95_runtime_seconds |   mean_regret_eur |   p90_regret_eur |   worst_regret_eur | qualification                                              |
|:----------------------|-----------------:|-------------:|----------------------------:|-----------------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|----------------------------:|------------------------:|------------------------:|----------------------:|----------------------:|----------------------:|--------------------------:|---------------------------:|---------------------------------:|---------------------------------:|-------------------------:|----------------------:|------------------:|-----------------:|-------------------:|:-----------------------------------------------------------|
| P0_NO_FLEX            |              273 |          273 |                           0 |                    28.447594 |               0.000000 |                 0.000000 |              0.000000 |              0.000000 |                           0 |               49.553120 |               55.769643 |             58.955770 |             65.956331 |             25.918068 |                  0.000000 |                   0.000000 |                                0 |                                0 |                 0.000025 |              0.000030 |          1.998817 |         3.098010 |          12.633250 | SECONDARY_POLICY_SPECIFIC_REALIZED_GRID_CAP_COMPLIANT_DAYS |
| P2_LGBM_DETERMINISTIC |              191 |          191 |                           0 |                    29.946908 |               2.326373 |                 2.263150 |              1.299450 |              3.250400 |                           0 |               49.996230 |               58.187300 |             58.754967 |             64.706620 |             32.369040 |                 56.276094 |                   1.000000 |                                0 |                                0 |                 0.019810 |              0.021033 |          0.000208 |         0.000000 |           0.024714 | SECONDARY_POLICY_SPECIFIC_REALIZED_GRID_CAP_COMPLIANT_DAYS |
| P3_S1_INDEPENDENT     |              257 |          257 |                           0 |                    27.336711 |               1.849406 |                 1.754521 |              0.642204 |              2.938214 |                           0 |               48.122993 |               53.438679 |             56.580876 |             62.943145 |             30.086378 |                 43.728270 |                   0.956578 |                                0 |                                0 |                 0.248608 |              0.266949 |          0.184315 |         0.497227 |           1.254330 | SECONDARY_POLICY_SPECIFIC_REALIZED_GRID_CAP_COMPLIANT_DAYS |
| P4_S2_COHERENT        |              262 |          262 |                           0 |                    27.188647 |               1.769705 |                 1.739652 |              0.538637 |              2.939350 |                           0 |               48.002477 |               53.285173 |             56.525719 |             62.645896 |             29.770687 |                 42.684822 |                   0.944794 |                                0 |                                0 |                 0.247559 |              0.268212 |          0.248188 |         0.620401 |           6.542013 | SECONDARY_POLICY_SPECIFIC_REALIZED_GRID_CAP_COMPLIANT_DAYS |
| P6_S2_CVAR            |              262 |          262 |                           0 |                    27.188647 |               1.769705 |                 1.739652 |              0.538637 |              2.939350 |                           0 |               48.002477 |               53.285173 |             56.525719 |             62.645896 |             29.515589 |                 40.281638 |                   0.943622 |                                0 |                                0 |                 0.303254 |              0.339150 |          0.248188 |         0.620401 |           6.542013 | SECONDARY_POLICY_SPECIFIC_REALIZED_GRID_CAP_COMPLIANT_DAYS |
| P7_ORACLE_UPPER_BOUND |              273 |          273 |                           0 |                    26.448777 |               1.998817 |                 1.842800 |              0.904932 |              3.098010 |                           0 |               47.696816 |               53.183035 |             55.804232 |             62.017802 |             29.781805 |                 56.342677 |                   1.000000 |                                0 |                                0 |                 0.020559 |              0.023725 |          0.000000 |         0.000000 |           0.000000 | SECONDARY_POLICY_SPECIFIC_REALIZED_GRID_CAP_COMPLIANT_DAYS |

| comparison                  | policy_A              | policy_B       | interpretation                                             |   common_days |   mean_cost_A_minus_B_eur |   median_cost_A_minus_B_eur |   ci95_lower_eur |   ci95_upper_eur |   block_ci95_lower_eur |   block_ci95_upper_eur |   policy_B_daily_win_fraction |   lag1_daily_difference |   mean_without_top1pct_abs_outliers_eur |   two_sided_bootstrap_p |   bootstrap_resamples |   block_length |     seed | qualification                                     |
|:----------------------------|:----------------------|:---------------|:-----------------------------------------------------------|--------------:|--------------------------:|----------------------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|------------------------------:|------------------------:|----------------------------------------:|------------------------:|----------------------:|---------------:|---------:|:--------------------------------------------------|
| P2_minus_P4_common_feasible | P2_LGBM_DETERMINISTIC | P4_S2_COHERENT | positive difference means policy B has lower realized cost |           191 |                 -0.191191 |                   -0.051716 |        -0.271593 |        -0.131960 |              -0.313554 |              -0.113278 |                      0.005236 |                0.143873 |                               -0.148814 |                0.000000 |                  2000 |              7 | 20261029 | SECONDARY_COMMON_REALIZED_GRID_CAP_COMPLIANT_DAYS |
| P3_minus_P4_common_feasible | P3_S1_INDEPENDENT     | P4_S2_COHERENT | positive difference means policy B has lower realized cost |           256 |                 -0.056938 |                    0.020600 |        -0.115541 |        -0.015121 |              -0.121450 |              -0.012869 |                      0.617188 |               -0.061561 |                               -0.024334 |                0.004000 |                  2000 |              7 | 20261029 | SECONDARY_COMMON_REALIZED_GRID_CAP_COMPLIANT_DAYS |

These tables are secondary diagnostics selected by a physical feasibility condition, not a replacement primary analysis. They prevent cap-violating schedules from being described as implementable value.

## Independent versus coherent scenarios

The mean interval-level absolute schedule difference between P3 and P4 was 1.018045 kW. The economic comparison isolates temporal dependence because physical parameters, prices, point forecast, scenario count, optimizer and days are identical.

## Physical-capacity sensitivity

| level   |   p_up_down_kw |   energy_capacity_each_direction_kwh |   available_days |   valid_days |   infeasible_or_failed_days |   mean_cost_eur_day |   mean_savings_eur_day |   median_savings_eur_day |   negative_savings_fraction |   mean_peak_import_kw |   mean_shifted_energy_kwh |   mean_backlog_utilization |   import_violation_days |   export_violation_days |   median_runtime_seconds |
|:--------|---------------:|-------------------------------------:|-----------------:|-------------:|----------------------------:|--------------------:|-----------------------:|-------------------------:|----------------------------:|----------------------:|--------------------------:|---------------------------:|------------------------:|------------------------:|-------------------------:|
| LOW     |              2 |                                    4 |              273 |          273 |                           0 |           27.706827 |               0.740766 |                 0.703180 |                           0 |             27.028554 |                 18.633925 |                   0.964441 |                       0 |                       0 |                 0.236498 |
| MEDIUM  |              5 |                                   10 |              273 |          273 |                           0 |           26.700058 |               1.747536 |                 1.693100 |                           0 |             29.443515 |                 42.609141 |                   0.947019 |                       0 |                      11 |                 0.247547 |
| HIGH    |             10 |                                   20 |              273 |          273 |                           0 |           25.389947 |               3.057647 |                 2.945953 |                           0 |             34.041140 |                 71.785146 |                   0.930743 |                       0 |                      54 |                 0.238289 |

## Grid-cap/peak sensitivity

| case             |   import_cap_kw |   peak_penalty_eur_per_kw_day |   available_days |   valid_days |   infeasible_or_failed_days |   mean_cost_eur_day |   mean_savings_eur_day |   median_savings_eur_day |   negative_savings_fraction |   mean_peak_import_kw |   mean_shifted_energy_kwh |   mean_backlog_utilization |   import_violation_days |   export_violation_days |   median_runtime_seconds |
|:-----------------|----------------:|------------------------------:|-----------------:|-------------:|----------------------------:|--------------------:|-----------------------:|-------------------------:|----------------------------:|----------------------:|--------------------------:|---------------------------:|------------------------:|------------------------:|-------------------------:|
| NO_IMPORT_CAP    |      nan        |                      0.000000 |              273 |          273 |                           0 |           26.700058 |               1.747536 |                 1.693100 |                    0.000000 |             29.443515 |                 42.609141 |                   0.947019 |                       0 |                      11 |                 0.247547 |
| IMPORT_CAP_40    |       40.000000 |                      0.000000 |              273 |          120 |                         153 |           20.608516 |               1.479334 |                 1.429642 |                    0.000000 |             22.481740 |                 35.232071 |                   0.894677 |                       6 |                       8 |                 0.239387 |
| IMPORT_CAP_30    |       30.000000 |                      0.000000 |              273 |           31 |                         242 |           10.900371 |               1.361855 |                 1.371585 |                    0.000000 |             15.832306 |                 28.269248 |                   0.833389 |                       0 |                       2 |                 0.272210 |
| IMPORT_CAP_20    |       20.000000 |                      0.000000 |              273 |            0 |                         273 |          nan        |             nan        |               nan        |                  nan        |            nan        |                nan        |                 nan        |                       0 |                       0 |               nan        |
| PEAK_SHADOW_0_05 |      nan        |                      0.050000 |              273 |          273 |                           0 |           28.135636 |               1.607861 |                 1.512133 |                    0.010989 |             28.141744 |                 36.652950 |                   0.943906 |                       0 |                      11 |                 0.282952 |
| PEAK_SHADOW_0_20 |      nan        |                      0.200000 |              273 |          273 |                           0 |           32.302075 |               1.329132 |                 1.270024 |                    0.054945 |             27.090580 |                 32.089691 |                   0.938957 |                       0 |                       9 |                 0.288414 |

## Risk-aware CVaR comparison

| case             |      alpha |   risk_lambda |   available_days |   valid_days |   infeasible_or_failed_days |   mean_cost_eur_day |   mean_savings_eur_day |   median_savings_eur_day |   negative_savings_fraction |   mean_peak_import_kw |   mean_shifted_energy_kwh |   mean_backlog_utilization |   import_violation_days |   export_violation_days |   median_runtime_seconds |   realized_p90_cost_eur |   realized_p95_cost_eur |   realized_cvar_eur |
|:-----------------|-----------:|--------------:|-----------------:|-------------:|----------------------------:|--------------------:|-----------------------:|-------------------------:|----------------------------:|----------------------:|--------------------------:|---------------------------:|------------------------:|------------------------:|-------------------------:|------------------------:|------------------------:|--------------------:|
| RISK_NEUTRAL     | nan        |      0.000000 |              273 |          273 |                           0 |           26.700058 |               1.747536 |                 1.693100 |                           0 |             29.443515 |                 42.609141 |                   0.947019 |                       0 |                      11 |                 0.247547 |               47.698721 |               53.183196 |           56.211531 |
| CVAR_A0.9_L0.25  |   0.900000 |      0.250000 |              273 |          273 |                           0 |           26.700058 |               1.747536 |                 1.693100 |                           0 |             29.195815 |                 40.190930 |                   0.945893 |                       0 |                      11 |                 0.303166 |               47.698721 |               53.183196 |           56.211531 |
| CVAR_A0.9_L1.0   |   0.900000 |      1.000000 |              273 |          273 |                           0 |           26.700058 |               1.747536 |                 1.693100 |                           0 |             29.221890 |                 40.317232 |                   0.945893 |                       0 |                      11 |                 0.295291 |               47.698721 |               53.183196 |           56.211531 |
| CVAR_A0.95_L0.25 |   0.950000 |      0.250000 |              273 |          273 |                           0 |           26.700058 |               1.747536 |                 1.693100 |                           0 |             29.220540 |                 40.263385 |                   0.945893 |                       0 |                      11 |                 0.293424 |               47.698721 |               53.183196 |           62.645896 |
| CVAR_A0.95_L1.0  |   0.950000 |      1.000000 |              273 |          273 |                           0 |           26.700058 |               1.747536 |                 1.693100 |                           0 |             29.231157 |                 40.325502 |                   0.945893 |                       0 |                      11 |                 0.292939 |               47.698721 |               53.183196 |           62.645896 |

The primary CVaR policy saved 1.747536 EUR/day. Forecast-scenario CVaR is not equated with realized historical tail risk.

## Market/tariff sensitivity

| case        |   export_multiplier |   available_days |   valid_days |   infeasible_or_failed_days |   mean_cost_eur_day |   mean_savings_eur_day |   median_savings_eur_day |   negative_savings_fraction |   mean_peak_import_kw |   mean_shifted_energy_kwh |   mean_backlog_utilization |   import_violation_days |   export_violation_days |   median_runtime_seconds |
|:------------|--------------------:|-----------------:|-------------:|----------------------------:|--------------------:|-----------------------:|-------------------------:|----------------------------:|----------------------:|--------------------------:|---------------------------:|------------------------:|------------------------:|-------------------------:|
| SYMMETRIC   |            1.000000 |              273 |          273 |                           0 |           26.700058 |               1.747536 |                 1.693100 |                           0 |             29.443515 |                 42.609141 |                   0.947019 |                       0 |                      11 |                 0.247547 |
| HALF_EXPORT |            0.500000 |              273 |          273 |                           0 |           26.700590 |               1.747003 |                 1.693100 |                           0 |             29.443515 |                 42.609141 |                   0.947019 |                       0 |                      11 |                 0.238903 |
| ZERO_EXPORT |            0.000000 |              273 |          273 |                           0 |           26.701123 |               1.746470 |                 1.693100 |                           0 |             29.443515 |                 42.609141 |                   0.947019 |                       0 |                      11 |                 0.237838 |

## Honda export-regime findings

**NOT EVALUABLE.** Honda has valid forecasts and signed scenarios but no accepted historical price/timing input. No economic export claim is fabricated.

## SPARK high-demand/holiday findings

| policy                | day_group      |   days |   mean_cost_eur |   mean_savings_eur |   mean_regret_eur |
|:----------------------|:---------------|-------:|----------------:|-------------------:|------------------:|
| P2_LGBM_DETERMINISTIC | HIGH_DEMAND    |    159 |       35.708719 |           2.204955 |         -0.000086 |
| P2_LGBM_DETERMINISTIC | NORMAL_WEEKDAY |     25 |       26.668718 |           1.508776 |         -0.001006 |
| P2_LGBM_DETERMINISTIC | PUBLIC_HOLIDAY |      9 |        8.711271 |           1.757372 |         -0.007607 |
| P2_LGBM_DETERMINISTIC | WEEKEND        |     80 |        9.968649 |           1.772149 |         -0.001390 |
| P3_S1_INDEPENDENT     | HIGH_DEMAND    |    159 |       35.845190 |           2.068483 |          0.136386 |
| P3_S1_INDEPENDENT     | NORMAL_WEEKDAY |     25 |       26.840548 |           1.336946 |          0.170824 |
| P3_S1_INDEPENDENT     | PUBLIC_HOLIDAY |      9 |        8.977011 |           1.491632 |          0.258133 |
| P3_S1_INDEPENDENT     | WEEKEND        |     80 |       10.292722 |           1.448075 |          0.322684 |
| P4_S2_COHERENT        | HIGH_DEMAND    |    159 |       35.905529 |           2.008145 |          0.196724 |
| P4_S2_COHERENT        | NORMAL_WEEKDAY |     25 |       26.903001 |           1.274493 |          0.233277 |
| P4_S2_COHERENT        | PUBLIC_HOLIDAY |      9 |        9.084608 |           1.384035 |          0.365730 |
| P4_S2_COHERENT        | WEEKEND        |     80 |       10.322501 |           1.418297 |          0.352462 |

Holiday groups are descriptive where small and are not used for policy tuning.

## Cross-dataset interpretation

Only SPARK has economic outcomes. The Honda row remains explicitly missing; raw euro or kW pooling is therefore avoided.

## Mechanisms explaining positive or negative value

Under risk-neutral expected linear energy cost, temporal correlation matters only through nonseparable constraints and the convex import/export split. The F1 state couples intervals, but scenario-wise no-export constraints and finite scenario means can dominate. If P3 and P4 are nearly identical, this is a structural consequence rather than evidence that H7 scenarios are statistically identical.

## Computational feasibility

All primary schedules use CPU HiGHS. Runtime coverage is reported per policy; failed or infeasible solves receive no favorable fallback.

## Negative controls and verification

| control                         | passed   | detail                                                |
|:--------------------------------|:---------|:------------------------------------------------------|
| N0_NO_FLEX                      | True     | zero control                                          |
| N1_IDENTICAL_SCENARIOS          | True     | deterministic/stochastic equivalence                  |
| N2_FLAT_PRICE                   | True     | no lossless arbitrage                                 |
| N3_ZERO_ENERGY                  | True     | no shifting                                           |
| N4_FUTURE_INFORMATION           | True     | actual absent from optimizer contract                 |
| N5_ORACLE_SEPARATION            | True     | oracle requires explicit P7 label and actual argument |
| N6_SIGN_INVERSION               | True     | signed grid split                                     |
| N7_REPRODUCIBLE_IDENTICAL_INPUT | True     | fixed solver input                                    |
| N8_SCENARIO_PERMUTATION         | True     | order invariance                                      |
| N9_COST_CONSISTENCY             | True     | optimizer versus standalone settlement                |

## Mathematical consistency

### Conservation and feasible polytope
`x[t+1]-x[t]-u[t]Δt=0`, fixed endpoints and box bounds define a convex polytope; summing dynamics proves daily net shifted energy is zero.

### Import/export convexity
With `G=I-E`, `I,E≥0` and buy price no lower than export remuneration, increasing I and E together cannot improve cost. Canonical settlement selects `I=max(G,0)`, `E=max(-G,0)`.

### Deterministic/stochastic equivalence
If all scenarios are identical and the risk term is absent, the stochastic LP reduces to the deterministic LP. Scenario ordering is immaterial because weights enter symmetrically.

### When dependence can matter
Pure expected separable linear costs depend mainly on marginal expectations. Whole-day correlation changes decisions through daily CVaR, scenario-wise joint caps, peak variables, or state-coupled tail patterns—not by name alone.

### Regret and oracle limit
`regret = realized policy cost − oracle cost`, where the oracle uses identical F1 and tariff constraints. It is an unattainable information upper bound, not an implementable policy.

## Research limitations

The only economic dataset is SPARK; asset/tariff parameters are hypothetical; prices are ex-post at decision time; 2024 was previously exposed; full industrial production constraints are unavailable; and 273 days limit tail inference.

## Publication-readiness assessment

**CATEGORY C — technically sound exploratory simulation.** Software and causal information controls are strong, but one-site ex-post pricing and hypothetical flexibility prevent Category A/B claims.

## Conclusions

The experiment quantifies whether uncertainty representation changes a shared constrained decision, rather than inferring value from forecast scores. The numerical verdicts above are conditional on the frozen F1 and wholesale assumptions.

## Recommended future work

Obtain authoritative Honda prices/publication metadata, real equipment and tariff constraints, and a prospective untouched period; then repeat the frozen analysis without parameter reselection.
