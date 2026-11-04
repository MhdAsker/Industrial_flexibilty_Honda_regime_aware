# Honda H7 coherent forecast-scenario study

## Executive summary

H7 evaluates leakage-safe residual scenarios before any flexibility optimization. Development selection froze **S2 with 50 equally weighted scenarios**. The method is classified **READY_WITH_LIMITATIONS**: it is causal, reproducible, timestamp-valid and dependence-aware, but rare DST support and conditional/tail calibration remain limited. Honda 2023 and SPARK 2024 are explicitly treated as previously exposed holdout diagnostics, not pristine confirmatory tests.

## Motivation from H1–H6 and SPARK

H1–H6 established a provenance-safe Honda point-forecast benchmark. The SPARK project established calibrated marginal quantiles, but q10/q50/q90 curves do not by themselves define plausible joint daily trajectories. H7 therefore compares independent marginal residual sampling with whole-day coherent residual sampling before asking whether uncertainty changes decisions.

## Literature-grounded methodological rationale

Proper multivariate scoring follows the framework of [Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437). Energy score evaluates joint distributional accuracy but can be less sensitive to misspecified dependence; the lag-weighted variogram score therefore directly tests intraday dependence, following [Scheuerer and Hamill (2015)](https://doi.org/10.1175/MWR-D-14-00269.1). The need to verify temporal interdependence rather than marginal distributions alone is also established in the energy-scenario literature by [Pinson and Girard (2012)](https://doi.org/10.1016/j.apenergy.2011.11.004). Marginal pinball loss, central coverage and weighted interval score remain complementary rather than substitutes for trajectory diagnostics.

## Frozen forecast protocols

Honda retains the H3 13:00 D-1, 15-minute signed-grid-exchange protocol and observed-original target policy. SPARK retains its frozen positive-demand target, issuance convention and primary quality mask. No deterministic model was retrained.

## Source and residual provenance

Residuals use `actual - forecast`. Honda donors are complete out-of-sample LightGBM Fold A/Fold B days. SPARK donors are complete out-of-sample 2023 validation days. TEST residuals never enter fitting.

| dataset_id   |   year |   physical_intervals | eligible_donor   |   size |
|:-------------|-------:|---------------------:|:-----------------|-------:|
| HONDA        |   2021 |                   14 | False            |      1 |
| HONDA        |   2021 |                   25 | False            |      1 |
| HONDA        |   2021 |                   28 | False            |      1 |
| HONDA        |   2021 |                   30 | False            |      1 |
| HONDA        |   2021 |                   31 | False            |      2 |
| HONDA        |   2021 |                   37 | False            |      1 |
| HONDA        |   2021 |                   45 | False            |      1 |
| HONDA        |   2021 |                   49 | False            |      1 |
| HONDA        |   2021 |                   50 | False            |      1 |
| HONDA        |   2021 |                   56 | False            |      1 |
| HONDA        |   2021 |                   66 | False            |      1 |
| HONDA        |   2021 |                   69 | False            |      1 |
| HONDA        |   2021 |                   78 | False            |      1 |
| HONDA        |   2021 |                   87 | False            |      1 |
| HONDA        |   2021 |                   92 | False            |      2 |
| HONDA        |   2021 |                   93 | False            |      1 |
| HONDA        |   2021 |                   94 | False            |      3 |
| HONDA        |   2021 |                   95 | False            |      2 |
| HONDA        |   2021 |                   96 | True             |    341 |
| HONDA        |   2021 |                  100 | True             |      1 |
| HONDA        |   2022 |                   38 | False            |      1 |
| HONDA        |   2022 |                   47 | False            |      1 |
| HONDA        |   2022 |                   89 | False            |      1 |
| HONDA        |   2022 |                   91 | False            |      2 |
| HONDA        |   2022 |                   92 | False            |      1 |
| HONDA        |   2022 |                   92 | True             |      1 |
| HONDA        |   2022 |                   96 | True             |    354 |
| HONDA        |   2022 |                  100 | True             |      1 |
| SPARK        |   2023 |                    1 | False            |      1 |
| SPARK        |   2023 |                   32 | False            |      1 |
| SPARK        |   2023 |                   37 | False            |      1 |
| SPARK        |   2023 |                   61 | False            |      1 |
| SPARK        |   2023 |                   62 | False            |      1 |
| SPARK        |   2023 |                   74 | False            |      1 |
| SPARK        |   2023 |                   76 | False            |      1 |
| SPARK        |   2023 |                   78 | False            |      1 |
| SPARK        |   2023 |                   90 | False            |      2 |
| SPARK        |   2023 |                   92 | False            |      3 |
| SPARK        |   2023 |                   92 | True             |      1 |
| SPARK        |   2023 |                   93 | False            |      2 |
| SPARK        |   2023 |                   94 | False            |     13 |
| SPARK        |   2023 |                   95 | False            |     37 |
| SPARK        |   2023 |                   96 | False            |      1 |
| SPARK        |   2023 |                   96 | True             |    278 |

## Experimental hypotheses and chronological scenario-fitting design

- Honda development: 2021 OOS residual donors, evaluated on 2022 OOS forecasts.
- SPARK development: expanding, past-only 2023 OOS residual library; evaluation starts only after 60 completed donors.
- Honda diagnostic: 2021–2022 donors applied to previously exposed 2023.
- SPARK diagnostic: 2023 donors applied to previously exposed 2024.

No donor can come from the evaluation day or a future date.

## Why TEST residuals are excluded

Honda 2023 and SPARK 2024 were previously examined in earlier stages. Their actual errors are used only for diagnostic scoring after configuration freeze. They are never sampled as donor trajectories.

## Scenario algorithms

- S0: deterministic point trajectory.
- S1: independent local-quarter-hour residual sampling.
- S2: whole-day exact-length residual-trajectory bootstrap.
- S3: S2 conditioned on season and weekday/weekend, with a global exact-length fallback below 20 donors.

## DST-aware trajectory handling

S2/S3 require exact 92/96/100 physical length and at least five donor days of that length. Rare transition days are reported unsupported rather than stretched, truncated or interpolated. S1 preserves the canonical timestamps and samples by local quarter-hour. No repeated autumn interval is collapsed and no spring interval is fabricated.

## Conditional context definition

Only future-known season and day type are used. S3 does not condition on realised load, export state, target-day PV, CHP or weather.

## Development scenario selection

| dataset_id   | scope                    | method   |   scenario_count |   eligible_complete_days |   supported_complete_days |   complete_day_coverage |   fallback_days |   coverage_50 |   coverage_80 |   coverage_90 |   width_50_kw |   width_80_kw |   width_90_kw |   mean_pinball_loss |   wis_kw |   energy_score_kw |   variogram_score |   acf_rmse |   observed_mean_abs_ramp_kw |   scenario_mean_abs_ramp_kw |   scenario_daily_energy_std_kwh |   spark_clipped_fraction |   spark_clipped_negative_mean_magnitude_kw |   mean_absolute_coverage_error |
|:-------------|:-------------------------|:---------|-----------------:|-------------------------:|--------------------------:|------------------------:|----------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------------:|---------:|------------------:|------------------:|-----------:|----------------------------:|----------------------------:|--------------------------------:|-------------------------:|-------------------------------------------:|-------------------------------:|
| HONDA        | DEVELOPMENT_2022         | S0       |                1 |                      356 |                       356 |                  1.0000 |               0 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |             33.2888 | 258.9128 |           85.8061 |           12.2457 |   nan      |                     26.4104 |                      0.0000 |                          0.0000 |                   0.0000 |                                     0.0000 |                         0.7333 |
| HONDA        | DEVELOPMENT_2022         | S1       |               10 |                      356 |                       356 |                  1.0000 |               0 |        0.3993 |        0.6496 |        0.7457 |       92.3063 |      172.1221 |      215.3296 |             21.9719 | 170.8925 |           69.6784 |           26.3183 |     0.5007 |                     26.4104 |                     93.3404 |                        197.6745 |                   0.0000 |                                     0.0000 |                         0.1351 |
| HONDA        | DEVELOPMENT_2022         | S1       |               25 |                      356 |                       356 |                  1.0000 |               0 |        0.4434 |        0.7291 |        0.8230 |       99.2028 |      193.7784 |      245.9196 |             20.4912 | 159.3763 |           65.6737 |           25.3565 |     0.5004 |                     26.4104 |                     93.2550 |                        211.7434 |                   0.0000 |                                     0.0000 |                         0.0682 |
| HONDA        | DEVELOPMENT_2022         | S1       |               50 |                      356 |                       356 |                  1.0000 |               0 |        0.4662 |        0.7544 |        0.8576 |      101.8029 |      201.6851 |      260.5620 |             19.9651 | 155.2845 |           64.3517 |           25.0395 |     0.4979 |                     26.4104 |                     93.1932 |                        216.3264 |                   0.0000 |                                     0.0000 |                         0.0406 |
| HONDA        | DEVELOPMENT_2022         | S2       |               10 |                      356 |                       354 |                  0.9944 |               0 |        0.4059 |        0.6543 |        0.7470 |       92.3988 |      172.5624 |      216.7460 |             21.7236 | 168.9616 |           67.0538 |            8.9229 |     0.2218 |                     26.4311 |                     28.6835 |                       1093.4530 |                   0.0000 |                                     0.0000 |                         0.1309 |
| HONDA        | DEVELOPMENT_2022         | S2       |               25 |                      356 |                       354 |                  0.9944 |               0 |        0.4469 |        0.7321 |        0.8252 |       98.6219 |      194.0607 |      245.7825 |             20.4835 | 159.3159 |           64.2577 |            8.3921 |     0.2118 |                     26.4311 |                     28.8609 |                       1127.7652 |                   0.0000 |                                     0.0000 |                         0.0652 |
| HONDA        | DEVELOPMENT_2022         | S2       |               50 |                      356 |                       354 |                  0.9944 |               0 |        0.4625 |        0.7521 |        0.8556 |      101.1545 |      200.6252 |      259.3175 |             20.0251 | 155.7509 |           63.3171 |            8.2562 |     0.2109 |                     26.4311 |                     28.7991 |                       1140.7322 |                   0.0000 |                                     0.0000 |                         0.0433 |
| HONDA        | DEVELOPMENT_2022         | S3       |               10 |                      356 |                       354 |                  0.9944 |               0 |        0.3857 |        0.6209 |        0.7124 |       91.2677 |      162.7177 |      197.8305 |             22.0919 | 171.8261 |           68.4433 |            8.8436 |     0.2231 |                     26.4311 |                     28.9594 |                        986.2651 |                   0.0000 |                                     0.0000 |                         0.1603 |
| HONDA        | DEVELOPMENT_2022         | S3       |               25 |                      356 |                       354 |                  0.9944 |               0 |        0.4410 |        0.7080 |        0.7970 |      101.1809 |      187.2858 |      231.1445 |             20.7034 | 161.0263 |           64.9107 |            8.4291 |     0.2133 |                     26.4311 |                     29.0414 |                       1074.4018 |                   0.0000 |                                     0.0000 |                         0.0847 |
| HONDA        | DEVELOPMENT_2022         | S3       |               50 |                      356 |                       354 |                  0.9944 |               0 |        0.4698 |        0.7321 |        0.8246 |      104.5460 |      193.6516 |      243.1477 |             20.0953 | 156.2971 |           63.2350 |            8.2415 |     0.2146 |                     26.4311 |                     29.1429 |                       1080.6782 |                   0.0000 |                                     0.0000 |                         0.0578 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S0       |                1 |                      279 |                       279 |                  1.0000 |               0 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |              1.1287 |   8.7788 |            3.0335 |            0.4289 |   nan      |                      0.7431 |                      0.0000 |                          0.0000 |                   0.0000 |                                     0.0000 |                         0.7333 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S1       |               10 |                      279 |                       219 |                  0.7849 |               0 |        0.3690 |        0.6118 |        0.7093 |        2.7016 |        5.7676 |        7.6584 |              0.8590 |   6.6813 |            2.5884 |            1.2940 |     0.5355 |                      0.7934 |                      3.2283 |                          7.5921 |                   0.0102 |                                     1.4398 |                         0.1700 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S1       |               25 |                      279 |                       219 |                  0.7849 |               0 |        0.3995 |        0.6762 |        0.7820 |        2.7167 |        6.5383 |        8.9134 |              0.8134 |   6.3262 |            2.4490 |            1.2575 |     0.5375 |                      0.7934 |                      3.2257 |                          8.0597 |                   0.0103 |                                     1.5452 |                         0.1141 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S1       |               50 |                      279 |                       219 |                  0.7849 |               0 |        0.4169 |        0.6993 |        0.8178 |        2.7176 |        6.7868 |        9.5320 |              0.7916 |   6.1571 |            2.3939 |            1.2377 |     0.5377 |                      0.7934 |                      3.2110 |                          8.0324 |                   0.0100 |                                     1.5703 |                         0.0887 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S2       |               10 |                      279 |                       218 |                  0.7814 |               0 |        0.3616 |        0.6021 |        0.7007 |        2.6776 |        5.7122 |        7.6225 |              0.8669 |   6.7424 |            2.5423 |            0.4258 |     0.1919 |                      0.7953 |                      0.7091 |                         43.7474 |                   0.0100 |                                     0.5282 |                         0.1785 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S2       |               25 |                      279 |                       218 |                  0.7814 |               0 |        0.4039 |        0.6807 |        0.7785 |        2.7160 |        6.4685 |        8.8041 |              0.8015 |   6.2341 |            2.3817 |            0.4072 |     0.1857 |                      0.7953 |                      0.6943 |                         47.1025 |                   0.0110 |                                     0.9420 |                         0.1123 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S2       |               50 |                      279 |                       218 |                  0.7814 |               0 |        0.4150 |        0.6974 |        0.8129 |        2.6803 |        6.7778 |        9.5388 |              0.7984 |   6.2095 |            2.3713 |            0.4021 |     0.1845 |                      0.7953 |                      0.6885 |                         47.9872 |                   0.0105 |                                     1.0189 |                         0.0915 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S3       |               10 |                      279 |                       218 |                  0.7814 |             113 |        0.3895 |        0.6271 |        0.7230 |        2.9897 |        6.0229 |        7.7381 |              0.8452 |   6.5740 |            2.5214 |            0.4221 |     0.1857 |                      0.7953 |                      0.7887 |                         44.6717 |                   0.0094 |                                     0.5558 |                         0.1535 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S3       |               25 |                      279 |                       218 |                  0.7814 |             113 |        0.4145 |        0.6928 |        0.7954 |        3.0283 |        6.7049 |        8.9880 |              0.8039 |   6.2524 |            2.3945 |            0.4054 |     0.1863 |                      0.7953 |                      0.7817 |                         47.7622 |                   0.0109 |                                     0.8244 |                         0.0991 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S3       |               50 |                      279 |                       218 |                  0.7814 |             113 |        0.4339 |        0.7176 |        0.8288 |        3.1420 |        7.1072 |        9.7124 |              0.7806 |   6.0712 |            2.3430 |            0.3992 |     0.1813 |                      0.7953 |                      0.7827 |                         49.4889 |                   0.0108 |                                     0.9310 |                         0.0732 |

Selection uses mean cross-dataset ranks for energy score, variogram score, autocorrelation RMSE and marginal coverage error, plus a coverage penalty. TEST performance and economic outcomes are absent from selection.

## Marginal calibration

| dataset_id   | scope                    | method   |   scenario_count |   coverage_50 |   coverage_80 |   coverage_90 |   width_50_kw |   width_80_kw |   width_90_kw |   mean_pinball_loss |   wis_kw |   mean_absolute_coverage_error |   complete_day_coverage |
|:-------------|:-------------------------|:---------|-----------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------------:|---------:|-------------------------------:|------------------------:|
| HONDA        | DEVELOPMENT_2022         | S0       |                1 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |             33.2888 | 258.9128 |                         0.7333 |                  1.0000 |
| HONDA        | DEVELOPMENT_2022         | S1       |               10 |        0.3993 |        0.6496 |        0.7457 |       92.3063 |      172.1221 |      215.3296 |             21.9719 | 170.8925 |                         0.1351 |                  1.0000 |
| HONDA        | DEVELOPMENT_2022         | S1       |               25 |        0.4434 |        0.7291 |        0.8230 |       99.2028 |      193.7784 |      245.9196 |             20.4912 | 159.3763 |                         0.0682 |                  1.0000 |
| HONDA        | DEVELOPMENT_2022         | S1       |               50 |        0.4662 |        0.7544 |        0.8576 |      101.8029 |      201.6851 |      260.5620 |             19.9651 | 155.2845 |                         0.0406 |                  1.0000 |
| HONDA        | DEVELOPMENT_2022         | S2       |               10 |        0.4059 |        0.6543 |        0.7470 |       92.3988 |      172.5624 |      216.7460 |             21.7236 | 168.9616 |                         0.1309 |                  0.9944 |
| HONDA        | DEVELOPMENT_2022         | S2       |               25 |        0.4469 |        0.7321 |        0.8252 |       98.6219 |      194.0607 |      245.7825 |             20.4835 | 159.3159 |                         0.0652 |                  0.9944 |
| HONDA        | DEVELOPMENT_2022         | S2       |               50 |        0.4625 |        0.7521 |        0.8556 |      101.1545 |      200.6252 |      259.3175 |             20.0251 | 155.7509 |                         0.0433 |                  0.9944 |
| HONDA        | DEVELOPMENT_2022         | S3       |               10 |        0.3857 |        0.6209 |        0.7124 |       91.2677 |      162.7177 |      197.8305 |             22.0919 | 171.8261 |                         0.1603 |                  0.9944 |
| HONDA        | DEVELOPMENT_2022         | S3       |               25 |        0.4410 |        0.7080 |        0.7970 |      101.1809 |      187.2858 |      231.1445 |             20.7034 | 161.0263 |                         0.0847 |                  0.9944 |
| HONDA        | DEVELOPMENT_2022         | S3       |               50 |        0.4698 |        0.7321 |        0.8246 |      104.5460 |      193.6516 |      243.1477 |             20.0953 | 156.2971 |                         0.0578 |                  0.9944 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S0       |                1 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |        0.0000 |              1.1287 |   8.7788 |                         0.7333 |                  1.0000 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S1       |               10 |        0.3690 |        0.6118 |        0.7093 |        2.7016 |        5.7676 |        7.6584 |              0.8590 |   6.6813 |                         0.1700 |                  0.7849 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S1       |               25 |        0.3995 |        0.6762 |        0.7820 |        2.7167 |        6.5383 |        8.9134 |              0.8134 |   6.3262 |                         0.1141 |                  0.7849 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S1       |               50 |        0.4169 |        0.6993 |        0.8178 |        2.7176 |        6.7868 |        9.5320 |              0.7916 |   6.1571 |                         0.0887 |                  0.7849 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S2       |               10 |        0.3616 |        0.6021 |        0.7007 |        2.6776 |        5.7122 |        7.6225 |              0.8669 |   6.7424 |                         0.1785 |                  0.7814 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S2       |               25 |        0.4039 |        0.6807 |        0.7785 |        2.7160 |        6.4685 |        8.8041 |              0.8015 |   6.2341 |                         0.1123 |                  0.7814 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S2       |               50 |        0.4150 |        0.6974 |        0.8129 |        2.6803 |        6.7778 |        9.5388 |              0.7984 |   6.2095 |                         0.0915 |                  0.7814 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S3       |               10 |        0.3895 |        0.6271 |        0.7230 |        2.9897 |        6.0229 |        7.7381 |              0.8452 |   6.5740 |                         0.1535 |                  0.7814 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S3       |               25 |        0.4145 |        0.6928 |        0.7954 |        3.0283 |        6.7049 |        8.9880 |              0.8039 |   6.2524 |                         0.0991 |                  0.7814 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S3       |               50 |        0.4339 |        0.7176 |        0.8288 |        3.1420 |        7.1072 |        9.7124 |              0.7806 |   6.0712 |                         0.0732 |                  0.7814 |

Empirical quantiles from 10/25/50 samples carry Monte Carlo uncertainty; coverage is not interpreted as exact from a small number of scenarios.

## Joint trajectory scoring and temporal dependence

| dataset_id   | scope                    | method   |   scenario_count |   acf_rmse |   observed_mean_abs_ramp_kw |   scenario_mean_abs_ramp_kw |   scenario_daily_energy_std_kwh |   energy_score_kw |   variogram_score |
|:-------------|:-------------------------|:---------|-----------------:|-----------:|----------------------------:|----------------------------:|--------------------------------:|------------------:|------------------:|
| HONDA        | DEVELOPMENT_2022         | S0       |                1 |   nan      |                     26.4104 |                      0.0000 |                          0.0000 |           85.8061 |           12.2457 |
| HONDA        | DEVELOPMENT_2022         | S1       |               10 |     0.5007 |                     26.4104 |                     93.3404 |                        197.6745 |           69.6784 |           26.3183 |
| HONDA        | DEVELOPMENT_2022         | S1       |               25 |     0.5004 |                     26.4104 |                     93.2550 |                        211.7434 |           65.6737 |           25.3565 |
| HONDA        | DEVELOPMENT_2022         | S1       |               50 |     0.4979 |                     26.4104 |                     93.1932 |                        216.3264 |           64.3517 |           25.0395 |
| HONDA        | DEVELOPMENT_2022         | S2       |               10 |     0.2218 |                     26.4311 |                     28.6835 |                       1093.4530 |           67.0538 |            8.9229 |
| HONDA        | DEVELOPMENT_2022         | S2       |               25 |     0.2118 |                     26.4311 |                     28.8609 |                       1127.7652 |           64.2577 |            8.3921 |
| HONDA        | DEVELOPMENT_2022         | S2       |               50 |     0.2109 |                     26.4311 |                     28.7991 |                       1140.7322 |           63.3171 |            8.2562 |
| HONDA        | DEVELOPMENT_2022         | S3       |               10 |     0.2231 |                     26.4311 |                     28.9594 |                        986.2651 |           68.4433 |            8.8436 |
| HONDA        | DEVELOPMENT_2022         | S3       |               25 |     0.2133 |                     26.4311 |                     29.0414 |                       1074.4018 |           64.9107 |            8.4291 |
| HONDA        | DEVELOPMENT_2022         | S3       |               50 |     0.2146 |                     26.4311 |                     29.1429 |                       1080.6782 |           63.2350 |            8.2415 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S0       |                1 |   nan      |                      0.7431 |                      0.0000 |                          0.0000 |            3.0335 |            0.4289 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S1       |               10 |     0.5355 |                      0.7934 |                      3.2283 |                          7.5921 |            2.5884 |            1.2940 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S1       |               25 |     0.5375 |                      0.7934 |                      3.2257 |                          8.0597 |            2.4490 |            1.2575 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S1       |               50 |     0.5377 |                      0.7934 |                      3.2110 |                          8.0324 |            2.3939 |            1.2377 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S2       |               10 |     0.1919 |                      0.7953 |                      0.7091 |                         43.7474 |            2.5423 |            0.4258 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S2       |               25 |     0.1857 |                      0.7953 |                      0.6943 |                         47.1025 |            2.3817 |            0.4072 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S2       |               50 |     0.1845 |                      0.7953 |                      0.6885 |                         47.9872 |            2.3713 |            0.4021 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S3       |               10 |     0.1857 |                      0.7953 |                      0.7887 |                         44.6717 |            2.5214 |            0.4221 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S3       |               25 |     0.1863 |                      0.7953 |                      0.7817 |                         47.7622 |            2.3945 |            0.4054 |
| SPARK        | ROLLING_DEVELOPMENT_2023 | S3       |               50 |     0.1813 |                      0.7953 |                      0.7827 |                         49.4889 |            2.3430 |            0.3992 |

## Honda export-probability evaluation

| scope                        | method   |   scenario_count |   brier_score |   log_loss |   export_prevalence |   mean_export_probability |   probability_any_export |
|:-----------------------------|:---------|-----------------:|--------------:|-----------:|--------------------:|--------------------------:|-------------------------:|
| TEST_2023_PREVIOUSLY_EXPOSED | S0       |                1 |        0.1109 |     1.5316 |              0.1353 |                    0.0577 |                   0.3775 |
| TEST_2023_PREVIOUSLY_EXPOSED | S1       |               50 |        0.0827 |     0.3287 |              0.1353 |                    0.1392 |                   0.9889 |
| TEST_2023_PREVIOUSLY_EXPOSED | S2       |               50 |        0.0834 |     0.3296 |              0.1354 |                    0.1387 |                   0.7776 |
| TEST_2023_PREVIOUSLY_EXPOSED | S3       |               50 |        0.0813 |     0.3189 |              0.1354 |                    0.1421 |                   0.7577 |

Scenario export probabilities are assessed without tuning a 2023 classification threshold.

## SPARK extreme-demand evaluation

| index   |   high_demand_brier_score |   q90_upper_tail_coverage |   high_demand_prevalence |   clipped_fraction |
|:--------|--------------------------:|--------------------------:|-------------------------:|-------------------:|
| mean    |                    0.0543 |                    0.8684 |                   0.1616 |             0.0087 |

SPARK scenario truncation at zero is explicit; the clipped fraction is reported rather than hidden.

## Frozen scenario configuration

The authoritative configuration is `config/scenarios/scenario_method_frozen.json`, SHA-256 `790d3574d468f10fbc565bef1ed0eb492d45ff9db00436f2700a74b7069171ba`.

## Previously exposed TEST diagnostics

| dataset_id   | scope                        | method   |   scenario_count |   eligible_complete_days |   supported_complete_days |   complete_day_coverage |   fallback_days |   coverage_50 |   coverage_80 |   coverage_90 |   width_50_kw |   width_80_kw |   width_90_kw |   mean_pinball_loss |   wis_kw |   energy_score_kw |   variogram_score |   acf_rmse |   observed_mean_abs_ramp_kw |   scenario_mean_abs_ramp_kw |   scenario_daily_energy_std_kwh |   spark_clipped_fraction |   spark_clipped_negative_mean_magnitude_kw |   mean_absolute_coverage_error |
|:-------------|:-----------------------------|:---------|-----------------:|-------------------------:|--------------------------:|------------------------:|----------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------------:|---------:|------------------:|------------------:|-----------:|----------------------------:|----------------------------:|--------------------------------:|-------------------------:|-------------------------------------------:|-------------------------------:|
| HONDA        | TEST_2023_PREVIOUSLY_EXPOSED | S2       |               50 |                      347 |                       346 |                  0.9971 |               0 |        0.4631 |        0.7340 |        0.8217 |      106.1579 |      205.2091 |      262.5616 |             21.7730 | 169.3459 |           66.4859 |           11.2553 |     0.2157 |                     37.3489 |                     27.6975 |                       1226.8428 |                   0.0000 |                                     0.0000 |                         0.0604 |
| SPARK        | TEST_2024_PREVIOUSLY_EXPOSED | S2       |               50 |                      274 |                       273 |                  0.9964 |               0 |        0.4215 |        0.7371 |        0.8468 |        2.8216 |        7.2394 |       10.2248 |              0.7776 |   6.0481 |            2.3198 |            0.3875 |     0.1938 |                      0.7555 |                      0.7384 |                         49.5731 |                   0.0087 |                                     1.2144 |                         0.0649 |

These results are exploratory diagnostics because the scenario study was motivated by known TEST weaknesses. No fresh external holdout is claimed.

## Statistical comparison

| dataset_id   | selected_method   | comparator   |   common_complete_days |   mean_paired_energy_score_difference_kw |   ci95_lower_kw |   ci95_upper_kw |   selected_daily_win_fraction |   bootstrap_resamples |     seed |
|:-------------|:------------------|:-------------|-----------------------:|-----------------------------------------:|----------------:|----------------:|------------------------------:|----------------------:|---------:|
| HONDA        | S2                | S0           |                    346 |                                 -24.4908 |        -25.6307 |        -23.3677 |                        0.9913 |                  2000 | 20261029 |
| HONDA        | S2                | S1           |                    346 |                                  -0.6339 |         -1.0393 |         -0.2535 |                        0.5925 |                  2000 | 20261029 |
| SPARK        | S2                | S0           |                    273 |                                  -0.7922 |         -0.8614 |         -0.7204 |                        0.9194 |                  2000 | 20261029 |
| SPARK        | S2                | S1           |                    273 |                                  -0.0403 |         -0.0588 |         -0.0204 |                        0.6264 |                  2000 | 20261029 |

Comparisons use 2,000 whole-day resamples with seed 20261029.

## Computational and reproducibility results

Residual resampling is CPU-only. Runtime was 658.4 seconds and recorded process RSS was 1.31 GiB. Deterministic generation is rerun and hash-checked.

## Decision-readiness verdict

**READY_WITH_LIMITATIONS.** The selected method has valid timestamps, finite values, equal weights, causal donors, deterministic generation and credible dependence diagnostics. Limitations are rare-day DST support, finite donor pools, conditional fallback, and imperfect tail/export calibration. Prompt 2 may use the scenarios for transparent sensitivity experiments but must not present them as fully validated operational uncertainty.

## Remaining methodological weaknesses

The study is single-site per dataset, relies on historical residual exchangeability, uses coarse conditioning, and cannot provide a pristine prospective holdout. SPARK nonnegative truncation changes a small reported fraction of draws. Scenario samples are empirical and finite rather than a continuous generative distribution.

## Requirements passed to Prompt 2

Prompt 2 must load complete scenario-day matrices by UTC key; preserve 92/96/100 intervals; use supplied nonnegative equal weights; retain Honda signed exchange; reject unsupported days; keep scenario generation fixed; report deterministic and stochastic coverage; and label all decision-value conclusions exploratory on previously exposed holdouts.

## What this experiment does not establish

H7 does not estimate flexibility, dispatch, production feasibility, market revenue, savings or operational value. It does not prove causal effects of CHP, PV, weather, season or modernization.

## Recommended next experiment

Stage 2 should define physically interpretable flexibility archetypes and market-timing assumptions, then test the frozen H7 scenarios only in transparent sensitivity analyses. It must not recalibrate scenario methods using optimization results.
