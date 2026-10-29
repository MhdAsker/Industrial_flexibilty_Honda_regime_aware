# Honda leakage-safe baseline forecasting benchmark

## Executive summary

H4 implemented only transparent baselines B0–B5 under the frozen H3 protocol. The P1-selected PRIMARY_BASELINE is **B5**. On untouched 2023 primary targets it achieved MAE 76.480 kW, RMSE 101.338 kW, NMAE 0.5417, and prediction coverage 100.00%. No machine-learning, foundation, probabilistic, or optimization model was run.

## Frozen H3 protocol

The benchmark forecasts signed grid exchange at 15-minute resolution at 13:00 Europe/Berlin on D-1. Positive values are imports and negative values exports. Historical intervals are usable only when interval_end_utc <= forecast_issue_time_utc. Primary targets are OBSERVED_ORIGINAL; 2023 remains the final TEST.

## Baseline definitions

B0 holds the most recent completed observed-original value at issuance. B1 searches backward by local calendar date for the most recent safe matching quarter-hour and preserves autumn fold identity. B2 uses exactly seven local calendar days earlier and never falls back. B3 uses the development-selected median weekly analogue (B3_N8_MIN4). B4 uses a training-only robust calendar climatology (B4_C3_MONTH_QUARTER_WEEKEND) with explicit broadening to a global training median. B5 is the fixed development-selected ensemble (E4_MEAN_B1_B2_B3_B4_MIN3). Negative predictions are retained.

## Leakage controls

The selection code cannot score 2023 until the frozen selection JSON exists. B4 statistics use only each fold's training years; historical sources must be OBSERVED_ORIGINAL and completed by issuance. The leakage audit contains 0 violations.

## Development Fold A

Fold A trains B4 on 2020 and validates on 2021. Full scores and coverage are in development_baseline_summary.csv.

## Development Fold B

Fold B trains B4 on 2020–2021 and validates on 2022. P1 selection concatenates both validation years with equal target weighting.

Concatenated development evidence:

| baseline_id   |   mae_kw |   rmse_kw |   bias_kw |   nmae |   prediction_coverage |   complete_day_coverage |
|:--------------|---------:|----------:|----------:|-------:|----------------------:|------------------------:|
| B0            |  158.421 |   205.205 |   -93.386 |  0.989 |                 1.000 |                   1.000 |
| B1            |   82.974 |   122.749 |     0.264 |  0.518 |                 1.000 |                   1.000 |
| B2            |   86.057 |   120.698 |     0.895 |  0.535 |                 0.983 |                   0.956 |
| B3            |   76.912 |   104.276 |    -1.248 |  0.480 |                 1.000 |                   0.997 |
| B4            |   85.580 |   117.802 |    33.664 |  0.534 |                 1.000 |                   1.000 |
| B5            |   68.569 |    93.222 |     8.586 |  0.428 |                 1.000 |                   1.000 |

## B3 selection

Candidates N=2/4/8 with all requested weeks and minimum-valid variants 2-of-4 and 4-of-8 were evaluated. The frozen P1 choice is B3_N8_MIN4 using the predefined coverage-aware rule.

## B4 selection

C1–C4 median calendar groupings were evaluated with training-only statistics and explicit fallback hierarchies. The frozen P1 choice is B4_C3_MONTH_QUARTER_WEEKEND.

## B5 selection

Four fixed mean/median ensembles were evaluated. The frozen P1 rule is E4_MEAN_B1_B2_B3_B4_MIN3; if its minimum member count is not met, it explicitly falls back to the selected B4.

## Frozen primary baseline

B5 minimized concatenated P1 validation MAE among candidates meeting at least 99% prediction coverage. Simplicity is the deterministic tie-breaker.

## Final 2023 TEST

The TEST was scored only after selection was serialized. Headline ranking uses MAE together with coverage; RMSE, bias, MedAE, NMAE and absolute-target-scaled WAPE are diagnostic.

| baseline_id   |   mae_kw |   rmse_kw |   bias_kw |   nmae |   prediction_coverage |   complete_day_coverage |
|:--------------|---------:|----------:|----------:|-------:|----------------------:|------------------------:|
| B0            |  153.040 |   198.779 |   -72.129 |  1.084 |                 1.000 |                   1.000 |
| B1            |   93.935 |   135.821 |    -0.158 |  0.665 |                 1.000 |                   1.000 |
| B2            |  100.106 |   136.968 |     0.625 |  0.707 |                 0.974 |                   0.945 |
| B3            |   85.798 |   113.992 |    -0.659 |  0.608 |                 1.000 |                   1.000 |
| B4            |   85.315 |   109.960 |    26.642 |  0.604 |                 1.000 |                   1.000 |
| B5            |   76.480 |   101.338 |     6.649 |  0.542 |                 1.000 |                   1.000 |

## Prediction coverage

Missing predictions are not silently filled. Every summary reports predicted targets and full-day coverage. B5 fallback use is explicit.

## Lead-time performance

The frozen H1–H4 bands are reported without retuning.

| lead_band   |   eligible_targets |   mae_kw |   rmse_kw |   bias_kw |   prediction_coverage |
|:------------|-------------------:|---------:|----------:|----------:|----------------------:|
| H1          |               8543 |   59.312 |    76.811 |    12.073 |                 1.000 |
| H2          |               8526 |   79.178 |   103.676 |     4.110 |                 1.000 |
| H3          |               8554 |  104.032 |   133.318 |     1.309 |                 1.000 |
| H4          |               8538 |   63.359 |    81.512 |     9.107 |                 1.000 |

## Import/export performance

Import-only and export-only MAE, sign accuracy, recalls, false-mode rates and zero-crossing error use epsilon=0 kW.

| baseline_id   | mode            |   count |   mae_kw |   sign_accuracy |   import_recall |   export_recall |   false_import_rate |   false_export_rate |   zero_crossing_error |
|:--------------|:----------------|--------:|---------:|----------------:|----------------:|----------------:|--------------------:|--------------------:|----------------------:|
| B5            | IMPORT          |   29526 |   69.349 |         nan     |         nan     |         nan     |             nan     |             nan     |               nan     |
| B5            | EXPORT          |    4635 |  121.900 |         nan     |         nan     |         nan     |             nan     |             nan     |               nan     |
| B5            | ALL_DIAGNOSTICS |   34161 |  nan     |           0.885 |           0.976 |           0.308 |               0.692 |               0.024 |                 0.115 |

## Calendar contexts

Normal weekdays, weekends, holidays, adjacent-to-holiday days and descriptive bridge days are reported with counts. Small holiday groups are not overinterpreted.

| calendar_context           |   eligible_targets |   mae_kw |   rmse_kw |   bias_kw |   prediction_coverage |
|:---------------------------|-------------------:|---------:|----------:|----------:|----------------------:|
| BRIDGE_DAY_DIAGNOSTIC_ONLY |                288 |   82.696 |   110.153 |    50.314 |                 1.000 |
| DAY_AFTER_PUBLIC_HOLIDAY   |                864 |   78.186 |   100.394 |    49.838 |                 1.000 |
| DAY_BEFORE_PUBLIC_HOLIDAY  |                768 |   79.731 |    99.123 |    52.606 |                 1.000 |
| NORMAL_WEEKDAY             |              24450 |   79.607 |   105.942 |     1.398 |                 1.000 |
| PUBLIC_HOLIDAY             |                864 |   80.520 |   108.766 |    60.000 |                 1.000 |
| WEEKEND                    |               9711 |   68.606 |    88.691 |    19.870 |                 1.000 |

## Monthly/seasonal performance

Monthly MAE, RMSE, bias, import/export prevalence and coverage are descriptive TEST diagnostics only.

|   month |   eligible_targets |   mae_kw |   rmse_kw |   bias_kw |   export_fraction |
|--------:|-------------------:|---------:|----------:|----------:|------------------:|
|   1.000 |           2975.000 |   60.971 |    73.908 |    17.101 |             0.168 |
|   2.000 |           2688.000 |   68.121 |    80.815 |   -12.559 |             0.142 |
|   3.000 |           2972.000 |   90.180 |   110.637 |    -5.843 |             0.174 |
|   4.000 |           2713.000 |   99.967 |   127.015 |    33.502 |             0.244 |
|   5.000 |           2733.000 |   86.203 |   112.126 |    17.089 |             0.243 |
|   6.000 |           2780.000 |   65.881 |   101.415 |    -7.365 |             0.182 |
|   7.000 |           2974.000 |   75.538 |   108.843 |    -6.964 |             0.118 |
|   8.000 |           2976.000 |   73.255 |   107.278 |    -2.595 |             0.092 |
|   9.000 |           2880.000 |   70.570 |    98.462 |     4.324 |             0.105 |
|  10.000 |           2707.000 |   94.565 |   114.081 |    27.911 |             0.080 |
|  11.000 |           2880.000 |   75.159 |    92.399 |     4.897 |             0.052 |
|  12.000 |           2883.000 |   59.593 |    76.071 |    12.687 |             0.041 |

## Structural diagnostics

Pre-June and June–December 2023 are evaluation-only modernization strata. They are not used for selection or causal claims.

| structural_stratum   |   eligible_targets |   mae_kw |   rmse_kw |   bias_kw |
|:---------------------|-------------------:|---------:|----------:|----------:|
| post_modernization   |              20080 |   73.372 |   100.422 |     4.471 |
| pre_modernization    |              14081 |   80.912 |   102.630 |     9.754 |

## Complete-day performance

Daily profile metrics use only complete primary days with full predictions and include interval error and signed/import/export energy errors.

| statistic                       |     mean |   median |      p90 |   maximum |
|:--------------------------------|---------:|---------:|---------:|----------:|
| daily_mae_kw                    |   76.424 |   74.016 |  108.660 |   212.406 |
| daily_rmse_kw                   |   95.938 |   93.270 |  132.362 |   261.166 |
| daily_bias_kw                   |    6.458 |   11.285 |   76.394 |   171.165 |
| max_absolute_interval_error_kw  |  244.936 |  226.678 |  361.257 |   689.390 |
| signed_daily_energy_error_kwh   |  155.144 |  270.830 | 1833.454 |  4107.964 |
| absolute_daily_energy_error_kwh | 1088.430 |  947.118 | 2176.954 |  4151.115 |
| import_energy_error_kwh         |  -19.435 |   83.406 | 1498.899 |  2948.186 |
| export_energy_error_kwh         | -174.579 |  -26.561 |    0.000 |   720.585 |

## P1 vs P2 sensitivity

P2 independently selected B5 after excluding 2020 from development. It is a sensitivity result and does not replace P1 post hoc.

## Bootstrap comparisons

Paired comparisons use 2,000 whole-local-day resamples with seed 20261029 on common fully predicted days.

| primary_baseline   | comparator   |   common_complete_days |   paired_mean_daily_mae_difference_kw |   ci95_lower_kw |   ci95_upper_kw |   primary_daily_win_rate |   bootstrap_resamples |     seed |
|:-------------------|:-------------|-----------------------:|--------------------------------------:|----------------:|----------------:|-------------------------:|----------------------:|---------:|
| B5                 | B0           |                    347 |                               -75.870 |         -86.005 |         -65.301 |                    0.778 |                  2000 | 20261029 |
| B5                 | B1           |                    347 |                               -16.833 |         -20.337 |         -13.296 |                    0.689 |                  2000 | 20261029 |
| B5                 | B2           |                    342 |                               -23.620 |         -26.782 |         -20.724 |                    0.810 |                  2000 | 20261029 |
| B5                 | B3           |                    347 |                                -9.451 |         -11.363 |          -7.408 |                    0.706 |                  2000 | 20261029 |
| B5                 | B4           |                    347 |                                -8.998 |         -11.273 |          -6.694 |                    0.614 |                  2000 | 20261029 |

## DST audit

Spring 92-interval and autumn 100-interval target days preserve unique UTC keys. B1/B2/B3 use local quarter-hour plus fold, so the repeated autumn hour is not collapsed.

## Comparison with SPARK baseline behavior

Honda forecasts signed net grid exchange, while SPARK is consumption-like demand. Raw kW MAE is not a fair cross-site comparison. Later comparison should focus on normalized errors, baseline ranking, temporal generalization, regime robustness and downstream value.

## What the baseline results imply for ML

Any later deterministic ML method must beat this leakage-safe, coverage-aware benchmark on the same frozen targets and common prediction subsets.

## Limitations

These baselines use only historical grid exchange and calendar statistics. They do not use future weather, future PV/CHP realization, probabilistic scenarios, or optimization.

## Recommended H5

H5 should be a deterministic machine-learning benchmark under the frozen Honda protocol: LightGBM with M0 first, followed by an M1 historical-context ablation. Foundation models should follow a strong site-specific deterministic benchmark.
