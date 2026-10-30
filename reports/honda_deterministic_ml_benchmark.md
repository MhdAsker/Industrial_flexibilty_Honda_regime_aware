# Honda deterministic machine-learning benchmark

## Executive summary

The frozen deterministic LightGBM benchmark achieved 71.513 kW MAE on untouched 2023, versus 76.480 kW for frozen B5. The evidence classification is **Category B**. Negative predictions were preserved; prediction coverage was 100.00%. No foundation, neural, probabilistic, quantile, or optimization model was run.

## H4 baseline checkpoint

H4 commit `11a490b6bff6a0ce12f6f67d686abd4355d3185b`, protocol hash `281f49331bc3b3c31aec0d33b50f6a83a149261db93ce04a29d67479cb62676e`, and baseline-selection hash `1bbbbac404675bb1bfb5c85067e1504fda843fc2bc52151a60025f4e3c8b7a04` were verified. Stored B5 predictions were loaded unchanged.

## Scientific question

H5 tests whether site-specific leakage-safe deterministic ML materially improves signed grid-exchange forecasts over B5.

## Frozen forecasting protocol

All models use the H3 13:00 D-1 issuance, physical 15-minute target days, OBSERVED_ORIGINAL labels, P1/P2 chronology, and interval-end cutoff.

## Feature availability and leakage controls

Historical features are computed separately at every historical issuance. Missing values stay NaN. PV/CHP use OBSERVED_ORIGINAL masks. Weather and thermal ablations are not evaluable because H2 did not establish primary provenance masks; they were not consumed.

## M0 feature design and ablation

|   rows |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |   coverage |   train_rows | validation_year   |   best_iteration | candidate   | fold         |
|-------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|-----------:|-------------:|:------------------|-----------------:|:------------|:-------------|
|  34281 |   85.179 |   125.461 |    43.749 |                     54.282 |  0.545 |  0.545 |      1.000 |    24145.000 | 2021              |              114 | M0-A        | FOLD_A       |
|  34624 |   76.032 |    97.312 |   -13.566 |                     64.300 |  0.464 |  0.464 |      1.000 |    58426.000 | 2022              |               88 | M0-A        | FOLD_B       |
|  68905 |   80.583 |   112.203 |    14.949 |                     59.150 |  0.503 |  0.503 |      1.000 |      nan     | 2021+2022         |              101 | M0-A        | CONCATENATED |
|  34281 |   67.443 |    92.977 |    13.299 |                     49.171 |  0.431 |  0.431 |      1.000 |    24145.000 | 2021              |              124 | M0-B        | FOLD_A       |
|  34624 |   69.054 |    90.865 |    -5.317 |                     55.007 |  0.422 |  0.422 |      1.000 |    58426.000 | 2022              |              109 | M0-B        | FOLD_B       |
|  68905 |   68.253 |    91.922 |     3.945 |                     51.849 |  0.426 |  0.426 |      1.000 |      nan     | 2021+2022         |              116 | M0-B        | CONCATENATED |
|  34281 |   68.293 |    94.301 |    17.059 |                     50.652 |  0.437 |  0.437 |      1.000 |    24145.000 | 2021              |              116 | M0-C        | FOLD_A       |
|  34624 |   68.201 |    90.739 |    -3.378 |                     53.198 |  0.416 |  0.416 |      1.000 |    58426.000 | 2022              |              103 | M0-C        | FOLD_B       |
|  68905 |   68.247 |    92.528 |     6.790 |                     51.929 |  0.426 |  0.426 |      1.000 |      nan     | 2021+2022         |              110 | M0-C        | CONCATENATED |
|  34281 |   67.689 |    92.950 |    14.627 |                     50.751 |  0.433 |  0.433 |      1.000 |    24145.000 | 2021              |              141 | M0-D        | FOLD_A       |
|  34624 |   67.978 |    90.392 |    -4.665 |                     53.534 |  0.415 |  0.415 |      1.000 |    58426.000 | 2022              |              143 | M0-D        | FOLD_B       |
|  68905 |   67.834 |    91.673 |     4.933 |                     52.048 |  0.424 |  0.424 |      1.000 |      nan     | 2021+2022         |              142 | M0-D        | CONCATENATED |

## LightGBM development selection

|   rows |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |   coverage |   train_rows | validation_year   |   best_iteration | candidate   | fold         |   num_leaves |   learning_rate |   min_child_samples |   feature_fraction |   reg_lambda |   reg_alpha |
|-------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|-----------:|-------------:|:------------------|-----------------:|:------------|:-------------|-------------:|----------------:|--------------------:|-------------------:|-------------:|------------:|
|  34281 |   66.558 |    92.174 |    13.803 |                     48.601 |  0.426 |  0.426 |      1.000 |    24145.000 | 2021              |              191 | HP01        | FOLD_A       |           15 |           0.030 |                  50 |              0.800 |        0.000 |       0.000 |
|  34624 |   67.788 |    90.780 |    -4.556 |                     51.963 |  0.414 |  0.414 |      1.000 |    58426.000 | 2022              |              259 | HP01        | FOLD_B       |           15 |           0.030 |                  50 |              0.800 |        0.000 |       0.000 |
|  68905 |   67.176 |    91.476 |     4.578 |                     50.240 |  0.420 |  0.420 |      1.000 |      nan     | 2021+2022         |              225 | HP01        | CONCATENATED |           15 |           0.030 |                  50 |              0.800 |        0.000 |       0.000 |
|  34281 |   67.918 |    94.100 |    18.461 |                     49.886 |  0.434 |  0.434 |      1.000 |    24145.000 | 2021              |               96 | HP02        | FOLD_A       |           15 |           0.050 |                 100 |              1.000 |        0.100 |       0.000 |
|  34624 |   67.985 |    90.806 |    -3.647 |                     52.330 |  0.415 |  0.415 |      1.000 |    58426.000 | 2022              |              123 | HP02        | FOLD_B       |           15 |           0.050 |                 100 |              1.000 |        0.100 |       0.000 |
|  68905 |   67.952 |    92.459 |     7.352 |                     51.091 |  0.424 |  0.424 |      1.000 |      nan     | 2021+2022         |              110 | HP02        | CONCATENATED |           15 |           0.050 |                 100 |              1.000 |        0.100 |       0.000 |
|  34281 |   67.059 |    93.230 |    16.914 |                     48.754 |  0.429 |  0.429 |      1.000 |    24145.000 | 2021              |               81 | HP03        | FOLD_A       |           15 |           0.050 |                 250 |              0.800 |        1.000 |       0.100 |
|  34624 |   67.771 |    90.592 |    -4.379 |                     52.128 |  0.414 |  0.414 |      1.000 |    58426.000 | 2022              |              144 | HP03        | FOLD_B       |           15 |           0.050 |                 250 |              0.800 |        1.000 |       0.100 |
|  68905 |   67.417 |    91.914 |     6.215 |                     50.282 |  0.421 |  0.421 |      1.000 |      nan     | 2021+2022         |              112 | HP03        | CONCATENATED |           15 |           0.050 |                 250 |              0.800 |        1.000 |       0.100 |
|  34281 |   68.593 |    93.763 |    19.362 |                     51.604 |  0.439 |  0.439 |      1.000 |    24145.000 | 2021              |              158 | HP04        | FOLD_A       |           31 |           0.030 |                 100 |              1.000 |        1.000 |       0.000 |
|  34624 |   68.071 |    90.408 |    -4.029 |                     53.399 |  0.416 |  0.416 |      1.000 |    58426.000 | 2022              |              187 | HP04        | FOLD_B       |           31 |           0.030 |                 100 |              1.000 |        1.000 |       0.000 |
|  68905 |   68.331 |    92.092 |     7.608 |                     52.454 |  0.427 |  0.427 |      1.000 |      nan     | 2021+2022         |              172 | HP04        | CONCATENATED |           31 |           0.030 |                 100 |              1.000 |        1.000 |       0.000 |
|  34281 |   68.523 |    93.848 |    18.629 |                     51.753 |  0.438 |  0.438 |      1.000 |    24145.000 | 2021              |              117 | HP05        | FOLD_A       |           31 |           0.050 |                  50 |              0.800 |        0.100 |       0.100 |
|  34624 |   67.964 |    90.362 |    -4.326 |                     53.594 |  0.415 |  0.415 |      1.000 |    58426.000 | 2022              |              128 | HP05        | FOLD_B       |           31 |           0.050 |                  50 |              0.800 |        0.100 |       0.100 |
|  68905 |   68.242 |    92.113 |     7.094 |                     52.680 |  0.426 |  0.426 |      1.000 |      nan     | 2021+2022         |              122 | HP05        | CONCATENATED |           31 |           0.050 |                  50 |              0.800 |        0.100 |       0.100 |
|  34281 |   68.822 |    94.273 |    21.158 |                     51.871 |  0.440 |  0.440 |      1.000 |    24145.000 | 2021              |               97 | HP06        | FOLD_A       |           31 |           0.050 |                 250 |              1.000 |        0.000 |       0.000 |
|  34624 |   67.919 |    90.453 |    -4.279 |                     52.732 |  0.415 |  0.415 |      1.000 |    58426.000 | 2022              |               96 | HP06        | FOLD_B       |           31 |           0.050 |                 250 |              1.000 |        0.000 |       0.000 |
|  68905 |   68.368 |    92.373 |     8.376 |                     52.325 |  0.427 |  0.427 |      1.000 |      nan     | 2021+2022         |               96 | HP06        | CONCATENATED |           31 |           0.050 |                 250 |              1.000 |        0.000 |       0.000 |
|  34281 |   70.689 |    96.463 |    21.853 |                     54.120 |  0.452 |  0.452 |      1.000 |    24145.000 | 2021              |              244 | HP07        | FOLD_A       |           63 |           0.030 |                  50 |              1.000 |        0.100 |       0.000 |
|  34624 |   68.009 |    90.240 |    -2.896 |                     53.458 |  0.415 |  0.415 |      1.000 |    58426.000 | 2022              |              190 | HP07        | FOLD_B       |           63 |           0.030 |                  50 |              1.000 |        0.100 |       0.000 |
|  68905 |   69.342 |    93.388 |     9.417 |                     53.819 |  0.433 |  0.433 |      1.000 |      nan     | 2021+2022         |              217 | HP07        | CONCATENATED |           63 |           0.030 |                  50 |              1.000 |        0.100 |       0.000 |
|  34281 |   68.772 |    94.600 |    20.345 |                     51.690 |  0.440 |  0.440 |      1.000 |    24145.000 | 2021              |              118 | HP08        | FOLD_A       |           63 |           0.030 |                 250 |              0.800 |        1.000 |       0.100 |
|  34624 |   67.740 |    89.938 |    -3.120 |                     53.694 |  0.414 |  0.414 |      1.000 |    58426.000 | 2022              |              151 | HP08        | FOLD_B       |           63 |           0.030 |                 250 |              0.800 |        1.000 |       0.100 |
|  68905 |   68.253 |    92.287 |     8.554 |                     52.621 |  0.426 |  0.426 |      1.000 |      nan     | 2021+2022         |              134 | HP08        | CONCATENATED |           63 |           0.030 |                 250 |              0.800 |        1.000 |       0.100 |
|  34281 |   70.740 |    97.654 |    25.022 |                     53.054 |  0.452 |  0.452 |      1.000 |    24145.000 | 2021              |              101 | HP09        | FOLD_A       |           63 |           0.050 |                 100 |              1.000 |        0.000 |       0.100 |
|  34624 |   67.858 |    89.916 |    -2.291 |                     53.247 |  0.414 |  0.414 |      1.000 |    58426.000 | 2022              |               92 | HP09        | FOLD_B       |           63 |           0.050 |                 100 |              1.000 |        0.000 |       0.100 |
|  68905 |   69.292 |    93.845 |    11.297 |                     53.173 |  0.433 |  0.433 |      1.000 |      nan     | 2021+2022         |               96 | HP09        | CONCATENATED |           63 |           0.050 |                 100 |              1.000 |        0.000 |       0.100 |

## M1 historical-context design and ablation

|      rows |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |    nmae |    wape |   coverage |   train_rows | validation_year   |   best_iteration | candidate   | fold         | status        | reason                                       |
|----------:|---------:|----------:|----------:|---------------------------:|--------:|--------:|-----------:|-------------:|:------------------|-----------------:|:------------|:-------------|:--------------|:---------------------------------------------|
| 34281.000 |   68.857 |    95.855 |    23.717 |                     50.004 |   0.440 |   0.440 |      1.000 |    24145.000 | 2021              |          150.000 | M1-PV       | FOLD_A       | nan           | nan                                          |
| 34624.000 |   68.377 |    91.103 |    -4.140 |                     52.829 |   0.417 |   0.417 |      1.000 |    58426.000 | 2022              |          193.000 | M1-PV       | FOLD_B       | nan           | nan                                          |
| 68905.000 |   68.616 |    93.497 |     9.719 |                     51.331 |   0.429 |   0.429 |      1.000 |      nan     | 2021+2022         |          172.000 | M1-PV       | CONCATENATED | nan           | nan                                          |
| 34281.000 |   66.536 |    91.736 |    15.899 |                     49.394 |   0.425 |   0.425 |      1.000 |    24145.000 | 2021              |          158.000 | M1-CHP      | FOLD_A       | nan           | nan                                          |
| 34624.000 |   66.524 |    89.684 |    -1.106 |                     49.926 |   0.406 |   0.406 |      1.000 |    58426.000 | 2022              |          234.000 | M1-CHP      | FOLD_B       | nan           | nan                                          |
| 68905.000 |   66.530 |    90.710 |     7.354 |                     49.679 |   0.416 |   0.416 |      1.000 |      nan     | 2021+2022         |          196.000 | M1-CHP      | CONCATENATED | nan           | nan                                          |
| 34281.000 |   68.975 |    96.113 |    23.835 |                     50.195 |   0.441 |   0.441 |      1.000 |    24145.000 | 2021              |          139.000 | M1-ALL      | FOLD_A       | nan           | nan                                          |
| 34624.000 |   67.258 |    90.415 |    -1.711 |                     50.957 |   0.411 |   0.411 |      1.000 |    58426.000 | 2022              |          196.000 | M1-ALL      | FOLD_B       | nan           | nan                                          |
| 68905.000 |   68.112 |    93.293 |    10.998 |                     50.595 |   0.425 |   0.425 |      1.000 |      nan     | 2021+2022         |          168.000 | M1-ALL      | CONCATENATED | nan           | nan                                          |
|   nan     |  nan     |   nan     |   nan     |                    nan     | nan     | nan     |    nan     |      nan     | nan               |          nan     | M1-W        | FOLD_A       | NOT_EVALUABLE | H2 weather provenance masks unavailable      |
|   nan     |  nan     |   nan     |   nan     |                    nan     | nan     | nan     |    nan     |      nan     | nan               |          nan     | M1-W        | FOLD_B       | NOT_EVALUABLE | H2 weather provenance masks unavailable      |
|   nan     |  nan     |   nan     |   nan     |                    nan     | nan     | nan     |    nan     |      nan     | nan               |          nan     | M1-W        | CONCATENATED | NOT_EVALUABLE | H2 weather provenance masks unavailable      |
|   nan     |  nan     |   nan     |   nan     |                    nan     | nan     | nan     |    nan     |      nan     | nan               |          nan     | M1-T        | FOLD_A       | NOT_EVALUABLE | Primary thermal provenance masks unavailable |
|   nan     |  nan     |   nan     |   nan     |                    nan     | nan     | nan     |    nan     |      nan     | nan               |          nan     | M1-T        | FOLD_B       | NOT_EVALUABLE | Primary thermal provenance masks unavailable |
|   nan     |  nan     |   nan     |   nan     |                    nan     | nan     | nan     |    nan     |      nan     | nan               |          nan     | M1-T        | CONCATENATED | NOT_EVALUABLE | Primary thermal provenance masks unavailable |

## Frozen deterministic model

Final information set: M1; feature family: M1-CHP; iterations: 196; objective: regression_l1.

## Final 2023 TEST and comparison with B5

| scope        | model      |   rows |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |   coverage |
|:-------------|:-----------|-------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|-----------:|
| P1_TEST_2023 | B5         |  34161 |   76.480 |   101.338 |     6.649 |                     60.345 |  0.542 |  0.542 |      1.000 |
| P1_TEST_2023 | LGBM_M0    |  34161 |   72.042 |    97.061 |     6.119 |                     53.974 |  0.510 |  0.510 |      1.000 |
| P1_TEST_2023 | LGBM_M1    |  34161 |   71.513 |    96.974 |     9.408 |                     52.618 |  0.507 |  0.507 |      1.000 |
| P1_TEST_2023 | PRIMARY_ML |  34161 |   71.513 |    96.974 |     9.408 |                     52.618 |  0.507 |  0.507 |      1.000 |

## Lead-time performance

| model      | lead_band   |   rows |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |   coverage |
|:-----------|:------------|-------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|-----------:|
| B5         | H1          |   8543 |   59.312 |    76.811 |    12.073 |                     47.807 |  0.434 |  0.434 |      1.000 |
| PRIMARY_ML | H1          |   8543 |   53.564 |    72.201 |    13.375 |                     39.503 |  0.392 |  0.392 |      1.000 |
| B5         | H2          |   8526 |   79.178 |   103.676 |     4.110 |                     62.236 |  0.587 |  0.587 |      1.000 |
| PRIMARY_ML | H2          |   8526 |   74.146 |    99.065 |     3.972 |                     55.755 |  0.550 |  0.550 |      1.000 |
| B5         | H3          |   8554 |  104.032 |   133.318 |     1.309 |                     84.427 |  0.721 |  0.721 |      1.000 |
| PRIMARY_ML | H3          |   8554 |   98.781 |   128.585 |     9.813 |                     77.734 |  0.685 |  0.685 |      1.000 |
| B5         | H4          |   8538 |   63.359 |    81.512 |     9.107 |                     51.151 |  0.426 |  0.426 |      1.000 |
| PRIMARY_ML | H4          |   8538 |   59.522 |    77.748 |    10.459 |                     46.465 |  0.400 |  0.400 |      1.000 |

## Import/export performance

| model      |   import_mae_kw |   export_mae_kw |   sign_accuracy |   import_recall |   export_recall |   false_import_rate |   false_export_rate |   zero_crossing_error |
|:-----------|----------------:|----------------:|----------------:|----------------:|----------------:|--------------------:|--------------------:|----------------------:|
| B5         |          69.349 |         121.900 |           0.885 |           0.976 |           0.308 |               0.692 |               0.024 |                 0.115 |
| PRIMARY_ML |          63.685 |         121.377 |           0.888 |           0.981 |           0.298 |               0.702 |               0.019 |                 0.112 |

## Calendar contexts

| model      | calendar_context           |   rows |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |   coverage |
|:-----------|:---------------------------|-------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|-----------:|
| B5         | BRIDGE_DAY_DIAGNOSTIC_ONLY |    288 |   82.696 |   110.153 |    50.314 |                     66.192 |  0.600 |  0.600 |      1.000 |
| PRIMARY_ML | BRIDGE_DAY_DIAGNOSTIC_ONLY |    288 |   78.869 |   106.977 |    54.952 |                     54.373 |  0.572 |  0.572 |      1.000 |
| B5         | DAY_AFTER_PUBLIC_HOLIDAY   |    864 |   78.186 |   100.394 |    49.838 |                     60.530 |  0.788 |  0.788 |      1.000 |
| PRIMARY_ML | DAY_AFTER_PUBLIC_HOLIDAY   |    864 |   71.983 |    94.827 |    38.877 |                     57.251 |  0.726 |  0.726 |      1.000 |
| B5         | DAY_BEFORE_PUBLIC_HOLIDAY  |    768 |   79.731 |    99.123 |    52.606 |                     65.851 |  0.828 |  0.828 |      1.000 |
| PRIMARY_ML | DAY_BEFORE_PUBLIC_HOLIDAY  |    768 |   69.185 |    89.795 |    43.622 |                     53.397 |  0.719 |  0.719 |      1.000 |
| B5         | NORMAL_WEEKDAY             |  24450 |   79.607 |   105.942 |     1.398 |                     62.189 |  0.524 |  0.524 |      1.000 |
| PRIMARY_ML | NORMAL_WEEKDAY             |  24450 |   74.808 |   101.369 |     6.314 |                     55.175 |  0.492 |  0.492 |      1.000 |
| B5         | PUBLIC_HOLIDAY             |    864 |   80.520 |   108.766 |    60.000 |                     58.782 |  0.805 |  0.805 |      1.000 |
| PRIMARY_ML | PUBLIC_HOLIDAY             |    864 |   72.301 |   105.210 |    53.570 |                     47.125 |  0.723 |  0.723 |      1.000 |
| B5         | WEEKEND                    |   9711 |   68.606 |    88.691 |    19.870 |                     56.189 |  0.602 |  0.602 |      1.000 |
| PRIMARY_ML | WEEKEND                    |   9711 |   63.216 |    84.907 |    17.196 |                     47.205 |  0.555 |  0.555 |      1.000 |

## Monthly/seasonal performance

|   month |   export_fraction |   B5_mae_kw |   LGBM_M0_mae_kw |   LGBM_M0_rmse_kw |   LGBM_M0_bias_kw |   LGBM_M0_improvement_vs_B5_kw |   LGBM_M1_mae_kw |   LGBM_M1_rmse_kw |   LGBM_M1_bias_kw |   LGBM_M1_improvement_vs_B5_kw |   PRIMARY_ML_mae_kw |   PRIMARY_ML_rmse_kw |   PRIMARY_ML_bias_kw |   PRIMARY_ML_improvement_vs_B5_kw |
|--------:|------------------:|------------:|-----------------:|------------------:|------------------:|-------------------------------:|-----------------:|------------------:|------------------:|-------------------------------:|--------------------:|---------------------:|---------------------:|----------------------------------:|
|   1.000 |             0.168 |      60.971 |           56.417 |            70.768 |             6.296 |                          4.555 |           56.625 |            70.548 |             8.367 |                          4.346 |              56.625 |               70.548 |                8.367 |                             4.346 |
|   2.000 |             0.142 |      68.121 |           55.099 |            67.937 |            -8.242 |                         13.023 |           50.460 |            64.697 |            -2.874 |                         17.661 |              50.460 |               64.697 |               -2.874 |                            17.661 |
|   3.000 |             0.174 |      90.180 |           79.970 |           103.481 |             3.278 |                         10.210 |           76.172 |           102.135 |             8.845 |                         14.008 |              76.172 |              102.135 |                8.845 |                            14.008 |
|   4.000 |             0.244 |      99.967 |           94.197 |           122.314 |            27.986 |                          5.770 |           91.830 |           119.916 |            21.584 |                          8.137 |              91.830 |              119.916 |               21.584 |                             8.137 |
|   5.000 |             0.243 |      86.203 |           82.216 |           109.057 |            19.634 |                          3.987 |           82.670 |           110.165 |            22.713 |                          3.533 |              82.670 |              110.165 |               22.713 |                             3.533 |
|   6.000 |             0.182 |      65.881 |           67.973 |           102.088 |             6.113 |                         -2.092 |           68.329 |           101.885 |             9.373 |                         -2.447 |              68.329 |              101.885 |                9.373 |                            -2.447 |
|   7.000 |             0.118 |      75.538 |           74.295 |           104.104 |             5.032 |                          1.244 |           74.255 |           104.890 |             9.244 |                          1.283 |              74.255 |              104.890 |                9.244 |                             1.283 |
|   8.000 |             0.092 |      73.255 |           73.589 |           104.561 |             2.202 |                         -0.334 |           74.521 |           105.767 |            12.621 |                         -1.266 |              74.521 |              105.767 |               12.621 |                            -1.266 |
|   9.000 |             0.105 |      70.570 |           69.390 |            95.721 |            -0.238 |                          1.180 |           70.512 |            96.921 |             5.307 |                          0.057 |              70.512 |               96.921 |                5.307 |                             0.057 |
|  10.000 |             0.080 |      94.565 |           89.129 |           108.873 |            11.308 |                          5.436 |           89.953 |           109.186 |            21.383 |                          4.612 |              89.953 |              109.186 |               21.383 |                             4.612 |
|  11.000 |             0.052 |      75.159 |           74.502 |            92.169 |             2.247 |                          0.658 |           74.698 |            92.582 |             0.473 |                          0.461 |              74.698 |               92.582 |                0.473 |                             0.461 |
|  12.000 |             0.041 |      59.593 |           49.448 |            66.947 |            -0.613 |                         10.146 |           49.642 |            67.482 |            -2.898 |                          9.951 |              49.642 |               67.482 |               -2.898 |                             9.951 |

## Modernization diagnostic

| model      | stratum            |   rows |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |   coverage |
|:-----------|:-------------------|-------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|-----------:|
| B5         | June_December_2023 |  20080 |   73.372 |   100.422 |     4.471 |                     52.320 |  0.470 |  0.470 |      1.000 |
| PRIMARY_ML | June_December_2023 |  20080 |   71.583 |    97.786 |     7.833 |                     50.062 |  0.458 |  0.458 |      1.000 |
| B5         | pre_June_2023      |  14081 |   80.912 |   102.630 |     9.754 |                     69.985 |  0.676 |  0.676 |      1.000 |
| PRIMARY_ML | pre_June_2023      |  14081 |   71.412 |    95.804 |    11.653 |                     55.410 |  0.597 |  0.597 |      1.000 |

## Daily-profile performance

| model      |   daily_mae_mean |   daily_mae_median |   daily_mae_p90 |   daily_mae_max |   daily_rmse_mean |   absolute_daily_energy_error_mean |   signed_daily_energy_error_mean |
|:-----------|-----------------:|-------------------:|----------------:|----------------:|------------------:|-----------------------------------:|---------------------------------:|
| B5         |           76.424 |             74.016 |         108.660 |         212.406 |            95.938 |                           1088.430 |                          155.144 |
| PRIMARY_ML |           71.302 |             68.437 |         104.780 |         211.768 |            90.989 |                            968.905 |                          215.656 |

## Bootstrap significance

| ML_model   | comparator   |   common_complete_days |   mean_daily_mae_difference_kw |   ci95_lower_kw |   ci95_upper_kw |   ML_daily_win_rate |   resamples |     seed |
|:-----------|:-------------|-----------------------:|-------------------------------:|----------------:|----------------:|--------------------:|------------:|---------:|
| PRIMARY_ML | B5           |                    347 |                         -5.122 |          -6.833 |          -3.514 |               0.643 |        2000 | 20261029 |
| PRIMARY_ML | B3           |                    347 |                        -14.573 |         -17.249 |         -11.978 |               0.738 |        2000 | 20261029 |

## P2 structural sensitivity

|   rows |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |   coverage | model         |
|-------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|-----------:|:--------------|
|  34161 |   72.175 |    96.998 |     2.314 |                     55.296 |  0.511 |  0.511 |      1.000 | PRIMARY_ML_P2 |

## Feature importance

Importance is predictive, not causal. Gain and split importance are reported by feature and group.

## Residual diagnostics

Residual summaries, frozen development-derived target bins, and conditional ML-minus-B5 errors are saved in deterministic tables. They were inspected only after model freeze.

## Where ML improves over baseline and where ML still fails

The conditional improvement tables identify lead bands, months, modes and target magnitudes where ML helps or degrades. No subgroup-specific retuning was performed.

## Comparison with SPARK deterministic findings

SPARK forecasts consumption-like demand; Honda forecasts signed net grid exchange. Raw kW errors are not treated as cross-site equivalents. Comparisons should emphasize ranking, normalized error, temporal generalization and regime robustness.

## Implications for foundation models

H5 provides a frozen B5 and site-specific ML reference for a future zero-shot foundation-model benchmark.

## Limitations

Historical weather and thermal context could not be admitted without provenance masks. Feature importance is non-causal, and 2023 diagnostics were not used for tuning.

## Recommended H6

Run a zero-shot foundation-model benchmark for Chronos-2 and TimesFM against frozen B5 and LightGBM under the same target, issuance, common-target and DST rules. Do not mix fine-tuning into that first comparison.
