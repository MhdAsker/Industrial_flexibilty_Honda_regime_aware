# Honda zero-shot foundation-model benchmark

## Executive summary

Chronos-2 achieved 103.672 kW TEST MAE and TimesFM 3.0 achieved 95.431 kW, versus 76.480 kW for B5 and 71.513 kW for frozen LightGBM. Both foundation models were evaluated locally and zero-shot; no weights, adapters, gradients, or Honda-specific parameters were trained.

## Scientific question and frozen Honda protocol

The experiment tests pretrained transfer under the frozen 13:00 D-1, 15-minute signed-grid-exchange protocol. Positive values are imports and negative values are exports. Only measurements completed by issuance were supplied.

## Frozen B5 and LightGBM benchmarks

Stored H4/H5 predictions were loaded unchanged. No reference model was rebuilt.

## Foundation-model versions and licensing

- Chronos-2: `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, `chronos-forecasting==2.3.2`, Apache-2.0. Official repository: https://github.com/amazon-science/chronos-forecasting
- TimesFM 3.0: `google/timesfm-3.0-pytorch`, revision `43046b85ec22d584a13f8098c2ed39c889e129c2`, `timesfm==3.0.2`. Source is Apache-2.0; downloaded 3.0 weights use `timesfm-non-commercial-license-v1.0`. Official repository: https://github.com/google-research/timesfm

## Hardware environment and zero-shot definition

Frozen inference ran locally on the NVIDIA GeForce GTX 1650: Chronos-2 used float16 and TimesFM used float32, with 15.7 GiB system RAM available. Historical Honda observations were inference context only; gradient mode was disabled.

## Context construction and missing-history handling

Contexts were exact physical UTC grids ending at `issue_time - 15 minutes`. Chronos used its native NaN mask. TimesFM used its official historical-only linear interpolation preprocessing; interpolated values were not reclassified as observations. The common subset requires valid predictions and at least one observed-original context value for both models.

## Chronos-2 and TimesFM implementation

Chronos-2 used its official median (quantile 0.5), native missing mask, and exact variable target horizons. TimesFM used its official median forecast with `make_positive=False`, which is essential for this signed target, plus the official historical-only interpolation performed inside its evaluator. Both ran locally with weights fixed and gradient mode disabled.

## Development context selection

Chronos-2 context candidates:

|   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw | model    | mode   |   context_days | fold         |   mean_context_observed_fraction |   valid_context_origins |
|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|:---------|:-------|---------------:|:-------------|---------------------------------:|------------------------:|
|              34281 |               34281 |          1 |  115.746 |   159.210 |    -4.181 | CHRONOS2 | U      |              7 | FOLD_A       |                            0.979 |                     365 |
|              34624 |               34624 |          1 |  115.342 |   153.919 |    -4.412 | CHRONOS2 | U      |              7 | FOLD_B       |                            0.990 |                     362 |
|              68905 |               68905 |          1 |  115.543 |   156.574 |    -4.297 | CHRONOS2 | U      |              7 | CONCATENATED |                            0.985 |                     727 |
|              34281 |               34281 |          1 |  115.533 |   158.658 |    -4.375 | CHRONOS2 | U      |             14 | FOLD_A       |                            0.979 |                     365 |
|              34624 |               34624 |          1 |  115.169 |   153.682 |    -5.498 | CHRONOS2 | U      |             14 | FOLD_B       |                            0.989 |                     362 |
|              68905 |               68905 |          1 |  115.350 |   156.178 |    -4.939 | CHRONOS2 | U      |             14 | CONCATENATED |                            0.984 |                     727 |
|              34281 |               34281 |          1 |  114.982 |   158.161 |    -4.818 | CHRONOS2 | U      |             28 | FOLD_A       |                            0.978 |                     365 |
|              34624 |               34624 |          1 |  114.568 |   153.413 |    -5.718 | CHRONOS2 | U      |             28 | FOLD_B       |                            0.988 |                     362 |
|              68905 |               68905 |          1 |  114.774 |   155.794 |    -5.271 | CHRONOS2 | U      |             28 | CONCATENATED |                            0.984 |                     727 |

TimesFM context candidates:

|   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw | model    | mode   |   context_days | fold         |   mean_context_observed_fraction |   valid_context_origins |
|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|:---------|:-------|---------------:|:-------------|---------------------------------:|------------------------:|
|              34281 |               34281 |          1 |  113.474 |   156.315 |    -5.661 | TIMESFM3 | U      |              7 | FOLD_A       |                            0.979 |                     365 |
|              34624 |               34624 |          1 |  114.331 |   151.976 |    -3.859 | TIMESFM3 | U      |              7 | FOLD_B       |                            0.990 |                     362 |
|              68905 |               68905 |          1 |  113.905 |   154.150 |    -4.756 | TIMESFM3 | U      |              7 | CONCATENATED |                            0.985 |                     727 |
|              34281 |               34281 |          1 |  113.051 |   155.658 |    -5.660 | TIMESFM3 | U      |             14 | FOLD_A       |                            0.979 |                     365 |
|              34624 |               34624 |          1 |  113.428 |   151.276 |    -3.660 | TIMESFM3 | U      |             14 | FOLD_B       |                            0.989 |                     362 |
|              68905 |               68905 |          1 |  113.241 |   153.472 |    -4.655 | TIMESFM3 | U      |             14 | CONCATENATED |                            0.984 |                     727 |
|              34281 |               34281 |          1 |  112.569 |   154.767 |    -4.631 | TIMESFM3 | U      |             28 | FOLD_A       |                            0.978 |                     365 |
|              34624 |               34624 |          1 |  112.593 |   150.486 |    -3.446 | TIMESFM3 | U      |             28 | FOLD_B       |                            0.988 |                     362 |
|              68905 |               68905 |          1 |  112.581 |   152.631 |    -4.035 | TIMESFM3 | U      |             28 | CONCATENATED |                            0.984 |                     727 |

The prospective 0.5% tie rule selected 14 days for Chronos-2 and 28 days for TimesFM. No 2023 label informed this choice.

## Covariate ablation and frozen zero-shot configurations

Chronos-2:

|   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw | model    | mode   |   context_days | fold         |   mean_context_observed_fraction |   valid_context_origins |
|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|:---------|:-------|---------------:|:-------------|---------------------------------:|------------------------:|
|              34281 |               34281 |          1 |  115.538 |   158.667 |    -4.375 | CHRONOS2 | U      |             14 | FOLD_A       |                            0.979 |                     365 |
|              34624 |               34624 |          1 |  115.173 |   153.690 |    -5.500 | CHRONOS2 | U      |             14 | FOLD_B       |                            0.989 |                     362 |
|              68905 |               68905 |          1 |  115.355 |   156.186 |    -4.941 | CHRONOS2 | U      |             14 | CONCATENATED |                            0.984 |                     727 |
|              34281 |               34281 |          1 |  104.254 |   142.914 |    -8.279 | CHRONOS2 | C1     |             14 | FOLD_A       |                            0.979 |                     365 |
|              34624 |               34624 |          1 |  105.730 |   143.012 |    -7.297 | CHRONOS2 | C1     |             14 | FOLD_B       |                            0.989 |                     362 |
|              68905 |               68905 |          1 |  104.996 |   142.963 |    -7.785 | CHRONOS2 | C1     |             14 | CONCATENATED |                            0.984 |                     727 |

TimesFM:

|   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw | model    | mode   |   context_days | fold         |   mean_context_observed_fraction |   valid_context_origins |
|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|:---------|:-------|---------------:|:-------------|---------------------------------:|------------------------:|
|              34281 |               34281 |          1 |  112.569 |   154.767 |    -4.631 | TIMESFM3 | U      |             28 | FOLD_A       |                            0.978 |                     365 |
|              34624 |               34624 |          1 |  112.593 |   150.486 |    -3.446 | TIMESFM3 | U      |             28 | FOLD_B       |                            0.988 |                     362 |
|              68905 |               68905 |          1 |  112.581 |   152.631 |    -4.035 | TIMESFM3 | U      |             28 | CONCATENATED |                            0.984 |                     727 |
|              34281 |               34281 |          1 |   92.219 |   128.754 |     2.588 | TIMESFM3 | C1     |             28 | FOLD_A       |                            0.978 |                     365 |
|              34624 |               34624 |          1 |   95.372 |   130.503 |     2.375 | TIMESFM3 | C1     |             28 | FOLD_B       |                            0.988 |                     362 |
|              68905 |               68905 |          1 |   93.803 |   129.636 |     2.481 | TIMESFM3 | C1     |             28 | CONCATENATED |                            0.984 |                     727 |

C1 contains future-known daily/annual cyclic, weekend and holiday covariates plus past-only observed-original CHP. Both C1 variants materially improved development MAE and were frozen. PV and weather were not introduced; this preserves the H3/H5 information contract and avoids an unsupported weather advantage.

Development comparison with the frozen references:

| model    | mode             |   context_days | fold         |   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw |   mean_context_observed_fraction |   valid_context_origins |
|:---------|:-----------------|---------------:|:-------------|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|---------------------------------:|------------------------:|
| B5       | FROZEN_REFERENCE |        nan     | FOLD_A       |              34281 |               34281 |          1 |   67.555 |    93.063 |    16.523 |                          nan     |                 nan     |
| LIGHTGBM | FROZEN_REFERENCE |        nan     | FOLD_A       |              34281 |               34281 |          1 |   66.536 |    91.736 |    15.899 |                          nan     |                 nan     |
| B5       | FROZEN_REFERENCE |        nan     | FOLD_B       |              34624 |               34624 |          1 |   69.573 |    93.379 |     0.727 |                          nan     |                 nan     |
| LIGHTGBM | FROZEN_REFERENCE |        nan     | FOLD_B       |              34624 |               34624 |          1 |   66.524 |    89.684 |    -1.106 |                          nan     |                 nan     |
| B5       | FROZEN_REFERENCE |        nan     | CONCATENATED |              68905 |               68905 |          1 |   68.569 |    93.222 |     8.586 |                          nan     |                 nan     |
| LIGHTGBM | FROZEN_REFERENCE |        nan     | CONCATENATED |              68905 |               68905 |          1 |   66.530 |    90.710 |     7.354 |                          nan     |                 nan     |
| CHRONOS2 | C1               |         14.000 | FOLD_A       |              34281 |               34281 |          1 |  104.254 |   142.914 |    -8.279 |                            0.979 |                 365.000 |
| CHRONOS2 | C1               |         14.000 | FOLD_B       |              34624 |               34624 |          1 |  105.730 |   143.012 |    -7.297 |                            0.989 |                 362.000 |
| CHRONOS2 | C1               |         14.000 | CONCATENATED |              68905 |               68905 |          1 |  104.996 |   142.963 |    -7.785 |                            0.984 |                 727.000 |
| TIMESFM3 | C1               |         28.000 | FOLD_A       |              34281 |               34281 |          1 |   92.219 |   128.754 |     2.588 |                            0.978 |                 365.000 |
| TIMESFM3 | C1               |         28.000 | FOLD_B       |              34624 |               34624 |          1 |   95.372 |   130.503 |     2.375 |                            0.988 |                 362.000 |
| TIMESFM3 | C1               |         28.000 | CONCATENATED |              68905 |               68905 |          1 |   93.803 |   129.636 |     2.481 |                            0.984 |                 727.000 |

## Final 2023 TEST and common-context comparison

| model    | evaluation_set         |   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |
|:---------|:-----------------------|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|
| B5       | MAXIMAL_PROTOCOL_VALID |              34161 |               34161 |      1.000 |   76.480 |   101.338 |     6.649 |                     60.345 |  0.542 |  0.542 |
| LIGHTGBM | MAXIMAL_PROTOCOL_VALID |              34161 |               34161 |      1.000 |   71.513 |    96.974 |     9.408 |                     52.618 |  0.507 |  0.507 |
| CHRONOS2 | MAXIMAL_PROTOCOL_VALID |              34161 |               34161 |      1.000 |  103.672 |   139.959 |    -6.184 |                     76.690 |  0.734 |  0.734 |
| TIMESFM3 | MAXIMAL_PROTOCOL_VALID |              34161 |               34161 |      1.000 |   95.431 |   129.221 |    -0.040 |                     69.993 |  0.676 |  0.676 |

| model    | evaluation_set        |   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |
|:---------|:----------------------|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|
| B5       | COMMON_CONTEXT_SUBSET |              34161 |               34161 |      1.000 |   76.480 |   101.338 |     6.649 |                     60.345 |  0.542 |  0.542 |
| LIGHTGBM | COMMON_CONTEXT_SUBSET |              34161 |               34161 |      1.000 |   71.513 |    96.974 |     9.408 |                     52.618 |  0.507 |  0.507 |
| CHRONOS2 | COMMON_CONTEXT_SUBSET |              34161 |               34161 |      1.000 |  103.672 |   139.959 |    -6.184 |                     76.690 |  0.734 |  0.734 |
| TIMESFM3 | COMMON_CONTEXT_SUBSET |              34161 |               34161 |      1.000 |   95.431 |   129.221 |    -0.040 |                     69.993 |  0.676 |  0.676 |

## Lead-time behavior

| model    | lead_band   |   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |
|:---------|:------------|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|
| B5       | H1          |               8543 |                8543 |      1.000 |   59.312 |    76.811 |    12.073 |                     47.807 |  0.434 |  0.434 |
| B5       | H2          |               8526 |                8526 |      1.000 |   79.178 |   103.676 |     4.110 |                     62.236 |  0.587 |  0.587 |
| B5       | H3          |               8554 |                8554 |      1.000 |  104.032 |   133.318 |     1.309 |                     84.427 |  0.721 |  0.721 |
| B5       | H4          |               8538 |                8538 |      1.000 |   63.359 |    81.512 |     9.107 |                     51.151 |  0.426 |  0.426 |
| LIGHTGBM | H1          |               8543 |                8543 |      1.000 |   53.564 |    72.201 |    13.375 |                     39.503 |  0.392 |  0.392 |
| LIGHTGBM | H2          |               8526 |                8526 |      1.000 |   74.146 |    99.065 |     3.972 |                     55.755 |  0.550 |  0.550 |
| LIGHTGBM | H3          |               8554 |                8554 |      1.000 |   98.781 |   128.585 |     9.813 |                     77.734 |  0.685 |  0.685 |
| LIGHTGBM | H4          |               8538 |                8538 |      1.000 |   59.522 |    77.748 |    10.459 |                     46.465 |  0.400 |  0.400 |
| CHRONOS2 | H1          |               8543 |                8543 |      1.000 |  104.131 |   141.407 |   -54.192 |                     76.414 |  0.761 |  0.761 |
| CHRONOS2 | H2          |               8526 |                8526 |      1.000 |   92.145 |   122.381 |     9.812 |                     70.227 |  0.683 |  0.683 |
| CHRONOS2 | H3          |               8554 |                8554 |      1.000 |  124.688 |   164.339 |    57.324 |                     95.141 |  0.864 |  0.864 |
| CHRONOS2 | H4          |               8538 |                8538 |      1.000 |   93.669 |   127.877 |   -37.748 |                     69.233 |  0.629 |  0.629 |
| TIMESFM3 | H1          |               8543 |                8543 |      1.000 |   95.143 |   128.410 |   -36.243 |                     71.261 |  0.696 |  0.696 |
| TIMESFM3 | H2          |               8526 |                8526 |      1.000 |   89.762 |   122.246 |    23.607 |                     64.641 |  0.665 |  0.665 |
| TIMESFM3 | H3          |               8554 |                8554 |      1.000 |  121.495 |   159.451 |    38.219 |                     94.078 |  0.842 |  0.842 |
| TIMESFM3 | H4          |               8538 |                8538 |      1.000 |   75.268 |    99.579 |   -25.758 |                     56.585 |  0.506 |  0.506 |

## Import/export performance

| model    |   import_mae_kw |   export_mae_kw |   sign_accuracy |   import_recall |   export_recall |   false_import_rate |   false_export_rate |
|:---------|----------------:|----------------:|----------------:|----------------:|----------------:|--------------------:|--------------------:|
| B5       |          69.349 |         121.900 |           0.885 |           0.976 |           0.308 |               0.692 |               0.024 |
| LIGHTGBM |          63.685 |         121.377 |           0.888 |           0.981 |           0.298 |               0.702 |               0.019 |
| CHRONOS2 |          91.412 |         181.771 |           0.798 |           0.906 |           0.115 |               0.885 |               0.094 |
| TIMESFM3 |          82.440 |         178.190 |           0.824 |           0.935 |           0.112 |               0.888 |               0.065 |

## Monthly performance

| model    |   month |   export_fraction |   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |
|:---------|--------:|------------------:|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|
| B5       |       1 |             0.168 |               2975 |                2975 |      1.000 |   60.971 |    73.908 |    17.101 |                     57.062 |  0.743 |  0.743 |
| B5       |       2 |             0.142 |               2688 |                2688 |      1.000 |   68.121 |    80.815 |   -12.559 |                     64.148 |  0.584 |  0.584 |
| B5       |       3 |             0.174 |               2972 |                2972 |      1.000 |   90.180 |   110.637 |    -5.843 |                     81.057 |  0.646 |  0.646 |
| B5       |       4 |             0.244 |               2713 |                2713 |      1.000 |   99.967 |   127.015 |    33.502 |                     88.010 |  0.856 |  0.856 |
| B5       |       5 |             0.243 |               2733 |                2733 |      1.000 |   86.203 |   112.126 |    17.089 |                     68.901 |  0.595 |  0.595 |
| B5       |       6 |             0.182 |               2780 |                2780 |      1.000 |   65.881 |   101.415 |    -7.365 |                     33.587 |  0.367 |  0.367 |
| B5       |       7 |             0.118 |               2974 |                2974 |      1.000 |   75.538 |   108.843 |    -6.964 |                     46.846 |  0.429 |  0.429 |
| B5       |       8 |             0.092 |               2976 |                2976 |      1.000 |   73.255 |   107.278 |    -2.595 |                     39.886 |  0.382 |  0.382 |
| B5       |       9 |             0.105 |               2880 |                2880 |      1.000 |   70.570 |    98.462 |     4.324 |                     47.715 |  0.371 |  0.371 |
| B5       |      10 |             0.080 |               2707 |                2707 |      1.000 |   94.565 |   114.081 |    27.911 |                     87.179 |  0.683 |  0.683 |
| B5       |      11 |             0.052 |               2880 |                2880 |      1.000 |   75.159 |    92.399 |     4.897 |                     66.770 |  0.648 |  0.648 |
| B5       |      12 |             0.041 |               2883 |                2883 |      1.000 |   59.593 |    76.071 |    12.687 |                     47.145 |  0.594 |  0.594 |
| LIGHTGBM |       1 |             0.168 |               2975 |                2975 |      1.000 |   56.625 |    70.548 |     8.367 |                     49.975 |  0.690 |  0.690 |
| LIGHTGBM |       2 |             0.142 |               2688 |                2688 |      1.000 |   50.460 |    64.697 |    -2.874 |                     41.888 |  0.433 |  0.433 |
| LIGHTGBM |       3 |             0.174 |               2972 |                2972 |      1.000 |   76.172 |   102.135 |     8.845 |                     59.613 |  0.546 |  0.546 |
| LIGHTGBM |       4 |             0.244 |               2713 |                2713 |      1.000 |   91.830 |   119.916 |    21.584 |                     74.921 |  0.786 |  0.786 |
| LIGHTGBM |       5 |             0.243 |               2733 |                2733 |      1.000 |   82.670 |   110.165 |    22.713 |                     62.605 |  0.570 |  0.570 |
| LIGHTGBM |       6 |             0.182 |               2780 |                2780 |      1.000 |   68.329 |   101.885 |     9.373 |                     37.172 |  0.381 |  0.381 |
| LIGHTGBM |       7 |             0.118 |               2974 |                2974 |      1.000 |   74.255 |   104.890 |     9.244 |                     47.217 |  0.422 |  0.422 |
| LIGHTGBM |       8 |             0.092 |               2976 |                2976 |      1.000 |   74.521 |   105.767 |    12.621 |                     45.721 |  0.389 |  0.389 |
| LIGHTGBM |       9 |             0.105 |               2880 |                2880 |      1.000 |   70.512 |    96.921 |     5.307 |                     47.194 |  0.371 |  0.371 |
| LIGHTGBM |      10 |             0.080 |               2707 |                2707 |      1.000 |   89.953 |   109.186 |    21.383 |                     81.542 |  0.650 |  0.650 |
| LIGHTGBM |      11 |             0.052 |               2880 |                2880 |      1.000 |   74.698 |    92.582 |     0.473 |                     63.951 |  0.644 |  0.644 |
| LIGHTGBM |      12 |             0.041 |               2883 |                2883 |      1.000 |   49.642 |    67.482 |    -2.898 |                     35.680 |  0.495 |  0.495 |
| CHRONOS2 |       1 |             0.168 |               2975 |                2975 |      1.000 |   61.003 |    78.297 |    -4.423 |                     48.498 |  0.743 |  0.743 |
| CHRONOS2 |       2 |             0.142 |               2688 |                2688 |      1.000 |   56.166 |    75.115 |    -5.681 |                     42.381 |  0.482 |  0.482 |
| CHRONOS2 |       3 |             0.174 |               2972 |                2972 |      1.000 |  101.114 |   131.022 |    -9.121 |                     83.928 |  0.725 |  0.725 |
| CHRONOS2 |       4 |             0.244 |               2713 |                2713 |      1.000 |  122.460 |   160.295 |   -19.998 |                     94.864 |  1.049 |  1.049 |
| CHRONOS2 |       5 |             0.243 |               2733 |                2733 |      1.000 |  157.975 |   194.758 |   -42.643 |                    136.718 |  1.090 |  1.090 |
| CHRONOS2 |       6 |             0.182 |               2780 |                2780 |      1.000 |  173.442 |   218.558 |    -7.492 |                    144.833 |  0.967 |  0.967 |
| CHRONOS2 |       7 |             0.118 |               2974 |                2974 |      1.000 |  121.332 |   156.936 |   -10.959 |                     94.865 |  0.690 |  0.690 |
| CHRONOS2 |       8 |             0.092 |               2976 |                2976 |      1.000 |  107.788 |   141.939 |     5.025 |                     83.385 |  0.563 |  0.563 |
| CHRONOS2 |       9 |             0.105 |               2880 |                2880 |      1.000 |  109.486 |   141.121 |    -3.438 |                     83.049 |  0.575 |  0.575 |
| CHRONOS2 |      10 |             0.080 |               2707 |                2707 |      1.000 |   98.323 |   122.723 |    11.333 |                     83.655 |  0.710 |  0.710 |
| CHRONOS2 |      11 |             0.052 |               2880 |                2880 |      1.000 |   77.785 |   100.039 |     5.881 |                     62.789 |  0.670 |  0.670 |
| CHRONOS2 |      12 |             0.041 |               2883 |                2883 |      1.000 |   60.810 |    79.929 |     5.493 |                     46.861 |  0.607 |  0.607 |
| TIMESFM3 |       1 |             0.168 |               2975 |                2975 |      1.000 |   59.307 |    76.826 |   -15.833 |                     46.111 |  0.723 |  0.723 |
| TIMESFM3 |       2 |             0.142 |               2688 |                2688 |      1.000 |   56.067 |    73.869 |    -6.874 |                     44.442 |  0.481 |  0.481 |
| TIMESFM3 |       3 |             0.174 |               2972 |                2972 |      1.000 |   97.991 |   130.979 |     7.833 |                     72.791 |  0.702 |  0.702 |
| TIMESFM3 |       4 |             0.244 |               2713 |                2713 |      1.000 |  115.888 |   152.389 |    -1.968 |                     88.477 |  0.992 |  0.992 |
| TIMESFM3 |       5 |             0.243 |               2733 |                2733 |      1.000 |  135.462 |   170.488 |    11.423 |                    112.313 |  0.935 |  0.935 |
| TIMESFM3 |       6 |             0.182 |               2780 |                2780 |      1.000 |  165.399 |   206.632 |     8.586 |                    136.221 |  0.923 |  0.923 |
| TIMESFM3 |       7 |             0.118 |               2974 |                2974 |      1.000 |   90.983 |   121.738 |     5.023 |                     63.677 |  0.517 |  0.517 |
| TIMESFM3 |       8 |             0.092 |               2976 |                2976 |      1.000 |   88.735 |   117.741 |    -0.639 |                     65.721 |  0.463 |  0.463 |
| TIMESFM3 |       9 |             0.105 |               2880 |                2880 |      1.000 |   99.017 |   129.162 |     4.529 |                     75.408 |  0.520 |  0.520 |
| TIMESFM3 |      10 |             0.080 |               2707 |                2707 |      1.000 |  100.318 |   124.847 |    10.803 |                     84.851 |  0.725 |  0.725 |
| TIMESFM3 |      11 |             0.052 |               2880 |                2880 |      1.000 |   77.627 |   102.103 |   -17.407 |                     59.579 |  0.669 |  0.669 |
| TIMESFM3 |      12 |             0.041 |               2883 |                2883 |      1.000 |   63.219 |    84.283 |    -4.855 |                     47.354 |  0.631 |  0.631 |

## Calendar contexts and structural diagnostic

| model    | calendar_context           |   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |
|:---------|:---------------------------|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|
| B5       | NORMAL_WEEKDAY             |              22626 |               22626 |      1.000 |   79.414 |   105.815 |    -2.960 |                     62.098 |  0.510 |  0.510 |
| B5       | WEEKEND                    |               9711 |                9711 |      1.000 |   68.606 |    88.691 |    19.870 |                     56.189 |  0.602 |  0.602 |
| B5       | PUBLIC_HOLIDAY             |               2304 |                2304 |      1.000 |   80.044 |   104.446 |    53.677 |                     62.198 |  0.775 |  0.775 |
| B5       | DAY_BEFORE_PUBLIC_HOLIDAY  |                768 |                 768 |      1.000 |   79.731 |    99.123 |    52.606 |                     65.851 |  0.828 |  0.828 |
| B5       | DAY_AFTER_PUBLIC_HOLIDAY   |                864 |                 864 |      1.000 |   78.186 |   100.394 |    49.838 |                     60.530 |  0.788 |  0.788 |
| B5       | BRIDGE_DAY_DIAGNOSTIC_ONLY |                288 |                 288 |      1.000 |   82.696 |   110.153 |    50.314 |                     66.192 |  0.600 |  0.600 |
| LIGHTGBM | NORMAL_WEEKDAY             |              22626 |               22626 |      1.000 |   74.717 |   101.168 |     2.702 |                     55.172 |  0.480 |  0.480 |
| LIGHTGBM | WEEKEND                    |               9711 |                9711 |      1.000 |   63.216 |    84.907 |    17.196 |                     47.205 |  0.555 |  0.555 |
| LIGHTGBM | PUBLIC_HOLIDAY             |               2304 |                2304 |      1.000 |   73.438 |    99.850 |    46.839 |                     54.009 |  0.711 |  0.711 |
| LIGHTGBM | DAY_BEFORE_PUBLIC_HOLIDAY  |                768 |                 768 |      1.000 |   69.185 |    89.795 |    43.622 |                     53.397 |  0.719 |  0.719 |
| LIGHTGBM | DAY_AFTER_PUBLIC_HOLIDAY   |                864 |                 864 |      1.000 |   71.983 |    94.827 |    38.877 |                     57.251 |  0.726 |  0.726 |
| LIGHTGBM | BRIDGE_DAY_DIAGNOSTIC_ONLY |                288 |                 288 |      1.000 |   78.869 |   106.977 |    54.952 |                     54.373 |  0.572 |  0.572 |
| CHRONOS2 | NORMAL_WEEKDAY             |              22626 |               22626 |      1.000 |   95.517 |   126.716 |   -15.510 |                     73.077 |  0.614 |  0.614 |
| CHRONOS2 | WEEKEND                    |               9711 |                9711 |      1.000 |  118.504 |   161.082 |    16.857 |                     85.384 |  1.040 |  1.040 |
| CHRONOS2 | PUBLIC_HOLIDAY             |               2304 |                2304 |      1.000 |  125.571 |   173.238 |   -11.714 |                     85.914 |  1.216 |  1.216 |
| CHRONOS2 | DAY_BEFORE_PUBLIC_HOLIDAY  |                768 |                 768 |      1.000 |  124.255 |   172.942 |    -8.525 |                     80.966 |  1.291 |  1.291 |
| CHRONOS2 | DAY_AFTER_PUBLIC_HOLIDAY   |                864 |                 864 |      1.000 |  125.060 |   174.551 |   -21.563 |                     85.914 |  1.261 |  1.261 |
| CHRONOS2 | BRIDGE_DAY_DIAGNOSTIC_ONLY |                288 |                 288 |      1.000 |  155.343 |   186.563 |     3.823 |                    136.519 |  1.127 |  1.127 |
| TIMESFM3 | NORMAL_WEEKDAY             |              22626 |               22626 |      1.000 |   90.036 |   119.949 |    -9.955 |                     67.803 |  0.579 |  0.579 |
| TIMESFM3 | WEEKEND                    |               9711 |                9711 |      1.000 |  104.375 |   144.292 |    20.767 |                     73.658 |  0.916 |  0.916 |
| TIMESFM3 | PUBLIC_HOLIDAY             |               2304 |                2304 |      1.000 |  109.933 |   147.710 |     9.543 |                     84.200 |  1.065 |  1.065 |
| TIMESFM3 | DAY_BEFORE_PUBLIC_HOLIDAY  |                768 |                 768 |      1.000 |  102.856 |   136.533 |    11.911 |                     75.679 |  1.069 |  1.069 |
| TIMESFM3 | DAY_AFTER_PUBLIC_HOLIDAY   |                864 |                 864 |      1.000 |  108.624 |   152.300 |     7.688 |                     77.677 |  1.095 |  1.095 |
| TIMESFM3 | BRIDGE_DAY_DIAGNOSTIC_ONLY |                288 |                 288 |      1.000 |  159.985 |   197.007 |    40.216 |                    140.053 |  1.160 |  1.160 |

| model    | stratum            |   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |
|:---------|:-------------------|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|
| B5       | pre_June_2023      |              14081 |               14081 |      1.000 |   80.912 |   102.630 |     9.754 |                     69.985 |  0.676 |  0.676 |
| B5       | June_December_2023 |              20080 |               20080 |      1.000 |   73.372 |   100.422 |     4.471 |                     52.320 |  0.470 |  0.470 |
| LIGHTGBM | pre_June_2023      |              14081 |               14081 |      1.000 |   71.412 |    95.804 |    11.653 |                     55.410 |  0.597 |  0.597 |
| LIGHTGBM | June_December_2023 |              20080 |               20080 |      1.000 |   71.583 |    97.786 |     7.833 |                     50.062 |  0.458 |  0.458 |
| CHRONOS2 | pre_June_2023      |              14081 |               14081 |      1.000 |   99.208 |   135.308 |   -16.074 |                     71.857 |  0.829 |  0.829 |
| CHRONOS2 | June_December_2023 |              20080 |               20080 |      1.000 |  106.803 |   143.130 |     0.751 |                     79.843 |  0.683 |  0.683 |
| TIMESFM3 | pre_June_2023      |              14081 |               14081 |      1.000 |   92.536 |   126.591 |    -1.166 |                     66.921 |  0.773 |  0.773 |
| TIMESFM3 | June_December_2023 |              20080 |               20080 |      1.000 |   97.461 |   131.034 |     0.750 |                     72.320 |  0.624 |  0.624 |

## Extreme regimes and daily-profile performance

| model    | actual_bin   |   lower_bound_kw |   upper_bound_kw |   eligible_targets |   predicted_targets |   coverage |   mae_kw |   rmse_kw |   bias_kw |   median_absolute_error_kw |   nmae |   wape |
|:---------|:-------------|-----------------:|-----------------:|-------------------:|--------------------:|-----------:|---------:|----------:|----------:|---------------------------:|-------:|-------:|
| B5       | Q1           |         -inf     |           83.266 |              13837 |               13837 |      1.000 |   90.829 |   110.946 |    83.839 |                     79.921 |  1.635 |  1.635 |
| B5       | Q2           |           83.266 |          169.776 |               6603 |                6603 |      1.000 |   50.952 |    66.236 |    -2.935 |                     40.387 |  0.401 |  0.401 |
| B5       | Q3           |          169.776 |          230.085 |               8459 |                8459 |      1.000 |   48.987 |    65.867 |   -37.558 |                     36.338 |  0.245 |  0.245 |
| B5       | Q4           |          230.085 |          inf     |               5262 |                5262 |      1.000 |  114.974 |   147.719 |  -113.239 |                     95.300 |  0.397 |  0.397 |
| LIGHTGBM | Q1           |         -inf     |           83.266 |              13837 |               13837 |      1.000 |   86.094 |   109.226 |    80.011 |                     70.740 |  1.550 |  1.550 |
| LIGHTGBM | Q2           |           83.266 |          169.776 |               6603 |                6603 |      1.000 |   46.328 |    60.214 |    -1.011 |                     36.560 |  0.365 |  0.365 |
| LIGHTGBM | Q3           |          169.776 |          230.085 |               8459 |                8459 |      1.000 |   45.273 |    60.126 |   -27.327 |                     35.049 |  0.226 |  0.226 |
| LIGHTGBM | Q4           |          230.085 |          inf     |               5262 |                5262 |      1.000 |  106.955 |   138.989 |  -104.125 |                     90.664 |  0.370 |  0.370 |
| CHRONOS2 | Q1           |         -inf     |           83.266 |              13837 |               13837 |      1.000 |  108.903 |   147.072 |    90.263 |                     81.411 |  1.960 |  1.960 |
| CHRONOS2 | Q2           |           83.266 |          169.776 |               6603 |                6603 |      1.000 |   63.925 |    84.862 |    -4.341 |                     51.020 |  0.503 |  0.503 |
| CHRONOS2 | Q3           |          169.776 |          230.085 |               8459 |                8459 |      1.000 |  109.999 |   149.235 |   -93.125 |                     79.253 |  0.550 |  0.550 |
| CHRONOS2 | Q4           |          230.085 |          inf     |               5262 |                5262 |      1.000 |  129.624 |   159.531 |  -122.351 |                    109.402 |  0.448 |  0.448 |
| TIMESFM3 | Q1           |         -inf     |           83.266 |              13837 |               13837 |      1.000 |  107.465 |   145.153 |    91.414 |                     79.353 |  1.934 |  1.934 |
| TIMESFM3 | Q2           |           83.266 |          169.776 |               6603 |                6603 |      1.000 |   59.794 |    75.355 |    -6.411 |                     52.312 |  0.471 |  0.471 |
| TIMESFM3 | Q3           |          169.776 |          230.085 |               8459 |                8459 |      1.000 |   88.952 |   122.652 |   -73.341 |                     63.454 |  0.445 |  0.445 |
| TIMESFM3 | Q4           |          230.085 |          inf     |               5262 |                5262 |      1.000 |  118.919 |   147.281 |  -114.692 |                    101.624 |  0.411 |  0.411 |

| model    |   complete_days |   daily_mae_mean |   daily_mae_median |   daily_mae_p90 |   daily_mae_max |   daily_rmse_mean |   absolute_daily_energy_error_mean |   signed_daily_energy_error_mean |
|:---------|----------------:|-----------------:|-------------------:|----------------:|----------------:|------------------:|-----------------------------------:|---------------------------------:|
| B5       |             347 |           76.424 |             74.016 |         108.660 |         212.406 |            95.938 |                           1088.430 |                          155.144 |
| CHRONOS2 |             347 |          102.882 |             90.932 |         174.996 |         326.381 |           124.623 |                           1270.517 |                         -178.649 |
| LIGHTGBM |             347 |           71.302 |             68.437 |         104.780 |         211.768 |            90.989 |                            968.905 |                          215.656 |
| TIMESFM3 |             347 |           94.271 |             82.898 |         153.522 |         280.357 |           116.625 |                           1024.445 |                          -18.193 |

## Bootstrap comparisons versus B5 and LightGBM

| foundation_model   | comparator   |   common_complete_days |   mean_daily_mae_difference_kw |   ci95_lower_kw |   ci95_upper_kw |   foundation_daily_win_rate |   resamples |     seed |
|:-------------------|:-------------|-----------------------:|-------------------------------:|----------------:|----------------:|----------------------------:|------------:|---------:|
| CHRONOS2           | B5           |                    347 |                         26.458 |          20.592 |          32.554 |                       0.392 |        2000 | 20261029 |
| TIMESFM3           | B5           |                    347 |                         17.847 |          13.277 |          22.854 |                       0.372 |        2000 | 20261029 |

| foundation_model   | comparator   |   common_complete_days |   mean_daily_mae_difference_kw |   ci95_lower_kw |   ci95_upper_kw |   foundation_daily_win_rate |   resamples |     seed |
|:-------------------|:-------------|-----------------------:|-------------------------------:|----------------:|----------------:|----------------------------:|------------:|---------:|
| CHRONOS2           | LIGHTGBM     |                    347 |                         31.580 |          26.291 |          37.161 |                       0.268 |        2000 | 20261029 |
| TIMESFM3           | LIGHTGBM     |                    347 |                         22.969 |          19.023 |          27.464 |                       0.251 |        2000 | 20261029 |

## Runtime and compute

| checkpoint_sha256                                                |   context_days | device   | dtype   |   forecast_origins | gradient_enabled   |   inference_seconds |   load_seconds | mode   | model    | model_id                   | native_missing_policy                                        | package             | package_version   |   peak_gpu_memory_bytes |   peak_rss_bytes | phase       | point_forecast                    |   prediction_rows | python   | revision                                 |   seconds_per_origin | torch_version   |
|:-----------------------------------------------------------------|---------------:|:---------|:--------|-------------------:|:-------------------|--------------------:|---------------:|:-------|:---------|:---------------------------|:-------------------------------------------------------------|:--------------------|:------------------|------------------------:|-----------------:|:------------|:----------------------------------|------------------:|:---------|:-----------------------------------------|---------------------:|:----------------|
| ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42 |             14 | cuda     | float16 |                730 | False              |             470.421 |          4.587 | C1     | CHRONOS2 | amazon/chronos-2           | Chronos-2 NaN context mask                                   | chronos-forecasting | 2.3.2             |           272898048.000 |       1084567552 | development | official quantile 0.5 (median)    |             70080 | 3.12.3   | 29ec3766d36d6f73f0696f85560a422f50e8498c |                0.644 | 2.7.1+cu118     |
| ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42 |             14 | cpu      | float32 |                730 | False              |             451.767 |          3.630 | U      | CHRONOS2 | amazon/chronos-2           | Chronos-2 NaN context mask                                   | chronos-forecasting | 2.3.2             |                 nan     |       1128161280 | development | official quantile 0.5 (median)    |             70080 | 3.12.3   | 29ec3766d36d6f73f0696f85560a422f50e8498c |                0.619 | 2.7.1+cpu       |
| ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42 |             28 | cpu      | float32 |                730 | False              |             497.328 |          1.478 | U      | CHRONOS2 | amazon/chronos-2           | Chronos-2 NaN context mask                                   | chronos-forecasting | 2.3.2             |                 nan     |       1346404352 | development | official quantile 0.5 (median)    |             70080 | 3.12.3   | 29ec3766d36d6f73f0696f85560a422f50e8498c |                0.681 | 2.7.1+cpu       |
| ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42 |              7 | cpu      | float32 |                730 | False              |             235.939 |          3.060 | U      | CHRONOS2 | amazon/chronos-2           | Chronos-2 NaN context mask                                   | chronos-forecasting | 2.3.2             |                 nan     |       1038159872 | development | official quantile 0.5 (median)    |             70080 | 3.12.3   | 29ec3766d36d6f73f0696f85560a422f50e8498c |                0.323 | 2.7.1+cpu       |
| ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42 |             14 | cuda     | float16 |                730 | False              |              50.254 |          1.654 | U      | CHRONOS2 | amazon/chronos-2           | Chronos-2 NaN context mask                                   | chronos-forecasting | 2.3.2             |           346904064.000 |       1095786496 | development | official quantile 0.5 (median)    |             70080 | 3.12.3   | 29ec3766d36d6f73f0696f85560a422f50e8498c |                0.069 | 2.7.1+cu118     |
| a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da |             28 | cuda     | float32 |                730 | False              |             405.455 |          5.933 | C1     | TIMESFM3 | google/timesfm-3.0-pytorch | official historical-only linear interpolation within context | timesfm             | 3.0.2             |          1514294784.000 |       2347347968 | development | official forecast output (median) |             70080 | 3.12.3   | 43046b85ec22d584a13f8098c2ed39c889e129c2 |                0.555 | 2.7.1+cu118     |
| a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da |             28 | cuda     | float32 |                730 | False              |             115.229 |          6.346 | U      | TIMESFM3 | google/timesfm-3.0-pytorch | official historical-only linear interpolation within context | timesfm             | 3.0.2             |          1373481472.000 |       2273239040 | development | official forecast output (median) |             70080 | 3.12.3   | 43046b85ec22d584a13f8098c2ed39c889e129c2 |                0.158 | 2.7.1+cu118     |
| ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42 |             14 | cuda     | float16 |                365 | False              |             221.354 |          1.355 | C1     | CHRONOS2 | amazon/chronos-2           | Chronos-2 NaN context mask                                   | chronos-forecasting | 2.3.2             |           272898048.000 |       1141727232 | test        | official quantile 0.5 (median)    |             35040 | 3.12.3   | 29ec3766d36d6f73f0696f85560a422f50e8498c |                0.606 | 2.7.1+cu118     |
| a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da |             28 | cuda     | float32 |                365 | False              |             206.129 |          5.958 | C1     | TIMESFM3 | google/timesfm-3.0-pytorch | official historical-only linear interpolation within context | timesfm             | 3.0.2             |          1515335168.000 |       2297057280 | test        | official forecast output (median) |             35040 | 3.12.3   | 43046b85ec22d584a13f8098c2ed39c889e129c2 |                0.565 | 2.7.1+cu118     |
| a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da |             14 | cpu      | float32 |                730 | False              |             458.404 |         11.257 | U      | TIMESFM3 | google/timesfm-3.0-pytorch | official historical-only linear interpolation within context | timesfm             | 3.0.2             |                 nan     |       1851916288 | development | official forecast output (median) |             70080 | 3.12.3   | 43046b85ec22d584a13f8098c2ed39c889e129c2 |                0.628 | 2.7.1+cpu       |
| a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da |             28 | cpu      | float32 |                730 | False              |             815.484 |          5.348 | U      | TIMESFM3 | google/timesfm-3.0-pytorch | official historical-only linear interpolation within context | timesfm             | 3.0.2             |                 nan     |       1879760896 | development | official forecast output (median) |             70080 | 3.12.3   | 43046b85ec22d584a13f8098c2ed39c889e129c2 |                1.117 | 2.7.1+cpu       |
| a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da |              7 | cpu      | float32 |                730 | False              |             237.821 |          6.534 | U      | TIMESFM3 | google/timesfm-3.0-pytorch | official historical-only linear interpolation within context | timesfm             | 3.0.2             |                 nan     |       1805672448 | development | official forecast output (median) |             70080 | 3.12.3   | 43046b85ec22d584a13f8098c2ed39c889e129c2 |                0.326 | 2.7.1+cpu       |

Both models ran locally. The frozen 2023 runs took about 221 seconds for Chronos-2 and 206 seconds for TimesFM, excluding load time. Their checkpoints are approximately 478 MB and 1.32 GB, respectively; the frozen LightGBM remains far smaller and requires site-specific training, whereas the foundation models required no Honda weight updates.

## Where foundation models help, where they fail, and comparison with LightGBM

Neither foundation model improved aggregate, lead-band, calendar, or export-state performance over the frozen references. TimesFM was the stronger zero-shot model, but its 95.431 kW MAE remained 23.919 kW above LightGBM. Both models performed relatively well in January/February and poorly in late spring/early summer, especially June. Their largest scientific failure was export recognition: export recall fell to roughly 11%, below both B5 and LightGBM, while export MAE increased to roughly 178–182 kW. The H3 band also worsened rather than improving.

This result supports the value of Honda-specific historical feature engineering and chronological adaptation. It does not invalidate pretrained models: the experiment demonstrates full-coverage transfer without site training, but that transfer is not competitive under this operational protocol.

## Implications for probabilistic forecasting and limitations

H6 evaluates only deterministic medians. It does not assess whether native samples or quantiles are calibrated or temporally coherent. Covariate work was deliberately limited to the H5-motivated C1 information set; no future weather, PV, or CHP realization was supplied. Results apply to the exact checkpoints, revisions, inference settings, hardware, and 2023 Honda test period recorded here.

## Interpretation, limitations, and recommended H7

- Chronos-2: **Category C — useful zero-shot coverage but behind the frozen references**.
- TimesFM 3.0: **Category C — useful zero-shot coverage but behind the frozen references**.

Foundation performance is interpreted jointly with coverage and compute. Predictive differences do not establish causal effects of CHP, calendar, modernization, or PV. H7 should evaluate probabilistic forecasting and temporally coherent daily scenarios using frozen deterministic references and any scientifically usable foundation-model uncertainty outputs. No H7 work was performed here.
