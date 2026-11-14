# Honda NVIDIA Kumo Forecast and DARR benchmark

## Executive answer

**Does domain-aware retrieval provide incremental value beyond pretrained foundation-model forecasting and strong site-specific baselines? Not yet established.**

The integration is implemented, causality-tested, DST-tested, pinned, and exercised through real Kumo inference. A single 2021-01-01 smoke origin produced mixed results: the external calendar-family wrapper improved substantially over direct Kumo, native DARR worsened, and neither retrieval result beat frozen LightGBM on that day. One origin cannot answer the research question, support statistical inference, select a configuration, or justify a 2023 test run. Full development experiments therefore remain `NOT RUN`, no final Kumo configuration is frozen, and the retrospective 2023 test remains blocked.

## Official capability audit

The study audited NVIDIA Kumo-TS 1.0.1 at commit `ad459475d1302fc8365e7117e39837a8ecb6a7e3` and the Hugging Face Kumo Forecast revision `402caa11b7ff585141e912c6743026141f648892`. The checkpoint and code are Apache-2.0. The backbone is MOMENT-1-large at revision `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc`. NVIDIA describes a roughly 340-million-parameter encoder, 72-step native forecasting, autoregressive horizon extension, and optional DARR retrieval. The public API exposes `alpha`, `k`, `temperature`, and `context_stride`, but no external regime-label filter and no returned neighbor metadata. [NVIDIA Kumo documentation](https://docs.nvidia.com/sdgm/structured-data-models/models/kumo-forecast), [official Kumo-TS source](https://github.com/NVIDIA/Kumo-TS), [official model card](https://huggingface.co/nvidia/Kumo-Forecast).

The official standard path fills numeric missing values with zero. The separate missingness-aware path cannot be combined with DARR in the audited release. The Honda adapter therefore handles direct-model context gaps before invocation, records every imputed point, and prohibits imputation in retrieval continuations.

## Honda protocol preservation

The signed target, negative exports, 13:00 D-1 issue time, `OBSERVED_ORIGINAL` target, and exact physical UTC chronology are unchanged. The implementation forecasts the gap between issue and midnight instead of observing it. Ordinary, spring-DST, and autumn-DST requests contain 140, 136, and 144 steps respectively; delivery-day extraction begins at position 44 and yields 96, 92, or 100 values. Negative predictions are never clipped.

Previously reported B5, LightGBM, TimesFM, and Chronos files were not edited or regenerated. Their historical 2023 MAEs remain:

| Frozen historical reference | MAE (kW) |
|---|---:|
| LightGBM | 71.513 |
| Historical B5 | 76.480 |
| TimesFM 3.0 | 95.431 |
| Chronos-2 | 103.672 |

These numbers are context only. Kumo has not yet been evaluated on the same complete 2023 target set, so no ranking against them is claimed.

## Implemented retrieval safeguards

External retrieval candidates contain only finite `OBSERVED_ORIGINAL` inputs and continuations. A continuation must be completely available before the new query issue and cannot overlap the unknown query period. Candidate source timestamps, availability timestamps, calendar context, pre-issuance regime, distance, rank, and weight are retained.

Native DARR has a stricter integration compromise. Because its public `context_df` interface supplies no candidate mask, the adapter uses a single continuous fully observed historical run. This avoids allowing Kumo's zero-fill behavior to fabricate retrieval continuations, but it narrows the archive and may reduce diversity.

Regime conditioning is an external Honda wrapper, not an NVIDIA capability. It filters by future-known calendar context and by load level/volatility calculated only from the pre-issuance history. Structural-era labels derived post hoc from targets are not predictive inputs.

## Empirical work completed

The local machine has 16.87 GB RAM and a driver-visible GTX 1650 with 4 GB VRAM, but the available PyTorch build is CPU-only. NVIDIA reports model testing on 74–80 GB A100/H100 systems; CPU fallback is supported, but GTX 1650 fit is not documented. The pinned artifacts occupy about 2.79 GB: 1.404 GB for the Kumo forecasting head and 1.386 GB for MOMENT-1-large.

Real smoke inference completed for the first 2021 development origin. These values are diagnostic only:

| Scope: 2021-01-01 only | MAE (kW) | RMSE (kW) | Bias (kW) | Runtime |
|---|---:|---:|---:|---:|
| Frozen B5 | 56.153 | 65.652 | 25.404 | historical artifact |
| Frozen LightGBM | 52.859 | 60.024 | 25.531 | historical artifact |
| Kumo direct | 84.540 | 99.487 | 78.409 | 40.2 s cold start |
| Native DARR, bounded archive | 99.937 | 115.404 | 31.086 | 104.5 s cold start |
| External calendar-family retrieval | 65.381 | 78.921 | 0.423 | 40.6 s cold start |

Direct Kumo generated the full 140-step path and all 96 delivery predictions. Native DARR built 12 archive batches from a bounded 1,024-row complete run; archive construction took roughly 63 seconds. The external calendar-family wrapper selected eight non-workday candidates known before issuance. It reduced the direct model's large positive bias, but remained worse than LightGBM and B5 on this one day.

The diagnostic figure at `results/kumo/diagnostics/retrieval_2021-01-01.png` displays the query, retrieved inputs, their observed continuations, both forecasts, and actual values. Nearest-profile distance is descriptive and is not proof of shared physical operating conditions.

## What is not run and why

- The 2021–2022 development ablation is not run. At measured CPU rates, direct inference alone is material; native DARR is substantially slower and its cost grows with archive size and stride density.
- No development selection artifact exists, so no final experiment configuration has been created.
- The exposed 2023 retrospective test is blocked by code until a full-coverage development configuration is frozen.
- Statistical paired-day intervals, regime conclusions, missing-history robustness, and structural-change conclusions require the complete development experiment.
- Downstream flexibility schedules are not run. A tested adapter can pass a future selected, complete Kumo point forecast to the existing Honda scenario contract without inventing uncertainty, but comparing decisions now would multiply an unselected point forecast into simulated economics. Honda tariffs and physical flexibility parameters also remain unverified, so no monetary or industrial-value claim is supportable.

## Threats to validity

First, Kumo is new and the audited API/version may change. Second, its training corpus includes general electricity datasets, but Honda signed grid exchange, exports, CHP/PV interaction, shutdowns, and the 13:00 bridge horizon are specialized. Third, DARR's continuous archive interface does not expose neighbor identities; native interpretability is therefore weaker than the external wrapper. Fourth, interpolating a pre-issuance input context is explicit and causal at issue time but can still alter model behavior. Fifth, the 2023 period is no longer an untouched holdout. Sixth, the one-day smoke result was chosen by chronological order for integration testing, not representativeness; it must not be generalized.

## Decision and next execution step

The integration passes the implementation gate but not the empirical promotion gate. The next valid step is to run the predefined 2021–2022 development grid on a CUDA environment with sufficient measured memory, beginning with direct Kumo and then a small native/external retrieval ablation. Every configuration must cover identical eligible targets, and selection must use concatenated development MAE with fold, regime, runtime, and whole-day uncertainty reporting. Only then may `honda-kumo freeze` create the locked config and permit the retrospective 2023 evaluation.

The present conclusion is deliberately negative/undetermined: retrieval is technically feasible and auditable, but incremental scientific value beyond Kumo direct and Honda LightGBM has not been demonstrated.
