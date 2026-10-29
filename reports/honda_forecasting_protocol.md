# Frozen Honda signed grid-exchange forecasting protocol

Protocol version: `honda-h3-v1.0.0`
Status: **FROZEN**

## Executive summary

All future Honda forecasting methods must solve the same task: at 13:00 Europe/Berlin on D-1, forecast every physical 15-minute signed net-grid-exchange interval of local day D. Positive values are grid imports and negative values are exports. The canonical evaluation target contains only `OBSERVED_ORIGINAL` intervals; known corrected observations are sensitivity-only and author-filled values are excluded.

P1 is the primary chronological protocol: 2020→2021, 2020–2021→2022, then a single refit on 2020–2022 and one untouched 2023 TEST evaluation. P2 is a separate post-2020 structural sensitivity. No H3B model was trained, prediction created, score calculated, or optimization run.

## Provenance foundation

H2 recovered meter-level issue intervals and created immutable 15-minute provenance flags. H3A then established target availability, physical-day construction, context completeness, holiday location, and temporal distribution differences. H3B preserves H2 commit `2d29de9f45dcd414baf53dd3b3dabb4627798fe3` and H3A commit `99bb2033e30c06aeb71486f65a347887735f5c95`.

Primary targets require `OBSERVED_ORIGINAL`. Sensitivity targets allow `OBSERVED_ORIGINAL` or `CORRECTED_NONIMPUTED`. `IMPUTED_OR_FILLED`, `UNKNOWN_PROVENANCE`, and `MISSING_UNRESOLVED` are never evaluation targets.

## Primary target

The frozen target is `grid_exchange_kw_signed`, in kW, at 15-minute resolution. UTC is the unique physical timestamp; Europe/Berlin defines operational calendar days. One-hour forecasting may later be reported as a secondary benchmark but cannot replace the 15-minute task.

## Signed-grid-exchange interpretation

Positive means import from the public grid; negative means export. Predictions remain real-valued and signed. Clipping at zero, converting exports to imports, or applying a non-reversible sign-unsafe transform is forbidden.

This is net grid exchange, not site load or gross industrial demand. On-site PV and CHP directly affect it. Reconstructed Honda gross load remains unapproved as a target.

## Operational issuance

The frozen issuance is 13:00 Europe/Berlin on D-1. It is 12:00 UTC in winter and 11:00 UTC in summer and is not ambiguous during DST transitions. A later price-aware decision may occur after verified price publication without changing the forecast issuance.

## Measurement availability

Historical measurements satisfy:

`interval_end_utc <= forecast_issue_time_utc`

A 12:45 label covers `[12:45, 13:00)` and is available at 13:00. A 13:00 label covers `[13:00, 13:15)` and is unavailable. Every baseline and future model must enforce this boundary.

## DST and physical target days

Target grids contain 96 intervals ordinarily, 92 on the spring transition, and 100 on the autumn transition. UTC prevents duplicate physical keys. The repeated autumn hour is retained with separate offsets/folds; the nonexistent spring hour is not fabricated.

Interval-start lead ranges are 11.00–34.75 h ordinarily, 11.00–33.75 h in spring DST, and 11.00–35.75 h in autumn DST.

## Primary and sensitivity target policies

Primary eligibility is exactly `provenance_class == OBSERVED_ORIGINAL`. Sensitivity eligibility is exactly membership in `{OBSERVED_ORIGINAL, CORRECTED_NONIMPUTED}`. Sample-size considerations cannot weaken these definitions.

Primary and sensitivity results must remain separate. Corrected observations never enter primary evaluation merely because they increase coverage.

## Historical predictor policy

Observed-original history is allowed. Corrected-nonimputed history may be used only as a separately flagged sensitivity-safe value. Imputed, unknown, and unresolved history is represented as unavailable. Protocol generation cannot silently fill it or reintroduce Honda author-filled values.

## Missing-history policy

A valid target is not removed solely because a lag or exogenous historical value is unavailable. The canonical representation is NaN plus an availability indicator where appropriate. A model-specific adapter may be introduced later if it is explicit, leakage-safe, audited, and leaves the canonical target set unchanged.

There is no global 95% or other context-completeness threshold. Foundation models that require dense context use a separately reported `COMMON_CONTEXT_SUBSET` for strict pairwise comparison; they do not redefine the benchmark for all models.

## Chronological design

### P1 Fold A

- TRAIN: eligible 2020 targets — 24,145 intervals, 187 complete days.
- VALIDATION: eligible 2021 targets — 34,281 intervals, 342 complete days.

### P1 Fold B

- TRAIN: eligible 2020–2021 targets — 58,426 intervals, 529 complete days.
- VALIDATION: eligible 2022 targets — 34,624 intervals, 356 complete days.

### Final evaluation

- Development/refit: eligible 2020–2022 targets — 93,050 intervals, 885 complete days.
- TEST: eligible 2023 targets — 34,161 intervals, 347 complete days.

TEST labels cannot be used for feature, baseline, model, context, calibration, threshold, scenario, bias, ensemble, or early-stopping choices. Reopening requires a new protocol version and scientific justification.

## Why P1 is primary

2020 contributes substantial volume, COVID/reduced-operation diversity, the PV-capacity transition, and a hard temporal-generalization stress test. It also has lower continuity, 105 zero-primary days, clock-correction history, selective coverage, and unusual conditions. Retaining it is deliberate; sparse intervals must not be treated as continuous history.

Model/configuration selection concatenates all eligible P1 validation errors from 2021 and 2022 and computes a single MAE, giving equal weight to each eligible target interval. Fold-specific results remain mandatory to expose instability. Selection solely on the more recent fold is forbidden.

## P2 structural sensitivity

`POST_2020_STRUCTURAL_SENSITIVITY` uses 2021 TRAIN, 2022 VALIDATION, 2021–2022 final refit, and the same 2023 TEST. It asks how conclusions depend on excluding 2020. P2 cannot replace P1 because it later scores better on TEST.

## Final 2023 TEST

After P1 development selection, refit once on eligible 2020–2022 data and evaluate 2023 once. P2 may separately refit on 2021–2022. Features and hyperparameters must be frozen before either TEST evaluation.

## Complete vs partial day evaluation

Headline interval metrics use every eligible primary TEST interval, including eligible intervals on partial days. Complete-day profile metrics use only days where every physical target interval is primary eligible. Sensitivity complete days require every interval to be observed-original or corrected-nonimputed and are reported separately.

Default downstream optimization uses complete primary days unless a separately versioned operational protocol justifies partial-day decisions.

## Lead-time bands

- H1: `11 <= lead < 17` hours.
- H2: `17 <= lead < 23` hours.
- H3: `23 <= lead < 29` hours.
- H4: `29 <= lead <= 36` hours.

Every indexed physical interval belongs to exactly one band, including DST days. Boundaries cannot be changed because of later performance.

## Calendar contexts

Future-known contexts are normal weekday, weekend, public holiday, day before public holiday, and day after public holiday, using Hessen holidays.

A bridge day is a non-holiday Monday–Friday with one adjacent local date a public holiday and the other a weekend day. Only nine such days occur during 2020–2023, so bridge day is frozen as descriptive diagnostic only, not a primary feature or required performance stratum.

## Structural-era diagnostics

COVID/reduced-operation, PV-capacity, post-COVID, and documented future-safe engineering dates may be evaluation strata. A date may be a predictor only when genuinely known at issuance and not derived from target behavior.

For 2023, `PRE_MODERNIZATION_2023` means before June 1 and `POST_MODERNIZATION_2023` means June–December. These are evaluation-only, month-precision labels. They do not establish causality and cannot tune the TEST model.

Official sources give February 13 and February 19, 2019 for the CHP control update. The interval is marked `CHP_CONTROL_BOUNDARY_UNCERTAIN`; neither date is invented as an exact predictive boundary until resolved.

## Feature availability contract

Classifications are `FUTURE_KNOWN`, `HISTORICAL_ONLY`, `FORECAST_REQUIRED`, `FORBIDDEN_REALIZED_FUTURE`, and `DIAGNOSTIC_ONLY`. The machine-readable contract covers grid exchange, calendar, holiday, Ta, Igm, PV, CHP, heating, cooling, structural eras, future prices, and realized grid mode.

## Weather/PV/CHP rules

Historical completed Ta, Igm, PV, and CHP may enter M1 when provenance-safe. Their future realized measurements are forbidden. Operational weather/PV/CHP forecasts require separate leakage-safe provenance and are not part of the primary benchmark.

Historical completed heating/cooling may enter M1 when provenance-safe. Future realized thermal values are forbidden; no future thermal schedule is presumed known.

## Core information sets

M0 uses historical signed grid exchange, future-known calendar variables, and structural dates classified future-safe.

M1 adds only pre-issuance provenance-safe historical PV, CHP, weather, and thermal measurements. It contains no realized future exogenous data. Development folds may later determine whether M1 improves on M0.

## Baseline specifications

- **B0 issuance persistence:** hold the most recent completed safe grid value constant across the target day.
- **B1 recent safe daily analogue:** most recent preceding local day with a safe matching quarter-hour; match fold/offset for repeated hours; otherwise missing.
- **B2 previous-week analogue:** exactly seven local calendar days earlier; no arbitrary alternate-week search; otherwise missing.
- **B3 recent-week robust analogue:** median of matching quarter-hours from prior N eligible weeks; choose N from 2, 4, or 8 using development folds.
- **B4 calendar climatology:** training-only robust calendar statistic; choose grouping/backoff on development folds without validation/TEST target encoding.
- **B5 simple baseline ensemble:** combine selected B0–B4 methods only when development evidence justifies a rule frozen before TEST.

Every source records timestamp, provenance, calendar lag, and availability. Primary baselines cannot use imputed history.

## Deterministic metrics

Primary metrics are MAE, RMSE, bias, median absolute error, and absolute-target-scaled NMAE. Exact NMAE is:

`mean(abs(pred - actual)) / mean(abs(actual))`

The denominator never uses the signed mean. Ordinary MAPE is forbidden as a headline metric because the target crosses and approaches zero.

Absolute-target-scaled WAPE is secondary:

`sum(abs(pred - actual)) / sum(abs(actual))`

It is numerically equivalent to the frozen NMAE on the same target set and must be labelled clearly rather than presented as positive-load WAPE.

## Import/export evaluation

No nonzero physical tolerance is documented, so primary `epsilon = 0 kW`. Import is `y > 0`, export `y < 0`, and neutral `y == 0`. Any later sensitivity tolerance must be predefined without TEST tuning.

Grid mode is diagnostic-only. Report import/export/neutral MAE where supported, sign accuracy, import/export recall, false-import/false-export rates, and zero-crossing error. Realized mode cannot be a feature.

## Daily-profile metrics

On complete primary days report daily MAE, RMSE, bias, maximum absolute interval error, signed energy-exchange error `sum(pred-actual)*0.25 h`, absolute daily energy error, and import/export energy errors where meaningful. Peak timing is secondary and must not be overinterpreted on export-dominant days.

## Model-selection rules

P1 selects models/configurations using the equal-target-weighted MAE over the union of eligible 2021 and 2022 validation targets. Report each fold separately. P2 assesses recency/structure robustness and cannot retrospectively select the primary model from TEST.

Prediction coverage is mandatory: eligible targets, predicted targets, coverage fraction, complete-day forecast coverage, and fallback usage. Methods covering only easy targets cannot be declared superior from raw MAE alone.

## Foundation-model fairness

Chronos/Chronos-2, TimesFM, TimeGPT, or any later foundation model must use the same issuance, target day, TEST, provenance rule, metrics, future-data constraint, and DST handling. Report its maximal protocol-valid set and a `COMMON_CONTEXT_SUBSET` when strict pairwise comparison requires one. Report architecture-natural and common-input comparisons where possible.

Zero-shot evaluation precedes fine-tuning, and the two are never mixed. Candidate contexts are 7, 14, 28, and optionally 56 days, selected with development folds only. Hidden author-imputation is forbidden.

## Probabilistic evaluation rules

Probabilistic methods use the same 2023 TEST. Calibration is chronological and restricted to development data; 2023 calibration is forbidden. The exact rolling/cross-fold calibration design requires a separately versioned extension before probabilistic training.

## Coherent-scenario requirement

Marginal quantiles alone are insufficient for downstream stochastic optimization. Later work must create temporally coherent daily trajectories, for example through whole-day residual blocks, trajectory bootstrap, dependence-aware reconstruction, or valid model samples.

## Bootstrap and statistical comparison

Major TEST comparisons use local calendar day as the bootstrap unit, at least 2,000 resamples, and seed `20261029`. Fifteen-minute intervals are never independently bootstrapped.

Pairwise comparisons use common eligible targets and common complete days. Report paired mean error difference, whole-day bootstrap 95% CI, daily win rate, common target/day counts, and coverage. Qualify comparisons on materially different subsets.

## Downstream market/flexibility compatibility

Forecast issuance stays 13:00 D-1. A later market decision may occur at a slightly later verified price-publication time. Optimization normally uses complete primary days, coverage-aware forecasts, and coherent trajectories. H3 does not assume prices, device constraints, or thermal schedules are already known.

## Cross-dataset SPARK comparison

SPARK is a gross/site-consumption-like demand task. Honda is signed net grid exchange. Raw kW MAE is therefore not a scientifically fair cross-site comparison. Cross-dataset work should emphasize normalized error, model ranking, calibration, temporal generalization, regime robustness, and downstream regret/value without implying target equivalence.

## Frozen protocol summary

The authoritative configuration is `config/honda_forecasting_protocol_frozen.json`. The forecasting index is a SILVER protocol index, not a GOLD feature table. All future Honda methods must preserve the canonical target set, chronology, information cutoff, feature timing, signed output, coverage reporting, and untouched 2023 TEST.

## Limitations

H3 freezes rules before model evidence by design. The CHP date remains ambiguous; modernization is month-precision; epsilon lacks a documented nonzero tolerance; future weather/PV/CHP forecasts and price timing are outside the primary information set; probabilistic calibration details require a versioned extension. None of these limitations authorizes use of 2023 labels during development.

## Next task

H4 should be **Leakage-safe adaptive baseline benchmark for Honda signed grid exchange** and implement only B0–B5 under this protocol. It must establish daily versus weekly structure, export predictability, sensitivity to including 2020, 2023 difficulty, and hard lead/structural periods before any machine-learning or foundation model is introduced.
