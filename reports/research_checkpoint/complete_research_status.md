# Industrial Energy Research: Complete Scientific Checkpoint

Audit date: 2026-11-05  
Honda repository: `C:\Users\asker\Documents\GenAIProjects\industrial_flexibilty_honda`  
SPARK repository: `C:\Users\asker\Documents\GenAIProjects\Industrial_energy_forecasting`

## 1. Executive summary

This audit inspected the actual Git history, frozen configurations, manifests, reports, metric tables and current test suites of the Honda and SPARK repositories. It did not retrain a model, regenerate forecasts, rerun an economic backtest, alter a scientific configuration or modify an upstream artifact. The only new files are the five checkpoint documents in `reports/research_checkpoint/`.

The forecasting evidence is mature. SPARK establishes a leakage-aware 15-minute facility-demand benchmark in which LightGBM improves on historical analogues (2024 MAE 2.352 kW versus B3 2.710 kW and B5 2.571 kW). Its calibrated marginal quantiles achieve WIS 1.4743, 80% coverage 80.44% and 90% coverage 89.91% on 2024, but holiday calibration remains poor and marginal calibration does not validate joint daily trajectories. Honda establishes a particularly strong provenance contribution: its primary target is signed public-grid exchange, not gross demand, and only `OBSERVED_ORIGINAL` labels are admitted. On 34,161 eligible 2023 intervals, site-specific LightGBM (MAE 71.513 kW) improves on the frozen B5 historical ensemble (76.480 kW) but both recognize export weakly (29.752% and 30.809% recall). Zero-shot TimesFM 3.0 (95.431 kW) and Chronos-2 (103.672 kW) are decisively worse than LightGBM in this implementation; this does not test fine-tuning or prove a general foundation-model hierarchy.

The uncertainty and decision layers were genuinely implemented after H6. Stage 1 froze 50 whole-day residual trajectories (`S2`) and demonstrated materially better dependence representation than independent residual sampling: for example, Honda development variogram score fell from 25.0395 to 8.2562 and ACF RMSE from 0.4979 to 0.2109. Marginal coverage remained imperfect, rare DST days were incompletely supported and the selected method is correctly labelled `READY_WITH_LIMITATIONS`, not fully calibrated operational uncertainty.

Stage 2 implemented and synthetically validated a linear virtual-production-buffer model, a synthetic thermal buffer, signed import/export accounting, connection/peak constraints, a shared first-stage stochastic schedule and CVaR. These are mathematically sound counterfactual archetypes, not identified factory assets. SPARK price publication timestamps were unavailable, so scheduling is `EX_POST_PRICE_AWARE`; Honda had no accepted 2023 price series and remained `UNSUPPORTED_TIMING`.

Stage 3 ran a predeclared exploratory comparison on 273 complete SPARK 2024 days. Under a hypothetical ±5 kW, ±10 kWh F1 buffer, deterministic LightGBM saved €1.999617/day, independent scenarios €1.800671/day and coherent scenarios €1.747536/day against a €28.447594/day no-flex wholesale-cost reference. Coherence did not add decision value: deterministic minus coherent cost was −€0.252081/day (95% CI [−€0.315669, −€0.199605]) and independent minus coherent was −€0.053135/day ([−€0.109641, −€0.011827]); negative means the first policy was cheaper. Physical capacity changed coherent-policy savings from €0.740766/day to €3.057647/day, much more than the forecast-policy effect. Yet realized zero-export caps were violated on 82 deterministic, 16 independent and 11 coherent days, prices were ex-post known, capacity was hypothetical, 2024 had already been examined, and only SPARK was economically evaluated. The report's Category C classification—technically sound exploratory simulation—is justified.

The strongest defensible thesis is therefore not “probabilistic forecasting creates savings.” It is that rigorous provenance, information timing, physical constraints and decision structure determine whether forecast sophistication has operational value; in the implemented SPARK case, coherent uncertainty improved statistical dependence but did not improve realized cost. A thesis can be drafted now. The most valuable next evidence is not another broad model search: it is independently timestamped market data, real asset/tariff constraints and a new untouched period, preferably including Honda or another signed-exchange site.

## 2. Audit scope and evidence labels

Evidence is labelled as follows:

- **VERIFIED FROM REPOSITORY**: read directly from a tracked artifact, a local artifact whose hash matches a manifest, or an executed non-mutating test.
- **REPORTED IN EXISTING DOCUMENTATION**: present in a committed report but not independently recomputed in this audit.
- **INFERRED**: a transparent interpretation of multiple verified artifacts.
- **NOT VERIFIED**: absent, inaccessible or not independently established.

The Honda worktree began at `be5ddb47ab58e43f70c72400c3eb09391a1b54a5` on `master`; SPARK began at `32cff65f7f5384d08095d8d0325c45e8ce002ec3` on `main`. Both were clean. All named H1–H6 commits are reachable. Stage checkpoints were discovered rather than assumed: Stage 1/H7 `bb0536df77f709a20452ed3c86ab16e66a5e0555`, Stage 2 `aff0c933...`, and Stage 3 `be5ddb47...`.

Current test status is not completely green. SPARK passed 104/104 tests. Honda passed 212/213; `tests/optimization/test_stage2_optimization.py::test_h1_h6_protected_artifacts_unchanged` fails because its post-H7 allow-list accepts Stage 2 directories only and therefore rejects the subsequently committed `decision_value` directories. This is a stale protection test, not evidence of a changed H1–H6 artifact, but it is still a real test-suite failure and was not edited during this audit.

## 3. Dataset identity

### Table A — target semantics and temporal coverage

| Dataset | Physical target | Sign | Unit / cadence | Source period | Forecast study | Issue time | Evidence |
|---|---|---|---|---|---|---|---|
| KIT SPARK | Facility/site electrical demand | Nonnegative demand | kW, 15 min | 2021–2024 | 2021–2023 development, 2024 test | 12:00 Europe/Berlin D−1 | `reports/gold_summary.md`; forecasting reports; commit `39c1faa` onward |
| Honda R&D Europe, Offenbach | Physical net exchange with public grid | Positive import; negative export | kW, 15 min | 2018–2023 | Primary labels 2020–2023; 2023 test | 13:00 Europe/Berlin D−1 | `reports/honda_forecasting_protocol.md`; H2/H3 manifests |

These targets are not interchangeable. SPARK approximates gross/site demand. Honda is already net of onsite PV, CHP and other site flows. Negative Honda values are exports, not “negative consumption.” Raw kW MAE must not be compared cross-site as a scientific ranking. Honda's reconstructed gross load is explicitly descriptive and not independently validated as a forecasting target.

## 4. Measurement quality and provenance

### Table B — SPARK measurement-quality results

| Finding | Value | Population / unit | Evidence |
|---|---:|---|---|
| Raw cadence | 5 s | Total-load archives | `reports/raw_quality/total_load_*.json` |
| 2021 valid samples | 4,968,471 / 6,307,200 (78.77%) | 5-s samples | same |
| 2022 valid samples | 5,023,226 / 6,307,200 (79.64%) | 5-s samples | same |
| 2023 valid samples | 4,520,517 / 6,307,200 (71.67%) | 5-s samples | same |
| 2024 valid samples | 4,320,457 / 6,324,480 (68.31%) | 5-s samples | same |
| Longest missing run | 1.58 h; 24.89 h; 13.70 d; 29.68 d | 2021–2024 respectively | same; seconds converted |
| Empirical 75% rule | MAE 0.0060 kW; P95 0.0212; P99 0.0495 | 3,600 reconstructed intervals | `reports/missingness/coverage_threshold_decision.csv` |
| Approx. 70% comparison | MAE 0.0416 kW; P95 0.1364; P99 0.2687 | 3,600 reconstructed intervals | same |
| GOLD targets | 133,847 | eligible 15-min intervals | `reports/gold_manifest.json` |
| Yearly GOLD | 34,948; 34,884; 32,676; 31,339 | 2021–2024 | same |

The 75% aggregation threshold is an empirical data-engineering finding for this meter/archive, not a universal physical law. DST ambiguity is explicitly audited; UTC is the unique physical key. The PV series overlaps only 2024 and is not a basis for multi-year causal decomposition. Intervals below the coverage threshold and unresolved ambiguous intervals are excluded rather than filled into the target.

### Table G — Honda provenance and eligibility

| Year | Expected physical intervals | Primary `OBSERVED_ORIGINAL` | Primary % | Sensitivity eligible | Complete primary days | Partial / zero days |
|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 35,136 | 24,145 | 68.72% | 31,874 | 187 | 74 / 105 |
| 2021 | 35,040 | 34,281 | 97.83% | 34,281 | 342 | 23 / 0 |
| 2022 | 35,040 | 34,624 | 98.81% | 34,624 | 356 | 6 / 3 |
| 2023 | 35,040 | 34,161 | 97.49% | 34,161 | 347 | 13 / 5 |

Source: `results/protocol_h3a/target_availability_by_year.csv` (VERIFIED FROM REPOSITORY). The observation-aware SILVER and provenance masks match H2 manifest hashes `d9d934...10df` and `f93d5b...b778`; these large generated Parquet files are intentionally Git-ignored, while their recipes, manifests and hashes are tracked. H2 distinguishes original observations, corrected non-imputed values, author-filled/imputed values, unresolved missingness and structural unavailability. Primary labels admit only original observations; corrected values are sensitivity-only. In 2018–2019 the released grid series is corrected/imputed rather than primary original under this policy. PV and CHP source hierarchies change over time, which is recorded as era metadata rather than silently harmonized.

## 5. Frozen forecasting protocols

SPARK uses a leakage-safe day-ahead task at noon D−1. Honda freezes: at 13:00 Europe/Berlin D−1, forecast every physical 15-minute interval of local day D; historical data are available only if `interval_end_utc <= forecast_issue_time_utc`. Honda primary targets are `OBSERVED_ORIGINAL`; predictors reject imputed, unknown and unresolved history. P1 uses train 2020/validate 2021 and train 2020–2021/validate 2022, then refits 2020–2022 and tests 2023. P2 excludes 2020 as a structural sensitivity. DST target days retain 92/96/100 physical intervals. The frozen H3 configuration hash is `281f49331bc3b3c31aec0d33b50f6a83a149261db93ce04a29d67479cb62676e`.

## 6. SPARK deterministic forecasting

### Table C — SPARK deterministic test performance

| Method | 2024 MAE kW | RMSE kW | Coverage | Notes |
|---|---:|---:|---:|---|
| B3 historical analogue | 2.7096 | 4.2319 | 100% | 31,339 eligible targets |
| B5 ensemble | 2.5713 | 4.1518 | 96.19% | 30,145 predictions; subset caveat |
| LightGBM | 2.3520 | 3.4659 | 100% | bias −0.4371 kW |

Source: `results/baselines/baseline_metrics.csv` and `results/lightgbm/lightgbm_metrics.csv`. The chronology is 2021–2022 training, 2023 validation/model selection and untouched-at-selection 2024 test. Features are future-known calendar variables plus issuance-safe historical demand summaries/analogues; weather/PV future realizations are not allowed. Against B3, paired daily LightGBM minus baseline MAE is −0.3537 kW (95% CI [−0.5089, −0.1958], 330 days). Against B5 on common predicted targets it is −0.2171 kW ([−0.3773, −0.0657], 320 days). Holiday MAE is still 3.4636 kW for LightGBM but much lower than B3/B5 (8.2790/9.6727). Errors increase in the middle lead bands and on high-demand periods; yearly distribution changes limit stationarity claims.

## 7. SPARK probabilistic forecasting and regime calibration

### Table D — SPARK probabilistic results

| Configuration / period | WIS kW | 50% cov. | 80% cov. | 90% cov. | Notes |
|---|---:|---:|---:|---:|---|
| R0 global, 2023 validation | 1.3924 | — | — | — | selected global calibration |
| R4 hierarchical, 2023 validation | 1.3910 | — | — | — | tiny WIS difference; added complexity |
| R0, 2024 test | 1.4743 | 50.06% | 80.44% | 89.91% | widths 3.971/8.030/10.950 kW |
| R4, 2024 diagnostic | 1.4646 | 47.76% | 80.54% | 90.09% | not a basis for retroactive selection |
| R0 holiday subgroup, 2024 | 2.1746 | — | 50.35% | 68.29% | severe conditional undercoverage |

Quantiles are q05, q10, q25, q50, q75, q90 and q95. Calibration is validation-derived and quantile crossing is corrected by monotonic rearrangement. R0 was retained because the R4 development advantage was negligible relative to complexity/robustness and because test diagnostics cannot retroactively choose R4. Operating-level regimes use broad pre-issuance information and are not tomorrow's realized regime. The experiment establishes useful **marginal** calibration overall; it does not establish a valid multivariate daily distribution, tail reliability on holidays, or causal regime effects.

## 8. Original SPARK flexibility simulation

### Table E — original flexibility economics (274 complete 2024 days)

| Policy | Mean savings €/day |
|---|---:|
| Perfect foresight | 0.12451 |
| B3 | 0.08366 |
| Deterministic LightGBM | 0.08972 |
| Marginal-quantile probabilistic | 0.09107 |

No-flex wholesale cost was €28.3816/day. The probabilistic increment over deterministic was €0.00135/day with whole-day bootstrap 95% CI [−€0.00007, +€0.00273], so no robust incremental value was demonstrated. The linear program assumed 20% shiftable demand, ±2 h movement, daily energy conservation and a peak no higher than the original day; q10/q50/q90 marginal curves served as scenarios. Historical prices were complete but their decision-time publication was not established, making the simulation ex-post price-aware. Perfect foresight is an oracle upper bound, not implementable scheduling.

### Table F — physical-assumption sensitivity

| Change | Mean savings €/day |
|---|---:|
| 5%, 10%, 20%, 30% shiftable at ±2 h | 0.0228; 0.0455; 0.0911; 0.1366 |
| 20% shiftable at ±1, ±2, ±4 h | 0.0211; 0.0911; 0.3271 |
| Peak cap 1.0×, 1.1×, 1.25× | 0.0911; 1.1739; 1.3963 |

The physical assumption effects are orders of magnitude larger than the €0.00135/day probabilistic increment. This supports a case-specific conclusion that assumed flexibility dominates this simulation, not a universal law about industrial flexibility.

## 9. Honda baseline benchmark

### Table H — Honda 2023 model comparison

| Model | MAE kW | RMSE kW | Bias kW | Coverage | Export recall |
|---|---:|---:|---:|---:|---:|
| B0 | 153.040 | — | — | 100% | — |
| B1 | 93.935 | — | — | 100% | — |
| B2 | 100.106 | — | — | 97.43% | — |
| B3 | 85.798 | — | — | 100% | — |
| B4 | 85.315 | — | — | 100% | — |
| B5 | 76.480 | 101.338 | +6.649 | 100% | 30.809% |
| LightGBM M1-CHP | 71.513 | 96.974 | +9.408 | 100% | 29.752% |
| TimesFM 3.0 zero-shot | 95.431 | 129.221 | −0.040 | 100% | 11.219% |
| Chronos-2 zero-shot | 103.672 | 139.959 | −6.184 | 100% | 11.543% |

B5 is an equal-mean ensemble of B1, B2, selected B3 `N8_MIN4` and B4 month × quarter-hour × weekend climatology, requiring at least three members. It was selected from 2021/2022 folds before 2023. Its hardest frozen lead band is H3 [23,29 h), MAE 104.032 kW. B5 import MAE is 69.349 kW and export MAE 121.900 kW. April is worst (99.967 kW) and December best (59.593 kW). P2 B5 test MAE is 75.853 kW, a small improvement when 2020 is excluded; this is structural sensitivity, not grounds to replace P1 after seeing test results.

## 10. Honda deterministic LightGBM

M0 ablations show that calendar alone is insufficient (concatenated development MAE 80.583 kW); historical analogues reduce this to 68.253, recent state to 68.247, and volatility/export summaries to 67.834. M1-CHP reaches 66.530 kW and is selected; M1-PV is 68.616 and M1-ALL 68.112. Weather and thermal variants were not admissible because primary provenance masks were insufficient. The final model uses 80 features, 196 iterations, objective `regression_l1`, 15 leaves, learning rate 0.03, minimum child samples 50 and feature fraction 0.8. Frozen config hash: `7fff8efbf51fc17ac2524ceeac30dbed6764d4ec55eaf1ea3993c47691dd21ea`.

On 2023, M1-CHP improves B5 by 4.967 kW (6.495%). Paired daily difference is −5.122 kW with 95% CI [−6.833, −3.514]. Yet export recall falls slightly, from 30.809% to 29.752%, while import recall is 98.059%. Thus aggregate MAE improvement is driven mainly by common import states, not successful classification of the rarer export state. Residual diagnostics identify export misrecognition, underprediction of extreme import, temporal autocorrelation and seasonal/structural variation. CHP feature importance is predictive association, not evidence that CHP causes the error pattern.

## 11. Honda zero-shot foundation models

Chronos uses `amazon/chronos-2`, revision `29ec376...`, `chronos-forecasting` 2.3.2, Apache-2.0, 14-day context and calendar + historical CHP covariates. TimesFM uses `google/timesfm-3.0-pytorch`, revision `43046...`, TimesFM 3.0.2, the checkpoint's non-commercial research terms, 28-day context and the same controlled covariate class. Both ran locally on CUDA without gradient updates; no fine-tuning, LoRA or test-based context selection occurred.

TimesFM minus LightGBM daily MAE is +22.969 kW (95% CI [19.023, 27.464]); Chronos minus LightGBM is +31.580 kW ([26.291, 37.161]). Inference for 365 origins took about 206 s and 221 s respectively, versus much lighter site-specific inference. Both are Category C in H6: reproducible zero-shot results that clearly trail the site-specific model. This result does not establish the effect of domain adaptation, other checkpoints, future model versions or fine-tuning. It also does not conclusively rule out benchmark exposure during pretraining.

### Table I — Honda modes and temporal structure

| Finding | Evidence |
|---|---|
| 2023 import/export prevalence | 86.43% / 13.57% of primary intervals |
| B5 import/export MAE | 69.349 / 121.900 kW |
| LightGBM import/export MAE | repository report: import materially lower; export 121.377 kW |
| Export recall | B5 30.809%; LGBM 29.752%; TimesFM 11.219%; Chronos 11.543% |
| Temporal shift | Mean grid exchange 184.73 kW (2020) versus 119.34 kW (2023); Wasserstein 66.07 kW |
| Hard lead range | B5 H3 104.032 kW; LightGBM H3 98.781 kW |

## 12. Stage 1: coherent scenarios

**Status: COMPLETED_AND_COMMITTED; readiness `READY_WITH_LIMITATIONS`.**

### Table J — selected scenario evidence

| Dataset / development scope | Method | Coverage 50/80/90 | WIS kW | Energy score kW | Variogram | ACF RMSE | Complete-day coverage |
|---|---|---|---:|---:|---:|---:|---:|
| Honda 2022 | S1 independent, 50 | .466/.754/.858 | 155.284 | 64.352 | 25.040 | .498 | 100% |
| Honda 2022 | S2 whole-day, 50 | .463/.752/.856 | 155.751 | 63.317 | 8.256 | .211 | 99.44% |
| SPARK rolling 2023 | S1 independent, 50 | .417/.699/.818 | 6.157 | 2.394 | 1.238 | .538 | 78.49% |
| SPARK rolling 2023 | S2 whole-day, 50 | .415/.697/.813 | 6.209 | 2.371 | .402 | .185 | 78.14% |

Residuals are `actual − forecast`. Honda donors are complete out-of-sample 2021 folds evaluated on 2022; SPARK uses expanding past-only 2023 validation donors after 60 completed donor days. Evaluation-day and future residuals are excluded. S2 samples whole physical-day residual trajectories with exact 92/96/100 length; unsupported rare lengths are not stretched. S3 conditions only on future-known season/day type and falls back below 20 donors.

Although S3/50 has slightly better aggregate scores in places, the predeclared robustness rule did not establish a consistent conditional advantage across both datasets, so the simpler S2/50 was frozen. On previously exposed Honda 2023, S2 export Brier score is 0.0834 and any-export probability 0.7776; no test threshold was tuned. S2 improves energy score over S1 by −0.634 kW on Honda and −0.040 kW on SPARK with bootstrap CIs below zero, and much more strongly improves dependence scores. Limitations are incomplete SPARK development coverage, finite donors, S3 fallback, imperfect tail/export calibration and rare DST support. Shape validity and dependence improvement are not equivalent to full probability calibration.

## 13. Stage 2: physical flexibility and markets

**Status: COMPLETED_AND_COMMITTED; readiness `READY_WITH_LIMITATIONS_FOR_STAGE_3`.**

### Table K — framework validation

| Component | Implemented result | Calibration status |
|---|---|---|
| F0 legacy | SPARK-only 20%, ±2 h comparator | legacy hypothetical |
| F1 process buffer | `G=baseline+u`; `x[t+1]=x[t]+u[t]Δt`; power/state bounds; terminal recovery | mathematically validated, not plant identified |
| Exact recovery | source-to-destination energy-flow arcs bounded by configured horizon | validated; state bound alone not mislabelled as deadline |
| F2 thermal | `E[t+1]=retention E[t]+(COP P_el−Q_demand)Δt` | synthetic demand only |
| F3 | import/export caps, scenario-wise/nominal/soft modes, peak shadow cost | validated; no customer contract |
| BESS | future interface only | not implemented / no Honda battery claim |
| Stochastic | one shared `u[t]` across all scenarios | non-anticipativity validated |
| CVaR | standard upper-tail linearization | analytic fixture validated |
| Solver | SciPy 1.17.1 HiGHS | 50×96 scenario fixture ~0.162 s |

F1 counterfactuals are SPARK LOW ±2 kW/±4 kWh, MEDIUM ±5/±10, HIGH ±10/±20; Honda LOW ±25/±50, MEDIUM ±50/±100, HIGH ±100/±200. All have a four-hour recovery-horizon assumption. These were predeclared, not inferred from production records. F2 COP levels 2.5/3.0/3.5 and thermal capacities 25/75/150 kWh_th are synthetic.

Grid accounting uses nonnegative import/export and tariff dominance checks to prevent artificial simultaneous buy/sell arbitrage. Negative wholesale prices are retained. Honda PV and CHP are already in signed exchange and are not subtracted again. Peak prices are daily shadow-price sensitivities, not monthly demand charges. The synthetic suite validates flat/negative prices, two-interval optima, infeasible caps, exact deadlines, thermal COP balance, DST lengths, deterministic/stochastic equivalence and CVaR. Stage 2 does not establish real asset availability or a customer tariff.

SPARK's historical German price file is hourly and expanded to four 15-minute delivery intervals; it lacks actual publication timestamps. Stage 2 therefore classifies it `EX_POST_PRICE_AWARE`. Honda has no accepted 2023 price file and is `UNSUPPORTED_TIMING`. EPEX/SDAC documentation establishes gate-closure conventions and that 15-minute SDAC products began only in October 2025, but does not repair missing historical publication metadata in the source file.

## 14. Stage 3: decision value

**Status: COMPLETED_AND_COMMITTED; publication readiness Category C. Honda economics: NOT EVALUABLE/BLOCKED.**

### Table L — Stage 3 SPARK results (273 common 2024 days)

| Policy | Cost €/day | Savings €/day | Mean regret €/day | Export-cap violation days | Median runtime s |
|---|---:|---:|---:|---:|---:|
| P0 no flex | 28.447594 | 0 | 1.998817 | 0 | .000025 |
| P2 deterministic LGBM | 26.447976 | 1.999617 | −0.000800* | 82 | .0199 |
| P3 independent S1 | 26.646923 | 1.800671 | .198146 | 16 | .2485 |
| P4 coherent S2 | 26.700058 | 1.747536 | .251281 | 11 | .2475 |
| P6 coherent CVaR | 26.700058 | 1.747536 | .251281 | 11 | .3032 |
| P7 oracle | 26.448777 | 1.998817 | 0 | 0 | .0206 |

`*` Negative deterministic regret is not a valid claim of beating the oracle; it arises because realized schedules violate the zero-export cap on 82 days while the oracle remains feasible. The all-day comparison measures forecast-risk exposure but is not fully operationally feasible. Common-feasible-day comparisons still favor deterministic/independent policies, but use different subsets and remain secondary.

The primary paired results use 2,000 whole-day resamples, seed 20261029, Holm adjustment for two comparisons and a seven-day moving-block sensitivity. P2−P4 mean cost is −€0.252081/day, 95% CI [−0.315669, −0.199605]; P3−P4 is −€0.053135/day, [−0.109641, −0.011827]. P4 wins 61.5% of days against P3 but loses on mean because adverse outliers are larger; removing the top 1% absolute differences reduces the mean disadvantage to −€0.022541/day. Coherence changes schedules (mean absolute P3/P4 interval difference 1.018 kW) but not beneficially under this objective.

Physical sensitivity is larger: coherent-policy savings LOW/MEDIUM/HIGH are €0.740766/€1.747536/€3.057647 per day. Import caps of 40/30/20 kW are feasible on only 120/31/0 of 273 days. Peak shadow prices reduce mean savings and can create negative-savings days. Export remuneration multipliers 1.0/0.5/0.0 alter savings only slightly here because SPARK export is constrained. CVaR changes dispatch/peak modestly but leaves realized energy cost and historical tail metrics unchanged in this configuration.

### Stage 3 hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Coherent beats independent | NOT_SUPPORTED | S2 costs €0.0531/day more on average; CI excludes zero |
| Probabilistic beats deterministic | NOT_SUPPORTED | S2 costs €0.2521/day more; CI excludes zero |
| Physical flexibility dominates | SUPPORTED conditionally | €2.3169/day capacity range vs €0.2521 policy effect; hypothetical capacities |
| Market structure matters | PARTIALLY_SUPPORTED | grid/peak sensitivity meaningful; tariff range small in this case |
| CVaR reduces realized risk | NOT_SUPPORTED | no realized cost/tail change |
| Cross-site heterogeneity | NOT EVALUABLE economically | Honda price/timing absent |
| Honda export-state value | NOT EVALUABLE | no Honda settlement |

The 273 evaluated days span 6 January–1 December 2024 and are all 96-interval days; the scenario interface excludes transition days, so no DST economic claim is made. The original Stage 3 frozen file contained a descriptive end-date typo corrected by a hash-tracked amendment without changing days or outcomes.

## 15. Cross-dataset interpretation

SPARK and Honda support a methodological comparison, not a pooled performance league. SPARK has a small positive-demand scale, excellent LightGBM MAE and externally available but ex-post-timed prices. Honda has a larger signed-exchange scale, frequent export, onsite generation and stronger provenance complexity, but no accepted economic price input. Both show temporal structure and distribution change; both benefit from site-specific historical features. Only SPARK supports the present economic simulation, so a two-site decision-value claim is not established.

## 16. Robust, negative and unresolved findings

### Robust findings

1. Provenance and causal information cutoffs materially change what counts as valid data; Honda's signed target is observation-aware and cannot be treated as gross load.
2. Site-specific LightGBM robustly improves aggregate deterministic accuracy over historical baselines on both sites, but improvement is conditional rather than universal across modes.
3. Whole-day residual sampling better represents temporal dependence than independent interval sampling on development data.
4. Under the frozen exploratory SPARK optimization, coherent scenarios do not improve realized cost over deterministic or independent decisions.
5. Assumed physical flexibility has a substantially larger effect than uncertainty representation in the tested setup.

### Negative findings

1. Honda LightGBM does not improve export recall despite lower overall MAE.
2. Zero-shot Chronos-2 and TimesFM are materially worse than B5 and LightGBM here.
3. Regime calibration does not robustly solve holiday uncertainty.
4. Original marginal-quantile flexibility adds only €0.00135/day with a CI crossing zero.
5. CVaR provides no realized cost/tail benefit in Stage 3.
6. A two-site economic conclusion is unavailable.

### Unresolved

- Whether future PV/CHP changes cause seasonal errors: **NOT ESTABLISHED**.
- Whether coherent scenarios matter under a real tariff, hard network constraint or a measured flexible asset: **NOT ESTABLISHED**.
- Whether improved export probabilities create Honda decision value: **NOT ESTABLISHED**.
- Whether Stage 3 findings generalize beyond exposed SPARK 2024 and hypothetical F1: **NOT ESTABLISHED**.

## 17. Independent answers to the major scientific questions

1. **Why LightGBM beats zero-shot models on Honda.** It uses site-specific analogue, recent-state, export-history and CHP-history features selected on chronological folds; zero-shot models receive generic pretrained representations and no site adaptation. This is a plausible mechanism, not a causal proof.
2. **Why MAE improves but export recognition does not.** Imports are 86.43% of intervals. Reducing continuous error in the majority regime lowers MAE even if the sign boundary and rare export events remain poorly classified.
3. **Are PV/CHP changes responsible?** **NOT ESTABLISHED.** Era associations and CHP predictive value exist, but no causal intervention or validated component balance identifies the cause.
4. **What SPARK calibration establishes.** Good aggregate marginal coverage/WIS; it does not establish joint daily paths or holiday-tail calibration.
5. **Why uncertainty adds little or negative value.** The objective is close to linear expected energy cost, the schedule is shared, terminal energy conservation cancels some errors, and physical/cap constraints dominate. Dependence matters only through nonseparable elements; in Stage 3 it changed schedules but hurt mean cost.
6. **Does physics matter more?** Supported within the frozen simulations, not universally.
7. **Does temporal dependence influence the objective?** Yes, schedules differ, but not beneficially. Pure separable risk-neutral expected cost mainly uses marginal means; state, cap, peak and CVaR couplings create the channels.
8. **Which constraints can create value for coherence?** Scenario-wise grid caps, daily peaks, asymmetric import/export tariffs, tail-risk objectives and state-coupled persistent errors.
9. **How realistic is flexibility?** Mathematically interpretable but hypothetical; no production orders or measured asset capacities validate it.
10. **Are timestamps defensible?** Forecast cutoffs are strong; SPARK market timing is ex-post and Honda is unsupported.
11. **Does signed exchange differ qualitatively?** Yes: sign transitions, export remuneration and already-netted PV/CHP fundamentally alter the decision balance.
12. **Are two sites enough?** Enough for comparative engineering evidence, not broad external generalization; only one has economics.
13. **What could be novel?** The integrated provenance-to-decision falsification study and empirical demonstration that better dependence need not create decision value may be distinctive; literature verification is required.
14. **Diminishing returns.** Additional zero-shot checkpoints, broad deterministic tuning and finer regime calibration have low expected value without new data/constraints.
15. **What should an academic reviewer inspect?** Market timing, asset identification, residual chronology, cap feasibility, exposed-test reuse and novelty against decision-focused energy forecasting literature.

## 18. Hypothesis status matrix

### Table M — supported versus unsupported scientific claims

| Claim | Dataset / experiment | Numerical evidence | Key assumption | Verdict |
|---|---|---|---|---|
| 75% raw coverage is adequate for 15-min SPARK means | SPARK reconstruction | MAE .0060 kW vs .0416 near 70% | sampled windows representative | SUPPORTED for this archive |
| LightGBM improves SPARK demand forecasts | SPARK 2024 | MAE 2.352 vs B3 2.710; paired CI below 0 | 2024 test validity | SUPPORTED |
| Marginal quantiles are calibrated overall | SPARK 2024 | 80%→80.44%; 90%→89.91% | aggregate population | SUPPORTED |
| Holiday uncertainty is solved | SPARK holiday | 80%→50.35%; 90%→68.29% | small subgroup | NOT_SUPPORTED |
| Marginal probabilistic scheduling adds value | SPARK original flexibility | +€0.00135/day; CI crosses 0 | hypothetical 20%, ex-post prices | INCONCLUSIVE |
| Honda M1-CHP improves aggregate accuracy | Honda 2023 | −4.967 kW; paired CI [−6.833,−3.514] | exposed final test but frozen selection | SUPPORTED |
| Honda M1-CHP improves export recognition | Honda 2023 | recall 29.752% vs B5 30.809% | epsilon 0 | NOT_SUPPORTED |
| Zero-shot FMs beat site model | Honda 2023 | +22.97/+31.58 kW vs LGBM | selected official checkpoints | NOT_SUPPORTED |
| Whole-day sampling improves dependence | Honda/SPARK development | variogram and ACF substantially lower | residual library chronology | SUPPORTED |
| S2 is fully calibrated operational uncertainty | Stage 1 | marginal errors, rare-day gaps | finite donors | NOT_SUPPORTED |
| F1 represents actual plant flexibility | Stage 2 | no equipment calibration | hypothetical capacities | INVALID_COMPARISON |
| Coherent scenarios improve realized cost | Stage 3 SPARK | P3−P4 −€0.0531; CI <0 | ex-post, hypothetical F1 | NOT_SUPPORTED |
| Physical capacity matters more in tested setup | Stage 3 | €2.3169 range vs €0.2521 policy effect | chosen grid | PARTIALLY_SUPPORTED |
| Honda export uncertainty has economic value | Honda | no accepted price input | — | NOT_TESTED |

## 19. Scientific integrity review

| Issue | Status | Audit conclusion |
|---|---|---|
| Dataset/target provenance | PASS_WITH_LIMITATION | strong manifests; large SILVER files are local ignored artifacts but hashes match |
| Target/future-feature leakage | PASS_WITH_LIMITATION | prior audits and tests are strong; this audit did not recompute every row |
| In-sample residual contamination | PASS_WITH_LIMITATION | Stage 1 chronology is explicit; not independently replayed |
| Test-based selection | PASS_WITH_LIMITATION | frozen configs precede scoring; years have been repeatedly examined across later hypotheses |
| Causal overstatement | PASS | reports generally label associations and counterfactuals correctly |
| Gross-load reconstruction | PASS | explicitly excluded from primary forecasting |
| Artificial net-export flexibility | PASS_WITH_LIMITATION | Stage 2 avoids fraction×abs(net); F1 remains hypothetical overlay |
| PV/CHP double counting | PASS | signed exchange model explicitly prevents re-subtraction |
| Physical realism | PASS_WITH_LIMITATION | constraints are correct; capacities/processes are unmeasured |
| Market timing | FAIL for operational claim | SPARK is ex-post; Honda unsupported; reports disclose this |
| Market-price resolution | PASS_WITH_LIMITATION | hourly products correctly repeated, not called separate quarter-hours |
| Customer tariff realism | NOT_ASSESSED | no customer contract; only hypothetical wholesale/shadow tariffs |
| Simultaneous import/export | PASS_WITH_LIMITATION | tariff dominance and synthetic tests; not every historical LP independently inspected |
| Units/energy balance | PASS_WITH_LIMITATION | centralized units and fixtures; not a metrological validation |
| Infeasible dispatch handling | PASS | infeasible caps reported; not silently zeroed |
| Hidden fallback scheduling | PASS | no predeclared hidden fallback found |
| Oracle comparability | PASS_WITH_LIMITATION | same F1/tariff, but deterministic cap violations make all-day regret non-interpretable |
| Bootstrap independence | PASS_WITH_LIMITATION | day bootstrap plus 7-day block sensitivity; only 273 days |
| Multiple comparisons | PASS_WITH_LIMITATION | Holm for two primary tests; sensitivities remain exploratory |
| Day-selection bias | PASS_WITH_LIMITATION | common intersection explicit; scenario coverage excludes rare/DST days |
| Operational deployability | FAIL if claimed | no such claim is supportable; existing reports classify exploratory |
| Current test suite | FAIL | Honda 212/213 due stale Stage 2 path allow-list; SPARK 104/104 |

## 20. Novelty audit

| Candidate contribution | Provisional classification |
|---|---|
| Empirical SPARK aggregation threshold | Useful applied contribution; potentially distinctive dataset finding |
| Observation-provenance-safe signed-exchange forecasting | Useful applied contribution; possibly distinctive empirical workflow |
| LightGBM vs zero-shot foundation models | Standard methodology with useful industrial evidence |
| Conditional regime calibration | Standard methodology; negative/applied contribution |
| Coherent residual trajectories | Established methodology, carefully applied |
| Virtual production buffer | Established storage/inventory-style abstraction |
| Market-timed forecast dispatch | Established optimization principle; current timing is not operationally validated |
| Physics vs forecast sophistication | Potentially distinctive empirical negative result, but simulation-dependent |
| CVaR scheduling | Standard methodology; negative result here |
| Cross-site decision value | Insufficient evidence; Honda economics absent |

External literature review is required before any novelty claim.

## 21. Research decision gates

| Activity | Gate | Reason |
|---|---|---|
| More deterministic forecasting | STOP | strong baselines and LGBM already answer the main accuracy question |
| More zero-shot foundation models | STOP | diminishing returns without a new scientific mechanism |
| Foundation-model fine-tuning | DEFER | expensive; only justified by a domain-adaptation research question |
| Further marginal calibration | DEFER | holiday/tail issue needs data, not merely more regimes |
| Coherent scenarios | CONTINUE only in decision studies | S2 exists and is adequate with limitations |
| Export-probability modelling | REDESIGN | potentially valuable; needs proper event objective and decision link |
| PV/CHP decomposition | NOT_YET_DECIDABLE | requires verified gross/component measurements |
| Virtual process buffer | CONTINUE as counterfactual | useful abstraction, not plant claim |
| Thermal flexibility | DEFER | no validated thermal demand/comfort parameters |
| Battery simulation | DEFER | separate asset question; avoid scope inflation |
| Market-timing validation | CONTINUE | highest-priority credibility gap |
| Physical parameter identification | CONTINUE | highest-priority operational gap |
| Risk-aware optimization | DEFER | current CVaR adds no value; revisit with credible constraints/scenarios |
| Decision-focused forecasting | NOT_YET_DECIDABLE | first secure real decision inputs and external holdout |
| Cross-site replication | REDESIGN | Honda economics blocked; seek compatible price/tariff data |
| Additional raw Honda acquisition | CONTINUE selectively | only if it resolves provenance/components or extends years |
| New external holdout | CONTINUE | essential for stronger confirmation |
| Thesis writing | CONTINUE | evidence is sufficient for an honest, bounded thesis |
| Publication preparation | CONTINUE cautiously | lead with negative decision result and limitations; literature audit first |

## 22. Ranked research risks (Table N)

| Rank | Risk | Severity | Consequence | Minimum mitigation |
|---:|---|---|---|---|
| 1 | Hypothetical asset capacities/process | Critical | economic value may not transfer to a factory | real process/asset constraints |
| 2 | Ex-post price timing | Critical | scheduling is not proven implementable | timestamped publication/decision data |
| 3 | Previously exposed years | High | confirmatory interpretation weakened | new untouched period/site |
| 4 | Honda economics absent | High | cross-site decision thesis incomplete | authoritative DE-LU prices and timing |
| 5 | Realized cap violations | High | some savings schedules are operationally infeasible | robust/soft cap design and measured limits |
| 6 | Scenario marginal/tail calibration | High | risk conclusions may be unreliable | additional OOS donor years and calibration |
| 7 | No production/tariff identification | High | bill savings cannot be claimed | plant and contract data |
| 8 | Rare DST/scenario-day coverage | Medium | incomplete calendar representativeness | more exact-length donors or principled DST method |
| 9 | Export event weakness | Medium | Honda operational risk poorly represented | event-aware but leakage-safe modelling |
| 10 | Current test regression | Medium | repository no longer fully green | update protection test in a separately authorized maintenance change |

## 23. Mathematical and economic glossary

- **Signed grid exchange:** public-grid power; Honda uses positive import and negative export.
- **Gross facility demand:** total consumption before behind-the-meter generation; SPARK approximates this, Honda primary target does not.
- **Forecast issuance:** timestamp at which forecast information is frozen (SPARK 12:00, Honda 13:00 D−1).
- **Target eligibility:** provenance and completeness rule deciding which realized labels may be scored.
- **Observed-original:** value traceable to an original released measurement, not a correction or fill.
- **MAE / RMSE:** mean absolute / root mean squared forecast error in kW.
- **NMAE:** MAE divided by mean absolute target, avoiding cancellation in signed data.
- **WIS:** weighted interval score combining interval width and penalties for misses; lower is better.
- **Coverage:** fraction of realized targets inside a nominal prediction interval.
- **Energy score:** proper multivariate score for scenario ensembles; lower is better but relatively insensitive to dependence misspecification.
- **Variogram score:** multivariate score emphasizing pairwise/temporal dependence; lower is better.
- **Export recall:** fraction of realized export intervals predicted as export at the frozen zero threshold.
- **Brier score:** mean squared error of event probabilities.
- **Whole-day bootstrap:** resampling local calendar days, preserving intraday dependence.
- **F1 buffer state:** `x[t+1]=x[t]+u[t]Δt`; positive state is advanced work, negative deferred work.
- **Non-anticipativity:** all future scenarios share the same committed first-stage control schedule.
- **Perfect foresight/oracle:** unattainable comparator using realized future values under the same physical constraints.
- **Realized savings:** no-flex realized cost minus policy realized cost; not customer profit.
- **Regret:** policy realized cost minus comparable oracle cost.
- **CVaR:** mean cost in the upper tail beyond a selected quantile, represented by a linear auxiliary formulation.
- **Import/export tariff:** separate price paid for imports and received for exports; dominance is required to avoid artificial buy/sell arbitrage.
- **Physical peak cap:** connection limit on power, distinct from a monetary peak charge.
- **Market gate closure:** deadline for day-ahead orders; not identical to publication time.
- **Forecast vs price availability:** a later scheduling decision may use published prices but may not silently refresh the earlier load forecast.
- **Coherent scenario:** full-day path whose errors retain plausible temporal dependence, not independently sampled marginal intervals.

## 24. Current research position and conclusion

The project began as industrial load forecasting and evolved into a provenance-aware, decision-focused comparison. Data engineering and deterministic forecasting are sufficiently answered for thesis purposes. Marginal probabilistic forecasting is useful but conditionally weak. Scenario generation has a validated interface and clear limitations. The optimization framework is mathematically usable but physically uncalibrated. The only integrated economic study is a previously exposed, ex-post-price-aware SPARK counterfactual, and it yields a scientifically useful negative result: better temporal dependence does not automatically create better decisions.

The work is ready for thesis drafting as a transparent Category C exploratory study with strong software and methodological controls. It is not ready for operational savings claims or a two-site economic publication. Independent literature review should now decide whether the integrated negative finding is sufficiently distinctive, and whether the minimum next experiment should be (a) real asset/tariff identification plus a new holdout, (b) Honda price/timing acquisition and export-focused decisions, or (c) no further modelling before thesis synthesis.

## 25. Principal evidence references

- SPARK: `reports/missingness_reliability_study.md`, `reports/gold_summary.md`, `reports/day_ahead_baselines.md`, `reports/deterministic_lightgbm.md`, `reports/probabilistic_lightgbm.md`, `reports/regime_aware_probabilistic.md`, `reports/flexibility_optimization.md` and corresponding `results/**` tables/manifests.
- Honda H1–H6: `reports/honda_reduced_dataset_audit.md`, `honda_provenance_and_silver.md`, `honda_forecasting_protocol*.md`, `honda_baseline_benchmark.md`, `honda_deterministic_ml_benchmark.md`, `honda_foundation_model_benchmark.md`.
- Stage 1: `reports/scenarios/honda_h7_coherent_scenario_study.md`, `config/scenarios/scenario_method_frozen.json`, `results/scenarios/honda_h7_manifest.json`.
- Stage 2: `reports/optimization/honda_stage2_flexibility_and_market_framework.md`, `config/optimization/*`, `results/optimization/honda_stage2_manifest.json`.
- Stage 3: `reports/decision_value/industrial_flexibility_decision_value_study.md`, `config/decision_value/stage3_experiment_frozen.json`, `results/decision_value/stage3_manifest.json`.

