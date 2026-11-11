# Perplexity Deep Research Handoff

## Independent review requested

Please act as a skeptical academic reviewer of an MSc-scale industrial-energy forecasting and decision-value programme. Do not automatically endorse the existing roadmap. Verify the literature and challenge the physical, statistical and market assumptions. This document is self-contained; repository references are supplied only for traceability.

Audit date: 2026-11-05. The audit found both source repositories accessible and clean before audit outputs were created. Honda HEAD was `be5ddb47ab58e43f70c72400c3eb09391a1b54a5`; SPARK HEAD was `32cff65f7f5384d08095d8d0325c45e8ce002ec3`.

## A. Researcher and project background

The project studies leakage-safe day-ahead industrial electricity forecasting and asks whether uncertainty representations create downstream value under physically constrained flexibility. It contains two public-data case studies:

1. **KIT SPARK:** positive facility electrical demand, 15-minute means, 2021–2024.
2. **Honda R&D Europe, Offenbach:** signed public-grid exchange, 15 minutes, 2018–2023, with primary forecasting labels concentrated in 2020–2023.

The work progressed from data forensics through deterministic and probabilistic forecasting, scenario generation, linear optimization and realized-price counterfactual settlement. It is software-intensive but intentionally distinguishes mathematical validity from plant validation.

## B. Research intention

The central question is not simply which model has the lowest MAE. It is:

> Under which information, market and physical-flexibility conditions do marginal or temporally coherent forecast uncertainties improve industrial energy decisions relative to strong deterministic forecasts?

The project seeks an honest negative result if uncertainty adds little value. It does not claim actual industrial trading profits, identified production schedules, an installed Honda battery, dispatchable Honda CHP or a known customer tariff.

## C. Dataset descriptions and target semantics

| Dataset | Target | Unit | Period | Issue time | Critical caveat |
|---|---|---|---|---|---|
| SPARK | Positive site/facility electricity demand | kW | 2021–2024 | 12:00 Europe/Berlin D−1 | PV overlaps only 2024; raw missingness is substantial |
| Honda | Signed net public-grid exchange: + import, − export | kW | 2018–2023; primary 2020–2023 | 13:00 Europe/Berlin D−1 | already nets PV/CHP; not gross demand |

The targets are physically different. A raw kW error comparison is not a cross-site model ranking. Honda gross demand can be reconstructed algebraically from incomplete components but is not independently measured/validated and is not an approved forecasting target.

## D. Completed stages and verified status

| Stage | Status | Commit / evidence | Readiness |
|---|---|---|---|
| Honda H1 reduced-data audit | COMPLETED_AND_COMMITTED | `80b8e303...` | validated forensic baseline |
| H2 provenance reconstruction | COMPLETED_AND_COMMITTED | `2d29de9...` | observation-aware SILVER |
| H3 protocol | COMPLETED_AND_COMMITTED | H3A `99bb203...`; H3B `b12d055...` | frozen |
| H4 historical baselines | COMPLETED_AND_COMMITTED | `11a490b...` | frozen B5 |
| H5 deterministic ML | COMPLETED_AND_COMMITTED | `b3764bd...` | frozen M1-CHP |
| H6 zero-shot foundation models | COMPLETED_AND_COMMITTED | `24e86e9...` | completed negative benchmark |
| Stage 1 / H7 scenarios | COMPLETED_AND_COMMITTED | `bb0536df...` | READY_WITH_LIMITATIONS |
| Stage 2 flexibility framework | COMPLETED_AND_COMMITTED | `aff0c933...` | READY_WITH_LIMITATIONS_FOR_STAGE_3 |
| Stage 3 decision value | COMPLETED_AND_COMMITTED | `be5ddb47...` | Category C exploratory; Honda economics blocked |

Current tests: SPARK 104/104 pass. Honda 212/213 pass. The single failure is a stale Stage 2 path-protection test that allows only Stage 2 directories after the Stage 1 commit and therefore rejects the legitimately committed Stage 3 paths. No scientific artifact mismatch was identified, but the suite is not currently fully green.

## E. Major numerical findings

### E1. SPARK data quality

Raw measurements are sampled every 5 seconds. Valid raw-sample availability declines from 78.77% (2021) and 79.64% (2022) to 71.67% (2023) and 68.31% (2024). Longest missing runs are approximately 1.58 hours, 24.89 hours, 13.70 days and 29.68 days, respectively.

An empirical reconstruction study supported a 75% within-interval coverage threshold: on 3,600 test intervals, reconstructed 15-minute mean error was MAE 0.0060 kW, P95 0.0212 and P99 0.0495. At roughly 70% coverage, MAE rose to 0.0416, P95 to 0.1364 and P99 to 0.2687 kW. This is a site/archive-specific threshold, not a universal rule.

GOLD includes 133,847 targets: 34,948 (2021), 34,884 (2022), 32,676 (2023), 31,339 (2024). Evidence: SPARK `reports/raw_quality/*.json`, `reports/missingness/coverage_threshold_decision.csv`, `reports/gold_manifest.json`.

### E2. SPARK deterministic forecasting

On 2024:

| Method | MAE kW | RMSE kW | Coverage |
|---|---:|---:|---:|
| B3 | 2.7096 | 4.2319 | 100% |
| B5 | 2.5713 | 4.1518 | 96.19% |
| LightGBM | 2.3520 | 3.4659 | 100% |

LightGBM minus B3 paired daily MAE: −0.3537 kW, 95% CI [−0.5089, −0.1958]. On B5-common targets: −0.2171 kW, [−0.3773, −0.0657]. Holiday LightGBM MAE is 3.4636 kW versus 8.2790/9.6727 for B3/B5. Chronology is 2021–2022 train, 2023 validation, 2024 test. Evidence: `results/baselines/baseline_metrics.csv`, `results/lightgbm/lightgbm_metrics.csv`.

### E3. SPARK probabilistic forecasting

Seven quantiles (q05/q10/q25/q50/q75/q90/q95) are trained and validation-calibrated. The selected global calibration R0 achieves 2024 WIS 1.4743, 50% coverage 50.06%, 80% 80.44% and 90% 89.91%. Interval widths are 3.971, 8.030 and 10.950 kW. Holiday performance is poor: WIS 2.1746, 80% coverage 50.35% and 90% 68.29%.

Development WIS differs minimally between R0 global (1.3924) and R4 hierarchical (1.3910). R4's 2024 diagnostic WIS is 1.4646, but this cannot retroactively select it. Broad pre-issuance regime variables do not equal knowledge of tomorrow's realized operating state. These results validate aggregate marginal uncertainty, not coherent full-day joint distributions.

### E4. Original SPARK flexibility simulation

On 274 complete 2024 days, with 20% assumed shiftable load, ±2 h movement, daily conservation, original-day peak cap and ex-post historical day-ahead prices:

| Quantity | €/day |
|---|---:|
| No-flex wholesale cost | 28.3816 |
| Oracle savings | 0.12451 |
| B3 savings | 0.08366 |
| Deterministic LightGBM savings | 0.08972 |
| Marginal q10/q50/q90 policy savings | 0.09107 |
| Probabilistic increment | 0.00135, CI [−0.00007, +0.00273] |

Physical sensitivity is much larger. At 20% flexibility, ±1/±2/±4 h produces €0.0211/€0.0911/€0.3271 per day. Peak caps 1.0×/1.1×/1.25× produce €0.0911/€1.1739/€1.3963. Thus probabilistic incremental value is not robust, and assumed physics dominates this simulation.

### E5. Honda provenance

Primary eligibility is exactly `OBSERVED_ORIGINAL`. Corrected non-imputed labels are sensitivity-only; imputed/filled, unknown and unresolved are rejected.

| Year | Primary intervals | Eligibility | Complete days | Partial / zero days |
|---:|---:|---:|---:|---:|
| 2020 | 24,145 | 68.72% | 187 | 74 / 105 |
| 2021 | 34,281 | 97.83% | 342 | 23 / 0 |
| 2022 | 34,624 | 98.81% | 356 | 6 / 3 |
| 2023 | 34,161 | 97.49% | 347 | 13 / 5 |

The local generated SILVER and mask hashes match H2: `d9d934...10df` and `f93d5b...b778`. 2018–2019 do not contain primary original grid targets under the frozen policy.

### E6. Honda B5, LightGBM and zero-shot foundation models

| Model | MAE kW | RMSE kW | Bias kW | Export recall |
|---|---:|---:|---:|---:|
| B5 | 76.480 | 101.338 | +6.649 | 30.809% |
| LightGBM M1-CHP | 71.513 | 96.974 | +9.408 | 29.752% |
| TimesFM 3.0 zero-shot | 95.431 | 129.221 | −0.040 | 11.219% |
| Chronos-2 zero-shot | 103.672 | 139.959 | −6.184 | 11.543% |

All cover 34,161 eligible targets and 347 complete days. LightGBM improves B5 by 4.967 kW (6.495%); daily difference is −5.122 kW with CI [−6.833, −3.514]. The selected model has 80 features and 196 boosting iterations. Development ablations are: M0 calendar 80.583 kW, calendar+analogues 68.253, +recent state 68.247, +volatility/export summaries 67.834, M1-PV 68.616, M1-CHP 66.530 and M1-ALL 68.112. Weather/thermal inputs were inadmissible under the primary provenance masks.

TimesFM minus LightGBM daily MAE is +22.969 kW, CI [19.023, 27.464]. Chronos minus LightGBM is +31.580, [26.291, 37.161]. Both are official zero-shot checkpoints run locally without fine-tuning. The result does not establish the value of domain adaptation or all foundation models.

The crucial conditional result is that lower MAE does not improve export recall. 2023 is 86.43% import and 13.57% export; a model can improve majority-state continuous errors while remaining weak around zero and export events.

### E7. Stage 1 coherent scenarios

Residuals are actual minus forecast. Honda uses complete out-of-sample fold residuals; SPARK uses expanding past-only 2023 validation residuals with at least 60 donor days. Test residuals are not sampled. S1 samples intervals independently; S2 samples whole-day exact-length trajectories; S3 conditions on future-known season/day type and falls back when donors are insufficient.

At 50 scenarios on Honda 2022, S1 versus S2 energy score is 64.352 versus 63.317 kW, variogram 25.040 versus 8.256 and ACF RMSE .498 versus .211. On rolling SPARK 2023 it is 2.394 versus 2.371, 1.238 versus .402 and .538 versus .185. S2 strongly improves dependence but marginal 80/90% coverage remains only 75.21/85.56% Honda and 69.74/81.29% SPARK. S2 is frozen because S3's added conditional complexity did not pass a robust across-dataset selection rule.

Previously exposed Honda 2023 S2 export Brier score is 0.0834; this is diagnostic and no event threshold was test-tuned. Readiness is `READY_WITH_LIMITATIONS`, not operational calibration.

### E8. Stage 2 physical/market framework

Primary F1 is a simulated deferrable process:

`G_t = baseline_t + u_t`

`x_{t+1} = x_t + u_t Δt`, with `Δt=0.25 h`

`−P_down ≤ u_t ≤ P_up`

`−E_backlog ≤ x_t ≤ E_advance`

and declared initial/terminal states. Positive state means advanced work; negative means deferred work. An exact recovery deadline uses flow arcs with maximum temporal separation; the simple energy-state capacity is not falsely called a duration constraint.

F2 is a synthetic thermal buffer:

`E_{t+1} = retention E_t + (COP P_el,t − Q_demand,t) Δt`.

F3 adds import/export caps and peak shadow costs. A shared first-stage schedule enforces non-anticipativity. CVaR uses standard upper-tail auxiliaries. SciPy 1.17.1 HiGHS solves the LPs; synthetic fixtures validate flat/negative price cases, two-interval optima, cap infeasibility, exact recovery, thermal balance, 92/96/100-day lengths, no artificial simultaneous import/export and identical-scenario equivalence.

These are counterfactual capacities: SPARK ±2/4, ±5/10, ±10/20 kW/kWh; Honda ±25/50, ±50/100, ±100/200. They are not measured equipment. F2 is synthetic; BESS is interface-only. Peak penalties are daily shadow prices, not actual demand charges.

SPARK prices are hourly German day-ahead values repeated across four 15-minute intervals. Their file lacks publication timestamps, so classification is `EX_POST_PRICE_AWARE`. Honda has no accepted 2023 price/timing file and is `UNSUPPORTED_TIMING`.

### E9. Stage 3 decision value

Stage 3 freezes an exploratory comparison before 2024 settlement and uses 273 common 96-interval days (6 January–1 December 2024). The primary hypothetical SPARK F1 asset is ±5 kW and ±10 kWh. Prices remain ex-post known.

| Policy | Cost €/day | Savings €/day | Realized zero-export violation days |
|---|---:|---:|---:|
| No flexibility | 28.447594 | 0 | 0 |
| Deterministic LightGBM | 26.447976 | 1.999617 | 82 |
| S1 independent scenarios | 26.646923 | 1.800671 | 16 |
| S2 coherent scenarios | 26.700058 | 1.747536 | 11 |
| S2 CVaR | 26.700058 | 1.747536 | 11 |
| Oracle | 26.448777 | 1.998817 | 0 |

Primary paired cost differences define negative as favoring the first policy:

- Deterministic minus coherent: −€0.252081/day, CI [−0.315669, −0.199605].
- Independent minus coherent: −€0.053135/day, CI [−0.109641, −0.011827].

Coherent wins 61.5% of individual days over independent but has worse mean due to adverse outliers. A 7-day block bootstrap reaches the same direction. Physical capacity changes coherent savings from €0.740766 (LOW) to €1.747536 (MEDIUM) to €3.057647/day (HIGH). Import caps 40/30/20 kW are feasible on 120/31/0 days. CVaR changes schedules but not realized historical costs/tails.

The deterministic cost being slightly below the oracle is not a discovery: deterministic schedules violate the zero-export cap on 82 days, making all-day regret operationally uninterpretable. Common-feasible-day secondary comparisons still favor deterministic/independent policies. No clipping or reoptimization occurs at settlement.

Honda economics are **NOT EVALUABLE** because authoritative price/timing data are absent. No two-site economic result exists.

## F. Strongest supported conclusions

1. **Provenance-aware target construction is scientifically consequential.** Honda's signed exchange can be forecast without silently treating corrected/fill values as observations or gross demand.
2. **Site-specific deterministic learning outperforms strong historical baselines, but conditional weaknesses remain.** Aggregate Honda MAE improves robustly while export recall does not.
3. **Zero-shot transfer is not competitive in this Honda implementation.** This is a bounded result about two checkpoints/modes, not a universal rejection of foundation models.
4. **Whole-day residual sampling improves temporal dependence representation.** This is statistically supported on development data.
5. **Better dependence does not automatically improve decisions.** Under the frozen SPARK objective, S2 changes schedules but increases realized mean cost.
6. **Physical assumptions dominate the tested economic effects.** Capacity and peak constraints have much larger effects than marginal/coherent forecast changes.

## G. Negative and inconclusive findings

- Regime calibration does not solve holiday uncertainty.
- Original marginal-quantile scheduling has an economically tiny, statistically inconclusive advantage.
- Honda LightGBM does not improve export recall.
- Chronos-2 and TimesFM do not beat B5 or LightGBM zero-shot.
- S2 is dependence-aware but not fully marginally calibrated.
- Coherent and CVaR policies do not improve realized SPARK economics in Stage 3.
- Honda economic export value and two-site decision heterogeneity are not tested.
- PV/CHP changes as the cause of seasonal errors are not established.

## H. Physical and economic assumptions

The flexibility resource is a virtual electrical-work buffer, not a named machine. Honda would be an additive behind-the-meter demand overlay on already-netted grid exchange; PV/CHP must not be subtracted again. Import/export accounting is piecewise linear with tariff-dominance checks. Prices are converted from EUR/MWh to EUR/kWh. Reported savings are wholesale-cost counterfactuals, not electricity bills, observed profits or verified customer value. F2 thermal assumptions and all F1 capacities are synthetic.

## I. Market-timing constraints

Forecast issuance is causally frozen at 12:00 (SPARK) or 13:00 (Honda) D−1. A later decision may use prices only if publication is established and may not refresh the forecast with post-issuance measurements. Historical SPARK price values prove realized prices, not when those values became available. Accordingly, the decision study is ex-post price-aware. Honda timing is unsupported because no accepted price series exists.

## J. Source and evaluation limitations

1. SPARK 2024 and Honda 2023 had been examined before later scenario/decision hypotheses; later results are exploratory extensions, not pristine confirmation.
2. Only one dataset has economic settlement.
3. No real production schedule, asset capacity, connection contract or tariff is identified.
4. Realized grid-cap violations undermine some all-day economic comparisons.
5. Scenario donor pools are finite and rare DST lengths are under-supported.
6. Aggregate calibration masks holiday and tail undercoverage.
7. Two public sites cannot establish general industrial external validity.
8. The present Honda test suite has one stale protection-test failure.

## K. Preliminary hypothesis matrix

| Hypothesis | Verdict | Basis |
|---|---|---|
| Site-specific LightGBM improves aggregate accuracy | SUPPORTED | paired CIs below zero on both site studies |
| Better aggregate MAE improves Honda export recognition | NOT_SUPPORTED | LGBM recall 29.752% vs B5 30.809% |
| Zero-shot foundation models are superior | NOT_SUPPORTED for tested models | large positive differences versus LGBM |
| SPARK marginal intervals are calibrated overall | SUPPORTED | nominal 80/90 coverage closely matched |
| SPARK holiday uncertainty is calibrated | NOT_SUPPORTED | severe holiday undercoverage |
| Whole-day residual sampling improves dependence | SUPPORTED | variogram/ACF and energy-score evidence |
| S2 is operationally calibrated uncertainty | NOT_SUPPORTED | marginal/coverage/donor limitations |
| Probabilistic scheduling adds value in original model | INCONCLUSIVE | €0.00135/day CI crosses zero |
| Coherence adds realized decision value | NOT_SUPPORTED | Stage 3 paired CIs favor deterministic/S1 |
| Physical assumptions matter more | PARTIALLY SUPPORTED | large sensitivity range, but counterfactual grid |
| CVaR reduces realized tail risk | NOT SUPPORTED | no change in realized tail metrics |
| Honda export uncertainty has economic value | NOT TESTED | absent price/timing input |

## L. Potential novelty claims requiring external verification

Do not accept these as novel without literature search:

1. An empirically validated raw-coverage threshold for industrial 15-minute aggregation.
2. Original-observation-provenance forecasting of signed industrial grid exchange.
3. A fair site-specific LightGBM versus modern zero-shot foundation-model comparison.
4. A negative result in conditional regime calibration under structural shifts.
5. Leakage-safe whole-day residual trajectories across positive demand and signed exchange.
6. The integration of coherent scenarios with a shared non-anticipative virtual-process schedule.
7. Empirical falsification of the assumption that better temporal dependence automatically improves economic decisions.
8. Quantification showing physical-capacity assumptions dominate forecast-policy differences.

Most underlying methods—gradient boosting, conformal/quantile calibration, residual bootstrap, energy/variogram scores, virtual storage buffers, LP, CVaR—are established. Potential contribution lies in careful industrial integration, provenance and falsification, not algorithmic novelty.

## M. Most serious reviewer objections

1. **The asset is hypothetical.** Savings may be an artifact of unmeasured power/energy bounds.
2. **Prices were not proven available at decision time.** Ex-post scheduling is not operational validation.
3. **The years were already exposed.** Later hypotheses are not independently confirmatory.
4. **Realized cap violations occur.** Some apparently good deterministic schedules are infeasible.
5. **Only SPARK has economics.** Cross-site decision claims cannot be made.
6. **Scenario calibration is incomplete.** Dependence quality is not full distributional validity.
7. **The objective may structurally suppress the value of dependence.** A linear risk-neutral expected-cost LP primarily depends on marginal expectations; coherence matters through nonseparable caps, peaks and risk.
8. **No actual tariff or production constraints exist.** Results are engineering counterfactuals.
9. **Multiple exploratory sensitivities are not confirmation.** Holm correction covers only two primary comparisons.
10. **The current suite has one regression.** Documentation and code-protection expectations are inconsistent after Stage 3.

## N. High-priority questions for independent research

For each question, please identify the best peer-reviewed evidence (preferably 2022–2026), any older foundational paper, and implications for this project.

1. **When does temporal scenario dependence affect a day-ahead LP?** We observe improved variogram scores but worse costs. Determine sufficient structural conditions—state coupling, joint caps, peaks, nonlinear tariffs, CVaR—for dependence to change an optimum.
2. **Are energy/variogram scores sufficient for decision readiness?** We have strong dependence gains but imperfect coverage. Identify decision-weighted validation alternatives.
3. **What is the best industrial deferrable-process abstraction without production orders?** Compare virtual storage, task/job-shop, inventory and age-structured backlog formulations; define minimum credibility criteria.
4. **How should recovery deadlines be modelled?** Stage 2 uses flow arcs; assess whether this is standard and what constraints prevent unrealistic pre-service or rebound.
5. **How can asset parameters be identified ethically and minimally?** We need a practical measurement protocol for power, energy, ramps and recovery without claiming proprietary production detail.
6. **What day-ahead German market timestamps were operationally available in 2024?** Verify gate closure, coupled-results publication, product resolution and archival sources with publication metadata.
7. **How should industrial wholesale exposure differ from retail tariffs?** Identify defensible tariff cases, export remuneration and billing-period peak models.
8. **Can signed-exchange export probabilities create decision value?** Honda deterministic recall is ~30%. Review event-aware, cost-sensitive and probabilistic sign-crossing models and when they improve scheduling.
9. **Would explicit gross demand, PV and CHP forecasts be scientifically preferable?** Determine when component forecasting improves net-exchange decisions and what data are necessary to avoid double counting.
10. **Why do zero-shot foundation models underperform site-specific boosting?** Review industrial/small-data evidence, context adaptation, covariates, pretraining frequency and domain shift; distinguish checkpoint limitations from family limitations.
11. **Is foundation-model fine-tuning worth testing?** We need an evidence-based stop/go rule given compute, limited years and a strong LightGBM baseline.
12. **How should holiday uncertainty be calibrated with very small samples?** Evaluate hierarchical pooling, event transfer, Bayesian shrinkage and whether the subgroup is too small for separate claims.
13. **How can Stage 3 be independently confirmed?** Recommend the smallest prospective holdout design or third site that separates model selection from evaluation.
14. **Are block bootstrap and 273 days adequate?** Evaluate inference for paired daily costs, heavy tails, serial dependence and tail/CVaR claims.
15. **What makes decision-focused forecasting scientifically justified here?** Determine whether end-to-end decision loss is warranted before real physics/tariffs are identified.
16. **Is a battery comparator helpful or distracting?** Battery physics are measurable but represent a different intervention. Assess its value for mechanism identification versus scope inflation.
17. **Can the negative coherence result itself be a publication contribution?** Find studies where better probabilistic scores fail to improve downstream decisions and evaluate novelty.
18. **Which parts should be removed from an MSc thesis?** Rank the minimum coherent story versus optional material that dilutes the contribution.

## O. Candidate next directions, pending your review

- **Highest scientific value:** obtain real asset/production constraints, actual tariff/connection data and an untouched evaluation period.
- **Highest cross-site value:** acquire authoritative Honda DE-LU prices with publication timing, then apply the already frozen decision protocol without retuning.
- **Focused modelling option:** export-event probability and cost-sensitive sign-crossing forecasting, only if tied to a verified decision.
- **Avoid by default:** more zero-shot checkpoint hunting, broad hyperparameter tuning, increasingly granular calibration regimes, or optional battery/thermal scope without new data.
- **Immediate action:** write the thesis around evidence, limitations and the negative decision-value result while conducting literature review.

## P. Requested literature review

Please:

1. Independently verify the scientific novelty of the work.
2. Find directly relevant peer-reviewed literature, prioritizing 2022–2026 while including foundational sources.
3. Verify every citation, DOI, publication status and—where relevant—license.
4. Challenge the physical flexibility assumptions and propose minimum data for validation.
5. Challenge the German market-timing assumptions with authoritative operator/regulatory sources.
6. Evaluate mathematically when scenario coherence is economically necessary.
7. Evaluate whether improved Honda export probabilities could plausibly improve decisions.
8. Compare virtual buffers, task scheduling, thermal storage and BESS as flexibility representations.
9. Identify where literature already solves or invalidates the proposed questions.
10. Distinguish standard methods, applied engineering contributions and potentially novel empirical findings.
11. Rank the remaining research directions by scientific value, feasibility and thesis relevance.
12. Decide whether another experiment is necessary.
13. If yes, specify the **smallest scientifically valuable next experiment**, including data, design, hypotheses and stopping rule.
14. Identify work that should stop or be removed from the thesis.
15. Propose a realistic MSc thesis structure.
16. Assess publication prospects without guaranteeing publication.
17. Provide a critical decision matrix.
18. Give a final evidence-grounded recommendation; do not automatically approve the prior roadmap.

## Q. Required Perplexity response format

Please return:

1. **Executive verdict** (maximum 500 words).
2. **Verified literature table** with claim, source, year, DOI/URL, peer-review status and relevance.
3. **Novelty matrix** for each candidate contribution: standard / applied / distinctive empirical / methodological / unsupported.
4. **Physical-model critique** with specific corrections and minimum validation data.
5. **Market-timing critique** with authoritative German/European evidence.
6. **Statistical critique** covering holdout exposure, scenario validation, bootstrap and multiplicity.
7. **Decision-value mechanism analysis** explaining when temporal dependence should and should not matter.
8. **Research-direction ranking** with expected scientific value, cost, risks and stop/go recommendation.
9. **Smallest next experiment** or an explicit recommendation to run none.
10. **Thesis scope**: proposed title, chapters, results to emphasize, results to demote/remove.
11. **Publication assessment**: plausible venues/areas, minimum missing evidence and likely reviewer objections.
12. **Final decision matrix**: CONTINUE / DEFER / STOP / REDESIGN for forecasting, scenarios, export modelling, physics, markets, optimization and writing.

Clearly separate established literature, your inference and recommendations. Do not infer real Honda/SPARK operations from the hypothetical optimization. Do not call the simulated savings actual customer savings.

## R. Repository evidence map

- SPARK reports: `reports/missingness_reliability_study.md`, `gold_summary.md`, `day_ahead_baselines.md`, `deterministic_lightgbm.md`, `probabilistic_lightgbm.md`, `regime_aware_probabilistic.md`, `flexibility_optimization.md`.
- Honda H1–H6: `reports/honda_reduced_dataset_audit.md`, `honda_provenance_and_silver.md`, `honda_forecasting_protocol.md`, `honda_baseline_benchmark.md`, `honda_deterministic_ml_benchmark.md`, `honda_foundation_model_benchmark.md`.
- Scenarios: `reports/scenarios/honda_h7_coherent_scenario_study.md`; frozen `config/scenarios/scenario_method_frozen.json`.
- Optimization: `reports/optimization/honda_stage2_flexibility_and_market_framework.md`; `config/optimization/`.
- Decision value: `reports/decision_value/industrial_flexibility_decision_value_study.md`; frozen `config/decision_value/stage3_experiment_frozen.json` and amendment.
- Audit companion: `reports/research_checkpoint/complete_research_status.md`.

