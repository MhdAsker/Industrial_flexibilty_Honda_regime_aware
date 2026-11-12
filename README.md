# Industrial Flexibility under Forecast Uncertainty

> A provenance-aware, leakage-safe research framework connecting day-ahead industrial-energy forecasting, temporally coherent uncertainty scenarios, physically constrained flexibility scheduling, and decision-value evaluation.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Research status](https://img.shields.io/badge/status-Category%20C%20exploratory-f59e0b)](#research-status)
[![Forecast horizon](https://img.shields.io/badge/horizon-day--ahead%20%7C%2015%20min-6f42c1)](config/honda_forecasting_protocol_frozen.json)
[![Honda data DOI](https://img.shields.io/badge/Honda%20data-10.5061%2Fdryad.73n5tb363-blue)](https://doi.org/10.5061/dryad.73n5tb363)

## Executive summary

This repository studies a question that is often skipped in industrial-energy forecasting:

> **Does a statistically better representation of forecast uncertainty produce a better industrial flexibility decision?**

The work builds an auditable chain from raw-data semantics to downstream scheduling:

1. forensic source and timestamp audit;
2. observation-level provenance and quality masks;
3. a frozen, leakage-safe day-ahead forecasting protocol;
4. transparent historical baselines;
5. site-specific deterministic LightGBM;
6. zero-shot Chronos-2 and TimesFM benchmarks;
7. temporally coherent daily scenario generation;
8. physically constrained flexibility optimization; and
9. a pre-specified decision-value comparison with a feasibility-restricted replication.

The forecasting contribution is strong: on the Honda 2023 test period, site-specific LightGBM improves on a competitive historical ensemble while preserving complete prediction coverage. Whole-day residual trajectories also represent temporal dependence substantially better than independent interval sampling.

The downstream result is deliberately negative and scientifically useful. In the SPARK counterfactual scheduling experiment, coherent scenarios substantially reduce modeled zero-export-cap violations, but they do **not** lower mean wholesale cost relative to deterministic scheduling. Better temporal dependence is therefore not automatically equivalent to better decision value.

All monetary outcomes are **synthetic counterfactual wholesale-cost differences**. They are not customer savings, trading profit, battery value, or validation of Honda or SPARK plant operation.

## Research status

| Dimension | Current status |
|---|---|
| Honda data provenance and target definition | Complete |
| Leakage-safe deterministic forecasting | Complete |
| Zero-shot foundation-model benchmark | Complete |
| Coherent daily scenario generation | Complete, with calibration limitations |
| Physical optimization framework | Complete as a synthetic counterfactual framework |
| SPARK decision-value study | Complete |
| Feasibility-restricted replication | Complete |
| Honda economic evaluation | Not evaluable with currently accepted price/timing evidence |
| Thesis drafting | Ready with explicit limitations |
| Exploratory paper preparation | Ready for literature verification and review |
| Operational deployment or savings claim | Not ready |

## Phase 2 shadow-scheduler foundation

The repository now includes a human-in-the-loop Phase 2 foundation for 48–72 hour rolling recommendations. It provides a strict partner-readiness gate, immutable issue-time snapshots, scenario-tree nonanticipative recourse, a deterministic fallback, operator approval/rejection/modification records, realized-operation settlement, and prospective cohort metrics. The partner-adaptive layer adds hash-linked online bias/conformal states, causal multi-day residual blocks, information-time scenario trees, transition and operator models, inverse preference estimation, value-of-information ablations, and a fail-closed prospective exit evaluator.

This does **not** mean Phase 2 is industrially validated. No current Honda or SPARK artifact satisfies the required partner-approval contract, and the software refuses to issue a prospective recommendation until all required evidence is approved and digested. Only the deferrable-process asset is currently implemented for Phase 2 claims; unsupported assets are rejected.

```powershell
shadow-scheduler readiness --partner path\to\partner-readiness.yaml
shadow-scheduler snapshot --issue path\to\issue.yaml --partner path\to\partner-readiness.yaml --output-root artifacts\shadow
shadow-scheduler recommend --snapshot artifacts\shadow\PARTNER\ISSUE
shadow-scheduler metrics --root artifacts\shadow
shadow-scheduler protocol --protocol examples\shadow\phase2_prospective_protocol.template.json
```

See [`docs/phase2_shadow_scheduler.md`](docs/phase2_shadow_scheduler.md) for the data contracts, optimization semantics, operating workflow, metrics, and remaining industrial work.

## Phase 3 multi-site platform foundation

An approval-gated `honda.platform` layer now provides versioned multi-site registries, leakage-safe connector snapshots, workflow state persistence, role- and site-scoped API credentials, a hash-chained audit log, monitoring gates, champion/rollback model aliases, portfolio allocation under network and risk limits, and authenticated operational API routes. The adaptive-ML layer adds an interval-level prediction/label ledger, identical-snapshot model comparisons, rolling-origin and semantic-safe cold-start evaluation, partial-pooling adapters, joint empirical-copula scenarios, and fail-closed global promotion gates.

Every tenant starts in `OFFLINE_SIMULATION`. The platform cannot authorize control from offline or prospective-shadow stages, and later stages require sequential evidence-backed promotion. Automated action additionally requires a separate safety-case digest and two-person approval. No current site has met these gates.

```powershell
flexops --database artifacts\platform\flexops.sqlite init
flexops --database artifacts\platform\flexops.sqlite verify-audit
flexops --database artifacts\platform\flexops.sqlite ledger-status --tenant TENANT
flexops --database artifacts\platform\flexops.sqlite ml-readiness --approved-sites 3 --seasonal-days 365 --pooled-rows 100000
```

See [`docs/phase3_multisite_platform.md`](docs/phase3_multisite_platform.md) for capabilities, threat boundaries, API routes, remaining production gaps and the recommended deployment sequence.

## Offline prescriptive ML challengers

The repository now includes exploratory regime-aware forecasting, quantile LightGBM, grouped conformal calibration, conditional ECC scenarios, scenario-reduction diagnostics, and an execute-first rolling-MPC evaluation contract.

On the already-exposed Honda 2023 period, the development-selected regime mixture reduced MAE from 71.513 to 70.447 kW and improved export recall from 29.75% to 33.03%, but missed the frozen promotion gates. Raw conditional ECC-200 improved WIS and dependence diagnostics while reaching approximately 89% coverage. Development-only selection chose bandwidth 2.0 and retained all 200 paths because reductions to 25 and 50 paths damaged calibration and score gains. All challengers remain `DO_NOT_PROMOTE` or development-only candidates.

See [`docs/prescriptive_ml_phase1.md`](docs/prescriptive_ml_phase1.md) for the protocol, quantitative results, claim boundary and next experiments.

The appropriate classification is:

> **READY_WITH_LIMITATIONS — CATEGORY_C_EXPLORATORY**

The software and methodological controls support a rigorous exploratory thesis or paper. They do not support an operational plant-performance claim, a customer-bill claim, or a two-site economic generalization.

## Research questions

1. **Target validity:** What does the released Honda signal physically represent, and which observations are valid original measurements?
2. **Forecast validity:** How should day-ahead performance be evaluated without future-information leakage, DST errors, target-quality mixing, or incomplete prediction coverage?
3. **Model value:** Do site-specific ML models or zero-shot foundation models outperform strong transparent historical baselines?
4. **Scenario value:** Does whole-day residual sampling preserve temporal dependence better than independent marginal sampling?
5. **Decision value:** Does improved scenario dependence reduce cost or improve feasibility under power, energy-state, recovery, and grid constraints?

## Datasets and physical semantics

### Honda R&D Europe

The primary Honda target is `grid_exchange_kw_signed`: physical net electrical exchange with the public grid at 15-minute resolution.

- Positive values are grid import.
- Negative values are grid export.
- The unit is kW.
- Behind-the-meter PV, CHP, and other site behavior are already reflected in the net exchange.

This signal must not be described as gross industrial demand. Gross reconstructed load was not validated as an independent measurement and is not used as the forecasting target.

The source is the Honda Research Institute Europe smart-building release on Dryad: [10.5061/dryad.73n5tb363](https://doi.org/10.5061/dryad.73n5tb363).

### KIT SPARK

SPARK supplies a physically different target: positive facility electrical demand. It is used as the economic evaluation site because the accepted Stage 3 inputs include a historical German price series. The price artifact does not contain observation-level publication timestamps, so the study assumes post-publication scheduling instead of claiming independently verified real-time availability.

Honda and SPARK raw kW errors are not directly comparable because their sites, scales, and target semantics differ.

## Observation-level provenance

Honda values are separated into explicit provenance classes:

| Class | Meaning | Primary evaluation? |
|---|---|---:|
| `OBSERVED_ORIGINAL` | Original observation accepted by the provenance policy | Yes |
| `CORRECTED_NONIMPUTED` | Corrected but non-imputed value | Sensitivity only |
| `IMPUTED_OR_FILLED` | Filled or reconstructed value | No |
| `UNKNOWN_PROVENANCE` | Provenance cannot be established | No |
| `MISSING_UNRESOLVED` | No usable value | No |

This prevents corrected or filled values from being silently treated as original ground truth.

| Year | Eligible original intervals | Complete primary days |
|---:|---:|---:|
| 2020 | 24,145 | 187 |
| 2021 | 34,281 | 342 |
| 2022 | 34,624 | 356 |
| 2023 | 34,161 | 347 |

## Frozen day-ahead protocol

| Element | Definition |
|---|---|
| Target | Signed public-grid exchange in kW |
| Forecast issue | 13:00 Europe/Berlin on D−1 |
| Horizon | Every physical 15-minute interval of local day D |
| Historical cutoff | `interval_end_utc <= forecast_issue_time_utc` |
| Primary label quality | `OBSERVED_ORIGINAL` only |
| Sensitivity labels | Original plus corrected non-imputed |
| Development folds | 2020→2021 and 2020–2021→2022 |
| Final training | 2020–2022 |
| Protected test | 2023 |
| Primary metric | MAE with explicit coverage |
| Statistical unit | Whole local calendar day |
| Bootstrap | 2,000 resamples; seed `20261029` |

UTC is the physical key; Europe/Berlin defines operational calendar semantics. The implementation supports 92-, 96-, and 100-interval local days rather than assuming every day contains 96 intervals. Future realized weather, PV, CHP, thermal values, and targets are forbidden inputs.

See [`config/honda_forecasting_protocol_frozen.json`](config/honda_forecasting_protocol_frozen.json) and [`reports/honda_forecasting_protocol.md`](reports/honda_forecasting_protocol.md).

## Research stages

| Stage | Purpose | Status | Main artifact |
|---|---|---|---|
| H1 | Source and semantic audit | Complete | [`honda_reduced_dataset_audit.md`](reports/honda_reduced_dataset_audit.md) |
| H2 | Provenance-aware SILVER layer | Complete | [`honda_provenance_and_silver.md`](reports/honda_provenance_and_silver.md) |
| H3A | Measurement and forecasting feasibility | Complete | [`honda_forecasting_protocol_h3a.md`](reports/honda_forecasting_protocol_h3a.md) |
| H3B | Frozen operational protocol | Complete | [`honda_forecasting_protocol.md`](reports/honda_forecasting_protocol.md) |
| H4 | Transparent baseline benchmark | Complete | [`honda_baseline_benchmark.md`](reports/honda_baseline_benchmark.md) |
| H5 | Deterministic LightGBM benchmark | Complete | [`honda_deterministic_ml_benchmark.md`](reports/honda_deterministic_ml_benchmark.md) |
| H6 | Zero-shot foundation-model benchmark | Complete | [`honda_foundation_model_benchmark.md`](reports/honda_foundation_model_benchmark.md) |
| H7 / Stage 1 | Coherent daily scenarios | Ready with limitations | [`honda_h7_coherent_scenario_study.md`](reports/scenarios/honda_h7_coherent_scenario_study.md) |
| Stage 2 | Flexibility and market framework | Ready with limitations | [`honda_stage2_flexibility_and_market_framework.md`](reports/optimization/honda_stage2_flexibility_and_market_framework.md) |
| Stage 3 | Decision-value study | Category C exploratory | [`industrial_flexibility_decision_value_study.md`](reports/decision_value/industrial_flexibility_decision_value_study.md) |
| Replication | Feasibility-restricted comparison | Category C exploratory | [`industrial_flexibility_decision_value_replication.md`](reports/decision_value/industrial_flexibility_decision_value_replication.md) |

## Forecasting methods

### Transparent baselines

- **B0:** most recent provenance-safe observation at issuance;
- **B1:** most recent safe matching local quarter-hour from a prior day;
- **B2:** matching local quarter-hour seven calendar days earlier;
- **B3:** robust median of selected previous weekly analogues;
- **B4:** training-only calendar climatology with a frozen fallback hierarchy; and
- **B5:** a fixed ensemble selected only on development data.

Analogue lookup uses local-calendar meaning and UTC physical keys. It does not use a fixed `t−96` shortcut across DST transitions.

### Site-specific LightGBM

- **M0:** calendar features and issuance-safe grid-history summaries;
- **M1:** M0 plus allowed historical energy context available before issuance.

The selected configuration is M1 with historical CHP context. Missing historical values remain missing rather than being replaced with invented measurements. Selection uses development folds only.

### Zero-shot foundation models

The benchmark evaluates Amazon Chronos-2 and Google TimesFM 3.0 without Honda-specific gradients, fine-tuning, adapters, LoRA, or prompt optimization. Isolated environments prevent their neural dependencies from silently altering the H1–H5 stack. Results apply to the frozen checkpoints and inference modes, not to every foundation model.

## Honda forecasting results

All headline values use the same 34,161 eligible 2023 observations with complete prediction coverage.

| Method | Adaptation | MAE (kW) | RMSE (kW) | Bias (kW) | Export recall |
|---|---|---:|---:|---:|---:|
| B5 ensemble | Development-selected rules | 76.480 | 101.338 | +6.649 | 30.81% |
| LightGBM M1-CHP | Site-specific training | **71.513** | **96.974** | +9.408 | 29.75% |
| TimesFM 3.0 C1 | Zero-shot | 95.431 | 129.221 | −0.040 | 11.22% |
| Chronos-2 C1 | Zero-shot | 103.672 | 139.959 | −6.184 | 11.54% |

Key findings:

- LightGBM improves MAE by 4.967 kW, or 6.49%, relative to B5.
- Whole-day bootstrap comparisons support a robust LightGBM improvement.
- Both zero-shot models provide complete coverage but underperform B5 and LightGBM.
- Export recognition remains weak and is especially poor for the zero-shot models.
- The 23–29 hour lead band is the hardest horizon across the evaluated methods.

## Temporally coherent scenarios

Stage 1 compares uncertainty constructions for downstream daily scheduling. The selected `S2_WHOLE_DAY_RESIDUAL_BOOTSTRAP` samples complete historical residual trajectories instead of independently sampling interval residuals. The frozen scenario count is 50 per day.

| Dataset | Diagnostic | Independent S1 | Coherent S2 |
|---|---|---:|---:|
| Honda | Variogram score | 25.0395 | **8.2562** |
| Honda | Residual ACF RMSE | 0.4979 | **0.2109** |
| SPARK | Variogram score | 1.2377 | **0.4021** |

This does not imply fully calibrated operational uncertainty. Limitations include finite donor pools, imperfect marginal and tail coverage, rare DST support, and limited evidence for unusual export regimes.

## Physical flexibility framework

Stage 2 implements auditable optimization contracts for signed exchange, import/export accounting, power and energy-state limits, terminal recovery, grid caps, shared first-stage stochastic schedules, expected cost, and CVaR.

The primary F1 archetype is a virtual production/work buffer, not a claimed battery or identified factory asset. The Stage 3 medium case assumes:

| Parameter | Value |
|---|---:|
| Power range | ±5 kW |
| State range | ±10 kWh |
| Initial / terminal state | 0 kWh |
| Interval duration | 0.25 h |
| Export cap | 0 kW |

These values are experimental assumptions, not measured industrial capacities.

## Decision-value experiment

The original Stage 3 study compares no flexibility, deterministic LightGBM, independent scenarios, and coherent scenarios on 273 complete SPARK 2024 days.

| Policy | Mean synthetic savings (EUR/day) | Export-cap violation days |
|---|---:|---:|
| Deterministic LightGBM | 1.999617 | 82 |
| Independent scenarios | 1.800671 | 16 |
| Coherent scenarios | 1.747536 | 11 |

Physical capacity changes the coherent-policy result more than forecast policy:

| F1 case | Coherent-policy synthetic savings (EUR/day) |
|---|---:|
| Low | 0.740766 |
| Medium | 1.747536 |
| High | 3.057647 |

This indicates that physical assumptions dominate the relatively small forecast-policy effects in the current experiment.

## Feasibility-restricted replication

The replication was frozen before outcome analysis. It does not retrain forecasts, regenerate scenarios, alter capacity, or add policies. Its primary population is the intersection of 191 days on which all three scheduled policies satisfy the zero-export cap.

| Comparison | Mean A−B (EUR/day) | 95% day-bootstrap interval | Interpretation |
|---|---:|---:|---|
| Deterministic − coherent | −0.191191 | [−0.271593, −0.131960] | Deterministic is cheaper |
| Independent − coherent | −0.047240 | [−0.122619, +0.009287] | Mean difference unresolved |
| Deterministic − independent | −0.143951 | [−0.169757, −0.120702] | Deterministic is cheaper |

Coherent scenarios nevertheless reduce realized cap-violation days from 82 under deterministic scheduling and 16 under independent scenarios to 11.

> Coherent scenarios materially improve modeled feasibility, but they do not demonstrate lower mean wholesale cost. Relative to independent scenarios, common-feasible-day cost is inconclusive; relative to deterministic LightGBM, coherent scheduling remains more expensive.

Scheduling is placed at an assumed 12:55 Europe/Berlin on D−1. Because archival publication timestamps are absent, timing is `POST_PUBLICATION_TIMING_ASSUMED`, not independently verified real-time operation.

## Scientific interpretation

Risk-neutral expected energy cost is driven mainly by marginal expectations. Temporal dependence changes decisions only through nonseparable mechanisms such as state coupling, grid caps, demand charges, ramping, recovery, recourse, or risk measures.

Here, coherent scenarios improve temporal structure and modeled feasibility, but that benefit does not offset higher mean cost under the frozen F1 problem. This does not show that probabilistic forecasting is generally useless. It shows that scenario quality must be assessed against the decision mechanism that could reward it.

## Claims and non-claims

### Supported

- Honda signed exchange can be evaluated with explicit provenance and leakage controls.
- Site-specific LightGBM outperforms the frozen baseline on Honda 2023.
- The evaluated zero-shot models underperform the site-specific model.
- Whole-day sampling represents temporal dependence better than independent interval sampling.
- Coherent scenarios reduce modeled zero-export violations in the SPARK counterfactual.
- They do not improve mean cost relative to deterministic scheduling in the frozen experiment.

### Not supported

- Customer savings, total-bill reduction, or trading profit.
- Validation of Honda or SPARK plant flexibility.
- Honda economic value.
- Battery or thermal-asset value.
- Universal superiority or inferiority of coherent scenarios.
- Causal attribution of Honda errors to PV, CHP, or modernization.
- Generalization to other industrial facilities.

## Repository structure

```text
config/                              Frozen protocols and experiments
data/                                Raw/staging placeholders; SILVER is local
docs/                                Source and schema documentation
reports/
  scenarios/                         H7 scenario study
  optimization/                      Stage 2 framework
  decision_value/                    Stage 3 and replication
  research_checkpoint/               Scientific audit and decision brief
  paper_overleaf/                    Paper manuscript source
  progress_overleaf/                 Stakeholder progress report
results/
  audit|provenance|protocol*/         Data and protocol evidence
  baselines|ml|foundation/            Forecasting evidence
  scenarios|optimization/             Stochastic and physical evidence
  decision_value*/                   Economic evaluation evidence
src/honda/
  scenarios|optimization/             Scenario and optimizer code
  decision_value*/                   Decision evaluation and replication
tests/                               Integrity and scientific regression tests
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

The base installation includes the data and optimization stack. The `all` extra adds LightGBM and the test runner; `forecasting` and `test` can instead be installed separately. Chronos-2 and TimesFM continue to use isolated environments recorded under `results/foundation/`.

## FlexLab v2 application

The repository now includes a reusable offline decision-lab layer above the frozen v1 research pipelines. One configuration can validate a site, compare deterministic/independent/coherent policies, settle realized outcomes, and export a hash-verifiable experiment bundle:

```powershell
flexlab validate --experiment examples/flexlab/synthetic/experiment.yaml
flexlab run --experiment examples/flexlab/synthetic/experiment.yaml --output results/flexlab/demo
flexlab reproduce --experiment examples/flexlab/synthetic/experiment.yaml
```

FlexLab v2 includes synthetic, Honda, and SPARK adapters; versioned site/asset/grid/tariff/timing contracts; ordinary and DST validation; calibrated-trajectory utilities; and a read-only dashboard that consumes `dashboard.json`. See [`docs/flexlab_v2.md`](docs/flexlab_v2.md).

Private dashboard deployment: [FlexLab Industrial Flexibility Decision Lab](https://flexlab-industrial-decision-lab.burlyaphid3.chatgpt.site)

Stage 2, Stage 3, and the replication expect the SPARK companion repository as a sibling checkout:

```text
parent/
  industrial_flexibilty_honda/
  Industrial_energy_forecasting/
```

Protected SPARK inputs are checked using frozen commits and file hashes. The integration suite also expects the companion checkout to be clean; unrelated untracked SPARK files can therefore fail external-cleanliness guards without changing computational results.

For a checkout in another location, set the companion path explicitly:

```powershell
$env:SPARK_RESEARCH_ROOT = "D:\research\Industrial_energy_forecasting"
```

## Reproduction

### Tests

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pytest -q -p no:cacheprovider
```

The suite contains 243 tests spanning provenance, signed-target preservation, issuance cutoffs, DST chronology, development/test separation, leakage controls, deterministic generation, optimization contracts, scenario integrity, FlexLab contracts, bootstrap reproducibility, and protected hashes. Four companion-repository cleanliness checks are skipped by default because that repository is independently managed. To run the strict release audit as well:

```powershell
$env:STRICT_EXTERNAL_REPO_CHECKS = "1"
python -m pytest -q -p no:cacheprovider -m external_integrity
```

### Honda H1–H5

```powershell
python -m src.honda.audit
python -m src.honda.h2_provenance
python -m src.honda.h3a_protocol_analysis
python -m src.honda.h3b_freeze_protocol
python -m src.honda.h4_baseline_benchmark
python -m src.honda.h5_deterministic_ml
```

H6 runners use isolated dependency stacks. Exact package states, checkpoints, hashes, devices, precision, and runtimes are recorded in [`results/foundation/`](results/foundation/).

### Later stages

```powershell
python -m src.honda.scenarios.pipeline
python -m src.honda.optimization.pipeline
python -m src.honda.decision_value.pipeline
python -m src.honda.decision_value_replication.replication
```

Generators verify frozen inputs and hashes. Reproduction from raw Honda data requires obtaining the Dryad release and following [`reports/honda_source_acquisition.md`](reports/honda_source_acquisition.md).

## Reproducibility safeguards

- Configurations are frozen before protected outcome analysis.
- UTC is the physical key; local time defines operational calendar meaning.
- Historical data are admitted only when available by forecast issuance.
- Model selection uses development years, not Honda 2023 test outcomes.
- Comparisons use common predicted targets and complete local days.
- Bootstrap resampling uses whole days, not individual intervals.
- Manifests record source, config, output, package, seed, and commit evidence.
- The replication consumes sealed schedules and does not rerun optimization.
- Repeated replication outputs are checked for byte-identical hashes.

## Principal limitations

1. Only SPARK is economically evaluated; Honda economics remain unsupported.
2. F1 capacity and state limits are hypothetical rather than plant-identified.
3. Price-publication timing is assumed, not verified from archival timestamps.
4. Wholesale prices are not a customer tariff or complete electricity bill.
5. Honda 2023 and SPARK 2024 were exposed in successive research stages.
6. Better dependence does not imply complete marginal, tail, holiday, or DST calibration.
7. Two sites cannot establish broad industrial external validity.
8. Honda export-state recognition remains weak.
9. No prospective shadow scheduling or operator validation was performed.
10. A formal software license must be selected before redistribution claims.

## Recommended next step

The next step should not be another broad forecasting-model survey. The highest-value extension is one evidence-rich case containing at least one of:

- measured process power, energy, ramp, and recovery constraints;
- an actual tariff, network-charge structure, and connection contract;
- verified price-publication and scheduling timestamps;
- a new untouched or prospective holdout; or
- an external site with a different flexibility mechanism.

Without those inputs, the current work should be finalized as a bounded exploratory thesis and paper. Additional zero-shot checkpoints, unconstrained capacity sweeps, synthetic batteries, or Honda economics without defensible timing would add complexity without resolving the principal validity limitations.

## Key reports

- [`complete_research_status.md`](reports/research_checkpoint/complete_research_status.md) — complete scientific checkpoint
- [`research_decision_brief.md`](reports/research_checkpoint/research_decision_brief.md) — concise decision brief
- [`industrial_flexibility_decision_value_replication.md`](reports/decision_value/industrial_flexibility_decision_value_replication.md) — feasibility replication
- [`thesis_contribution_assessment.md`](reports/decision_value/thesis_contribution_assessment.md) — thesis contribution assessment
- [`paper_overleaf/main.tex`](reports/paper_overleaf/main.tex) — research-paper source
- [`progress_overleaf/main.tex`](reports/progress_overleaf/main.tex) — stakeholder progress source

## Citation

Until a software DOI is assigned, cite the repository, exact Git commit, and Honda source dataset separately.

```text
Asker, M. Industrial Flexibility under Forecast Uncertainty:
Provenance-Safe Forecasting, Coherent Scenarios, and Decision-Value Evaluation.
GitHub repository, exact commit required.
```

Honda source data:

```text
Honda Research Institute Europe GmbH.
Dataset DOI: 10.5061/dryad.73n5tb363
```

## Responsible interpretation

When reporting results:

- distinguish signed grid exchange from gross demand;
- state label-quality and prediction-coverage rules;
- distinguish forecasting, scenario, optimization, and settlement evidence;
- label monetary results as synthetic counterfactual wholesale-cost differences;
- report feasibility violations alongside costs;
- distinguish assumed from verified information timing; and
- do not translate modeled differences into customer savings.

> **Better temporal dependence can improve modeled feasibility without improving mean decision cost. Forecast quality, scenario quality, and operational value must therefore be evaluated as related but distinct scientific objects.**
