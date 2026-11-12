# Industrial Flexibility under Forecast Uncertainty

## Three-page comparative status report — 12 November 2026

## Page 1 — Executive assessment and scientific performance

This project is substantially stronger than a typical applied forecasting repository in research control, provenance, reproducibility, and the connection from prediction to operational decisions. It is not yet comparable to a completed industrial flexibility pilot. Its most defensible classification is **strong exploratory research and credible pre-pilot engineering, with no demonstrated industrial savings**.

The strongest contribution is not a new neural architecture. It is an auditable chain from source semantics and information timing through deterministic and probabilistic forecasting, coherent scenarios, nonanticipative optimization, operator approval, settlement, monitoring, and model promotion. Just as importantly, the project reports negative results: better dependence modeling improved modeled feasibility but did not improve mean cost. That is scientifically more valuable than presenting every forecast improvement as operational value.

| Dimension | Current standing | Comparative judgment |
|---|---|---|
| Data provenance and leakage control | Strong | Better than many applied energy-ML studies |
| Deterministic forecasting | Strong single-site evidence | Competitive applied work; local LightGBM clearly beats generic zero-shot models here |
| Probabilistic calibration | Credible exploratory evidence | Good marginal coverage; limited tail, regime and independent-test evidence |
| Coherent scenario generation | Promising | Methodologically aligned with established ECC/copula research, not algorithmically novel |
| Decision value | Scientifically useful negative result | Stronger falsification than many simulation-only studies; no positive industrial-value result |
| Reproducibility and MLOps controls | Excellent for research | Ahead of most academic repositories; behind production MLOps platforms |
| Industrial validation | Not established | Behind field pilots with measured assets, operators and activations |
| Multi-site generalization | Architecture only | No three-site seasonal evidence; global models remain unvalidated |

Raw errors should not be compared directly across unrelated studies or sites because target scale, semantics, horizon, weather information and eligibility differ. The comparisons below concern experimental rigor, relative improvement, operational evidence and methodological maturity.

### Scientific performance and position relative to research

### Forecasting: genuinely good, but approaching offline diminishing returns

On Honda 2023, the frozen site-specific LightGBM reduced MAE from the strong historical ensemble's **76.480 kW to 71.513 kW**, a **6.49% improvement**, with complete prediction coverage and a whole-day bootstrap interval below zero. This is a meaningful applied result. It is stronger than a comparison against persistence alone because the benchmark ensemble combined multiple development-selected historical analogues and climatologies.

The zero-shot foundation-model comparison is particularly informative. TimesFM 3.0 and Chronos-2 produced **95.431 kW** and **103.672 kW** MAE, respectively, versus **71.513 kW** for local LightGBM; export recall was about 11% for both foundation models versus 29.75% for LightGBM. This does not contradict broad foundation-model benchmark papers. It shows that a signed industrial exchange series—with rare exports, CHP/PV interactions, distribution shift and strict issuance-safe covariates—still rewards local structure. The result is a useful domain-specific counterexample to treating zero-shot forecasting as a default replacement for supervised local models.

The Phase 1 regime mixture improved LightGBM further to **70.447 kW MAE**, and daily paired improvement was statistically favorable, but the gain was only **1.49%**. Export recall rose to **33.03%**, missing the 40% gate. The correct verdict was `DO_NOT_PROMOTE`. Compared with researchers who choose a model after inspecting a test year, the project's frozen promotion gates and exposed-test warning are major strengths.

This is now partly saturated on the current Honda data. More LightGBM tuning, AutoML or repeated testing against 2023 has low scientific value. New untouched observations, verified forecast vintages, and better export-regime information would be more valuable than another hyperparameter search.

### Probabilistic forecasts and scenarios: good calibration, incomplete tail evidence

Quantile LightGBM achieved **89.61% coverage** for a nominal 90% interval, only 0.39 percentage points from nominal. However, conformal correction improved WIS by just **0.42%**, far below the 5% gate. The model therefore demonstrated reliability without demonstrating enough sharpness or decision improvement for promotion.

Conditional ECC-200 performed better: on 346 supported Honda 2023 days it reached **89.17% coverage** and improved WIS by **3.89%**, energy score by **1.59%**, variogram score by **3.74%**, and ACF error by **1.87%** relative to frozen S2. Development-only selection chose bandwidth 2.0 and retained 200 scenarios. Reducing to 25 paths dropped coverage to **75.75%** and worsened WIS, so rejecting reduction was scientifically correct.

ECC itself is established rather than novel; its purpose is to join calibrated marginals with empirical dependence, as in [Schefzik et al.](https://arxiv.org/abs/1302.7149). The repository's contribution is the causal donor-availability check, whole-day evaluation, explicit marginal/dependence separation, and downstream decision audit. This aligns with probabilistic-load research using post-processed weather ensembles and ECC, but the project has less data, weaker verified weather input and no independent future holdout. Its provenance discipline is stronger than many such studies.

### Decision value: the negative result is a strength

On 273 SPARK days, deterministic LightGBM produced synthetic savings of **€1.9996/day**, independent scenarios **€1.8007/day**, and coherent scenarios **€1.7475/day**. Coherent scenarios reduced zero-export-cap violation days from 82 under deterministic scheduling to 11, but did not reduce cost. On 191 common-feasible days, deterministic remained cheaper than coherent by **€0.191/day** with the confidence interval below zero; independent-versus-coherent was unresolved.

This is consistent with research showing that a full joint predictive distribution is valuable only when the decision structure actually rewards dependence; expected values or marginals can suffice in simpler settings ([Beykirch et al.](https://doi.org/10.1109/TPWRS.2024.3381472)). It also contrasts appropriately with industrial-process studies where identified two-stage models and CVaR can reduce risk without large expected-cost penalties ([Schäfer et al.](https://doi.org/10.1002/aic.17828)). The present project lacks those identified plant mechanisms.

The finding that assumed flexibility capacity changes savings by much more than forecast policy is crucial. It says the next research priority is physical identification—not more elaborate forecasting. Decision-focused learning is appropriately deferred: training against invented constraints would optimize the wrong system.

---

## Page 2 — Engineering quality, industrial comparison, advantages and disadvantages

### Where the repository is unusually strong

The repository now has **287 collected tests**, cross-platform CI, protected-artifact checks, immutable issue snapshots, hash-linked calibration states, explicit target semantics, delayed-label status, same-snapshot champion/challenger comparison, model rollback identities, role-based access, audit chaining, human approval, deterministic fallback, and fail-closed `UNKNOWN` monitoring. Relative to a normal research repository, this is exceptional.

The prediction/label ledger records site, issue, interval, model, information cutoff, semantics, prediction/scenario identity, eventual label, model/feature/code/environment hashes, calibration version, operator action and decision outcome. Honda signed exchange and SPARK nonnegative demand cannot be silently pooled. This is exactly the kind of control needed before global forecasting, and it is often absent from papers reporting only aggregate MAE.

Claim discipline is another major advantage:

- synthetic wholesale differences are not called customer savings;
- ex-post price availability is labeled;
- Honda economics is blocked without acceptable prices;
- hypothetical capacities are not called plant capability;
- ex-post correction is not relabeled as recourse;
- interval rows are not treated as independent experimental units; and
- engineering fixtures are not presented as prospective validation.

These practices make the work easier to trust and reproduce. The architecture also follows the right industrial direction: the IEA emphasizes that demand flexibility can provide network, peak, resilience and system-cost value, but uptake and cybersecurity determine whether that value is realizable ([IEA, 2025](https://www.iea.org/reports/the-value-of-demand-flexibility)).

### Where leading industrial projects are ahead

Field programs are ahead on the evidence that matters most. Industrial demand-response studies have conducted multi-month factory campaigns and measured actual reducible power and recovery. The California Energy Commission has field-tested industrial refrigeration and water-pumping controls ([CEC field-test report](https://www.energy.ca.gov/sites/default/files/2023-11/CEC-500-2023-060.pdf)). Large European projects such as [OneNet](https://cordis.europa.eu/project/id/957739/reporting) and [InterFlex](https://cordis.europa.eu/project/id/731289/reporting) combined real pilots, market/DSO interfaces, demonstrations, KPI evaluation and business-model work.

This project currently has none of the following: an approved flexible plant, a measured response envelope, an authenticated SCADA connector, a real tariff engine validated against bills, prospective operator decisions, accepted recommendations, seasonal service evidence, or realized net value after integration and labor costs. Its roughly €2/day synthetic SPARK value could easily be overwhelmed by software and operator cost. Therefore, its engineering breadth must not be mistaken for product maturity.

Production MLOps is also incomplete. SQLite and synchronous orchestration are reference implementations, not high-availability infrastructure. There is no managed time-series database, durable job queue, retry/idempotency service, live feature store, centralized artifact store, SSO/OIDC, managed secret rotation, external penetration test, disaster-recovery evidence, alert/on-call process, SBOM/signing, load testing or demonstrated service-level performance. The platform is approximately mature research software, but early-stage production software.

### Main advantages

1. Strong end-to-end scientific control rather than isolated model accuracy.
2. Rare and valuable reporting of falsified hypotheses.
3. Excellent provenance, timing and semantic separation.
4. Strong transparent baselines and equal-support model comparisons.
5. Correct human-in-the-loop and fallback design.
6. Clear path from offline work to shadow mode and later multi-site evaluation.
7. Conservative handling of foundation models, generative scenarios and offline RL.

### Main disadvantages

1. Honda 2023 and SPARK 2024 are exposed; further analyses are not confirmatory.
2. Only two sites exist, and their physical targets are not comparable.
3. Process physics, tariffs, availability and recovery are hypothetical or missing.
4. Only the deferrable-process asset supports current Phase 2 claims.
5. Export events remain poorly classified.
6. No real operator acceptance, modification or preparation-time data exist.
7. No causal savings design has run.
8. Scenario count remains high because present reduction damages tails.
9. The architecture is now ahead of the evidence and risks over-engineering.

---

## Page 3 — Saturation, what can and cannot improve, and recommended direction

### Are we at saturation?

**Offline algorithmic work is near partial saturation; the project as a whole is not.** The evidence is visible in the diminishing gains: 1.49% point improvement from the regime mixture, 0.42% WIS improvement from conformal correction, and 3.89% WIS improvement from conditional ECC—all below promotion gates. Zero-shot foundation models are not close. More complex generative models would increase compute and researcher degrees of freedom without repairing missing industrial information.

By contrast, operational learning is almost untouched. A partner could change the ranking of policies by supplying actual state coupling, recovery deadlines, production loss, demand charges, imbalance exposure and operator constraints. Global forecasting research shows that pooled models can generalize well across multiple series ([Montero-Manso and Hyndman](https://doi.org/10.1016/j.ijforecast.2021.03.004)), but this cannot be tested credibly until several sites share comparable target semantics. TACTiS and TimeGrad are reasonable future challengers only at much larger multivariate scale; their published settings involve high-dimensional or thousands-of-dimensions data, unlike the present effective sample.

### What can still be improved internally

- asymmetric or class-balanced export modeling on development data;
- tail- and duration-preserving scenario reduction with measured optimization distortion;
- dependency locking, artifact signing, security scanning and static typing;
- database migrations and a PostgreSQL/time-series adapter;
- durable workers, retries, telemetry, dashboards and performance budgets;
- failure-injection, backup/restore and fallback exercises;
- clearer cost accounting for compute, integration and operator time; and
- automatic ledger-backed reports for whole-day and site-level promotion gates.

These are worthwhile, but none will establish industrial value by itself.

### What cannot be repaired with the present repository

- missing historical publication timestamps;
- an already exposed test period;
- unknown plant-level meaning behind aggregate exchange;
- unmeasured power, ramp, state and recovery constraints;
- missing tariff and network-contract clauses;
- operator preferences and acceptance behavior;
- causal savings without an approved comparison design;
- cross-site transfer without comparable sites; or
- real reliability, cybersecurity and service-level evidence.

These require new people, measurements, contracts and prospective time—not a different algorithm.

### Recommended next decision

Stop broadening the offline model zoo. The highest-value plan is:

1. Secure one partner and freeze asset, tariff, information-timing, baseline and safety contracts.
2. Build one authenticated read-only connector set and retain every forecast, price, production and availability vintage.
3. Verify physical flexibility through controlled response tests before making value claims.
4. Run 8–12 weeks of prospective shadow operation, including missed deadlines and rejected recommendations.
5. Measure hard violations, tail duration, total-bill impact, acceptance, modification reasons, preparation time and net value after operating cost.
6. Use a safe switchback only between already-approved policies; otherwise keep comparisons descriptive.
7. Add second and third semantically comparable sites only after the first site shows safe value.
8. Then run the frozen local → pooled → local-adapter → partial-pooling hierarchy. Keep neural global models, diffusion/flow scenarios and offline RL deferred until their data gates are genuinely met.

### Final verdict

Compared with other researchers, the project is **above average in experimental discipline, provenance, reproducibility, falsification and end-to-end design**. It is **competitive but not frontier-novel in forecasting and probabilistic methods**. It is **below field-demonstrated industrial projects in physical validation, operators, integrations, seasonal operation and realized economics**.

The project has not failed; it has reached the point where better science requires a different kind of evidence. Its most valuable asset is now the infrastructure and discipline to learn safely from a real partner without confusing predictive improvement with industrial value.
