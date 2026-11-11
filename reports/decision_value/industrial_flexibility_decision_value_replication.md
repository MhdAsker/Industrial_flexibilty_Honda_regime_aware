# Feasibility-Restricted SPARK Decision-Value Replication

## Executive conclusion

This pre-specified replication does **not** reverse the original negative decision-value result. On the 191 days where deterministic LightGBM (P2), independent residual scenarios (P3), and coherent whole-day scenarios (P4) are all compliant with the frozen zero-export cap, P2 is cheaper than P4 by **0.191191 EUR/day** (P2−P4; 95% day-bootstrap CI **[−0.271593, −0.131960]**). P3 is cheaper than P4 by **0.047240 EUR/day**, but this comparison is not statistically resolved (95% CI **[−0.122619, +0.009287]**; Holm-adjusted p=0.131). P2 is cheaper than P3 by **0.143951 EUR/day** (95% CI **[−0.169757, −0.120702]**).

P4 does materially reduce realized zero-export-cap violations: **11 days**, versus **82** for P2 and **16** for P3. That reduction is operationally relevant as a risk diagnostic, but it does not produce lower mean wholesale cost in this frozen counterfactual. The explicit all-day settlement correction gives the same qualitative ranking.

All monetary results are labelled **SYNTHETIC_COUNTERFACTUAL_WHOLESALE_COST_DIFFERENCE**, **NOT_CUSTOMER_SAVINGS**, **NOT_TRADING_PROFIT**, and **NOT_PLANT_VALIDATION**. Timing is **POST_PUBLICATION_TIMING_ASSUMED**, not independently timestamp-verified operation. Overall status remains **READY_WITH_LIMITATIONS** and **CATEGORY_C_EXPLORATORY**.

## Objective

The replication asks whether the apparent disadvantage of coherent S2 scheduling persists when:

1. the scheduling decision is explicitly placed after assumed day-ahead publication;
2. primary comparisons use a single intersection of days feasible for all three policies; and
3. nominal cap violations receive a transparent, pre-specified all-day settlement correction.

No forecast was retrained, no scenario was regenerated, no physical capacity changed, and no new model, scenario count, market design, CVaR setting, battery, thermal case, or Honda economic result was added.

## Frozen inputs

The replication configuration was frozen before replication outcomes at:

`config/decision_value_replication/stage3_feasibility_replication_frozen.json`

SHA-256:

`f00a24c55072cce5ec9399d506231d6f7fe67d29ae50448a8ea9ba1d3e39c0c3`

The analysis consumes the already sealed Stage 3 schedules in `results/decision_value/policy_decisions.parquet`, the original Stage 3 daily metrics, the frozen SPARK LightGBM target/forecast artifact, and the same historical price vector. It does not call the scenario generator.

Population:

- Dataset: SPARK only.
- Candidate days: 273 complete 96-interval days.
- Date range: 6 January–1 December 2024.
- Policies: P2 deterministic LightGBM, P3 S1 independent residual scenarios, P4 S2 coherent scenarios.
- Asset: frozen F1 MEDIUM, ±5 kW power and ±10 kWh advance/backlog state.
- Grid constraint: zero export; no import cap.
- Interval duration: 0.25 h.

The original Stage 3 results and artifacts remain unchanged.

## Timing assumption

The underlying SPARK forecast remains frozen at 12:00 Europe/Berlin on D−1. The replication places scheduling at an assumed 12:55 Europe/Berlin on D−1, after the documentary EPEX publication convention supplied to this experiment. No post-12:00 load measurements enter the forecast or schedule.

The historical price file has no observation-level publication timestamps. Therefore the exact label is **POST_PUBLICATION_TIMING_ASSUMED**. The source experiment remains **EX_POST_PRICE_AWARE**. This is not real-time operational validation.

## Feasibility definitions

For realized baseline `y_t` and the already sealed F1 schedule `u_t`:

`g_t = y_t + u_t`.

The frozen zero-export cap requires `g_t >= 0`. A day is policy-feasible if every realized interval meets this condition within `1e-7 kW`. A common-feasible day is one on which P2, P3, and P4 are all feasible. This produces one identical 191-day population for all primary paired comparisons.

The original schedules retain exact daily energy conservation and terminal recovery. Across the replication, maximum absolute schedule-energy residual is below `6.7e-15 kWh` and terminal-state residual is zero.

## Explicit settlement correction

The secondary all-day correction is:

`c_t = max(0, -g_t)`

`g_corrected,t = g_t + c_t`.

This is the unique componentwise minimum nonnegative intervention needed to satisfy the zero-export cap. The intervention cannot exceed the violation amount, including during negative-price intervals, so it cannot create additional negative-price consumption arbitrage. Corrected exchange is priced with the unchanged historical interval prices.

The committed F1 schedule is never changed, clipped, or reoptimized. Actuals are accessed only after the schedule is sealed. The correction is external to the F1 work-buffer state and is an ex-post settlement/feasibility sensitivity—not an implementable forecast-time decision or plant capability.

## Pre-registered hypotheses

- **H1:** On common-feasible days, S2 does not significantly reduce mean realized wholesale cost relative to deterministic scheduling.
- **H2:** S2 reduces cap-violation frequency relative to deterministic scheduling.
- **H3:** Any S2 advantage is concentrated on days with binding state or cap constraints.

Exactly three policy comparisons were frozen: P2−P4, P3−P4, and P2−P3. A positive A−B difference means policy B is cheaper. Primary uncertainty uses 2,000 whole-day bootstrap resamples with seed 20261029. Seven-day circular moving-block intervals are sensitivity results. Holm adjustment covers the three primary tests.

## Policy-level results

| Policy | Nominal cost, all 273 EUR/day | Nominal savings EUR/day | Violation days | Common-feasible cost EUR/day | Common-feasible savings EUR/day | Mean correction cost EUR/day | Corrected cost EUR/day | Corrected savings EUR/day |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P2 deterministic | 26.447976 | 1.999617 | 82 | 29.946908 | 2.326373 | 0.017155 | 26.465131 | 1.982463 |
| P3 independent | 26.646923 | 1.800671 | 16 | 30.090860 | 2.182422 | 0.001031 | 26.647953 | 1.799640 |
| P4 coherent | 26.700058 | 1.747536 | 11 | 30.138099 | 2.135182 | 0.001066 | 26.701123 | 1.746470 |

The common-feasible population contains 191 days for every policy. Its no-flex mean is approximately 32.273281 EUR/day. Differences in these subgroup means must not be compared with all-day means without acknowledging the population change.

## Primary common-feasible comparisons

Classification: **PRE_REGISTERED_PRIMARY**.

| Comparison | Days | Mean A−B EUR/day | 95% day-bootstrap CI | 7-day block CI | B daily win fraction | Holm-adjusted p |
|---|---:|---:|---|---|---:|---:|
| P2−P4 | 191 | −0.191191 | [−0.271593, −0.131960] | [−0.313554, −0.113278] | 0.0052 | 0.000 |
| P3−P4 | 191 | −0.047240 | [−0.122619, +0.009287] | [−0.135098, +0.005784] | 0.6597 | 0.131 |
| P2−P3 | 191 | −0.143951 | [−0.169757, −0.120702] | [−0.188752, −0.104662] | 0.0000 | 0.000 |

Interpretation:

- P2 remains robustly cheaper than P4 and P3 on the identical feasible-day set.
- P4 beats P3 on about 66% of days, but its adverse days are large enough that its mean cost is higher. The CI crosses zero, so the average difference is unresolved.
- Removing infeasible days weakens the original P3−P4 conclusion from a significant S2 disadvantage to an inconclusive mean disadvantage. It does not produce evidence that S2 is cheaper.

## Pre-specified corrected all-day comparisons

Classification: **PRE_SPECIFIED_SECONDARY**.

| Comparison | Days | Mean A−B EUR/day | 95% day-bootstrap CI | 7-day block CI | B daily win fraction |
|---|---:|---:|---|---|---:|
| P2−P4 | 273 | −0.235992 | [−0.300416, −0.184113] | [−0.321147, −0.172512] | 0.0330 |
| P3−P4 | 273 | −0.053170 | [−0.109739, −0.011935] | [−0.113356, −0.011379] | 0.6154 |
| P2−P3 | 273 | −0.182822 | [−0.209122, −0.157245] | [−0.221738, −0.145244] | 0.0293 |

The explicit correction reduces the deterministic apparent advantage modestly, from nominal P2−P4 −0.252081 to corrected −0.235992 EUR/day, but it does not reverse the ranking. P3 remains cheaper than P4 on mean even after correction.

## Nominal all-day diagnostic

Classification: **EXPLORATORY_DIAGNOSTIC** because the population contains cap-violating schedules.

The replication exactly reproduces the original Stage 3 nominal differences:

- P2−P4: −0.252081 EUR/day, CI [−0.315669, −0.199605].
- P3−P4: −0.053135 EUR/day, CI [−0.109641, −0.011827].
- P2−P3: −0.198946 EUR/day, CI [−0.226065, −0.173479].

These numbers are retained for traceability, not as the primary feasibility-restricted result.

## Feasibility diagnostics

| Policy | Compliant days | Violation days | Violation fraction | Mean violating-day energy kWh | P90 violating-day energy kWh | Maximum energy kWh | Maximum kW |
|---|---:|---:|---:|---:|---:|---:|---:|
| P2 | 191 | 82 | 30.04% | 0.7093 | 2.0154 | 9.4713 | 1.1259 |
| P3 | 257 | 16 | 5.86% | 0.2174 | 0.3281 | 2.0182 | 0.5017 |
| P4 | 262 | 11 | 4.03% | 0.3495 | 0.5171 | 2.0295 | 0.4599 |

Violation overlap is highly structured:

- 191 days: no policy violates.
- 65 days: P2 only.
- 6 days: P2 and P3.
- 1 day: P2 and P4.
- 10 days: all three.
- There are no days where P3 or P4 violates without P2 also violating.

Thus H2 is descriptively supported: S2 reduces violation frequency by 71 days versus deterministic scheduling and by 5 days versus independent scenarios. This is a counterfactual risk result, not verified plant operation.

## Binding-constraint diagnostics

Classification: **EXPLORATORY_DIAGNOSTIC**.

| Policy | Power-limit binding days | Energy-state binding days | Realized grid-cap violation days |
|---|---:|---:|---:|
| P2 | 273 | 273 | 82 |
| P3 | 271 | 253 | 16 |
| P4 | 266 | 251 | 11 |

The F1 power and state constraints bind on nearly all days, so they provide little subgroup separation. On the 11 days where P4 violates the realized grid cap, P4 is 0.036878 EUR/day cheaper than P3 on mean; on the other 262 days, P3 is 0.056914 EUR/day cheaper. The 11-day result is small, infeasible, and exploratory. It cannot establish H3. There is no robust evidence that a feasible, binding-constraint subgroup creates a coherent-scenario advantage.

## Hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: S2 does not significantly beat deterministic on common-feasible days | SUPPORTED | P2−P4 = −0.191191 EUR/day; CI entirely below zero, favoring P2 |
| H2: S2 reduces cap violations versus deterministic | SUPPORTED DESCRIPTIVELY | 11 versus 82 violation days; all P4 violations are also P2 violations |
| H3: S2 advantage is concentrated on binding days | INCONCLUSIVE / NOT ESTABLISHED | power/state bind almost universally; only an infeasible 11-day grid-cap subgroup favors P4 over P3 |

## Scientific interpretation

The feasibility restriction changes one nuance, not the overall conclusion. On all nominal days, P3 is significantly cheaper than P4. On the common-feasible intersection, the P3−P4 confidence interval crosses zero. Therefore the defensible statement is now:

> Coherent scenarios strongly reduce zero-export-cap violation frequency, but they do not demonstrate lower mean wholesale cost. Relative to independent scenarios, the feasible-day mean difference is inconclusive; relative to deterministic LightGBM, coherent scheduling remains robustly more expensive.

The result is compatible with the mathematical structure. Risk-neutral expected energy cost is driven primarily by marginal expectations; temporal dependence affects decisions only through state, cap, peak, or other nonseparable constraints. Here, F1 state and power bounds bind often, but coherence's risk reduction does not offset its cost. Better dependence representation is therefore not equivalent to better decision value.

## Limitations

1. The F1 resource is hypothetical, not an identified industrial process.
2. Historical price publication timestamps are absent; 12:55 is assumed from documentary convention.
3. SPARK 2024 was previously exposed, so this is not independent confirmation.
4. The correction is an ex-post external feasibility ledger, not an implementable plant action.
5. Common-feasible filtering changes the evaluated population and may select less difficult days.
6. Only SPARK is evaluated; no Honda economic inference is made.
7. Every evaluated day has 96 intervals; the experiment provides no DST-transition economic result.
8. Wholesale exposure is not a customer tariff or total electricity bill.

## Reproducibility and tests

Two consecutive complete replications produced byte-identical hashes for every deterministic table and Parquet output. `results/decision_value_replication/reproducibility_check.json` reports `passed=true` and no mismatches.

The full Honda suite passes **233/233 tests**, including 20 new replication tests. The stale Stage 2 protection test was repaired by explicitly allowing the committed Stage 3 and isolated replication namespaces while continuing to reject changes elsewhere in protected report/config/result trees. SPARK remained clean.

## Artifact paths and hashes

| Artifact | SHA-256 |
|---|---|
| `config/decision_value_replication/stage3_feasibility_replication_frozen.json` | `f00a24c55072cce5ec9399d506231d6f7fe67d29ae50448a8ea9ba1d3e39c0c3` |
| `src/honda/decision_value_replication/replication.py` | `2dfc078f061ad9a282557f7af653f79b19f697f4f230d2da5ead9d61927cd4b0` |
| `results/decision_value_replication/replication_manifest.json` | `88bc917eed7cce7601ba2837d8884fca798e33a280e1f7702f1b6dc3ff1b7949` |
| `results/decision_value_replication/reproducibility_check.json` | `2015c726d850d73ec8cd1a520a723dfa7f69b4a2a91040dfa02b65edb9aa14e5` |
| `results/decision_value_replication/replication_policy_summary.csv` | `7e4c5d5d6d8f97913a69d697033d5ad2a7266b348684adc3c9e1b9127bcc49fa` |
| `results/decision_value_replication/primary_common_feasible_comparisons.csv` | `30819f4ecb7ee452bbdec74776504818ac20f59f559d723d07057127a129083a` |
| `results/decision_value_replication/secondary_corrected_comparisons.csv` | `577a7ceea27781051f2df78c9985a04acf245fb8b0a535abaa84582de48f4160` |
| `results/decision_value_replication/feasibility_diagnostics.csv` | `ccf4291e36de30a315db066342603ae5c7234160ffcc254f6230710a1256c995` |
| `results/decision_value_replication/binding_constraint_diagnostics.csv` | `6278be55e527c15fa257e6e1efd9c15f318d66b916d74af6c4761ca1c0fb222e` |

Additional machine-readable outputs are in `results/decision_value_replication/`.

## Final classification

**READY_WITH_LIMITATIONS — CATEGORY_C_EXPLORATORY.**

The stopping rule is satisfied. S2 did not show lower feasible-day decision cost, so no further model, capacity, scenario, CVaR, market, or Honda experiment is added. The scientifically valuable result is bounded: coherent scenarios reduce modeled cap violations, but deterministic scheduling remains cheaper and independent-versus-coherent feasible-day cost is unresolved.
