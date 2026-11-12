# Phase 2 — Industrial Partner Shadow Scheduler

## Status and claim boundary

Phase 2 is now an implemented **engineering foundation**, not a completed industrial validation.

The repository can validate partner readiness, capture an immutable issue-time snapshot, solve a 48–72 hour two-stage schedule, issue a human-readable recommendation, record the operator response, settle against realized operation, and aggregate prospective cohort metrics. It does not control equipment.

The prospective shadow phase remains blocked until an industrial owner approves every required physical, operational, timing, and commercial input. The CLI enforces this boundary:

- incomplete or unapproved inputs: `OFFLINE_ENGINEERING_ONLY`;
- complete, digested, partner-approved inputs: `PARTNER_APPROVED_SHADOW`;
- issued recommendation: `PROSPECTIVE_SHADOW_ONLY_NO_AUTOMATIC_CONTROL`;
- settlement: `SHADOW_COMPARISON_NOT_CAUSAL_SAVINGS`.

No Honda or SPARK artifact currently satisfies the partner-readiness gate.

## Implemented architecture

```text
partner-readiness.yaml ──┐
                        ├─> immutable issue snapshot ─> readiness gate
issue.yaml ─────────────┘                              │
                                                       v
                                      two-stage rolling optimization
                                         │                     │
                                  recommendation       deterministic fallback
                                         │
                                  operator decision
                                         │
                                 realized settlement
                                         │
                                  8–12 week metrics
```

The operational record is write-once. A changed input requires a new `issue_id`; an existing snapshot cannot be silently overwritten.

## Partner-readiness contract

`shadow.partner-readiness.v1` requires approval evidence for all of the following:

1. measured power limits;
2. process-state or energy limits;
3. ramp constraints;
4. recovery requirements;
5. availability and maintenance windows;
6. production deadlines;
7. tariff and network-charge rules;
8. connection-contract limits;
9. verified price-publication timestamps;
10. operator overrides and reason codes; and
11. realized process schedules.

Every approval record needs a source identifier, owner role, timezone-aware approval time, and SHA-256 digest. A policy statement without evidence is not sufficient.

Run the gate with:

```powershell
shadow-scheduler readiness --partner path\to\partner-readiness.yaml
```

## Issue-time contract

`shadow.issue.v1` contains:

- issue identity and partner identity;
- the exact SHA-256 set of approved evidence to which the issue is bound;
- timezone-aware issue, price-publication, decision-deadline, and target timestamps;
- 48–72 physical hours of point and scenario forecasts;
- scenario weights and a scenario-tree node ID for every scenario and interval;
- interval-specific availability;
- approved deferrable-process constraints;
- energy, export, network, demand, imbalance, recourse, service-state, and connection-violation prices; and
- first-stage commitment, recourse start, adjustment, and connection-limit settings.

The scenario-node matrix is not decorative. Corrections belonging to the same information node are constrained to be equal. They may diverge only after the scenario tree branches, which makes the recourse explicitly nonanticipative.

Every issue must also contain valid SHA-256 identities for the production plan, maintenance/availability state, process state, weather forecast vintage, load forecast vintage, price vintage, tariff version, and adaptive-calibration state. A recommendation cannot be reproduced if any of those identities is missing.

## Partner-adaptive ML layer

The Phase 2 ML layer is now implemented as reusable, fail-closed components. It is intentionally data-empty until approved partner records arrive.

### Versioned online calibration

`AdaptiveCalibrationState` combines an exponentially updated bias with adaptive conformal miscoverage control. Each post-realization update produces a new immutable state containing:

- partner and model identity;
- monotonically increasing sequence number;
- target and current adaptive miscoverage;
- bias and learning rates;
- bounded nonconformity-score history;
- effective timestamp; and
- the previous state's cryptographic hash.

The exact state used for an issue must be included in that issue's provenance. Calibration is updated only after outcomes become available, never by rewriting a historical issue. The implementation follows the online feedback principle of [Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html); prospective empirical coverage still has to be demonstrated for the partner stream.

```powershell
shadow-scheduler adaptive-init `
  --partner-id PARTNER --model-id LOAD_V1 `
  --effective-at-utc 2026-01-01T00:00:00+00:00 `
  --output artifacts\calibration\state-0000.json

shadow-scheduler adaptive-update `
  --state artifacts\calibration\state-0000.json `
  --prediction prediction.json --realized realized.json `
  --effective-at-utc 2026-01-02T00:00:00+00:00 `
  --output artifacts\calibration\state-0001.json
```

### Multi-day blocks and genuine update trees

The multi-day module constructs 48–72 hour residual blocks that cross midnight and records the latest time at which every residual in a donor block was observable. Conditional selection refuses future donors. Scenario-tree node paths branch only at explicitly supplied forecast, price, production, or availability update timestamps and never merge after divergence.

### Process and operator learning

The implemented scientific interfaces include:

- a regularized learned transition/recovery-residual model;
- an operator acceptance-probability model;
- nonnegative inverse-preference estimation from chosen versus available schedules; and
- matched value-of-information ablations for weather, production, availability, and price updates.

These models are descriptive challengers. Operator acceptance must not be used to manipulate explanations or bypass safety constraints. Inverse preferences are not identifiable until repeated choices with meaningful alternatives exist.

## Predeclared prospective experiment

[`examples/shadow/phase2_prospective_protocol.template.json`](../examples/shadow/phase2_prospective_protocol.template.json) requires the operator baseline, deterministic ML baseline, calibrated stochastic policy, eligibility rules, missed-deadline rule, tariff version, primary safety endpoints, economic endpoint, and unacceptable-loss margin to be frozen before the first eligible live issue.

```powershell
shadow-scheduler protocol --protocol controlled\shadow\partner_protocol_frozen.json
shadow-scheduler protocol --protocol controlled\shadow\partner_protocol_frozen.json --block-id 2026-W10
shadow-scheduler exit-report --protocol controlled\shadow\partner_protocol_frozen.json --metrics results\shadow\prospective_metrics.json
```

A deterministic hash assigns switchback blocks only when the partner has explicitly enabled a switchback between two already-approved recommendation policies. Without that approval, the protocol reports `DESCRIPTIVE_ONLY`; counterfactual settlement differences are not called causal savings.

The exit evaluator fails closed unless all required endpoints are supplied and verifies:

- at least the predeclared prospective duration, never less than eight weeks;
- zero hard violations among accepted recommendations;
- all eligible recommendations delivered before their deadlines;
- reliable multi-day tail coverage;
- improvement in tail cost, intervention count, or violation duration;
- operator acceptance and preparation-time evidence; and
- an economic confidence interval whose upper bound is below the predeclared unacceptable-loss margin.

## Optimization semantics

The current production-supported asset model is `deferrable_process`.

For interval `t` and scenario `s`:

```text
physical adjustment[s,t] = day-ahead schedule[t] + intraday correction[s,t]
state[s,t+1] = state[s,t] + physical adjustment[s,t] × interval hours[t]
grid[s,t] = baseline scenario[s,t] + physical adjustment[s,t]
```

The day-ahead schedule is common to all scenarios. Intraday corrections are bounded, unavailable before their declared start, and equal within each information node. State is carried across midnight and reaches the approved recovery target at the end of the rolling horizon.

The objective can include:

- asymmetric import and export prices;
- volumetric network charges;
- demand/peak charges;
- imbalance and recourse energy costs;
- service-state costs; and
- hard connection limits or explicit violation penalties.

This is genuine modeled recourse. It is distinct from the legacy ex-post feasibility correction, which remains settlement-only and is not relabeled.

Only `deferrable_process` is currently implemented for Phase 2 value claims. Thermal buffers, storage, curtailment, CHP/PV, and multi-asset site balance remain planned interfaces; the validator rejects them rather than silently applying the wrong physics.

## Operating workflow

### 1. Capture the issue-time state

```powershell
shadow-scheduler snapshot `
  --issue path\to\issue.yaml `
  --partner path\to\partner-readiness.yaml `
  --output-root artifacts\shadow
```

The output directory is `artifacts/shadow/<partner_id>/<issue_id>/` and contains exact source copies, hashes, normalized decision identity, capture time, and the readiness report.

### 2. Issue the recommendation

```powershell
shadow-scheduler recommend --snapshot artifacts\shadow\<partner_id>\<issue_id>
```

The recommendation includes:

- expected and worst-case cost;
- price spread and flexible energy shifted;
- binding power, state, and grid constraints;
- worst-case grid-cap margin;
- the tail scenario driving risk;
- differences from the deterministic fallback; and
- the full recommended, recourse, state, fallback, and configuration records.

### 3. Record the operator decision

```powershell
shadow-scheduler operator-decision `
  --snapshot artifacts\shadow\<partner_id>\<issue_id> `
  --disposition APPROVED `
  --operator-role shift_operator
```

`REJECTED` and `MODIFIED` require a reason code. A modified schedule is rejected if it violates availability, power, state, recovery, or ramp constraints.

### 4. Settle against realized operation

```powershell
shadow-scheduler settle `
  --snapshot artifacts\shadow\<partner_id>\<issue_id> `
  --realized-baseline realized_baseline.json `
  --realized-schedule realized_flexible_schedule.json
```

Settlement reports actual operation, no-flex, recommended, and deterministic-fallback costs and peaks under the realized baseline. Recommended and deterministic results are counterfactual comparisons, not causal savings estimates.

### 5. Aggregate the prospective run

```powershell
shadow-scheduler metrics --root artifacts\shadow
```

The cohort output covers issue timeliness, approval/rejection/modification, hard violations, operator preparation time, delivered flexibility, correction energy, and cost differences. A predeclared analysis plan must determine which metrics control the go/no-go decision before the first live issue.

## Verification already present

Automated tests verify:

- 48-hour rolling state continuity and cross-midnight recovery;
- explicit scenario-tree nonanticipativity;
- deterministic fallback without recourse;
- partner-readiness refusal;
- immutable recommendation and operator records;
- mandatory rejection reason codes; and
- realized hard-constraint evaluation.

These are software tests using engineering fixtures. They do not constitute evidence of industrial value.

## What is still required for Phase 2 completion

### Partner work

- Obtain signed model ownership and approval evidence.
- Map source-system fields to the issue schema.
- Verify each publication and availability timestamp at observation level.
- Agree the baseline, override taxonomy, escalation route, and operational deadline.
- Approve the explanation vocabulary with operators.

### Engineering work

- Add authenticated source connectors and a scheduler that captures snapshots before every deadline.
- Implement multi-asset composition only for partner-approved asset types.
- Add forecast/scenario monitoring across successive issues and forecast updates.
- Add role-based access, retention policy, secrets management, and incident logging.
- Add service health, missed-deadline alerts, and deterministic fallback delivery.
- Freeze a versioned tariff engine against real bills and contract clauses.

### Scientific work

- Predeclare operator and deterministic baselines before the shadow run.
- Compare rolling coherent and independent scenario policies under the same information sets.
- Separate forecast value, scenario-dependence value, and recourse value with ablations.
- Report uncertainty across operational days, not only aggregate EUR/day.
- Run prospectively for at least 8–12 weeks before a formal go/no-go decision.
- Fit and validate the adaptive, transition, operator, and inverse-preference models on approved prospective records; their software implementation alone is not evidence of predictive value.

## Exit decision

Phase 2 may be called complete only when the industrial owner approves the models, timing is verified, nonanticipative rolling simulations are leakage-safe, at least 8–12 weeks of prospective records exist, accepted recommendations have zero approved hard-constraint violations, explanations are actionable, and the predeclared comparison demonstrates useful value beyond offline simulation.
