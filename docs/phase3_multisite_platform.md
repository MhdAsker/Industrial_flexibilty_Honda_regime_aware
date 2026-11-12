# Phase 3 — Multi-site Flexibility Operations Platform

## Current status

Phase 3 now has a tested **platform foundation**. It is not a deployed industrial platform and it does not authorize equipment control.

The prerequisite chain remains explicit:

```text
FlexLab offline evidence
        ↓
partner-approved Phase 2 shadow run
        ↓
human-approved recommendations
        ↓
limited controlled pilot
        ↓
separate safety case and automated action
```

No real site has yet completed the Phase 2 prospective exit criteria. Every new tenant therefore starts in `OFFLINE_SIMULATION`, and control is blocked in both offline and prospective-shadow stages.

## Implemented foundation

### Tenant, site and asset boundaries

`honda.platform` adds immutable versioned registries for:

- tenants and sites;
- multiple assets per site;
- live connector definitions;
- forecast, scenario, optimizer and tariff models; and
- champion aliases with an explicit rollback model.

Site records carry timezone, contractual import/export limits and ownership. Asset and model records must reference evidence or artifact SHA-256 digests. Duplicate registry versions are rejected instead of overwritten.

### Connector contract

The connector interface covers:

- meter/SCADA;
- production plans;
- asset availability;
- weather;
- market prices;
- tariffs/contracts;
- operator overrides; and
- settlement data.

Every batch carries separate observation, source-availability and retrieval timestamps. Snapshot validation rejects:

- naive timestamps;
- future observations;
- data retrieved after the issue time;
- data published after the issue time;
- stale data;
- empty batches; and
- connector/batch identity mismatches.

Credential values are not stored in connector records. Only `vault://` and `keyring://` references are accepted, and credential-like query parameters are rejected from endpoints.

The included `InMemoryConnector` is for deterministic tests and offline replay only. It is not represented as a live industrial integration.

### Workflow orchestration

The reference orchestrator executes:

```text
connector snapshot → forecast → scenarios → optimization → pending approval
```

Each state transition is persisted and added to the audit chain. Failed optimization stops before a decision is issued. Successful output creates an immutable pending decision with:

- issue and deadline times;
- input snapshot digest;
- forecasts and scenarios;
- optimization and deterministic fallback output;
- deployment stage; and
- `control_authorized: false`.

The reference implementation is synchronous so its state transitions are easy to test. A production deployment still needs a durable scheduled/event-driven worker and retry queue.

### Deployment and safety gates

The permitted stages are:

1. `OFFLINE_SIMULATION`
2. `PROSPECTIVE_SHADOW`
3. `HUMAN_APPROVED`
4. `LIMITED_CONTROLLED_PILOT`
5. `AUTOMATED_ACTION`

Promotions must advance one stage at a time. Every promotion requires owner approval, audit verification, rollback testing and fallback testing. Controlled-pilot and automated stages additionally require prospective-pilot and incident-review evidence.

Automated action additionally requires:

- seasonal value evidence;
- a disaster-recovery test;
- a separately digested safety case; and
- two-person approval.

Even after promotion, the current implementation requires human approval and a healthy monitoring report before returning `control_authorized: true`. This is intentionally more conservative than the final optional automation target.

### Security and audit

The foundation provides:

- salted `scrypt` API-token hashes;
- viewer, operator, site-admin, platform-admin and auditor roles;
- per-site scopes;
- indirect credential references;
- immutable decision and settlement identifiers; and
- a SHA-256 hash-chained audit log.

Tokens are revealed once and never stored in plaintext. Audit verification recomputes every link and reports the first failing sequence.

This is application-level security scaffolding, not a completed security certification. Production still requires TLS termination, managed identity, key rotation, centralized secrets, database encryption, network segmentation, dependency scanning and an external penetration test.

### Monitoring

Health evaluation covers:

- data age and missingness;
- clock drift;
- forecast MAE and bias;
- scenario coverage and dependence;
- distribution shift;
- solver feasibility and runtime;
- hard-constraint violations;
- recommendation/actual deviation;
- realized versus expected value; and
- tariff/model-rule changes.

Freshness, clock, solver-feasibility and hard-constraint failures are critical. Any critical or warning state prevents the monitoring layer from approving automatic action.

Missing forecast accuracy, dependence, or distribution-shift values now produce `UNKNOWN`, not a passing check. `UNKNOWN` blocks controlled action exactly as a failed health gate does.

## Global multi-site adaptive ML

### Prediction and label ledger

The platform now persists one immutable row per `site × issue × interval × model`. Each row records:

- information cutoff and target timestamp;
- explicit physical target semantics;
- prediction and optional scenario identity;
- model, feature, code, environment and immutable-snapshot hashes;
- calibration version;
- delayed-label state (`PENDING`, `OBSERVED`, or `UNAVAILABLE`);
- eventual label;
- operator action; and
- decision outcome.

Predictions cannot be overwritten, and a delayed label can be attached only once. Champion/challenger comparison requires exactly the same observed site/issue/interval support, snapshot hash, target semantics, and label. A mismatch raises an error instead of silently comparing different information sets.

```powershell
flexops --database artifacts\platform\flexops.sqlite ledger-status --tenant TENANT
flexops --database artifacts\platform\flexops.sqlite compare-models `
  --tenant TENANT --champion LOCAL_LGBM --challenger GLOBAL_ADAPTER
```

### Hierarchy and evaluation order

The benchmark contract fixes this order:

1. local LightGBM per site;
2. pooled LightGBM with site identity and within-site normalized features;
3. global prediction plus local residual/calibration adapter;
4. hierarchical probabilistic partial pooling;
5. compact neural global model with site embeddings; and
6. joint cross-site scenarios.

Implemented evaluation primitives include rolling-origin known-site splits, leave-one-site-out cold-start splits, training-only within-site normalization, partial-pooling residual adaptation, per-site metrics, macro and worst-site aggregation, and joint empirical-copula scenarios.

Cold-start training is restricted to sites with the same target semantics. Honda signed grid exchange is therefore never used as though it were a SPARK nonnegative-demand label. Joint scenarios may coordinate separately calibrated site marginals, but the physical target arrays remain separate.

The execution template is [`examples/platform/multisite_ml_policy.template.json`](../examples/platform/multisite_ml_policy.template.json).

```powershell
flexops --database artifacts\platform\flexops.sqlite ml-readiness `
  --approved-sites 3 --seasonal-days 365 --pooled-rows 100000
```

### Promotion controls

The fail-closed promotion evaluator requires all site and service values. Missing values return `UNKNOWN` and block promotion. A challenger must satisfy every gate:

- at least 2% macro MAE improvement by default;
- no site degradation beyond 2%;
- no degradation in worst-site MAE;
- per-site bias and coverage thresholds;
- non-inferior decision cost and operator acceptance;
- inference and solver p95 service levels;
- drift alarms preceding safety-threshold breaches; and
- zero hard violations.

Model promotion does not change the deployment stage or authorize control.
Registering a global champion additionally requires the SHA-256 identities of both its identical-snapshot ledger comparison and its complete promotion report; ordinary approval and rollback references alone are insufficient.

### Advanced-model boundary

TACTiS, normalizing flows, diffusion models and graph neural networks remain behind a separate scale gate. The conservative default software gate requires at least ten comparable approved sites, one year of seasonal evidence and 100,000 pooled rows before they are even eligible as challengers. Passing that gate is not performance evidence.

This boundary reflects the mismatch between the present project and the settings for which these methods were introduced: [TACTiS](https://proceedings.mlr.press/v162/drouin22a.html) targets high-dimensional joint distributions and reports scaling to hundreds of series, while [TimeGrad](https://proceedings.mlr.press/v139/rasul21a.html) was demonstrated on multivariate datasets with thousands of correlated dimensions.

Offline reinforcement learning remains prohibited until prospective logs contain valid state, action, reward, next-state and behavior-policy information with adequate action coverage. This follows the distribution-shift and dataset-coverage limitations emphasized in the [offline RL review by Levine et al.](https://arxiv.org/abs/2005.01643).

### Portfolio allocation

The portfolio allocator coordinates a requested response across multiple sites using:

- site response capacities;
- site marginal costs;
- network-group limits;
- a site covariance matrix;
- risk aversion; and
- an optional portfolio variance budget.

It reports site allocations, cost, portfolio variance, binding network groups and a covariance-based diversification diagnostic. This is an allocation primitive, not evidence that multi-site value exists. Real claims require approved site response envelopes and prospective realized results.

### Operational API

Install the optional service dependencies:

```powershell
python -m pip install -e ".[platform]"
```

Initialize and serve:

```powershell
flexops --database artifacts\platform\flexops.sqlite init
flexops --database artifacts\platform\flexops.sqlite serve --host 127.0.0.1 --port 8000
```

The FastAPI application exposes authenticated routes:

```text
GET  /forecasts
GET  /scenarios
GET  /assets
GET  /tariffs
GET  /models
GET  /decisions
GET  /settlements
GET  /health
GET  /prediction-ledger
GET  /model-comparisons
POST /decisions/{decision_id}/operator-action
```

Operator actions require role authorization. Rejections and modifications require reason codes, and the API never turns an operator action directly into equipment control.

## Basic administration

Initialize storage:

```powershell
flexops --database artifacts\platform\flexops.sqlite init
```

Register a site:

```powershell
flexops --database artifacts\platform\flexops.sqlite register-site `
  --file examples\platform\site.template.json `
  --actor platform-admin
```

Create a scoped operator credential:

```powershell
flexops --database artifacts\platform\flexops.sqlite create-principal `
  --tenant REPLACE_WITH_TENANT `
  --principal shift-operator-1 `
  --role operator `
  --site REPLACE_WITH_SITE `
  --actor platform-admin
```

The token is printed once. It should be transferred immediately to a managed secrets system.

Verify audit integrity:

```powershell
flexops --database artifacts\platform\flexops.sqlite verify-audit
```

## Verification

Automated tests currently cover:

- immutable registry versions;
- audit-chain verification and tamper detection;
- salted token storage, role permissions and site scope;
- rejection of future information;
- persisted workflow transitions;
- offline/shadow control blocking;
- sequential stage promotion and safety-case requirements;
- champion evidence and rollback identity;
- monitoring-based action blocking;
- portfolio network/risk constraints; and
- presence of all required API routes.
- delayed-label ledger immutability and identical-snapshot comparison;
- target-semantic separation in known-site and cold-start evaluation;
- partial-pooling cold-start behavior;
- joint scenario marginal preservation;
- fail-closed global promotion; and
- advanced-model deferral without sufficient scale.

CI runs the platform and Phase 2 shadow tests on Windows and Linux with Python 3.10 and 3.12.

## Important gaps before an industrial pilot

The following are intentionally not claimed as complete:

- no authenticated Honda, SPARK or partner SCADA connector exists;
- SQLite is a reference persistence layer, not the target high-availability time-series architecture;
- no durable job queue, retry policy or high-availability scheduler is deployed;
- no partner-approved multi-asset site optimizer exists beyond the Phase 2 deferrable process;
- no market bidding or dispatch connector is enabled;
- no infrastructure-as-code, backup environment or disaster-recovery exercise exists;
- no external security assessment has been performed;
- no real operator has used the API prospectively; and
- no prospective multi-site diversification value has been measured.

## Next implementation sequence

1. Complete Phase 2 with one partner site and 8–12 weeks of prospective evidence.
2. Implement one authenticated, read-only connector set against that partner’s systems.
3. Move persistence to managed PostgreSQL/time-series storage with encryption, backups and retention controls.
4. Deploy the orchestrator as a durable worker with missed-deadline alerts and idempotent retries.
5. Run the human-approved recommendation stage without control.
6. Add a second approved site and predeclare the portfolio diversification experiment.
7. Conduct failure injection, fallback, rollback and disaster-recovery exercises.
8. Prepare a separate safety case before any limited controlled pilot.
9. After at least three comparable sites have seasonal evidence, run the hierarchy in its frozen order and promote only through the ledger-backed gates.

## Phase 3 completion rule

Phase 3 remains incomplete until at least one real site finishes a prospective pilot, every decision is reproducible from an immutable available-information snapshot, service levels are demonstrated, operators retain rejection authority, critical provenance and safety incidents remain at zero, net realized value exceeds total operating cost, multi-site transfer is evaluated, and rollback/fallback/disaster recovery have been exercised.
