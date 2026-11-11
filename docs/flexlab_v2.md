# FlexLab v2 — Offline Flexibility Decision Lab

FlexLab v2 is the reusable application layer above the repository's frozen v1 scientific evidence. It does not modify H1–H7 or Stage 2/3 artifacts. It provides versioned input contracts, common site adapters, one CLI, calibrated-trajectory utilities, policy comparison, and reproducible decision bundles.

The read-only dashboard is privately deployed at [FlexLab Industrial Flexibility Decision Lab](https://flexlab-industrial-decision-lab.burlyaphid3.chatgpt.site). It loads the synthetic demonstration by default and accepts a locally generated `dashboard.json` without uploading experiment data to a backend.

## Quick start

```powershell
python -m pip install -e ".[all]"
flexlab validate --experiment examples/flexlab/synthetic/experiment.yaml
flexlab run --experiment examples/flexlab/synthetic/experiment.yaml --output results/flexlab/demo
flexlab reproduce --experiment examples/flexlab/synthetic/experiment.yaml
```

The `run`, `evaluate`, and `optimize` commands execute the complete offline chain. `forecast` and `scenarios` validate and summarize their configured inputs. This deliberately keeps one source of truth rather than creating independent command-specific data transformations.

## Experiment schema

Every experiment uses `flexlab.experiment.v2` and contains five contracts:

- `site`: adapter, target semantics, unit, sign, timezone, day, and seed;
- `asset`: F1 power, state, ramp, and recovery values;
- `grid`: connection limits and constraint formulation;
- `tariff`: import/export vectors or scalar prices; and
- `timing`: forecast issue, price publication, decision time, and evidence class.

Invalid units, sign conventions, horizons, tariff lengths, timing, weights, timestamps, and finite-value assumptions fail before optimization.

## Site adapters

| Adapter | Purpose | Data status |
|---|---|---|
| `synthetic` | Portable CI, demos, DST and analytical tests | Always available |
| `honda` | Frozen Honda 2023 point/scenario evidence | Requires local generated scenario Parquet |
| `spark` | Frozen SPARK 2024 point/scenario evidence | Requires `SPARK_RESEARCH_ROOT` and local scenario Parquet |

All adapters return the same `SiteData` contract: unique UTC timestamps, actuals, point forecasts, independent scenarios, coherent scenarios, weights, and provenance.

## Decision bundle

Each successful run exports:

```text
bundle/
  experiment.yaml
  policy_summary.csv
  schedules.csv
  states.csv
  summary.json
  dashboard.json
  manifest.json
```

The manifest records the Git commit, package versions, experiment configuration, input provenance, output hashes, and allowed/non-allowed claims. `dashboard.json` can be loaded directly into the read-only FlexLab dashboard.

## Uncertainty calibration

`src/honda/scenarios/calibration.py` provides a development-only global trajectory scale. It expands or contracts deviations around the point forecast while retaining each scenario's within-day temporal shape. Selection minimizes mean absolute marginal coverage error over predeclared central coverage levels.

The module also reports:

- marginal interval coverage;
- simultaneous whole-day coverage;
- daily-energy coverage; and
- maximum-ramp upper coverage.

The scale must be fitted on chronological development observations. It must not use Honda 2023, SPARK 2024 economic outcomes, or optimizer results for tuning.

## Grid formulations

The existing optimizer supports the common v2 contract:

- `nominal`;
- `scenario_wise`;
- `chance_approximation`;
- `soft` with an explicit violation coefficient.

This makes constraint design comparable without creating separate optimization implementations.

## Claim boundary

Every bundle carries supported and unsupported claims. Synthetic configurations must not be represented as customer savings, trading profit, or plant validation. Operational timing can be labelled verified only when price publication exists and the decision timestamp follows it.

## Testing

FlexLab's portable suite covers:

- configuration validation;
- duplicate timestamps;
- tariff horizons;
- information timing;
- ordinary, spring-DST, and autumn-DST days;
- synthetic end-to-end optimization;
- deterministic reruns; and
- calibration and joint-reliability behavior.

GitHub Actions runs the synthetic suite on Windows and Linux with Python 3.10 and 3.12. External Honda/SPARK evidence remains covered by the repository's existing integrity and scientific tests.
