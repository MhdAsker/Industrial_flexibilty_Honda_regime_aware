# Offline Prescriptive ML Upgrade — Implementation and Exploratory Results

## Status

The first Offline Prescriptive ML implementation is complete as an exploratory challenger framework.

It adds:

- a regime-aware signed-exchange mixture;
- quantile LightGBM at seven probability levels;
- deterministic non-crossing quantile projection;
- lead-band × season conformal interval calibration;
- conditional similarity-weighted residual trajectories;
- empirical copula coupling;
- weighted scenario reduction diagnostics;
- an execute-first rolling-MPC simulation contract;
- CVaR, VSS, EVPI, recourse-value and oracle-regret utilities; and
- fail-closed promotion gates.

The Honda 2023 period was already exposed by earlier research stages. All new 2023 numbers below are therefore exploratory diagnostics, not a new confirmatory model promotion.

## Frozen design

The experiment uses the existing H5 issuance-safe `M1-CHP` feature contract. No future realized weather, target, PV, CHP, thermal or operational values were added.

Two point challengers were evaluated:

1. `REGIME_MIXTURE`: export classifier, conditional import/export magnitudes, direct signed regression and a development-selected soft blend.
2. `QUANTILE_MEDIAN`: the median of seven independently trained quantile LightGBM models.

Development selection used 2021–2022 only. Internal mixture weights and conformal corrections were fitted on training-period calibration tails rather than the evaluation year. The exact exploratory protocol is recorded in [`config/prescriptive_ml/phase1_exploratory_protocol.json`](../config/prescriptive_ml/phase1_exploratory_protocol.json).

## Point-forecast results

### Development selection

| Candidate | MAE (kW) | RMSE (kW) | Bias (kW) | Export recall |
|---|---:|---:|---:|---:|
| Regime mixture | **67.638** | **92.615** | +11.705 | **29.67%** |
| Quantile median | 71.606 | 96.385 | +25.393 | 25.21% |

The regime mixture was selected without using 2023.

### Exposed 2023 diagnostic

| Model | MAE (kW) | RMSE (kW) | Bias (kW) | Export recall | False-export rate |
|---|---:|---:|---:|---:|---:|
| Existing LightGBM champion | 71.513 | 96.974 | +9.408 | 29.75% | 1.94% |
| Regime mixture | **70.447** | **95.350** | +9.356 | **33.03%** | 1.93% |
| Quantile median | 73.295 | 97.079 | +10.962 | 27.36% | 1.68% |

For the regime mixture, the paired whole-day MAE difference relative to the champion was −1.081 kW, with a 2,000-sample bootstrap interval of −1.662 to −0.493 kW. It won on 62.22% of the 360 evaluated local days.

This is promising but insufficient for promotion:

- the 70.447 kW MAE misses the 69.4 kW gate;
- export recall improved by only 3.28 percentage points, missing the 40% gate;
- the effect is measured on an exposed test period; and
- the point-only mixture has not yet passed probabilistic or rolling-decision gates.

The fail-closed decision is `DO_NOT_PROMOTE`.

## Reproduction commands

With the provenance-controlled SILVER artifacts present locally:

```powershell
prescriptive-ml build-features
prescriptive-ml run --output results\prescriptive_ml
prescriptive-ml select-scenarios --output results\prescriptive_ml
prescriptive-ml scenarios --output results\prescriptive_ml --raw-count 200 --reduced-count 25 --bandwidth 2.0
prescriptive-ml audit --output results\prescriptive_ml\current_audit.json
```

Large feature and prediction Parquet files remain local and ignored by Git. The compact protocol, summaries, day-level scenario scores and fail-closed promotion reports are versioned.

## Quantile calibration

The calibrated quantile model achieved:

| Metric | Result | Gate |
|---|---:|---:|
| Nominal 90% interval coverage | **89.61%** | 87–93% |
| Absolute coverage error | **0.39 percentage points** | ≤3 points |
| Mean 90% interval width | 293.79 kW | diagnostic |
| Raw WIS | 163.641 | — |
| Calibrated WIS | 162.949 | ≥5% improvement required |
| WIS improvement | **0.42%** | ≥5% |

Calibration succeeded, but sharpness/value improvement was too small and the median forecast was worse than the existing champion. The quantile model is therefore not promoted.

## Conditional coherent scenarios

The scenario challenger used:

- forecast daily energy, extrema, variability, ramps and export fraction;
- season and day type;
- issuance-safe recent grid level and export fraction;
- issuance-safe CHP on-fraction;
- development residual trajectories completed before the target issue time;
- calibrated marginal samples; and
- empirical copula coupling.

The first run caught and rejected a donor day that was not complete at the first 2023 issue time. The final comparison filters donors by their actual availability timestamp.

### Exposed Honda 2023 diagnostic, 346 supported days

| Method | WIS | Energy score | Variogram score | 90% coverage | Width (kW) | ACF RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Frozen coherent S2-50 | 169.346 | 66.486 | 11.255 | 82.17% | 262.56 | 0.2157 |
| Conditional ECC-200 raw | **162.763** | **65.426** | **10.835** | **89.17%** | 289.88 | **0.2117** |
| Conditional ECC reduced to 25 | 170.205 | 67.584 | 12.012 | 75.75% | 222.89 | 0.2287 |

Relative to frozen S2, the unreduced ECC challenger improved:

- WIS by 3.89%;
- energy score by 1.59%;
- variogram score by 3.74%; and
- ACF RMSE by 1.87%.

Its 89.17% coverage satisfies the 90% reliability band. However, WIS improvement remains below the 5% promotion target.

Development-only selection on 354 supported days in 2022 compared bandwidths 0.5, 1.0 and 2.0. Their WIS values tied at 158.663, so the declared energy-score and variogram-score tie breakers selected bandwidth 2.0. Reductions to 25 and 50 paths both worsened WIS and coverage; the frozen configuration therefore retains all 200 paths. No 2023 observation was used for this selection.

The 25-path reduction destroyed the gains and sharply undercovered. Mean-path L1 reduction is therefore rejected for this use. Future reduction must explicitly preserve marginal tails, event duration and decision distortion, or retain more weighted paths.

## Rolling MPC and decision-focused learning

The new rolling harness executes only the next commitment block, carries state forward and calls the next vintage without exposing future realized values to the policy. CVaR and formal information-value calculations are implemented and tested.

An empirical Honda rolling-MPC value claim is not currently possible because the repository does not contain genuine sequential forecast vintages, partner-approved process states, recovery rules or validated Honda tariffs. Running it with invented values would contradict the project’s claim boundary.

Decision-focused or differentiable learning remains deferred until the same policies can be evaluated through a leakage-safe rolling benchmark. Offline reinforcement learning remains out of scope.

## Promotion verdict

| Component | Verdict | Reason |
|---|---|---|
| Regime mixture | Continue as challenger | Significant exploratory MAE improvement, but misses magnitude and export gates |
| Quantile LightGBM | Do not promote | Good coverage, weak WIS improvement and worse point forecast |
| Raw conditional ECC-200 | Continue as challenger | Development-selected and improves all principal scenario scores, but misses the 5% WIS gate |
| Reduced ECC-25 | Reject | Tail collapse and worse proper/dependence scores |
| Rolling MPC | Engineering-ready, empirical evaluation blocked | No real sequential vintages and approved physical/commercial inputs |
| Decision-focused learning | Deferred | Rolling benchmark must exist first |

## Next development step

1. Improve the regime model through class-balanced or asymmetric export loss, using development data only.
2. Replace mean-path reduction with tail-constrained reduction; the development-selected configuration currently retains 200 paths.
3. Collect genuine sequential forecast vintages and approved physical/commercial inputs for rolling MPC.
4. Measure optimization runtime and decision distortion as scenario count changes.
5. Acquire forward data or partner forecast vintages for confirmatory promotion and rolling MPC.

No current Phase 1 challenger should replace the frozen champion yet.
