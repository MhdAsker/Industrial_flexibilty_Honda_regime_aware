# Industrial Energy Research — Decision Brief

Audit date: 2026-11-05

## Where we stand

The research programme is substantially complete as an **exploratory, reproducible two-site forecasting study and one-site decision-value simulation**. Honda H1–H6, coherent-scenario Stage 1, physical-optimization Stage 2 and decision-value Stage 3 are all committed. SPARK's complete historical pipeline is accessible. The work is not operationally validated and is not yet a two-site economic study.

Current software verification is almost, but not fully, green: SPARK passes 104/104 tests; Honda passes 212/213. The single Honda failure is a stale Stage 2 path allow-list that rejects the later committed Stage 3 directories. No upstream scientific hash mismatch was found, but the regression should be repaired separately before release.

## What is established

- SPARK's empirical 75% raw-coverage threshold produces very small 15-minute aggregation error (MAE 0.0060 kW in its reconstruction study), while roughly 70% coverage is materially worse.
- Leakage-safe site-specific LightGBM beats strong historical baselines on both sites. SPARK 2024 MAE is 2.352 kW versus B3 2.710 and B5 2.571. Honda 2023 MAE is 71.513 kW versus B5 76.480, with a paired daily improvement of −5.122 kW (95% CI [−6.833, −3.514]).
- Honda's target is rigorously defined as signed net grid exchange, with positive import, negative export and primary labels restricted to original observations.
- Zero-shot TimesFM 3.0 and Chronos-2 are substantially worse than Honda LightGBM (95.431 and 103.672 kW MAE), although this does not test fine-tuning or all foundation models.
- Whole-day residual sampling better preserves temporal dependence than independent interval sampling, but its marginal calibration and rare-day support remain limited.

## Strongest scientific finding

**Better uncertainty representation did not automatically produce better decisions.** In the frozen Stage 3 SPARK simulation, deterministic LightGBM saved €1.999617/day, independent scenarios €1.800671 and coherent scenarios €1.747536 under the same hypothetical ±5 kW/±10 kWh process buffer. Coherent scenarios cost €0.053135/day more than independent scenarios (95% CI [€0.011827, €0.109641] in that direction) and €0.252081/day more than deterministic scheduling. Meanwhile, changing physical capacity moved coherent savings from €0.740766 to €3.057647/day. This is a valuable falsification result, not evidence that probabilistic forecasting is generally useless.

## Biggest weakness

The economic layer is not physically or operationally identified. Flexibility capacities and tariffs are hypothetical; SPARK prices are ex-post known because publication timestamps are absent; 2024 was previously exposed; and realized zero-export limits are violated on 82 deterministic, 16 independent and 11 coherent days. Honda economics were not run because no authoritative 2023 price/timing series was accepted. Therefore simulated savings are counterfactual wholesale-cost effects, not customer bill savings or trading P&L.

## What remains missing

1. Real process/asset power, energy, ramp and recovery constraints.
2. Actual tariff/connection terms and timestamped market-price availability.
3. An untouched prospective period or genuinely external site.
4. Honda economic data sufficient to test signed-exchange/export decisions.
5. Better tail/holiday and rare-DST scenario evidence.
6. A repaired all-green test suite.

## Are we ready to write a thesis?

**Yes.** The thesis should be framed as a rigorous, Category C exploratory study: provenance-safe forecasting, scenario dependence, physically explicit counterfactual optimization and a negative decision-value result. It should foreground the distinction between statistical forecast quality and operational value, and avoid operational savings claims.

## Do we need another experiment?

**Not another broad modelling experiment now.** The highest-value next evidence is data/validation, not model proliferation. Run a further experiment only if it adds at least one of: real asset constraints, defensible price/tariff timing, or a new untouched holdout. If Honda market data with publication metadata become available, the smallest useful extension is the already-frozen Honda export-aware decision study without retuning.

## What Perplexity should investigate

- Whether the integrated negative finding is novel relative to decision-focused energy forecasting literature.
- Mathematical conditions under which temporal dependence affects risk-neutral scheduling.
- Best-practice industrial deferrable-load formulations and minimum plant data for credibility.
- Authoritative German day-ahead publication timing and 2024 product resolution.
- Whether export-event probability is a valuable Honda research direction.
- Whether another experiment is necessary for an MSc thesis or paper.

## Decision after independent review

Choose among:

1. **Write now** with the current bounded contribution;
2. **Add one minimal confirmatory study** using real physics/timing and a new holdout;
3. **Add Honda economics only** if authoritative prices/timing can be secured;
4. **Stop further modelling** if literature shows the decision-value mechanism is already well established and the missing physical data cannot be obtained.

Do not default to more zero-shot models, fine-tuning, regime proliferation, thermal/BESS scope or larger parameter grids.

