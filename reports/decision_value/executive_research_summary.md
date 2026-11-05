# Executive research summary

## What was tested

We compared frozen deterministic LightGBM scheduling with independent and whole-day coherent uncertainty under the same simulated ±5 kW / ±10 kWh deferrable process on 273 SPARK 2024 days.

## Main result

- Deterministic savings: **1.999617 EUR/day**.
- Independent-scenario savings: **1.800671 EUR/day**.
- Coherent-scenario savings: **1.747536 EUR/day**.
- Coherent-versus-independent paired cost effect: **-0.053135 EUR/day**, CI **[-0.109641, -0.011827]**.

## Interpretation

All schedules were generated before settlement and were physically energy-conserving. Physical-capacity and market assumptions must be considered alongside forecast sophistication. Results use ex-post-known wholesale prices and hypothetical flexibility, so they are not actual customer savings.

The zero-export cap was violated ex post on 82 deterministic, 16 independent-scenario and 11 coherent-scenario days. Those all-day savings are settlement diagnostics, not fully feasible operating value; common-feasible-day results are reported separately in the scientific report.

## Operational credibility

SPARK price publication timing was not encoded, so live implementability is not established. Honda could not be economically evaluated because no authoritative price series was available.

## What requires real-site validation

Equipment power/energy limits, recovery deadlines, connection contracts, actual procurement/export tariffs, production constraints and a prospective untouched evaluation period.
