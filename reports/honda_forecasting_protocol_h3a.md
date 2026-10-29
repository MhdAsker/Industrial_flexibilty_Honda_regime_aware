# Honda forecasting-protocol evidence: H3A

## Executive summary

Signed 15-minute grid exchange is viable as the Honda primary forecasting candidate when primary targets are restricted to `OBSERVED_ORIGINAL`. It measures net exchange with the public grid: positive values are imports and negative values are exports. It is not gross industrial demand and negative values must remain intact.

The candidate 13:00 Europe/Berlin D-1 issuance is technically sound. It is cadence-aligned, unambiguous, and leaves an 11–34.75 hour ordinary-day interval-start horizon. DST produces the correct 92- and 100-interval target days and changes the maximum lead by one physical hour.

Availability supports later protocol design but is structurally uneven. Primary complete days are 187 in 2020, 342 in 2021, 356 in 2022, and 347 in 2023. H3B should carry P1 forward as the broader-regime primary candidate and P2 as a post-2020 structural-sensitivity candidate. This is not a frozen selection. No model, performance metric, fold, lead band, or final feature set is selected here.

## H2 checkpoint

H3A began from exact commit `2d29de9f45dcd414baf53dd3b3dabb4627798fe3`. The worktree was clean. Both H1 and H2 manifests validated. All 22 pre-H3A tests passed. The SILVER SHA-256 is `d9d93445391985faa7a0aa015cdb65b949f3909636d284c59449db876ae310df`; the provenance-mask SHA-256 is `f93d5bf4597d8f768cc63437f2ce78d21f4050f4dfcfbbe0e59d2fa0ef4eb778`.

## Primary target semantics

The candidate target is `grid_exchange_kw_signed` at 15-minute resolution. The Honda convention is positive grid import and negative grid export. No clipping, absolute-value transform, or gross-load reinterpretation is permitted.

The target differs fundamentally from SPARK's industrial-demand target. Honda grid exchange is a net boundary flow after on-site PV and CHP effects. The same underlying facility demand can produce different grid exchange when local generation changes, including negative export. Gross Honda demand remains an unvalidated reconstruction and is not an H3A target.

## Signed grid-exchange distribution

Only primary-eligible observations are summarized.

| Year | Count | Mean kW | Median kW | Std. kW | Min kW | Q05 kW | Q95 kW | Max kW | Import | Export |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 24,145 | 184.73 | 199.22 | 109.46 | -419.94 | -4.41 | 338.01 | 572.14 | 94.81% | 5.19% |
| 2021 | 34,281 | 131.18 | 138.96 | 119.28 | -506.43 | -104.24 | 287.57 | 481.57 | 89.61% | 10.39% |
| 2022 | 34,624 | 144.49 | 171.05 | 120.00 | -399.66 | -72.72 | 308.23 | 546.34 | 89.98% | 10.02% |
| 2023 | 34,161 | 119.34 | 126.47 | 122.19 | -419.69 | -84.52 | 294.38 | 655.49 | 86.43% | 13.57% |

Exact-zero incidence is zero in all four years. Extremes were retained. These statistics describe target scale and sign behavior; they do not select a metric.

## Candidate issuance time

The candidate task is: at 13:00 local on D-1, forecast every physical 15-minute signed-grid-exchange interval of local day D. 13:00 is aligned to the data cadence and is 12:00 UTC in winter or 11:00 UTC in summer. It is never a DST-ambiguous local time.

There is no measurement-availability obstacle. The time is also coherent for later analysis after the usual day-ahead auction process, but H3B must verify the precise market/gate/result convention for the intended operational use before freezing it. H3A does not silently substitute another issuance time.

## Measurement availability cutoff

The candidate cutoff is `interval_end_utc <= forecast_issue_time_utc`. A left-labelled 12:45 value covers 12:45–13:00 and becomes available at 13:00; a 13:00 value has not completed. The separate measurement-availability report gives winter, summer, and context examples.

## Physical local-day construction

UTC is the unique physical key and local timestamps carry their UTC offsets and fold values. Each year contains one 92-interval spring day, 363 or 364 ordinary 96-interval days, and one 100-interval autumn day. The repeated autumn 02:00 hour is retained twice with different offsets/folds; the nonexistent spring hour is not fabricated.

Interval-start lead ranges are:

| Day type | Intervals per day | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| Ordinary | 96 | 11.00 h | 22.875 h | 34.75 h |
| Spring DST | 92 | 11.00 h | 22.375 h | 33.75 h |
| Autumn DST | 100 | 11.00 h | 23.375 h | 35.75 h |

## Primary target availability

`complete_primary_day` is true only when every physical interval is `OBSERVED_ORIGINAL`; there is no percentage threshold.

| Year | Expected intervals | Primary | Sensitivity | Primary % | Complete primary days | Partial primary days | Zero-primary days |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 35,136 | 24,145 | 31,874 | 68.72% | 187 | 74 | 105 |
| 2021 | 35,040 | 34,281 | 34,281 | 97.83% | 342 | 23 | 0 |
| 2022 | 35,040 | 34,624 | 34,624 | 98.81% | 356 | 6 | 3 |
| 2023 | 35,040 | 34,161 | 34,161 | 97.49% | 347 | 13 | 5 |

Sensitivity eligibility remains separate and uses observed-original plus corrected-nonimputed only.

## Complete-day availability

| Year | Complete-day blocks | Median run | P90 run | Longest run | Longest start | Longest end |
|---|---:|---:|---:|---:|---|---|
| 2020 | 5 | 34.0 d | 55.4 d | 63 d | 2020-06-18 | 2020-08-19 |
| 2021 | 14 | 18.5 d | 33.6 d | 138 d | 2021-07-29 | 2021-12-13 |
| 2022 | 6 | 17.0 d | 159.0 d | 287 d | 2022-03-20 | 2022-12-31 |
| 2023 | 9 | 27.0 d | 95.2 d | 116 d | 2023-07-04 | 2023-10-27 |

Complete-profile analysis is feasible, but fragmentation differs materially by year.

## Context availability

Across all 1,461 origins, descriptive observed-original context availability is:

| Window | Median | P10 | Origins 100% | Origins ≥95% | Origins ≥90% | Origins ≥80% |
|---|---:|---:|---:|---:|---:|---:|
| 24 h | 100.00% | 87.50% | 84.39% | 85.69% | 89.39% | 90.49% |
| 48 h | 100.00% | 73.44% | 82.27% | 85.08% | 89.19% | 89.80% |
| 7 d | 100.00% | 68.01% | 72.62% | 81.52% | 84.94% | 88.57% |
| 14 d | 100.00% | 70.09% | 62.35% | 76.45% | 84.26% | 86.79% |
| 28 d | 99.96% | 78.79% | 46.61% | 74.74% | 79.95% | 88.30% |
| 56 d | 99.74% | 78.12% | 30.60% | 65.57% | 81.45% | 88.84% |

The low tails are driven mainly by 2020. H3A does not convert these descriptive percentages into a context-eligibility threshold and performs no filling.

## Holiday calendar

The official dataset places the facility in Offenbach am Main, which is in Hessen. H3A uses the ten annual statutory holidays in Hessisches Feiertagsgesetz §1 for 2020–2023, including Corpus Christi. It does not reuse Baden-Württemberg. The 40 exact dates and names are stored in `holiday_calendar.csv` with the location and legal-source URLs.

Bridge days are not created. A candidate H3B definition is a non-holiday Monday–Friday directly between a public holiday and a weekend, with adjacency evaluated on local calendar dates. H3B must decide whether to adopt or broaden that definition.

## Weather/PV/CHP leakage rules

Historical Ta, Igm, PV, CHP, and potentially thermal values are candidate-safe only for completed intervals ending no later than issuance and subject to their own provenance. Tomorrow's observed weather, measured PV, measured CHP, and realized heating/cooling are forbidden. Future forecasts or genuinely fixed control schedules require separate leakage-safe provenance and are later experiments, not H3A features.

Calendar fields, public holidays, DST state, and documented dates known by issuance are candidate future-safe. Existence in the released dataset does not imply operational availability.

## Structural-era policy

Documented engineering dates may later be features only when they were known by issuance. Retrospective processing eras and target-behavior regimes are not future features. All predeclared eras may be evaluation strata if they do not use TEST outcomes.

The June 2023 modernization is documented only to month precision, so it is allowed as an evaluation stratum but not yet an interval-level future-safe feature. The CHP update has an official February 13 versus February 19 inconsistency and also remains unresolved for H3B.

## 2020 assessment

2020 provides 24,145 primary intervals, 187 complete days, 74 partial days, and 105 zero-primary days. Its complete days form five blocks; the longest is 63 days. Only seven complete days occur before the final PV capacity date. The year spans the PV transition, pre-COVID and COVID periods, and the conservative Tixi correction boundary.

2020 is usable as training candidate data but not as a uniformly observed year. It broadens regime diversity and preserves evidence around major structural transitions. Risks are fragmentation, no primary targets during much of the early year, PV-capacity change, COVID conditions, and a distribution dominated by a different subset of dates. H3B must prevent sparse primary rows from being interpreted as continuous history.

## 2023 candidate TEST assessment

2023 has 34,161 primary intervals, 347 complete days, 13 partial days, and 5 zero-primary days. Imports account for 86.43% and exports for 13.57%. Before June modernization there are 145 complete days and 14,081 primary intervals; from June onward there are 202 complete days and 20,080 primary intervals.

Context is strong: median observed-original availability is 100% through 28 days and 99.96% at 56 days. The year contains both pre- and post-modernization operation and shows a lower annual mean, lower median, higher export share, and greater standard deviation than 2020. The paper also documents changed test-equipment use and CHP integration. These make 2023 a scientifically useful, challenging untouched TEST candidate rather than grounds to reject it. H3B should predeclare modernization strata and keep 2023 out of development.

## Candidate protocol P1

P1 availability is:

- Fold A train 2020: 24,145 primary intervals, 187 complete days; validation 2021: 34,281 and 342.
- Fold B train 2020–2021: 58,426 primary intervals, 529 complete days; validation 2022: 34,624 and 356.
- Final development/refit 2020–2022: 93,050 primary intervals, 885 complete days; candidate TEST 2023: 34,161 and 347.

P1 covers all seasons and more structural diversity, but it must handle 2020 fragmentation explicitly. H3A recommends that H3B evaluate P1 as the leading primary candidate, not that it freeze P1 now.

## Candidate protocol P2

P2 availability is:

- Train 2021: 34,281 primary intervals and 342 complete days; validation 2022: 34,624 and 356.
- Final development/refit 2021–2022: 68,905 primary intervals and 698 complete days; candidate TEST 2023: 34,161 and 347.

P2 has cleaner, nearly continuous context and stable final PV capacity. It omits the 2020 PV transition and early COVID evidence. H3B should retain it as a post-2020 structural-sensitivity candidate.

## Temporal distribution differences

From 2020 to 2023, the primary-target mean decreases from 184.73 to 119.34 kW, median from 199.22 to 126.47 kW, and import fraction from 94.81% to 86.43%; export fraction rises from 5.19% to 13.57%. Standard deviation rises from 109.46 to 122.19 kW. The 2020–2023 Wasserstein distance is 66.07 kW, larger than 2021–2022 (16.35 kW) and 2021–2023 (19.63 kW).

Complete-day mean medians decline from 171.51 kW in 2020 to 116.81 kW in 2023. Seasonal summaries show particularly low spring means and higher export fractions in 2021–2023. These are temporal distribution or structural differences, not a claim of concept drift or causality.

## Foundation-model context readiness

The data generally support 7-, 14-, and 28-day clean-history experiments: median observed-original context is approximately 100% for each. However, 100% availability falls from 72.62% of origins at 7 days to 46.61% at 28 days, and 2020 has a severe lower tail. Chronos/Chronos-2, TimesFM, and TimeGPT are therefore data-ready for later feasibility experiments only if H3B defines an explicit missing-context policy. H3A makes no API, architecture, or performance claim and runs no foundation model.

## Questions left for H3B

H3B still needs to freeze the primary versus sensitivity protocol, exact rolling-origin folds, context eligibility and missing-history handling, metric set, lead-time bands, baseline/model set, feature timing contracts, archived weather-forecast treatment, market timing source, CHP-date resolution, modernization boundary, bridge-day definition, subgroup reporting, and the treatment of partial target days.

## Recommendation to H3B

Carry forward signed 15-minute grid exchange at 13:00 D-1, with primary targets restricted to observed-original intervals. Evaluate P1 as the leading broad-regime design and P2 as the required post-2020 sensitivity. Preserve 2023 as untouched candidate TEST. Freeze nothing until H3B resolves the listed operational and evaluation choices.
