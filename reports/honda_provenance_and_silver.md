# Honda H2 provenance recovery and observation-aware SILVER

## Executive summary

H2 recovered exact official issue intervals and a time-aware electrical source mapping without downloading the 103 GB full archive. It created conservative 15-minute provenance masks and an observation-aware SILVER layer with 210,336 rows. The primary rule admits only `OBSERVED_ORIGINAL`; deterministic corrections are sensitivity-only; imputed, unknown, and unresolved values are excluded.

The result is **H2 Category B**. Grid-exchange forecasting can proceed later using the primary mask, and PV/CHP can be studied as secondary masked targets. Gross electrical load is reconstructed for analysis but is not an eligible forecasting target because no independent total-load measurement was available. H2 trains no model and defines no train/validation/test split.

## H1 checkpoint

The H1 checkpoint `80b8e3033547341f0223d6ccd4e54da4de741909` was reproduced before H2. The repository was clean, the reduced archive SHA-256 remained `2a35a629fdf863793ab1a0c1bf80110ec7fc909ea65f716613a529da4539571f`, all 27 H1 manifest artifacts verified, and the 11 pre-H2 tests passed in 4.80 s. The finalized H1 report and manifest remain unchanged.

## Why reduced-data non-nullness is insufficient

The official processing pipeline detects issues, deletes invalid measurements, fills gaps, aligns clocks, resamples, and exports fully processed series. Therefore a non-null reduced value can be synthetic. The paper explicitly describes short-gap cloning from the immediately preceding interval and long-gap cloning from an earlier corresponding weekly period. H1's nearly complete reduced tables consequently did not establish measured completeness.

## Official source acquisition

Dryad exposes `issues.zip` independently. H2 acquired and verified it (2,825,237 bytes; SHA-256 `594e09131fc797af0a0cfb4355fd2dabe95d88a72fd78e73dbb3017240452a85`) and the official release README (15,222 bytes; SHA-256 `7433ae4227aa9253c3a4c3f007486f5567d7fec2fdcf33b1600799217ff7cdee`). The paper HTML and hierarchy figures were also preserved. The full object inventory and checksums are in `reports/honda_source_acquisition.md` and `results/provenance/source_acquisition_inventory.csv`.

The 103,090,137,247-byte `data.zip` was not downloaded. Dryad provides no independently downloadable meter-level subset.

## Issue metadata

After excluding the two-record manual template, H2 parsed **148,968** official records from 87 YAML files: 148,958 automatic and 10 manual. Each record preserves issue ID, meter/reference sensor, measurement scope, UTC start/end, reason, correction method, parameters, source file, confidence, and manual/automatic origin.

Issue categories are: gap 142,207; zero 6,752; transformer-factor 4; single leap 3; lasting leap 1; errors 1. Exact affected intervals are recoverable. Every catalog record has an explicit mapped, non-target, or out-of-era disposition; no issue is silently omitted. Measurement-specific issues are conservatively propagated to aggregate `P` when they belong to a contributing source meter, so masks may over-exclude but do not knowingly under-exclude.

The YAML `correction` field identifies the pipeline operation. `applied_corrections` is empty in the supplied records, but the release README and paper state that issue corrections were applied to the corrected and reduced products. This discrepancy is retained in catalog notes rather than normalized away.

## Correction and filling taxonomy

The observed official methods and counts are: `interpolate` 126,935; `integrate_pw` 15,144; `delete` 6,755; `substitute_V.K21` 99; `substitute` 28; `transformer_factor` 4; `delete_range` 1; `pw_correction` 1; and `normalize_leap` 1.

`transformer_factor`, `normalize_leap`, clock alignment, and sign/unit harmonization are deterministic corrections. Resampling is observation-preserving when its inputs are observed. Deletion followed by reduced-series completion, interpolation/cloning, combined power-energy reconstruction, and substitution are classified as imputation. The paper makes historical-week cloning explicit and gives its deterministic donor-offset formula, but individual YAML records do not store the selected donor timestamps; the affected target timestamps are exact, while donor lineage is not.

## Meter hierarchy recovery

`meter_hierarchy_full.csv` contains 81 expanded electrical/thermal/weather meter records derived from the official Tables 2/3 and Figures 1/2. `(E)` entries are expanded into their Z and ZE meters. Known parent/child relationships are recorded with confidence; unspecified parents remain `unknown` and are not invented from correlations.

## Aggregate-to-source mapping

The exact documented sources are:

- grid total: `V.Z81`, `V.Z82`, old office meters `H2.Z35`/`H2.Z36`, then replacements `H2.Z351`/`H2.Z361`;
- PV: `V.Z84`, `H1.Z310`, `H2.Z311`, and `H3.Z312` according to Tables 2 and Figure 1;
- CHP: `H1.Z20`.

Table 8 prints `H1.Z311` and `H1.Z312` for two PV sources, conflicting with Tables 2 and Figure 1. H2 documents this as an apparent prefix typo and uses the consistent physical URNs `H2.Z311` and `H3.Z312`. `H1.ZE20` is a later redundant CHP meter and is not an independent source for the reduced aggregate. The exact duplicate H1 found between `electricity_P.CHP` and `heating_P.CHP_elec` therefore cannot be treated as independent validation.

## Electrical topology eras

Mappings change over time. Parking-lot PV began June 29, 2019 local time; rooftop PV groups 4–6 began June 25, 2020. Old office transformer meters ended September 9, 2020 at 12:00 UTC and replacements began September 15 at 10:00 UTC, leaving a partial-aggregate gap. The CHP control update is dated February 19, 2019 in Table 10; a usage-note paragraph says February 13, an official internal inconsistency retained as a limitation. Heating/CHP integration changed in June 2023.

The paper says Tixi clocks were unsynchronized until March 2020 without an exact day. H2 conservatively treats Tixi-fed grid/CHP values before April 1, 2020 UTC as `CORRECTED_NONIMPUTED`, making them sensitivity-only.

## Grid-exchange provenance

Percentages by local year (observed / imputed / corrected / unresolved) are:

| Year | Observed original | Imputed/filled | Corrected non-imputed | Missing unresolved |
|---|---:|---:|---:|---:|
| 2018 | 0.000% | 3.122% | 96.878% | 0.000% |
| 2019 | 0.000% | 8.228% | 91.772% | 0.000% |
| 2020 | 68.719% | 7.667% | 21.997% | 1.617% |
| 2021 | 97.834% | 2.166% | 0.000% | 0.000% |
| 2022 | 98.813% | 1.187% | 0.000% | 0.000% |
| 2023 | 97.491% | 2.509% | 0.000% | 0.000% |

Early years have no primary-eligible grid rows because conservative clock-correction evidence prevents calling them original observations; they remain sensitivity-eligible except where issue filling applies.

## PV provenance

| Year | Observed original | Imputed/filled | Corrected non-imputed | Structural absence |
|---|---:|---:|---:|---:|
| 2018 | 0.000% | 0.000% | 0.000% | 100.000% |
| 2019 | 50.819% | 0.151% | 0.000% | 49.030% |
| 2020 | 45.657% | 4.090% | 50.253% | 0.000% |
| 2021 | 86.946% | 2.272% | 10.782% | 0.000% |
| 2022 | 97.765% | 2.235% | 0.000% | 0.000% |
| 2023 | 95.240% | 4.760% | 0.000% | 0.000% |

Pre-commissioning PV is `STRUCTURAL_NOT_AVAILABLE` with documented physical generation zero, not sensor missingness. The first/final capacity phases are explicit.

## CHP provenance

| Year | Observed original | Imputed/filled | Corrected non-imputed |
|---|---:|---:|---:|
| 2018 | 0.000% | 2.106% | 97.894% |
| 2019 | 0.000% | 0.845% | 99.155% |
| 2020 | 66.342% | 8.786% | 24.872% |
| 2021 | 99.952% | 0.049% | 0.000% |
| 2022 | 99.969% | 0.031% | 0.000% |
| 2023 | 86.852% | 13.148% | 0.000% |

## Weather provenance

Weather issues are present in the complete issue catalog, but H2 does not create weather masks. Weather is not a target in this electrical-provenance stage, and the paper says extended weather gaps were intentionally not filled. A dedicated later feature-provenance stage should handle irradiance and ambient temperature.

## 15-minute aggregation semantics

The official February 2025 release states that corrections and alignment precede resampling and that non-cumulative 15-minute values are arithmetic means. H1 independently reproduced left-labelled UTC 15-minute means from the released one-minute data. H2 maps any inclusive issue interval to each half-open 15-minute bin `[t, t+15 min)` it overlaps.

This confirms reduced-resolution downsampling but does not independently reproduce raw → corrected → resampled processing; the source series needed for that test are inside `data.zip`. Details are in `results/provenance/aggregation_semantics.md`.

## Observation-aware quality masks

`data/silver/honda/provenance_masks_15min.parquet` contains 841,344 rows: one row for each of 210,336 timestamps and each of grid, PV, CHP, and reconstructed gross load. Classes remain distinct: `OBSERVED_ORIGINAL`, `CORRECTED_NONIMPUTED`, `IMPUTED_OR_FILLED`, `STRUCTURAL_NOT_AVAILABLE`, `MISSING_UNRESOLVED`, and `UNKNOWN_PROVENANCE`.

Primary eligibility is exactly `provenance_class == OBSERVED_ORIGINAL`. Sensitivity eligibility for measured targets is exactly `OBSERVED_ORIGINAL` or `CORRECTED_NONIMPUTED`. Imputed, unknown, and unresolved intervals are never eligible. Gross load is disabled for both policies pending independent validation.

## Gross-load reconstruction

The SILVER audit column uses the H1 source-sign equation:

`gross_load_reconstructed_kw = grid_exchange_kw_signed - pv_source_kw_signed - chp_source_kw_signed`

Equivalently it is signed grid exchange plus canonical positive PV and CHP generation. Gross provenance follows `IMPUTED` > `UNKNOWN` > `MISSING` > `CORRECTED` > `OBSERVED`; documented structural PV absence acts as a known zero.

## Independent gross-load validation

Independent validation was impossible: the reduced release has no independent gross-load channel and the meter-level transformer/generator series are only inside `data.zip`. MAE, RMSE, bias, median absolute error, P95 absolute error, relative error, coverage, and era-specific performance are therefore explicitly `unknown`, not zero. Gross load is useful only as an unvalidated descriptive reconstruction and is not scientifically approved as a future target.

## Trustworthy contiguous periods

Observed-only block summaries are:

| Signal | Blocks | Median | P90 | Longest |
|---|---:|---:|---:|---:|
| Grid exchange | 370 | 3.50 h | 15.675 h | 7,557.50 h |
| PV | 555 | 3.50 h | 63.30 h | 2,524.25 h |
| CHP | 1,128 | 0.25 h | 0.25 h | 6,596.75 h |

Sensitivity block statistics and start-year distributions are in `trustworthy_blocks.csv`. These results establish availability but do not define forecasting splits.

## Future modelling eligibility policy

For later work, observed-original rows may be targets and historical context. Corrected non-imputed rows are sensitivity targets and flagged predictors. Imputed predictors require a separate operational-availability/leakage policy and are never evaluation targets. Structural non-availability can be a known zero only when physically documented. Unresolved and unknown rows are excluded from modelling targets.

## Structural eras versus measurement issues

Commissioning, COVID lockdown, CHP control changes, office-meter replacement, and heating modernization are real regimes, not automatically missing data. Genuine low or zero operation remains observed unless an issue overlaps. The office replacement period is explicitly unresolved because the grid aggregate lacks two contributions; pre-PV is structural zero.

## Whether selective acquisition is sufficient

Selective metadata acquisition is sufficient to build conservative grid, PV, and CHP target masks, recover the published hierarchy, and construct SILVER. It is insufficient for raw-to-reduced numerical replication, independent gross-load validation, subsystem source analysis, or validating whether an unflagged anomaly escaped issue detection.

## Whether the 103 GB archive is necessary

- Grid-exchange forecasting: **no**, for a conservative primary target based on official issue metadata.
- PV forecasting: **no**, for the masked reduced PV target; yes only for independent meter-level replication.
- CHP forecasting: **no**, for the masked reduced CHP target; the same validation caveat applies.
- Gross-load forecasting: **yes or another independent source is required** before making it a scientific target.
- Subsystem forecasting: **yes**, because meter-level time series are in the monolithic archive.
- Regime analysis: reduced-data operational eras can be studied without it; meter/subsystem regimes need it.
- Flexibility optimization: descriptive aggregate exploration can proceed, but defensible device/subsystem constraints and independently validated balances need meter-level data.

## H2 sufficiency decision

**Category B.** The primary grid target can be masked rigorously enough for later selected-target forecasting, while independent gross-load validation and richer subsystem work require additional source data. Conservative masking controls known reduced-data target contamination; it cannot prove that all unflagged measurements were error-free.

## Recommended H3

H3 should be a target-design and evaluation-protocol study for signed grid exchange only: quantify forecast-horizon coverage over the observed-only blocks, design era-aware rolling-origin evaluation without choosing eras to maximize sample count, and define causal predictor-quality rules. It should not begin model training until that protocol is approved.

## Limitations

The masks trust completeness of official issue detection outside documented issues. Measurement-specific issues are deliberately over-propagated to aggregate `P`. Exact donor timestamps for cloned gaps are not stored per issue. The Tixi synchronization cutoff is month-level and treated conservatively. The paper contains PV-URN and CHP-date inconsistencies. No raw/intermediate meter values were acquired, so source comparisons and balance residual metrics remain unavailable.
