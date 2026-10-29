# Honda reduced dataset forensic audit (Prompt H1)

## Executive summary

The February 26, 2025 reduced Honda release is intact and internally coherent. It
contains 21 gzip-compressed CSV tables spanning operational years 2018–2023 at
1-minute, 15-minute, and 1-hour resolution. The data provide signed grid exchange,
aggregate PV and CHP generation, heating and cooling production, cooling-machine
electricity, and two onsite weather observations. Timestamps form duplicate-free,
regular UTC sequences; published coarser power/weather series are consistent with
left-labelled arithmetic means of the finer data.

The decisive limitation is provenance. The reduced files are aggregates of fully
processed measurements. The authors document deletion, correction, cloned gap
filling (including earlier-week substitution), power/energy reconstruction,
interpolation/forward fill, clock alignment, and downsampling. The reduced archive
contains no issue YAML, meter hierarchy, raw series, or per-value
observed/corrected/imputed mask. Nearly complete target columns therefore do not
mean nearly complete original observation. Rigorous forecast evaluation cannot
exclude reconstructed targets.

Decision: **Category C**. The reduced release alone is not sufficient for rigorous,
contamination-safe forecasting evaluation. It is sufficient for descriptive
analysis, exploratory multi-energy research, prototype pipelines, and hypothesis
generation if every result is labelled as using corrected/resampled data with
mixed or unknown provenance.

Recommended later primary target, once quality masks are obtained: signed site
grid exchange (`electricity_P.total`) at 15-minute resolution. Secondary targets:
PV generation, CHP state/output, and heating/cooling regimes. No model was trained.

## Dataset provenance

- DOI: `10.5061/dryad.73n5tb363`
- Version: February 26, 2025 (corrected downsampling release)
- Source: `data/raw/honda/reduced_data.zip`
- Bytes: `320157291`
- MD5: `8db608ae7efa029493e559d2a2fa5e98`
- SHA-256: `2a35a629fdf863793ab1a0c1bf80110ec7fc909ea65f716613a529da4539571f`
- ZIP integrity: passed (`ZipFile.testzip()` returned no bad entry)
- Source hash before/after extraction: unchanged
- Data license: CC0-1.0 on the Dryad record

The approximately 103 GB `data.zip` was not downloaded. The separate `issues.zip`
was also not used as a data source in H1. Authoritative external documentation is
listed in `docs/external_sources.md`.

## Reduced-release contents

The archive has 25 ZIP entries: four directories and 21 files. Each of the three
resolution directories contains seven CSV.GZ tables. There are 54 measurable
table-column instances in the signal catalogue (18 logical columns repeated at
three resolutions). The reduced release has three primary electrical aggregate
channels (`total`, `PV`, `CHP`), five thermal/coupled logical channels (heating
total, CHP heat, duplicated CHP electricity, cooling total, cooling-machine
electricity), and two weather variables.

Coverage is January 1, 2018 00:00 local winter time through January 1, 2024 00:00
local winter time, corresponding for the principal power tables to
2017-12-31 23:00 UTC through 2023-12-31 22:59/22:45/22:00 UTC. Operational years
are exactly **2018–2023**. Weather begins roughly one day earlier.

## Documentation and metadata

No documentation file is bundled inside `reduced_data.zip`. The official Dryad
README and the Scientific Data paper define the variables and processing. Detailed
notes are in `reports/honda_metadata_notes.md`.

The release is harmonized, corrected, aligned, resampled, and aggregated. It is
not a raw or observation-only release. February 2025 corrected the 15-minute and
hourly downsampling of non-cumulative measurements.

## Meter hierarchy

No meter hierarchy is recoverable from the reduced archive. It has no URNs,
parent IDs, transformer nodes, PV group IDs, or subsystem meters. The full release
documents 72 electrical meters and nine heat/cooling meters, but those cannot be
reconstructed from aggregate correlations without inventing relationships.
`meter_hierarchy.csv` records this explicit unavailability with high confidence.

## Electrical system architecture

Official documentation describes four grid transformers, a 749 kWp six-group PV
system installed in two phases, and a 199 kW electrical gas CHP coupled to the
heating system. The reduced release collapses this architecture to grid exchange,
PV, and CHP aggregates. Positive values represent inflow/consumption; negative
values represent outflow/production.

## Candidate site-demand signals

There is no direct gross site-demand channel in the reduced release. The supported
reconstruction hypothesis after PV commissioning is:

`GrossLoad = signed_grid_exchange - PV_signed - CHP_signed`

because PV and CHP production are negative. This is a useful derived research
quantity but is not independently validated against a direct gross meter and must
not silently become the canonical target.

## Grid import/export

`electricity_P.total` is the signed grid-exchange aggregate described by Dryad as
electricity drawn from the main grid. It contains negative intervals (3,291 hourly
samples), consistent with export. Import and export can be derived as positive and
negative parts, but no separate import/export measurements are supplied. The
signed channel is the most operationally interpretable direct electrical target.

## PV generation

`electricity_P.PV` uses negative generation sign. The first valid 15-minute value
is 2019-06-28 22:00 UTC. The paper documents the initial smaller installation in
June 2019 and the upgrade to the final 749 kWp capacity in June 2020. All six PV
groups are aggregated. At 15 minutes there are 52,220 pre-commissioning NaNs in a
single run; these are structural nonexistence, not missing measurements. The
observed 15-minute minimum is -610.7 kW and hourly minimum -560.5 kW.

Small positive PV values and negative values under an irradiance-below-1 W/m²
screen exist. These are flagged, not removed; a simple irradiance threshold is not
a solar-position proof and no arbitrary PV-quality threshold was created.

## CHP generation

`electricity_P.CHP` is negative during generation and is exactly duplicated by
`heating_P.CHP_elec` at all resolutions (zero residual across every overlapping
sample). The hourly range is -239.7 to +21.5 kW. Positive near-off readings and
outputs exceeding nominal rating at fine resolution are flagged for later
investigation, not cleaned.

The paper documents binary 0/100% control before February 13, 2019, 50–100%
modulation afterward, and heating/ventilation modernization in June 2023. CHP is
heat-led and dispatchable within thermal constraints; it is not classified as
renewable.

## Thermal system

The release includes total heat production, CHP heat production, total cooling
production, cooling-machine electrical consumption, and the duplicate CHP
electrical output. These support aggregate heating/cooling season and
thermal-electric coupling studies. They do not expose zone-level HVAC,
temperatures, volume flows, boilers, individual chillers, or ventilation meters.
The paper documents reconstruction of extended main cooling-meter failures, but
the reduced release does not identify the affected output intervals.

## Weather

Two onsite observed variables are present: mean global normal irradiance `Igm`
(10-minute moving average, W/m²) and ambient air temperature `Ta` (°C). Across the
hourly series their ranges are 0–990.36 W/m² and -11.20–39.35 °C. These are
observations, not forecasts, and must not later be used as target-time operational
forecast inputs without a genuine forecast-weather source.

## Timestamp and DST handling

All timestamp columns are named `datetime_utc`; official documentation explicitly
defines UTC. The stored strings lack an offset suffix and are therefore syntactically
naive, but their semantics are documented rather than guessed. All 21 tables are
monotonic, have zero duplicate timestamps, and have zero internal non-cadence steps.
The 2020 leap day is present.

UTC is continuous through DST. Conversion to documented `Europe/Berlin` local time
produces the normal missing spring clock hour and repeated autumn clock hour while
retaining distinct UTC instants. No destructive conversion was performed in H1.

## Resolutions and aggregation semantics

Actual resolutions are 1 minute, 15 minutes, and 1 hour. Cross-resolution tests
support arithmetic means over left-closed, left-labelled UTC intervals for all
non-cumulative power and weather signals. Relative mean absolute differences are
very small (generally numerical/rounding scale; the largest tested relative MAE is
below 4×10⁻⁶). DST does not change bin sizes because aggregation is in UTC.

Energy (`W`) columns are cumulative kWh series and should not be summed as interval
energy. The paper warns that finite differences of W and instantaneous P need not
match exactly because of sampling and correction mechanisms.

## Missingness

At 15 minutes, grid exchange, CHP, heating, CHP heat, cooling, and cooling electrical
input have 100% non-null coverage. This is not proof of fully observed data because
authors filled/corrected gaps. PV has 75.173% non-null coverage solely because of
its pre-commissioning period.

Weather retains gaps: `Igm` has 5,007 NaNs in 27 runs (97.621% non-null), `Ta` has
5,005 NaNs in 25 runs (97.622% non-null), and both have a longest 336.75-hour NaN
run. Year/month/hour/weekday patterns are in `missingness_patterns.csv`.

## Corrections and imputations

Author-corrected and reconstructed values are present. Documented methods include:

- deletion and offset correction;
- clone filling from the immediately preceding interval for short gaps;
- selection of an earlier corresponding interval in whole-week steps for longer gaps;
- joint power/energy gap reconstruction with scaling and offsets;
- linear interpolation and conditional forward fill during one-minute resampling;
- cooling main-meter reconstruction from submeters;
- conversion-factor and clock-drift correction;
- downsampling and calculation of dependent energy measurements.

Weather gaps are retained as NaN. For energy targets, filled intervals are not
reliably identifiable in the reduced release because it has no flags or issue
metadata. “Imputed” here includes cloned/reconstructed/filled values; it does not
assert that the authors use that exact label for every correction mechanism.

## Forecast-target contamination risk

The contamination risk is material. Weekly clone selection can insert historical
patterns into gaps, and power/energy correction uses replacement data. A model with
weekly lags could receive artificially favorable scores if evaluated on those
unmarked intervals. All candidate targets are therefore classified
`IMPUTED_NOT_TRACEABLE`, not because every value is imputed, but because affected
target timestamps cannot be reliably separated from observations.

## Physical plausibility

NaNs, infinities, sign counts, zeros, extrema, jumps, and constant runs were audited
without removal. No infinities or duplicate timestamps were found. Negative PV/CHP
is expected by convention; negative grid total denotes export. Small opposite-sign
generation readings, fine-resolution CHP extrema above nominal rating, PV nighttime
screen flags, and large cooling peaks require contextual investigation before any
cleaning policy is designed.

## Energy-balance validation

The cross-file identity `electricity_P.CHP = heating_P.CHP_elec` is exact at all
three resolutions and is strong internal validation. The gross-load equation is
sign-consistent and physically plausible, but no direct gross-load observation is
present, so residual statistics are unavailable. It is not strong enough to
declare reconstructed gross load canonical without full hierarchy evidence.

## Structural operating eras

Documented eras include CHP binary control through 2019-02-12; modulating CHP from
2019-02-13; initial PV from June 2019; final PV capacity from June 2020; broad
COVID/reduced operation; and June 2023 HVAC modernization. Four documented gateway
failures in 2020–2022 are also listed because their corrections are relevant to
quality provenance. No algorithmic change-point model was trained and no causal
meaning was attached to purely visual changes.

## Regime-support assessment

Directly supported: calendar, weekday/weekend, PV availability, CHP on/off, and
heating/cooling season. Metadata-supported: COVID/reduced activity and structural
eras. Inferable: high/low grid exchange, holiday/shutdown behavior, and grid
import/export. Subsystem-driven regimes are not supported by the reduced release.

## Subsystem interpretability

Site demand cannot be decomposed without double counting because parent/child
meters and subsystem channels are absent. The paper reports facility-level shares,
but H1 does not reproduce them from reduced data or treat them as measured reduced
channels. Subsystem regime interpretation requires the full meter-level release.

## Candidate forecasting targets

After quality masks are recovered, the recommended order is:

1. signed grid exchange / net-grid load;
2. PV generation with forecast weather;
3. CHP state/output conditioned on thermal information;
4. aggregate heating and cooling;
5. reconstructed gross load only after balance validation.

The scorecard records availability, quality, traceability, history,
interpretability, and suitability. No final modelling target is selected in H1.

## Candidate optimization targets

The dataset may later support multi-energy scheduling, PV self-consumption, CHP
coordination, and HVAC/flexibility studies. The reduced release lacks subsystem
constraints, controllability labels, storage state, tariffs, and quality masks;
therefore no flexibility or optimization experiment is justified yet.

## Recommended canonical resolution

**Primary: 15 minutes.** It preserves PV/grid/CHP dynamics relevant to energy
management while reducing 1-minute volume by 15×, and it matches common operational
decision intervals. Its semantics are empirically verified against 1-minute data.

**Secondary: 1 hour** for long-horizon baselines, structural-era summaries, and
cross-dataset comparisons. Retain 1 minute for diagnostic event studies and
resampling validation, not as the default modelling table.

## Whether reduced release is sufficient

**Category C.** It is not sufficient by itself for rigorous forecasting research
where target scores must be restricted to originally observed values. The reason
is traceability, not coverage or cadence. It remains sufficient for exploratory
and descriptive work that explicitly accepts corrected/resampled mixed provenance.

## What would require the full raw dataset

Rigorous target-quality masking requires the separate issue metadata and a mapping
from all contributing meters/corrections into each aggregate. Raw-versus-corrected
comparisons, meter hierarchy recovery, transformer balances, subsystem forecasts,
PV-group analysis, and validation of reconstructed gross load require meter-level
files from `data.zip`. The entire 103 GB archive is not scientifically necessary
if the authors/Dryad can provide the relevant contributing meter files separately;
with the current monolithic packaging, obtaining those files may require the full
download. H2 should first request or obtain `issues.zip` and a minimal extracted
subset rather than automatically downloading 103 GB.

## Proposed next Honda phases

- H2: obtain quality/issue provenance, map aggregate contributors, and build
  observation-aware SILVER/GOLD data at 15 minutes;
- H3: contamination-safe deterministic baselines for signed grid exchange;
- H4: deterministic ML only after chronological evaluation design is frozen;
- H5: probabilistic forecasting and calibration;
- H6: documented structural-era and CHP/PV regime analysis;
- H7: PV and net-grid-load forecasting using genuine forecast weather;
- H8: coherent multi-energy daily scenarios;
- H9: flexibility optimization only after controls/constraints are documented;
- H10: semantic, uncertainty-aware comparison with SPARK.

## Cross-dataset comparison potential with SPARK

A minimal semantic schema is proposed in
`docs/cross_dataset_schema_proposal.md`. Honda should retain signed grid exchange
and explicit provenance rather than being forced into SPARK names. Comparable
fields can be exposed as a view after direct versus reconstructed quantities are
clearly marked.

## Lessons transferred from SPARK

### TRANSFERRED

Immutable provenance, hashes, leakage safety, quality before modelling,
chronological evaluation later, deterministic tests, reproducible manifests,
honest uncertainty, explicit operational decision timing, and separation of site
demand, onsite generation, grid exchange, and net load.

### NOT TRANSFERRED

No 75% coverage rule, 5-second assumption, 180-sample aggregation rule, PV
threshold, weather list, holiday logic, issue time, split years, forecast features,
regime thresholds, flexibility percentage, ±2-hour shift, peak cap, or price-market
interpretation was reused.

### NEW HONDA-SPECIFIC ISSUES

Processed aggregate targets without correction masks; a strong but absent
meter hierarchy; phased 749 kWp PV commissioning; heat-led CHP with control and
HVAC changes; multiple onsite energy domains; cloned gap correction; reconstructed
cooling data; signed grid export; and six years spanning COVID-related operation.

## Limitations

H1 uses only the reduced archive for numerical evidence. No issue YAML or raw
meter data was inspected. A nighttime PV screen is not a solar-geometry test.
Structural era dates are documentation-based and sometimes only month-level.
Energy balance lacks an independent gross-load observation. No causal claim,
forecast, cluster, regime classifier, optimization, or final split was produced.
