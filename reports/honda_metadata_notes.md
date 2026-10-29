# Honda reduced-release metadata notes

## What is bundled

The ZIP contains 21 gzip-compressed CSV files: seven files at each of `1min`,
`15min`, and `1h`. It contains no README, YAML, JSON, data dictionary, issue list,
meter URNs, or hierarchy file. Documentation therefore comes from the official
Dryad record and associated Scientific Data paper.

Each resolution contains:

- `electricity_P`: signed grid exchange (`total`), aggregate PV, aggregate CHP;
- `electricity_W`: cumulative versions of those electrical quantities;
- `heating_P` and `heating_W`: total heat, CHP heat, and duplicate CHP electricity;
- `cooling_P` and `cooling_W`: total cooling and cooling-machine electricity;
- `weather`: onsite mean global irradiance and ambient temperature.

## Processing terminology

The reduced release is not raw. The authors describe harmonization, issue
correction, time alignment, one-minute resampling, calculation of dependent
measurements, and aggregation. Non-cumulative 15-minute and hourly data are
downsampled means; the February 2025 release corrected that downsampling.

Documented corrections include deletion, offset correction, clone gap filling,
combined power/energy correction, cooling-meter reconstruction, conversion-factor
correction, and clock-drift alignment. Short cloned gaps copy the immediately
preceding period; longer gaps select a corresponding earlier period in whole-week
steps. Power/energy pairs use jointly scaled replacement series. Resampling uses
linear interpolation, with forward filling when the nearest measurement is more
than five minutes away. Weather gaps remain NaN.

The reduced files have no per-timestamp observed/corrected/imputed flag. The
separate `issues.zip` and full processed/raw files contain the traceability needed
to locate corrections, but they are outside H1's reduced-only data source.

## Signs, units, and timestamps

The documented harmonized sign convention is positive inflow/consumption and
negative outflow/production. This is observed in PV and CHP values and in negative
grid-exchange intervals. Power is W, cumulative energy is kWh, temperature is °C,
and irradiance is W/m².

The CSV timestamp strings have no offset suffix, but the column name is
`datetime_utc` and the official documentation explicitly defines them as UTC.
Local facility time is `Europe/Berlin`. Analysis preserves the strings and parses
them on the documented UTC basis without destructive rewriting.

## Metadata unavailable in the reduced release

- 72 individual electricity meters and nine thermal meters;
- parent/child hierarchy and subsystem URNs;
- separate transformer totals;
- PV group/meter channels;
- raw, harmonized, and corrected intermediate values;
- manual and automatic issue YAML;
- per-value or interval correction/imputation masks.

Unknowns are retained as unknown rather than inferred from correlations.
