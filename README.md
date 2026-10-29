# Honda industrial energy dataset audit

This repository contains a forensic, pre-modelling audit of the reduced Honda R&D
Europe smart-building energy dataset, Dryad DOI
[`10.5061/dryad.73n5tb363`](https://doi.org/10.5061/dryad.73n5tb363).

Only the corrected February 26, 2025 reduced release (320,157,291 bytes) is used.
The approximately 103 GB full archive is not downloaded or used. The source ZIP is
immutable, extracted files are staging inputs, and no forecasting model,
clustering/regime model, optimization, GOLD data, or train/test split is created.

## Current conclusion

H1 concludes **Category C for rigorous forecasting evaluation using the reduced
release alone**. The aggregates are analytically useful, but author corrections
and cloned/reconstructed gaps cannot be identified per timestamp. Consequently,
corrected targets cannot be masked and contamination-safe forecast scoring cannot
be guaranteed. The reduced release remains useful for exploratory multi-energy
analysis and method prototyping with this limitation stated explicitly.

## Structure

- `config/`: audit configuration;
- `data/raw/honda/`: immutable ZIP and provenance note;
- `data/staging/honda/`: extracted source structure (ignored by Git);
- `src/honda/`: deterministic audit implementation;
- `tests/honda/`: provenance, semantics, and scope tests;
- `results/audit/`: generated audit tables, figures, and manifest;
- `reports/`: metadata notes and main audit report;
- `docs/`: sources and cross-dataset schema proposal.

## Reproduce

```powershell
python -m src.honda.audit
pytest
```

The audit refuses to run if the ZIP SHA-256 differs from the official February
2025 reduced release. Generated CSVs and figures are deterministic for unchanged
input and package versions; the manifest execution timestamp is intentionally
runtime-specific.
