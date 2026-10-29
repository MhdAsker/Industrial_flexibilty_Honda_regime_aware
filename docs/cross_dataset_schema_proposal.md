# Cross-dataset schema proposal

This is a semantic proposal, not a Honda GOLD schema. Fields must remain nullable
when Honda does not directly support them.

| Field | Unit/type | Honda mapping | Qualification |
|---|---|---|---|
| `timestamp_utc` | timezone-aware datetime | parse `datetime_utc` as UTC | Source strings are syntactically naive but explicitly documented as UTC. |
| `timestamp_local` | timezone-aware datetime | convert UTC to `Europe/Berlin` | Derived; preserve UTC as canonical. |
| `site_load_kw` | kW | `grid_exchange_kw - pv_generation_signed_kw - chp_generation_signed_kw` | Reconstructed hypothesis, not directly measured in reduced release. |
| `grid_exchange_kw` | kW | `electricity_P.total / 1000` | Direct processed aggregate; positive import, negative export. |
| `grid_import_kw` | kW | positive part of grid exchange | Derived. |
| `grid_export_kw` | kW | negative part magnitude | Derived; no separate channel. |
| `pv_generation_kw` | kW | `-electricity_P.PV / 1000` | Positive reporting convention derived from negative source sign. |
| `chp_generation_kw` | kW | `-electricity_P.CHP / 1000` | Positive reporting convention derived from negative source sign. |
| `heating_production_kw` | kW | `heating_P.total / 1000` | Direct processed aggregate. |
| `cooling_production_kw` | kW | `cooling_P.total / 1000` | Direct processed aggregate. |
| `temperature_c` | °C | `WeatherStation.Weather.Ta` | Onsite observation, not forecast weather. |
| `irradiance_w_m2` | W/m² | `WeatherStation.Weather.Igm` | Onsite 10-minute moving-average observation. |
| `quality_observed` | nullable boolean | unavailable | Must not be fabricated. |
| `quality_corrected` | nullable boolean | unavailable | Reduced release has no per-value mask. |
| `quality_imputed` | nullable boolean | unavailable | Reduced release has no per-value mask. |
| `dataset_id` | string | `honda_dryad_73n5tb363_reduced_v2025-02-26` | Provenance identifier. |

Honda names and signs should be preserved in SILVER; normalized positive generation
fields belong only in a documented comparison view.
