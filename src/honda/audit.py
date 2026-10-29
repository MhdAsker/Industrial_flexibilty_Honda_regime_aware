"""Deterministic forensic audit of the reduced Honda energy release.

This module deliberately performs no imputation, cleaning, forecasting, model
training, clustering, regime classification, or optimisation. It reads the
immutable ZIP and extracted CSV.GZ files and writes descriptive audit artifacts.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/honda/reduced_data.zip"
STAGING = ROOT / "data/staging/honda/reduced_data"
OUT = ROOT / "results/audit"
FIG = OUT / "figures"
EXPECTED_SHA256 = "2a35a629fdf863793ab1a0c1bf80110ec7fc909ea65f716613a529da4539571f"

RESOLUTION_MINUTES = {"1min": 1, "15min": 15, "1h": 60}


@dataclass(frozen=True)
class Meta:
    quantity: str
    unit: str
    subsystem: str
    sign: str
    notes: str


METADATA: dict[tuple[str, str], Meta] = {
    ("electricity_P", "total"): Meta("grid electrical power", "W", "site grid connection", "positive import; negative export inferred from signed values", "Official README calls this power drawn from the main grid; it is not gross site demand."),
    ("electricity_P", "PV"): Meta("PV electrical power", "W", "photovoltaic", "generation is negative after harmonisation", "Aggregate PV channel."),
    ("electricity_P", "CHP"): Meta("CHP electrical power", "W", "CHP", "generation is negative after harmonisation", "Aggregate CHP electrical channel."),
    ("electricity_W", "total"): Meta("cumulative grid electrical energy", "kWh", "site grid connection", "increases for net import; may decrease during export", "Cumulative energy, not interval energy."),
    ("electricity_W", "PV"): Meta("cumulative PV electrical energy", "kWh", "photovoltaic", "decreases with generation", "Unavailable before first PV commissioning."),
    ("electricity_W", "CHP"): Meta("cumulative CHP electrical energy", "kWh", "CHP", "decreases with generation", "Cumulative generation under negative sign convention."),
    ("heating_P", "total"): Meta("total heat production power", "W", "heating", "production positive", "Aggregate heat production."),
    ("heating_P", "CHP_heat"): Meta("CHP heat production power", "W", "CHP/heating", "production positive", "CHP thermal output."),
    ("heating_P", "CHP_elec"): Meta("CHP electrical power", "W", "CHP", "generation is negative after harmonisation", "Duplicate representation of electricity_P.CHP."),
    ("heating_W", "total"): Meta("cumulative total heat energy", "kWh", "heating", "increases with production", "Cumulative heat production."),
    ("heating_W", "CHP_heat"): Meta("cumulative CHP heat energy", "kWh", "CHP/heating", "increases with production", "Cumulative CHP heat."),
    ("heating_W", "CHP_elec"): Meta("cumulative CHP electrical energy", "kWh", "CHP", "decreases with generation", "Duplicate representation of electricity_W.CHP."),
    ("cooling_P", "total"): Meta("total cooling production power", "W", "cooling", "production positive", "Aggregate output of cooling machines."),
    ("cooling_P", "cool_elec"): Meta("cooling-machine electrical consumption power", "W", "cooling", "consumption positive", "Electrical input to cooling machines."),
    ("cooling_W", "total"): Meta("cumulative cooling production energy", "kWh", "cooling", "increases with production", "Cumulative cooling output."),
    ("cooling_W", "cool_elec"): Meta("cumulative cooling-machine electrical energy", "kWh", "cooling", "increases with consumption", "Cumulative electrical input."),
    ("weather", "WeatherStation.Weather.Igm"): Meta("mean global normal irradiance (10 min moving average)", "W/m2", "onsite weather station", "non-negative", "Observed onsite weather, not a forecast."),
    ("weather", "WeatherStation.Weather.Ta"): Meta("ambient air temperature", "degC", "onsite weather station", "signed temperature", "Observed onsite weather, not a forecast."),
}


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(name: str, rows: Iterable[dict], columns: list[str] | None = None) -> None:
    rows = list(rows)
    if columns is None:
        columns = list(rows[0]) if rows else []
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table_key(path: Path) -> str:
    return path.name.removesuffix(".csv.gz")


def resolution_of(path: Path) -> str:
    return path.parent.name


def data_files() -> list[Path]:
    return sorted(STAGING.glob("*/*.csv.gz"), key=lambda p: (p.parent.name, p.name))


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["datetime_utc"], low_memory=False)


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def null_runs(mask: pd.Series) -> list[int]:
    if not mask.any():
        return []
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    return mask[mask].groupby(groups[mask]).size().astype(int).tolist()


def archive_inventory() -> None:
    rows = []
    with zipfile.ZipFile(SOURCE) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP entry: {bad}")
        for info in sorted(zf.infolist(), key=lambda x: x.filename):
            if info.is_dir():
                continue
            p = Path(info.filename)
            res = next((r for r in RESOLUTION_MINUTES if r in p.parts), "unknown")
            stem = p.name.lower()
            dtype = "weather" if "weather" in stem else "electricity" if "electricity" in stem else "heating" if "heating" in stem else "cooling" if "cooling" in stem else "unknown"
            rows.append({
                "archive_path": info.filename,
                "filename": p.name,
                "extension": ".csv.gz" if p.name.endswith(".csv.gz") else p.suffix,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "directory": str(p.parent).replace("\\", "/"),
                "suspected_resolution": res,
                "suspected_data_type": dtype,
            })
    write_csv("archive_inventory.csv", rows)


def extracted_manifest() -> None:
    rows = []
    for path in data_files():
        rows.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "MD5": digest(path, "md5"),
            "SHA256": digest(path, "sha256"),
            "file_type": "gzip-compressed CSV",
        })
    write_csv("extracted_file_manifest.csv", rows)


def signal_and_timestamp_audits() -> tuple[list[dict], list[dict], list[dict]]:
    signals: list[dict] = []
    times: list[dict] = []
    plausibility: list[dict] = []
    for path in data_files():
        df = load(path)
        t = df["datetime_utc"]
        res = resolution_of(path)
        expected_seconds = RESOLUTION_MINUTES[res] * 60
        deltas = t.diff().dt.total_seconds().dropna()
        gap_count = int((deltas > expected_seconds).sum())
        times.append({
            "dataset_file": path.relative_to(ROOT).as_posix(),
            "resolution": res,
            "timestamp_column": "datetime_utc",
            "timestamp_format": "YYYY-MM-DD HH:MM:SS",
            "documentation_timezone": "UTC",
            "timezone_aware_in_file": False,
            "first_timestamp": t.min().isoformat(sep=" "),
            "last_timestamp": t.max().isoformat(sep=" "),
            "row_count": len(df),
            "monotonic_increasing": bool(t.is_monotonic_increasing),
            "duplicate_timestamps": int(t.duplicated().sum()),
            "non_cadence_steps": int((deltas != expected_seconds).sum()),
            "gap_steps": gap_count,
            "minimum_step_seconds": float(deltas.min()) if len(deltas) else "unknown",
            "maximum_step_seconds": float(deltas.max()) if len(deltas) else "unknown",
            "dst_handling": "continuous UTC cadence; Europe/Berlin conversion yields normal spring skip and autumn repeated local hour",
            "leap_year_handling": "2020 leap day present" if ((t.dt.year == 2020) & (t.dt.month == 2) & (t.dt.day == 29)).any() else "not present",
            "notes": "Timestamp strings are naive syntactically, but column name and official documentation explicitly define UTC.",
        })
        key = table_key(path)
        for col in [c for c in df.columns if c != "datetime_utc"]:
            s = pd.to_numeric(df[col], errors="coerce")
            meta = METADATA.get((key, col), Meta("unknown", "unknown", "unknown", "unknown", "No supported mapping."))
            nonnull = s.dropna()
            signals.append({
                "dataset/file": path.relative_to(ROOT).as_posix(),
                "signal_name": col,
                "meter_id": "unavailable_in_reduced_release",
                "meter_name": "aggregate_only",
                "parent_meter": "unavailable_in_reduced_release",
                "subsystem": meta.subsystem,
                "physical_quantity": meta.quantity,
                "unit": meta.unit,
                "resolution": res,
                "start_timestamp": t.min().isoformat(sep=" "),
                "end_timestamp": t.max().isoformat(sep=" "),
                "row_count": len(s),
                "non_null_count": int(s.notna().sum()),
                "null_count": int(s.isna().sum()),
                "minimum": float(nonnull.min()) if len(nonnull) else "unknown",
                "maximum": float(nonnull.max()) if len(nonnull) else "unknown",
                "mean": float(nonnull.mean()) if len(nonnull) else "unknown",
                "median": float(nonnull.median()) if len(nonnull) else "unknown",
                "sign_convention": meta.sign,
                "quality_metadata_available": False,
                "suspected_imputation_metadata": "none in reduced release",
                "notes": meta.notes,
            })
            diff = s.diff().abs()
            longest_constant = max(null_runs(s.eq(s.shift()) & s.notna()) or [0])
            plausibility.append({
                "dataset_file": path.relative_to(ROOT).as_posix(),
                "signal_name": col,
                "resolution": res,
                "nan_count": int(s.isna().sum()),
                "infinity_count": int(np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).sum()),
                "negative_count": int((s < 0).sum()),
                "zero_count": int((s == 0).sum()),
                "p01": float(nonnull.quantile(0.01)) if len(nonnull) else "unknown",
                "p99": float(nonnull.quantile(0.99)) if len(nonnull) else "unknown",
                "maximum_absolute_step": float(diff.max()) if diff.notna().any() else "unknown",
                "longest_constant_run_samples": int(longest_constant),
                "interpretation": "flag only; no observations removed or modified",
            })
    write_csv("signal_catalog.csv", signals)
    write_csv("timestamp_audit.csv", times)
    write_csv("physical_plausibility_audit.csv", plausibility)
    return signals, times, plausibility


def hierarchy_and_candidates(signals: list[dict]) -> None:
    hierarchy = [
        {"meter_id": "unavailable", "meter_name": "reduced aggregate release", "parent_meter_id": "unavailable", "hierarchy_level": "unavailable", "subsystem": "site aggregates only", "energy_domain": "electricity/heating/cooling/weather", "is_site_level": "partial aggregates", "is_generation": "mixed", "is_consumption": "mixed", "is_grid_exchange": "mixed", "confidence": "high", "source_of_mapping": "Archive inspection and official Dryad README: meter URNs/hierarchy exist only in full data.zip."}
    ]
    write_csv("meter_hierarchy.csv", hierarchy)
    candidates = [
        {"candidate_type": "A gross site electricity consumption", "meter/channel": "reconstructed: electricity_P.total - electricity_P.PV - electricity_P.CHP", "unit": "W", "sign_convention": "positive consumption", "coverage": "PV begins 2019-06-28 22:00 UTC; otherwise full", "resolution": "1min, 15min, 1h", "source_documentation": "Scientific Data Fig. 6 distinguishes total consumption from grid, PV, and CHP; reduced release has no direct gross channel", "measurement_status": "reconstructed hypothesis", "confidence": "medium"},
        {"candidate_type": "B grid import", "meter/channel": "electricity_P.total where value > 0", "unit": "W", "sign_convention": "positive", "coverage": "full six-year grid-exchange series", "resolution": "1min, 15min, 1h", "source_documentation": "Dryad README: electricity drawn from main grid; negative values show signed exchange", "measurement_status": "derived positive part of directly measured aggregate", "confidence": "medium"},
        {"candidate_type": "C grid export", "meter/channel": "-electricity_P.total where value < 0", "unit": "W", "sign_convention": "reported as positive export magnitude after derivation", "coverage": "full six-year grid-exchange series", "resolution": "1min, 15min, 1h", "source_documentation": "Empirical signed values; no separately documented export column", "measurement_status": "derived negative part of directly measured aggregate", "confidence": "medium"},
        {"candidate_type": "D PV production", "meter/channel": "electricity_P.PV", "unit": "W", "sign_convention": "negative generation", "coverage": "from 2019-06-28 22:00 UTC", "resolution": "1min, 15min, 1h", "source_documentation": "Dryad README and paper", "measurement_status": "aggregate of fully processed meters", "confidence": "high"},
        {"candidate_type": "E CHP electrical generation", "meter/channel": "electricity_P.CHP (duplicated as heating_P.CHP_elec)", "unit": "W", "sign_convention": "negative generation", "coverage": "full six years", "resolution": "1min, 15min, 1h", "source_documentation": "Dryad README and exact duplicate cross-file channel", "measurement_status": "aggregate of fully processed meters", "confidence": "high"},
        {"candidate_type": "F net grid load", "meter/channel": "electricity_P.total", "unit": "W", "sign_convention": "positive import, negative export", "coverage": "full six years", "resolution": "1min, 15min, 1h", "source_documentation": "Dryad README plus empirical sign", "measurement_status": "direct aggregate", "confidence": "high for grid exchange; medium for import/export semantics"},
        {"candidate_type": "G transformer-level totals", "meter/channel": "not present", "unit": "unknown", "sign_convention": "unknown", "coverage": "unavailable", "resolution": "unavailable", "source_documentation": "81-URN full archive only", "measurement_status": "unavailable in reduced release", "confidence": "high"},
    ]
    write_csv("electrical_target_candidates.csv", candidates)


def pv_chp_thermal_weather() -> None:
    pv_rows, chp_rows, thermal_rows, weather_rows = [], [], [], []
    for res in RESOLUTION_MINUTES:
        ep = load(STAGING / res / "electricity_P.csv.gz")
        t = ep["datetime_utc"]
        pv = ep["PV"]
        valid = pv.notna()
        daylight = pd.read_csv(STAGING / res / "weather.csv.gz", parse_dates=["datetime_utc"])
        joined = ep[["datetime_utc", "PV"]].merge(daylight[["datetime_utc", "WeatherStation.Weather.Igm"]], on="datetime_utc", how="left")
        night = joined["WeatherStation.Weather.Igm"].fillna(0) < 1
        pv_rows.append({
            "channel": "electricity_P.PV", "resolution": res,
            "documented_first_commissioning": "June 2019", "documented_final_capacity_upgrade": "June 2020",
            "first_valid_timestamp_utc": t[valid].min().isoformat(sep=" "), "last_valid_timestamp_utc": t[valid].max().isoformat(sep=" "),
            "unit": "W", "sign": "negative generation", "null_count": int(pv.isna().sum()), "zero_count": int((pv == 0).sum()),
            "positive_value_count": int((pv > 0).sum()), "negative_value_count": int((pv < 0).sum()),
            "minimum_W": float(pv.min()), "maximum_W": float(pv.max()),
            "nighttime_generation_below_minus_100W_count": int(((joined["PV"] < -100) & night).sum()),
            "multiple_groups": "yes, six groups documented; only aggregate present", "nominal_capacity": "six groups totalling 749 kWp documented in paper; group channels absent",
            "quality_status": "corrected/resampled mixed; original-vs-corrected mask unavailable",
        })
        chp = ep["CHP"]
        on = chp < -20_000
        runs_off = null_runs(~on)
        chp_rows.append({
            "channel": "electricity_P.CHP", "resolution": res, "unit": "W", "sign": "negative generation",
            "first_timestamp_utc": t.min().isoformat(sep=" "), "last_timestamp_utc": t.max().isoformat(sep=" "),
            "null_count": int(chp.isna().sum()), "minimum_W": float(chp.min()), "maximum_W": float(chp.max()),
            "on_samples_empirical_lt_minus_20kW": int(on.sum()), "longest_empirical_off_run_samples": max(runs_off or [0]),
            "dispatch_interpretation": "heat-led: paper states CHP runs above heat-demand threshold and with safe return temperatures",
            "documented_control_changes": "0/100% before 2019-02-13; 50-100% modulation thereafter; heating/ventilation modernization June 2023",
            "quality_status": "corrected/resampled mixed; original-vs-corrected mask unavailable",
        })
        for fname in ["heating_P.csv.gz", "cooling_P.csv.gz"]:
            df = load(STAGING / res / fname)
            key = table_key(STAGING / res / fname)
            for col in df.columns[1:]:
                m = METADATA[(key, col)]
                s = df[col]
                thermal_rows.append({"dataset_file": fname, "signal": col, "resolution": res, "physical_quantity": m.quantity, "unit": m.unit, "start_utc": df.datetime_utc.min().isoformat(sep=" "), "end_utc": df.datetime_utc.max().isoformat(sep=" "), "non_null": int(s.notna().sum()), "null": int(s.isna().sum()), "minimum": float(s.min()), "maximum": float(s.max()), "future_support": "thermal-electric coupling, CHP-state and heating/cooling regime analysis; no optimization performed"})
        w = load(STAGING / res / "weather.csv.gz")
        for col in w.columns[1:]:
            m = METADATA[("weather", col)]
            s = w[col]
            weather_rows.append({"variable": col, "description": m.quantity, "unit": m.unit, "resolution": res, "coverage_start_utc": w.datetime_utc.min().isoformat(sep=" "), "coverage_end_utc": w.datetime_utc.max().isoformat(sep=" "), "row_count": len(s), "non_null_count": int(s.notna().sum()), "null_count": int(s.isna().sum()), "minimum": float(s.min()), "maximum": float(s.max()), "weather_type": "observed onsite weather station", "forecast_safe_later": "no: target-time observations are not point-in-time forecasts"})
    write_csv("pv_audit.csv", pv_rows)
    write_csv("chp_audit.csv", chp_rows)
    write_csv("thermal_signal_summary.csv", thermal_rows)
    write_csv("weather_audit.csv", weather_rows)


def resolution_consistency() -> None:
    rows = []
    pairs = [("1min", "15min"), ("15min", "1h")]
    for fine_res, coarse_res in pairs:
        freq = f"{RESOLUTION_MINUTES[coarse_res]}min"
        for fname in ["electricity_P.csv.gz", "heating_P.csv.gz", "cooling_P.csv.gz", "weather.csv.gz"]:
            fine = load(STAGING / fine_res / fname).set_index("datetime_utc")
            published = load(STAGING / coarse_res / fname).set_index("datetime_utc")
            aggregate = fine.resample(freq, label="left", closed="left").mean()
            joined = aggregate.join(published, lsuffix="_aggregate", rsuffix="_published", how="inner")
            for col in published.columns:
                a, b = joined[f"{col}_aggregate"], joined[f"{col}_published"]
                mask = a.notna() & b.notna()
                diff = (a[mask] - b[mask]).abs()
                scale = b[mask].abs().mean()
                rows.append({
                    "source_resolution": fine_res, "published_resolution": coarse_res, "dataset_file": fname, "signal": col,
                    "tested_semantics": "arithmetic mean; left-labelled, left-closed UTC interval", "overlap_count": int(mask.sum()),
                    "mean_absolute_difference": float(diff.mean()) if len(diff) else "unknown",
                    "maximum_absolute_difference": float(diff.max()) if len(diff) else "unknown",
                    "relative_MAE_to_mean_absolute_signal": float(diff.mean() / scale) if len(diff) and scale else "unknown",
                    "conclusion": "consistent with mean aggregation" if len(diff) and (diff.mean() / max(scale, 1e-12)) < 1e-5 else "small discrepancy or missingness requires caution",
                    "dst_behavior": "aggregation performed in continuous UTC; no DST-sized bins",
                })
    write_csv("resolution_consistency.csv", rows)


def missingness() -> None:
    rows, patterns = [], []
    for fname, cols in [("electricity_P.csv.gz", ["total", "PV", "CHP"]), ("heating_P.csv.gz", ["total", "CHP_heat"]), ("cooling_P.csv.gz", ["total", "cool_elec"]), ("weather.csv.gz", ["WeatherStation.Weather.Igm", "WeatherStation.Weather.Ta"])]:
        path = STAGING / "15min" / fname
        df = load(path)
        for col in cols:
            mask = df[col].isna()
            runs = null_runs(mask)
            structural = col == "PV"
            rows.append({
                "dataset_file": fname, "signal": col, "resolution": "15min", "total_rows": len(df),
                "expected_rows_from_first_to_last": int((df.datetime_utc.max() - df.datetime_utc.min()).total_seconds() / 900) + 1,
                "non_null_fraction": float((~mask).mean()), "null_count": int(mask.sum()), "number_of_null_runs": len(runs),
                "longest_null_gap_samples": max(runs or [0]), "longest_null_gap_hours": max(runs or [0]) * 0.25,
                "median_null_gap_samples": percentile(runs, 50), "p95_null_gap_samples": percentile(runs, 95),
                "missingness_classification": "structural pre-commissioning, not measurement missingness" if structural else "remaining NaN; author corrections are not separately identifiable",
            })
            d = df.assign(missing=mask, year=df.datetime_utc.dt.year, month=df.datetime_utc.dt.month, hour=df.datetime_utc.dt.hour, weekday=df.datetime_utc.dt.weekday)
            for dimension in ["year", "month", "hour", "weekday"]:
                for value, grp in d.groupby(dimension):
                    patterns.append({"dataset_file": fname, "signal": col, "dimension": dimension, "value": int(value), "rows": len(grp), "missing_count": int(grp.missing.sum()), "missing_fraction": float(grp.missing.mean())})
    write_csv("missingness_summary.csv", rows)
    write_csv("missingness_patterns.csv", patterns)


def balance_and_integrity() -> None:
    rows = []
    for res in RESOLUTION_MINUTES:
        e = load(STAGING / res / "electricity_P.csv.gz")
        h = load(STAGING / res / "heating_P.csv.gz")
        x = e.merge(h[["datetime_utc", "CHP_elec"]], on="datetime_utc", how="inner")
        d = x["CHP"] - x["CHP_elec"]
        gross = x["total"] - x["PV"] - x["CHP"]
        rows.extend([
            {"resolution": res, "diagnostic": "duplicate CHP channel equality", "candidate_equation": "electricity_P.CHP = heating_P.CHP_elec", "unit": "W", "period_start_utc": x.datetime_utc.min().isoformat(sep=" "), "period_end_utc": x.datetime_utc.max().isoformat(sep=" "), "sample_count": int(d.notna().sum()), "residual_mean": float(d.mean()), "residual_MAE": float(d.abs().mean()), "residual_max_abs": float(d.abs().max()), "validation_strength": "strong exact cross-file identity"},
            {"resolution": res, "diagnostic": "gross-load reconstruction hypothesis", "candidate_equation": "GrossLoad = signed grid exchange - PV - CHP (PV/CHP negative)", "unit": "W", "period_start_utc": x.datetime_utc.min().isoformat(sep=" "), "period_end_utc": x.datetime_utc.max().isoformat(sep=" "), "sample_count": int(gross.notna().sum()), "residual_mean": "unavailable", "residual_MAE": "unavailable", "residual_max_abs": "unavailable", "validation_strength": "cannot be independently validated: reduced release has no direct gross-load channel"},
        ])
    write_csv("electrical_balance_diagnostics.csv", rows)
    correction = []
    for res in RESOLUTION_MINUTES:
        for domain in ["electricity", "heating", "cooling", "weather"]:
            correction.append({"domain": domain, "resolution": res, "raw_values_present": False, "harmonized": True, "corrected": domain != "weather", "gap_filled_or_reconstructed": domain != "weather", "resampled": True, "observed_vs_corrected_distinguishable": False, "imputed_intervals_identifiable": False, "quality_flag_granularity": "none in reduced release", "methods_documented": "yes at dataset/paper level; issue-specific YAML only in separate issues.zip", "historical_week_substitution": "documented clone selection for corrected energy/thermal/electrical gaps; weather gaps remain NaN", "traceability_conclusion": "processed aggregates without per-value provenance"})
    write_csv("correction_imputation_audit.csv", correction)
    integrity = [
        {"candidate_target": "signed grid exchange / grid import", "channel": "electricity_P.total", "classification": "IMPUTED_NOT_TRACEABLE", "contamination_risk": "high/unknown: corrected gap values cannot be masked; forecasting evaluation may reward reconstruction patterns", "reason": "fully processed aggregate; issue masks absent"},
        {"candidate_target": "reconstructed gross site load", "channel": "total - PV - CHP", "classification": "IMPUTED_NOT_TRACEABLE", "contamination_risk": "high/unknown and compounded across three processed inputs", "reason": "no directly measured gross-load target or provenance masks"},
        {"candidate_target": "PV generation", "channel": "electricity_P.PV", "classification": "IMPUTED_NOT_TRACEABLE", "contamination_risk": "high/unknown after commissioning; pre-commissioning NaNs are identifiable", "reason": "processed aggregate; correction intervals absent"},
        {"candidate_target": "CHP output/state", "channel": "electricity_P.CHP", "classification": "IMPUTED_NOT_TRACEABLE", "contamination_risk": "high/unknown; corrected values cannot be excluded", "reason": "processed aggregate; duplicate channel does not restore provenance"},
        {"candidate_target": "thermal production", "channel": "heating/cooling totals", "classification": "IMPUTED_NOT_TRACEABLE", "contamination_risk": "high/unknown, especially reconstructed cooling-meter failures", "reason": "paper documents reconstruction; masks absent"},
    ]
    write_csv("forecast_target_integrity.csv", integrity)


def eras_regimes_scorecards() -> None:
    eras = [
        {"start": "2018-01-01", "end": "2019-02-12", "era_name": "CHP binary control", "domain": "CHP", "evidence_type": "DOCUMENTED ERA", "evidence_source": "Scientific Data paper", "confidence": "high", "notes": "CHP operated at 0% or 100%."},
        {"start": "2019-02-13", "end": "2023-05-31", "era_name": "CHP 50-100% modulation", "domain": "CHP", "evidence_type": "DOCUMENTED ERA", "evidence_source": "Scientific Data paper", "confidence": "high", "notes": "Control logic update; integration still limited utilization."},
        {"start": "2019-06-01", "end": "2020-05-31", "era_name": "Initial smaller PV system", "domain": "PV", "evidence_type": "DOCUMENTED ERA", "evidence_source": "Scientific Data paper and first valid data 2019-06-28 UTC", "confidence": "high", "notes": "First PV installation; exact June commissioning day not stated in paper."},
        {"start": "2020-06-01", "end": "2023-12-31", "era_name": "Final PV capacity", "domain": "PV", "evidence_type": "DOCUMENTED ERA", "evidence_source": "Scientific Data paper", "confidence": "high", "notes": "PV upgrade to final capacity in June 2020."},
        {"start": "2020-03-01", "end": "2021-12-31", "era_name": "COVID/reduced-operation period", "domain": "site operation", "evidence_type": "DOCUMENTED ERA", "evidence_source": "Dataset abstract; exact operational boundaries not provided", "confidence": "medium", "notes": "Broad period only; do not treat dates as causal labels without facility records."},
        {"start": "2023-06-01", "end": "2023-12-31", "era_name": "Heating/ventilation modernization", "domain": "CHP/HVAC", "evidence_type": "DOCUMENTED ERA", "evidence_source": "Scientific Data paper", "confidence": "high", "notes": "Modernization improved CHP contribution; modulation scheme unchanged."},
        {"start": "2020-02-13", "end": "2020-03-06", "era_name": "Workshop gateway failure (corrected)", "domain": "submeters", "evidence_type": "DOCUMENTED ERA", "evidence_source": "Scientific Data paper", "confidence": "high", "notes": "Reduced aggregates do not retain interval-level correction flags."},
        {"start": "2020-08-20", "end": "2020-09-17", "era_name": "Emission-lab gateway failure (corrected)", "domain": "submeters", "evidence_type": "DOCUMENTED ERA", "evidence_source": "Scientific Data paper", "confidence": "high", "notes": "Reduced aggregates do not retain interval-level correction flags."},
        {"start": "2021-11-15", "end": "2021-12-10", "era_name": "Distribution gateway failure (corrected)", "domain": "submeters", "evidence_type": "DOCUMENTED ERA", "evidence_source": "Scientific Data paper", "confidence": "high", "notes": "Reduced aggregates do not retain interval-level correction flags."},
        {"start": "2022-05-06", "end": "2022-07-14", "era_name": "Workshop gateway failure (corrected)", "domain": "submeters", "evidence_type": "DOCUMENTED ERA", "evidence_source": "Scientific Data paper", "confidence": "high", "notes": "Reduced aggregates do not retain interval-level correction flags."},
    ]
    write_csv("structural_eras.csv", eras)
    regimes = [
        ("calendar regime", "SUPPORTED_DIRECTLY", "UTC timestamps convert to documented Europe/Berlin local time"),
        ("weekday/weekend", "SUPPORTED_DIRECTLY", "timestamp-derived"),
        ("holiday/shutdown", "INFERABLE", "no holiday or closure labels in reduced release"),
        ("COVID/reduced activity", "SUPPORTED_FROM_METADATA", "documented broadly, exact boundaries unavailable"),
        ("high/low site load", "INFERABLE", "signed grid exchange observed; gross load reconstructed only"),
        ("PV availability", "SUPPORTED_DIRECTLY", "PV and onsite irradiance observed"),
        ("CHP on/off", "SUPPORTED_DIRECTLY", "CHP electrical and thermal aggregate channels"),
        ("heating/cooling season", "SUPPORTED_DIRECTLY", "thermal power and temperature"),
        ("grid import/export", "INFERABLE", "signed total channel; no separate import/export columns"),
        ("subsystem-driven load", "NOT_SUPPORTED", "meter hierarchy and subsystem channels absent"),
        ("structural-era", "SUPPORTED_FROM_METADATA", "PV and CHP/HVAC changes documented"),
    ]
    write_csv("regime_support_matrix.csv", [{"regime_type": a, "support": b, "basis": c} for a, b, c in regimes])
    write_csv("subsystem_energy_summary.csv", [{"period": "2018-2023", "subsystem": "unavailable", "energy_domain": "electricity", "energy_kWh": "unknown", "share_of_site_energy": "unknown", "hierarchy_level_used": "none", "double_counting_check": "not applicable", "reason": "Reduced release contains site aggregates but no meter hierarchy or subsystem breakdown."}])
    targets = [
        ("T1", "gross site electricity-load forecasting", "MEDIUM", "LOW", "LOW", "HIGH", "MEDIUM", "MEDIUM", "LOW", "reconstructed target; correction masks absent"),
        ("T2", "grid-import forecasting", "HIGH", "MEDIUM", "LOW", "HIGH", "HIGH", "MEDIUM", "MEDIUM", "signed aggregate but author-filled intervals unmaskable"),
        ("T3", "PV forecasting", "HIGH", "MEDIUM", "LOW", "MEDIUM", "HIGH", "HIGH", "MEDIUM", "phased commissioning; observed weather is not forecast weather"),
        ("T4", "CHP-state/output forecasting", "HIGH", "MEDIUM", "LOW", "HIGH", "MEDIUM", "HIGH", "MEDIUM", "heat-led controls and structural changes; correction masks absent"),
        ("T5", "net-grid-load forecasting", "HIGH", "MEDIUM", "LOW", "HIGH", "HIGH", "HIGH", "HIGH", "best direct aggregate, but contamination cannot be masked"),
        ("T6", "multi-energy regime analysis", "HIGH", "MEDIUM", "LOW", "HIGH", "MEDIUM", "HIGH", "MEDIUM", "aggregates only; no meter hierarchy"),
        ("T7", "subsystem load forecasting", "LOW", "LOW", "LOW", "LOW", "LOW", "LOW", "LOW", "subsystem channels absent"),
        ("T8", "flexibility / energy-management study", "MEDIUM", "MEDIUM", "LOW", "MEDIUM", "LOW", "MEDIUM", "MEDIUM", "no direct controllability/constraint metadata; no optimization in H1"),
    ]
    cols = ["target_id", "research_target", "data_availability", "quality", "traceability", "history_length", "operational_interpretability", "forecasting_suitability", "regime_suitability", "optimization_suitability", "main_limitations"]
    write_csv("research_target_scorecard.csv", [dict(zip(cols, x)) for x in targets], cols)


def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    e = load(STAGING / "1h/electricity_P.csv.gz").set_index("datetime_utc")
    w = load(STAGING / "1h/weather.csv.gz").set_index("datetime_utc")

    monthly = e.resample("MS").mean() / 1000
    ax = monthly.plot(figsize=(11, 4), title="Honda reduced release: monthly mean signed electrical power\n(corrected/resampled; observed-vs-imputed status unknown)")
    ax.set_ylabel("kW (generation negative)"); ax.figure.tight_layout(); ax.figure.savefig(FIG / "electrical_overview.png", dpi=140); plt.close(ax.figure)

    availability = pd.DataFrame({c: e[c].notna().groupby(e.index.year).mean() for c in e.columns})
    ax = availability.plot.bar(figsize=(9, 4), ylim=(0, 1.05), title="Yearly value availability (NaN-based only; corrections not traceable)")
    ax.set_ylabel("non-null fraction"); ax.figure.tight_layout(); ax.figure.savefig(FIG / "yearly_availability.png", dpi=140); plt.close(ax.figure)

    pv = (-e.PV / 1000).clip(lower=0)
    ax = pv.groupby([e.index.year, e.index.month]).mean().unstack(0).plot(figsize=(10, 4), title="PV generation seasonal profile by year\n(corrected/resampled; provenance mixed/unknown)")
    ax.set_xlabel("month"); ax.set_ylabel("mean generation kW"); ax.figure.tight_layout(); ax.figure.savefig(FIG / "pv_by_year.png", dpi=140); plt.close(ax.figure)

    chp = (-e.CHP / 1000).clip(lower=0)
    ax = chp.groupby(e.index.year).mean().plot.bar(figsize=(8, 4), title="Mean CHP electrical generation by UTC year\n(corrected/resampled; provenance mixed/unknown)")
    ax.set_ylabel("kW"); ax.figure.tight_layout(); ax.figure.savefig(FIG / "chp_by_year.png", dpi=140); plt.close(ax.figure)

    ax = (e.total / 1000).groupby([e.index.month, e.index.hour]).mean().unstack(0).plot(figsize=(11, 4), legend=False, title="Signed grid-exchange hourly profiles by month\n(corrected/resampled; provenance mixed/unknown)")
    ax.set_xlabel("UTC hour"); ax.set_ylabel("kW"); ax.figure.tight_layout(); ax.figure.savefig(FIG / "grid_seasonal_profile.png", dpi=140); plt.close(ax.figure)

    miss = pd.DataFrame({"grid": e.total.isna(), "PV": e.PV.isna(), "CHP": e.CHP.isna(), "irradiance": w.iloc[:, 0].isna(), "temperature": w.iloc[:, 1].isna()})
    heat = miss.groupby([miss.index.year, miss.index.month]).mean().unstack(1)
    fig, ax = plt.subplots(figsize=(12, 4)); im=ax.imshow(heat.to_numpy(dtype=float), aspect="auto", interpolation="nearest"); fig.colorbar(im, ax=ax, label="NaN fraction"); ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index); ax.set_title("Monthly NaN patterns (does not reveal author-corrected intervals)"); fig.tight_layout(); fig.savefig(FIG / "missingness_heatmap.png", dpi=140); plt.close(fig)


def artifact_manifest() -> None:
    packages = {}
    for p in ["numpy", "pandas", "matplotlib", "pytest"]:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = "not installed"
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        git_hash = "uncommitted repository (no commit yet)"
    artifacts = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "honda_audit_manifest.json":
            artifacts[p.relative_to(ROOT).as_posix()] = {"bytes": p.stat().st_size, "sha256": digest(p, "sha256")}
    stat = SOURCE.stat()
    manifest = {
        "source_zip_filename": SOURCE.name,
        "source_zip_project_relative_path": SOURCE.relative_to(ROOT).as_posix(),
        "source_zip_bytes": stat.st_size,
        "source_zip_md5": digest(SOURCE, "md5"),
        "source_zip_sha256": digest(SOURCE, "sha256"),
        "source_zip_modification_timestamp": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "extraction_audit_execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": packages,
        "git_commit": git_hash,
        "audit_configuration": json.loads((ROOT / "config/honda_audit.json").read_text(encoding="utf-8")),
        "generated_artifacts": artifacts,
        "determinism_note": "CSV tables and figures are deterministic for unchanged inputs and package versions; execution timestamp is intentionally runtime-specific.",
    }
    (OUT / "honda_audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if digest(SOURCE, "sha256") != EXPECTED_SHA256:
        raise RuntimeError("Source SHA-256 does not match the official February 2025 reduced release")
    archive_inventory()
    extracted_manifest()
    signals, _, _ = signal_and_timestamp_audits()
    hierarchy_and_candidates(signals)
    pv_chp_thermal_weather()
    resolution_consistency()
    missingness()
    balance_and_integrity()
    eras_regimes_scorecards()
    figures()
    artifact_manifest()


if __name__ == "__main__":
    run()
