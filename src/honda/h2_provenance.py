"""H2 provenance recovery and observation-aware 15-minute SILVER construction.

The pipeline uses only the reduced release, the independently downloaded official
issue metadata, and immutable documentation from the associated paper. It never
downloads or opens the 103 GB full data archive and never trains a model.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/honda_h2.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RAW = ROOT / "data/raw/honda"
PROVENANCE_RAW = RAW / "provenance"
ISSUES_STAGING = ROOT / "data/staging/honda/provenance/issues"
REDUCED_15M = ROOT / "data/staging/honda/reduced_data/15min/electricity_P.csv.gz"
OUT = ROOT / "results/provenance"
FIG = OUT / "figures"
SILVER = ROOT / "data/silver/honda"

H1_COMMIT = CONFIG["h1_checkpoint"]
ISSUES_SHA256 = "594e09131fc797af0a0cfb4355fd2dabe95d88a72fd78e73dbb3017240452a85"

OBSERVED = "OBSERVED_ORIGINAL"
CORRECTED = "CORRECTED_NONIMPUTED"
IMPUTED = "IMPUTED_OR_FILLED"
STRUCTURAL = "STRUCTURAL_NOT_AVAILABLE"
MISSING = "MISSING_UNRESOLVED"
UNKNOWN = "UNKNOWN_PROVENANCE"

PRIORITY = {OBSERVED: 0, STRUCTURAL: 0, CORRECTED: 1, MISSING: 3, UNKNOWN: 4, IMPUTED: 5}

METHODS = {
    "integrate_pw": (IMPUTED, "Historical-clone power replacement jointly reconciled with cumulative energy."),
    "interpolate": (IMPUTED, "Honda issue correction name; paper documents immediate-prior or weekly historical cloning for gaps."),
    "delete": (IMPUTED, "Invalid observation deleted; the fully processed reduced value is subsequently resampled/interpolated."),
    "substitute_V.K21": (IMPUTED, "Main cooling meter reconstructed from downstream meters."),
    "substitute": (IMPUTED, "Measurement replaced from another documented source."),
    "transformer_factor": (CORRECTED, "Deterministic current-transformer scaling correction."),
    "delete_range": (IMPUTED, "Invalid range deleted before a separate gap-filling correction."),
    "pw_correction": (IMPUTED, "Joint power/energy reconstruction for an unusable interval."),
    "normalize_leap": (CORRECTED, "Deterministic offset correction after a lasting leap."),
}


def digest(path: Path, algorithm: str = "sha256") -> str:
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


def parse_time(value: object) -> pd.Timestamp:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.to_datetime(int(value), unit="s", utc=True)
    return pd.to_datetime(value, utc=True)


def overlapping_bin_slice(
    timestamps: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, bin_width: str = "15min"
) -> tuple[int, int]:
    """Return the half-open index slice for bins overlapping an inclusive issue interval."""
    values = timestamps.asi8
    width_ns = pd.Timedelta(bin_width).value
    first = int(np.searchsorted(values, start.value - width_ns, side="right"))
    last = int(np.searchsorted(values, end.value, side="right"))
    first = max(0, min(first, len(timestamps)))
    return first, max(first, min(last, len(timestamps)))


def combined_provenance(component_states: Iterable[str]) -> str:
    states = set(component_states)
    for state in [IMPUTED, UNKNOWN, MISSING, CORRECTED]:
        if state in states:
            return state
    return OBSERVED


def canonical_generation(source_signed_kw: pd.Series) -> pd.Series:
    return (-source_signed_kw).clip(lower=0)


def split_grid_exchange(source_signed_kw: pd.Series, tolerance_kw: float) -> tuple[np.ndarray, np.ndarray]:
    imports = np.where(source_signed_kw > tolerance_kw, source_signed_kw, 0.0)
    exports = np.where(source_signed_kw < -tolerance_kw, -source_signed_kw, 0.0)
    return imports, exports


def verify_inputs() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", H1_COMMIT, head],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("H1 checkpoint is not an ancestor of the current repository")
    if digest(RAW / "reduced_data.zip") != "2a35a629fdf863793ab1a0c1bf80110ec7fc909ea65f716613a529da4539571f":
        raise RuntimeError("H1 reduced archive changed")
    if digest(PROVENANCE_RAW / "issues.zip") != ISSUES_SHA256:
        raise RuntimeError("Official issues.zip checksum mismatch")
    with zipfile.ZipFile(PROVENANCE_RAW / "issues.zip") as zf:
        if zf.testzip() is not None:
            raise RuntimeError("issues.zip is corrupt")


def source_acquisition_inventory() -> None:
    official = [
        ("data.zip", 103_090_137_247, "a81cb72e4a9f5a2c8c0f10c2015bdc4b84f1945e605d5b9f7887ebeabcd711d2", "3910978", False, "Meter-level raw/intermediate validation; prohibited without explicit authorization."),
        ("issues.zip", 2_825_237, ISSUES_SHA256, "3910975", True, "Exact issue intervals and recommended correction methods."),
        ("issues.zip", 2_825_237, ISSUES_SHA256, "3910976", False, "Duplicate Dryad listing with identical digest; not downloaded twice."),
        ("plots.zip", 1_430_037, "bd1d095b45d24a2c9e93dd6d5acf625b8f494fe6ead892df106711bfe4ab8591", "3910977", False, "Not required for provenance recovery."),
        ("README.md", 15_222, "7433ae4227aa9253c3a4c3f007486f5567d7fec2fdcf33b1600799217ff7cdee", "3910982", True, "Official units, files, aggregate definitions, and release description."),
        ("reduced_data.zip", 320_157_291, "2a35a629fdf863793ab1a0c1bf80110ec7fc909ea65f716613a529da4539571f", "3910979", True, "Existing immutable H1 source; not newly downloaded in H2."),
    ]
    rows = []
    for filename, size, sha, file_id, required, acquired in official:
        local = PROVENANCE_RAW / filename if filename in {"issues.zip", "README.md"} else RAW / filename
        duplicate_listing = file_id == "3910976"
        exists = not duplicate_listing and local.exists() and local.stat().st_size == size and digest(local) == sha
        newly = filename in {"issues.zip", "README.md"} and exists
        rows.append({
            "filename": filename,
            "official_file_id": file_id,
            "source_url_or_doi": f"https://datadryad.org/downloads/file_stream/{file_id}",
            "description": acquired,
            "bytes": size,
            "official_sha256": sha,
            "independently_downloadable": True,
            "required": required,
            "acquisition_status": "duplicate_listing_not_downloaded" if duplicate_listing else "acquired_and_verified" if exists else "not_acquired",
            "local_path": local.relative_to(ROOT).as_posix() if exists else "",
            "local_md5": digest(local, "md5") if exists else "",
            "local_sha256": digest(local) if exists else "",
            "newly_acquired_in_h2": newly,
            "relationship_to_h1": "supplements reduced release" if filename in {"issues.zip", "README.md"} else "H1 reduced source" if filename == "reduced_data.zip" else "not used",
        })
    for filename, url, description in [
        ("scientific_data_article.html", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12103535/", "Associated paper HTML with machine-readable tables and methods."),
        ("paper_figures/Fig1_electrical_hierarchy.jpg", "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/205a/12103535/55efb32c4536/41597_2025_5186_Fig1_HTML.jpg", "Official electrical hierarchy figure."),
        ("paper_figures/Fig2_thermal_hierarchy.jpg", "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/205a/12103535/c842a427b1e2/41597_2025_5186_Fig2_HTML.jpg", "Official thermal hierarchy figure."),
    ]:
        local = PROVENANCE_RAW / filename
        rows.append({"filename": filename, "official_file_id": "associated_paper", "source_url_or_doi": url, "description": description, "bytes": local.stat().st_size, "official_sha256": "not separately published", "independently_downloadable": True, "required": True, "acquisition_status": "acquired_and_verified_by_https", "local_path": local.relative_to(ROOT).as_posix(), "local_md5": digest(local, "md5"), "local_sha256": digest(local), "newly_acquired_in_h2": True, "relationship_to_h1": "authoritative interpretation and hierarchy metadata"})
    write_csv("source_acquisition_inventory.csv", rows)


def parse_issue_catalog() -> pd.DataFrame:
    cached = OUT / "issue_catalog.csv"
    if cached.exists():
        # The immutable source checksum is verified before this function. Reusing
        # the deterministic catalog makes repeated full-pipeline verification fast.
        return pd.read_csv(cached, keep_default_na=False)
    rows: list[dict] = []
    files = sorted(p for p in ISSUES_STAGING.rglob("*.yaml") if p.name != "1_issues_manual_template.yaml")
    for path in files:
        source_kind = "manual" if path.parent.name.startswith("manual") else "automatic"
        meter_from_file = path.name.split("_issues_", 1)[0]
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for issue_id, item in data.items():
            reference = str(item.get("reference") or meter_from_file)
            measurement = "all_measurements"
            if reference.startswith(meter_from_file + "."):
                measurement = reference[len(meter_from_file) + 1 :]
            start, end = parse_time(item["time_start"]), parse_time(item["time_end"])
            method = str(item.get("correction") or "unknown")
            method_class, fill_method = METHODS.get(method, (UNKNOWN, "Undocumented correction method."))
            confidence = "HIGH" if reference == meter_from_file or reference.startswith(meter_from_file + ".") else "LOW"
            rows.append({
                "issue_id": str(issue_id), "meter_id": meter_from_file, "sensor_id": reference,
                "urn": meter_from_file, "measurement": measurement,
                "start_utc": start.isoformat(), "end_utc": end.isoformat(),
                "issue_type": str(item.get("reason") or "unknown"), "correction_type": method,
                "correction_class": method_class, "fill_method": fill_method,
                "manual_or_automatic": source_kind,
                "source_file": path.relative_to(ROOT).as_posix(), "confidence": confidence,
                "comment": str(item.get("comment") or ""),
                "correction_params_json": json.dumps(item.get("correction_params") or {}, sort_keys=True),
                "reference_mismatch": reference != meter_from_file and not reference.startswith(meter_from_file + "."),
                "notes": "applied_corrections is empty in the official files; correction records the pipeline method described by the authors.",
            })
    df = pd.DataFrame(rows).sort_values(["start_utc", "meter_id", "issue_id"], kind="mergesort").reset_index(drop=True)
    write_csv("issue_catalog.csv", df.to_dict("records"), list(df.columns))
    return df


def correction_method_catalog(issues: pd.DataFrame) -> None:
    counts = issues.groupby(["correction_type", "correction_class"]).size().to_dict()
    rows = []
    for method, (classification, semantics) in METHODS.items():
        rows.append({
            "correction_method": method, "classification": classification,
            "issue_record_count": int(sum(v for (m, _), v in counts.items() if m == method)),
            "actual_honda_semantics": semantics,
            "observation_replaced": classification == IMPUTED,
            "primary_target_eligible_when_affected": False,
            "sensitivity_target_eligible_when_affected": classification == CORRECTED,
        })
    rows.extend([
        {"correction_method": "time_alignment", "classification": CORRECTED, "issue_record_count": "global era", "actual_honda_semantics": "Documented gateway clock-drift curve applied before March 2020; exact transition day unavailable.", "observation_replaced": False, "primary_target_eligible_when_affected": False, "sensitivity_target_eligible_when_affected": True},
        {"correction_method": "harmonize_sign_unit", "classification": CORRECTED, "issue_record_count": "global processing", "actual_honda_semantics": "Deterministic sign, name, and unit harmonisation.", "observation_replaced": False, "primary_target_eligible_when_affected": False, "sensitivity_target_eligible_when_affected": True},
        {"correction_method": "resample_1min_then_mean", "classification": "OBSERVATION_PRESERVING", "issue_record_count": "global processing", "actual_honda_semantics": "One-minute aligned series followed by left-labelled arithmetic means for non-cumulative 15-minute data.", "observation_replaced": False, "primary_target_eligible_when_affected": True, "sensitivity_target_eligible_when_affected": True},
    ])
    write_csv("correction_method_catalog.csv", rows)


def paper_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = pd.read_html(PROVENANCE_RAW / "scientific_data_article.html")
    return tables[1], tables[2], tables[17]


def expand_urn(urn: str) -> list[str]:
    if "(E)" not in urn:
        return [urn]
    return [urn.replace("(E)", ""), urn.replace("(E)", "E")]


def meter_hierarchy() -> None:
    t1, t2, _ = paper_tables()
    rows = []
    categories = {"Electricity Main", "Local Generators", "Servers", "Cooling", "Heating", "Ventilation", "Workshops", "Emission Lab", "Offices & Distribution", "Design Studio", "Weather station"}
    subsystem = "unknown"
    parents = {
        "V.Z81": "external_grid", "V.Z82": "external_grid", "H2.Z35": "external_grid", "H2.Z36": "external_grid", "H2.Z351": "external_grid", "H2.Z361": "external_grid",
        "H1.Z20": "site_electrical_bus", "H1.ZE20": "site_electrical_bus", "V.Z84": "site_electrical_bus", "V.ZE84": "site_electrical_bus", "H1.Z310": "site_electrical_bus", "H2.Z311": "site_electrical_bus", "H3.Z312": "site_electrical_bus",
        "V.K21": "cooling_machines", "H1.W11": "heating_system", "H1.W12": "CHP", "H1.K12": "V.K21", "H1.K16": "V.K21", "H1.K11": "V.K21", "H1.K14": "V.K21", "H1.K15": "H1.K11", "H2.K21": "H1.K14",
    }
    for table in [t1, t2]:
        for record in table.to_dict("records"):
            urn = str(record["URN"])
            if urn in categories:
                subsystem = urn
                continue
            for expanded in expand_urn(urn):
                is_thermal = expanded.endswith(("K11", "K12", "K14", "K15", "K16", "K21", "W11", "W12"))
                domain = "weather" if expanded.startswith("WeatherStation") else "thermal" if is_thermal else "electricity"
                description = str(record["Description"])
                is_generation = any(x in description.lower() for x in ["pv", "chp production", "heat generation"])
                is_grid = "transformer" in description.lower()
                parent = parents.get(expanded, "unknown")
                rows.append({
                    "meter_id": expanded, "urn": expanded, "meter_type": record["Type"], "description": description,
                    "physical_location_or_subsystem": subsystem, "parent_meter_id": parent,
                    "hierarchy_level": 1 if is_grid else 2 if "feed" in description.lower() or "distribution" in description.lower() else 3,
                    "energy_domain": domain, "is_site_level": is_grid, "is_generation": is_generation,
                    "is_consumption": not is_generation and domain != "weather", "is_grid_exchange": is_grid,
                    "mapping_confidence": "HIGH" if parent != "unknown" else "MEDIUM",
                    "source_of_mapping": "Scientific Data Tables 2/3 and Figures 1/2; unknown parents are not inferred.",
                })
    write_csv("meter_hierarchy_full.csv", sorted(rows, key=lambda r: r["meter_id"]))


def aggregate_mapping() -> pd.DataFrame:
    start = "2017-12-31T23:00:00Z"; end = "2023-12-31T23:00:00Z"
    rows = []
    def add(signal: str, meter: str, relationship: str, valid_from: str, valid_to: str, confidence: str, source: str) -> None:
        rows.append({"reduced_signal": signal, "source_meter_id": meter, "source_urn": meter, "relationship": relationship, "aggregation_rule": "sum fully processed P measurements", "unit": "W", "sign": "positive inflow/consumption; negative outflow/production", "valid_from": valid_from, "valid_to": valid_to, "confidence": confidence, "metadata_source": source})
    for meter in ["V.Z81", "V.Z82"]:
        add("electricity_P.total", meter, "parallel site transformer contribution", start, end, "HIGH", "Scientific Data Table 8 and Figure 1")
    for meter in ["H2.Z35", "H2.Z36"]:
        add("electricity_P.total", meter, "old office transformer contribution", start, "2020-09-09T12:00:00Z", "HIGH", "Scientific Data Table 8 and Usage Notes")
    for meter in ["H2.Z351", "H2.Z361"]:
        add("electricity_P.total", meter, "replacement office transformer contribution", "2020-09-15T10:00:00Z", end, "HIGH", "Scientific Data Table 8 and Usage Notes")
    add("electricity_P.PV", "V.Z84", "parking-lot PV groups 1/2", "2019-06-28T22:00:00Z", end, "HIGH", "Scientific Data Tables 1/2/8 and Fig. 1")
    for meter, rel in [("H1.Z310", "rooftop PV west"), ("H2.Z311", "rooftop PV central"), ("H3.Z312", "rooftop PV east and groups 5/6")]:
        add("electricity_P.PV", meter, rel, "2020-06-25T00:00:00Z", end, "HIGH", "Tables 1/2 and Fig. 1; Table 8 H1.Z311/H1.Z312 prefixes are documented typographical inconsistencies")
    add("electricity_P.CHP", "H1.Z20", "CHP electrical production source", start, end, "HIGH", "Scientific Data Table 8; H1.ZE20 is redundant and not listed in the aggregate")
    df = pd.DataFrame(rows)
    write_csv("aggregate_source_mapping.csv", df.to_dict("records"), list(df.columns))
    return df


def topology_eras() -> None:
    rows = [
        {"start_utc": "2017-12-31T23:00:00Z", "end_utc": "2020-04-01T00:00:00Z", "era_name": "Tixi clock-drift correction era", "domain": "grid/CHP and other Tixi meters", "change_type": "timestamp alignment", "evidence": "Paper states Tixi synchronization began in March 2020; full month conservatively treated as corrected", "confidence": "MEDIUM"},
        {"start_utc": "2017-12-31T23:00:00Z", "end_utc": "2019-02-19T00:00:00Z", "era_name": "CHP binary control", "domain": "CHP", "change_type": "control", "evidence": "Scientific Data Table 10", "confidence": "HIGH"},
        {"start_utc": "2019-02-19T00:00:00Z", "end_utc": "2023-05-31T22:00:00Z", "era_name": "CHP 50-100% modulation", "domain": "CHP", "change_type": "control", "evidence": "Scientific Data Table 10 and Usage Notes", "confidence": "HIGH"},
        {"start_utc": "2019-06-28T22:00:00Z", "end_utc": "2020-06-25T00:00:00Z", "era_name": "PV phase 1", "domain": "PV", "change_type": "commissioning", "evidence": "June 29, 2019 local event and reduced data onset", "confidence": "HIGH"},
        {"start_utc": "2020-06-25T00:00:00Z", "end_utc": "2023-12-31T23:00:00Z", "era_name": "PV final capacity", "domain": "PV", "change_type": "commissioning", "evidence": "Scientific Data Table 10", "confidence": "HIGH"},
        {"start_utc": "2020-09-09T12:00:00Z", "end_utc": "2020-09-15T10:00:00Z", "era_name": "Office transformer meter replacement gap", "domain": "grid", "change_type": "meter replacement", "evidence": "Scientific Data Usage Notes", "confidence": "HIGH"},
        {"start_utc": "2023-06-01T00:00:00Z", "end_utc": "2023-12-31T23:00:00Z", "era_name": "Modernized heating/CHP integration", "domain": "CHP/HVAC", "change_type": "system modernization", "evidence": "Scientific Data Table 10 (month precision)", "confidence": "MEDIUM"},
    ]
    write_csv("electrical_topology_eras.csv", rows)


def active_sources(mapping: pd.DataFrame, signal: str, timestamp: pd.Timestamp) -> list[str]:
    part = mapping[mapping.reduced_signal == signal]
    result = []
    for r in part.itertuples():
        if parse_time(r.valid_from) <= timestamp < parse_time(r.valid_to):
            result.append(r.source_meter_id)
    return result


def build_masks(issues: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reduced = pd.read_csv(REDUCED_15M, parse_dates=["datetime_utc"])
    reduced["timestamp_utc"] = reduced["datetime_utc"].dt.tz_localize("UTC")
    ts = pd.DatetimeIndex(reduced["timestamp_utc"])
    n = len(ts)
    signals = {"grid_exchange": ("electricity_P.total", "total"), "PV": ("electricity_P.PV", "PV"), "CHP": ("electricity_P.CHP", "CHP")}
    states: dict[str, np.ndarray] = {k: np.full(n, OBSERVED, dtype=object) for k in signals}
    issue_ids = {k: np.full(n, "", dtype=object) for k in signals}
    issue_types = {k: np.full(n, "", dtype=object) for k in signals}
    methods = {k: np.full(n, "", dtype=object) for k in signals}
    confidences = {k: np.full(n, "HIGH", dtype=object) for k in signals}
    issue_counts = {k: np.zeros(n, dtype=np.int32) for k in signals}
    source_meters = {k: np.full(n, "", dtype=object) for k in signals}
    reduced_to_signal = {reduced_signal: signal for signal, (reduced_signal, _) in signals.items()}
    for source in mapping.itertuples(index=False):
        signal = reduced_to_signal[source.reduced_signal]
        active = (ts >= parse_time(source.valid_from)) & (ts < parse_time(source.valid_to))
        previous = source_meters[signal][active]
        source_meters[signal][active] = np.where(
            previous == "", source.source_meter_id, previous + ";" + source.source_meter_id
        )

    # Deterministic time alignment affects Tixi-fed office transformers and CHP.
    align_end = parse_time(CONFIG["tixi_unsynchronised_conservative_end_utc"])
    align_mask = ts < align_end
    for signal in ["grid_exchange", "CHP"]:
        states[signal][align_mask] = CORRECTED
        issue_ids[signal][align_mask] = "DOCUMENTED_TIME_ALIGNMENT"
        issue_types[signal][align_mask] = "clock_drift"
        methods[signal][align_mask] = "time_alignment"
        confidences[signal][align_mask] = "MEDIUM"

    pv_start = parse_time(CONFIG["pv_first_phase_documented_start_utc"])
    pv_struct = ts < pv_start
    states["PV"][pv_struct] = STRUCTURAL
    issue_ids["PV"][pv_struct] = "PV_NOT_COMMISSIONED"
    issue_types["PV"][pv_struct] = "structural_absence"
    methods["PV"][pv_struct] = "documented_zero_generation"

    mapping_audit: list[dict] = []
    target_meters = set(mapping.source_meter_id)
    routes_by_meter: dict[str, list[tuple[str, object]]] = {}
    for source in mapping.itertuples(index=False):
        routes_by_meter.setdefault(source.source_meter_id, []).append(
            (reduced_to_signal[source.reduced_signal], source)
        )
    for row in issues.itertuples(index=False):
        mapped_this_issue = False
        routes = routes_by_meter.get(row.meter_id, [])
        if routes:
            start, end = parse_time(row.start_utc), parse_time(row.end_utc)
        for signal, source in routes:
                clipped_start = max(start, parse_time(source.valid_from), ts[0])
                clipped_end = min(end, parse_time(source.valid_to), ts[-1] + pd.Timedelta("15min"))
                if clipped_end < clipped_start:
                    continue
                first, last = overlapping_bin_slice(ts, clipped_start, clipped_end)
                if last == first:
                    continue
                mapped_this_issue = True
                new_state = row.correction_class
                idx = np.arange(first, last)
                issue_counts[signal][idx] += 1
                replace = np.array([PRIORITY[new_state] > PRIORITY[x] for x in states[signal][idx]])
                chosen = idx[replace]
                states[signal][chosen] = new_state
                issue_ids[signal][chosen] = row.issue_id
                issue_types[signal][chosen] = row.issue_type
                methods[signal][chosen] = row.correction_type
                if row.measurement != "all_measurements":
                    confidences[signal][chosen] = "MEDIUM"
                mapping_audit.append({"issue_id": row.issue_id, "source_file": row.source_file, "signal": signal, "source_meter": row.meter_id, "disposition": "mapped_to_target", "mapped_interval_count": last - first, "first_mapped_timestamp_utc": ts[first].isoformat(), "last_mapped_timestamp_utc": ts[last - 1].isoformat(), "mapping_scope": "all source-meter issues conservatively affect aggregate P; measurement-specific issues may over-mask", "omitted": False})
        if not mapped_this_issue:
            disposition = "not_a_target_source_meter" if row.meter_id not in target_meters else "outside_source_active_period_or_reduced_time_range"
            mapping_audit.append({"issue_id": row.issue_id, "source_file": row.source_file, "signal": "", "source_meter": row.meter_id, "disposition": disposition, "mapped_interval_count": 0, "first_mapped_timestamp_utc": "", "last_mapped_timestamp_utc": "", "mapping_scope": "Every official issue receives an explicit disposition; no record is silently dropped.", "omitted": False})

    # Known topology gap: two office transformer contributions absent.
    gap_start = parse_time(CONFIG["office_transformers_old_end_utc"])
    gap_end = parse_time(CONFIG["office_transformers_new_start_utc"])
    gap = (ts >= gap_start) & (ts < gap_end)
    states["grid_exchange"][gap] = MISSING
    issue_ids["grid_exchange"][gap] = "OFFICE_TRANSFORMER_REPLACEMENT_GAP"
    issue_types["grid_exchange"][gap] = "structural_meter_gap"
    methods["grid_exchange"][gap] = "partial_aggregate_unresolved"
    confidences["grid_exchange"][gap] = "HIGH"

    # Existing NaNs after structural handling remain unresolved.
    for signal, (_, column) in signals.items():
        null = reduced[column].isna().to_numpy() & (states[signal] != STRUCTURAL)
        states[signal][null] = MISSING
        issue_ids[signal][null] = np.where(issue_ids[signal][null] == "", "REDUCED_VALUE_NAN", issue_ids[signal][null])
        issue_types[signal][null] = np.where(issue_types[signal][null] == "", "missing_value", issue_types[signal][null])
        methods[signal][null] = np.where(methods[signal][null] == "", "none", methods[signal][null])

    mask_rows = []
    for i, stamp in enumerate(ts):
        local = stamp.tz_convert("Europe/Berlin")
        for signal, (reduced_signal, _) in signals.items():
            state = states[signal][i]
            mask_rows.append({
                "timestamp_utc": stamp, "timestamp_local": local, "signal": signal,
                "provenance_class": state, "issue_id": issue_ids[signal][i], "issue_type": issue_types[signal][i],
                "correction_method": methods[signal][i], "source_meter": source_meters[signal][i],
                "issue_count": int(issue_counts[signal][i]), "confidence": confidences[signal][i],
                "eligible_primary_target": state == OBSERVED,
                "eligible_sensitivity_target": state in {OBSERVED, CORRECTED},
            })
    masks = pd.DataFrame(mask_rows)

    # Component-based gross provenance. Structural PV absence is a documented zero.
    gross_rows = []
    for i, stamp in enumerate(ts):
        comps = [states["grid_exchange"][i], states["PV"][i], states["CHP"][i]]
        state = combined_provenance(comps)
        gross_rows.append({"timestamp_utc": stamp, "timestamp_local": stamp.tz_convert("Europe/Berlin"), "signal": "gross_load_reconstructed", "provenance_class": state, "issue_id": ";".join(x for x in [issue_ids[s][i] for s in signals] if x), "issue_type": "component_combination", "correction_method": "conservative_component_precedence", "source_meter": "grid_exchange;PV;CHP", "issue_count": int(sum(issue_counts[s][i] for s in signals)), "confidence": "MEDIUM", "eligible_primary_target": False, "eligible_sensitivity_target": False})
    masks = pd.concat([masks, pd.DataFrame(gross_rows)], ignore_index=True)
    SILVER.mkdir(parents=True, exist_ok=True)
    masks.to_parquet(SILVER / "provenance_masks_15min.parquet", index=False)
    write_csv("issue_target_mapping_audit.csv", mapping_audit)

    # Canonical SILVER values.
    source_grid = reduced["total"] / 1000.0
    source_pv = reduced["PV"] / 1000.0
    source_chp = reduced["CHP"] / 1000.0
    tol = float(CONFIG["import_export_split_tolerance_kw"])
    silver = pd.DataFrame({"timestamp_utc": ts, "timestamp_local": ts.tz_convert("Europe/Berlin")})
    silver["grid_exchange_kw_signed"] = source_grid
    silver["grid_import_kw"], silver["grid_export_kw"] = split_grid_exchange(source_grid, tol)
    silver["pv_source_kw_signed"] = source_pv
    silver["chp_source_kw_signed"] = source_chp
    silver["pv_generation_kw"] = canonical_generation(source_pv)
    silver["chp_generation_kw"] = canonical_generation(source_chp)
    silver.loc[pv_struct, "pv_generation_kw"] = 0.0
    silver["pv_opposite_sign_flag"] = source_pv > 0
    silver["chp_opposite_sign_flag"] = source_chp > 0
    silver["gross_load_reconstructed_kw"] = source_grid - source_pv - source_chp
    for signal, prefix in [("grid_exchange", "grid_exchange"), ("PV", "pv"), ("CHP", "chp")]:
        silver[f"{prefix}_provenance"] = states[signal]
        silver[f"{prefix}_primary_eligible"] = states[signal] == OBSERVED
        silver[f"{prefix}_sensitivity_eligible"] = np.isin(states[signal], [OBSERVED, CORRECTED])
        silver[f"{prefix}_issue_flag"] = issue_ids[signal] != ""
    gross = masks[masks.signal == "gross_load_reconstructed"].reset_index(drop=True)
    silver["gross_load_provenance"] = gross.provenance_class
    silver["gross_load_primary_eligible"] = False
    silver["gross_load_sensitivity_eligible"] = False
    local = silver.timestamp_local
    silver["structural_era"] = np.select([local < pd.Timestamp("2020-03-16", tz="Europe/Berlin"), local <= pd.Timestamp("2021-01-17 23:59:59", tz="Europe/Berlin")], ["pre_COVID", "COVID_lockdown_documented"], default="post_lockdown")
    silver["pv_capacity_era"] = np.select([ts < pv_start, ts < parse_time(CONFIG["pv_final_phase_start_utc"])], ["pre_PV", "PV_phase_1"], default="PV_final_capacity")
    silver["chp_control_era"] = np.select([ts < pd.Timestamp("2019-02-19", tz="UTC"), ts < pd.Timestamp("2023-06-01", tz="UTC")], ["binary_0_100", "modulating_50_100"], default="post_HVAC_modernization")
    silver["dataset_id"] = "honda_dryad_73n5tb363_reduced_v2025-02-26_h2"
    silver.to_parquet(SILVER / "honda_energy_15min.parquet", index=False)
    return masks, silver, pd.DataFrame(mapping_audit)


def provenance_summaries(masks: pd.DataFrame) -> None:
    d = masks.copy()
    d["year"] = d.timestamp_local.dt.year
    d["month"] = d.timestamp_local.dt.strftime("%Y-%m")
    year = d.groupby(["signal", "year", "provenance_class"]).size().rename("interval_count").reset_index()
    year["fraction"] = year["interval_count"] / year.groupby(["signal", "year"])["interval_count"].transform("sum")
    month = d.groupby(["signal", "month", "provenance_class"]).size().rename("interval_count").reset_index()
    month["fraction"] = month["interval_count"] / month.groupby(["signal", "month"])["interval_count"].transform("sum")
    write_csv("provenance_by_year.csv", year.to_dict("records"), list(year.columns))
    write_csv("provenance_by_month.csv", month.to_dict("records"), list(month.columns))

    blocks = []
    for signal, group in d.groupby("signal", sort=True):
        group = group.sort_values("timestamp_utc").reset_index(drop=True)
        for mode, allowed in [("observed_only", {OBSERVED}), ("observed_plus_corrected", {OBSERVED, CORRECTED})]:
            good = group.provenance_class.isin(allowed)
            run_id = good.ne(good.shift(fill_value=False)).cumsum()
            runs = []
            years = Counter()
            for _, run in group[good].groupby(run_id[good]):
                duration_h = len(run) * 0.25
                runs.append(duration_h)
                years.update([int(run.timestamp_local.iloc[0].year)])
            blocks.append({"signal": signal, "eligibility_mode": mode, "number_of_blocks": len(runs), "median_duration_hours": float(np.median(runs)) if runs else 0.0, "p90_duration_hours": float(np.percentile(runs, 90)) if runs else 0.0, "longest_duration_hours": max(runs or [0.0]), "block_start_year_distribution": json.dumps(dict(sorted(years.items())))})
    write_csv("trustworthy_blocks.csv", blocks)


def policies_and_validation() -> None:
    rows = [
        (OBSERVED, "YES", "YES", "YES", "YES", "YES", "No documented issue correction/fill affects target interval."),
        (CORRECTED, "SENSITIVITY_ONLY", "NO", "YES_WITH_FLAG", "YES_WITH_FLAG", "YES", "Deterministic correction is traceable; excluded from primary scoring."),
        (IMPUTED, "NO", "NO", "LATER_POLICY_REQUIRED", "LATER_POLICY_REQUIRED", "YES_WITH_FLAG", "Copied/synthesized target may encode unavailable or future-pattern information."),
        (STRUCTURAL, "NO_AS_TARGET", "NO_AS_TARGET", "KNOWN_ZERO_IF_DOCUMENTED", "KNOWN_ZERO_IF_DOCUMENTED", "YES", "Physical nonexistence such as pre-commissioning PV is not sensor missingness."),
        (MISSING, "NO", "NO", "NO", "NO", "YES_WITH_FLAG", "No trustworthy target value."),
        (UNKNOWN, "NO", "NO", "NO", "NO", "DESCRIPTIVE_ONLY", "Provenance not demonstrated."),
    ]
    cols = ["provenance_class", "training_target", "validation_test_target", "lag_feature", "historical_context", "descriptive_analysis", "rationale"]
    write_csv("future_model_usage_policy.csv", [dict(zip(cols, x)) for x in rows], cols)
    write_csv("reduced_vs_source_validation.csv", [{"signal": "grid/PV/CHP", "validation_status": "NOT_PERFORMED", "source_data_available": False, "reason": "Relevant meter-level data are packaged only inside the prohibited 103,090,137,247-byte data.zip object.", "representative_periods": "unavailable", "result": "Issue-based provenance mapping created; raw-to-reduced numerical validation remains unresolved."}])
    write_csv("gross_load_validation.csv", [{"validation_status": "NOT_INDEPENDENTLY_VALIDATED", "formula": "grid_exchange_kw_signed - pv_source_kw_signed - chp_source_kw_signed", "MAE": "unknown", "RMSE": "unknown", "bias": "unknown", "median_absolute_error": "unknown", "P95_absolute_error": "unknown", "relative_error": "unknown", "coverage": "unknown", "performance_by_structural_era": "unknown", "reason": "Reduced release has no independent gross-load channel and source meter series are inside monolithic data.zip."}])
    (OUT / "aggregation_semantics.md").write_text(
        "# Fifteen-minute aggregation semantics\n\n"
        "The associated paper states that issue correction and clock alignment occur before one-minute resampling. "
        "The February 2025 release then forms non-cumulative 15-minute values as arithmetic means. H1 independently "
        "confirmed left-closed, left-labelled UTC arithmetic means from the released one-minute data.\n\n"
        "H2 could not independently reproduce raw/intermediate → corrected one-minute → reduced 15-minute processing "
        "because all meter-level series are inside the 103 GB monolithic `data.zip`. Issue intervals are mapped to every "
        "15-minute bin whose half-open interval overlaps the inclusive issue interval. Pre-April-2020 Tixi-fed grid/CHP "
        "values are conservatively classified as deterministic clock corrections because the exact March synchronization "
        "day is not documented. Therefore downsampling semantics are confirmed, while full raw-to-reduced semantics are not.\n",
        encoding="utf-8",
    )


def figures(masks: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    d = masks[masks.signal.isin(["grid_exchange", "PV", "CHP"])].copy()
    d["year"] = d.timestamp_local.dt.year
    order = [OBSERVED, CORRECTED, IMPUTED, STRUCTURAL, MISSING, UNKNOWN]
    colors = {OBSERVED: "#2ca02c", CORRECTED: "#1f77b4", IMPUTED: "#d62728", STRUCTURAL: "#bdbdbd", MISSING: "#ff7f0e", UNKNOWN: "#9467bd"}
    for signal in ["grid_exchange", "PV", "CHP"]:
        p = d[d.signal == signal].groupby(["year", "provenance_class"]).size().unstack(fill_value=0)
        p = p.div(p.sum(axis=1), axis=0).reindex(columns=order, fill_value=0)
        ax = p.plot.bar(stacked=True, color=[colors[c] for c in order], figsize=(9, 4), title=f"{signal}: 15-minute provenance composition by local year")
        ax.set_ylabel("fraction"); ax.legend(fontsize=7, loc="upper right"); ax.figure.tight_layout(); ax.figure.savefig(FIG / f"{signal}_provenance_by_year.png", dpi=140); plt.close(ax.figure)
    monthly = d[d.provenance_class == IMPUTED].groupby([d.timestamp_local.dt.to_period("M"), "signal"]).size().unstack(fill_value=0)
    ax = monthly.plot(figsize=(11, 4), title="Issue-filled 15-minute intervals by month (conservative source-meter mapping)")
    ax.set_ylabel("interval count"); ax.figure.tight_layout(); ax.figure.savefig(FIG / "issue_fill_timeline.png", dpi=140); plt.close(ax.figure)


def manifest() -> None:
    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "PyYAML", "matplotlib", "pytest"]:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "not installed"
    try: code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError: code_commit = "unavailable"
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        code_commit = f"uncommitted_worktree_based_on_{code_commit}"
    sources = {}
    for p in sorted(PROVENANCE_RAW.rglob("*")):
        if p.is_file():
            sources[p.relative_to(ROOT).as_posix()] = {"bytes": p.stat().st_size, "md5": digest(p, "md5"), "sha256": digest(p), "download_timestamp_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()}
    outputs = {}
    for base in [OUT, SILVER]:
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.name != "honda_h2_manifest.json":
                outputs[p.relative_to(ROOT).as_posix()] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    for p in [ROOT / "reports/honda_source_acquisition.md", ROOT / "reports/honda_provenance_and_silver.md"]:
        if p.is_file():
            outputs[p.relative_to(ROOT).as_posix()] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    data = {"h1_commit": H1_COMMIT, "h2_code_commit": code_commit, "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(), "package_versions": packages, "configuration_sha256": digest(CONFIG_PATH), "official_sources": sources, "official_urls_dois": ["https://doi.org/10.5061/dryad.73n5tb363", "https://doi.org/10.1038/s41597-025-05186-3", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12103535/"], "generated_outputs": outputs, "full_103gb_archive_downloaded": False, "determinism_note": "Tables and Parquet outputs are deterministic for identical immutable inputs/package versions; execution timestamps are runtime-specific."}
    (OUT / "honda_h2_manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True); SILVER.mkdir(parents=True, exist_ok=True)
    verify_inputs(); source_acquisition_inventory()
    issues = parse_issue_catalog(); correction_method_catalog(issues)
    meter_hierarchy(); mapping = aggregate_mapping(); topology_eras()
    masks, silver, _ = build_masks(issues, mapping)
    provenance_summaries(masks); policies_and_validation(); figures(masks); manifest()


if __name__ == "__main__":
    run()
