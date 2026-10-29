"""H3A evidence generation for Honda forecasting-protocol feasibility.

This stage performs provenance, chronology, calendar, context, and distribution
analysis only. It does not train models, score forecasts, or freeze a protocol.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dateutil.easter import easter

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/honda_h3a_protocol_analysis.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SILVER_PATH = ROOT / "data/silver/honda/honda_energy_15min.parquet"
MASK_PATH = ROOT / "data/silver/honda/provenance_masks_15min.parquet"
ISSUES_PATH = ROOT / "data/raw/honda/provenance/issues.zip"
OUT = ROOT / "results/protocol_h3a"
REPORT = ROOT / "reports/honda_forecasting_protocol_h3a.md"
MEASUREMENT_REPORT = ROOT / "reports/h3a_measurement_availability.md"
TZ = CONFIG["timezone"]
H2_COMMIT = CONFIG["h2_checkpoint"]
OBSERVED = "OBSERVED_ORIGINAL"
CORRECTED = "CORRECTED_NONIMPUTED"
PRIMARY = {OBSERVED}
SENSITIVITY = {OBSERVED, CORRECTED}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(name: str, frame: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False, lineterminator="\n", float_format="%.10g")


def verify_inputs() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", H2_COMMIT, head], cwd=ROOT).returncode:
        raise RuntimeError("Verified H2 checkpoint is not an ancestor of HEAD")
    expected = {
        SILVER_PATH: "d9d93445391985faa7a0aa015cdb65b949f3909636d284c59449db876ae310df",
        MASK_PATH: "f93d5bf4597d8f768cc63437f2ce78d21f4050f4dfcfbbe0e59d2fa0ef4eb778",
        ISSUES_PATH: "594e09131fc797af0a0cfb4355fd2dabe95d88a72fd78e73dbb3017240452a85",
    }
    for path, wanted in expected.items():
        if digest(path) != wanted:
            raise RuntimeError(f"Input checksum mismatch: {path}")


def issue_time(target_day: date) -> pd.Timestamp:
    previous = target_day - timedelta(days=1)
    return pd.Timestamp(f"{previous.isoformat()} {CONFIG['candidate_issue_time_local']}", tz=TZ)


def historical_interval_available(interval_start_utc: pd.Timestamp, forecast_issue_time_utc: pd.Timestamp) -> bool:
    """A left-labelled 15-minute measurement is available only after its interval ends."""
    return interval_start_utc + pd.Timedelta("15min") <= forecast_issue_time_utc


def physical_day_grid(target_day: date) -> pd.DataFrame:
    start_local = pd.Timestamp(target_day, tz=TZ)
    end_local = pd.Timestamp(target_day + timedelta(days=1), tz=TZ)
    utc = pd.date_range(start_local.tz_convert("UTC"), end_local.tz_convert("UTC"), freq="15min", inclusive="left")
    local = utc.tz_convert(TZ)

    def offset_text(stamp: pd.Timestamp) -> str:
        minutes = int(stamp.utcoffset().total_seconds() // 60)
        sign = "+" if minutes >= 0 else "-"
        minutes = abs(minutes)
        return f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"

    return pd.DataFrame({
        "target_day_local": target_day.isoformat(),
        "target_timestamp_utc": utc,
        "target_timestamp_local": [x.isoformat() for x in local],
        "utc_offset": [offset_text(x) for x in local],
        "local_hour": local.hour,
        "local_quarter_hour": local.minute // 15,
        "dst_active": [bool(x.dst().total_seconds()) for x in local],
        "fold": [int(x.fold) for x in local],
    })


def all_target_intervals(silver: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = silver.set_index("timestamp_utc")
    holidays = holiday_calendar()
    holiday_dates = set(holidays.holiday_date)
    interval_frames: list[pd.DataFrame] = []
    origin_rows: list[dict] = []
    for stamp in pd.date_range(CONFIG["analysis_start_local_date"], CONFIG["analysis_end_local_date"], freq="D"):
        day = stamp.date()
        grid = physical_day_grid(day)
        issue_local = issue_time(day)
        issue_utc = issue_local.tz_convert("UTC")
        joined = indexed.reindex(pd.DatetimeIndex(grid.target_timestamp_utc))
        grid["grid_exchange_kw_signed"] = joined.grid_exchange_kw_signed.to_numpy()
        grid["grid_exchange_provenance"] = joined.grid_exchange_provenance.to_numpy()
        grid["primary_eligible"] = grid.grid_exchange_provenance.eq(OBSERVED)
        grid["sensitivity_eligible"] = grid.grid_exchange_provenance.isin(SENSITIVITY)
        grid["forecast_issue_time_local"] = issue_local.isoformat()
        grid["forecast_issue_time_utc"] = issue_utc.isoformat()
        grid["lead_time_hours"] = (pd.DatetimeIndex(grid.target_timestamp_utc) - issue_utc) / pd.Timedelta("1h")
        n = len(grid)
        spring, autumn = n == 92, n == 100
        day_text = day.isoformat()
        grid["spring_dst_day"] = spring
        grid["autumn_dst_day"] = autumn
        grid["year"] = day.year
        grid["weekday"] = day.weekday() < 5
        grid["weekend"] = day.weekday() >= 5
        grid["public_holiday"] = day_text in holiday_dates
        interval_frames.append(grid)
        primary = int(grid.primary_eligible.sum())
        sensitivity = int(grid.sensitivity_eligible.sum())
        era = era_labels(day)
        origin_rows.append({
            "target_day_local": day_text,
            "forecast_issue_time_local": issue_local.isoformat(),
            "forecast_issue_time_utc": issue_utc.isoformat(),
            "expected_target_intervals": n,
            "primary_eligible_targets": primary,
            "sensitivity_eligible_targets": sensitivity,
            "primary_eligible_fraction": primary / n,
            "sensitivity_eligible_fraction": sensitivity / n,
            "complete_primary_day": primary == n,
            "complete_sensitivity_day": sensitivity == n,
            "spring_dst_day": spring,
            "autumn_dst_day": autumn,
            "weekday": day.weekday() < 5,
            "weekend": day.weekday() >= 5,
            "public_holiday": day_text in holiday_dates,
            "pv_capacity_era": era["pv_capacity_era"],
            "covid_era": era["covid_era"],
            "chp_control_era": era["chp_control_era"],
            "modernization_era": era["modernization_era"],
            "year": day.year,
        })
    intervals = pd.concat(interval_frames, ignore_index=True)
    origins = pd.DataFrame(origin_rows)
    write_csv("lead_time_distribution.csv", intervals)
    write_csv("forecast_origins_candidate.csv", origins)
    return intervals, origins


def era_labels(day: date) -> dict[str, str]:
    covid_start = date.fromisoformat(CONFIG["covid_start_local_date"])
    covid_end = date.fromisoformat(CONFIG["covid_end_local_date"])
    pv_final_local = pd.Timestamp(CONFIG["pv_final_capacity_start_utc"]).tz_convert(TZ).date()
    modernization = date.fromisoformat(CONFIG["modernization_start_local_date"])
    return {
        "pv_capacity_era": "PV_phase_1" if day < pv_final_local else "PV_final_capacity",
        "covid_era": "pre_COVID" if day < covid_start else "COVID_reduced_operation" if day <= covid_end else "post_COVID",
        "chp_control_era": "modulating_50_100" if day < modernization else "post_heating_modernization",
        "modernization_era": "pre_modernization" if day < modernization else "post_modernization",
    }


def holiday_calendar() -> pd.DataFrame:
    fixed = [(1, 1, "New Year's Day"), (5, 1, "Labour Day"), (10, 3, "German Unity Day"), (12, 25, "Christmas Day"), (12, 26, "Second Day of Christmas")]
    rows = []
    for year in range(2020, 2024):
        easter_sunday = easter(year)
        variable = [
            (easter_sunday - timedelta(days=2), "Good Friday"),
            (easter_sunday + timedelta(days=1), "Easter Monday"),
            (easter_sunday + timedelta(days=39), "Ascension Day"),
            (easter_sunday + timedelta(days=50), "Whit Monday"),
            (easter_sunday + timedelta(days=60), "Corpus Christi"),
        ]
        dated = [(date(year, month, day), name) for month, day, name in fixed] + variable
        for holiday_date, name in sorted(dated):
            rows.append({
                "holiday_date": holiday_date.isoformat(), "holiday_name": name,
                "weekday": holiday_date.strftime("%A"), "federal_state": "Hessen", "region_code": "DE-HE",
                "facility_location_source": "Official Dryad README: Honda R&D Europe facility, Offenbach am Main",
                "holiday_source": "Hessisches Feiertagsgesetz §1",
                "holiday_source_url": "https://www.rv.hessenrecht.hessen.de/bshe/document/jlr-FeiertGHErahmen",
            })
    frame = pd.DataFrame(rows)
    write_csv("holiday_calendar.csv", frame)
    return frame


def target_summary(intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in intervals[intervals.primary_eligible].groupby("year"):
        x = group.grid_exchange_kw_signed.dropna()
        rows.append({
            "year": year, "target_count": len(x), "mean_kw": x.mean(), "std_kw": x.std(ddof=1),
            "minimum_kw": x.min(), "q01_kw": x.quantile(.01), "q05_kw": x.quantile(.05),
            "q25_kw": x.quantile(.25), "median_kw": x.median(), "q75_kw": x.quantile(.75),
            "q95_kw": x.quantile(.95), "q99_kw": x.quantile(.99), "maximum_kw": x.max(),
            "positive_import_fraction": (x > 0).mean(), "negative_export_fraction": (x < 0).mean(),
            "exact_zero_fraction": (x == 0).mean(),
        })
    frame = pd.DataFrame(rows)
    write_csv("grid_target_summary.csv", frame)
    return frame


def availability_by_year(origins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in origins.groupby("year"):
        expected = int(group.expected_target_intervals.sum())
        primary = int(group.primary_eligible_targets.sum())
        sensitivity = int(group.sensitivity_eligible_targets.sum())
        rows.append({
            "year": year, "expected_physical_target_intervals": expected,
            "primary_eligible_intervals": primary, "sensitivity_eligible_intervals": sensitivity,
            "primary_eligibility_fraction": primary / expected, "sensitivity_eligibility_fraction": sensitivity / expected,
            "local_days": len(group), "complete_primary_days": int(group.complete_primary_day.sum()),
            "complete_sensitivity_days": int(group.complete_sensitivity_day.sum()),
            "partial_primary_days": int(((group.primary_eligible_targets > 0) & ~group.complete_primary_day).sum()),
            "zero_primary_eligible_days": int((group.primary_eligible_targets == 0).sum()),
        })
    frame = pd.DataFrame(rows)
    write_csv("target_availability_by_year.csv", frame)
    return frame


def complete_day_blocks(origins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in origins.groupby("year"):
        dates = pd.to_datetime(group.target_day_local)
        good = group.complete_primary_day.to_numpy()
        runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
        start = None
        for stamp, ok in zip(dates, good):
            if ok and start is None: start = stamp
            if not ok and start is not None:
                end = stamp - pd.Timedelta(days=1); runs.append((start, end, (end - start).days + 1)); start = None
        if start is not None:
            end = dates.iloc[-1]; runs.append((start, end, (end - start).days + 1))
        lengths = [x[2] for x in runs]
        longest = max(runs, key=lambda x: x[2]) if runs else (pd.NaT, pd.NaT, 0)
        rows.append({
            "year": year, "contiguous_complete_day_blocks": len(runs),
            "median_block_length_days": float(np.median(lengths)) if lengths else 0,
            "p90_block_length_days": float(np.percentile(lengths, 90)) if lengths else 0,
            "longest_complete_day_run_days": longest[2],
            "longest_run_start_local": "" if pd.isna(longest[0]) else longest[0].date().isoformat(),
            "longest_run_end_local": "" if pd.isna(longest[1]) else longest[1].date().isoformat(),
        })
    frame = pd.DataFrame(rows)
    write_csv("complete_day_blocks.csv", frame)
    return frame


def longest_true_run(values: Iterable[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def context_availability(silver: pd.DataFrame, origins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = silver.set_index("timestamp_utc").grid_exchange_provenance
    rows = []
    for origin in origins.itertuples(index=False):
        issue = pd.Timestamp(origin.forecast_issue_time_utc)
        for hours in CONFIG["context_windows_hours"]:
            expected = pd.date_range(issue - pd.Timedelta(hours=hours), issue, freq="15min", inclusive="left")
            values = state.reindex(expected)
            observed = values.eq(OBSERVED)
            corrected = values.eq(CORRECTED)
            unavailable = ~(observed | corrected)
            rows.append({
                "target_day_local": origin.target_day_local, "forecast_issue_time_utc": origin.forecast_issue_time_utc,
                "year": origin.year, "context_hours": hours, "expected_intervals": len(expected),
                "observed_original_intervals": int(observed.sum()), "corrected_nonimputed_intervals": int(corrected.sum()),
                "unavailable_intervals": int(unavailable.sum()), "observed_original_fraction": observed.mean(),
                "observed_plus_corrected_fraction": (observed | corrected).mean(),
                "longest_unavailable_gap_intervals": longest_true_run(unavailable.to_numpy()),
                "longest_unavailable_gap_hours": longest_true_run(unavailable.to_numpy()) * .25,
                "latest_context_interval_start_utc": expected[-1].isoformat(),
                "latest_context_interval_end_utc": (expected[-1] + pd.Timedelta("15min")).isoformat(),
            })
    detail = pd.DataFrame(rows)
    write_csv("context_availability.csv", detail)
    summary_rows = []
    for (year, hours), group in detail.groupby(["year", "context_hours"]):
        x = group.observed_original_fraction
        summary_rows.append({
            "year": year, "context_hours": hours, "forecast_origins": len(group),
            "median_observed_original_fraction": x.median(), "p10_observed_original_fraction": x.quantile(.1),
            "p90_observed_original_fraction": x.quantile(.9), "fraction_origins_100pct_observed": (x == 1).mean(),
            "fraction_origins_ge_95pct_observed": (x >= .95).mean(), "fraction_origins_ge_90pct_observed": (x >= .90).mean(),
            "fraction_origins_ge_80pct_observed": (x >= .80).mean(),
            "median_observed_plus_corrected_fraction": group.observed_plus_corrected_fraction.median(),
        })
    summary = pd.DataFrame(summary_rows)
    write_csv("context_availability_summary.csv", summary)
    return detail, summary


def feature_policies() -> None:
    rows = [
        ("local calendar features", "target_time", "SAFE_CANDIDATE", "Known deterministically at issuance"),
        ("Hessen public holiday", "target_time", "SAFE_CANDIDATE", "Published statutory calendar"),
        ("known structural-era date", "target_time", "CONDITIONAL_CANDIDATE", "Only engineering dates known by issuance; no target-derived labels"),
        ("Ta ambient temperature", "historical_completed", "SAFE_CANDIDATE", "Only intervals ending at or before issuance"),
        ("Ta ambient temperature", "future_observed", "FORBIDDEN", "Realized target-time weather is unavailable at issuance"),
        ("Igm irradiance", "historical_completed", "SAFE_CANDIDATE", "Only intervals ending at or before issuance"),
        ("Igm irradiance", "future_observed", "FORBIDDEN", "Realized target-time weather is unavailable at issuance"),
        ("weather forecast", "future_forecast", "LATER_EXPERIMENT", "Requires archived operational forecasts and separate provenance"),
        ("PV generation", "historical_completed", "SAFE_CANDIDATE", "Measured interval must complete before issuance"),
        ("PV generation", "future_measured", "FORBIDDEN", "Direct component of future grid exchange"),
        ("PV forecast", "future_forecast", "LATER_EXPERIMENT", "Requires a leakage-safe forecast system"),
        ("CHP generation/state", "historical_completed", "SAFE_CANDIDATE", "Measured interval must complete before issuance"),
        ("CHP generation/state", "future_measured", "FORBIDDEN", "Future realized CHP is unavailable at issuance"),
        ("CHP forecast or known schedule", "future", "LATER_EXPERIMENT", "Requires a leakage-safe forecast or genuinely fixed schedule"),
        ("heating/cooling", "historical_completed", "SAFE_CANDIDATE", "Potentially safe after provenance and cutoff checks"),
        ("heating/cooling", "future_realized", "FORBIDDEN", "Do not assume future thermal operation is known"),
    ]
    write_csv("feature_availability_matrix.csv", pd.DataFrame(rows, columns=["feature", "time_relation", "h3a_classification", "rationale"]))

    eras = [
        ("Tixi clock-drift correction era", "RETROSPECTIVE_PROCESSING_ONLY", False, True, "Correction provenance, not an operational future feature"),
        ("CHP modulation update", "KNOWN_ENGINEERING_DATE", True, True, "Documented date; paper contains Feb 13/19 inconsistency to resolve in H3B"),
        ("PV phase 1 commissioning", "KNOWN_ENGINEERING_DATE", True, True, "Documented commissioning date"),
        ("PV final-capacity commissioning", "KNOWN_ENGINEERING_DATE", True, True, "Documented commissioning date"),
        ("COVID/reduced-operation period", "KNOWN_POLICY_DATE_AT_ISSUANCE", True, True, "Calendar/policy state, not a target-derived behavior label"),
        ("Office transformer replacement gap", "RETROSPECTIVE_PROVENANCE_EVENT", False, True, "Use for eligibility/audit, not as a predictive feature"),
        ("2023 heating/CHP modernization", "KNOWN_ENGINEERING_MONTH_ONLY", False, True, "Month precision is insufficient for a frozen interval-level feature"),
        ("target-behavior regime inferred after observation", "FUTURE_TARGET_INFERRED", False, False, "Would leak future outcome behavior"),
    ]
    write_csv("structural_era_feature_policy.csv", pd.DataFrame(eras, columns=["era_label", "knowledge_class", "candidate_future_safe_feature", "allowed_evaluation_stratum", "rationale"]))


def day_type_availability(origins: pd.DataFrame) -> pd.DataFrame:
    labels = np.select([origins.public_holiday, origins.weekend], ["public_holiday", "weekend"], default="weekday_nonholiday")
    d = origins.assign(day_type=labels)
    rows = []
    for (year, day_type), group in d.groupby(["year", "day_type"]):
        rows.append({
            "year": year, "day_type": day_type, "local_days": len(group),
            "expected_intervals": int(group.expected_target_intervals.sum()),
            "primary_eligible_intervals": int(group.primary_eligible_targets.sum()),
            "primary_eligible_fraction": group.primary_eligible_targets.sum() / group.expected_target_intervals.sum(),
            "complete_primary_days": int(group.complete_primary_day.sum()),
            "partial_primary_days": int(((group.primary_eligible_targets > 0) & ~group.complete_primary_day).sum()),
        })
    frame = pd.DataFrame(rows)
    write_csv("calendar_day_type_availability.csv", frame)
    return frame


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    values = np.sort(np.concatenate([a, b]))
    if len(values) < 2: return 0.0
    deltas = np.diff(values)
    a_cdf = np.searchsorted(np.sort(a), values[:-1], side="right") / len(a)
    b_cdf = np.searchsorted(np.sort(b), values[:-1], side="right") / len(b)
    return float(np.sum(np.abs(a_cdf - b_cdf) * deltas))


def distribution_shift(intervals: pd.DataFrame, origins: pd.DataFrame) -> pd.DataFrame:
    primary = intervals[intervals.primary_eligible].copy()
    primary["season"] = np.select(
        [primary.target_timestamp_utc.dt.month.isin([12, 1, 2]), primary.target_timestamp_utc.dt.month.isin([3, 4, 5]), primary.target_timestamp_utc.dt.month.isin([6, 7, 8])],
        ["winter", "spring", "summer"], default="autumn")
    rows = []
    yearly: dict[int, np.ndarray] = {}
    for year, group in primary.groupby("year"):
        x = group.grid_exchange_kw_signed.dropna(); yearly[int(year)] = x.to_numpy()
        rows.append({"record_type": "annual_interval", "year_a": year, **distribution_fields(x)})
    complete_dates = set(origins.loc[origins.complete_primary_day, "target_day_local"])
    complete = primary[primary.target_day_local.isin(complete_dates)]
    daily = complete.groupby(["year", "target_day_local"]).grid_exchange_kw_signed.mean().reset_index()
    for year, group in daily.groupby("year"):
        rows.append({"record_type": "complete_day_mean", "year_a": year, **distribution_fields(group.grid_exchange_kw_signed)})
    for (year, season), group in primary.groupby(["year", "season"]):
        rows.append({"record_type": "seasonal_interval", "year_a": year, "season": season, **distribution_fields(group.grid_exchange_kw_signed)})
    years = sorted(yearly)
    for i, a in enumerate(years):
        for b in years[i + 1:]:
            rows.append({"record_type": "pairwise_wasserstein", "year_a": a, "year_b": b, "wasserstein_distance_kw": wasserstein_1d(yearly[a], yearly[b])})
    frame = pd.DataFrame(rows)
    write_csv("yearly_distribution_shift.csv", frame)
    return frame


def distribution_fields(x: pd.Series) -> dict:
    return {
        "count": len(x), "mean_kw": x.mean(), "std_kw": x.std(ddof=1), "minimum_kw": x.min(),
        "q05_kw": x.quantile(.05), "q25_kw": x.quantile(.25), "median_kw": x.median(),
        "q75_kw": x.quantile(.75), "q95_kw": x.quantile(.95), "maximum_kw": x.max(),
        "import_fraction": (x > 0).mean(), "export_fraction": (x < 0).mean(), "zero_fraction": (x == 0).mean(),
    }


def era_availability(origins: pd.DataFrame) -> pd.DataFrame:
    d = origins.copy()
    d["analysis_era"] = np.select([
        pd.to_datetime(d.target_day_local) < pd.Timestamp("2020-06-25"),
        pd.to_datetime(d.target_day_local) <= pd.Timestamp(CONFIG["covid_end_local_date"]),
        pd.to_datetime(d.target_day_local) < pd.Timestamp(CONFIG["modernization_start_local_date"]),
    ], ["2020_pre_final_PV_capacity", "COVID_after_final_PV_capacity", "post_COVID_pre_modernization"], default="2023_post_modernization")
    rows = []
    for era, group in d.groupby("analysis_era"):
        rows.append({"analysis_era": era, "start_local": group.target_day_local.min(), "end_local": group.target_day_local.max(),
                     "days": len(group), "primary_eligible_intervals": int(group.primary_eligible_targets.sum()),
                     "complete_primary_days": int(group.complete_primary_day.sum()),
                     "partial_primary_days": int(((group.primary_eligible_targets > 0) & ~group.complete_primary_day).sum())})
    frame = pd.DataFrame(rows)
    write_csv("structural_era_target_availability.csv", frame)
    return frame


def protocol_analysis(intervals: pd.DataFrame, origins: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol, definitions in CONFIG["candidate_protocols"].items():
        for definition in definitions:
            years = definition["years"]
            od = origins[origins.year.isin(years)]
            iv = intervals[intervals.year.isin(years) & intervals.primary_eligible]
            context_part = context[context.year.isin(years) & context.context_hours.isin([168, 336, 672])]
            seasons = sorted(set(pd.to_datetime(od.target_day_local).dt.month.map(lambda m: "winter" if m in [12,1,2] else "spring" if m in [3,4,5] else "summer" if m in [6,7,8] else "autumn")))
            eras = sorted(set(od.pv_capacity_era) | set(od.covid_era) | set(od.modernization_era))
            context_summary = {str(h): float(g.observed_original_fraction.median()) for h, g in context_part.groupby("context_hours")}
            rows.append({
                "protocol": protocol, "stage": definition["stage"], "role": definition["role"], "years": ";".join(map(str, years)),
                "primary_eligible_intervals": int(od.primary_eligible_targets.sum()),
                "sensitivity_eligible_intervals": int(od.sensitivity_eligible_targets.sum()),
                "complete_primary_days": int(od.complete_primary_day.sum()),
                "partial_primary_days": int(((od.primary_eligible_targets > 0) & ~od.complete_primary_day).sum()),
                "zero_primary_days": int((od.primary_eligible_targets == 0).sum()),
                "structural_eras_included": ";".join(eras), "seasonal_coverage": ";".join(seasons),
                "import_fraction_primary": (iv.grid_exchange_kw_signed > 0).mean(),
                "export_fraction_primary": (iv.grid_exchange_kw_signed < 0).mean(),
                "median_observed_context_fraction_json": json.dumps(context_summary, sort_keys=True),
                "availability_assessment": "Counts reported; H3B must define adequacy without model outcomes",
                "test_2023_untouched": not (definition["role"] != "test" and 2023 in years),
            })
    frame = pd.DataFrame(rows)
    write_csv("protocol_candidate_analysis.csv", frame)
    return frame


def manifest() -> None:
    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "python-dateutil", "pytest"]:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "not installed"
    outputs = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "honda_h3a_manifest.json":
            outputs[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    for path in [REPORT, MEASUREMENT_REPORT]:
        if path.exists():
            outputs[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    data = {
        "h2_commit": H2_COMMIT, "silver_sha256": digest(SILVER_PATH), "provenance_mask_sha256": digest(MASK_PATH),
        "issues_source_sha256": digest(ISSUES_PATH), "configuration_sha256": digest(CONFIG_PATH),
        "generator_sha256": digest(Path(__file__)), "holiday_source_metadata": {
            "facility": "Honda R&D Europe facility, Offenbach am Main, Hessen, Germany",
            "facility_source": "Dryad README and Scientific Data article DOI 10.1038/s41597-025-05186-3",
            "holiday_law": "Hessisches Feiertagsgesetz §1",
            "holiday_source_url": "https://www.rv.hessenrecht.hessen.de/bshe/document/jlr-FeiertGHErahmen",
        },
        "python_version": platform.python_version(), "package_versions": packages,
        "generated_outputs": outputs, "deterministic_runtime_timestamp_omitted": True,
        "model_training_performed": False, "final_protocol_frozen": False,
    }
    (OUT / "honda_h3a_manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> None:
    verify_inputs(); OUT.mkdir(parents=True, exist_ok=True)
    silver = pd.read_parquet(SILVER_PATH)
    silver["timestamp_utc"] = pd.to_datetime(silver.timestamp_utc, utc=True)
    intervals, origins = all_target_intervals(silver)
    target_summary(intervals); availability_by_year(origins); complete_day_blocks(origins)
    context, _ = context_availability(silver, origins)
    feature_policies(); day_type_availability(origins); distribution_shift(intervals, origins)
    era_availability(origins); protocol_analysis(intervals, origins, context); manifest()


if __name__ == "__main__":
    run()
