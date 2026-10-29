"""Freeze and materialize the leakage-safe Honda forecasting protocol.

No prediction, model fitting, optimization, or performance calculation occurs here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/honda_forecasting_protocol_frozen.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
H2_COMMIT = CONFIG["H2_commit"]
H3A_COMMIT = CONFIG["H3A_commit"]
SILVER_PATH = ROOT / "data/silver/honda/honda_energy_15min.parquet"
MASK_PATH = ROOT / "data/silver/honda/provenance_masks_15min.parquet"
INDEX_PATH = ROOT / "data/silver/honda/honda_forecasting_index_15min.parquet"
H3A_OUT = ROOT / "results/protocol_h3a"
OUT = ROOT / "results/protocol"
REPORT = ROOT / "reports/honda_forecasting_protocol.md"
TZ = CONFIG["timezone"]
OBSERVED = "OBSERVED_ORIGINAL"
CORRECTED = "CORRECTED_NONIMPUTED"


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
    if subprocess.run(["git", "merge-base", "--is-ancestor", H3A_COMMIT, head], cwd=ROOT).returncode:
        raise RuntimeError("Verified H3A checkpoint is not an ancestor of HEAD")
    expected = {
        SILVER_PATH: "d9d93445391985faa7a0aa015cdb65b949f3909636d284c59449db876ae310df",
        MASK_PATH: "f93d5bf4597d8f768cc63437f2ce78d21f4050f4dfcfbbe0e59d2fa0ef4eb778",
    }
    for path, wanted in expected.items():
        if digest(path) != wanted:
            raise RuntimeError(f"Input checksum mismatch: {path}")
    h3a_manifest = json.loads((H3A_OUT / "honda_h3a_manifest.json").read_text(encoding="utf-8"))
    for relative, metadata in h3a_manifest["generated_outputs"].items():
        path = ROOT / relative
        if not path.exists() or digest(path) != metadata["sha256"]:
            raise RuntimeError(f"H3A artifact mismatch: {relative}")


def historical_interval_available(interval_start_utc: pd.Timestamp, issue_time_utc: pd.Timestamp) -> bool:
    return interval_start_utc + pd.Timedelta(minutes=CONFIG["resolution_minutes"]) <= issue_time_utc


def lead_band(lead_hours: float) -> str:
    if 11 <= lead_hours < 17: return "H1"
    if 17 <= lead_hours < 23: return "H2"
    if 23 <= lead_hours < 29: return "H3"
    if 29 <= lead_hours <= 36: return "H4"
    raise ValueError(f"Lead outside frozen physical horizon: {lead_hours}")


def grid_mode(value_kw: float) -> str:
    epsilon = float(CONFIG["near_zero_tolerance"]["epsilon_kw"])
    if value_kw > epsilon: return "IMPORT"
    if value_kw < -epsilon: return "EXPORT"
    return "NEUTRAL"


def assert_development_test_guard(timestamp: pd.Timestamp, purpose: str) -> None:
    forbidden = set(CONFIG["test_access_guard"]["forbidden_uses"])
    if timestamp.tz_convert(TZ).year == 2023 and purpose in forbidden:
        raise PermissionError(f"2023 TEST labels are forbidden for {purpose}")


def is_bridge_day(day: date, holidays: set[date]) -> bool:
    if day.weekday() >= 5 or day in holidays:
        return False
    previous, following = day - timedelta(days=1), day + timedelta(days=1)
    return (previous in holidays and following.weekday() >= 5) or (following in holidays and previous.weekday() >= 5)


def holiday_context(day: date, holidays: set[date]) -> str:
    labels = ["WEEKEND" if day.weekday() >= 5 else "NORMAL_WEEKDAY"]
    if day in holidays: labels.append("PUBLIC_HOLIDAY")
    if day + timedelta(days=1) in holidays: labels.append("DAY_BEFORE_PUBLIC_HOLIDAY")
    if day - timedelta(days=1) in holidays: labels.append("DAY_AFTER_PUBLIC_HOLIDAY")
    if is_bridge_day(day, holidays): labels.append("BRIDGE_DAY_DIAGNOSTIC_ONLY")
    return ";".join(labels)


def p1_roles(year: int) -> tuple[str, str]:
    roles = {
        2020: "P1_FOLD_A_TRAIN;P1_FOLD_B_TRAIN;P1_FINAL_DEVELOPMENT",
        2021: "P1_FOLD_A_VALIDATION;P1_FOLD_B_TRAIN;P1_FINAL_DEVELOPMENT",
        2022: "P1_FOLD_B_VALIDATION;P1_FINAL_DEVELOPMENT",
        2023: "P1_FINAL_TEST",
    }
    return roles[year], "DEVELOPMENT_2020_2022" if year <= 2022 else "TEST_2023"


def p2_roles(year: int) -> str:
    return {
        2020: "OUTSIDE_POST_2020_SENSITIVITY",
        2021: "P2_TRAIN;P2_FINAL_DEVELOPMENT",
        2022: "P2_VALIDATION;P2_FINAL_DEVELOPMENT",
        2023: "P2_TEST",
    }[year]


def build_forecasting_index() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lead = pd.read_csv(H3A_OUT / "lead_time_distribution.csv")
    origins = pd.read_csv(H3A_OUT / "forecast_origins_candidate.csv")
    holidays_frame = pd.read_csv(H3A_OUT / "holiday_calendar.csv")
    holidays = set(pd.to_datetime(holidays_frame.holiday_date).dt.date)
    lead["timestamp_utc"] = pd.to_datetime(lead.target_timestamp_utc, utc=True)
    lead["timestamp_local"] = lead.timestamp_utc.dt.tz_convert(TZ)
    lead["target_day_date"] = pd.to_datetime(lead.target_day_local).dt.date
    day_complete = lead.groupby("target_day_local").primary_eligible.all()
    origin_map = origins.set_index("target_day_local")
    era_map = {
        day: json.dumps({
            "pv_capacity_era": row.pv_capacity_era,
            "covid_era": row.covid_era,
            "chp_control_era": row.chp_control_era,
            "modernization_era": row.modernization_era,
        }, sort_keys=True, separators=(",", ":"))
        for day, row in origin_map.iterrows()
    }
    holiday_context_map = {day.isoformat(): holiday_context(day, holidays) for day in sorted(set(lead.target_day_date))}
    roles = lead.year.map(lambda year: p1_roles(int(year)))
    index = pd.DataFrame({
        "timestamp_utc": lead.timestamp_utc,
        "timestamp_local": lead.timestamp_local,
        "target_day_local": lead.target_day_local,
        "forecast_issue_time_utc": pd.to_datetime(lead.forecast_issue_time_utc, utc=True),
        "forecast_issue_time_local": pd.to_datetime(lead.forecast_issue_time_utc, utc=True).dt.tz_convert(TZ),
        "lead_time_minutes": (lead.lead_time_hours * 60).round().astype("int16"),
        "lead_band": lead.lead_time_hours.map(lead_band),
        "target_primary_eligible": lead.primary_eligible.astype(bool),
        "target_sensitivity_eligible": lead.sensitivity_eligible.astype(bool),
        "complete_primary_day": lead.target_day_local.map(day_complete).astype(bool),
        "development_fold_role": [value[0] for value in roles],
        "primary_protocol_split": [value[1] for value in roles],
        "sensitivity_protocol_split": lead.year.map(lambda year: p2_roles(int(year))),
        "holiday_context": lead.target_day_local.map(holiday_context_map),
        "structural_era_metadata": lead.target_day_local.map(era_map),
        "grid_mode_diagnostic_only": lead.grid_exchange_kw_signed.map(grid_mode),
    })
    index.to_parquet(INDEX_PATH, index=False)
    return index, lead, origins


def frozen_feature_contract() -> pd.DataFrame:
    rows = [
        ("signed grid exchange", "historical completed", "HISTORICAL_ONLY", "M0,M1", "Subject to predictor provenance policy and issuance cutoff"),
        ("signed grid exchange", "future target", "DIAGNOSTIC_ONLY", "target", "Target value inaccessible until evaluation"),
        ("local calendar", "target time", "FUTURE_KNOWN", "M0,M1", "Hour, quarter, weekday, weekend, month, day-of-year, DST"),
        ("Hessen public holiday", "target time", "FUTURE_KNOWN", "M0,M1", "Statutory calendar"),
        ("future-safe structural date", "target time", "FUTURE_KNOWN", "M0,M1", "Only documented dates known at issuance"),
        ("retrospective structural era", "target time", "DIAGNOSTIC_ONLY", "none", "Evaluation stratum, never target-derived model feature"),
        ("Ta ambient temperature", "historical completed", "HISTORICAL_ONLY", "M1", "Provenance-safe pre-issuance observation"),
        ("Ta ambient temperature", "future realized", "FORBIDDEN_REALIZED_FUTURE", "none", "Unavailable operationally"),
        ("Ta forecast", "future", "FORECAST_REQUIRED", "later experiment", "Archived operational weather forecast required"),
        ("Igm irradiance", "historical completed", "HISTORICAL_ONLY", "M1", "Provenance-safe pre-issuance observation"),
        ("Igm irradiance", "future realized", "FORBIDDEN_REALIZED_FUTURE", "none", "Unavailable operationally"),
        ("Igm forecast", "future", "FORECAST_REQUIRED", "later experiment", "Archived operational weather forecast required"),
        ("PV generation", "historical completed", "HISTORICAL_ONLY", "M1", "Provenance-safe pre-issuance observation"),
        ("PV generation", "future measured", "FORBIDDEN_REALIZED_FUTURE", "none", "Direct future grid-exchange component"),
        ("PV forecast", "future", "FORECAST_REQUIRED", "later experiment", "Independent leakage-safe PV forecast required"),
        ("CHP generation/state", "historical completed", "HISTORICAL_ONLY", "M1", "Provenance-safe pre-issuance observation"),
        ("CHP generation/state", "future measured", "FORBIDDEN_REALIZED_FUTURE", "none", "Unavailable operationally"),
        ("CHP forecast/schedule", "future", "FORECAST_REQUIRED", "later experiment", "Independent forecast or proven fixed schedule required"),
        ("heating", "historical completed", "HISTORICAL_ONLY", "M1", "Provenance-safe pre-issuance observation"),
        ("heating", "future realized", "FORBIDDEN_REALIZED_FUTURE", "none", "Future thermal operation not assumed known"),
        ("cooling", "historical completed", "HISTORICAL_ONLY", "M1", "Provenance-safe pre-issuance observation"),
        ("cooling", "future realized", "FORBIDDEN_REALIZED_FUTURE", "none", "Future thermal operation not assumed known"),
        ("future electricity price", "future", "FORECAST_REQUIRED", "later market experiment", "Requires verified publication timing and data provenance"),
        ("realized target grid mode", "target time", "DIAGNOSTIC_ONLY", "none", "Defined from y_true; forbidden as feature"),
    ]
    frame = pd.DataFrame(rows, columns=["variable_or_domain", "time_relation", "classification", "feature_contract", "rule"])
    write_csv("frozen_feature_contract.csv", frame)
    return frame


def frozen_baseline_specifications() -> pd.DataFrame:
    rows = [
        ("B0", "Issuance persistence", "Most recent completed safe signed-grid value", "Hold constant across target day", "OBSERVED_ORIGINAL primary; corrected only in flagged sensitivity", "Source interval ends by issuance", "none"),
        ("B1", "Most recent safe daily analogue", "Most recent preceding local day matching local quarter-hour and repeated-hour fold/offset", "Missing prediction if no safe match", "Frozen historical predictor policy", "Source exists, is safe, and completed by issuance", "none"),
        ("B2", "Previous-week analogue", "Same local quarter-hour exactly seven local calendar days earlier; fold/offset match where applicable", "Missing prediction; no arbitrary alternate week", "Frozen historical predictor policy", "Source exists, is safe, and completed by issuance", "none"),
        ("B3", "Recent-week robust analogue", "Median of same local quarter-hour from prior N eligible calendar weeks", "Missing if protocol-valid sources insufficient; exact minimum frozen in H4 development", "Frozen historical predictor policy", "Every included source completed by issuance", "N in {2,4,8}; minimum source count"),
        ("B4", "Calendar climatology", "Training-only robust statistic grouped by calendar context and local quarter-hour", "Backoff allowed only if hierarchy frozen on development folds", "Training targets/predictors follow frozen provenance", "Training partition only; no validation/test target encoding", "Grouping/backoff selected on development folds"),
        ("B5", "Simple baseline ensemble", "Predetermined combination of selected strong B0-B4 baselines", "Missing/fallback behavior frozen before TEST and reported", "All components obey their source provenance rules", "Development-fold evidence only", "Members and weights selected before TEST"),
    ]
    frame = pd.DataFrame(rows, columns=["baseline_id", "name", "definition", "fallback_behavior", "provenance_requirement", "availability_rule", "development_selected_hyperparameters"])
    write_csv("frozen_baseline_specifications.csv", frame)
    return frame


def season(month: int) -> str:
    if month in [12, 1, 2]: return "winter"
    if month in [3, 4, 5]: return "spring"
    if month in [6, 7, 8]: return "summer"
    return "autumn"


def frozen_fold_summary(index: pd.DataFrame, lead: pd.DataFrame, origins: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("P1", "FOLD_A", "TRAIN", [2020]), ("P1", "FOLD_A", "VALIDATION", [2021]),
        ("P1", "FOLD_B", "TRAIN", [2020, 2021]), ("P1", "FOLD_B", "VALIDATION", [2022]),
        ("P1", "FINAL", "DEVELOPMENT_REFIT", [2020, 2021, 2022]), ("P1", "FINAL", "TEST", [2023]),
        ("P2", "DEVELOPMENT", "TRAIN", [2021]), ("P2", "DEVELOPMENT", "VALIDATION", [2022]),
        ("P2", "FINAL", "DEVELOPMENT_REFIT", [2021, 2022]), ("P2", "FINAL", "TEST", [2023]),
    ]
    rows = []
    origins = origins.copy(); origins["date"] = pd.to_datetime(origins.target_day_local)
    origins["season"] = origins.date.dt.month.map(season)
    for protocol, fold, role, years in definitions:
        day = origins[origins.year.isin(years)]
        iv = lead[lead.year.isin(years) & lead.primary_eligible]
        era_values = sorted(set(day.pv_capacity_era) | set(day.covid_era) | set(day.chp_control_era) | set(day.modernization_era))
        holiday = day.public_holiday
        rows.append({
            "protocol": protocol, "fold": fold, "role": role, "years": ";".join(map(str, years)),
            "primary_eligible_interval_count": int(day.primary_eligible_targets.sum()),
            "sensitivity_eligible_interval_count": int(day.sensitivity_eligible_targets.sum()),
            "complete_primary_day_count": int(day.complete_primary_day.sum()),
            "partial_primary_day_count": int(((day.primary_eligible_targets > 0) & ~day.complete_primary_day).sum()),
            "zero_primary_day_count": int((day.primary_eligible_targets == 0).sum()),
            "import_fraction": (iv.grid_exchange_kw_signed > 0).mean(),
            "export_fraction": (iv.grid_exchange_kw_signed < 0).mean(),
            "public_holiday_day_count": int(holiday.sum()),
            "complete_primary_public_holiday_count": int((holiday & day.complete_primary_day).sum()),
            "seasonal_coverage": ";".join(sorted(day.season.unique())),
            "known_structural_eras": ";".join(era_values),
        })
    frame = pd.DataFrame(rows)
    write_csv("frozen_fold_summary.csv", frame)
    return frame


def manifest() -> None:
    h3a_manifest = json.loads((H3A_OUT / "honda_h3a_manifest.json").read_text(encoding="utf-8"))
    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "pytest"]:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "not installed"
    outputs = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "honda_h3_manifest.json":
            outputs[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    for path in [CONFIG_PATH, INDEX_PATH, REPORT]:
        if path.exists(): outputs[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    data = {
        "H2_commit": H2_COMMIT, "H3A_commit": H3A_COMMIT,
        "silver_sha256": digest(SILVER_PATH), "provenance_mask_sha256": digest(MASK_PATH),
        "H3A_manifest_sha256": digest(H3A_OUT / "honda_h3a_manifest.json"),
        "H3A_artifact_hashes": h3a_manifest["generated_outputs"],
        "frozen_protocol_sha256": digest(CONFIG_PATH), "forecast_index_sha256": digest(INDEX_PATH),
        "feature_contract_sha256": digest(OUT / "frozen_feature_contract.csv"),
        "baseline_specification_sha256": digest(OUT / "frozen_baseline_specifications.csv"),
        "generator_sha256": digest(Path(__file__)), "python_version": platform.python_version(),
        "package_versions": packages, "deterministic_outputs": outputs,
        "model_training_performed": False, "predictions_created": False, "optimization_performed": False,
    }
    (OUT / "honda_h3_manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> None:
    verify_inputs(); OUT.mkdir(parents=True, exist_ok=True)
    index, lead, origins = build_forecasting_index()
    frozen_feature_contract(); frozen_baseline_specifications(); frozen_fold_summary(index, lead, origins)
    manifest()


if __name__ == "__main__":
    run()
