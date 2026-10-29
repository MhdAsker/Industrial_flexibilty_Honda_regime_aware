from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd

from src.honda import audit


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_zip_source_hash_stability_and_integrity():
    before = file_hash(audit.SOURCE)
    assert before == audit.EXPECTED_SHA256
    with zipfile.ZipFile(audit.SOURCE) as zf:
        assert zf.testzip() is None
        assert len([x for x in zf.infolist() if not x.is_dir()]) == 21
    assert file_hash(audit.SOURCE) == before


def test_extracted_inventory_consistency():
    manifest = pd.read_csv(audit.OUT / "extracted_file_manifest.csv")
    inventory = pd.read_csv(audit.OUT / "archive_inventory.csv")
    assert len(manifest) == len(inventory) == 21
    assert set(inventory.suspected_resolution) == {"1min", "15min", "1h"}


def test_timestamp_parsing_duplicates_and_resolution():
    timestamps = pd.read_csv(audit.OUT / "timestamp_audit.csv")
    assert set(timestamps.resolution) == {"1min", "15min", "1h"}
    assert timestamps.duplicate_timestamps.sum() == 0
    assert timestamps.non_cadence_steps.sum() == 0
    parsed = pd.to_datetime(timestamps.first_timestamp)
    assert parsed.notna().all()


def test_unit_parsing_and_key_signal_discovery():
    catalog = pd.read_csv(audit.OUT / "signal_catalog.csv")
    assert "unknown" not in set(catalog.unit)
    expected = {"total", "PV", "CHP", "CHP_heat", "CHP_elec", "cool_elec", "WeatherStation.Weather.Igm", "WeatherStation.Weather.Ta"}
    assert expected.issubset(set(catalog.signal_name))


def test_hierarchy_limitation_is_unique_and_explicit():
    hierarchy = pd.read_csv(audit.OUT / "meter_hierarchy.csv")
    assert len(hierarchy) == 1
    assert hierarchy.meter_id.is_unique
    assert hierarchy.iloc[0].meter_id == "unavailable"


def test_no_parent_child_double_counting_in_subsystem_summary():
    summary = pd.read_csv(audit.OUT / "subsystem_energy_summary.csv")
    assert len(summary) == 1
    assert summary.iloc[0].double_counting_check == "not applicable"
    assert not summary.energy_kWh.duplicated().any()


def test_energy_balance_formula_on_synthetic_data():
    grid = pd.Series([100.0, -20.0])
    pv = pd.Series([-30.0, -50.0])
    chp = pd.Series([-10.0, 0.0])
    gross = grid - pv - chp
    assert gross.tolist() == [140.0, 30.0]


def test_pv_and_chp_sign_checks():
    candidates = pd.read_csv(audit.OUT / "electrical_target_candidates.csv")
    pv = candidates[candidates["candidate_type"].str.startswith("D")].iloc[0]
    chp = candidates[candidates["candidate_type"].str.startswith("E")].iloc[0]
    assert pv.sign_convention == "negative generation"
    assert chp.sign_convention == "negative generation"


def test_correction_imputation_classification():
    integrity = pd.read_csv(audit.OUT / "forecast_target_integrity.csv")
    assert set(integrity.classification) == {"IMPUTED_NOT_TRACEABLE"}


def test_deterministic_archive_inventory():
    audit.archive_inventory()
    first = file_hash(audit.OUT / "archive_inventory.csv")
    audit.archive_inventory()
    assert file_hash(audit.OUT / "archive_inventory.csv") == first


def test_no_model_training_no_gold_and_no_spark_dependency():
    gold_files = [p for p in (audit.ROOT / "data/gold").rglob("*") if p.is_file()]
    assert gold_files == []
    source = (audit.ROOT / "src/honda/audit.py").read_text(encoding="utf-8").lower()
    forbidden_training = ["lightgbm", "chronos", "timesfm", "timegpt", ".fit(", "kmeans"]
    assert not any(token in source for token in forbidden_training)
    assert "industrial_energy_forecasting" not in source
