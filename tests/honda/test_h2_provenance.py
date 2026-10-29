from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.honda import h2_provenance as h2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_h1_checkpoint_and_finalized_artifacts_unchanged():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=h2.ROOT, text=True).strip()
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", h2.H1_COMMIT, head], cwd=h2.ROOT, check=False
    ).returncode == 0
    for relative in ["reports/honda_reduced_dataset_audit.md", "results/audit/honda_audit_manifest.json"]:
        assert subprocess.run(
            ["git", "diff", "--quiet", h2.H1_COMMIT, "--", relative], cwd=h2.ROOT, check=False
        ).returncode == 0


def test_source_checksum_stability_and_issue_count():
    assert sha256(h2.RAW / "reduced_data.zip") == "2a35a629fdf863793ab1a0c1bf80110ec7fc909ea65f716613a529da4539571f"
    assert sha256(h2.PROVENANCE_RAW / "issues.zip") == h2.ISSUES_SHA256
    assert sha256(h2.PROVENANCE_RAW / "README.md") == "7433ae4227aa9253c3a4c3f007486f5567d7fec2fdcf33b1600799217ff7cdee"
    assert len(pd.read_csv(h2.OUT / "issue_catalog.csv", usecols=["issue_id"])) == 148_968


def test_issue_interval_parsing_and_utc_overlap_edges():
    assert h2.parse_time("2020-01-01 01:00:00+01:00") == pd.Timestamp("2020-01-01T00:00:00Z")
    ts = pd.date_range("2020-01-01", periods=4, freq="15min", tz="UTC")
    assert h2.overlapping_bin_slice(ts, ts[1], ts[1]) == (1, 2)
    assert h2.overlapping_bin_slice(ts, ts[1] + pd.Timedelta("1s"), ts[2] - pd.Timedelta("1s")) == (1, 2)
    assert h2.overlapping_bin_slice(ts, ts[1], ts[2]) == (1, 3)


def test_every_issue_has_explicit_mapping_disposition():
    issues = pd.read_csv(h2.OUT / "issue_catalog.csv", usecols=["source_file", "issue_id"], dtype=str)
    audit = pd.read_csv(h2.OUT / "issue_target_mapping_audit.csv", usecols=["source_file", "issue_id", "disposition"], dtype=str)
    expected = set(map(tuple, issues[["source_file", "issue_id"]].drop_duplicates().to_numpy()))
    actual = set(map(tuple, audit[["source_file", "issue_id"]].drop_duplicates().to_numpy()))
    assert actual == expected
    assert audit.disposition.notna().all()


def test_structural_pv_absence_and_canonical_generation_signs():
    masks = pd.read_parquet(h2.SILVER / "provenance_masks_15min.parquet")
    silver = pd.read_parquet(h2.SILVER / "honda_energy_15min.parquet")
    before = silver.timestamp_utc < h2.parse_time(h2.CONFIG["pv_first_phase_documented_start_utc"])
    pv_mask = masks[(masks.signal == "PV") & (masks.timestamp_utc < h2.parse_time(h2.CONFIG["pv_first_phase_documented_start_utc"]))]
    assert set(pv_mask.provenance_class) == {h2.STRUCTURAL}
    assert (silver.loc[before, "pv_generation_kw"] == 0).all()
    sample = pd.Series([-5.0, 0.0, 2.0])
    assert h2.canonical_generation(sample).tolist() == [5.0, -0.0, 0.0]
    assert (silver.pv_generation_kw >= 0).all() and (silver.chp_generation_kw >= 0).all()


def test_import_export_split_and_gross_equation():
    imports, exports = h2.split_grid_exchange(pd.Series([10.0, -4.0, 0.05]), 0.1)
    assert imports.tolist() == [10.0, 0.0, 0.0]
    assert exports.tolist() == [0.0, 4.0, 0.0]
    silver = pd.read_parquet(h2.SILVER / "honda_energy_15min.parquet")
    expected = silver.grid_exchange_kw_signed - silver.pv_source_kw_signed - silver.chp_source_kw_signed
    assert np.allclose(expected, silver.gross_load_reconstructed_kw, equal_nan=True)
    assert not silver.gross_load_primary_eligible.any()


def test_combined_gross_provenance_precedence():
    assert h2.combined_provenance([h2.OBSERVED, h2.STRUCTURAL, h2.OBSERVED]) == h2.OBSERVED
    assert h2.combined_provenance([h2.OBSERVED, h2.CORRECTED]) == h2.CORRECTED
    assert h2.combined_provenance([h2.CORRECTED, h2.MISSING]) == h2.MISSING
    assert h2.combined_provenance([h2.UNKNOWN, h2.MISSING]) == h2.UNKNOWN
    assert h2.combined_provenance([h2.IMPUTED, h2.UNKNOWN]) == h2.IMPUTED


def test_target_eligibility_rules_are_conservative():
    masks = pd.read_parquet(h2.SILVER / "provenance_masks_15min.parquet")
    forbidden = masks.provenance_class.isin([h2.IMPUTED, h2.UNKNOWN, h2.MISSING])
    assert not masks.loc[forbidden, "eligible_primary_target"].any()
    corrected = masks.provenance_class == h2.CORRECTED
    assert not masks.loc[corrected, "eligible_primary_target"].any()
    assert masks.loc[corrected & (masks.signal != "gross_load_reconstructed"), "eligible_sensitivity_target"].all()
    assert h2.CONFIG["primary_eligibility"] == [h2.OBSERVED]


def test_meter_mapping_uniqueness_and_topology_boundaries():
    mapping = pd.read_csv(h2.OUT / "aggregate_source_mapping.csv")
    assert not mapping.duplicated(["reduced_signal", "source_meter_id", "valid_from", "valid_to"]).any()
    old = mapping[mapping.source_meter_id.isin(["H2.Z35", "H2.Z36"])]
    new = mapping[mapping.source_meter_id.isin(["H2.Z351", "H2.Z361"])]
    assert set(old.valid_to) == {h2.CONFIG["office_transformers_old_end_utc"]}
    assert set(new.valid_from) == {h2.CONFIG["office_transformers_new_start_utc"]}


def test_no_forecasting_models_or_split_artifacts():
    source = (h2.ROOT / "src/honda/h2_provenance.py").read_text(encoding="utf-8").lower()
    forbidden = ["lightgbm", "xgboost", "chronos", "timesfm", "timegpt", "n-beats", ".fit(", "kmeans"]
    assert not any(token in source for token in forbidden)
    assert not [path for path in (h2.ROOT / "data/gold").rglob("*") if path.is_file()]
    assert not any("train" in p.name.lower() or "validation_split" in p.name.lower() for p in h2.OUT.rglob("*"))


def test_deterministic_rerun_of_core_generated_tables():
    targets = [
        h2.OUT / "aggregate_source_mapping.csv",
        h2.OUT / "electrical_topology_eras.csv",
        h2.OUT / "future_model_usage_policy.csv",
        h2.OUT / "gross_load_validation.csv",
        h2.OUT / "reduced_vs_source_validation.csv",
    ]
    before = {path: sha256(path) for path in targets}
    h2.aggregate_mapping()
    h2.topology_eras()
    h2.policies_and_validation()
    assert {path: sha256(path) for path in targets} == before
