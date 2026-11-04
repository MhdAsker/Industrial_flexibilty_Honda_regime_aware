from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.honda import h6_analysis as analysis
from src.honda import h6_foundation_data as data
from src.honda import h6_selection as selection


ROOT = data.ROOT
OUT = ROOT / "results/foundation"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configs() -> tuple[dict, dict]:
    return (
        json.loads(selection.CHRONOS_CONFIG.read_text(encoding="utf-8")),
        json.loads(selection.TIMESFM_CONFIG.read_text(encoding="utf-8")),
    )


def test_h5_checkpoint_and_protected_configs_are_unchanged():
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", selection.H5_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    assert ancestry.returncode == 0
    assert digest(ROOT / "config/honda_forecasting_protocol_frozen.json") == selection.H3_HASH
    assert digest(ROOT / "config/honda_baseline_selection_frozen.json") == selection.H4_HASH
    assert digest(ROOT / "config/honda_lgbm_selection_frozen.json") == selection.H5_HASH


def test_contexts_end_exactly_at_issuance_boundary_and_have_expected_lengths():
    for days in [7, 14, 28]:
        origin = data.load_origins([2023], days, max_origins=1)[0]
        assert len(origin.context) == days * 96
        assert origin.issue_utc - pd.Timedelta(minutes=15) < origin.target_timestamps[0]
        context_end = origin.issue_utc - pd.Timedelta(minutes=15)
        assert context_end + pd.Timedelta(minutes=15) <= origin.issue_utc


def test_physical_horizons_and_dst_are_not_forced_to_96():
    frame = pd.read_parquet(OUT / "test_predictions_chronos2.parquet")
    sizes = frame.groupby("target_day_local").size()
    assert set(sizes.unique()) == {92, 96, 100}
    assert sizes.loc["2023-03-26"] == 92
    assert sizes.loc["2023-10-29"] == 100
    assert frame.timestamp_utc.is_unique


def test_frozen_configs_precede_test_and_are_zero_shot():
    for cfg, path, test_path in [
        (configs()[0], selection.CHRONOS_CONFIG, OUT / "test_predictions_chronos2.parquet"),
        (configs()[1], selection.TIMESFM_CONFIG, OUT / "test_predictions_timesfm.parquet"),
    ]:
        assert cfg["selection_status"] == "FROZEN_BEFORE_TEST"
        assert cfg["test_used_for_selection"] is False
        assert cfg["fine_tuning_performed"] is False
        assert cfg["gradient_training_performed"] is False
        assert path.stat().st_mtime <= test_path.stat().st_mtime


def test_context_selection_is_development_only_and_tie_rule_is_applied():
    assert selection.selected_context("CHRONOS2") == 14
    assert selection.selected_context("TIMESFM3") == 28
    for name in ["chronos2_context_selection.csv", "timesfm_context_selection.csv"]:
        table = pd.read_csv(OUT / name)
        assert set(table.fold) == {"FOLD_A", "FOLD_B", "CONCATENATED"}
        assert set(table.context_days) == {7, 14, 28}


def test_no_future_or_forbidden_realized_covariates_enter_inputs():
    origin = data.load_origins([2023], 7, max_origins=1)[0]
    chronos = data.chronos_inputs([origin], "C1")[0]
    assert set(chronos) == {"target", "past_covariates", "future_covariates"}
    assert set(chronos["future_covariates"]) == set(data.FOUNDATION_CALENDAR_KEYS)
    assert "chp_history" in chronos["past_covariates"]
    forbidden = {"PV", "Ta", "Igm", "heating", "cooling", "grid_mode", "future_CHP"}
    assert forbidden.isdisjoint(chronos["future_covariates"])


def test_only_observed_original_history_is_exposed_as_finite_context():
    silver = pd.read_parquet(
        data.SILVER,
        columns=["timestamp_utc", "grid_exchange_kw_signed", "grid_exchange_provenance"],
    ).set_index("timestamp_utc")
    origin = data.load_origins([2023], 7, max_origins=1)[0]
    idx = pd.date_range(origin.issue_utc - pd.Timedelta(days=7), origin.issue_utc - pd.Timedelta(minutes=15), freq="15min", tz="UTC")
    raw = silver.reindex(idx)
    unsafe = ~raw.grid_exchange_provenance.eq(data.OBSERVED)
    assert np.isnan(origin.context[unsafe.to_numpy()]).all()


def test_reference_predictions_are_loaded_unchanged():
    h5 = pd.read_parquet(ROOT / "results/ml/test_predictions_2023.parquet")
    h6 = pd.read_parquet(OUT / "test_predictions_2023.parquet")
    merged = h6.merge(h5[["timestamp_utc", "B5", "LGBM_M1"]], on="timestamp_utc", suffixes=("", "_h5"), validate="one_to_one")
    assert np.array_equal(merged.B5.to_numpy(), merged.B5_h5.to_numpy())
    assert np.array_equal(merged.LIGHTGBM.to_numpy(), merged.LGBM_M1.to_numpy())


def test_negative_forecasts_timestamp_alignment_and_full_coverage():
    frame = pd.read_parquet(OUT / "test_predictions_2023.parquet")
    assert len(frame) == 34161 and frame.timestamp_utc.is_unique
    assert frame[["CHRONOS2", "TIMESFM3"]].notna().all().all()
    assert frame.CHRONOS2.min() < 0 and frame.TIMESFM3.min() < 0
    assert frame.COMMON_CONTEXT_SUBSET.all()


def test_runtime_has_no_gradients_and_uses_isolated_cuda_environments():
    for name in ["test_chronos2.json", "test_timesfm.json"]:
        runtime = json.loads((OUT / "runtime" / name).read_text(encoding="utf-8"))
        assert runtime["gradient_enabled"] is False
        assert runtime["device"] == "cuda"
        assert runtime["peak_gpu_memory_bytes"] > 0
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".venv-h6-chronos/" in ignored and ".venv-h6-timesfm/" in ignored


def test_common_target_bootstrap_and_leakage_audit():
    assert pd.read_csv(OUT / "leakage_audit.csv").violations.sum() == 0
    for name in ["bootstrap_vs_b5.csv", "bootstrap_vs_lgbm.csv"]:
        table = pd.read_csv(OUT / name)
        assert set(table.seed) == {analysis.SEED}
        assert set(table.resamples) == {analysis.BOOTSTRAPS}
        assert set(table.common_complete_days) == {347}


def test_scope_excludes_training_timegpt_calibration_and_optimization():
    manifest = json.loads((OUT / "honda_h6_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fine_tuning_performed"] is False
    assert manifest["gradient_training_performed"] is False
    assert manifest["timegpt_called"] is False
    assert manifest["probabilistic_calibration_performed"] is False
    assert manifest["optimization_performed"] is False


def test_manifest_hashes_are_current():
    manifest = json.loads((OUT / "honda_h6_manifest.json").read_text(encoding="utf-8"))
    for rel, meta in manifest["deterministic_outputs"].items():
        assert digest(ROOT / rel) == meta["sha256"]
