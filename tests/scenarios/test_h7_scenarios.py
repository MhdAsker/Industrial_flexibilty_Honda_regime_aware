from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.honda.scenarios import core, metrics
from src.honda.scenarios.adapters import (
    HONDA_ROOT,
    expected_intervals,
    load_honda_development,
    load_honda_test,
    spark_root,
)
from src.honda.scenarios import pipeline


ROOT = HONDA_ROOT
OUT = ROOT / "results/scenarios"


def synthetic_day(date: str, actual_offset: float = 1.0, complete: bool = True) -> pd.DataFrame:
    start = pd.Timestamp(date, tz="Europe/Berlin")
    end = pd.Timestamp(pd.Timestamp(date).date() + pd.Timedelta(days=1), tz="Europe/Berlin")
    utc = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="15min", inclusive="left")
    local = utc.tz_convert("Europe/Berlin")
    point = np.sin(np.arange(len(utc)) / 8) * 5
    return pd.DataFrame(
        {
            "dataset_id": "HONDA",
            "model_id": "synthetic",
            "split": "synthetic",
            "target_day_local": date,
            "timestamp_utc": utc,
            "timestamp_local": local,
            "forecast_issue_time_utc": pd.Timestamp(date, tz="Europe/Berlin").tz_convert("UTC") - pd.Timedelta(hours=11),
            "forecast_issue_time_local": pd.Timestamp(date, tz="Europe/Berlin") - pd.Timedelta(hours=11),
            "lead_time_minutes": np.arange(len(utc)) * 15 + 660,
            "actual_kw": point + actual_offset,
            "point_forecast_kw": point,
            "complete_day": complete,
            "local_hour": local.hour,
            "local_quarter_hour": local.minute // 15,
            "month": local.month,
            "season": "winter",
            "day_type": "weekday",
            "holiday": False,
            "quality_policy": "OBSERVED_ORIGINAL",
        }
    )


def donors(n: int = 70) -> list[pd.DataFrame]:
    return [synthetic_day(str((pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)).date()), float(i % 7 - 3)) for i in range(n)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_h6_checkpoint_is_preserved_as_ancestor():
    result = subprocess.run(["git", "merge-base", "--is-ancestor", pipeline.H6_COMMIT, "HEAD"], cwd=ROOT)
    assert result.returncode == 0


def test_frozen_honda_protocol_hash_is_preserved():
    assert digest(ROOT / "config/honda_forecasting_protocol_frozen.json") == "281f49331bc3b3c31aec0d33b50f6a83a149261db93ce04a29d67479cb62676e"


def test_spark_repository_is_read_only_and_clean():
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=spark_root(), text=True).strip() == ""


def test_source_hash_verification_succeeds():
    hashes = pipeline.verify_source_hashes()
    assert len(hashes) == 4 and all(len(x) == 64 for x in hashes.values())


def test_residual_sign_is_actual_minus_forecast():
    day = synthetic_day("2020-04-01", 2.5)
    assert np.allclose(core.residual_vector(day), 2.5)


def test_incomplete_day_is_rejected_as_donor():
    good = synthetic_day("2020-04-01")
    bad = synthetic_day("2020-04-02", complete=False)
    selected, audit = core.complete_donor_days(pd.concat([good, bad]))
    assert len(selected) == 1
    assert not audit.loc[audit.target_day_local.eq("2020-04-02"), "eligible_donor"].iloc[0]


def test_nonfinite_donor_is_rejected():
    bad = synthetic_day("2020-04-01")
    bad.loc[0, "actual_kw"] = np.nan
    selected, _ = core.complete_donor_days(bad)
    assert not selected


def test_no_self_day_or_future_donor_selection():
    target = synthetic_day("2020-03-20")
    library = donors(90)
    matrix, meta = core.generate_scenarios(target, library, pipeline.config("S2", 10))
    assert matrix is not None
    assert all(pd.Timestamp(x) < pd.Timestamp("2020-03-20") for x in meta["donor_ids"])


def test_insufficient_past_library_is_unsupported():
    target = synthetic_day("2020-01-20")
    matrix, meta = core.generate_scenarios(target, donors(90), pipeline.config("S2", 10))
    assert matrix is None and meta["conditioning"] == "insufficient_global_history"


@pytest.mark.parametrize("method", ["S0", "S1", "S2", "S3"])
def test_all_declared_methods_generate_valid_shapes(method: str):
    target = synthetic_day("2020-04-01")
    count = 1 if method == "S0" else 10
    matrix, _ = core.generate_scenarios(target, donors(90), pipeline.config(method, count))
    assert matrix is not None and matrix.shape == (count, 96)


@pytest.mark.parametrize("count", [10, 25, 50])
def test_supported_scenario_counts(count: int):
    matrix, _ = core.generate_scenarios(synthetic_day("2020-04-01"), donors(90), pipeline.config("S2", count))
    assert matrix.shape[0] == count


def test_conditional_fallback_is_explicit():
    target = synthetic_day("2020-04-01")
    target["season"] = "rare"
    matrix, meta = core.generate_scenarios(target, donors(90), pipeline.config("S3", 10))
    assert matrix is not None and meta["fallback_used"] and "fallback" in meta["conditioning"]


@pytest.mark.parametrize(("date", "count"), [("2023-03-26", 92), ("2023-03-27", 96), ("2023-10-29", 100)])
def test_physical_dst_interval_counts(date: str, count: int):
    assert expected_intervals(date) == count
    assert len(synthetic_day(date)) == count


def test_autumn_repeated_hour_uses_unique_utc_keys():
    day = synthetic_day("2023-10-29")
    assert day.timestamp_utc.is_unique and len(day) == 100
    assert day.timestamp_local.astype(str).duplicated().sum() == 0  # offsets distinguish the repeated hour


def test_spring_nonexistent_hour_is_not_fabricated():
    day = synthetic_day("2023-03-26")
    assert len(day) == 92
    assert not ((day.timestamp_local.dt.hour == 2)).any()


def test_rare_dst_coherent_sampling_is_unsupported_without_five_donors():
    target = synthetic_day("2023-03-26")
    library = donors(90) + [synthetic_day("2022-03-27")]
    matrix, meta = core.generate_scenarios(target, library, pipeline.config("S2", 10))
    assert matrix is None and meta["conditioning"] == "unsupported_physical_day_length"


def test_scenario_weights_sum_to_one():
    target = synthetic_day("2020-04-01")
    cfg = pipeline.config("S2", 10)
    matrix, meta = core.generate_scenarios(target, donors(90), cfg)
    long = core.scenario_long_frame(target, matrix, cfg, "a" * 64, meta)
    weights = long.groupby("scenario_id").scenario_weight.first()
    assert np.isclose(weights.sum(), 1.0)


def test_negative_honda_scenarios_are_preserved():
    target = synthetic_day("2020-04-01", -20)
    matrix, _ = core.generate_scenarios(target, [synthetic_day(str((pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)).date()), -20) for i in range(90)], pipeline.config("S2", 10))
    assert matrix.min() < 0


def test_nan_and_inf_scenarios_are_rejected():
    with pytest.raises(ValueError):
        core.validate_scenario_matrix(np.array([[1.0, np.inf]]), 2, allow_negative=True)


def test_deterministic_random_generation():
    args = (synthetic_day("2020-04-01"), donors(90), pipeline.config("S2", 25))
    first, _ = core.generate_scenarios(*args)
    second, _ = core.generate_scenarios(*args)
    assert np.array_equal(first, second)


def test_marginal_quantile_extraction_is_monotonic():
    matrix = np.arange(50 * 3, dtype=float).reshape(50, 3)
    quantiles = metrics.ensemble_quantiles(matrix)
    assert quantiles.shape == (7, 3) and np.all(np.diff(quantiles, axis=0) >= 0)


def test_energy_score_zero_for_perfect_degenerate_ensemble():
    actual = np.array([1.0, 2.0, 3.0])
    assert metrics.energy_score(actual, np.tile(actual, (10, 1))) == pytest.approx(0.0)


def test_variogram_score_zero_for_perfect_ensemble():
    actual = np.arange(20, dtype=float)
    assert metrics.variogram_score(actual, np.tile(actual, (10, 1))) == pytest.approx(0.0)


def test_temporal_dependence_metric_recognizes_exact_structure():
    residual = np.sin(np.arange(96) / 8)
    result = metrics.dependence_scores(residual, np.tile(residual, (10, 1)))
    assert result["acf_rmse"] == pytest.approx(0.0)


def test_export_probability_and_brier_score():
    actual = np.array([-1.0, 1.0])
    matrix = np.array([[-2.0, 2.0], [-3.0, 3.0]])
    result = metrics.export_scores(actual, matrix)
    assert result["brier_score"] == 0.0 and result["probability_any_export"] == 1.0


def test_calibration_bins_handle_probability_one():
    bins = metrics.calibration_bins(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert bins["count"].sum() == 2


def test_honda_adapter_contains_only_frozen_oos_development_folds():
    frame = load_honda_development()
    assert set(frame.split) == {"FOLD_A", "FOLD_B"}
    assert set(pd.to_datetime(frame.target_day_local).dt.year) == {2021, 2022}


def test_no_future_realized_covariates_in_canonical_adapter():
    forbidden = {"future_pv", "future_chp", "future_weather", "realized_export_state"}
    assert forbidden.isdisjoint(load_honda_test().columns)


def test_frozen_configuration_precedes_final_diagnostics():
    assert pipeline.FROZEN.exists()
    assert pipeline.FROZEN.stat().st_mtime <= (OUT / "scenario_diagnostic_scores.csv").stat().st_mtime
    cfg = json.loads(pipeline.FROZEN.read_text(encoding="utf-8"))
    assert cfg["test_residuals_used_for_fitting"] is False


def test_scenario_artifact_round_trip_and_weight_contract():
    cfg = json.loads(pipeline.FROZEN.read_text(encoding="utf-8"))
    path = OUT / "honda_2023_diagnostic_scenarios.parquet"
    frame = pd.read_parquet(path)
    assert frame.configuration_sha256.nunique() == 1
    per_day = frame.groupby(["target_day_local", "scenario_id"]).scenario_weight.first().groupby(level=0).sum()
    assert np.allclose(per_day, 1.0)
    assert set(frame.groupby("target_day_local").target_timestamp_utc.nunique().unique()) <= {92, 96, 100}
    assert frame.scenario_id.nunique() == cfg["scenario_count"]


def test_bootstrap_is_reproducible():
    frame = pd.DataFrame(
        {
            "dataset_id": ["HONDA"] * 6,
            "method": ["S2"] * 3 + ["S1"] * 3,
            "target_day_local": ["a", "b", "c"] * 2,
            "energy_score_kw": [1, 2, 3, 2, 3, 4],
        }
    )
    assert pipeline._bootstrap(frame, "S2", "S1", "HONDA") == pipeline._bootstrap(frame, "S2", "S1", "HONDA")


def test_no_flexibility_optimization_was_added_to_h7_scope():
    cfg = json.loads(pipeline.FROZEN.read_text(encoding="utf-8"))
    manifest = json.loads(pipeline.MANIFEST.read_text(encoding="utf-8"))
    assert cfg["flexibility_optimization_performed"] is False
    assert manifest["flexibility_optimization_performed"] is False
    assert not any("optim" in p.name.lower() for p in (ROOT / "src/honda/scenarios").glob("*.py"))
