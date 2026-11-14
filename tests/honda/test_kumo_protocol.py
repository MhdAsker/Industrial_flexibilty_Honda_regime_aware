from __future__ import annotations

import hashlib
import json

import pandas as pd

from src.honda.h6_foundation_data import ROOT
from src.honda.kumo import runner
from src.honda.kumo.downstream import point_forecast_adapter


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_honda_and_foundation_configs_are_unchanged():
    expected = {
        "config/honda_forecasting_protocol_frozen.json": "281f49331bc3b3c31aec0d33b50f6a83a149261db93ce04a29d67479cb62676e",
        "config/honda_baseline_selection_frozen.json": "1bbbbac404675bb1bfb5c85067e1504fda843fc2bc52151a60025f4e3c8b7a04",
        "config/honda_lgbm_selection_frozen.json": "7fff8efbf51fc17ac2524ceeac30dbed6764d4ec55eaf1ea3993c47691dd21ea",
    }
    for relative, expected_hash in expected.items():
        assert digest(ROOT / relative) == expected_hash


def test_kumo_protocol_is_retrospective_and_does_not_claim_completed_evaluation():
    config = json.loads(runner.EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    assert config["study_classification"] == "RETROSPECTIVE_POST_BENCHMARK_EXTENSION"
    assert config["test_used_for_selection"] is False
    assert config["protocol_status"] == "DEVELOPMENT_PROTOCOL_FROZEN_BEFORE_KUMO_INFERENCE"
    assert not runner.FROZEN_CONFIG.exists()


def test_official_versions_and_api_are_pinned():
    config = json.loads(runner.EXPERIMENT_CONFIG.read_text(encoding="utf-8"))["official_kumo"]
    assert config["repository_commit"] == runner.KUMO_TS_COMMIT
    assert config["model_revision"] == runner.MODEL_REVISION
    assert config["backbone_revision"] == runner.BACKBONE_REVISION
    assert config["native_regime_filtering_supported"] is False


def test_downstream_adapter_preserves_point_forecast_and_invents_no_scenarios():
    reference = pd.read_parquet(ROOT / "results/ml/development_predictions.parquet")
    day = str(reference.target_day_local.min())
    reference = reference[reference.target_day_local.astype(str).eq(day)].copy()
    protocol = pd.read_parquet(
        ROOT / "data/silver/honda/honda_forecasting_index_15min.parquet",
        columns=["timestamp_utc", "forecast_issue_time_utc"],
    )
    predictions = reference[["timestamp_utc", "target_day_local", "LGBM_M1"]].merge(
        protocol, on="timestamp_utc", validate="one_to_one"
    )
    predictions = predictions.rename(columns={"LGBM_M1": "prediction_kw"})
    predictions["prediction_available"] = True
    adapted = point_forecast_adapter(predictions, phase="development")
    assert adapted.complete_day.all()
    assert adapted.point_forecast_kw.to_numpy().tolist() == predictions.prediction_kw.to_numpy().tolist()
    assert not any("scenario" in column for column in adapted.columns)
