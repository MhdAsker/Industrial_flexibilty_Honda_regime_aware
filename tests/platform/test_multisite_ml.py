from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from honda.platform.contracts import ModelRegistration
from honda.platform.joint_scenarios import joint_empirical_copula
from honda.platform.ledger import PredictionLedgerRecord
from honda.platform.multisite_ml import (
    PartialPoolingResidualAdapter,
    advanced_model_readiness,
    hierarchy_execution_plan,
    leave_one_site_out_splits,
    normalize_features_by_site,
    rolling_origin_splits,
    validate_multisite_frame,
)
from honda.platform.promotion import evaluate_global_promotion
from honda.platform.storage import PlatformStore


def _ledger(model: str, prediction: float, snapshot: str = "a" * 64) -> PredictionLedgerRecord:
    return PredictionLedgerRecord(
        "TENANT", "SITE_A", "ISSUE_1", "2026-01-02T00:00:00+00:00", model,
        f"{model}-prediction", "2026-01-01T12:00:00+00:00", "SIGNED_GRID_EXCHANGE",
        prediction, "b" * 64, "c" * 64, "d" * 64, "e" * 64, snapshot, "CAL-1",
    )


def test_prediction_ledger_is_delayed_label_safe_and_same_snapshot_paired(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    store.append_prediction(_ledger("champion", 12), "runner")
    store.append_prediction(_ledger("challenger", 10), "runner")
    for model in ("champion", "challenger"):
        store.attach_label(
            "TENANT", "SITE_A", "ISSUE_1", "2026-01-02T00:00:00+00:00", model, 9.0,
            "settler", operator_action="APPROVED", decision_outcome="FEASIBLE",
        )
    comparison = store.compare_ledger_models("TENANT", "champion", "challenger")
    assert comparison["macro_relative_improvement"] == pytest.approx(2 / 3)
    assert comparison["support_contract"] == "IDENTICAL_SITE_ISSUE_INTERVAL_SNAPSHOT"
    with pytest.raises(ValueError, match="only be attached once"):
        store.attach_label("TENANT", "SITE_A", "ISSUE_1", "2026-01-02T00:00:00+00:00", "champion", 9.0, "settler")


def test_ledger_comparison_rejects_different_snapshots(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    store.append_prediction(_ledger("champion", 12), "runner")
    store.append_prediction(_ledger("challenger", 10, "f" * 64), "runner")
    for model in ("champion", "challenger"):
        store.attach_label("TENANT", "SITE_A", "ISSUE_1", "2026-01-02T00:00:00+00:00", model, 9.0, "settler")
    with pytest.raises(ValueError, match="same immutable snapshot"):
        store.compare_ledger_models("TENANT", "champion", "challenger")


def _frame() -> pd.DataFrame:
    rows = []
    for site, semantics in (("HONDA", "SIGNED_GRID_EXCHANGE"), ("SPARK_A", "NONNEGATIVE_DEMAND"), ("SPARK_B", "NONNEGATIVE_DEMAND")):
        for issue in pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC"):
            rows.append({
                "site_id": site, "issue_time_utc": issue,
                "target_timestamp_utc": issue + pd.Timedelta(days=1),
                "target_semantics": semantics, "actual_kw": -2.0 if site == "HONDA" else 20.0,
                "power_normalized": 0.5, "hour": 12.0,
            })
    return pd.DataFrame(rows)


def test_multisite_splits_preserve_chronology_and_target_semantics():
    frame = _frame(); features = ("power_normalized", "hour")
    validate_multisite_frame(frame, features)
    rolling = rolling_origin_splits(frame, minimum_train_origins=2, evaluation_origins=1)
    assert len(rolling) == 3
    cold_start = leave_one_site_out_splits(frame)
    assert {site for site, _, _ in cold_start} == {"SPARK_A", "SPARK_B"}
    assert "HONDA" not in {site for site, _, _ in cold_start}
    train = frame.iloc[rolling[0][0]]; test = frame.iloc[rolling[0][1]]
    train_x, test_x, state = normalize_features_by_site(train, test, features)
    assert train_x.shape[1] == test_x.shape[1] == 2
    assert set(state) == {"HONDA", "SPARK_A", "SPARK_B"}


def test_partial_pooling_supports_known_and_cold_start_sites():
    sites = np.asarray(["A", "A", "B", "B"])
    global_prediction = np.asarray([10, 10, 10, 10], dtype=float)
    actual = np.asarray([12, 12, 8, 8], dtype=float)
    adapter = PartialPoolingResidualAdapter.fit(sites, actual, global_prediction, pooling_strength=2)
    prediction = adapter.predict(np.asarray(["A", "C"]), np.asarray([10, 10], dtype=float))
    assert prediction[0] > 10
    assert prediction[1] == pytest.approx(10)


def test_global_promotion_is_fail_closed_on_missing_values_and_site_regression():
    champion = {
        "A": {"mae": 10, "bias": 1, "coverage": .9, "decision_cost": 100, "operator_acceptance": .8},
        "B": {"mae": 20, "bias": 1, "coverage": .9, "decision_cost": 100, "operator_acceptance": .8},
    }
    challenger = {
        "A": {"mae": 9, "bias": 1, "coverage": .9, "decision_cost": 99, "operator_acceptance": .8},
        "B": {"mae": 20.5, "bias": 1, "coverage": .9, "decision_cost": 99, "operator_acceptance": .8},
    }
    service = {"inference_p95_seconds": 2, "solver_p95_seconds": 3, "drift_alarm_precedes_safety": True, "hard_violations": 0}
    assert evaluate_global_promotion(champion, challenger, service)["status"] == "DO_NOT_PROMOTE"
    challenger["B"]["coverage"] = None
    assert evaluate_global_promotion(champion, challenger, service)["status"] == "UNKNOWN"


def test_advanced_models_remain_deferred_without_scale():
    result = advanced_model_readiness(comparable_approved_sites=3, seasonal_days=200, pooled_rows=50_000)
    assert result["status"] == "DEFER_ADVANCED_GENERATIVE_MODELS"
    assert hierarchy_execution_plan(3, 200)[0]["status"] == "IMPLEMENTATION_READY_DATA_REQUIRED"


def test_global_champion_promotion_requires_ledger_and_gate_evidence(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    store.register_model(ModelRegistration("TENANT", None, "global", "forecast", "1", "a" * 64), "ml-owner")
    basic = {"evaluation_sha256": "b" * 64, "approval_sha256": "c" * 64, "rollback_model_id": "local:1"}
    with pytest.raises(ValueError, match="Global model promotion evidence"):
        store.promote_model("TENANT", "global", "1", basic, "ml-owner")
    store.promote_model(
        "TENANT", "global", "1",
        {**basic, "ledger_comparison_sha256": "d" * 64, "promotion_report_sha256": "e" * 64},
        "ml-owner",
    )


def test_joint_copula_preserves_site_marginals_without_semantic_pooling():
    marginal = {
        "HONDA": np.asarray([[-2, -1], [0, 1], [2, 3]], dtype=float),
        "SPARK": np.asarray([[10, 11], [20, 21], [30, 31]], dtype=float),
    }
    template = {
        "HONDA": np.asarray([[2, 2], [1, 1], [3, 3]], dtype=float),
        "SPARK": np.asarray([[3, 3], [1, 1], [2, 2]], dtype=float),
    }
    coupled = joint_empirical_copula(
        marginal, template,
        {"HONDA": "SIGNED_GRID_EXCHANGE", "SPARK": "NONNEGATIVE_DEMAND"},
    )
    for site, values in marginal.items():
        assert np.array_equal(np.sort(coupled[site], axis=0), np.sort(values, axis=0))
