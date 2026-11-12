from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from honda.shadow.adaptive import AdaptiveCalibrationState, load_state, write_state_once
from honda.shadow.learning import (
    OperatorAcceptanceModel,
    OperatorModificationModel,
    RidgeTransitionModel,
    infer_nonnegative_preferences,
    value_of_information_ablation,
)
from honda.shadow.multiday import (
    build_multiday_residual_blocks,
    construct_information_tree,
    select_conditional_blocks,
)
from honda.shadow.protocol import (
    ProspectiveProtocol,
    economic_margin_interval,
    phase2_exit_report,
    switchback_assignment,
)


def _protocol() -> ProspectiveProtocol:
    return ProspectiveProtocol(
        "P2-001", "PARTNER", "2026-01-01T00:00:00+00:00", "OPERATOR_AS_OPERATED",
        "DETERMINISTIC_ML", "CALIBRATED_STOCHASTIC", ("complete approved issue",),
        "missed deadlines count as failures", "TARIFF-1", ("accepted_hard_constraint_violations",),
        "economic_difference_eur", 5.0, True, 8,
    )


def test_adaptive_state_is_hash_linked_versioned_and_immutable(tmp_path):
    initial = AdaptiveCalibrationState.initial("P", "M", "2026-01-01T00:00:00+00:00")
    for day in range(3):
        initial = initial.update(np.zeros(10), np.full(10, day + 1.0), f"2026-01-0{day + 2}T00:00:00+00:00")
    assert initial.sequence == 3
    assert initial.previous_state_hash is not None
    assert initial.radius_kw() is not None
    path = tmp_path / "state.json"
    write_state_once(path, initial)
    assert load_state(path).state_hash == initial.state_hash
    altered = initial.update(np.zeros(2), np.ones(2), "2026-01-06T00:00:00+00:00")
    with pytest.raises(FileExistsError):
        write_state_once(path, altered)


def test_multiday_blocks_are_causal_and_tree_branches_only_at_updates():
    stamps = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "timestamp_utc": stamps,
        "available_at_utc": stamps + pd.Timedelta(hours=1),
        "actual_kw": np.sin(np.arange(120) / 5),
        "forecast_kw": np.zeros(120),
    })
    blocks = build_multiday_residual_blocks(frame, block_hours=48, stride_hours=24)
    selected = select_conditional_blocks(blocks, blocks[-1].descriptor, "2026-01-05T00:00:00+00:00", 2)
    assert all(pd.Timestamp(block.issue_time_utc) <= pd.Timestamp("2026-01-05T00:00:00+00:00") for block in selected)
    scenarios = np.asarray([[0, 0, -2, -3, -4], [0, 0, 2, 3, 4]], dtype=float)
    tree_stamps = tuple(pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC").astype(str))
    nodes = construct_information_tree(scenarios, tree_stamps, (tree_stamps[2],))
    assert np.all(nodes[:, :2] == "root")
    assert nodes[0, 2] != nodes[1, 2]


def test_partner_learning_models_and_information_ablation():
    x = np.arange(40, dtype=float).reshape(20, 2)
    transition = RidgeTransitionModel.fit(x, 2 + x @ np.asarray([.3, -.1]), ("state", "delay"), ridge=.01)
    assert transition.predict(x).shape == (20,)
    accepted = np.asarray([0] * 10 + [1] * 10)
    operator = OperatorAcceptanceModel.fit(x, accepted, ("cost", "risk"), iterations=100)
    assert np.all((operator.predict_probability(x) > 0) & (operator.predict_probability(x) < 1))
    modifications = np.column_stack((x[:, 0] * .1, x[:, 1] * -.2))
    modification_model = OperatorModificationModel.fit(x, modifications, ("cost", "risk"), ridge=.01)
    assert modification_model.predict(x).shape == modifications.shape
    weights = infer_nonnegative_preferences(np.zeros((4, 2)), np.asarray([[1, 2], [2, 1], [1, 1], [3, 1]]))
    assert np.isclose(weights.sum(), 1) and np.all(weights >= 0)
    voi = value_of_information_ablation(np.asarray([10, 11]), {"weather": np.asarray([12, 13])})
    assert voi["weather"]["mean_value_eur"] == 2


def test_protocol_switchback_and_fail_closed_exit_report(tmp_path):
    protocol = _protocol(); protocol.validate()
    assert switchback_assignment(protocol, "week-1") in {"DETERMINISTIC_ML", "CALIBRATED_STOCHASTIC"}
    interval = economic_margin_interval(np.asarray([-2, -1, 0, 1]), samples=100)
    metrics = {
        "prospective_weeks": 2,
        "accepted_hard_constraint_violations": 0,
        "deadline_success_rate": 1,
        "multiday_tail_coverage": True,
        "tail_cost_eur": -1,
        "operator_interventions": 0,
        "violation_duration_hours": 0,
        "operator_acceptance_rate": .8,
        "operator_preparation_minutes": 3,
        "economic_difference_eur": interval,
    }
    report = phase2_exit_report(protocol, metrics)
    assert report["status"] == "PHASE2_EXIT_NOT_YET_MET"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol.__dict__), encoding="utf-8")
    assert ProspectiveProtocol.load(path).protocol_hash == protocol.protocol_hash
