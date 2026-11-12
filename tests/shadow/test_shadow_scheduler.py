from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from honda.shadow.contracts import (
    REQUIRED_PARTNER_INPUTS,
    CommercialTerms,
    Evidence,
    PartnerReadiness,
    RollingDecisionInput,
    ShadowAsset,
    TwoStageConfig,
)
from honda.shadow.optimizer import solve_deterministic_fallback, solve_two_stage
from honda.shadow.workflow import (
    capture_issue_snapshot,
    cohort_metrics,
    issue_recommendation,
    record_operator_decision,
    settle_shadow,
)


def _case():
    horizon = 48
    timestamps = tuple(str(x) for x in np.arange(np.datetime64("2026-01-02T00:00"), np.datetime64("2026-01-04T00:00"), np.timedelta64(1, "h")).astype("datetime64[s]"))
    point = np.full(horizon, 50.0)
    scenarios = np.vstack([point - 3, point + 3])
    nodes = np.asarray([["root" if t < 16 else "low" for t in range(horizon)], ["root" if t < 16 else "high" for t in range(horizon)]])
    decision = RollingDecisionInput(
        issue_time_utc="2026-01-01T12:00:00+00:00",
        deadline_time_utc="2026-01-01T14:00:00+00:00",
        timestamps_utc=tuple(f"{stamp}+00:00" for stamp in timestamps),
        interval_hours=tuple(float(x) for x in np.ones(horizon)),
        point_forecast_kw=tuple(float(x) for x in point),
        scenario_matrix_kw=tuple(tuple(float(x) for x in row) for row in scenarios),
        scenario_weights=(0.5, 0.5),
        scenario_node_ids=tuple(tuple(str(x) for x in row) for row in nodes),
        price_publication_time_utc="2026-01-01T13:00:00+00:00",
        availability=tuple(True for _ in range(horizon)),
        provenance_hashes={"forecast": "test"},
    )
    asset = ShadowAsset("process-1", "deferrable_process", 8, 8, -20, 20, 4, -2, 5, approved=True)
    prices = np.r_[np.full(16, .08), np.full(16, .30), np.full(16, .10)]
    commercial = CommercialTerms(tuple(float(x) for x in prices), tuple(0.0 for _ in range(horizon)), tuple(.02 for _ in range(horizon)), demand_charge_eur_per_kw=.1, imbalance_eur_per_kwh=.01, recourse_eur_per_kwh=.01, service_state_eur_per_kwh_interval=.001, label="TEST_APPROVED")
    config = TwoStageConfig(24, 8, 2, 70, 0, True)
    return decision, asset, commercial, config


def test_two_stage_optimizer_carries_state_and_enforces_nonanticipativity():
    decision, asset, commercial, config = _case()
    result = solve_two_stage(decision, asset, commercial, config)
    assert result.solver_status == "optimal"
    assert result.metadata["physical_hours"] == 48
    assert result.metadata["genuine_recourse"] is True
    assert result.nonanticipativity_max_residual_kw < 1e-8
    assert np.allclose(result.scenario_state_kwh[:, 0], asset.initial_state_kwh)
    assert np.allclose(result.scenario_state_kwh[:, -1], asset.terminal_state_kwh)
    assert np.allclose(result.scenario_recourse_kw[0, :16], result.scenario_recourse_kw[1, :16])


def test_deterministic_fallback_has_no_recourse():
    decision, asset, commercial, config = _case()
    fallback = solve_deterministic_fallback(decision, asset, commercial, config)
    assert fallback.solver_status == "optimal"
    assert np.max(np.abs(fallback.scenario_recourse_kw)) < 1e-9


def test_two_stage_cvar_is_part_of_the_optimized_objective():
    decision, asset, commercial, config = _case()
    risk_config = replace(config, risk_lambda=.4, cvar_alpha=.9)
    result = solve_two_stage(decision, asset, commercial, risk_config)
    assert result.solver_status == "optimal"
    assert result.cvar_eur is not None
    assert result.objective_eur == pytest.approx(result.expected_cost_eur + .4 * result.cvar_eur, abs=1e-6)


def test_scenario_tree_cannot_merge_after_information_diverges():
    decision, _asset, _commercial, _config = _case()
    nodes = [list(row) for row in decision.scenario_node_ids]
    nodes[0][20] = "merged"; nodes[1][20] = "merged"
    broken = RollingDecisionInput(**{**asdict(decision), "scenario_node_ids": tuple(tuple(row) for row in nodes)})
    with pytest.raises(ValueError, match="cannot merge"):
        broken.arrays()


def _write_case(tmp_path: Path, approved: bool) -> tuple[Path, Path]:
    decision, asset, commercial, optimizer = _case()
    evidence = {
        name: asdict(Evidence(f"partner-register://{name}", approved, "energy_manager" if approved else None, "2025-12-01T12:00:00+00:00" if approved else None, "a" * 64 if approved else None))
        for name in REQUIRED_PARTNER_INPUTS
    }
    issue = {
        "schema_version": "shadow.issue.v1",
        "issue_id": "2026-01-01T1200Z",
        "partner_id": "TEST_PARTNER",
        "approved_evidence_hashes": {name: value["sha256"] for name, value in evidence.items() if value["approved"]},
        "decision": asdict(decision),
        "asset": asdict(asset),
        "commercial": asdict(commercial),
        "optimizer": asdict(optimizer),
    }
    issue_path = tmp_path / "issue.yaml"
    issue_path.write_text(yaml.safe_dump(issue, sort_keys=False), encoding="utf-8")
    readiness = PartnerReadiness("TEST_PARTNER", "Europe/Berlin", "industrial_owner", evidence)
    readiness_path = tmp_path / "partner.yaml"
    readiness_path.write_text(yaml.safe_dump(asdict(readiness), sort_keys=False), encoding="utf-8")
    return issue_path, readiness_path


def test_readiness_gate_blocks_unapproved_shadow_recommendation(tmp_path):
    issue, partner = _write_case(tmp_path, approved=False)
    snapshot = capture_issue_snapshot(issue, partner, tmp_path / "snapshots")
    with pytest.raises(ValueError, match="readiness gate"):
        issue_recommendation(snapshot)


def test_snapshot_rejects_issue_not_bound_to_approved_evidence(tmp_path):
    issue, partner = _write_case(tmp_path, approved=True)
    value = yaml.safe_load(issue.read_text(encoding="utf-8"))
    value["approved_evidence_hashes"]["tariff_rules"] = "b" * 64
    issue.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exact approved evidence"):
        capture_issue_snapshot(issue, partner, tmp_path / "snapshots")


def test_shadow_record_is_immutable_and_settles_realized_operation(tmp_path):
    issue, partner = _write_case(tmp_path, approved=True)
    snapshot = capture_issue_snapshot(issue, partner, tmp_path / "snapshots")
    recommendation = issue_recommendation(snapshot)
    assert recommendation["claim_boundary"] == "PROSPECTIVE_SHADOW_ONLY_NO_AUTOMATIC_CONTROL"
    assert recommendation["explanation"]["tail_scenario"]["scenario_index"] in {0, 1}
    record_operator_decision(snapshot, "APPROVED", "shift_operator")
    schedule = recommendation["recommended"]["day_ahead_schedule_kw"]
    settlement = settle_shadow(snapshot, [50.0] * 48, schedule)
    assert settlement["hard_constraint_violation"] is False
    assert settlement["claim_boundary"] == "SHADOW_COMPARISON_NOT_CAUSAL_SAVINGS"
    metrics = cohort_metrics(tmp_path / "snapshots")
    assert metrics["recommendations_issued"] == 1
    assert metrics["accepted"] == 1
    assert metrics["settlements_recorded"] == 1
    assert metrics["hard_constraint_violation_count"] == 0
    with pytest.raises(FileExistsError):
        record_operator_decision(snapshot, "REJECTED", "shift_operator", "PRODUCTION_RISK")


def test_rejection_requires_reason_code(tmp_path):
    issue, partner = _write_case(tmp_path, approved=True)
    snapshot = capture_issue_snapshot(issue, partner, tmp_path / "snapshots")
    issue_recommendation(snapshot)
    with pytest.raises(ValueError, match="reason code"):
        record_operator_decision(snapshot, "REJECTED", "shift_operator")
