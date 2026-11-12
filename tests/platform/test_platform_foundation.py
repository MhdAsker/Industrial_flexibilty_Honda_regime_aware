from __future__ import annotations

import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from honda.platform.connectors import (
    ConnectorBatch,
    InMemoryConnector,
    capture_connector_snapshot,
)
from honda.platform.contracts import (
    DeploymentStage,
    ModelRegistration,
    PromotionEvidence,
    SiteRegistration,
)
from honda.platform.monitoring import (
    MonitoringThresholds,
    OperationalMetrics,
    evaluate_health,
)
from honda.platform.orchestration import PipelineComponents, WorkflowOrchestrator
from honda.platform.portfolio import PortfolioSite, allocate_portfolio
from honda.platform.storage import PlatformStore


def _site() -> SiteRegistration:
    return SiteRegistration("TENANT", "SITE_A", "Plant A", "Europe/Berlin", 500, 50, "energy_manager", approved=True)


def _batch(available: str = "2026-01-01T11:05:00+00:00") -> ConnectorBatch:
    return ConnectorBatch(
        "meter-a", "meter_scada", "2026-01-01T11:00:00+00:00", available,
        "2026-01-01T11:10:00+00:00", ({"timestamp_utc": "2026-01-01T11:00:00+00:00", "kw": 120.0},),
    )


def test_registry_audit_chain_and_tamper_detection(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    digest = store.register_site(_site(), "admin")
    assert len(digest) == 64
    with pytest.raises(sqlite3.IntegrityError):
        store.register_site(_site(), "admin")
    assert store.verify_audit_chain()["valid"] is True
    with store.connection() as connection:
        connection.execute("UPDATE audit_log SET payload_json = ? WHERE sequence = 1", ('{"tampered":true}',))
    assert store.verify_audit_chain()["valid"] is False


def test_api_credentials_are_hashed_and_role_and_site_scoped(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    store.register_site(_site(), "admin")
    token = store.create_principal("TENANT", "operator-1", "operator", ("SITE_A",), "admin")
    identity = store.authenticate(token, "approve", "TENANT", "SITE_A")
    assert identity["principal_id"] == "operator-1"
    with pytest.raises(PermissionError, match="site scope"):
        store.authenticate(token, "read", "TENANT", "SITE_B")
    with pytest.raises(PermissionError, match="permission"):
        store.authenticate(token, "register", "TENANT", "SITE_A")
    assert token not in (tmp_path / "platform.sqlite").read_bytes().decode("latin-1")


def test_connector_snapshot_rejects_future_information():
    future_batch = replace(
        _batch("2026-01-01T12:05:00+00:00"),
        retrieved_at_utc="2026-01-01T12:10:00+00:00",
    )
    connector = InMemoryConnector("meter-a", "meter_scada", 7200, future_batch)
    with pytest.raises(ValueError, match="unavailable at issue time"):
        capture_connector_snapshot((connector,), "2026-01-01T12:00:00+00:00")


def test_workflow_is_immutable_and_control_remains_stage_gated(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    store.register_site(_site(), "admin")
    connector = InMemoryConnector("meter-a", "meter_scada", 7200, _batch())
    components = PipelineComponents(
        forecast=lambda snapshot: {"point_kw": [121.0], "snapshot": snapshot["connectors"]["meter-a"]["sha256"]},
        scenarios=lambda data: {"scenario_kw": [[119.0], [123.0]], "forecast": data["forecast"]["point_kw"]},
        optimize=lambda data: {"solver_status": "optimal", "schedule_kw": [-2.0], "fallback_kw": [0.0]},
    )
    orchestrator = WorkflowOrchestrator(store)
    result = orchestrator.run(
        "TENANT", "SITE_A", "2026-01-01T12:00:00+00:00", "2026-01-01T13:00:00+00:00",
        (connector,), components, "scheduler", run_id="run-1",
    )
    assert result["state"] == "PENDING_APPROVAL"
    action = orchestrator.authorize_action("TENANT", result["decision_id"], result["run_id"], "operator-1", True)
    assert action["control_authorized"] is False
    assert "forbids_control" in action["reason"]
    assert store.verify_audit_chain()["valid"] is True


def test_automated_action_requires_sequential_promotion_and_safety_case(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    store.register_site(_site(), "admin")
    base = {"owner_approval": "a", "rollback_test": "b", "fallback_test": "c", "audit_verification": "d"}
    with pytest.raises(ValueError, match="exactly one stage"):
        PromotionEvidence("TENANT", DeploymentStage.OFFLINE_SIMULATION, DeploymentStage.AUTOMATED_ACTION, base, ("owner",), "2026-01-01T00:00:00+00:00").validate()
    for source, target in zip(tuple(DeploymentStage)[:-1], tuple(DeploymentStage)[1:]):
        evidence = dict(base)
        if target in {DeploymentStage.LIMITED_CONTROLLED_PILOT, DeploymentStage.AUTOMATED_ACTION}:
            evidence.update({"prospective_pilot": "e", "incident_review": "f"})
        if target == DeploymentStage.AUTOMATED_ACTION:
            evidence.update({"seasonal_value_evidence": "g", "disaster_recovery_test": "h"})
        promotion = PromotionEvidence(
            "TENANT", source, target, evidence,
            ("industrial_owner", "safety_owner") if target == DeploymentStage.AUTOMATED_ACTION else ("industrial_owner",),
            "2026-01-01T00:00:00+00:00", "a" * 64 if target == DeploymentStage.AUTOMATED_ACTION else None,
        )
        store.promote(promotion, "platform-admin")
    assert store.deployment_stage("TENANT") == DeploymentStage.AUTOMATED_ACTION
    orchestrator = WorkflowOrchestrator(store)
    blocked = orchestrator.authorize_action("TENANT", "decision-a", "run-a", "operator", True)
    assert blocked["control_authorized"] is False
    allowed = orchestrator.authorize_action(
        "TENANT", "decision-b", "run-b", "operator", True,
        {"status": "HEALTHY", "automatic_action_allowed": True},
    )
    assert allowed["control_authorized"] is True


def test_champion_promotion_requires_evidence_and_preserves_rollback_identity(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    model = ModelRegistration("TENANT", "SITE_A", "forecast-lgbm", "forecast", "2.0", "a" * 64)
    store.register_model(model, "ml-owner")
    with pytest.raises(ValueError, match="incomplete"):
        store.promote_model("TENANT", "forecast-lgbm", "2.0", {}, "ml-owner")
    evidence = {
        "evaluation_sha256": "b" * 64,
        "approval_sha256": "c" * 64,
        "rollback_model_id": "forecast-lgbm:1.0",
    }
    store.promote_model("TENANT", "forecast-lgbm", "2.0", evidence, "ml-owner")
    champions = store.champion_models("TENANT")
    assert champions[0]["model_id"] == "forecast-lgbm"
    assert champions[0]["evidence"]["rollback_model_id"] == "forecast-lgbm:1.0"


def test_monitoring_health_blocks_automatic_action_on_critical_failure():
    thresholds = MonitoringThresholds(900, .02, 5, 10, .85, 60, 20, 50)
    metrics = OperationalMetrics(100, .01, 1, 2, .9, .1, .1, True, 5, 0, 3, 100, 95)
    assert evaluate_health(metrics, thresholds)["automatic_action_allowed"] is True
    failed = replace(metrics, hard_constraint_violations=1)
    health = evaluate_health(failed, thresholds)
    assert health["status"] == "CRITICAL"
    assert health["automatic_action_allowed"] is False


def test_portfolio_allocation_respects_network_and_reports_diversification():
    sites = (
        PortfolioSite("A", 10, 1.0, "north"),
        PortfolioSite("B", 10, 1.2, "north"),
        PortfolioSite("C", 10, 1.5, "south"),
    )
    covariance = np.asarray([[1.0, .2, -.1], [.2, 1.2, .1], [-.1, .1, .8]])
    result = allocate_portfolio(sites, 15, covariance, {"north": 9, "south": 10}, risk_aversion=.1, risk_budget=100)
    assert result.status == "optimal"
    assert sum(result.allocation_kw.values()) == pytest.approx(15)
    assert result.allocation_kw["A"] + result.allocation_kw["B"] <= 9 + 1e-6
    assert result.portfolio_variance <= 100 + 1e-6
    assert result.diversification_value is not None


def test_api_exposes_required_operational_routes(tmp_path):
    pytest.importorskip("fastapi")
    from honda.platform.api import create_app

    app = create_app(tmp_path / "platform.sqlite")
    paths = {route.path for route in app.routes}
    required = {"/forecasts", "/scenarios", "/assets", "/tariffs", "/decisions", "/settlements", "/health"}
    assert required.issubset(paths)
