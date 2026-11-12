from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pandas as pd

from .connectors import LiveConnector, capture_connector_snapshot
from .contracts import DeploymentStage, stable_hash
from .storage import PlatformStore

PipelineFunction = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class PipelineComponents:
    forecast: PipelineFunction
    scenarios: PipelineFunction
    optimize: PipelineFunction


class WorkflowOrchestrator:
    """Synchronous reference orchestrator; production runners can call the same stages."""

    def __init__(self, store: PlatformStore) -> None:
        self.store = store

    def run(
        self,
        tenant_id: str,
        site_id: str,
        issue_time_utc: str,
        deadline_time_utc: str,
        connectors: tuple[LiveConnector, ...],
        components: PipelineComponents,
        actor: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        issue = pd.Timestamp(issue_time_utc)
        deadline = pd.Timestamp(deadline_time_utc)
        if issue.tzinfo is None or deadline.tzinfo is None or issue > deadline:
            raise ValueError("Workflow issue and deadline times are invalid")
        run_id = run_id or uuid4().hex
        snapshot = capture_connector_snapshot(connectors, issue_time_utc)
        snapshot["tenant_id"] = tenant_id; snapshot["site_id"] = site_id
        snapshot_hash = self.store.save_run_snapshot(tenant_id, run_id, site_id, issue_time_utc, deadline_time_utc, snapshot, actor)
        forecast = components.forecast(snapshot)
        self.store.transition_run(tenant_id, run_id, "SNAPSHOT", "FORECASTED", {"forecast": forecast}, actor)
        scenarios = components.scenarios({"snapshot": snapshot, "forecast": forecast})
        self.store.transition_run(tenant_id, run_id, "FORECASTED", "SCENARIOS", {"forecast": forecast, "scenarios": scenarios}, actor)
        optimization = components.optimize({"snapshot": snapshot, "forecast": forecast, "scenarios": scenarios})
        if optimization.get("solver_status") != "optimal":
            self.store.transition_run(tenant_id, run_id, "SCENARIOS", "FAILED", {"optimization": optimization}, actor)
            raise RuntimeError("Optimization did not return an optimal recommendation")
        self.store.transition_run(tenant_id, run_id, "SCENARIOS", "PENDING_APPROVAL", {"forecast": forecast, "scenarios": scenarios, "optimization": optimization}, actor)
        decision_id = stable_hash({"tenant": tenant_id, "run": run_id, "snapshot": snapshot_hash, "optimization": optimization})[:24]
        decision = {
            "site_id": site_id, "issue_time_utc": issue_time_utc, "deadline_time_utc": deadline_time_utc,
            "snapshot_sha256": snapshot_hash, "forecast": forecast, "scenarios": scenarios,
            "optimization": optimization, "control_authorized": False,
            "deployment_stage": self.store.deployment_stage(tenant_id).value,
        }
        self.store.save_decision(tenant_id, decision_id, run_id, "PENDING", decision, actor)
        return {"run_id": run_id, "decision_id": decision_id, "state": "PENDING_APPROVAL", "decision": decision}

    def authorize_action(
        self,
        tenant_id: str,
        decision_id: str,
        run_id: str,
        actor: str,
        human_approved: bool,
        health_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage = self.store.deployment_stage(tenant_id)
        if not human_approved:
            payload = {"control_authorized": False, "reason": "operator_rejected"}
            self.store.save_decision(tenant_id, f"{decision_id}-rejected", run_id, "REJECTED", payload, actor, supersedes=decision_id)
            return payload
        if stage in {DeploymentStage.OFFLINE_SIMULATION, DeploymentStage.PROSPECTIVE_SHADOW}:
            payload = {"control_authorized": False, "reason": f"deployment_stage_{stage.value}_forbids_control"}
            self.store.save_decision(tenant_id, f"{decision_id}-approved-shadow", run_id, "APPROVED", payload, actor, supersedes=decision_id)
            return payload
        health_ok = health_report is not None and health_report.get("status") == "HEALTHY" and health_report.get("automatic_action_allowed") is True
        payload = {
            "control_authorized": stage in {DeploymentStage.LIMITED_CONTROLLED_PILOT, DeploymentStage.AUTOMATED_ACTION} and health_ok,
            "human_approved": True,
            "deployment_stage": stage.value,
            "health_gate_passed": health_ok,
        }
        self.store.save_decision(tenant_id, f"{decision_id}-approved", run_id, "APPROVED", payload, actor, supersedes=decision_id)
        return payload

    def rollback_to_fallback(self, tenant_id: str, run_id: str, decision_id: str, fallback_payload: dict[str, Any], actor: str) -> str:
        rollback_id = f"{decision_id}-fallback-{stable_hash(fallback_payload)[:8]}"
        self.store.save_decision(tenant_id, rollback_id, run_id, "FALLBACK", {**fallback_payload, "control_authorized": False}, actor, supersedes=decision_id)
        return rollback_id
