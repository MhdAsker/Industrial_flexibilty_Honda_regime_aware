from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .storage import PlatformStore


def create_app(database: str | Path):
    """Create the optional FastAPI service without making it a core dependency."""
    try:
        from fastapi import FastAPI, Header, HTTPException, Query
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("Install the platform extra: pip install -e '.[platform]'") from exc

    store = PlatformStore(database)
    app = FastAPI(title="FlexOps Multi-site Platform", version="0.1.0")

    def authorize(token: str | None, tenant_id: str, permission: str = "read", site_id: str | None = None) -> dict[str, Any]:
        if token is None:
            raise HTTPException(status_code=401, detail="Missing X-API-Key")
        try:
            return store.authenticate(token, permission, tenant_id, site_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    def collection(table: str, tenant_id: str, site_id: str | None, token: str | None) -> list[dict[str, Any]]:
        authorize(token, tenant_id, "read", site_id)
        return store.list_payloads(table, tenant_id, site_id)

    @app.get("/assets")
    def assets(tenant_id: str, site_id: str | None = None, x_api_key: str | None = Header(default=None)):
        return collection("assets", tenant_id, site_id, x_api_key)

    @app.get("/tariffs")
    def tariffs(tenant_id: str, x_api_key: str | None = Header(default=None)):
        return [row for row in collection("models", tenant_id, None, x_api_key) if row.get("payload", {}).get("model_kind") == "tariff"]

    @app.get("/models")
    def models(tenant_id: str, x_api_key: str | None = Header(default=None)):
        authorize(x_api_key, tenant_id)
        return {"registry": store.list_payloads("models", tenant_id), "champions": store.champion_models(tenant_id)}

    @app.get("/forecasts")
    def forecasts(tenant_id: str, site_id: str | None = None, x_api_key: str | None = Header(default=None)):
        rows = collection("runs", tenant_id, site_id, x_api_key)
        return [{"run_id": row["run_id"], "forecast": row.get("output", {}).get("forecast")} for row in rows if row.get("output", {}).get("forecast") is not None]

    @app.get("/scenarios")
    def scenarios(tenant_id: str, site_id: str | None = None, x_api_key: str | None = Header(default=None)):
        rows = collection("runs", tenant_id, site_id, x_api_key)
        return [{"run_id": row["run_id"], "scenarios": row.get("output", {}).get("scenarios")} for row in rows if row.get("output", {}).get("scenarios") is not None]

    @app.get("/decisions")
    def decisions(tenant_id: str, x_api_key: str | None = Header(default=None)):
        return collection("decisions", tenant_id, None, x_api_key)

    @app.get("/settlements")
    def settlements(tenant_id: str, x_api_key: str | None = Header(default=None)):
        return collection("settlements", tenant_id, None, x_api_key)

    @app.get("/health")
    def health(tenant_id: str = Query(...), x_api_key: str | None = Header(default=None)):
        authorize(x_api_key, tenant_id)
        events = store.list_payloads("monitoring_events", tenant_id)
        critical = sum(row["severity"] == "CRITICAL" for row in events)
        audit = store.verify_audit_chain()
        return {
            "status": "CRITICAL" if critical or not audit["valid"] else "HEALTHY",
            "deployment_stage": store.deployment_stage(tenant_id).value,
            "critical_monitoring_events": critical,
            "audit_chain": audit,
        }

    @app.post("/decisions/{decision_id}/operator-action")
    def operator_action(decision_id: str, body: dict[str, Any], tenant_id: str, x_api_key: str | None = Header(default=None)):
        principal = authorize(x_api_key, tenant_id, "approve")
        disposition = str(body.get("disposition", ""))
        if disposition not in {"APPROVED", "REJECTED", "MODIFIED"}:
            raise HTTPException(status_code=422, detail="Invalid disposition")
        if disposition != "APPROVED" and not body.get("reason_code"):
            raise HTTPException(status_code=422, detail="Reason code required")
        new_id = f"{decision_id}-{disposition.lower()}"
        try:
            store.save_decision(
                tenant_id, new_id, str(body["run_id"]), disposition,
                {"operator_action": body, "control_authorized": False},
                principal["principal_id"], supersedes=decision_id,
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Decision action already recorded") from exc
        return {"decision_id": new_id, "disposition": disposition, "control_authorized": False}

    return app
