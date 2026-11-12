from __future__ import annotations

import json
import os
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import scrypt, sha256
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    AssetRegistration,
    ConnectorRegistration,
    DeploymentStage,
    ModelRegistration,
    PromotionEvidence,
    SiteRegistration,
    stable_hash,
    stable_json,
)
from .ledger import PredictionLedgerRecord, paired_ledger_comparison

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sites (
  tenant_id TEXT NOT NULL, site_id TEXT NOT NULL, version INTEGER NOT NULL,
  payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
  PRIMARY KEY (tenant_id, site_id, version)
);
CREATE TABLE IF NOT EXISTS assets (
  tenant_id TEXT NOT NULL, site_id TEXT NOT NULL, asset_id TEXT NOT NULL, version INTEGER NOT NULL,
  payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
  PRIMARY KEY (tenant_id, site_id, asset_id, version)
);
CREATE TABLE IF NOT EXISTS connectors (
  tenant_id TEXT NOT NULL, site_id TEXT NOT NULL, connector_id TEXT NOT NULL, version INTEGER NOT NULL,
  payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
  PRIMARY KEY (tenant_id, site_id, connector_id, version)
);
CREATE TABLE IF NOT EXISTS models (
  tenant_id TEXT NOT NULL, model_id TEXT NOT NULL, version TEXT NOT NULL,
  payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
  PRIMARY KEY (tenant_id, model_id, version)
);
CREATE TABLE IF NOT EXISTS model_aliases (
  tenant_id TEXT NOT NULL, site_key TEXT NOT NULL, model_kind TEXT NOT NULL,
  model_id TEXT NOT NULL, version TEXT NOT NULL, evidence_json TEXT NOT NULL,
  promoted_at_utc TEXT NOT NULL,
  PRIMARY KEY (tenant_id, site_key, model_kind)
);
CREATE TABLE IF NOT EXISTS deployment_stage (
  tenant_id TEXT PRIMARY KEY, stage TEXT NOT NULL, evidence_json TEXT NOT NULL, updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  tenant_id TEXT NOT NULL, run_id TEXT NOT NULL, site_id TEXT,
  state TEXT NOT NULL, issue_time_utc TEXT NOT NULL, deadline_time_utc TEXT NOT NULL,
  snapshot_json TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL,
  output_json TEXT, updated_at_utc TEXT NOT NULL,
  PRIMARY KEY (tenant_id, run_id)
);
CREATE TABLE IF NOT EXISTS decisions (
  tenant_id TEXT NOT NULL, decision_id TEXT NOT NULL, run_id TEXT NOT NULL,
  status TEXT NOT NULL, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
  supersedes_decision_id TEXT, created_at_utc TEXT NOT NULL,
  PRIMARY KEY (tenant_id, decision_id)
);
CREATE TABLE IF NOT EXISTS settlements (
  tenant_id TEXT NOT NULL, settlement_id TEXT NOT NULL, decision_id TEXT NOT NULL,
  payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, created_at_utc TEXT NOT NULL,
  PRIMARY KEY (tenant_id, settlement_id)
);
CREATE TABLE IF NOT EXISTS monitoring_events (
  tenant_id TEXT NOT NULL, event_id TEXT NOT NULL, site_id TEXT,
  severity TEXT NOT NULL, metric TEXT NOT NULL, payload_json TEXT NOT NULL, created_at_utc TEXT NOT NULL,
  PRIMARY KEY (tenant_id, event_id)
);
CREATE TABLE IF NOT EXISTS prediction_ledger (
  tenant_id TEXT NOT NULL, site_id TEXT NOT NULL, issue_id TEXT NOT NULL,
  target_timestamp_utc TEXT NOT NULL, model_id TEXT NOT NULL,
  prediction_id TEXT NOT NULL, scenario_id TEXT,
  information_cutoff_utc TEXT NOT NULL, target_semantics TEXT NOT NULL,
  prediction_kw REAL NOT NULL, label_kw REAL,
  model_sha256 TEXT NOT NULL, feature_sha256 TEXT NOT NULL,
  code_sha256 TEXT NOT NULL, environment_sha256 TEXT NOT NULL,
  snapshot_sha256 TEXT NOT NULL, calibration_version TEXT NOT NULL,
  operator_action TEXT, decision_outcome TEXT, label_status TEXT NOT NULL,
  created_at_utc TEXT NOT NULL, labeled_at_utc TEXT,
  PRIMARY KEY (tenant_id, site_id, issue_id, target_timestamp_utc, model_id)
);
CREATE INDEX IF NOT EXISTS prediction_ledger_model_idx
  ON prediction_ledger (tenant_id, model_id, label_status);
CREATE TABLE IF NOT EXISTS principals (
  tenant_id TEXT NOT NULL, principal_id TEXT NOT NULL, role TEXT NOT NULL,
  site_scope_json TEXT NOT NULL, token_salt BLOB NOT NULL, token_hash BLOB NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (tenant_id, principal_id)
);
CREATE TABLE IF NOT EXISTS audit_log (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at_utc TEXT NOT NULL,
  tenant_id TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
  resource TEXT NOT NULL, payload_json TEXT NOT NULL, previous_hash TEXT NOT NULL,
  entry_hash TEXT NOT NULL UNIQUE
);
"""


ROLE_PERMISSIONS = {
    "viewer": {"read"},
    "operator": {"read", "approve", "reject", "modify"},
    "site_admin": {"read", "approve", "reject", "modify", "register"},
    "platform_admin": {"read", "approve", "reject", "modify", "register", "promote", "rollback"},
    "auditor": {"read", "audit"},
}


class PlatformStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _audit(self, connection: sqlite3.Connection, tenant: str, actor: str, action: str, resource: str, payload: Any) -> str:
        previous = connection.execute("SELECT entry_hash FROM audit_log ORDER BY sequence DESC LIMIT 1").fetchone()
        previous_hash = "GENESIS" if previous is None else str(previous[0])
        occurred = self.now()
        payload_json = stable_json(payload)
        entry_hash = stable_hash({
            "occurred_at_utc": occurred, "tenant_id": tenant, "actor": actor,
            "action": action, "resource": resource, "payload_json": payload_json,
            "previous_hash": previous_hash,
        })
        connection.execute(
            "INSERT INTO audit_log (occurred_at_utc, tenant_id, actor, action, resource, payload_json, previous_hash, entry_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (occurred, tenant, actor, action, resource, payload_json, previous_hash, entry_hash),
        )
        return entry_hash

    def verify_audit_chain(self) -> dict[str, Any]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM audit_log ORDER BY sequence").fetchall()
        previous = "GENESIS"
        for row in rows:
            expected = stable_hash({
                "occurred_at_utc": row["occurred_at_utc"], "tenant_id": row["tenant_id"],
                "actor": row["actor"], "action": row["action"], "resource": row["resource"],
                "payload_json": row["payload_json"], "previous_hash": previous,
            })
            if row["previous_hash"] != previous or row["entry_hash"] != expected:
                return {"valid": False, "failed_sequence": row["sequence"], "entries": len(rows)}
            previous = row["entry_hash"]
        return {"valid": True, "failed_sequence": None, "entries": len(rows), "head_hash": previous}

    def _register(self, table: str, identity: tuple[Any, ...], payload: dict[str, Any], actor: str) -> str:
        payload_json = stable_json(payload)
        digest = sha256(payload_json.encode("utf-8")).hexdigest()
        columns = {
            "sites": "tenant_id, site_id, version",
            "assets": "tenant_id, site_id, asset_id, version",
            "connectors": "tenant_id, site_id, connector_id, version",
            "models": "tenant_id, model_id, version",
        }[table]
        placeholders = ", ".join("?" for _ in identity)
        tenant = str(identity[0])
        with self.connection() as connection:
            connection.execute(
                f"INSERT INTO {table} ({columns}, payload_json, payload_sha256) VALUES ({placeholders}, ?, ?)",
                (*identity, payload_json, digest),
            )
            self._audit(connection, tenant, actor, "REGISTER", f"{table}:{':'.join(map(str, identity[1:]))}", {"sha256": digest})
        return digest

    def register_site(self, value: SiteRegistration, actor: str) -> str:
        payload = value.payload()
        digest = self._register("sites", (value.tenant_id, value.site_id, value.version), payload, actor)
        with self.connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO deployment_stage VALUES (?, ?, ?, ?)",
                (value.tenant_id, DeploymentStage.OFFLINE_SIMULATION.value, "{}", self.now()),
            )
        return digest

    def register_asset(self, value: AssetRegistration, actor: str) -> str:
        return self._register("assets", (value.tenant_id, value.site_id, value.asset_id, value.version), value.payload(), actor)

    def register_connector(self, value: ConnectorRegistration, actor: str) -> str:
        return self._register("connectors", (value.tenant_id, value.site_id, value.connector_id, value.version), value.payload(), actor)

    def register_model(self, value: ModelRegistration, actor: str) -> str:
        return self._register("models", (value.tenant_id, value.model_id, value.version), value.payload(), actor)

    def promote_model(self, tenant_id: str, model_id: str, version: str, evidence: dict[str, str], actor: str) -> None:
        required = {"evaluation_sha256", "approval_sha256", "rollback_model_id"}
        missing = sorted(required - evidence.keys())
        if missing:
            raise ValueError(f"Model promotion evidence is incomplete: {', '.join(missing)}")
        with self.connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM models WHERE tenant_id = ? AND model_id = ? AND version = ?",
                (tenant_id, model_id, version),
            ).fetchone()
            if row is None:
                raise ValueError("Cannot promote an unregistered model")
            payload = json.loads(row[0])
            if payload.get("site_id") is None:
                global_required = {"ledger_comparison_sha256", "promotion_report_sha256"}
                global_missing = sorted(global_required - evidence.keys())
                if global_missing:
                    raise ValueError(f"Global model promotion evidence is incomplete: {', '.join(global_missing)}")
            site_key = payload.get("site_id") or "*"
            kind = payload["model_kind"]
            connection.execute(
                "INSERT INTO model_aliases VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(tenant_id, site_key, model_kind) DO UPDATE SET model_id=excluded.model_id, version=excluded.version, evidence_json=excluded.evidence_json, promoted_at_utc=excluded.promoted_at_utc",
                (tenant_id, site_key, kind, model_id, version, stable_json(evidence), self.now()),
            )
            self._audit(connection, tenant_id, actor, "PROMOTE_MODEL", f"model-alias:{site_key}:{kind}", {"model_id": model_id, "version": version, "evidence": evidence})

    def champion_models(self, tenant_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM model_aliases WHERE tenant_id = ?", (tenant_id,)).fetchall()
        return [{**dict(row), "evidence": json.loads(row["evidence_json"])} for row in rows]

    def list_payloads(self, table: str, tenant_id: str, site_id: str | None = None) -> list[dict[str, Any]]:
        if table not in {"sites", "assets", "connectors", "models", "runs", "decisions", "settlements", "monitoring_events"}:
            raise ValueError("Unsupported platform collection")
        query = f"SELECT * FROM {table} WHERE tenant_id = ?"
        parameters: list[Any] = [tenant_id]
        if site_id is not None and table in {"sites", "assets", "connectors", "runs", "monitoring_events"}:
            query += " AND site_id = ?"
            parameters.append(site_id)
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for field in ("payload_json", "snapshot_json", "output_json"):
                if field in item and item[field] is not None:
                    item[field.removesuffix("_json")] = json.loads(item.pop(field))
            result.append(item)
        return result

    def create_principal(self, tenant_id: str, principal_id: str, role: str, site_scope: tuple[str, ...], actor: str) -> str:
        if role not in ROLE_PERMISSIONS:
            raise ValueError("Unsupported platform role")
        token = secrets.token_urlsafe(32)
        salt = os.urandom(16)
        digest = scrypt(token.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO principals VALUES (?, ?, ?, ?, ?, ?, 1)",
                (tenant_id, principal_id, role, stable_json(site_scope), salt, digest),
            )
            self._audit(connection, tenant_id, actor, "CREATE_PRINCIPAL", f"principal:{principal_id}", {"role": role, "site_scope": site_scope})
        return token

    def authenticate(self, token: str, required_permission: str, tenant_id: str, site_id: str | None = None) -> dict[str, Any]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM principals WHERE tenant_id = ? AND enabled = 1", (tenant_id,)).fetchall()
        for row in rows:
            candidate = scrypt(token.encode("utf-8"), salt=row["token_salt"], n=2**14, r=8, p=1)
            if secrets.compare_digest(candidate, row["token_hash"]):
                role = row["role"]
                scope = tuple(json.loads(row["site_scope_json"]))
                if required_permission not in ROLE_PERMISSIONS[role]:
                    raise PermissionError("Principal lacks required permission")
                if site_id is not None and scope and site_id not in scope:
                    raise PermissionError("Principal is outside the requested site scope")
                return {"principal_id": row["principal_id"], "role": role, "site_scope": scope}
        raise PermissionError("Invalid platform credential")

    def deployment_stage(self, tenant_id: str) -> DeploymentStage:
        with self.connection() as connection:
            row = connection.execute("SELECT stage FROM deployment_stage WHERE tenant_id = ?", (tenant_id,)).fetchone()
        return DeploymentStage.OFFLINE_SIMULATION if row is None else DeploymentStage(row[0])

    def promote(self, evidence: PromotionEvidence, actor: str) -> None:
        evidence.validate()
        current = self.deployment_stage(evidence.tenant_id)
        if current != evidence.from_stage:
            raise ValueError(f"Tenant is at {current.value}, not {evidence.from_stage.value}")
        payload = {
            "from": evidence.from_stage.value, "to": evidence.to_stage.value,
            "evidence": evidence.evidence, "approved_by": evidence.approved_by,
            "approved_at_utc": evidence.approved_at_utc, "safety_case_sha256": evidence.safety_case_sha256,
        }
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO deployment_stage VALUES (?, ?, ?, ?) ON CONFLICT(tenant_id) DO UPDATE SET stage=excluded.stage, evidence_json=excluded.evidence_json, updated_at_utc=excluded.updated_at_utc",
                (evidence.tenant_id, evidence.to_stage.value, stable_json(payload), self.now()),
            )
            self._audit(connection, evidence.tenant_id, actor, "PROMOTE", "deployment-stage", payload)

    def save_run_snapshot(self, tenant_id: str, run_id: str, site_id: str | None, issue_time_utc: str, deadline_time_utc: str, snapshot: dict[str, Any], actor: str) -> str:
        snapshot_json = stable_json(snapshot)
        digest = sha256(snapshot_json.encode("utf-8")).hexdigest()
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (tenant_id, run_id, site_id, "SNAPSHOT", issue_time_utc, deadline_time_utc, snapshot_json, digest, self.now()),
            )
            self._audit(connection, tenant_id, actor, "SNAPSHOT", f"run:{run_id}", {"sha256": digest, "site_id": site_id})
        return digest

    def transition_run(self, tenant_id: str, run_id: str, expected_state: str, new_state: str, output: dict[str, Any], actor: str) -> None:
        with self.connection() as connection:
            row = connection.execute("SELECT state FROM runs WHERE tenant_id = ? AND run_id = ?", (tenant_id, run_id)).fetchone()
            if row is None or row[0] != expected_state:
                raise ValueError("Run transition does not match the persisted state")
            connection.execute(
                "UPDATE runs SET state = ?, output_json = ?, updated_at_utc = ? WHERE tenant_id = ? AND run_id = ?",
                (new_state, stable_json(output), self.now(), tenant_id, run_id),
            )
            self._audit(connection, tenant_id, actor, "TRANSITION", f"run:{run_id}", {"from": expected_state, "to": new_state, "output_sha256": stable_hash(output)})

    def save_decision(self, tenant_id: str, decision_id: str, run_id: str, status: str, payload: dict[str, Any], actor: str, supersedes: str | None = None) -> str:
        if status not in {"PENDING", "APPROVED", "REJECTED", "MODIFIED", "ROLLED_BACK", "FALLBACK"}:
            raise ValueError("Unsupported decision status")
        digest = stable_hash(payload)
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tenant_id, decision_id, run_id, status, stable_json(payload), digest, supersedes, self.now()),
            )
            self._audit(connection, tenant_id, actor, "DECISION", f"decision:{decision_id}", {"status": status, "sha256": digest, "supersedes": supersedes})
        return digest

    def save_settlement(self, tenant_id: str, settlement_id: str, decision_id: str, payload: dict[str, Any], actor: str) -> str:
        digest = stable_hash(payload)
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO settlements VALUES (?, ?, ?, ?, ?, ?)",
                (tenant_id, settlement_id, decision_id, stable_json(payload), digest, self.now()),
            )
            self._audit(connection, tenant_id, actor, "SETTLEMENT", f"settlement:{settlement_id}", {"decision_id": decision_id, "sha256": digest})
        return digest

    def save_monitoring_event(self, tenant_id: str, event_id: str, site_id: str | None, severity: str, metric: str, payload: dict[str, Any], actor: str = "monitor") -> None:
        if severity not in {"INFO", "WARNING", "CRITICAL"}:
            raise ValueError("Unsupported monitoring severity")
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO monitoring_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (tenant_id, event_id, site_id, severity, metric, stable_json(payload), self.now()),
            )
            self._audit(connection, tenant_id, actor, "MONITOR", f"event:{event_id}", {"severity": severity, "metric": metric})

    def append_prediction(self, record: PredictionLedgerRecord, actor: str) -> None:
        payload = record.payload()
        columns = tuple(payload)
        with self.connection() as connection:
            connection.execute(
                f"INSERT INTO prediction_ledger ({', '.join(columns)}, created_at_utc, labeled_at_utc) "
                f"VALUES ({', '.join('?' for _ in columns)}, ?, ?)",
                (*[payload[column] for column in columns], self.now(), self.now() if record.label_status != "PENDING" else None),
            )
            self._audit(
                connection, record.tenant_id, actor, "LEDGER_PREDICTION",
                f"prediction:{record.site_id}:{record.issue_id}:{record.target_timestamp_utc}:{record.model_id}",
                {"prediction_id": record.prediction_id, "snapshot_sha256": record.snapshot_sha256},
            )

    def attach_label(
        self,
        tenant_id: str,
        site_id: str,
        issue_id: str,
        target_timestamp_utc: str,
        model_id: str,
        label_kw: float | None,
        actor: str,
        *,
        unavailable: bool = False,
        operator_action: str | None = None,
        decision_outcome: str | None = None,
    ) -> None:
        status = "UNAVAILABLE" if unavailable else "OBSERVED"
        if (unavailable and label_kw is not None) or (not unavailable and (label_kw is None or not np.isfinite(label_kw))):
            raise ValueError("Ledger label value does not match its status")
        identity = (tenant_id, site_id, issue_id, target_timestamp_utc, model_id)
        with self.connection() as connection:
            row = connection.execute(
                "SELECT label_status FROM prediction_ledger WHERE tenant_id=? AND site_id=? AND issue_id=? AND target_timestamp_utc=? AND model_id=?",
                identity,
            ).fetchone()
            if row is None or row[0] != "PENDING":
                raise ValueError("Ledger label can only be attached once to a pending prediction")
            connection.execute(
                "UPDATE prediction_ledger SET label_kw=?, label_status=?, operator_action=?, decision_outcome=?, labeled_at_utc=? "
                "WHERE tenant_id=? AND site_id=? AND issue_id=? AND target_timestamp_utc=? AND model_id=?",
                (label_kw, status, operator_action, decision_outcome, self.now(), *identity),
            )
            self._audit(
                connection, tenant_id, actor, "LEDGER_LABEL",
                f"prediction:{site_id}:{issue_id}:{target_timestamp_utc}:{model_id}",
                {"label_status": status, "operator_action": operator_action, "decision_outcome": decision_outcome},
            )

    def ledger_rows(self, tenant_id: str, model_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM prediction_ledger WHERE tenant_id = ?"
        parameters: list[Any] = [tenant_id]
        if model_id is not None:
            query += " AND model_id = ?"; parameters.append(model_id)
        query += " ORDER BY site_id, issue_id, target_timestamp_utc, model_id"
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(query, parameters).fetchall()]

    def compare_ledger_models(self, tenant_id: str, champion_model_id: str, challenger_model_id: str) -> dict[str, Any]:
        return paired_ledger_comparison(
            self.ledger_rows(tenant_id, champion_model_id),
            self.ledger_rows(tenant_id, challenger_model_id),
        )
