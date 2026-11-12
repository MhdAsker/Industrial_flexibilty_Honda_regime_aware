from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return sha256(stable_json(value).encode("utf-8")).hexdigest()


class DeploymentStage(str, Enum):
    OFFLINE_SIMULATION = "OFFLINE_SIMULATION"
    PROSPECTIVE_SHADOW = "PROSPECTIVE_SHADOW"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    LIMITED_CONTROLLED_PILOT = "LIMITED_CONTROLLED_PILOT"
    AUTOMATED_ACTION = "AUTOMATED_ACTION"


STAGE_ORDER = tuple(DeploymentStage)


@dataclass(frozen=True)
class SiteRegistration:
    tenant_id: str
    site_id: str
    name: str
    timezone: str
    connection_import_limit_kw: float | None
    connection_export_limit_kw: float | None
    owner_role: str
    version: int = 1
    approved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not all((self.tenant_id.strip(), self.site_id.strip(), self.name.strip(), self.owner_role.strip())):
            raise ValueError("Site identity, name, and owner role are required")
        ZoneInfo(self.timezone)
        if self.version < 1:
            raise ValueError("Site registry version must be positive")
        if any(value is not None and value < 0 for value in (self.connection_import_limit_kw, self.connection_export_limit_kw)):
            raise ValueError("Connection limits must be nonnegative")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class AssetRegistration:
    tenant_id: str
    site_id: str
    asset_id: str
    asset_type: str
    model_version: str
    approved: bool
    parameters: dict[str, Any]
    evidence_sha256: str
    version: int = 1

    def validate(self) -> None:
        if not all((self.tenant_id, self.site_id, self.asset_id, self.asset_type, self.model_version)):
            raise ValueError("Asset identity, type, and model version are required")
        if self.version < 1:
            raise ValueError("Asset registry version must be positive")
        if len(self.evidence_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in self.evidence_sha256):
            raise ValueError("Asset registration requires a valid evidence SHA-256")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ConnectorRegistration:
    tenant_id: str
    site_id: str
    connector_id: str
    connector_type: str
    endpoint: str
    credential_ref: str
    maximum_age_seconds: int
    enabled: bool = True
    version: int = 1

    def validate(self) -> None:
        allowed = {
            "meter_scada", "production_plan", "asset_availability", "weather",
            "market_price", "tariff_contract", "operator_override", "settlement",
        }
        if self.connector_type not in allowed:
            raise ValueError("Unsupported connector type")
        if not self.credential_ref.startswith(("vault://", "keyring://")):
            raise ValueError("Credentials must be indirect vault:// or keyring:// references")
        if self.maximum_age_seconds <= 0 or self.version < 1:
            raise ValueError("Connector freshness and version must be positive")
        endpoint_lower = self.endpoint.lower()
        if any(token in endpoint_lower for token in ("password=", "token=", "apikey=", "api_key=")):
            raise ValueError("Connector endpoints must not embed credentials")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ModelRegistration:
    tenant_id: str
    site_id: str | None
    model_id: str
    model_kind: str
    version: str
    artifact_sha256: str
    status: str = "CHALLENGER"
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.model_kind not in {"forecast", "scenario", "optimizer", "tariff"}:
            raise ValueError("Unsupported model kind")
        if self.status not in {"CHALLENGER", "CHAMPION", "RETIRED"}:
            raise ValueError("Unsupported model status")
        if len(self.artifact_sha256) != 64:
            raise ValueError("Model artifact requires a SHA-256 digest")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PromotionEvidence:
    tenant_id: str
    from_stage: DeploymentStage
    to_stage: DeploymentStage
    evidence: dict[str, str]
    approved_by: tuple[str, ...]
    approved_at_utc: str
    safety_case_sha256: str | None = None

    def validate(self) -> None:
        current = STAGE_ORDER.index(self.from_stage)
        target = STAGE_ORDER.index(self.to_stage)
        if target != current + 1:
            raise ValueError("Deployment promotion must advance exactly one stage")
        stamp = pd.Timestamp(self.approved_at_utc)
        if stamp.tzinfo is None:
            raise ValueError("Promotion approval time must be timezone-aware")
        required = {"owner_approval", "rollback_test", "fallback_test", "audit_verification"}
        if self.to_stage in {DeploymentStage.LIMITED_CONTROLLED_PILOT, DeploymentStage.AUTOMATED_ACTION}:
            required |= {"prospective_pilot", "incident_review"}
        if self.to_stage == DeploymentStage.AUTOMATED_ACTION:
            required |= {"seasonal_value_evidence", "disaster_recovery_test"}
            if self.safety_case_sha256 is None or len(self.safety_case_sha256) != 64:
                raise ValueError("Automated action requires a separately approved safety case")
            if len(self.approved_by) < 2:
                raise ValueError("Automated action requires two-person approval")
        missing = sorted(required - self.evidence.keys())
        if missing:
            raise ValueError(f"Promotion evidence is incomplete: {', '.join(missing)}")
