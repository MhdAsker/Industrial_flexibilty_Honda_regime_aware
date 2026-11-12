from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REQUIRED_PARTNER_INPUTS = (
    "power_limits",
    "state_limits",
    "ramp_constraints",
    "recovery_requirements",
    "availability_windows",
    "production_deadlines",
    "tariff_rules",
    "connection_contract",
    "price_publication_timestamps",
    "operator_overrides",
    "realized_process_schedules",
    "production_plans",
    "maintenance_records",
    "process_state_and_recovery",
    "verified_weather_forecasts",
    "rolling_forecast_vintages",
    "updated_price_vintages",
)

REQUIRED_ISSUE_PROVENANCE = (
    "production_plan",
    "maintenance_availability",
    "process_state",
    "weather_forecast_vintage",
    "load_forecast_vintage",
    "price_vintage",
    "tariff_version",
    "calibration_state",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Evidence:
    source: str
    approved: bool = False
    approved_by_role: str | None = None
    approved_at_utc: str | None = None
    sha256: str | None = None

    def validate(self, name: str) -> list[str]:
        problems: list[str] = []
        if not self.source.strip():
            problems.append(f"{name}: evidence source is missing")
        if self.approved and (not self.approved_by_role or not self.approved_at_utc):
            problems.append(f"{name}: approved evidence needs role and UTC approval time")
        if self.approved and (self.sha256 is None or len(self.sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in self.sha256)):
            problems.append(f"{name}: approved evidence needs a valid SHA-256 digest")
        if self.approved_at_utc:
            stamp = pd.Timestamp(self.approved_at_utc)
            if stamp.tzinfo is None:
                problems.append(f"{name}: approval time must be timezone-aware")
        return problems


@dataclass(frozen=True)
class PartnerReadiness:
    partner_id: str
    timezone: str
    model_owner: str
    evidence: dict[str, Evidence]
    schema_version: str = "shadow.partner-readiness.v1"

    @classmethod
    def load(cls, path: str | Path) -> PartnerReadiness:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("Partner-readiness file must contain a mapping")
        allowed = {"schema_version", "partner_id", "timezone", "model_owner", "evidence"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown readiness fields: {', '.join(unknown)}")
        evidence_raw = raw.get("evidence")
        if not isinstance(evidence_raw, dict):
            raise TypeError("Readiness evidence must be a mapping")
        evidence: dict[str, Evidence] = {}
        for name, value in evidence_raw.items():
            if not isinstance(value, dict):
                raise TypeError(f"Evidence {name} must be a mapping")
            evidence[name] = Evidence(**value)
        result = cls(
            partner_id=str(raw.get("partner_id", "")),
            timezone=str(raw.get("timezone", "")),
            model_owner=str(raw.get("model_owner", "")),
            evidence=evidence,
            schema_version=str(raw.get("schema_version", "shadow.partner-readiness.v1")),
        )
        if result.schema_version != "shadow.partner-readiness.v1":
            raise ValueError("Unsupported partner-readiness schema")
        return result

    def report(self) -> dict[str, Any]:
        missing = sorted(set(REQUIRED_PARTNER_INPUTS) - set(self.evidence))
        problems = [f"{name}: missing" for name in missing]
        for name, value in self.evidence.items():
            if name not in REQUIRED_PARTNER_INPUTS:
                problems.append(f"{name}: unrecognized evidence category")
            problems.extend(value.validate(name))
            if not value.approved:
                problems.append(f"{name}: not partner-approved")
        ready = bool(self.partner_id and self.timezone and self.model_owner) and not problems
        return {
            "schema_version": "shadow.readiness-report.v1",
            "partner_id": self.partner_id,
            "ready_for_prospective_shadow": ready,
            "approved_categories": sorted(k for k, v in self.evidence.items() if v.approved),
            "required_categories": list(REQUIRED_PARTNER_INPUTS),
            "problems": sorted(problems),
            "claim_boundary": "PARTNER_APPROVED_SHADOW" if ready else "OFFLINE_ENGINEERING_ONLY",
            "configuration_hash": stable_hash(asdict(self)),
        }


@dataclass(frozen=True)
class ShadowAsset:
    asset_id: str
    asset_type: str
    p_up_kw: float
    p_down_kw: float
    state_min_kwh: float
    state_max_kwh: float
    initial_state_kwh: float
    terminal_state_kwh: float
    ramp_limit_kw_per_interval: float | None = None
    approved: bool = False

    def validate(self) -> None:
        if self.asset_type != "deferrable_process":
            raise ValueError(
                f"Asset type {self.asset_type!r} is registered but not implemented; "
                "Phase 2 value claims currently support deferrable_process only"
            )
        if not self.asset_id or min(self.p_up_kw, self.p_down_kw) < 0:
            raise ValueError("Invalid asset identity or power limits")
        if self.state_min_kwh > self.state_max_kwh:
            raise ValueError("Asset state minimum exceeds maximum")
        if not self.state_min_kwh <= self.initial_state_kwh <= self.state_max_kwh:
            raise ValueError("Initial state is outside approved bounds")
        if not self.state_min_kwh <= self.terminal_state_kwh <= self.state_max_kwh:
            raise ValueError("Terminal state is outside approved bounds")
        if self.ramp_limit_kw_per_interval is not None and self.ramp_limit_kw_per_interval < 0:
            raise ValueError("Ramp limit must be nonnegative")


@dataclass(frozen=True)
class CommercialTerms:
    import_eur_per_kwh: tuple[float, ...]
    export_eur_per_kwh: tuple[float, ...]
    network_eur_per_kwh_import: tuple[float, ...]
    demand_charge_eur_per_kw: float = 0.0
    imbalance_eur_per_kwh: float = 0.0
    recourse_eur_per_kwh: float = 0.0
    service_state_eur_per_kwh_interval: float = 0.0
    connection_violation_eur_per_kw_interval: float = 0.0
    label: str = "PARTNER_APPROVAL_REQUIRED"

    def arrays(self, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        buy = np.asarray(self.import_eur_per_kwh, dtype=float)
        sell = np.asarray(self.export_eur_per_kwh, dtype=float)
        network = np.asarray(self.network_eur_per_kwh_import, dtype=float)
        if any(len(v) != horizon for v in (buy, sell, network)):
            raise ValueError("Commercial vectors must match the rolling horizon")
        if not all(np.isfinite(v).all() for v in (buy, sell, network)):
            raise ValueError("Commercial vectors contain nonfinite values")
        if np.any(buy + network < sell - 1e-12):
            raise ValueError("Import plus network price must dominate export remuneration")
        scalars = (
            self.demand_charge_eur_per_kw,
            self.imbalance_eur_per_kwh,
            self.recourse_eur_per_kwh,
            self.service_state_eur_per_kwh_interval,
            self.connection_violation_eur_per_kw_interval,
        )
        if min(scalars) < 0:
            raise ValueError("Commercial penalty coefficients must be nonnegative")
        return buy, sell, network


@dataclass(frozen=True)
class TwoStageConfig:
    commitment_intervals: int
    recourse_start_interval: int
    adjustment_limit_kw: float
    import_limit_kw: float | None
    export_limit_kw: float | None
    hard_connection_limits: bool = True
    risk_lambda: float = 0.0
    cvar_alpha: float = 0.95

    def validate(self, horizon: int) -> None:
        if not 1 <= self.commitment_intervals <= horizon:
            raise ValueError("Commitment intervals must lie inside the horizon")
        if not 0 <= self.recourse_start_interval < horizon:
            raise ValueError("Recourse start must lie inside the horizon")
        if self.adjustment_limit_kw < 0:
            raise ValueError("Adjustment limit must be nonnegative")
        if self.risk_lambda < 0 or not 0 < self.cvar_alpha < 1:
            raise ValueError("Two-stage CVaR settings are invalid")
        if any(v is not None and v < 0 for v in (self.import_limit_kw, self.export_limit_kw)):
            raise ValueError("Connection limits must be nonnegative")


@dataclass(frozen=True)
class RollingDecisionInput:
    issue_time_utc: str
    deadline_time_utc: str
    timestamps_utc: tuple[str, ...]
    interval_hours: tuple[float, ...]
    point_forecast_kw: tuple[float, ...]
    scenario_matrix_kw: tuple[tuple[float, ...], ...]
    scenario_weights: tuple[float, ...]
    scenario_node_ids: tuple[tuple[str, ...], ...]
    price_publication_time_utc: str
    availability: tuple[bool, ...]
    provenance_hashes: dict[str, str] = field(default_factory=dict)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stamps = pd.DatetimeIndex(pd.to_datetime(self.timestamps_utc, utc=True))
        point = np.asarray(self.point_forecast_kw, dtype=float)
        scenarios = np.asarray(self.scenario_matrix_kw, dtype=float)
        weights = np.asarray(self.scenario_weights, dtype=float)
        nodes = np.asarray(self.scenario_node_ids, dtype=str)
        dt = np.asarray(self.interval_hours, dtype=float)
        available = np.asarray(self.availability, dtype=bool)
        horizon = len(stamps)
        if len(set(stamps)) != horizon or not stamps.is_monotonic_increasing:
            raise ValueError("Rolling UTC timestamps must be unique and increasing")
        if not 48 <= float(dt.sum()) <= 72:
            raise ValueError("Phase 2 rolling horizon must span 48–72 physical hours")
        if scenarios.ndim != 2 or scenarios.shape[1] != horizon:
            raise ValueError("Scenario matrix does not match rolling horizon")
        if nodes.shape != scenarios.shape:
            raise ValueError("Scenario-tree node matrix must match scenarios")
        if any(len(v) != horizon for v in (point, dt, available)):
            raise ValueError("Rolling input vectors do not share a horizon")
        if horizon > 1:
            observed_spacing = np.diff(stamps.view("i8")) / 3.6e12
            if not np.allclose(observed_spacing, dt[:-1]):
                raise ValueError("Interval durations do not match consecutive UTC timestamps")
        if len(weights) != len(scenarios) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise ValueError("Scenario weights must be nonnegative and sum to one")
        if not np.isfinite(point).all() or not np.isfinite(scenarios).all() or not np.isfinite(dt).all() or np.any(dt <= 0):
            raise ValueError("Rolling inputs contain invalid numeric values")
        issue = pd.Timestamp(self.issue_time_utc)
        deadline = pd.Timestamp(self.deadline_time_utc)
        publication = pd.Timestamp(self.price_publication_time_utc)
        if any(x.tzinfo is None for x in (issue, deadline, publication)):
            raise ValueError("Issue, deadline, and publication times must be timezone-aware")
        if issue > deadline or publication > deadline:
            raise ValueError("Recommendation deadline precedes available issue/price information")
        missing_provenance = sorted(set(REQUIRED_ISSUE_PROVENANCE) - set(self.provenance_hashes))
        if missing_provenance:
            raise ValueError(f"Rolling issue is missing provenance hashes: {missing_provenance}")
        invalid_hashes = sorted(
            name for name, digest in self.provenance_hashes.items()
            if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest)
        )
        if invalid_hashes:
            raise ValueError(f"Rolling issue has invalid provenance hashes: {invalid_hashes}")
        for t in range(1, horizon):
            for left in range(len(scenarios)):
                for right in range(left + 1, len(scenarios)):
                    if nodes[left, t] == nodes[right, t] and nodes[left, t - 1] != nodes[right, t - 1]:
                        raise ValueError("Scenario-tree branches cannot merge after diverging")
        return point, scenarios, weights, dt, available

    def hash(self) -> str:
        return stable_hash(asdict(self))
