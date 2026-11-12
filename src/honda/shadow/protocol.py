from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_ENDPOINTS = (
    "accepted_hard_constraint_violations",
    "deadline_success_rate",
    "multiday_tail_coverage",
    "tail_cost_eur",
    "operator_interventions",
    "violation_duration_hours",
    "operator_acceptance_rate",
    "operator_preparation_minutes",
    "economic_difference_eur",
)


@dataclass(frozen=True)
class ProspectiveProtocol:
    protocol_id: str
    partner_id: str
    frozen_at_utc: str
    operator_baseline: str
    deterministic_baseline: str
    stochastic_policy: str
    eligibility_rules: tuple[str, ...]
    missed_deadline_rule: str
    tariff_version: str
    safety_endpoints: tuple[str, ...]
    economic_endpoint: str
    unacceptable_loss_margin_eur: float
    switchback_enabled: bool
    minimum_prospective_weeks: int = 8
    schema_version: str = "shadow.prospective-protocol.v1"

    @classmethod
    def load(cls, path: str | Path) -> ProspectiveProtocol:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        frozen = pd.Timestamp(self.frozen_at_utc)
        if frozen.tzinfo is None or self.schema_version != "shadow.prospective-protocol.v1":
            raise ValueError("Prospective protocol version or frozen timestamp is invalid")
        if not all((self.protocol_id, self.partner_id, self.operator_baseline, self.deterministic_baseline, self.stochastic_policy)):
            raise ValueError("Prospective protocol identities and baselines are required")
        if not self.eligibility_rules or not self.missed_deadline_rule or not self.tariff_version:
            raise ValueError("Prospective eligibility, deadline, and tariff rules are required")
        if self.minimum_prospective_weeks < 8 or self.unacceptable_loss_margin_eur < 0:
            raise ValueError("Prospective duration or unacceptable-loss margin is invalid")
        if "accepted_hard_constraint_violations" not in self.safety_endpoints:
            raise ValueError("Hard-constraint violations must be a primary safety endpoint")
        if self.economic_endpoint != "economic_difference_eur":
            raise ValueError("The primary economic endpoint must be economic_difference_eur")

    @property
    def protocol_hash(self) -> str:
        value = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return sha256(value.encode("utf-8")).hexdigest()


def switchback_assignment(protocol: ProspectiveProtocol, block_id: str) -> str:
    protocol.validate()
    if not protocol.switchback_enabled:
        return "DESCRIPTIVE_STOCHASTIC_SHADOW"
    digest = sha256(f"{protocol.protocol_hash}:{block_id}".encode()).digest()
    return protocol.deterministic_baseline if digest[0] % 2 == 0 else protocol.stochastic_policy


def economic_margin_interval(
    economic_difference_eur: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 5_000,
    seed: int = 20261112,
) -> dict[str, float]:
    values = np.asarray(economic_difference_eur, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or not 0 < confidence < 1:
        raise ValueError("Economic endpoint observations are invalid")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "mean_eur": float(np.mean(values)),
        "lower_eur": float(np.quantile(draws, alpha)),
        "upper_eur": float(np.quantile(draws, 1 - alpha)),
        "confidence": confidence,
        "days": float(len(values)),
    }


def phase2_exit_report(protocol: ProspectiveProtocol, metrics: dict[str, Any]) -> dict[str, Any]:
    protocol.validate()
    missing = sorted(set(REQUIRED_ENDPOINTS) - set(metrics))
    if missing:
        raise ValueError(f"Phase 2 exit metrics are missing: {missing}")
    interval = metrics["economic_difference_eur"]
    checks = {
        "minimum_duration": float(metrics.get("prospective_weeks", 0)) >= protocol.minimum_prospective_weeks,
        "zero_accepted_hard_violations": metrics["accepted_hard_constraint_violations"] == 0,
        "all_eligible_recommendations_on_time": metrics["deadline_success_rate"] == 1,
        "multiday_tail_coverage_reliable": bool(metrics["multiday_tail_coverage"]),
        "tail_or_intervention_improvement": bool(
            metrics["tail_cost_eur"] < 0
            or metrics["operator_interventions"] < 0
            or metrics["violation_duration_hours"] < 0
        ),
        "operator_evidence_available": (
            metrics["operator_acceptance_rate"] is not None
            and metrics["operator_preparation_minutes"] is not None
        ),
        "economic_margin_passed": float(interval["upper_eur"]) < protocol.unacceptable_loss_margin_eur,
    }
    return {
        "schema_version": "shadow.phase2-exit-report.v1",
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
        "checks": checks,
        "status": "PHASE2_EXIT_PASSED" if all(checks.values()) else "PHASE2_EXIT_NOT_YET_MET",
        "causal_interpretation": "SWITCHBACK" if protocol.switchback_enabled else "DESCRIPTIVE_ONLY",
    }
