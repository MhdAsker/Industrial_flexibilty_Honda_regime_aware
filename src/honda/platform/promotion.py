from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GlobalPromotionGates:
    minimum_macro_improvement: float = 0.02
    maximum_site_degradation: float = 0.02
    maximum_absolute_bias_kw: float = 10.0
    minimum_coverage: float = 0.87
    maximum_coverage: float = 0.93
    maximum_inference_p95_seconds: float = 30.0
    maximum_solver_p95_seconds: float = 60.0


def evaluate_global_promotion(
    champion_by_site: dict[str, dict[str, float | None]],
    challenger_by_site: dict[str, dict[str, float | None]],
    service_metrics: dict[str, float | bool | None],
    gates: GlobalPromotionGates | None = None,
) -> dict[str, Any]:
    gates = gates or GlobalPromotionGates()
    if not champion_by_site or set(champion_by_site) != set(challenger_by_site):
        return {"status": "UNKNOWN", "reason": "site_support_mismatch", "automatic_action_allowed": False}
    required = {"mae", "bias", "coverage", "decision_cost", "operator_acceptance"}
    service_required = {"inference_p95_seconds", "solver_p95_seconds", "drift_alarm_precedes_safety", "hard_violations"}
    if any(required - set(metrics) or any(metrics[key] is None for key in required) for metrics in (*champion_by_site.values(), *challenger_by_site.values())):
        return {"status": "UNKNOWN", "reason": "missing_site_monitoring_value", "automatic_action_allowed": False}
    if service_required - set(service_metrics) or any(service_metrics[key] is None for key in service_required):
        return {"status": "UNKNOWN", "reason": "missing_service_monitoring_value", "automatic_action_allowed": False}
    improvements = {}
    for site in champion_by_site:
        champion_mae = float(champion_by_site[site]["mae"])
        challenger_mae = float(challenger_by_site[site]["mae"])
        if champion_mae <= 0:
            return {"status": "UNKNOWN", "reason": "invalid_champion_mae", "automatic_action_allowed": False}
        improvements[site] = (champion_mae - challenger_mae) / champion_mae
    checks = {
        "macro_improvement": float(np.mean(list(improvements.values()))) >= gates.minimum_macro_improvement,
        "no_site_worse_than_limit": min(improvements.values()) >= -gates.maximum_site_degradation,
        "worst_site_not_degraded": max(float(value["mae"]) for value in challenger_by_site.values()) <= max(float(value["mae"]) for value in champion_by_site.values()),
        "site_bias": all(abs(float(value["bias"])) <= gates.maximum_absolute_bias_kw for value in challenger_by_site.values()),
        "site_calibration": all(gates.minimum_coverage <= float(value["coverage"]) <= gates.maximum_coverage for value in challenger_by_site.values()),
        "decision_noninferior": all(float(challenger_by_site[site]["decision_cost"]) <= float(champion_by_site[site]["decision_cost"]) for site in champion_by_site),
        "operator_noninferior": all(float(challenger_by_site[site]["operator_acceptance"]) >= float(champion_by_site[site]["operator_acceptance"]) for site in champion_by_site),
        "inference_service_level": float(service_metrics["inference_p95_seconds"]) <= gates.maximum_inference_p95_seconds,
        "solver_service_level": float(service_metrics["solver_p95_seconds"]) <= gates.maximum_solver_p95_seconds,
        "drift_early_warning": service_metrics["drift_alarm_precedes_safety"] is True,
        "zero_hard_violations": int(service_metrics["hard_violations"]) == 0,
    }
    return {
        "status": "PROMOTE" if all(checks.values()) else "DO_NOT_PROMOTE",
        "checks": checks,
        "macro_relative_improvement": float(np.mean(list(improvements.values()))),
        "worst_site_relative_improvement": float(min(improvements.values())),
        "per_site_relative_improvement": improvements,
        "automatic_action_allowed": False,
        "claim_boundary": "MODEL_PROMOTION_ONLY_DEPLOYMENT_STAGE_UNCHANGED",
    }
