from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .contracts import (
    CommercialTerms,
    PartnerReadiness,
    RollingDecisionInput,
    ShadowAsset,
    TwoStageConfig,
    stable_hash,
)
from .optimizer import TwoStageResult, solve_deterministic_fallback, solve_two_stage


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Immutable record already exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")


def load_issue(path: str | Path) -> tuple[dict[str, Any], RollingDecisionInput, ShadowAsset, CommercialTerms, TwoStageConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Shadow issue file must contain a mapping")
    allowed = {"schema_version", "issue_id", "partner_id", "approved_evidence_hashes", "decision", "asset", "commercial", "optimizer"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown shadow issue fields: {', '.join(unknown)}")
    if raw.get("schema_version") != "shadow.issue.v1":
        raise ValueError("Unsupported shadow issue schema")
    if not raw.get("issue_id") or not raw.get("partner_id"):
        raise ValueError("Shadow issue needs issue_id and partner_id")
    if not isinstance(raw.get("approved_evidence_hashes"), dict):
        raise ValueError("Shadow issue needs approved_evidence_hashes")
    decision = RollingDecisionInput(**raw["decision"])
    asset = ShadowAsset(**raw["asset"])
    commercial = CommercialTerms(**raw["commercial"])
    optimizer = TwoStageConfig(**raw["optimizer"])
    point, _, _, _, _ = decision.arrays()
    asset.validate(); commercial.arrays(len(point)); optimizer.validate(len(point))
    return raw, decision, asset, commercial, optimizer


def readiness_report(path: str | Path) -> dict[str, Any]:
    return PartnerReadiness.load(path).report()


def capture_issue_snapshot(issue_path: str | Path, readiness_path: str | Path, output_root: str | Path) -> Path:
    issue_path = Path(issue_path).resolve()
    readiness_path = Path(readiness_path).resolve()
    raw, decision, _asset, _commercial, _optimizer = load_issue(issue_path)
    readiness = PartnerReadiness.load(readiness_path)
    report = readiness.report()
    if raw["partner_id"] != readiness.partner_id:
        raise ValueError("Issue partner does not match readiness owner")
    expected_hashes = {name: evidence.sha256 for name, evidence in readiness.evidence.items() if evidence.approved}
    if raw["approved_evidence_hashes"] != expected_hashes:
        raise ValueError("Issue is not bound to the exact approved evidence digests")
    target = Path(output_root).resolve() / str(raw["partner_id"]) / str(raw["issue_id"])
    target.mkdir(parents=True, exist_ok=True)
    copies = {"issue.yaml": issue_path, "partner-readiness.yaml": readiness_path}
    for name, source in copies.items():
        destination = target / name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Immutable snapshot input already differs: {destination}")
        if not destination.exists():
            shutil.copyfile(source, destination)
    manifest = {
        "schema_version": "shadow.snapshot-manifest.v1",
        "issue_id": raw["issue_id"],
        "partner_id": raw["partner_id"],
        "issue_time_utc": decision.issue_time_utc,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": {name: _file_hash(source) for name, source in copies.items()},
        "normalized_decision_hash": decision.hash(),
        "readiness": report,
        "immutability": "write-once; a changed source requires a new issue_id",
    }
    manifest_path = target / "snapshot-manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Capture time is observational; all identity-bearing fields must match.
        manifest["captured_at_utc"] = existing["captured_at_utc"]
    _write_once(manifest_path, manifest)
    return target


def _result_payload(result: TwoStageResult) -> dict[str, Any]:
    return {
        "solver_status": result.solver_status,
        "objective_eur": result.objective_eur,
        "expected_cost_eur": result.expected_cost_eur,
        "worst_case_cost_eur": result.worst_case_cost_eur,
        "day_ahead_schedule_kw": result.day_ahead_schedule_kw.tolist(),
        "scenario_recourse_kw": result.scenario_recourse_kw.tolist(),
        "scenario_state_kwh": result.scenario_state_kwh.tolist(),
        "scenario_cost_eur": result.scenario_cost_eur.tolist(),
        "active_constraints": result.active_constraints,
        "nonanticipativity_max_residual_kw": result.nonanticipativity_max_residual_kw,
        "runtime_seconds": result.runtime_seconds,
        "configuration_hash": result.configuration_hash,
        "metadata": result.metadata,
    }


def _explain(
    result: TwoStageResult,
    fallback: TwoStageResult,
    decision: RollingDecisionInput,
    commercial: CommercialTerms,
    config: TwoStageConfig,
) -> dict[str, Any]:
    buy = np.asarray(commercial.import_eur_per_kwh) + np.asarray(commercial.network_eur_per_kwh_import)
    schedule = result.day_ahead_schedule_kw
    shifted = float(np.sum(np.maximum(schedule, 0) * np.asarray(decision.interval_hours)))
    grid_margin = None
    if config.import_limit_kw is not None:
        grid_margin = float(config.import_limit_kw - np.max(result.scenario_grid_kw))
    costs = result.scenario_cost_eur
    worst = int(np.argmax(costs))
    difference = schedule - fallback.day_ahead_schedule_kw
    return {
        "price_spread_eur_per_kwh": float(np.max(buy) - np.min(buy)),
        "flexible_energy_shifted_kwh": shifted,
        "binding_constraints": result.active_constraints,
        "worst_case_grid_cap_margin_kw": grid_margin,
        "tail_scenario": {"scenario_index": worst, "cost_eur": float(costs[worst])},
        "expected_cost_eur": result.expected_cost_eur,
        "worst_case_cost_eur": result.worst_case_cost_eur,
        "difference_from_deterministic": {
            "maximum_absolute_schedule_difference_kw": float(np.max(np.abs(difference))),
            "different_intervals": int(np.sum(~np.isclose(difference, 0))),
            "reason": "scenario-weighted energy, peak, service-state, and recourse costs under nonanticipative corrections",
        },
    }


def issue_recommendation(snapshot_dir: str | Path) -> dict[str, Any]:
    snapshot = Path(snapshot_dir).resolve()
    existing_path = snapshot / "recommendation.json"
    if existing_path.exists():
        return json.loads(existing_path.read_text(encoding="utf-8"))
    manifest = json.loads((snapshot / "snapshot-manifest.json").read_text(encoding="utf-8"))
    if not manifest["readiness"]["ready_for_prospective_shadow"]:
        raise ValueError("Partner readiness gate is not satisfied; remain in OFFLINE_ENGINEERING_ONLY mode")
    raw, decision, asset, commercial, optimizer = load_issue(snapshot / "issue.yaml")
    if not asset.approved:
        raise ValueError("The configured asset instance is not marked partner-approved")
    result = solve_two_stage(decision, asset, commercial, optimizer)
    fallback = solve_deterministic_fallback(decision, asset, commercial, optimizer)
    if result.solver_status != "optimal" or fallback.solver_status != "optimal":
        raise RuntimeError(f"Recommendation/fallback optimization failed: {result.solver_status} / {fallback.solver_status}")
    recommendation_id = stable_hash({"snapshot": manifest["input_hashes"], "configuration": result.configuration_hash})[:20]
    payload = {
        "schema_version": "shadow.recommendation.v1",
        "recommendation_id": recommendation_id,
        "issue_id": raw["issue_id"],
        "partner_id": raw["partner_id"],
        "status": "PENDING_OPERATOR_DECISION",
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_time_utc": decision.deadline_time_utc,
        "commitment_intervals": optimizer.commitment_intervals,
        "recommended": _result_payload(result),
        "deterministic_fallback": _result_payload(fallback),
        "explanation": _explain(result, fallback, decision, commercial, optimizer),
        "claim_boundary": "PROSPECTIVE_SHADOW_ONLY_NO_AUTOMATIC_CONTROL",
        "snapshot_manifest_sha256": _file_hash(snapshot / "snapshot-manifest.json"),
    }
    _write_once(existing_path, payload)
    return payload


def record_operator_decision(
    snapshot_dir: str | Path,
    disposition: str,
    operator_role: str,
    reason_code: str | None = None,
    note: str | None = None,
    modified_schedule_kw: list[float] | None = None,
) -> dict[str, Any]:
    if disposition not in {"APPROVED", "REJECTED", "MODIFIED"}:
        raise ValueError("Disposition must be APPROVED, REJECTED, or MODIFIED")
    if disposition != "APPROVED" and not reason_code:
        raise ValueError("Rejected or modified recommendations require a reason code")
    snapshot = Path(snapshot_dir).resolve()
    recommendation = json.loads((snapshot / "recommendation.json").read_text(encoding="utf-8"))
    if disposition == "MODIFIED":
        expected = len(recommendation["recommended"]["day_ahead_schedule_kw"])
        if modified_schedule_kw is None or len(modified_schedule_kw) != expected:
            raise ValueError("Modified schedule must match the recommendation horizon")
        _raw, decision, asset, _commercial, _optimizer = load_issue(snapshot / "issue.yaml")
        schedule = np.asarray(modified_schedule_kw, dtype=float)
        dt = np.asarray(decision.interval_hours)
        available = np.asarray(decision.availability)
        state = asset.initial_state_kwh + np.r_[0.0, np.cumsum(schedule * dt)]
        if np.any(np.abs(schedule[~available]) > 1e-8):
            raise ValueError("Modified schedule operates during an unavailable interval")
        if np.any(schedule > asset.p_up_kw + 1e-8) or np.any(schedule < -asset.p_down_kw - 1e-8):
            raise ValueError("Modified schedule exceeds approved power limits")
        if np.any(state > asset.state_max_kwh + 1e-8) or np.any(state < asset.state_min_kwh - 1e-8):
            raise ValueError("Modified schedule exceeds approved state limits")
        if abs(state[-1] - asset.terminal_state_kwh) > 1e-8:
            raise ValueError("Modified schedule does not satisfy the recovery target")
        if asset.ramp_limit_kw_per_interval is not None and np.any(np.abs(np.diff(schedule)) > asset.ramp_limit_kw_per_interval + 1e-8):
            raise ValueError("Modified schedule exceeds the approved ramp limit")
    payload = {
        "schema_version": "shadow.operator-decision.v1",
        "recommendation_id": recommendation["recommendation_id"],
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator_role": operator_role,
        "disposition": disposition,
        "reason_code": reason_code,
        "note": note,
        "modified_schedule_kw": modified_schedule_kw,
    }
    _write_once(snapshot / "operator-decision.json", payload)
    return payload


def settle_shadow(
    snapshot_dir: str | Path,
    realized_baseline_kw: list[float],
    realized_flexible_schedule_kw: list[float],
) -> dict[str, Any]:
    snapshot = Path(snapshot_dir).resolve()
    recommendation = json.loads((snapshot / "recommendation.json").read_text(encoding="utf-8"))
    operator = json.loads((snapshot / "operator-decision.json").read_text(encoding="utf-8"))
    _raw, decision, asset, commercial, optimizer = load_issue(snapshot / "issue.yaml")
    horizon = len(decision.timestamps_utc)
    baseline = np.asarray(realized_baseline_kw, dtype=float)
    realized = np.asarray(realized_flexible_schedule_kw, dtype=float)
    if len(baseline) != horizon or len(realized) != horizon:
        raise ValueError("Realized vectors must match the issue horizon")
    dt = np.asarray(decision.interval_hours)
    buy, sell, network = commercial.arrays(horizon)
    grid = baseline + realized
    actual_cost = float(np.sum((np.maximum(grid, 0) * (buy + network) - np.maximum(-grid, 0) * sell) * dt))
    actual_cost += commercial.demand_charge_eur_per_kw * float(np.max(np.maximum(grid, 0)))
    recommended = np.asarray(recommendation["recommended"]["day_ahead_schedule_kw"])
    fallback = np.asarray(recommendation["deterministic_fallback"]["day_ahead_schedule_kw"])
    def settled_cost(schedule: np.ndarray) -> tuple[float, float]:
        candidate_grid = baseline + schedule
        value = float(np.sum((np.maximum(candidate_grid, 0) * (buy + network) - np.maximum(-candidate_grid, 0) * sell) * dt))
        value += commercial.demand_charge_eur_per_kw * float(np.max(np.maximum(candidate_grid, 0)))
        return value, float(np.max(np.maximum(candidate_grid, 0)))
    no_flex_cost, no_flex_peak = settled_cost(np.zeros(horizon))
    recommended_cost, recommended_peak = settled_cost(recommended)
    fallback_cost, fallback_peak = settled_cost(fallback)
    state = asset.initial_state_kwh + np.r_[0.0, np.cumsum(realized * dt)]
    import_violation = 0.0 if optimizer.import_limit_kw is None else float(np.max(np.maximum(grid - optimizer.import_limit_kw, 0)))
    export_violation = 0.0 if optimizer.export_limit_kw is None else float(np.max(np.maximum(-grid - optimizer.export_limit_kw, 0)))
    power_violation = float(max(np.max(realized - asset.p_up_kw), np.max(-asset.p_down_kw - realized), 0))
    ramp_violation = 0.0 if asset.ramp_limit_kw_per_interval is None else float(max(np.max(np.abs(np.diff(realized)) - asset.ramp_limit_kw_per_interval), 0))
    availability_violation = float(np.max(np.abs(realized[~np.asarray(decision.availability)]))) if not all(decision.availability) else 0.0
    state_violation = float(max(np.max(state - asset.state_max_kwh), np.max(asset.state_min_kwh - state), 0))
    terminal_residual = float(abs(state[-1] - asset.terminal_state_kwh))
    hard_violation = any(value > 1e-8 for value in (import_violation, export_violation, power_violation, ramp_violation, availability_violation, state_violation, terminal_residual))
    payload = {
        "schema_version": "shadow.settlement.v1",
        "recommendation_id": recommendation["recommendation_id"],
        "operator_disposition": operator["disposition"],
        "settled_at_utc": datetime.now(timezone.utc).isoformat(),
        "realized_cost_eur": actual_cost,
        "no_flex_realized_cost_eur": no_flex_cost,
        "recommended_counterfactual_cost_eur": recommended_cost,
        "deterministic_fallback_counterfactual_cost_eur": fallback_cost,
        "recommendation_minus_realized_cost_eur": recommended_cost - actual_cost,
        "recommendation_minus_deterministic_cost_eur": recommended_cost - fallback_cost,
        "recommended_expected_cost_eur": recommendation["recommended"]["expected_cost_eur"],
        "schedule_mae_kw": float(np.mean(np.abs(realized - recommended))),
        "flexible_energy_delivered_kwh": float(np.sum(np.abs(realized) * dt)),
        "realized_correction_energy_kwh": float(np.sum(np.abs(realized - recommended) * dt)),
        "peak_import_kw": float(np.max(np.maximum(grid, 0))),
        "no_flex_peak_import_kw": no_flex_peak,
        "recommended_counterfactual_peak_import_kw": recommended_peak,
        "deterministic_fallback_counterfactual_peak_import_kw": fallback_peak,
        "max_import_violation_kw": import_violation,
        "max_export_violation_kw": export_violation,
        "max_power_violation_kw": power_violation,
        "max_ramp_violation_kw": ramp_violation,
        "max_availability_violation_kw": availability_violation,
        "max_state_violation_kwh": state_violation,
        "terminal_state_residual_kwh": terminal_residual,
        "hard_constraint_violation": hard_violation,
        "claim_boundary": "SHADOW_COMPARISON_NOT_CAUSAL_SAVINGS",
    }
    _write_once(snapshot / "settlement.json", payload)
    return payload


def cohort_metrics(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    recommendation_paths = sorted(root.rglob("recommendation.json"))
    rows: list[dict[str, Any]] = []
    for recommendation_path in recommendation_paths:
        folder = recommendation_path.parent
        recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
        operator_path = folder / "operator-decision.json"
        settlement_path = folder / "settlement.json"
        operator = json.loads(operator_path.read_text(encoding="utf-8")) if operator_path.exists() else None
        settlement = json.loads(settlement_path.read_text(encoding="utf-8")) if settlement_path.exists() else None
        issued = pd.Timestamp(recommendation["issued_at_utc"])
        deadline = pd.Timestamp(recommendation["deadline_time_utc"])
        preparation_minutes = None
        if operator is not None:
            preparation_minutes = (pd.Timestamp(operator["recorded_at_utc"]) - issued).total_seconds() / 60
        rows.append({"recommendation": recommendation, "operator": operator, "settlement": settlement, "on_time": issued <= deadline, "preparation_minutes": preparation_minutes})
    dispositions = [row["operator"]["disposition"] for row in rows if row["operator"]]
    settled = [row["settlement"] for row in rows if row["settlement"]]
    def mean(field: str) -> float | None:
        return None if not settled else float(np.mean([item[field] for item in settled]))
    return {
        "schema_version": "shadow.cohort-metrics.v1",
        "recommendations_issued": len(rows),
        "recommendations_on_time": sum(row["on_time"] for row in rows),
        "operator_decisions_recorded": len(dispositions),
        "accepted": dispositions.count("APPROVED"),
        "rejected": dispositions.count("REJECTED"),
        "modified": dispositions.count("MODIFIED"),
        "acceptance_rate": None if not dispositions else dispositions.count("APPROVED") / len(dispositions),
        "settlements_recorded": len(settled),
        "hard_constraint_violation_count": sum(item["hard_constraint_violation"] for item in settled),
        "mean_recommendation_minus_realized_cost_eur": mean("recommendation_minus_realized_cost_eur"),
        "mean_recommendation_minus_deterministic_cost_eur": mean("recommendation_minus_deterministic_cost_eur"),
        "mean_realized_correction_energy_kwh": mean("realized_correction_energy_kwh"),
        "mean_flexible_energy_delivered_kwh": mean("flexible_energy_delivered_kwh"),
        "mean_operator_preparation_minutes": None if not any(row["preparation_minutes"] is not None for row in rows) else float(np.mean([row["preparation_minutes"] for row in rows if row["preparation_minutes"] is not None])),
        "claim_boundary": "DESCRIPTIVE_PROSPECTIVE_SHADOW_METRICS_NOT_CAUSAL_SAVINGS",
    }
