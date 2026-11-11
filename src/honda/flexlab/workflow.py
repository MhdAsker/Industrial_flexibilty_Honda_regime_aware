from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from honda.optimization.contracts import DecisionInput, F1Config, F3Config, TariffConfig
from honda.optimization.optimizer import settle_schedule, solve_f1

from .adapters import SiteData, adapter_for
from .bundle import digest, manifest_for, stable_json
from .schemas import ExperimentConfig


def _local_timestamp(day: str, local_time: str, timezone: str) -> pd.Timestamp:
    value = pd.Timestamp(f"{day} {local_time}", tz=timezone)
    return value.tz_convert("UTC")


def validate_experiment(config_or_path: ExperimentConfig | str | Path) -> tuple[ExperimentConfig, SiteData]:
    config = config_or_path if isinstance(config_or_path, ExperimentConfig) else ExperimentConfig.load(config_or_path)
    data = adapter_for(config.site.adapter).load(config.site)
    config.tariff.vectors(len(data.timestamps_utc))
    issue_day = str((pd.Timestamp(config.site.target_day_local) - pd.Timedelta(days=1)).date())
    issue = _local_timestamp(issue_day, config.timing.forecast_issue_local, config.site.timezone)
    decision = _local_timestamp(issue_day, config.timing.decision_time_local, config.site.timezone)
    if decision < issue:
        raise ValueError("Decision time cannot precede forecast issue")
    if config.timing.classification == "OPERATIONAL_TIMING_VERIFIED":
        publication = _local_timestamp(issue_day, str(config.timing.price_publication_local), config.site.timezone)
        if decision < publication:
            raise ValueError("Decision time precedes verified price publication")
    return config, data


def _decision(config: ExperimentConfig, data: SiteData, scenarios: np.ndarray) -> DecisionInput:
    issue_day = str((pd.Timestamp(config.site.target_day_local) - pd.Timedelta(days=1)).date())
    issue = _local_timestamp(issue_day, config.timing.forecast_issue_local, config.site.timezone)
    decision = _local_timestamp(issue_day, config.timing.decision_time_local, config.site.timezone)
    publication = None
    if config.timing.price_publication_local is not None:
        publication = _local_timestamp(issue_day, config.timing.price_publication_local, config.site.timezone).isoformat()
    weights = np.full(len(scenarios), 1 / len(scenarios))
    return DecisionInput(
        dataset_id=config.site.dataset_id, target_day_local=config.site.target_day_local,
        timestamp_utc=data.timestamps_utc, interval_hours=tuple(np.full(len(data.timestamps_utc), .25)),
        deterministic_forecast_kw=tuple(data.point_forecast_kw), scenario_matrix_kw=tuple(map(tuple, scenarios)),
        scenario_weights=tuple(weights), forecast_issue_time_utc=issue.isoformat(), decision_time_utc=decision.isoformat(),
        price_publication_time_utc=publication, timing_classification=config.timing.classification,
        provenance_hashes=data.provenance,
    )


def evaluate_experiment(config_or_path: ExperimentConfig | str | Path, output: str | Path) -> dict:
    config, data = validate_experiment(config_or_path)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    buy, sell = config.tariff.vectors(len(data.timestamps_utc))
    tariff = TariffConfig.from_eur_per_mwh(buy, sell, label=config.tariff.label)
    asset = F1Config(**asdict(config.asset))
    grid = F3Config(
        import_max_kw=config.grid.import_max_kw, export_max_kw=config.grid.export_max_kw,
        peak_penalty_eur_per_kw_per_day=config.grid.peak_penalty_eur_per_kw_per_day,
        grid_limit_mode=config.grid.mode, chance_quantile=config.grid.chance_quantile,
        soft_violation_penalty_eur_per_kw_interval=config.grid.soft_violation_penalty_eur_per_kw_interval,
    )
    scenario_map = {
        "deterministic": data.point_forecast_kw[None, :],
        "independent": data.independent_scenarios_kw,
        "coherent": data.coherent_scenarios_kw,
    }
    if config.coherent_calibration_scale is not None:
        scenario_map["calibrated_coherent"] = data.point_forecast_kw[None, :] + config.coherent_calibration_scale * (data.coherent_scenarios_kw - data.point_forecast_kw[None, :])
    no_flex = settle_schedule(np.zeros(len(data.actual_kw)), data.actual_kw, np.full(len(data.actual_kw), .25), tariff)
    no_flex_peak_charge = config.grid.peak_penalty_eur_per_kw_per_day * float(np.max(no_flex["import_kw"]))
    no_flex_total_cost = float(no_flex["cost_eur"]) + no_flex_peak_charge
    summaries: list[dict] = []; schedules: list[dict] = []; states: list[dict] = []
    for policy in config.policies:
        result = solve_f1(_decision(config, data, scenario_map[policy]), asset, grid, tariff, risk_lambda=config.risk_lambda, cvar_alpha=config.cvar_alpha)
        if result.solver_status != "optimal":
            raise RuntimeError(f"Policy {policy} failed: {result.solver_status}")
        settled = settle_schedule(result.schedule_kw, data.actual_kw, np.full(len(data.actual_kw), .25), tariff)
        actual_grid = np.asarray(settled["grid_kw"])
        import_violation = 0.0 if config.grid.import_max_kw is None else float(np.maximum(actual_grid - config.grid.import_max_kw, 0).sum() * .25)
        export_violation = 0.0 if config.grid.export_max_kw is None else float(np.maximum(-actual_grid - config.grid.export_max_kw, 0).sum() * .25)
        peak_charge = config.grid.peak_penalty_eur_per_kw_per_day * float(np.max(settled["import_kw"]))
        violation_charge = config.grid.soft_violation_penalty_eur_per_kw_interval * 4 * (import_violation + export_violation)
        realized_total_cost = float(settled["cost_eur"]) + peak_charge + violation_charge
        summaries.append({
            "policy": policy, "solver_status": result.solver_status, "scenario_count": int(len(scenario_map[policy])),
            "expected_objective_eur": result.objective_eur, "realized_energy_cost_eur": float(settled["cost_eur"]),
            "realized_peak_charge_eur": peak_charge, "realized_violation_charge_eur": violation_charge,
            "realized_total_cost_eur": realized_total_cost, "no_flex_total_cost_eur": no_flex_total_cost,
            "realized_savings_eur": no_flex_total_cost - realized_total_cost,
            "peak_import_kw": float(np.max(settled["import_kw"])), "export_energy_kwh": float(np.sum(settled["export_kw"]) * .25),
            "import_cap_violation_kwh": import_violation, "export_cap_violation_kwh": export_violation,
            "terminal_residual_kwh": float(result.feasibility["terminal_residual_kwh"]),
            "power_binding_intervals": result.active_constraints.get("power", 0), "state_binding_points": result.active_constraints.get("state", 0),
            "configuration_hash": result.configuration_hash,
        })
        for i, stamp in enumerate(data.timestamps_utc):
            schedules.append({"policy": policy, "timestamp_utc": stamp, "actual_kw": data.actual_kw[i], "forecast_kw": data.point_forecast_kw[i], "schedule_kw": result.schedule_kw[i], "realized_grid_kw": actual_grid[i]})
        for i, value in enumerate(result.state_kwh):
            states.append({"policy": policy, "state_index": i, "state_kwh": value})
    policy_frame = pd.DataFrame(summaries).sort_values("policy")
    policy_frame.to_csv(output / "policy_summary.csv", index=False, float_format="%.10g", lineterminator="\n")
    pd.DataFrame(schedules).to_csv(output / "schedules.csv", index=False, float_format="%.10g", lineterminator="\n")
    pd.DataFrame(states).to_csv(output / "states.csv", index=False, float_format="%.10g", lineterminator="\n")
    if config.source_path is not None:
        shutil.copy2(config.source_path, output / "experiment.yaml")
    summary = {
        "schema_version": "flexlab.result.v2", "experiment_id": config.experiment_id,
        "dataset_id": config.site.dataset_id, "target_day_local": config.site.target_day_local,
        "target_semantics": config.site.target_semantics, "timing_classification": config.timing.classification,
        "asset_label": config.asset.label, "tariff_label": config.tariff.label,
        "policies": policy_frame.to_dict("records"), "supported_claims": list(config.supported_claims),
        "unsupported_claims": list(config.unsupported_claims),
    }
    stable_json(output / "summary.json", summary)
    schedule_frame = pd.DataFrame(schedules)
    schedule_pivot = schedule_frame.pivot(index="timestamp_utc", columns="policy", values="schedule_kw").reindex(data.timestamps_utc)
    dashboard = {
        "experiment": {k: summary[k] for k in ("experiment_id", "dataset_id", "target_day_local", "target_semantics", "timing_classification", "asset_label", "tariff_label")},
        "policies": policy_frame.to_dict("records"),
        "trajectory": {
            "timestamp_utc": list(data.timestamps_utc), "actual_kw": data.actual_kw.tolist(), "forecast_kw": data.point_forecast_kw.tolist(),
            "coherent_q10_kw": np.quantile(data.coherent_scenarios_kw, .1, axis=0).tolist(), "coherent_q90_kw": np.quantile(data.coherent_scenarios_kw, .9, axis=0).tolist(),
            "schedules_kw": {column: schedule_pivot[column].tolist() for column in schedule_pivot.columns},
        },
        "labels": {"supported": list(config.supported_claims), "unsupported": list(config.unsupported_claims)},
    }
    stable_json(output / "dashboard.json", dashboard)
    manifest = manifest_for(output, experiment=config.as_dict(), provenance=data.provenance, supported=list(config.supported_claims), unsupported=list(config.unsupported_claims))
    stable_json(output / "manifest.json", manifest)
    return summary


def reproduce_experiment(config_path: str | Path) -> dict:
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        evaluate_experiment(config_path, first); evaluate_experiment(config_path, second)
        names = ("policy_summary.csv", "schedules.csv", "states.csv", "summary.json", "dashboard.json")
        hashes_a = {name: digest(Path(first) / name) for name in names}
        hashes_b = {name: digest(Path(second) / name) for name in names}
        return {"passed": hashes_a == hashes_b, "hashes": hashes_a, "mismatches": sorted(name for name in names if hashes_a[name] != hashes_b[name])}
