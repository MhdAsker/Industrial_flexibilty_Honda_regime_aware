from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.honda.decision_value.statistics import (
    bootstrap_mean,
    holm_adjust,
    lag1,
    moving_block_bootstrap,
)
from src.honda.scenarios.adapters import load_spark_test, spark_root


ROOT = Path(__file__).resolve().parents[3]
SPARK = spark_root()
CONFIG = ROOT / "config/decision_value_replication/stage3_feasibility_replication_frozen.json"
OUT = ROOT / "results/decision_value_replication"
REPORT = ROOT / "reports/decision_value/industrial_flexibility_decision_value_replication.md"
EXPECTED_CONFIG_HASH = "f00a24c55072cce5ec9399d506231d6f7fe67d29ae50448a8ea9ba1d3e39c0c3"
POLICIES = ("P2_LGBM_DETERMINISTIC", "P3_S1_INDEPENDENT", "P4_S2_COHERENT")
TOL = 1e-7


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


def explicit_zero_export_settlement_correction(
    realized_baseline_kw: np.ndarray,
    sealed_schedule_kw: np.ndarray,
    price_eur_per_mwh: np.ndarray,
    *,
    interval_hours: float = 0.25,
) -> dict[str, np.ndarray | float | bool]:
    """Apply the pre-specified settlement-only zero-export projection.

    The committed schedule is never changed.  The correction is exactly the
    pointwise amount required to remove realized export.  It is external to the
    F1 state and is a secondary ex-post accounting sensitivity, not a forecast-
    time control or plant-feasibility claim.
    """
    actual = np.asarray(realized_baseline_kw, dtype=float)
    schedule = np.asarray(sealed_schedule_kw, dtype=float)
    price = np.asarray(price_eur_per_mwh, dtype=float)
    if actual.ndim != 1 or schedule.shape != actual.shape or price.shape != actual.shape:
        raise ValueError("Actual, sealed schedule, and price must be aligned one-dimensional arrays")
    if not (np.isfinite(actual).all() and np.isfinite(schedule).all() and np.isfinite(price).all()):
        raise ValueError("Settlement correction inputs must be finite")
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")

    original_schedule = schedule.copy()
    nominal_grid = actual + schedule
    correction_kw = np.maximum(-nominal_grid, 0.0)
    corrected_grid = nominal_grid + correction_kw
    price_eur_per_kwh = price / 1000.0
    nominal_cost = float(np.sum(price_eur_per_kwh * nominal_grid * interval_hours))
    corrected_cost = float(np.sum(price_eur_per_kwh * corrected_grid * interval_hours))
    no_flex_cost = float(np.sum(price_eur_per_kwh * actual * interval_hours))
    if not np.array_equal(schedule, original_schedule):
        raise RuntimeError("Settlement correction mutated the sealed schedule")
    if np.min(corrected_grid) < -TOL:
        raise RuntimeError("Explicit settlement correction failed to enforce zero export")
    if np.any(correction_kw - np.maximum(-nominal_grid, 0.0) > TOL):
        raise RuntimeError("Settlement correction exceeded the minimum required intervention")
    return {
        "nominal_grid_kw": nominal_grid,
        "correction_kw": correction_kw,
        "corrected_grid_kw": corrected_grid,
        "nominal_cost_eur": nominal_cost,
        "corrected_cost_eur": corrected_cost,
        "no_flex_cost_eur": no_flex_cost,
        "nominal_savings_eur": no_flex_cost - nominal_cost,
        "corrected_savings_eur": no_flex_cost - corrected_cost,
        "correction_cost_eur": corrected_cost - nominal_cost,
        "correction_energy_kwh": float(np.sum(correction_kw) * interval_hours),
        "nominal_cap_compliant": bool(np.max(correction_kw) <= TOL),
        "violation_intervals": int(np.sum(correction_kw > TOL)),
        "violation_max_kw": float(np.max(correction_kw)),
        "violation_energy_kwh": float(np.sum(correction_kw) * interval_hours),
    }


def _load_prices() -> pd.Series:
    path = SPARK / "data/silver/exogenous/canonical/electricity_prices_15min_de_2021_2024.parquet"
    prices = pd.read_parquet(path, columns=["utc_datetime", "price_eur_per_mwh"])
    prices["utc_datetime"] = pd.to_datetime(prices.utc_datetime, utc=True)
    if prices.utc_datetime.duplicated().any() or prices.price_eur_per_mwh.isna().any():
        raise ValueError("Invalid canonical SPARK price series")
    return prices.set_index("utc_datetime").price_eur_per_mwh.sort_index()


def _load_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    if digest(CONFIG) != EXPECTED_CONFIG_HASH:
        raise RuntimeError("Frozen replication configuration hash mismatch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_REPLICATION_OUTCOME_ANALYSIS":
        raise RuntimeError("Replication specification is not frozen")
    for item in config["frozen_inputs"].values():
        text_path = item["path"]
        path = (SPARK / text_path.removeprefix("SPARK/")) if text_path.startswith("SPARK/") else (ROOT / text_path)
        if digest(path) != item["sha256"]:
            raise RuntimeError(f"Frozen input hash mismatch: {text_path}")

    daily = pd.read_parquet(ROOT / "results/decision_value/policy_daily_metrics.parquet")
    decisions = pd.read_parquet(ROOT / "results/decision_value/policy_decisions.parquet")
    daily = daily[daily.policy.isin(POLICIES)].copy()
    decisions = decisions[decisions.policy.isin(POLICIES)].copy()
    target = load_spark_test()
    target = target[target.complete_day].copy()
    target["timestamp_utc"] = pd.to_datetime(target.timestamp_utc, utc=True)
    prices = _load_prices()
    return daily, decisions, target, prices, config


def _day_level_metrics() -> tuple[pd.DataFrame, dict]:
    original_daily, decisions, target, prices, config = _load_frozen_inputs()
    decision_map = decisions.set_index(["target_day_local", "policy"])
    original_map = original_daily.set_index(["target_day_local", "policy"])
    rows: list[dict] = []
    for day in sorted(original_daily.target_day_local.unique()):
        target_day = target[target.target_day_local.eq(day)].sort_values("timestamp_utc")
        if len(target_day) != 96:
            raise RuntimeError(f"Replication is frozen to 96-interval days; found {len(target_day)} on {day}")
        price = prices.reindex(pd.DatetimeIndex(target_day.timestamp_utc))
        if price.isna().any():
            raise RuntimeError(f"Missing price on frozen day {day}")
        actual = target_day.actual_kw.to_numpy(float)
        for policy in POLICIES:
            record = decision_map.loc[(day, policy)]
            original = original_map.loc[(day, policy)]
            schedule = np.asarray(record.schedule_kw, dtype=float)
            state = np.asarray(record.state_kwh, dtype=float)
            settled = explicit_zero_export_settlement_correction(actual, schedule, price.to_numpy(float))
            if not np.isclose(settled["nominal_cost_eur"], float(original.realized_cost_eur), atol=1e-8):
                raise RuntimeError(f"Nominal cost does not reproduce frozen Stage 3 on {day} / {policy}")
            if bool(settled["nominal_cap_compliant"]) != bool(original.realized_grid_cap_compliant):
                raise RuntimeError(f"Feasibility classification does not reproduce Stage 3 on {day} / {policy}")
            rows.append({
                "target_day_local": day,
                "policy": policy,
                "result_classification_nominal": "EXPLORATORY_DIAGNOSTIC",
                "result_classification_common_feasible": "PRE_REGISTERED_PRIMARY",
                "result_classification_corrected": "PRE_SPECIFIED_SECONDARY",
                "economic_label": "SYNTHETIC_COUNTERFACTUAL_WHOLESALE_COST_DIFFERENCE",
                "timing_classification": "POST_PUBLICATION_TIMING_ASSUMED",
                "customer_value_label": "NOT_CUSTOMER_SAVINGS",
                "profit_label": "NOT_TRADING_PROFIT",
                "physical_label": "NOT_PLANT_VALIDATION",
                "nominal_cost_eur": settled["nominal_cost_eur"],
                "nominal_savings_eur": settled["nominal_savings_eur"],
                "corrected_cost_eur": settled["corrected_cost_eur"],
                "corrected_savings_eur": settled["corrected_savings_eur"],
                "correction_cost_eur": settled["correction_cost_eur"],
                "correction_energy_kwh": settled["correction_energy_kwh"],
                "no_flex_cost_eur": settled["no_flex_cost_eur"],
                "cap_compliant": settled["nominal_cap_compliant"],
                "violation_intervals": settled["violation_intervals"],
                "violation_max_kw": settled["violation_max_kw"],
                "violation_energy_kwh": settled["violation_energy_kwh"],
                "power_upper_bind": bool(np.any(schedule >= 5.0 - TOL)),
                "power_lower_bind": bool(np.any(schedule <= -5.0 + TOL)),
                "power_limit_bind": bool(np.any(np.abs(schedule) >= 5.0 - TOL)),
                "advance_state_bind": bool(np.any(state >= 10.0 - TOL)),
                "backlog_state_bind": bool(np.any(state <= -10.0 + TOL)),
                "energy_state_bind": bool(np.any(np.abs(state) >= 10.0 - TOL)),
                "realized_grid_cap_active_or_violated": bool(np.min(np.asarray(settled["nominal_grid_kw"])) <= TOL),
                "schedule_energy_residual_kwh": float(np.sum(schedule) * 0.25),
                "terminal_state_residual_kwh": float(abs(state[-1])),
            })
    result = pd.DataFrame(rows).sort_values(["target_day_local", "policy"]).reset_index(drop=True)
    feasible_by_day = result.groupby("target_day_local").cap_compliant.all()
    result["common_feasible_day"] = result.target_day_local.map(feasible_by_day).astype(bool)
    if result.target_day_local.nunique() != 273 or set(result.groupby("policy").size()) != {273}:
        raise RuntimeError("Replication day/policy coverage differs from frozen specification")
    return result, config


def _comparison(
    daily: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    *,
    comparison: str,
    cost_column: str,
    classification: str,
    population: str,
    config: dict,
) -> dict:
    a = daily[daily.policy.eq(policy_a)][["target_day_local", cost_column]].rename(columns={cost_column: "a"})
    b = daily[daily.policy.eq(policy_b)][["target_day_local", cost_column]].rename(columns={cost_column: "b"})
    merged = a.merge(b, on="target_day_local", how="inner").dropna().sort_values("target_day_local")
    difference = (merged.a - merged.b).to_numpy(float)
    settings = config["statistics"]
    ordinary = bootstrap_mean(difference, resamples=settings["bootstrap_resamples"], seed=settings["bootstrap_seed"])
    block = moving_block_bootstrap(
        difference,
        block_length=settings["block_length_days"],
        resamples=settings["bootstrap_resamples"],
        seed=settings["bootstrap_seed"],
    )
    p = min(1.0, 2 * min(float(np.mean(ordinary <= 0)), float(np.mean(ordinary >= 0))))
    return {
        "comparison": comparison,
        "policy_A": policy_a,
        "policy_B": policy_b,
        "cost_column": cost_column,
        "population": population,
        "result_classification": classification,
        "interpretation": "positive A-minus-B means policy B has lower wholesale cost",
        "common_days": len(difference),
        "mean_cost_A_minus_B_eur_day": float(np.mean(difference)),
        "median_cost_A_minus_B_eur_day": float(np.median(difference)),
        "ci95_lower_eur_day": float(np.quantile(ordinary, 0.025)),
        "ci95_upper_eur_day": float(np.quantile(ordinary, 0.975)),
        "block_ci95_lower_eur_day": float(np.quantile(block, 0.025)),
        "block_ci95_upper_eur_day": float(np.quantile(block, 0.975)),
        "policy_B_daily_win_fraction": float(np.mean(difference > 0)),
        "lag1_daily_difference": lag1(difference),
        "two_sided_bootstrap_p": p,
        "bootstrap_resamples": settings["bootstrap_resamples"],
        "block_length_days": settings["block_length_days"],
        "seed": settings["bootstrap_seed"],
    }


def _comparisons(daily: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = config["primary_comparisons"]
    common = daily[daily.common_feasible_day].copy()
    primary = pd.DataFrame([
        _comparison(
            common, spec["policy_a"], spec["policy_b"], comparison=spec["id"],
            cost_column="nominal_cost_eur", classification="PRE_REGISTERED_PRIMARY",
            population="COMMON_FEASIBLE_ALL_THREE_POLICIES", config=config,
        ) for spec in specs
    ])
    primary = holm_adjust(primary.rename(columns={"two_sided_bootstrap_p": "two_sided_bootstrap_p"}))
    corrected = pd.DataFrame([
        _comparison(
            daily, spec["policy_a"], spec["policy_b"], comparison=spec["id"],
            cost_column="corrected_cost_eur", classification="PRE_SPECIFIED_SECONDARY",
            population="ALL_273_EXPLICITLY_CORRECTED_DAYS", config=config,
        ) for spec in specs
    ])
    nominal = pd.DataFrame([
        _comparison(
            daily, spec["policy_a"], spec["policy_b"], comparison=spec["id"],
            cost_column="nominal_cost_eur", classification="EXPLORATORY_DIAGNOSTIC",
            population="ALL_273_NOMINAL_DAYS_CONTAMINATED_BY_CAP_VIOLATIONS", config=config,
        ) for spec in specs
    ])
    return primary, corrected, nominal


def _summaries(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = daily[daily.common_feasible_day]
    summary_rows = []
    for policy, group in daily.groupby("policy", sort=True):
        feasible = common[common.policy.eq(policy)]
        summary_rows.append({
            "policy": policy,
            "result_label": "SYNTHETIC_COUNTERFACTUAL_WHOLESALE_COST_DIFFERENCE",
            "all_day_nominal_mean_cost_eur": float(group.nominal_cost_eur.mean()),
            "all_day_nominal_mean_savings_eur": float(group.nominal_savings_eur.mean()),
            "cap_violation_days": int((~group.cap_compliant).sum()),
            "common_feasible_day_count": int(feasible.target_day_local.nunique()),
            "common_feasible_mean_cost_eur": float(feasible.nominal_cost_eur.mean()),
            "common_feasible_mean_savings_eur": float(feasible.nominal_savings_eur.mean()),
            "mean_correction_cost_eur": float(group.correction_cost_eur.mean()),
            "mean_correction_energy_kwh": float(group.correction_energy_kwh.mean()),
            "mean_corrected_cost_eur": float(group.corrected_cost_eur.mean()),
            "mean_corrected_savings_eur": float(group.corrected_savings_eur.mean()),
            "timing_classification": "POST_PUBLICATION_TIMING_ASSUMED",
            "primary_result_classification": "PRE_REGISTERED_PRIMARY",
            "corrected_result_classification": "PRE_SPECIFIED_SECONDARY",
            "nominal_result_classification": "EXPLORATORY_DIAGNOSTIC",
        })
    summary = pd.DataFrame(summary_rows)

    feasibility_rows = []
    for policy, group in daily.groupby("policy", sort=True):
        violating = group[~group.cap_compliant]
        feasibility_rows.append({
            "policy": policy,
            "days": len(group),
            "compliant_days": int(group.cap_compliant.sum()),
            "violation_days": int((~group.cap_compliant).sum()),
            "violation_day_fraction": float((~group.cap_compliant).mean()),
            "mean_violation_energy_kwh_all_days": float(group.violation_energy_kwh.mean()),
            "mean_violation_energy_kwh_violating_days": float(violating.violation_energy_kwh.mean()) if len(violating) else 0.0,
            "p90_violation_energy_kwh_violating_days": float(violating.violation_energy_kwh.quantile(.9)) if len(violating) else 0.0,
            "maximum_violation_energy_kwh": float(group.violation_energy_kwh.max()),
            "maximum_violation_kw": float(group.violation_max_kw.max()),
            "classification": "EXPLORATORY_DIAGNOSTIC",
        })
    feasibility = pd.DataFrame(feasibility_rows)

    pivot = daily.pivot(index="target_day_local", columns="policy", values="cap_compliant")
    overlap = (
        pivot.assign(
            P2_violation=~pivot["P2_LGBM_DETERMINISTIC"],
            P3_violation=~pivot["P3_S1_INDEPENDENT"],
            P4_violation=~pivot["P4_S2_COHERENT"],
        )
        .groupby(["P2_violation", "P3_violation", "P4_violation"], dropna=False)
        .size().rename("days").reset_index()
    )
    overlap["classification"] = "EXPLORATORY_DIAGNOSTIC"

    binding = daily.groupby("policy", as_index=False).agg(
        days=("target_day_local", "nunique"),
        power_limit_binding_days=("power_limit_bind", "sum"),
        power_upper_binding_days=("power_upper_bind", "sum"),
        power_lower_binding_days=("power_lower_bind", "sum"),
        energy_state_binding_days=("energy_state_bind", "sum"),
        advance_state_binding_days=("advance_state_bind", "sum"),
        backlog_state_binding_days=("backlog_state_bind", "sum"),
        realized_grid_cap_active_or_violation_days=("realized_grid_cap_active_or_violated", "sum"),
        realized_grid_cap_violation_days=("cap_compliant", lambda x: int((~x).sum())),
        maximum_abs_schedule_energy_residual_kwh=("schedule_energy_residual_kwh", lambda x: float(np.max(np.abs(x)))),
        maximum_terminal_state_residual_kwh=("terminal_state_residual_kwh", "max"),
    )
    binding["classification"] = "EXPLORATORY_DIAGNOSTIC"
    return summary, feasibility, overlap, binding


def _binding_effects(daily: pd.DataFrame) -> pd.DataFrame:
    wide_cost = daily.pivot(index="target_day_local", columns="policy", values="nominal_cost_eur")
    p4 = daily[daily.policy.eq("P4_S2_COHERENT")].set_index("target_day_local")
    rows = []
    flags = {
        "P4_POWER_LIMIT_BIND": p4.power_limit_bind,
        "P4_ENERGY_STATE_BIND": p4.energy_state_bind,
        "P4_REALIZED_GRID_CAP_ACTIVE_OR_VIOLATED": p4.realized_grid_cap_active_or_violated,
        "P4_ANY_BINDING_CONDITION": p4[["power_limit_bind", "energy_state_bind", "realized_grid_cap_active_or_violated"]].any(axis=1),
    }
    for label, flag in flags.items():
        for value in [False, True]:
            dates = flag.index[flag.eq(value)]
            subset = wide_cost.loc[wide_cost.index.intersection(dates)]
            for comparison, a, b in [
                ("P2_MINUS_P4", "P2_LGBM_DETERMINISTIC", "P4_S2_COHERENT"),
                ("P3_MINUS_P4", "P3_S1_INDEPENDENT", "P4_S2_COHERENT"),
            ]:
                diff = subset[a] - subset[b]
                rows.append({
                    "binding_definition": label,
                    "binding_value": value,
                    "comparison": comparison,
                    "days": len(diff),
                    "mean_cost_A_minus_B_eur_day": float(diff.mean()) if len(diff) else np.nan,
                    "policy_B_daily_win_fraction": float((diff > 0).mean()) if len(diff) else np.nan,
                    "classification": "EXPLORATORY_DIAGNOSTIC",
                })
    return pd.DataFrame(rows)


def _manifest(config: dict, outputs: list[Path]) -> dict:
    return {
        "experiment_id": config["experiment_id"],
        "starting_honda_commit": config["starting_honda_commit"],
        "spark_commit": config["spark_commit"],
        "frozen_config_sha256": EXPECTED_CONFIG_HASH,
        "timing_classification": "POST_PUBLICATION_TIMING_ASSUMED",
        "candidate_days": 273,
        "policies": list(POLICIES),
        "bootstrap_seed": config["statistics"]["bootstrap_seed"],
        "bootstrap_resamples": config["statistics"]["bootstrap_resamples"],
        "source_hashes": {key: value["sha256"] for key, value in config["frozen_inputs"].items()},
        "output_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in outputs},
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyarrow", "scipy", "pytest"]
        },
        "reproducibility_claim": "Deterministic table generation from immutable schedules, actuals, and prices; no optimization or scenario regeneration.",
        "labels": config["required_labels"],
    }


def run_replication() -> dict:
    daily, config = _day_level_metrics()
    primary, corrected, nominal = _comparisons(daily, config)
    summary, feasibility, overlap, binding = _summaries(daily)
    binding_effects = _binding_effects(daily)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "daily": OUT / "day_level_replication_metrics.parquet",
        "summary": OUT / "replication_policy_summary.csv",
        "primary": OUT / "primary_common_feasible_comparisons.csv",
        "corrected": OUT / "secondary_corrected_comparisons.csv",
        "nominal": OUT / "nominal_all_day_diagnostic_comparisons.csv",
        "feasibility": OUT / "feasibility_diagnostics.csv",
        "overlap": OUT / "violation_day_overlap.csv",
        "binding": OUT / "binding_constraint_diagnostics.csv",
        "binding_effects": OUT / "binding_effect_diagnostics.csv",
    }
    daily.to_parquet(paths["daily"], index=False, compression="zstd")
    for key, frame in [
        ("summary", summary), ("primary", primary), ("corrected", corrected), ("nominal", nominal),
        ("feasibility", feasibility), ("overlap", overlap), ("binding", binding),
        ("binding_effects", binding_effects),
    ]:
        write_csv(paths[key], frame)
    output_paths = [paths[key] for key in paths]
    manifest = _manifest(config, output_paths)
    manifest_path = OUT / "replication_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "config_hash": EXPECTED_CONFIG_HASH,
        "common_feasible_days": int(daily.loc[daily.common_feasible_day, "target_day_local"].nunique()),
        "summary": summary.to_dict(orient="records"),
        "primary": primary.to_dict(orient="records"),
        "corrected": corrected.to_dict(orient="records"),
        "nominal": nominal.to_dict(orient="records"),
        "manifest_path": str(manifest_path),
    }


def verify_reproducibility() -> dict:
    run_replication()
    paths = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "reproducibility_check.json")
    first = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in paths}
    run_replication()
    second = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in paths}
    result = {
        "passed": first == second,
        "comparison": "Two consecutive complete replications from immutable frozen inputs",
        "first_hashes": first,
        "second_hashes": second,
        "mismatches": sorted(key for key in first if first[key] != second.get(key)),
        "frozen_config_sha256": EXPECTED_CONFIG_HASH,
    }
    path = OUT / "reproducibility_check.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError("Replication outputs are not deterministic")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen SPARK Stage 3 feasibility replication")
    parser.add_argument("--json", action="store_true", help="Print compact JSON summary")
    parser.add_argument("--verify-reproducibility", action="store_true", help="Run twice and compare output hashes")
    args = parser.parse_args()
    result = verify_reproducibility() if args.verify_reproducibility else run_replication()
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))


if __name__ == "__main__":
    main()
