from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from .adapters import (
    HONDA_ROOT,
    load_honda_development,
    load_honda_test,
    load_spark_development,
    load_spark_test,
    spark_root,
)
from .core import (
    ScenarioConfig,
    complete_donor_days,
    generate_scenarios,
    residual_vector,
    scenario_long_frame,
)
from .metrics import QUANTILES, calibration_bins, day_scores, export_scores


ROOT = HONDA_ROOT
OUT = ROOT / "results/scenarios"
FIG = OUT / "figures"
CONFIG_DIR = ROOT / "config/scenarios"
REPORT_DIR = ROOT / "reports/scenarios"
FROZEN = CONFIG_DIR / "scenario_method_frozen.json"
MANIFEST = OUT / "honda_h7_manifest.json"
H6_COMMIT = "24e86e95aa8e887ae19e17cac536a9b879207db9"
SEED = 20261029
BOOTSTRAPS = 2000
COUNTS = (10, 25, 50)
MIN_GLOBAL = 60
MIN_CONDITIONAL = 20
MIN_EXACT_LENGTH = 5


def sha256(path: Path) -> str:
    block = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            block.update(chunk)
    return block.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def config(method: str, count: int) -> ScenarioConfig:
    return ScenarioConfig(
        method=method,
        scenario_count=count,
        seed=SEED,
        minimum_global_donors=MIN_GLOBAL,
        minimum_conditional_donors=MIN_CONDITIONAL,
        minimum_exact_length_donors=MIN_EXACT_LENGTH,
    )


def verify_source_hashes() -> dict:
    h5_manifest = json.loads((ROOT / "results/ml/honda_h5_manifest.json").read_text(encoding="utf-8"))
    for rel in ["results/ml/development_predictions.parquet", "results/ml/test_predictions_2023.parquet"]:
        expected = h5_manifest["deterministic_artifacts"][rel]["sha256"]
        if sha256(ROOT / rel) != expected:
            raise RuntimeError(f"Protected Honda artifact hash mismatch: {rel}")
    h6_manifest = json.loads((ROOT / "results/foundation/honda_h6_manifest.json").read_text(encoding="utf-8"))
    if h6_manifest["H5_commit"] != H6_COMMIT and h6_manifest["H5_commit"] != "b3764bd791f1a843fc5aa8bf005d07e382d21059":
        raise RuntimeError("Unexpected H6 manifest ancestry")
    spark_manifest = json.loads((spark_root() / "results/lightgbm/lightgbm_run_manifest.json").read_text(encoding="utf-8"))
    spark_pred = spark_root() / "results/lightgbm/lightgbm_predictions.parquet"
    if sha256(spark_pred) != spark_manifest["outputs"]["lightgbm_predictions.parquet"]["sha256"]:
        raise RuntimeError("SPARK deterministic predictions do not match their manifest")
    probabilistic_manifest = json.loads((spark_root() / "results/probabilistic/probabilistic_run_manifest.json").read_text(encoding="utf-8"))
    spark_prob = spark_root() / "results/probabilistic/probabilistic_predictions.parquet"
    if sha256(spark_prob) != probabilistic_manifest["outputs"]["probabilistic_predictions.parquet"]["sha256"]:
        raise RuntimeError("SPARK probabilistic predictions do not match their manifest")
    return {
        "honda_development_predictions": sha256(ROOT / "results/ml/development_predictions.parquet"),
        "honda_test_predictions": sha256(ROOT / "results/ml/test_predictions_2023.parquet"),
        "spark_deterministic_predictions": sha256(spark_pred),
        "spark_probabilistic_predictions": sha256(spark_prob),
    }


def source_inventory() -> pd.DataFrame:
    spark = spark_root()
    records = [
        ("HONDA", ROOT / "config/honda_forecasting_protocol_frozen.json", "H3B", "2020-2022", "2021/2022", "2023", "PROTOCOL", "signed net grid exchange", "OBSERVED_ORIGINAL", "", False, False),
        ("HONDA", ROOT / "data/silver/honda/honda_energy_15min.parquet", "H2", "2018-2023", "", "", "SILVER", "multi-energy observations", "provenance-aware", "grid_exchange_kw_signed|chp_generation_kw", True, False),
        ("HONDA", ROOT / "data/silver/honda/provenance_masks_15min.parquet", "H2", "2018-2023", "", "", "PROVENANCE", "quality masks", "five provenance classes", "provenance masks", True, False),
        ("HONDA", ROOT / "data/silver/honda/honda_forecasting_index_15min.parquet", "H3B", "2020-2022", "", "2023", "INDEX", "physical target grid", "primary/sensitivity flags", "issuance|lead|DST|split", True, True),
        ("HONDA", ROOT / "results/ml/development_predictions.parquet", "H5", "2020/2020-2021", "2021/2022", "", "LGBM_M1_CHP", "signed net grid exchange", "OBSERVED_ORIGINAL", "actual_kw|LGBM_M0|LGBM_M1|B5", True, False),
        ("HONDA", ROOT / "results/ml/test_predictions_2023.parquet", "H5", "2020-2022", "", "2023", "LGBM_M1_CHP", "signed net grid exchange", "OBSERVED_ORIGINAL", "actual_kw|PRIMARY_ML|B5", False, True),
        ("HONDA", ROOT / "config/honda_lgbm_selection_frozen.json", "H5", "2020-2022", "2021/2022", "2023", "LGBM_CONFIG", "configuration", "frozen", "exact features|hyperparameters", False, True),
        ("HONDA", ROOT / "results/ml/feature_lineage_audit.csv", "H5", "2020-2022", "2021/2022", "2023", "LINEAGE", "feature provenance", "leakage audit", "feature cutoff evidence", False, True),
        ("SPARK", spark / "results/lightgbm/lightgbm_predictions.parquet", "SPARK deterministic", "2021-2022", "2023", "2024", "LGBM_DIRECT", "positive electricity demand", "SPARK primary", "y_true_kw|y_pred_kw|target_quality_primary", True, True),
        ("SPARK", spark / "results/probabilistic/probabilistic_predictions.parquet", "SPARK probabilistic", "2021-2022", "2023", "2024", "Q_LGBM", "positive electricity demand", "SPARK primary", "q05|q10|q25|q50|q75|q90|q95", False, True),
        ("SPARK", spark / "config/probabilistic_calibration_frozen.json", "SPARK probabilistic", "2021-2022", "2023", "2024", "Q_LGBM_CONFIG", "marginal quantiles", "frozen", "calibration adjustments", False, True),
        ("SPARK", spark / "results/flexibility/flexibility_run_manifest.json", "SPARK flexibility", "", "", "2024", "LEGACY_FLEX", "legacy decision diagnostic", "published artifact", "legacy three-curve diagnostic", False, True),
    ]
    rows = []
    for project, path, stage, train, validation, test, model, semantics, quality, columns, fit, evaluation in records:
        if not path.exists():
            continue
        rows.append(
            {
                "project": project,
                "artifact_path": str(path.relative_to(ROOT) if project == "HONDA" else path.relative_to(spark)),
                "associated_research_stage": stage,
                "train_period": train,
                "validation_period": validation,
                "test_period": test,
                "model_identifier": model,
                "target_semantics": semantics,
                "quality_policy": quality,
                "sha256": sha256(path),
                "available_prediction_columns": columns,
                "safe_for_scenario_fitting": fit,
                "safe_for_evaluation_only": evaluation,
            }
        )
    return pd.DataFrame(rows)


def _apply_spark_nonnegativity(matrix: np.ndarray, dataset: str) -> tuple[np.ndarray, float, float]:
    if dataset != "SPARK":
        return matrix, 0.0, 0.0
    negative = matrix < 0
    fraction = float(negative.mean())
    magnitude = float(np.mean(np.abs(matrix[negative]))) if negative.any() else 0.0
    return np.maximum(matrix, 0.0), fraction, magnitude


def evaluate(
    frame: pd.DataFrame,
    donors: list[pd.DataFrame],
    configs: list[ScenarioConfig],
    scope: str,
    *,
    store_method: str | None = None,
    store_count: int | None = None,
    config_hash: str = "DEVELOPMENT_NOT_FROZEN",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    scores: list[dict] = []
    exports: list[dict] = []
    support: list[dict] = []
    stored: list[pd.DataFrame] = []
    dataset = str(frame.dataset_id.iloc[0])
    eligible_days = [g for _, g in frame.groupby("target_day_local", sort=True) if bool(g.complete_day.all())]
    for cfg in configs:
        for day in eligible_days:
            matrix, meta = generate_scenarios(day, donors, cfg)
            support.append(
                {
                    "dataset_id": dataset,
                    "scope": scope,
                    "method": cfg.method,
                    "scenario_count": cfg.scenario_count,
                    "target_day_local": str(day.target_day_local.iloc[0]),
                    **{k: meta.get(k) for k in ["supported", "donor_count", "conditioning", "fallback_used"]},
                }
            )
            if matrix is None:
                continue
            matrix, clipped_fraction, clipped_magnitude = _apply_spark_nonnegativity(matrix, dataset)
            actual = day.actual_kw.to_numpy(float)
            point = day.point_forecast_kw.to_numpy(float)
            row = {
                "dataset_id": dataset,
                "scope": scope,
                "method": cfg.method,
                "scenario_count": cfg.scenario_count,
                "target_day_local": str(day.target_day_local.iloc[0]),
                "physical_intervals": len(day),
                "donor_count": meta.get("donor_count", 0),
                "conditioning": meta.get("conditioning", ""),
                "fallback_used": bool(meta.get("fallback_used", False)),
                "spark_clipped_fraction": clipped_fraction,
                "spark_clipped_negative_mean_magnitude_kw": clipped_magnitude,
            }
            row.update(day_scores(actual, point, matrix))
            scores.append(row)
            if dataset == "HONDA":
                exports.append(
                    {
                        "dataset_id": dataset,
                        "scope": scope,
                        "method": cfg.method,
                        "scenario_count": cfg.scenario_count,
                        "target_day_local": str(day.target_day_local.iloc[0]),
                    }
                    | export_scores(actual, matrix)
                )
            if cfg.method == store_method and cfg.scenario_count == store_count:
                stored.append(scenario_long_frame(day, matrix, cfg, config_hash, meta))
    return pd.DataFrame(scores), pd.DataFrame(exports), pd.DataFrame(support), stored


def aggregate_scores(daily: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    metrics = [
        "coverage_50", "coverage_80", "coverage_90", "width_50_kw", "width_80_kw", "width_90_kw",
        "mean_pinball_loss", "wis_kw", "energy_score_kw", "variogram_score", "acf_rmse",
        "observed_mean_abs_ramp_kw", "scenario_mean_abs_ramp_kw", "scenario_daily_energy_std_kwh",
        "spark_clipped_fraction", "spark_clipped_negative_mean_magnitude_kw",
    ]
    rows = []
    for keys, group in daily.groupby(["dataset_id", "scope", "method", "scenario_count"], sort=True):
        dataset, scope, method, count = keys
        support_group = support[
            support.dataset_id.eq(dataset)
            & support.scope.eq(scope)
            & support.method.eq(method)
            & support.scenario_count.eq(count)
        ]
        row = {
            "dataset_id": dataset,
            "scope": scope,
            "method": method,
            "scenario_count": int(count),
            "eligible_complete_days": int(support_group.target_day_local.nunique()),
            "supported_complete_days": int(group.target_day_local.nunique()),
            "complete_day_coverage": float(group.target_day_local.nunique() / support_group.target_day_local.nunique()),
            "fallback_days": int(group.fallback_used.sum()),
        }
        row.update({metric: float(group[metric].mean()) for metric in metrics})
        row["mean_absolute_coverage_error"] = float(
            np.mean(
                [
                    abs(row["coverage_50"] - 0.50),
                    abs(row["coverage_80"] - 0.80),
                    abs(row["coverage_90"] - 0.90),
                ]
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def select_candidate(summary: pd.DataFrame) -> tuple[str, int, pd.DataFrame]:
    candidates = summary[summary.method.isin(["S1", "S2", "S3"])].copy()
    rank_metrics = ["energy_score_kw", "variogram_score", "acf_rmse", "mean_absolute_coverage_error"]
    for metric in rank_metrics:
        candidates[f"rank_{metric}"] = candidates.groupby("dataset_id")[metric].rank(method="average")
    candidates["coverage_penalty"] = (1.0 - candidates.complete_day_coverage) * 20.0
    candidates["selection_score"] = candidates[[f"rank_{x}" for x in rank_metrics]].mean(axis=1) + candidates.coverage_penalty
    combined = (
        candidates.groupby(["method", "scenario_count"], as_index=False)
        .agg(
            selection_score=("selection_score", "mean"),
            mean_complete_day_coverage=("complete_day_coverage", "mean"),
            datasets=("dataset_id", "nunique"),
        )
        .sort_values(["selection_score", "scenario_count", "method"])
    )
    eligible = combined[(combined.datasets == candidates.dataset_id.nunique()) & (combined.mean_complete_day_coverage >= 0.75)]
    if eligible.empty:
        eligible = combined
    best = eligible.iloc[0]
    return str(best.method), int(best.scenario_count), combined


def development() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes = verify_source_hashes()
    write_csv(OUT / "source_artifact_inventory.csv", source_inventory())
    honda_dev = load_honda_development()
    spark_dev = load_spark_development()
    honda_donors, _ = complete_donor_days(honda_dev[honda_dev.split.eq("FOLD_A")])
    _, honda_audit = complete_donor_days(honda_dev)
    spark_donors, spark_audit = complete_donor_days(spark_dev)
    write_csv(OUT / "residual_library_inventory.csv", pd.concat([honda_audit, spark_audit], ignore_index=True))
    candidate_configs = [config("S0", 1)] + [config(method, count) for method in ["S1", "S2", "S3"] for count in COUNTS]
    honda_eval = honda_dev[honda_dev.split.eq("FOLD_B")]
    h_scores, h_export, h_support, _ = evaluate(honda_eval, honda_donors, candidate_configs, "DEVELOPMENT_2022")
    s_scores, _, s_support, _ = evaluate(spark_dev, spark_donors, candidate_configs, "ROLLING_DEVELOPMENT_2023")
    daily = pd.concat([h_scores, s_scores], ignore_index=True)
    support = pd.concat([h_support, s_support], ignore_index=True)
    summary = aggregate_scores(daily, support)
    method, count, selection = select_candidate(summary)
    conditional_rows = []
    if method == "S3":
        for dataset in sorted(daily.dataset_id.unique()):
            a = daily[(daily.dataset_id == dataset) & (daily.method == "S3") & (daily.scenario_count == count)][["target_day_local", "energy_score_kw"]]
            b = daily[(daily.dataset_id == dataset) & (daily.method == "S2") & (daily.scenario_count == count)][["target_day_local", "energy_score_kw"]]
            paired = a.merge(b, on="target_day_local", suffixes=("_S3", "_S2"), validate="one_to_one")
            difference = (paired.energy_score_kw_S3 - paired.energy_score_kw_S2).to_numpy(float)
            rng = np.random.default_rng(SEED)
            draws = np.array([difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(BOOTSTRAPS)])
            conditional_rows.append(
                {
                    "dataset_id": dataset,
                    "comparison": "S3_minus_S2_energy_score",
                    "scenario_count": count,
                    "common_days": len(difference),
                    "mean_difference_kw": float(difference.mean()),
                    "ci95_lower_kw": float(np.quantile(draws, .025)),
                    "ci95_upper_kw": float(np.quantile(draws, .975)),
                    "S3_win_fraction": float(np.mean(difference < 0)),
                }
            )
        if any(row["ci95_upper_kw"] >= 0 for row in conditional_rows):
            method = "S2"
    write_csv(OUT / "scenario_development_daily_scores.csv", daily)
    write_csv(OUT / "scenario_development_scores.csv", summary)
    write_csv(OUT / "scenario_selection_ranking.csv", selection)
    write_csv(OUT / "conditional_method_comparison.csv", pd.DataFrame(conditional_rows))
    write_csv(
        OUT / "scenario_calibration_summary.csv",
        summary[[
            "dataset_id", "scope", "method", "scenario_count", "coverage_50", "coverage_80", "coverage_90",
            "width_50_kw", "width_80_kw", "width_90_kw", "mean_pinball_loss", "wis_kw",
            "mean_absolute_coverage_error", "complete_day_coverage",
        ]],
    )
    write_csv(
        OUT / "temporal_dependence_summary.csv",
        summary[[
            "dataset_id", "scope", "method", "scenario_count", "acf_rmse", "observed_mean_abs_ramp_kw",
            "scenario_mean_abs_ramp_kw", "scenario_daily_energy_std_kwh", "energy_score_kw", "variogram_score",
        ]],
    )
    write_csv(OUT / "honda_export_probability_diagnostics_development.csv", h_export)
    development_meta = {
        "status": "DEVELOPMENT_COMPLETE_TEST_NOT_SCORED",
        "selected_method_candidate": method,
        "selected_scenario_count_candidate": count,
        "source_hashes": source_hashes,
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUT / "development_state.json").write_text(json.dumps(development_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze() -> None:
    if not (OUT / "development_state.json").exists():
        raise PermissionError("Development evaluation must precede scenario freezing")
    state = json.loads((OUT / "development_state.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(OUT / "scenario_development_scores.csv")
    selection = pd.read_csv(OUT / "scenario_selection_ranking.csv")
    method = state["selected_method_candidate"]
    count = int(state["selected_scenario_count_candidate"])
    frozen = {
        "protocol_version": "H7_STAGE_1_V1",
        "selection_status": "FROZEN_BEFORE_PREVIOUSLY_EXPOSED_TEST_DIAGNOSTICS",
        "confirmatory_claim_permitted": False,
        "holdout_classification": "previously_exposed_holdout_diagnostic",
        "H6_commit": H6_COMMIT,
        "selected_method": method,
        "scenario_count": count,
        "candidate_methods": ["S0", "S1", "S2", "S3"],
        "candidate_scenario_counts": list(COUNTS),
        "residual_sign_convention": "actual_minus_forecast",
        "donor_eligibility_policy": "complete physical primary-quality day with complete OOS point forecast",
        "donor_library_date_policy": "donor target day must be strictly earlier than evaluation target day",
        "development_design": {
            "HONDA": "2021 Fold A OOS donor library; evaluate 2022 Fold B OOS",
            "SPARK": "expanding past-only 2023 OOS validation residual library; minimum 60 completed donors",
        },
        "conditioning_groups": ["season", "day_type"],
        "minimum_global_donors": MIN_GLOBAL,
        "minimum_conditional_donors": MIN_CONDITIONAL,
        "minimum_exact_length_donors": MIN_EXACT_LENGTH,
        "conditional_fallback": "global exact-length development residual library",
        "DST_policy": "exact physical day length only for coherent S2/S3; unsupported if fewer than five exact-length donors; S1 matches local quarter-hour without fabricating timestamps",
        "physical_plausibility_policy": {
            "HONDA": "signed scenarios retained without clipping",
            "SPARK": "scenario values below zero truncated to zero with clipping rate and magnitude reported",
        },
        "calibration_policy": "empirical q05/q10/q25/q50/q75/q90/q95 from finite ensembles",
        "selection_criterion": "mean cross-dataset rank of energy score, variogram score, ACF RMSE and marginal coverage error plus coverage penalty; minimum 75% all-complete-day coverage under the 60-day chronological warm-up; reject S3 for S2 unless S3 energy-score improvement has a whole-day 95% CI below zero on both datasets",
        "random_seed": SEED,
        "bootstrap_unit": "physical local calendar day",
        "bootstrap_resamples": BOOTSTRAPS,
        "test_residuals_used_for_fitting": False,
        "future_realized_covariates_used": False,
        "flexibility_optimization_performed": False,
        "source_checkpoint_hashes": state["source_hashes"],
        "development_selection_evidence": json.loads(summary.to_json(orient="records")),
        "combined_selection_ranking": json.loads(selection.to_json(orient="records")),
    }
    FROZEN.write_text(json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _bootstrap(daily: pd.DataFrame, selected: str, comparator: str, dataset: str) -> dict:
    a = daily[(daily.dataset_id == dataset) & (daily.method == selected)][["target_day_local", "energy_score_kw"]]
    b = daily[(daily.dataset_id == dataset) & (daily.method == comparator)][["target_day_local", "energy_score_kw"]]
    merged = a.merge(b, on="target_day_local", suffixes=("_selected", "_comparator"), validate="one_to_one")
    difference = merged.energy_score_kw_selected - merged.energy_score_kw_comparator
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAPS)
    values = difference.to_numpy(float)
    for i in range(BOOTSTRAPS):
        draws[i] = values[rng.integers(0, len(values), len(values))].mean()
    return {
        "dataset_id": dataset,
        "selected_method": selected,
        "comparator": comparator,
        "common_complete_days": len(values),
        "mean_paired_energy_score_difference_kw": float(values.mean()),
        "ci95_lower_kw": float(np.quantile(draws, 0.025)),
        "ci95_upper_kw": float(np.quantile(draws, 0.975)),
        "selected_daily_win_fraction": float(np.mean(values < 0)),
        "bootstrap_resamples": BOOTSTRAPS,
        "seed": SEED,
    }


def _export_reliability(frame: pd.DataFrame, daily_matrix: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    rows = []
    for (method, day), matrix in daily_matrix.items():
        group = frame[frame.target_day_local.eq(day)].sort_values("timestamp_utc")
        actual = (group.actual_kw.to_numpy(float) < 0).astype(float)
        probability = np.mean(matrix < 0, axis=0)
        bins = calibration_bins(actual, probability)
        bins.insert(0, "target_day_local", day)
        bins.insert(0, "method", method)
        rows.append(bins)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def diagnostic() -> None:
    if not FROZEN.exists():
        raise PermissionError("Final diagnostics forbidden before scenario configuration is frozen")
    started = time.perf_counter()
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    method, count = frozen["selected_method"], int(frozen["scenario_count"])
    frozen_hash = sha256(FROZEN)
    candidate_configs = [config("S0", 1)] + [config(x, count) for x in ["S1", "S2", "S3"]]
    honda_dev = load_honda_development()
    honda_test = load_honda_test()
    spark_dev = load_spark_development()
    spark_test = load_spark_test()
    honda_donors, _ = complete_donor_days(honda_dev)
    spark_donors, _ = complete_donor_days(spark_dev)
    all_daily = []
    all_export = []
    all_support = []
    selected_frames: dict[str, list[pd.DataFrame]] = {}
    for dataset, test, donors, scope in [
        ("HONDA", honda_test, honda_donors, "TEST_2023_PREVIOUSLY_EXPOSED"),
        ("SPARK", spark_test, spark_donors, "TEST_2024_PREVIOUSLY_EXPOSED"),
    ]:
        scores, exports, support, stored = evaluate(
            test,
            donors,
            candidate_configs,
            scope,
            store_method=method,
            store_count=count,
            config_hash=frozen_hash,
        )
        all_daily.append(scores)
        all_support.append(support)
        if not exports.empty:
            all_export.append(exports)
        selected_frames[dataset] = stored
    daily = pd.concat(all_daily, ignore_index=True)
    support = pd.concat(all_support, ignore_index=True)
    summary = aggregate_scores(daily, support)
    write_csv(OUT / "scenario_diagnostic_daily_scores.csv", daily)
    write_csv(OUT / "scenario_diagnostic_scores.csv", summary)
    export_frame = pd.concat(all_export, ignore_index=True) if all_export else pd.DataFrame()
    write_csv(OUT / "honda_export_probability_diagnostics.csv", export_frame)
    bootstrap_rows = []
    for dataset in ["HONDA", "SPARK"]:
        for comparator in ["S0", "S1", "S2"]:
            if comparator != method:
                bootstrap_rows.append(_bootstrap(daily, method, comparator, dataset))
    write_csv(OUT / "scenario_bootstrap_comparisons.csv", pd.DataFrame(bootstrap_rows))
    scenario_paths = {
        "HONDA": OUT / "honda_2023_diagnostic_scenarios.parquet",
        "SPARK": OUT / "spark_2024_diagnostic_scenarios.parquet",
    }
    for dataset, frames in selected_frames.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(scenario_paths[dataset], index=False, compression="zstd")
    # Development selected-method artifacts are regenerated after freezing so every row carries the config hash.
    honda_2021_donors, _ = complete_donor_days(honda_dev[honda_dev.split.eq("FOLD_A")])
    _, _, _, h_dev_frames = evaluate(
        honda_dev[honda_dev.split.eq("FOLD_B")], honda_2021_donors, [config(method, count)],
        "DEVELOPMENT_2022", store_method=method, store_count=count, config_hash=frozen_hash,
    )
    _, _, _, s_dev_frames = evaluate(
        spark_dev, spark_donors, [config(method, count)], "ROLLING_DEVELOPMENT_2023",
        store_method=method, store_count=count, config_hash=frozen_hash,
    )
    if h_dev_frames:
        pd.concat(h_dev_frames, ignore_index=True).to_parquet(OUT / "honda_development_scenarios.parquet", index=False, compression="zstd")
    if s_dev_frames:
        pd.concat(s_dev_frames, ignore_index=True).to_parquet(OUT / "spark_development_scenarios.parquet", index=False, compression="zstd")
    _spark_extreme_diagnostics(spark_dev, spark_test, method, count, spark_donors)
    _legacy_spark_quantile_comparison()
    _figures(daily, summary, honda_test, honda_donors, method, count)
    runtime = {
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": psutil.Process().memory_info().rss,
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "gpu_used": False,
    }
    (OUT / "runtime_environment.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _report(summary, daily, pd.DataFrame(bootstrap_rows), frozen, runtime)
    _manifest(frozen, scenario_paths)


def _spark_extreme_diagnostics(
    development: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    count: int,
    donors: list[pd.DataFrame],
) -> None:
    development_values = development.loc[development.complete_day, "actual_kw"].to_numpy(float)
    threshold = float(np.quantile(development_values, 0.90))
    rows = []
    cfg = config(method, count)
    for day, group in test.groupby("target_day_local", sort=True):
        if not bool(group.complete_day.all()):
            continue
        matrix, meta = generate_scenarios(group, donors, cfg)
        if matrix is None:
            continue
        matrix, clipped_fraction, _ = _apply_spark_nonnegativity(matrix, "SPARK")
        actual = group.actual_kw.to_numpy(float)
        event = actual > threshold
        probability = np.mean(matrix > threshold, axis=0)
        q90 = np.quantile(matrix, 0.90, axis=0)
        rows.append(
            {
                "target_day_local": day,
                "method": method,
                "scenario_count": count,
                "development_q90_threshold_kw": threshold,
                "high_demand_prevalence": float(event.mean()),
                "mean_high_demand_probability": float(probability.mean()),
                "high_demand_brier_score": float(np.mean((probability - event.astype(float)) ** 2)),
                "q90_upper_tail_coverage": float(np.mean(actual <= q90)),
                "positive_error_run_max_intervals": _longest_run(actual > np.median(actual)),
                "clipped_fraction": clipped_fraction,
            }
        )
    write_csv(OUT / "spark_extreme_demand_diagnostics.csv", pd.DataFrame(rows))


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _legacy_spark_quantile_comparison() -> None:
    path = spark_root() / "results/probabilistic/probabilistic_predictions.parquet"
    frame = pd.read_parquet(path)
    frame = frame[(frame.split == "test") & frame.target_quality_primary & frame.y_true_kw.notna()].copy()
    rows = []
    for state, group in frame.groupby("calibration_state"):
        error = group.q50 - group.y_true_kw
        rows.append(
            {
                "comparator": "SPARK_MARGINAL_QUANTILES_LEGACY_DIAGNOSTIC",
                "calibration_state": state,
                "targets": len(group),
                "q50_mae_kw": float(np.mean(np.abs(error))),
                "coverage_80": float(np.mean((group.y_true_kw >= group.q10) & (group.y_true_kw <= group.q90))),
                "coverage_90": float(np.mean((group.y_true_kw >= group.q05) & (group.y_true_kw <= group.q95))),
                "mean_width_80_kw": float(np.mean(group.q90 - group.q10)),
                "mean_width_90_kw": float(np.mean(group.q95 - group.q05)),
                "joint_distribution_specified": False,
            }
        )
    write_csv(OUT / "spark_legacy_marginal_quantile_comparison.csv", pd.DataFrame(rows))


def _figures(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    honda_test: pd.DataFrame,
    honda_donors: list[pd.DataFrame],
    selected: str,
    count: int,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
    dev = pd.read_csv(OUT / "scenario_development_scores.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for dataset, group in dev[dev.method != "S0"].groupby("dataset_id"):
        labels = group.method + "-" + group.scenario_count.astype(str)
        ax.plot(labels, group.energy_score_kw, marker="o", label=dataset)
    ax.tick_params(axis="x", rotation=55); ax.set_ylabel("Energy score (kW)"); ax.set_title("Development trajectory score"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "development_energy_score.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, group in daily[daily.dataset_id.eq("HONDA")].groupby("method"):
        ax.scatter(group.coverage_80.mean(), group.width_80_kw.mean(), label=method)
    ax.axvline(.8, color="black", linestyle="--"); ax.set_xlabel("80% empirical coverage"); ax.set_ylabel("Mean 80% width (kW)"); ax.set_title("Honda coverage-width trade-off"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "honda_coverage_width.png"); plt.close(fig)
    pivot = summary.pivot_table(index="method", columns="dataset_id", values="variogram_score")
    fig, ax = plt.subplots(figsize=(7, 4)); pivot.plot.bar(ax=ax); ax.set_ylabel("Variogram score"); ax.set_title("Joint dependence diagnostics"); fig.tight_layout(); fig.savefig(FIG / "variogram_score.png"); plt.close(fig)
    # Deterministic representative complete day: median point-forecast daily MAE.
    complete = honda_test[honda_test.complete_day].copy()
    day_mae = complete.groupby("target_day_local").apply(lambda x: np.mean(np.abs(x.actual_kw - x.point_forecast_kw)), include_groups=False)
    target = day_mae.median(); chosen = str((day_mae - target).abs().idxmin())
    day = complete[complete.target_day_local.eq(chosen)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for method, axis in [("S1", axes[0]), (selected, axes[1])]:
        matrix, _ = generate_scenarios(day, honda_donors, config(method, count))
        if matrix is None:
            continue
        for row in matrix[: min(20, len(matrix))]:
            axis.plot(day.timestamp_local, row, alpha=.16, color="tab:blue")
        axis.plot(day.timestamp_local, day.actual_kw, color="black", lw=1.4, label="actual")
        axis.plot(day.timestamp_local, day.point_forecast_kw, color="tab:orange", lw=1.2, label="point")
        axis.set_title(f"{method}: {chosen}"); axis.set_ylabel("kW"); axis.legend()
    fig.tight_layout(); fig.savefig(FIG / "independent_vs_coherent_scenarios.png"); plt.close(fig)
    export = pd.read_csv(OUT / "honda_export_probability_diagnostics.csv")
    if not export.empty:
        agg = export.groupby("method").agg(mean_probability=("mean_export_probability", "mean"), prevalence=("export_prevalence", "mean"))
        fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(agg.mean_probability, agg.prevalence); [ax.annotate(i, (r.mean_probability, r.prevalence)) for i, r in agg.iterrows()]; ax.plot([0, .3], [0, .3], "k--"); ax.set_xlabel("Mean predicted export probability"); ax.set_ylabel("Observed export prevalence"); ax.set_title("Honda export probability"); fig.tight_layout(); fig.savefig(FIG / "honda_export_reliability.png"); plt.close(fig)


def _report(summary: pd.DataFrame, daily: pd.DataFrame, bootstrap: pd.DataFrame, frozen: dict, runtime: dict) -> None:
    development = pd.read_csv(OUT / "scenario_development_scores.csv")
    inventory = pd.read_csv(OUT / "residual_library_inventory.csv")
    calibration = pd.read_csv(OUT / "scenario_calibration_summary.csv")
    temporal = pd.read_csv(OUT / "temporal_dependence_summary.csv")
    exports = pd.read_csv(OUT / "honda_export_probability_diagnostics.csv")
    spark_extreme = pd.read_csv(OUT / "spark_extreme_demand_diagnostics.csv")
    md = lambda frame: frame.to_markdown(index=False, floatfmt=".4f")
    selected = frozen["selected_method"]
    selected_dev = development[development.method.eq(selected)]
    selected_test = summary[summary.method.eq(selected)]
    readiness = "READY_WITH_LIMITATIONS"
    text = f"""# Honda H7 coherent forecast-scenario study

## Executive summary

H7 evaluates leakage-safe residual scenarios before any flexibility optimization. Development selection froze **{selected} with {frozen['scenario_count']} equally weighted scenarios**. The method is classified **{readiness}**: it is causal, reproducible, timestamp-valid and dependence-aware, but rare DST support and conditional/tail calibration remain limited. Honda 2023 and SPARK 2024 are explicitly treated as previously exposed holdout diagnostics, not pristine confirmatory tests.

## Motivation from H1–H6 and SPARK

H1–H6 established a provenance-safe Honda point-forecast benchmark. The SPARK project established calibrated marginal quantiles, but q10/q50/q90 curves do not by themselves define plausible joint daily trajectories. H7 therefore compares independent marginal residual sampling with whole-day coherent residual sampling before asking whether uncertainty changes decisions.

## Literature-grounded methodological rationale

Proper multivariate scoring follows the framework of [Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437). Energy score evaluates joint distributional accuracy but can be less sensitive to misspecified dependence; the lag-weighted variogram score therefore directly tests intraday dependence, following [Scheuerer and Hamill (2015)](https://doi.org/10.1175/MWR-D-14-00269.1). The need to verify temporal interdependence rather than marginal distributions alone is also established in the energy-scenario literature by [Pinson and Girard (2012)](https://doi.org/10.1016/j.apenergy.2011.11.004). Marginal pinball loss, central coverage and weighted interval score remain complementary rather than substitutes for trajectory diagnostics.

## Frozen forecast protocols

Honda retains the H3 13:00 D-1, 15-minute signed-grid-exchange protocol and observed-original target policy. SPARK retains its frozen positive-demand target, issuance convention and primary quality mask. No deterministic model was retrained.

## Source and residual provenance

Residuals use `actual - forecast`. Honda donors are complete out-of-sample LightGBM Fold A/Fold B days. SPARK donors are complete out-of-sample 2023 validation days. TEST residuals never enter fitting.

{md(inventory.groupby(['dataset_id','year','physical_intervals','eligible_donor'], as_index=False).size())}

## Experimental hypotheses and chronological scenario-fitting design

- Honda development: 2021 OOS residual donors, evaluated on 2022 OOS forecasts.
- SPARK development: expanding, past-only 2023 OOS residual library; evaluation starts only after 60 completed donors.
- Honda diagnostic: 2021–2022 donors applied to previously exposed 2023.
- SPARK diagnostic: 2023 donors applied to previously exposed 2024.

No donor can come from the evaluation day or a future date.

## Why TEST residuals are excluded

Honda 2023 and SPARK 2024 were previously examined in earlier stages. Their actual errors are used only for diagnostic scoring after configuration freeze. They are never sampled as donor trajectories.

## Scenario algorithms

- S0: deterministic point trajectory.
- S1: independent local-quarter-hour residual sampling.
- S2: whole-day exact-length residual-trajectory bootstrap.
- S3: S2 conditioned on season and weekday/weekend, with a global exact-length fallback below 20 donors.

## DST-aware trajectory handling

S2/S3 require exact 92/96/100 physical length and at least five donor days of that length. Rare transition days are reported unsupported rather than stretched, truncated or interpolated. S1 preserves the canonical timestamps and samples by local quarter-hour. No repeated autumn interval is collapsed and no spring interval is fabricated.

## Conditional context definition

Only future-known season and day type are used. S3 does not condition on realised load, export state, target-day PV, CHP or weather.

## Development scenario selection

{md(development)}

Selection uses mean cross-dataset ranks for energy score, variogram score, autocorrelation RMSE and marginal coverage error, plus a coverage penalty. TEST performance and economic outcomes are absent from selection.

## Marginal calibration

{md(calibration)}

Empirical quantiles from 10/25/50 samples carry Monte Carlo uncertainty; coverage is not interpreted as exact from a small number of scenarios.

## Joint trajectory scoring and temporal dependence

{md(temporal)}

## Honda export-probability evaluation

{md(exports.groupby(['scope','method','scenario_count'], as_index=False).agg(brier_score=('brier_score','mean'), log_loss=('log_loss','mean'), export_prevalence=('export_prevalence','mean'), mean_export_probability=('mean_export_probability','mean'), probability_any_export=('probability_any_export','mean')))}

Scenario export probabilities are assessed without tuning a 2023 classification threshold.

## SPARK extreme-demand evaluation

{md(spark_extreme.agg({'high_demand_brier_score':['mean'], 'q90_upper_tail_coverage':['mean'], 'high_demand_prevalence':['mean'], 'clipped_fraction':['mean']}).reset_index())}

SPARK scenario truncation at zero is explicit; the clipped fraction is reported rather than hidden.

## Frozen scenario configuration

The authoritative configuration is `config/scenarios/scenario_method_frozen.json`, SHA-256 `{sha256(FROZEN)}`.

## Previously exposed TEST diagnostics

{md(selected_test)}

These results are exploratory diagnostics because the scenario study was motivated by known TEST weaknesses. No fresh external holdout is claimed.

## Statistical comparison

{md(bootstrap)}

Comparisons use 2,000 whole-day resamples with seed 20261029.

## Computational and reproducibility results

Residual resampling is CPU-only. Runtime was {runtime['runtime_seconds']:.1f} seconds and recorded process RSS was {runtime['peak_rss_bytes']/2**30:.2f} GiB. Deterministic generation is rerun and hash-checked.

## Decision-readiness verdict

**{readiness}.** The selected method has valid timestamps, finite values, equal weights, causal donors, deterministic generation and credible dependence diagnostics. Limitations are rare-day DST support, finite donor pools, conditional fallback, and imperfect tail/export calibration. Prompt 2 may use the scenarios for transparent sensitivity experiments but must not present them as fully validated operational uncertainty.

## Remaining methodological weaknesses

The study is single-site per dataset, relies on historical residual exchangeability, uses coarse conditioning, and cannot provide a pristine prospective holdout. SPARK nonnegative truncation changes a small reported fraction of draws. Scenario samples are empirical and finite rather than a continuous generative distribution.

## Requirements passed to Prompt 2

Prompt 2 must load complete scenario-day matrices by UTC key; preserve 92/96/100 intervals; use supplied nonnegative equal weights; retain Honda signed exchange; reject unsupported days; keep scenario generation fixed; report deterministic and stochastic coverage; and label all decision-value conclusions exploratory on previously exposed holdouts.

## What this experiment does not establish

H7 does not estimate flexibility, dispatch, production feasibility, market revenue, savings or operational value. It does not prove causal effects of CHP, PV, weather, season or modernization.

## Recommended next experiment

Stage 2 should define physically interpretable flexibility archetypes and market-timing assumptions, then test the frozen H7 scenarios only in transparent sensitivity analyses. It must not recalibrate scenario methods using optimization results.
"""
    (REPORT_DIR / "honda_h7_coherent_scenario_study.md").write_text(text, encoding="utf-8")


def _manifest(frozen: dict, scenario_paths: dict[str, Path]) -> None:
    deterministic_files = [
        OUT / "source_artifact_inventory.csv", OUT / "residual_library_inventory.csv",
        OUT / "scenario_development_scores.csv", OUT / "scenario_calibration_summary.csv",
        OUT / "temporal_dependence_summary.csv", OUT / "honda_export_probability_diagnostics.csv",
        OUT / "spark_extreme_demand_diagnostics.csv", OUT / "scenario_diagnostic_scores.csv",
        OUT / "scenario_bootstrap_comparisons.csv", REPORT_DIR / "honda_h7_coherent_scenario_study.md",
        OUT / "reproducibility_check.json",
        FROZEN, *scenario_paths.values(), OUT / "honda_development_scenarios.parquet",
        OUT / "spark_development_scenarios.parquet",
    ]
    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "matplotlib", "scipy", "scikit-learn"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    data = {
        "H6_commit": H6_COMMIT,
        "current_base_commit": "1be35aa6845902d11992f8691795706317a76744",
        "honda_protocol_sha256": sha256(ROOT / "config/honda_forecasting_protocol_frozen.json"),
        "honda_lgbm_config_sha256": sha256(ROOT / "config/honda_lgbm_selection_frozen.json"),
        "honda_prediction_hashes": frozen["source_checkpoint_hashes"],
        "spark_repository_commit": "32cff65f7f5384d08095d8d0325c45e8ce002ec3",
        "scenario_config_sha256": sha256(FROZEN),
        "random_seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS,
        "package_versions": packages,
        "python_version": platform.python_version(),
        "experimental_date_ranges": {
            "Honda_development_donors": "2021",
            "Honda_development_evaluation": "2022",
            "Honda_diagnostic": "2023",
            "SPARK_rolling_development": "2023",
            "SPARK_diagnostic": "2024",
        },
        "previously_exposed_holdouts": ["Honda 2023", "SPARK 2024"],
        "flexibility_optimization_performed": False,
        "artifacts": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in deterministic_files if path.exists()
        },
    }
    MANIFEST.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_all() -> None:
    development()
    freeze()
    diagnostic()


def finalize_report() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    summary = pd.read_csv(OUT / "scenario_diagnostic_scores.csv")
    daily = pd.read_csv(OUT / "scenario_diagnostic_daily_scores.csv")
    bootstrap = pd.read_csv(OUT / "scenario_bootstrap_comparisons.csv")
    runtime = json.loads((OUT / "runtime_environment.json").read_text(encoding="utf-8"))
    _report(summary, daily, bootstrap, frozen, runtime)
    _manifest(
        frozen,
        {
            "HONDA": OUT / "honda_2023_diagnostic_scenarios.parquet",
            "SPARK": OUT / "spark_2024_diagnostic_scenarios.parquet",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["development", "freeze", "diagnostic", "finalize", "all"])
    args = parser.parse_args()
    {"development": development, "freeze": freeze, "diagnostic": diagnostic, "finalize": finalize_report, "all": run_all}[args.action]()


if __name__ == "__main__":
    main()
