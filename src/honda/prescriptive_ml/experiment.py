from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import (
    PromotionGates,
    evaluate_point_forecast,
    interval_reliability,
    paired_day_bootstrap,
    promotion_report,
    weighted_interval_score,
    write_json,
)
from .forecasting import GroupedIntervalCalibrator, QuantileLightGBM, RegimeAwareMixture

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ROOT / "results/ml/h5_feature_matrix.parquet"
CONFIG = ROOT / "config/honda_lgbm_selection_frozen.json"
DEV_CHAMPION = ROOT / "results/ml/development_predictions.parquet"
TEST_CHAMPION = ROOT / "results/ml/test_predictions_2023.parquet"
OUTPUT = ROOT / "results/prescriptive_ml"
QUANTILES = (.05, .10, .25, .50, .75, .90, .95)


def _season(month: pd.Series) -> pd.Series:
    return month.map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"})


def _eligible(frame: pd.DataFrame) -> pd.Series:
    return frame.target_primary_eligible & frame.actual_kw.notna()


def _calibration_days(frame: pd.DataFrame, train_years: tuple[int, ...], count: int) -> set[str]:
    values = frame[_eligible(frame) & frame.year.isin(train_years)].target_day_local.astype(str)
    days = sorted(values.unique())
    if len(days) <= count:
        raise ValueError("Insufficient training days for a held-out calibration tail")
    return set(days[-count:])


def _regime_prediction(
    frame: pd.DataFrame,
    features: list[str],
    train_years: tuple[int, ...],
    evaluation_year: int,
    calibration_days: int,
) -> tuple[np.ndarray, dict[str, float]]:
    calibration = _calibration_days(frame, train_years, calibration_days)
    train = _eligible(frame) & frame.year.isin(train_years)
    fit = train & ~frame.target_day_local.astype(str).isin(calibration)
    tune = train & frame.target_day_local.astype(str).isin(calibration)
    evaluate = _eligible(frame) & frame.year.eq(evaluation_year)
    model = RegimeAwareMixture({"n_estimators": 196, "num_leaves": 15, "learning_rate": .03, "min_child_samples": 50, "feature_fraction": .8, "n_jobs": 4})
    model.fit(frame.loc[fit, features], frame.loc[fit, "actual_kw"], frame.loc[tune, features], frame.loc[tune, "actual_kw"])
    model.refit_components(frame.loc[train, features], frame.loc[train, "actual_kw"])
    prediction = model.predict(frame.loc[evaluate, features])
    return prediction, {"blend_weight": model.blend_weight, "fit_rows": int(fit.sum()), "calibration_rows": int(tune.sum()), "refit_rows": int(train.sum())}


def _quantile_prediction(
    frame: pd.DataFrame,
    features: list[str],
    fit_years: tuple[int, ...],
    calibration_year: int,
    evaluation_year: int,
    calibration_tail_days: int = 60,
) -> tuple[np.ndarray, dict[str, object]]:
    fit = _eligible(frame) & frame.year.isin(fit_years)
    calibration = _eligible(frame) & frame.year.eq(calibration_year)
    if calibration_year in fit_years:
        days = _calibration_days(frame, fit_years, calibration_tail_days)
        calibration = _eligible(frame) & frame.target_day_local.astype(str).isin(days)
        fit = fit & ~frame.target_day_local.astype(str).isin(days)
    evaluate = _eligible(frame) & frame.year.eq(evaluation_year)
    model = QuantileLightGBM(QUANTILES, {"n_estimators": 196, "num_leaves": 15, "learning_rate": .03, "min_child_samples": 50, "feature_fraction": .8, "n_jobs": 4})
    model.fit(frame.loc[fit, features], frame.loc[fit, "actual_kw"])
    calibration_raw = model.predict(frame.loc[calibration, features])
    evaluation_raw = model.predict(frame.loc[evaluate, features])
    calibration_keys = list(zip(frame.loc[calibration, "lead_band"].astype(str), _season(frame.loc[calibration, "month"])))
    evaluation_keys = list(zip(frame.loc[evaluate, "lead_band"].astype(str), _season(frame.loc[evaluate, "month"])))
    calibrator = GroupedIntervalCalibrator(.9, minimum_group_rows=300, shrinkage_rows=500)
    calibrator.fit(frame.loc[calibration, "actual_kw"].to_numpy(), calibration_raw[:, 0], calibration_raw[:, 6], calibration_keys)
    lower, upper = calibrator.transform(evaluation_raw[:, 0], evaluation_raw[:, 6], evaluation_keys)
    calibrated = evaluation_raw.copy(); calibrated[:, 0] = lower; calibrated[:, 6] = upper
    calibrated = np.sort(calibrated, axis=1)
    return calibrated, {
        "fit_rows": int(fit.sum()), "calibration_rows": int(calibration.sum()),
        "global_correction_kw": calibrator.global_correction,
        "groups": {"|".join(key): asdict(value) for key, value in calibrator.groups.items()},
        "raw_predictions": evaluation_raw,
    }


def _join_champion(frame: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if any(year in {2021, 2022} for year in years):
        dev = pd.read_parquet(DEV_CHAMPION)
        dev["year"] = pd.to_datetime(dev.target_day_local).dt.year
        parts.append(dev[dev.year.isin(years)][["timestamp_utc", "PRIMARY_ML" if "PRIMARY_ML" in dev else "LGBM_M1", "B5"]].rename(columns={"LGBM_M1": "PRIMARY_ML"}))
    if 2023 in years:
        test = pd.read_parquet(TEST_CHAMPION)
        parts.append(test[["timestamp_utc", "PRIMARY_ML", "B5"]])
    champion = pd.concat(parts, ignore_index=True)
    selected = frame[_eligible(frame) & frame.year.isin(years)].copy()
    return selected.merge(champion, on="timestamp_utc", how="left", validate="one_to_one")


def _maximum_slice_degradation(frame: pd.DataFrame, challenger: str) -> float:
    values: list[float] = []
    for column in ("lead_band", "season"):
        for _, group in frame.groupby(column):
            champion = np.mean(np.abs(group.PRIMARY_ML - group.actual_kw))
            candidate = np.mean(np.abs(group[challenger] - group.actual_kw))
            values.append((candidate - champion) / champion)
    return float(max(values))


def run(output: str | Path = OUTPUT) -> dict[str, object]:
    started = time.perf_counter()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if not FEATURES.exists():
        raise ValueError("Missing issuance-safe feature cache; run the frozen H5 feature builder first")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    features = list(config["exact_features"]["PRIMARY"])
    frame = pd.read_parquet(FEATURES)
    frame["season"] = _season(frame.month)

    development_rows: list[pd.DataFrame] = []
    fold_metadata: dict[str, object] = {}
    for fold, train_years, evaluation_year, calibration_days in (
        ("FOLD_A", (2020,), 2021, 60), ("FOLD_B", (2020, 2021), 2022, 90),
    ):
        selected = _join_champion(frame, (evaluation_year,))
        regime, regime_meta = _regime_prediction(frame, features, train_years, evaluation_year, calibration_days)
        selected["REGIME_MIXTURE"] = regime
        if fold == "FOLD_A":
            quantiles, quantile_meta = _quantile_prediction(frame, features, (2020,), 2020, 2021)
        else:
            quantiles, quantile_meta = _quantile_prediction(frame, features, (2020,), 2021, 2022)
        for index, value in enumerate(QUANTILES):
            selected[f"Q{int(value * 100):02d}"] = quantiles[:, index]
        selected["QUANTILE_MEDIAN"] = quantiles[:, QUANTILES.index(.5)]
        selected["fold"] = fold
        development_rows.append(selected)
        fold_metadata[fold] = {"regime": regime_meta, "quantile": {key: value for key, value in quantile_meta.items() if key != "raw_predictions"}}
    development = pd.concat(development_rows, ignore_index=True)
    development.to_parquet(output / "development_predictions.parquet", index=False)
    candidates = ("REGIME_MIXTURE", "QUANTILE_MEDIAN")
    development_summary = []
    for candidate in candidates:
        metrics = evaluate_point_forecast(development.actual_kw.to_numpy(), development[candidate].to_numpy())
        development_summary.append({"candidate": candidate, **metrics})
    development_summary_frame = pd.DataFrame(development_summary).sort_values(["mae_kw", "candidate"])
    development_summary_frame.to_csv(output / "development_summary.csv", index=False, float_format="%.10g", lineterminator="\n")
    selected_candidate = str(development_summary_frame.iloc[0].candidate)

    test = _join_champion(frame, (2023,))
    regime_test, regime_meta = _regime_prediction(frame, features, (2020, 2021, 2022), 2023, 90)
    test["REGIME_MIXTURE"] = regime_test
    quantile_test, quantile_meta = _quantile_prediction(frame, features, (2020, 2021), 2022, 2023)
    for index, value in enumerate(QUANTILES):
        test[f"Q{int(value * 100):02d}"] = quantile_test[:, index]
    test["QUANTILE_MEDIAN"] = quantile_test[:, QUANTILES.index(.5)]
    test.to_parquet(output / "test_predictions_2023_exploratory.parquet", index=False)

    test_summaries = []
    reports: dict[str, object] = {}
    raw_quantiles = np.asarray(quantile_meta.pop("raw_predictions"))
    raw_wis = weighted_interval_score(test.actual_kw.to_numpy(), raw_quantiles, QUANTILES)
    calibrated_wis = weighted_interval_score(test.actual_kw.to_numpy(), quantile_test, QUANTILES)
    interval = interval_reliability(test.actual_kw.to_numpy(), test.Q05.to_numpy(), test.Q95.to_numpy(), .9)
    for candidate in candidates:
        metrics = evaluate_point_forecast(test.actual_kw.to_numpy(), test[candidate].to_numpy())
        metrics["evaluation_exposed"] = True
        bootstrap = paired_day_bootstrap(test, candidate, "PRIMARY_ML")
        degradation = _maximum_slice_degradation(test, candidate)
        candidate_interval = interval if candidate == "QUANTILE_MEDIAN" else None
        wis_improvement = (raw_wis - calibrated_wis) / raw_wis if candidate == "QUANTILE_MEDIAN" else None
        reports[candidate] = promotion_report(
            metrics, bootstrap, gates=PromotionGates(), interval_metrics=candidate_interval,
            wis_improvement=wis_improvement, maximum_slice_degradation=degradation,
        )
        test_summaries.append({"candidate": candidate, **metrics, "maximum_slice_degradation": degradation})
    pd.DataFrame(test_summaries).to_csv(output / "test_summary_exploratory.csv", index=False, float_format="%.10g", lineterminator="\n")
    result: dict[str, object] = {
        "schema_version": "prescriptive-ml.experiment.v1",
        "selection_period": "2021-2022 development only",
        "selected_candidate": selected_candidate,
        "test_period": "2023 exposed exploratory diagnostic",
        "quantiles": QUANTILES,
        "development_summary": development_summary,
        "fold_metadata": fold_metadata,
        "test_metadata": {"regime": regime_meta, "quantile": quantile_meta},
        "probabilistic_metrics": {"raw_wis": raw_wis, "calibrated_wis": calibrated_wis, "interval_90": interval},
        "promotion_reports": reports,
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": "EXPLORATORY_EXPOSED_2023_HOLDOUT_NO_CONFIRMATORY_PROMOTION",
    }
    write_json(output / "experiment_summary.json", result)
    return result
