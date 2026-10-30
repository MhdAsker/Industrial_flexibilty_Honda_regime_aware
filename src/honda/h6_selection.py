from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.honda.h6_foundation_data import ROOT


OUT = ROOT / "results/foundation"
CHRONOS_CONFIG = ROOT / "config/honda_chronos2_zero_shot_frozen.json"
TIMESFM_CONFIG = ROOT / "config/honda_timesfm_zero_shot_frozen.json"
H5_COMMIT = "b3764bd791f1a843fc5aa8bf005d07e382d21059"
H3_HASH = "281f49331bc3b3c31aec0d33b50f6a83a149261db93ce04a29d67479cb62676e"
H4_HASH = "1bbbbac404675bb1bfb5c85067e1504fda843fc2bc52151a60025f4e3c8b7a04"
H5_HASH = "7fff8efbf51fc17ac2524ceeac30dbed6764d4ec55eaf1ea3993c47691dd21ea"
MODELS = {
    "CHRONOS2": {
        "slug": "chronos2",
        "model_id": "amazon/chronos-2",
        "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "package": "chronos-forecasting",
        "version": "2.3.2",
        "license": "Apache-2.0",
        "checkpoint": ROOT / "artifacts/foundation_models/chronos-2/model.safetensors",
        "config": CHRONOS_CONFIG,
        "missing": "native NaN mask",
        "point": "quantile 0.5 median",
    },
    "TIMESFM3": {
        "slug": "timesfm",
        "model_id": "google/timesfm-3.0-pytorch",
        "revision": "43046b85ec22d584a13f8098c2ed39c889e129c2",
        "package": "timesfm",
        "version": "3.0.2",
        "license": "timesfm-non-commercial-license-v1.0",
        "checkpoint": ROOT / "artifacts/foundation_models/timesfm-3.0-pytorch/model.safetensors",
        "config": TIMESFM_CONFIG,
        "missing": "official historical-only linear interpolation within pre-issuance context",
        "point": "official median forecast",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference() -> pd.DataFrame:
    return pd.read_parquet(OUT.parent / "ml/development_predictions.parquet")[
        ["timestamp_utc", "target_day_local", "actual_kw", "fold", "B5", "LGBM_M1"]
    ].rename(columns={"LGBM_M1": "LIGHTGBM"})


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    valid = np.isfinite(prediction)
    y = actual[valid]
    p = prediction[valid]
    error = p - y
    return {
        "eligible_targets": len(actual),
        "predicted_targets": int(valid.sum()),
        "coverage": float(valid.mean()),
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(error**2))),
        "bias_kw": float(np.mean(error)),
    }


def evaluate(path: Path, model: str, mode: str, context_days: int) -> list[dict]:
    pred = pd.read_parquet(path)[["timestamp_utc", "prediction_kw", "context_valid", "context_observed_fraction"]]
    frame = reference().merge(pred, on="timestamp_utc", how="left", validate="one_to_one")
    rows = []
    for fold, group in list(frame.groupby("fold", sort=True)) + [("CONCATENATED", frame)]:
        row = metrics(group.actual_kw.to_numpy(float), group.prediction_kw.to_numpy(float))
        row.update(
            {
                "model": model,
                "mode": mode,
                "context_days": context_days,
                "fold": fold,
                "mean_context_observed_fraction": float(group.context_observed_fraction.mean()),
                "valid_context_origins": int(group.loc[group.context_valid.fillna(False), "target_day_local"].nunique()),
            }
        )
        rows.append(row)
    return rows


def select_contexts() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model, meta in MODELS.items():
        rows = []
        for days in [7, 14, 28]:
            path = OUT / f"development_{meta['slug']}_u_{days}d.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.extend(evaluate(path, model, "U", days))
        table = pd.DataFrame(rows)
        table.to_csv(OUT / f"{meta['slug']}_context_selection.csv", index=False, lineterminator="\n", float_format="%.10g")


def selected_context(model: str) -> int:
    meta = MODELS[model]
    table = pd.read_csv(OUT / f"{meta['slug']}_context_selection.csv")
    candidates = table[table.fold.eq("CONCATENATED")].sort_values(["mae_kw", "context_days"])
    best = candidates.iloc[0]
    simpler = candidates[candidates.context_days < best.context_days]
    if not simpler.empty:
        relative_gain = (simpler.mae_kw - best.mae_kw) / simpler.mae_kw
        simplest_near = simpler[relative_gain < 0.005]
        if not simplest_near.empty:
            return int(simplest_near.sort_values("context_days").iloc[0].context_days)
    return int(best.context_days)


def freeze() -> None:
    for model, meta in MODELS.items():
        days = selected_context(model)
        matched_u = OUT / f"development_{meta['slug']}_u_gpu_{days}d.parquet"
        univariate_path = matched_u if matched_u.exists() else OUT / f"development_{meta['slug']}_u_{days}d.parquet"
        rows = evaluate(univariate_path, model, "U", days)
        c1 = OUT / f"development_{meta['slug']}_c1_{days}d.parquet"
        if c1.exists():
            rows.extend(evaluate(c1, model, "C1", days))
        table = pd.DataFrame(rows)
        table.to_csv(OUT / f"{meta['slug']}_covariate_ablation.csv", index=False, lineterminator="\n", float_format="%.10g")
        concat = table[table.fold.eq("CONCATENATED")].sort_values(["mae_kw", "mode"])
        best = concat.iloc[0]
        univariate = concat[concat["mode"].eq("U")].iloc[0]
        mode = str(best["mode"]) if best.mae_kw < univariate.mae_kw * 0.995 else "U"
        chosen = concat[concat["mode"].eq(mode)].iloc[0]
        config = {
            "selection_status": "FROZEN_BEFORE_TEST",
            "test_used_for_selection": False,
            "H5_commit": H5_COMMIT,
            "H3_protocol_sha256": H3_HASH,
            "H4_baseline_selection_sha256": H4_HASH,
            "H5_LightGBM_config_sha256": H5_HASH,
            "model_id": meta["model_id"],
            "revision": meta["revision"],
            "checkpoint_sha256": sha256(meta["checkpoint"]),
            "checkpoint_bytes": meta["checkpoint"].stat().st_size,
            "package": meta["package"],
            "package_version": meta["version"],
            "license": meta["license"],
            "selected_context_days": days,
            "selected_mode": mode,
            "covariates": [] if mode == "U" else ["future-known calendar", "historical observed-original CHP"],
            "missing_context_policy": meta["missing"],
            "point_forecast_definition": meta["point"],
            "device": "cuda" if mode == "C1" or matched_u.exists() else "cpu",
            "dtype": "float16" if model == "CHRONOS2" and (mode == "C1" or matched_u.exists()) else "float32",
            "random_seed": 20261029,
            "development_selection_metric": "equal-target-weighted concatenated 2021+2022 MAE",
            "tie_rule": "prefer univariate when covariate MAE improvement is below 0.5%",
            "development_metrics": table.to_dict("records"),
            "selected_development_mae_kw": float(chosen.mae_kw),
            "fine_tuning_performed": False,
            "gradient_training_performed": False,
        }
        meta["config"].write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    summaries = []
    ref = reference()
    for fold, group in list(ref.groupby("fold", sort=True)) + [("CONCATENATED", ref)]:
        for model in ["B5", "LIGHTGBM"]:
            summaries.append({"model": model, "mode": "FROZEN_REFERENCE", "context_days": np.nan, "fold": fold} | metrics(group.actual_kw.to_numpy(float), group[model].to_numpy(float)))
    for model, meta in MODELS.items():
        cfg = json.loads(meta["config"].read_text(encoding="utf-8"))
        path = OUT / f"development_{meta['slug']}_{cfg['selected_mode'].lower()}_{cfg['selected_context_days']}d.parquet"
        gpu_u = OUT / f"development_{meta['slug']}_u_gpu_{cfg['selected_context_days']}d.parquet"
        if cfg["selected_mode"] == "U" and gpu_u.exists():
            path = gpu_u
        summaries.extend(evaluate(path, model, cfg["selected_mode"], cfg["selected_context_days"]))
    pd.DataFrame(summaries).to_csv(OUT / "development_summary.csv", index=False, lineterminator="\n", float_format="%.10g")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["select-contexts", "freeze"])
    args = parser.parse_args()
    select_contexts()
    if args.action == "freeze":
        freeze()


if __name__ == "__main__":
    main()
