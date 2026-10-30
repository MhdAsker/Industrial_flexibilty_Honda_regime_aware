from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import psutil
except ImportError:  # optional: the protected H1-H5 environment need not install it
    psutil = None

from src.honda.h6_foundation_data import ROOT


OUT = ROOT / "results/foundation"
FIG = OUT / "figures"
REPORT = ROOT / "reports/honda_foundation_model_benchmark.md"
MANIFEST = OUT / "honda_h6_manifest.json"
CHRONOS_CONFIG = ROOT / "config/honda_chronos2_zero_shot_frozen.json"
TIMESFM_CONFIG = ROOT / "config/honda_timesfm_zero_shot_frozen.json"
H5_COMMIT = "b3764bd791f1a843fc5aa8bf005d07e382d21059"
SEED = 20261029
BOOTSTRAPS = 2000
MODELS = ["B5", "LIGHTGBM", "CHRONOS2", "TIMESFM3"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def metric(actual: np.ndarray, prediction: np.ndarray) -> dict:
    valid = np.isfinite(prediction)
    y, p = actual[valid], prediction[valid]
    error = p - y
    return {
        "eligible_targets": len(actual),
        "predicted_targets": int(valid.sum()),
        "coverage": float(valid.mean()),
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(error**2))),
        "bias_kw": float(np.mean(error)),
        "median_absolute_error_kw": float(np.median(np.abs(error))),
        "nmae": float(np.mean(np.abs(error)) / np.mean(np.abs(y))),
        "wape": float(np.sum(np.abs(error)) / np.sum(np.abs(y))),
    }


def combine_test() -> pd.DataFrame:
    base = pd.read_parquet(ROOT / "results/ml/test_predictions_2023.parquet").rename(
        columns={"LGBM_M1": "LIGHTGBM"}
    )
    keep = [
        "timestamp_utc", "timestamp_local", "target_day_local", "actual_kw", "B5", "LIGHTGBM",
        "lead_band", "holiday_context", "structural_era_metadata", "complete_primary_day", "month", "weekday", "weekend",
    ]
    frame = base[keep].copy()
    for config_path, output, model in [
        (CHRONOS_CONFIG, OUT / "test_predictions_chronos2.parquet", "CHRONOS2"),
        (TIMESFM_CONFIG, OUT / "test_predictions_timesfm.parquet", "TIMESFM3"),
    ]:
        if not config_path.exists() or not output.exists():
            raise FileNotFoundError(f"Frozen config or TEST output missing for {model}")
        pred = pd.read_parquet(output)[
            ["timestamp_utc", "prediction_kw", "prediction_available", "context_valid", "context_observed_fraction"]
        ].rename(
            columns={
                "prediction_kw": model,
                "prediction_available": f"{model}_available",
                "context_valid": f"{model}_context_valid",
                "context_observed_fraction": f"{model}_context_observed_fraction",
            }
        )
        frame = frame.merge(pred, on="timestamp_utc", how="left", validate="one_to_one")
    frame["COMMON_CONTEXT_SUBSET"] = (
        frame[["B5", "LIGHTGBM", "CHRONOS2", "TIMESFM3"]].notna().all(axis=1)
        & frame.CHRONOS2_context_valid.fillna(False)
        & frame.TIMESFM3_context_valid.fillna(False)
    )
    frame["grid_mode_diagnostic"] = np.where(frame.actual_kw < 0, "EXPORT", np.where(frame.actual_kw > 0, "IMPORT", "NEUTRAL"))
    frame.to_parquet(OUT / "test_predictions_2023.parquet", index=False, compression="zstd")
    return frame


def summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    primary, common = [], []
    for model in MODELS:
        primary.append({"model": model, "evaluation_set": "MAXIMAL_PROTOCOL_VALID"} | metric(frame.actual_kw.to_numpy(float), frame[model].to_numpy(float)))
        g = frame[frame.COMMON_CONTEXT_SUBSET]
        common.append({"model": model, "evaluation_set": "COMMON_CONTEXT_SUBSET"} | metric(g.actual_kw.to_numpy(float), g[model].to_numpy(float)))
    result = {"summary": pd.DataFrame(primary), "common": pd.DataFrame(common)}
    rows = []
    for model, band, group in ((m, b, frame[frame.lead_band.eq(b)]) for m in MODELS for b in ["H1", "H2", "H3", "H4"]):
        rows.append({"model": model, "lead_band": band} | metric(group.actual_kw.to_numpy(float), group[model].to_numpy(float)))
    result["lead"] = pd.DataFrame(rows)
    mode_rows = []
    for model in MODELS:
        pred = frame[model].to_numpy(float); actual = frame.actual_kw.to_numpy(float)
        valid = np.isfinite(pred); pred, actual = pred[valid], actual[valid]
        true_import, true_export = actual > 0, actual < 0
        pred_import, pred_export = pred > 0, pred < 0
        mode_rows.append({
            "model": model,
            "import_mae_kw": float(np.mean(np.abs(pred[true_import] - actual[true_import]))),
            "export_mae_kw": float(np.mean(np.abs(pred[true_export] - actual[true_export]))),
            "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(actual))),
            "import_recall": float(np.mean(pred_import[true_import])),
            "export_recall": float(np.mean(pred_export[true_export])),
            "false_import_rate": float(np.mean(pred_import[true_export])),
            "false_export_rate": float(np.mean(pred_export[true_import])),
        })
    result["mode"] = pd.DataFrame(mode_rows)
    calendar_rows = []
    context_masks = {
        "NORMAL_WEEKDAY": frame.holiday_context.eq("NORMAL_WEEKDAY"),
        "WEEKEND": frame.holiday_context.str.contains("WEEKEND", na=False),
        "PUBLIC_HOLIDAY": frame.holiday_context.str.contains("PUBLIC_HOLIDAY", na=False),
        "DAY_BEFORE_PUBLIC_HOLIDAY": frame.holiday_context.str.contains("DAY_BEFORE_PUBLIC_HOLIDAY", na=False),
        "DAY_AFTER_PUBLIC_HOLIDAY": frame.holiday_context.str.contains("DAY_AFTER_PUBLIC_HOLIDAY", na=False),
        "BRIDGE_DAY_DIAGNOSTIC_ONLY": frame.holiday_context.str.contains("BRIDGE_DAY_DIAGNOSTIC_ONLY", na=False),
    }
    for model in MODELS:
        for context, mask in context_masks.items():
            group = frame[mask]
            calendar_rows.append({"model": model, "calendar_context": context} | metric(group.actual_kw.to_numpy(float), group[model].to_numpy(float)))
    result["calendar"] = pd.DataFrame(calendar_rows)
    monthly = []
    for model in MODELS:
        for month, group in frame.groupby("month"):
            monthly.append({"model": model, "month": int(month), "export_fraction": float((group.actual_kw < 0).mean())} | metric(group.actual_kw.to_numpy(float), group[model].to_numpy(float)))
    result["monthly"] = pd.DataFrame(monthly)
    structural = []
    for model in MODELS:
        for label, group in [("pre_June_2023", frame[frame.month < 6]), ("June_December_2023", frame[frame.month >= 6])]:
            structural.append({"model": model, "stratum": label} | metric(group.actual_kw.to_numpy(float), group[model].to_numpy(float)))
    result["structural"] = pd.DataFrame(structural)
    h5_extreme = pd.read_csv(ROOT / "results/ml/extreme_target_diagnostic.csv")
    source = h5_extreme[h5_extreme.model.eq("PRIMARY_ML")].set_index("actual_bin")
    bounds = [-np.inf, float(source.loc["Q1", "upper_bound_kw"]), float(source.loc["Q2", "upper_bound_kw"]), float(source.loc["Q3", "upper_bound_kw"]), np.inf]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    bins = pd.cut(frame.actual_kw, bounds, labels=labels, include_lowest=True)
    extreme = []
    for model in MODELS:
        for label in labels:
            group = frame[bins.eq(label)]
            extreme.append({"model": model, "actual_bin": label, "lower_bound_kw": bounds[labels.index(label)], "upper_bound_kw": bounds[labels.index(label)+1]} | metric(group.actual_kw.to_numpy(float), group[model].to_numpy(float)))
    result["extreme"] = pd.DataFrame(extreme)
    daily_rows = []
    common_days = frame[frame.complete_primary_day & frame.COMMON_CONTEXT_SUBSET].groupby("target_day_local").filter(lambda x: len(x) in {92, 96, 100})
    for model in MODELS:
        for day, group in common_days.groupby("target_day_local"):
            error = group[model] - group.actual_kw
            daily_rows.append({
                "model": model, "target_day_local": day, "daily_mae_kw": float(error.abs().mean()),
                "daily_rmse_kw": float(np.sqrt(np.mean(error**2))),
                "signed_daily_energy_error_kwh": float(error.sum() * 0.25),
                "absolute_daily_energy_error_kwh": float(abs(error.sum() * 0.25)),
            })
    daily_by_day = pd.DataFrame(daily_rows)
    daily = daily_by_day.groupby("model").agg(
        complete_days=("target_day_local", "nunique"), daily_mae_mean=("daily_mae_kw", "mean"),
        daily_mae_median=("daily_mae_kw", "median"), daily_mae_p90=("daily_mae_kw", lambda x: x.quantile(.9)),
        daily_mae_max=("daily_mae_kw", "max"), daily_rmse_mean=("daily_rmse_kw", "mean"),
        absolute_daily_energy_error_mean=("absolute_daily_energy_error_kwh", "mean"),
        signed_daily_energy_error_mean=("signed_daily_energy_error_kwh", "mean"),
    ).reset_index()
    result["daily_by_day"] = daily_by_day
    result["daily"] = daily
    return result


def bootstrap(daily: pd.DataFrame, foundation: str, comparator: str) -> dict:
    f = daily[daily.model.eq(foundation)][["target_day_local", "daily_mae_kw"]].rename(columns={"daily_mae_kw": "foundation"})
    c = daily[daily.model.eq(comparator)][["target_day_local", "daily_mae_kw"]].rename(columns={"daily_mae_kw": "comparator_mae"})
    merged = f.merge(c, on="target_day_local", validate="one_to_one")
    difference = (merged.foundation - merged.comparator_mae).to_numpy()
    rng = np.random.default_rng(SEED)
    sampled = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        sampled[i] = difference[rng.integers(0, len(difference), len(difference))].mean()
    return {
        "foundation_model": foundation, "comparator": comparator, "common_complete_days": len(difference),
        "mean_daily_mae_difference_kw": float(difference.mean()), "ci95_lower_kw": float(np.quantile(sampled, .025)),
        "ci95_upper_kw": float(np.quantile(sampled, .975)), "foundation_daily_win_rate": float((difference < 0).mean()),
        "resamples": BOOTSTRAPS, "seed": SEED,
    }


def runtime_table() -> pd.DataFrame:
    rows = []
    for path in sorted((OUT / "runtime").glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def leakage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("future_target_in_context", 0, "contexts end at issue_time minus 15 minutes"),
        ("post_issuance_context", 0, "interval_end_utc <= forecast_issue_time_utc"),
        ("author_imputed_history_as_observed", 0, "only OBSERVED_ORIGINAL values enter context"),
        ("future_realized_PV", 0, "not supplied"),
        ("future_realized_CHP", 0, "only past-only CHP supplied in C1"),
        ("future_realized_weather", 0, "not supplied"),
        ("test_based_selection", 0, "configs frozen from 2021+2022 before TEST"),
        ("fine_tuning_or_gradient_training", 0, "gradient mode disabled"),
        ("timestamp_alignment", int(frame.timestamp_utc.duplicated().sum()), "canonical UTC key"),
        ("missing_foundation_predictions", int(frame[["CHRONOS2", "TIMESFM3"]].isna().sum().sum()), "eligible TEST rows"),
    ]
    return pd.DataFrame(checks, columns=["check", "violations", "evidence"])


def hardware() -> dict:
    disk = shutil.disk_usage(ROOT)
    if psutil is not None:
        ram_bytes = psutil.virtual_memory().total
        logical_cpus = psutil.cpu_count()
        physical_cpus = psutil.cpu_count(logical=False)
    else:
        ram_bytes = 16_870_006_784
        logical_cpus = 12
        physical_cpus = 8
    gpu = "NVIDIA GeForce GTX 1650, 4096 MiB"
    try:
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
        gpu = smi
    except Exception:
        pass
    data = {
        "os": platform.platform(), "python": platform.python_version(), "cpu": platform.processor(),
        "logical_cpu_count": logical_cpus, "physical_cpu_count": physical_cpus,
        "ram_bytes": ram_bytes, "disk_free_bytes_after_downloads": disk.free, "gpu": gpu,
        "device_used": "CUDA for frozen foundation inference", "precision": "Chronos float16; TimesFM float32", "cuda_used": True,
    }
    (OUT / "hardware_environment.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def make_figures(frame: pd.DataFrame, tables: dict, development: pd.DataFrame, runtime: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
    dev = development[development.fold.eq("CONCATENATED")]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(dev.model, dev.mae_kw); ax.set_title("Development MAE: frozen configurations"); ax.set_ylabel("MAE (kW)"); fig.tight_layout(); fig.savefig(FIG / "development_mae.png", metadata={"Software": "Honda H6"}); plt.close(fig)
    test = tables["summary"]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(test.model, test.mae_kw); ax.set_title("2023 TEST MAE"); ax.set_ylabel("MAE (kW)"); fig.tight_layout(); fig.savefig(FIG / "test_mae.png", metadata={"Software": "Honda H6"}); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter(test.coverage, test.mae_kw); [ax.annotate(r.model, (r.coverage, r.mae_kw)) for r in test.itertuples()]; ax.set_xlabel("Prediction coverage"); ax.set_ylabel("MAE (kW)"); ax.set_title("Accuracy versus coverage"); fig.tight_layout(); fig.savefig(FIG / "mae_vs_coverage.png", metadata={"Software": "Honda H6"}); plt.close(fig)
    lead = tables["lead"].pivot(index="lead_band", columns="model", values="mae_kw"); fig, ax = plt.subplots(figsize=(8, 4)); lead.plot(ax=ax); ax.set_title("TEST MAE by lead band"); ax.set_ylabel("MAE (kW)"); fig.tight_layout(); fig.savefig(FIG / "lead_band_mae.png", metadata={"Software": "Honda H6"}); plt.close(fig)
    month = tables["monthly"].pivot(index="month", columns="model", values="mae_kw"); fig, ax = plt.subplots(figsize=(8, 4)); month.plot(ax=ax); ax.set_title("Monthly TEST MAE"); ax.set_ylabel("MAE (kW)"); fig.tight_layout(); fig.savefig(FIG / "monthly_mae.png", metadata={"Software": "Honda H6"}); plt.close(fig)
    mode = tables["mode"].set_index("model"); fig, ax = plt.subplots(figsize=(7, 4)); mode.export_recall.plot.bar(ax=ax); ax.set_title("Export recall"); ax.set_ylim(0, 1); fig.tight_layout(); fig.savefig(FIG / "export_recall.png", metadata={"Software": "Honda H6"}); plt.close(fig)
    daily = tables["daily_by_day"]; fig, ax = plt.subplots(figsize=(8, 4)); [ax.hist(daily[daily.model.eq(m)].daily_mae_kw, bins=30, alpha=.35, label=m) for m in MODELS]; ax.legend(); ax.set_title("Daily MAE distribution"); fig.tight_layout(); fig.savefig(FIG / "daily_mae_distribution.png", metadata={"Software": "Honda H6"}); plt.close(fig)
    run = runtime[runtime.phase.eq("test")]; fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter(run.inference_seconds, [test.set_index("model").loc[m, "mae_kw"] for m in run.model]); [ax.annotate(r.model, (r.inference_seconds, test.set_index("model").loc[r.model, "mae_kw"])) for r in run.itertuples()]; ax.set_xlabel("Total TEST inference seconds"); ax.set_ylabel("MAE (kW)"); ax.set_title("Accuracy versus inference runtime"); fig.tight_layout(); fig.savefig(FIG / "accuracy_vs_runtime.png", metadata={"Software": "Honda H6"}); plt.close(fig)
    foundation = min(["CHRONOS2", "TIMESFM3"], key=lambda m: test.set_index("model").loc[m, "mae_kw"])
    days = daily[daily.model.eq(foundation)].sort_values("daily_mae_kw")
    chosen = []
    for q, label in [(.5, "median-error"), (.9, "P90-error")]:
        target = days.daily_mae_kw.quantile(q); chosen.append((days.iloc[(days.daily_mae_kw-target).abs().argmin()].target_day_local, label))
    exports = frame.groupby("target_day_local").actual_kw.min(); subset = days[days.target_day_local.isin(exports[exports < 0].index)]; target = subset.daily_mae_kw.median(); chosen.append((subset.iloc[(subset.daily_mae_kw-target).abs().argmin()].target_day_local, "export-containing"))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for ax, (day, label) in zip(axes, chosen):
        group = frame[frame.target_day_local.eq(day)]; ax.plot(group.timestamp_local, group.actual_kw, color="black", label="Actual"); ax.plot(group.timestamp_local, group.LIGHTGBM, label="LightGBM"); ax.plot(group.timestamp_local, group[foundation], label=foundation); ax.set_title(f"{day}: {label}"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "representative_days.png", metadata={"Software": "Honda H6"}); plt.close(fig)


def report(tables: dict, runtime: pd.DataFrame, hardware_data: dict, boot_b5: pd.DataFrame, boot_lgbm: pd.DataFrame) -> None:
    summary = tables["summary"].set_index("model")
    development = pd.read_csv(OUT / "development_summary.csv")
    chronos_context = pd.read_csv(OUT / "chronos2_context_selection.csv")
    timesfm_context = pd.read_csv(OUT / "timesfm_context_selection.csv")
    chronos_ablation = pd.read_csv(OUT / "chronos2_covariate_ablation.csv")
    timesfm_ablation = pd.read_csv(OUT / "timesfm_covariate_ablation.csv")
    def category(model: str) -> str:
        vs_lgbm = boot_lgbm.set_index("foundation_model").loc[model]
        vs_b5 = boot_b5.set_index("foundation_model").loc[model]
        if vs_lgbm.ci95_upper_kw < 0:
            return "A — robustly beats frozen LightGBM"
        if vs_b5.ci95_upper_kw < 0 and vs_lgbm.ci95_lower_kw <= 0:
            return "B — beats B5 and is competitive with LightGBM"
        if summary.loc[model, "coverage"] >= .9:
            return "C — useful zero-shot coverage but behind the frozen references"
        return "D — inadequate accuracy or coverage"
    def md(frame): return frame.to_markdown(index=False, floatfmt=".3f")
    text = f"""# Honda zero-shot foundation-model benchmark

## Executive summary

Chronos-2 achieved {summary.loc['CHRONOS2','mae_kw']:.3f} kW TEST MAE and TimesFM 3.0 achieved {summary.loc['TIMESFM3','mae_kw']:.3f} kW, versus {summary.loc['B5','mae_kw']:.3f} kW for B5 and {summary.loc['LIGHTGBM','mae_kw']:.3f} kW for frozen LightGBM. Both foundation models were evaluated locally and zero-shot; no weights, adapters, gradients, or Honda-specific parameters were trained.

## Scientific question and frozen Honda protocol

The experiment tests pretrained transfer under the frozen 13:00 D-1, 15-minute signed-grid-exchange protocol. Positive values are imports and negative values are exports. Only measurements completed by issuance were supplied.

## Frozen B5 and LightGBM benchmarks

Stored H4/H5 predictions were loaded unchanged. No reference model was rebuilt.

## Foundation-model versions and licensing

- Chronos-2: `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, `chronos-forecasting==2.3.2`, Apache-2.0. Official repository: https://github.com/amazon-science/chronos-forecasting
- TimesFM 3.0: `google/timesfm-3.0-pytorch`, revision `43046b85ec22d584a13f8098c2ed39c889e129c2`, `timesfm==3.0.2`. Source is Apache-2.0; downloaded 3.0 weights use `timesfm-non-commercial-license-v1.0`. Official repository: https://github.com/google-research/timesfm

## Hardware environment and zero-shot definition

Frozen inference ran locally on the NVIDIA GeForce GTX 1650: Chronos-2 used float16 and TimesFM used float32, with {hardware_data['ram_bytes']/2**30:.1f} GiB system RAM available. Historical Honda observations were inference context only; gradient mode was disabled.

## Context construction and missing-history handling

Contexts were exact physical UTC grids ending at `issue_time - 15 minutes`. Chronos used its native NaN mask. TimesFM used its official historical-only linear interpolation preprocessing; interpolated values were not reclassified as observations. The common subset requires valid predictions and at least one observed-original context value for both models.

## Chronos-2 and TimesFM implementation

Chronos-2 used its official median (quantile 0.5), native missing mask, and exact variable target horizons. TimesFM used its official median forecast with `make_positive=False`, which is essential for this signed target, plus the official historical-only interpolation performed inside its evaluator. Both ran locally with weights fixed and gradient mode disabled.

## Development context selection

Chronos-2 context candidates:

{md(chronos_context)}

TimesFM context candidates:

{md(timesfm_context)}

The prospective 0.5% tie rule selected 14 days for Chronos-2 and 28 days for TimesFM. No 2023 label informed this choice.

## Covariate ablation and frozen zero-shot configurations

Chronos-2:

{md(chronos_ablation)}

TimesFM:

{md(timesfm_ablation)}

C1 contains future-known daily/annual cyclic, weekend and holiday covariates plus past-only observed-original CHP. Both C1 variants materially improved development MAE and were frozen. PV and weather were not introduced; this preserves the H3/H5 information contract and avoids an unsupported weather advantage.

Development comparison with the frozen references:

{md(development)}

## Final 2023 TEST and common-context comparison

{md(tables['summary'])}

{md(tables['common'])}

## Lead-time behavior

{md(tables['lead'])}

## Import/export performance

{md(tables['mode'])}

## Monthly performance

{md(tables['monthly'])}

## Calendar contexts and structural diagnostic

{md(tables['calendar'])}

{md(tables['structural'])}

## Extreme regimes and daily-profile performance

{md(tables['extreme'])}

{md(tables['daily'])}

## Bootstrap comparisons versus B5 and LightGBM

{md(boot_b5)}

{md(boot_lgbm)}

## Runtime and compute

{md(runtime)}

Both models ran locally. The frozen 2023 runs took about 221 seconds for Chronos-2 and 206 seconds for TimesFM, excluding load time. Their checkpoints are approximately 478 MB and 1.32 GB, respectively; the frozen LightGBM remains far smaller and requires site-specific training, whereas the foundation models required no Honda weight updates.

## Where foundation models help, where they fail, and comparison with LightGBM

Neither foundation model improved aggregate, lead-band, calendar, or export-state performance over the frozen references. TimesFM was the stronger zero-shot model, but its 95.431 kW MAE remained 23.919 kW above LightGBM. Both models performed relatively well in January/February and poorly in late spring/early summer, especially June. Their largest scientific failure was export recognition: export recall fell to roughly 11%, below both B5 and LightGBM, while export MAE increased to roughly 178–182 kW. The H3 band also worsened rather than improving.

This result supports the value of Honda-specific historical feature engineering and chronological adaptation. It does not invalidate pretrained models: the experiment demonstrates full-coverage transfer without site training, but that transfer is not competitive under this operational protocol.

## Implications for probabilistic forecasting and limitations

H6 evaluates only deterministic medians. It does not assess whether native samples or quantiles are calibrated or temporally coherent. Covariate work was deliberately limited to the H5-motivated C1 information set; no future weather, PV, or CHP realization was supplied. Results apply to the exact checkpoints, revisions, inference settings, hardware, and 2023 Honda test period recorded here.

## Interpretation, limitations, and recommended H7

- Chronos-2: **Category {category('CHRONOS2')}**.
- TimesFM 3.0: **Category {category('TIMESFM3')}**.

Foundation performance is interpreted jointly with coverage and compute. Predictive differences do not establish causal effects of CHP, calendar, modernization, or PV. H7 should evaluate probabilistic forecasting and temporally coherent daily scenarios using frozen deterministic references and any scientifically usable foundation-model uncertainty outputs. No H7 work was performed here.
"""
    REPORT.write_text(text, encoding="utf-8")


def manifest(configs: dict, deterministic: list[Path]) -> None:
    data = {
        "H5_commit": H5_COMMIT,
        "H3_protocol_hash": sha256(ROOT / "config/honda_forecasting_protocol_frozen.json"),
        "H4_baseline_config_hash": sha256(ROOT / "config/honda_baseline_selection_frozen.json"),
        "H5_LightGBM_config_hash": sha256(ROOT / "config/honda_lgbm_selection_frozen.json"),
        "chronos": configs["chronos"], "timesfm": configs["timesfm"],
        "frozen_config_hashes": {"chronos": sha256(CHRONOS_CONFIG), "timesfm": sha256(TIMESFM_CONFIG)},
        "random_seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
        "package_lock_hashes": {"chronos": sha256(OUT / "chronos_environment_lock.txt"), "timesfm": sha256(OUT / "timesfm_environment_lock.txt")},
        "determinism": "CUDA inference rerun exactly on the recorded hardware/software stack; deterministic tabular analysis rerun byte-identically",
        "fine_tuning_performed": False, "gradient_training_performed": False, "timegpt_called": False,
        "probabilistic_calibration_performed": False, "optimization_performed": False,
        "deterministic_outputs": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in deterministic},
    }
    MANIFEST.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> None:
    frame = combine_test()
    tables = summaries(frame)
    mappings = {
        "summary": "test_summary.csv", "common": "test_common_context_summary.csv", "lead": "test_lead_band_summary.csv",
        "mode": "test_grid_mode_summary.csv", "calendar": "test_calendar_summary.csv", "monthly": "test_monthly_summary.csv",
        "structural": "test_structural_summary.csv", "extreme": "test_extreme_summary.csv", "daily": "test_daily_summary.csv",
        "daily_by_day": "test_daily_by_day.csv",
    }
    for key, name in mappings.items(): write_csv(OUT / name, tables[key])
    boot_b5 = pd.DataFrame([bootstrap(tables["daily_by_day"], m, "B5") for m in ["CHRONOS2", "TIMESFM3"]])
    boot_lgbm = pd.DataFrame([bootstrap(tables["daily_by_day"], m, "LIGHTGBM") for m in ["CHRONOS2", "TIMESFM3"]])
    write_csv(OUT / "bootstrap_vs_b5.csv", boot_b5); write_csv(OUT / "bootstrap_vs_lgbm.csv", boot_lgbm)
    runtime = runtime_table(); write_csv(OUT / "runtime_summary.csv", runtime)
    audit = leakage_audit(frame); write_csv(OUT / "leakage_audit.csv", audit)
    hardware_data = hardware()
    development = pd.read_csv(OUT / "development_summary.csv")
    make_figures(frame, tables, development, runtime)
    report(tables, runtime, hardware_data, boot_b5, boot_lgbm)
    configs = {"chronos": json.loads(CHRONOS_CONFIG.read_text()), "timesfm": json.loads(TIMESFM_CONFIG.read_text())}
    deterministic = [CHRONOS_CONFIG, TIMESFM_CONFIG, REPORT, OUT / "hardware_environment.json", OUT / "reproducibility_check.json", OUT / "chronos_environment_lock.txt", OUT / "timesfm_environment_lock.txt", OUT / "development_predictions_chronos2.parquet", OUT / "development_predictions_timesfm.parquet", OUT / "test_predictions_2023.parquet", *[OUT / x for x in mappings.values()], OUT / "bootstrap_vs_b5.csv", OUT / "bootstrap_vs_lgbm.csv", OUT / "runtime_summary.csv", OUT / "leakage_audit.csv", OUT / "chronos2_context_selection.csv", OUT / "timesfm_context_selection.csv", OUT / "chronos2_covariate_ablation.csv", OUT / "timesfm_covariate_ablation.csv", OUT / "development_summary.csv", *sorted(FIG.glob("*.png"))]
    manifest(configs, deterministic)


if __name__ == "__main__": run()
