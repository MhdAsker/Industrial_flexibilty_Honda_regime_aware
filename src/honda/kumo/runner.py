from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:  # optional runtime telemetry
    psutil = None

from src.honda.h6_foundation_data import INDEX, ROOT

from .data_adapter import HondaKumoOrigin, _observed_grid, load_honda_origins, result_frame
from .diagnostics import write_retrieval_diagnostic
from .evaluation import write_evaluation
from .regimes import PreissueThresholds, preissue_features, preissue_regime
from .retrieval import (
    build_retrieval_candidates,
    longest_complete_archive,
    retrieval_forecast,
    select_candidates,
)


EXPERIMENT_CONFIG = ROOT / "config/honda_kumo_experiment.json"
FROZEN_CONFIG = ROOT / "config/honda_kumo_final_frozen.json"
ARTIFACT_DIR = ROOT / "artifacts/kumo"
RESULT_DIR = ROOT / "results/kumo"
MODEL_REPO = "nvidia/Kumo-Forecast"
MODEL_REVISION = "402caa11b7ff585141e912c6743026141f648892"
BACKBONE_REPO = "AutonLab/MOMENT-1-large"
BACKBONE_REVISION = "ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc"
KUMO_TS_COMMIT = "ad459475d1302fc8365e7117e39837a8ecb6a7e3"
SEED = 20261029


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _kumo_api():
    try:
        from sdk.forecasting import ForecastingConfig, perform_forecasting
    except ImportError as exc:
        raise RuntimeError(
            "Kumo-Forecast is optional and unavailable. Create an isolated environment and install "
            f"git+https://github.com/NVIDIA/Kumo-TS.git@{KUMO_TS_COMMIT}#subdirectory=Kumo-Forecast"
        ) from exc
    return ForecastingConfig, perform_forecasting


def prepare_weights() -> dict[str, dict[str, str | int]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface-hub to prepare the pinned Kumo checkpoint") from exc
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    for filename in ["standardizer.pkl", "moment_head_512_6hr.pt"]:
        downloaded = Path(
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                revision=MODEL_REVISION,
                local_dir=ARTIFACT_DIR,
            )
        )
        result[filename] = {
            "path": downloaded.as_posix(),
            "bytes": downloaded.stat().st_size,
            "sha256": _sha256(downloaded),
        }
    backbone_dir = ARTIFACT_DIR / "moment-1-large"
    for filename in ["config.json", "model.safetensors"]:
        downloaded = Path(
            hf_hub_download(
                repo_id=BACKBONE_REPO,
                filename=filename,
                revision=BACKBONE_REVISION,
                local_dir=backbone_dir,
            )
        )
        key = f"moment-1-large/{filename}"
        result[key] = {
            "path": downloaded.as_posix(),
            "bytes": downloaded.stat().st_size,
            "sha256": _sha256(downloaded),
        }
    (ARTIFACT_DIR / "weights_manifest.json").write_text(
        json.dumps(
            {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "backbone_repo": BACKBONE_REPO,
                "backbone_revision": BACKBONE_REVISION,
                "files": result,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _hardware() -> dict[str, Any]:
    gpu = {"available_to_torch": False, "name": None, "memory_total_bytes": None}
    try:
        import torch

        gpu["available_to_torch"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            gpu["name"] = torch.cuda.get_device_name(0)
            gpu["memory_total_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except ImportError:
        pass
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        name, memory = [x.strip() for x in smi.splitlines()[0].split(",")]
        gpu["driver_visible_name"] = name
        gpu["driver_visible_memory_mib"] = int(memory)
    except Exception:
        gpu["driver_visible_name"] = None
    ram_bytes = psutil.virtual_memory().total if psutil is not None else None
    if ram_bytes is None and platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            ram_bytes = int(status.total_physical)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": psutil.cpu_count() if psutil is not None else os.cpu_count(),
        "ram_bytes": ram_bytes,
        "gpu": gpu,
    }


def audit_environment(output: Path) -> dict[str, Any]:
    package_version = None
    try:
        package_version = importlib.metadata.version("kumo-forecast")
    except importlib.metadata.PackageNotFoundError:
        pass
    packages = {}
    for name in [
        "kumo-forecast",
        "torch",
        "pandas",
        "numpy",
        "transformers",
        "huggingface-hub",
        "datasetsforecast",
        "joblib",
        "pyarrow",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    artifacts = {}
    for name, path in {
        "standardizer": ARTIFACT_DIR / "standardizer.pkl",
        "forecast_head": ARTIFACT_DIR / "moment_head_512_6hr.pt",
        "backbone": ARTIFACT_DIR / "moment-1-large/model.safetensors",
    }.items():
        artifacts[name] = (
            {"bytes": path.stat().st_size, "sha256": _sha256(path)} if path.exists() else {"status": "ABSENT"}
        )
    audit = {
        "study_classification": "RETROSPECTIVE_POST_BENCHMARK_EXTENSION",
        "official_repository": "https://github.com/NVIDIA/Kumo-TS",
        "official_commit_audited": KUMO_TS_COMMIT,
        "official_package_version": "1.0.1",
        "installed_package_version": package_version,
        "package_versions": packages,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "backbone_repo": BACKBONE_REPO,
        "backbone_revision": BACKBONE_REVISION,
        "license": "Apache-2.0",
        "verified_api": "sdk.forecasting.perform_forecasting(df, config=ForecastingConfig(...), context_df=None)",
        "verified_native_horizon": 72,
        "verified_max_horizon": 512,
        "verified_sequence_lengths": [256, 512, 1024, 2048],
        "native_regime_filtering": False,
        "native_neighbor_metadata": False,
        "native_missing_policy": "fills numeric NaNs with zero unless separate missingness path is used; DARR and missingness path are incompatible in 1.0.1",
        "local_artifacts": artifacts,
        "random_seed": SEED,
        "hardware": _hardware(),
        "empirical_status": "SMOKE_RUN_ONLY_FULL_DEVELOPMENT_AND_TEST_NOT_RUN",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _official_forecast(
    origin: HondaKumoOrigin,
    *,
    device: str,
    context_df: pd.DataFrame | None,
    alpha: float,
    k: int,
    temperature: float,
    context_stride: int,
) -> np.ndarray:
    ForecastingConfig, perform_forecasting = _kumo_api()
    standardizer = ARTIFACT_DIR / "standardizer.pkl"
    checkpoint = ARTIFACT_DIR / "moment_head_512_6hr.pt"
    backbone = ARTIFACT_DIR / "moment-1-large"
    if not standardizer.exists() or not checkpoint.exists() or not (backbone / "model.safetensors").exists():
        raise FileNotFoundError("Pinned Kumo weights are absent; run `honda-kumo prepare` first")
    config = ForecastingConfig(
        timestamp_column="timestamp",
        target_column="grid_exchange_kw_signed",
        standardizer_pkl=str(standardizer),
        ckpt=str(checkpoint),
        model_name=str(backbone),
        seq_len=len(origin.input_frame),
        forecast_horizon=origin.forecast_horizon,
        model_horizon=72,
        alpha=alpha,
        k=k,
        temperature=temperature,
        context_stride=context_stride,
        seed=SEED,
        device=device,
        use_cross_channel=False,
        num_workers=0,
        local_files_only=True,
    )
    output = perform_forecasting(df=origin.input_frame, config=config, context_df=context_df)
    expected_timestamps = origin.forecast_path_timestamps.tz_localize(None)
    actual_timestamps = pd.DatetimeIndex(pd.to_datetime(output.timestamp))
    if len(actual_timestamps) != len(expected_timestamps) or not (actual_timestamps == expected_timestamps).all():
        raise RuntimeError("Kumo output timestamps do not match the audited UTC forecast path")
    prediction = output.grid_exchange_kw_signed_forecast.to_numpy(float)
    if len(prediction) != origin.forecast_horizon or not np.isfinite(prediction).all():
        raise RuntimeError("Kumo returned an incomplete or nonfinite forecast")
    return prediction


def _calendar_contexts() -> dict[str, str]:
    frame = pd.read_parquet(INDEX, columns=["target_day_local", "holiday_context"])
    return frame.groupby("target_day_local").holiday_context.first().astype(str).to_dict()


def _thresholds_from_config(config: dict[str, Any]) -> PreissueThresholds:
    values = config["regime_thresholds"]
    if any(values[key] is None for key in ["low_abs_level", "high_abs_level", "high_volatility"]):
        raise RuntimeError("Development-fitted regime thresholds are not frozen")
    return PreissueThresholds(**values)


def run_origins(
    origins: list[HondaKumoOrigin],
    *,
    method: str,
    device: str,
    config: dict[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    observed = _observed_grid()
    protocol_index = pd.read_parquet(INDEX)
    calendars = _calendar_contexts()
    predictions: list[np.ndarray] = []
    retrieval_rows: list[dict[str, Any]] = []
    thresholds = _thresholds_from_config(config) if method.startswith("REGIME_") else None
    for origin in origins:
        direct = _official_forecast(
            origin,
            device=device,
            context_df=None,
            alpha=1.0,
            k=config["k"],
            temperature=config["temperature"],
            context_stride=config["context_stride"],
        )
        if method == "DIRECT":
            predictions.append(direct)
            continue
        if method == "NATIVE_DARR":
            archive = longest_complete_archive(
                observed,
                query_issue_utc=origin.issue_utc,
                seq_len=len(origin.input_frame),
                continuation_steps=origin.forecast_horizon,
                max_rows=config.get("native_archive_max_rows"),
            )
            predictions.append(
                _official_forecast(
                    origin,
                    device=device,
                    context_df=archive,
                    alpha=config["alpha"],
                    k=config["k"],
                    temperature=config["temperature"],
                    context_stride=config["context_stride"],
                )
            )
            continue
        if not method.startswith("REGIME_") or thresholds is None:
            raise ValueError(f"Unsupported Kumo method: {method}")
        strategy = method.removeprefix("REGIME_").lower()
        candidates = build_retrieval_candidates(
            observed,
            protocol_index,
            seq_len=len(origin.input_frame),
            continuation_steps=origin.forecast_horizon,
            query_issue_utc=origin.issue_utc,
            thresholds=thresholds,
        )
        query_features = preissue_features(origin.input_frame.grid_exchange_kw_signed.to_numpy(float))
        selected = select_candidates(
            origin,
            candidates,
            query_calendar_context=calendars[origin.target_day_local],
            query_preissue_regime=preissue_regime(query_features, thresholds),
            strategy=strategy,
            k=config["k"],
        )
        retrieved, weights = retrieval_forecast(selected, temperature=config["temperature"])
        prediction = config["alpha"] * direct + (1 - config["alpha"]) * retrieved
        predictions.append(prediction)
        for rank, (row, weight) in enumerate(zip(selected.itertuples(), weights, strict=True), start=1):
            retrieval_rows.append(
                {
                    "query_target_day_local": origin.target_day_local,
                    "query_issue_utc": origin.issue_utc,
                    "strategy": strategy,
                    "rank": rank,
                    "candidate_id": row.candidate_id,
                    "candidate_issue_utc": row.candidate_issue_utc,
                    "candidate_target_day_local": row.candidate_target_day_local,
                    "similarity_distance": row.similarity_distance,
                    "weight": float(weight),
                    "calendar_context": row.calendar_context,
                    "preissue_regime": row.preissue_regime,
                    "available_at_utc": row.available_at_utc,
                }
            )
    return predictions, retrieval_rows


def run_command(args: argparse.Namespace) -> None:
    config_path = FROZEN_CONFIG if args.phase == "test" else EXPERIMENT_CONFIG
    if args.phase == "test" and not FROZEN_CONFIG.exists():
        raise PermissionError("Kumo TEST is forbidden until development selection has produced a frozen config")
    config = _load_json(config_path)
    if args.phase == "test" and config.get("selection_status") != "FROZEN_AFTER_DEVELOPMENT_SELECTION":
        raise PermissionError("Kumo TEST configuration is not development-frozen")
    if args.phase == "test":
        selected = config["selected_configuration"]
        invocation = {
            "method": args.method,
            "seq_len": args.seq_len,
            "alpha": args.alpha,
            "k": args.k,
            "temperature": args.temperature,
            "context_stride": args.context_stride,
        }
        mismatches = {
            key: (selected[key], value)
            for key, value in invocation.items()
            if selected[key] != value
        }
        if mismatches:
            raise PermissionError(f"Kumo TEST invocation differs from the development-frozen configuration: {mismatches}")
    years = [2021, 2022] if args.phase == "development" else [2023]
    origins = load_honda_origins(years, seq_len=args.seq_len, max_origins=args.max_origins)
    runtime_config = {
        "alpha": args.alpha,
        "k": args.k,
        "temperature": args.temperature,
        "context_stride": args.context_stride,
        "native_archive_max_rows": args.native_archive_max_rows,
        "regime_thresholds": config.get("regime_thresholds"),
    }
    started = time.perf_counter()
    process = psutil.Process(os.getpid()) if psutil is not None else None
    before_rss = process.memory_info().rss if process is not None else None
    predictions, retrieval_rows = run_origins(
        origins,
        method=args.method,
        device=args.device,
        config=runtime_config,
    )
    elapsed = time.perf_counter() - started
    configuration_id = (
        f"{args.method.lower()}_l{args.seq_len}_a{args.alpha:g}_k{args.k}_"
        f"t{args.temperature:g}_s{args.context_stride}"
    )
    frame = result_frame(
        origins,
        predictions,
        configuration_id=configuration_id,
        method=args.method,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False, compression="zstd")
    if retrieval_rows:
        pd.DataFrame(retrieval_rows).to_parquet(
            args.output.with_name(args.output.stem + "_retrieval_metadata.parquet"),
            index=False,
            compression="zstd",
        )
    runtime = {
        "phase": args.phase,
        "method": args.method,
        "configuration_id": configuration_id,
        "forecast_origins": len(origins),
        "prediction_rows": len(frame),
        "elapsed_seconds": elapsed,
        "seconds_per_origin": elapsed / max(len(origins), 1),
        "rss_before_bytes": before_rss,
        "rss_after_bytes": process.memory_info().rss if process is not None else None,
        "device": args.device,
        "hardware": _hardware(),
        "kumo_ts_commit": KUMO_TS_COMMIT,
        "model_revision": MODEL_REVISION,
        "backbone_revision": BACKBONE_REVISION,
        "complete_issue_to_delivery_horizon": True,
        "test_used_for_selection": False,
    }
    args.runtime_output.parent.mkdir(parents=True, exist_ok=True)
    args.runtime_output.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze_command(selection_table: Path) -> dict[str, Any]:
    table = pd.read_csv(selection_table)
    required = {
        "configuration_id",
        "method",
        "seq_len",
        "alpha",
        "k",
        "temperature",
        "context_stride",
        "fold",
        "coverage",
        "mae_kw",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Development selection table is missing {sorted(missing)}")
    candidates = table[(table.fold == "CONCATENATED") & (table.coverage == 1.0)].copy()
    if candidates.empty:
        raise ValueError("No full-coverage concatenated development configuration can be frozen")
    best_mae = float(candidates.mae_kw.min())
    near = candidates[candidates.mae_kw <= best_mae * 1.005].copy()
    near["direct_priority"] = near.method.ne("DIRECT").astype(int)
    near = near.sort_values(
        ["direct_priority", "alpha", "k", "seq_len", "context_stride"],
        ascending=[True, False, True, True, False],
    )
    selected = near.iloc[0]
    protocol = _load_json(EXPERIMENT_CONFIG)
    frozen = {
        **protocol,
        "selection_status": "FROZEN_AFTER_DEVELOPMENT_SELECTION",
        "protocol_status": "FINAL_CONFIGURATION_FROZEN_BEFORE_RETROSPECTIVE_TEST_INFERENCE",
        "selection_table": selection_table.as_posix(),
        "selection_table_sha256": _sha256(selection_table),
        "selected_configuration": {
            key: selected[key].item() if hasattr(selected[key], "item") else selected[key]
            for key in [
                "configuration_id",
                "method",
                "seq_len",
                "alpha",
                "k",
                "temperature",
                "context_stride",
                "mae_kw",
                "coverage",
            ]
        },
        "development_results": table.to_dict("records"),
    }
    FROZEN_CONFIG.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frozen


def diagnose_command(args: argparse.Namespace) -> tuple[Path, Path]:
    day = pd.Timestamp(args.day)
    origins = load_honda_origins([day.year], seq_len=args.seq_len)
    matched = [origin for origin in origins if origin.target_day_local == args.day]
    if len(matched) != 1:
        raise ValueError(f"Expected exactly one Honda origin for {args.day}")
    origin = matched[0]
    config = _load_json(EXPERIMENT_CONFIG)
    thresholds = _thresholds_from_config(config)
    observed = _observed_grid()
    protocol_index = pd.read_parquet(INDEX)
    candidates = build_retrieval_candidates(
        observed,
        protocol_index,
        seq_len=args.seq_len,
        continuation_steps=origin.forecast_horizon,
        query_issue_utc=origin.issue_utc,
        thresholds=thresholds,
    )
    query_features = preissue_features(origin.input_frame.grid_exchange_kw_signed.to_numpy(float))
    selected = select_candidates(
        origin,
        candidates,
        query_calendar_context=_calendar_contexts()[origin.target_day_local],
        query_preissue_regime=preissue_regime(query_features, thresholds),
        strategy=args.strategy,
        k=args.k,
    )

    def embed_target(path: Path) -> np.ndarray:
        frame = pd.read_parquet(path).sort_values("timestamp_utc")
        frame = frame[frame.target_day_local.astype(str).eq(args.day)]
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame.timestamp_utc, utc=True))
        if len(timestamps) != len(origin.target_timestamps) or not (timestamps == origin.target_timestamps).all():
            raise ValueError(f"Diagnostic predictions in {path} do not align to {args.day}")
        full = np.full(origin.forecast_horizon, np.nan)
        full[origin.target_positions] = frame.prediction_kw.to_numpy(float)
        return full

    direct = embed_target(args.direct_predictions)
    enhanced = embed_target(args.enhanced_predictions)
    actual = observed.reindex(origin.forecast_path_timestamps).to_numpy(float)
    return write_retrieval_diagnostic(
        origin,
        selected,
        direct_full_path=direct,
        retrieval_full_path=enhanced,
        actual_full_path=actual,
        output_dir=args.output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-safe Honda NVIDIA Kumo research runner")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="download the pinned official checkpoint into ignored artifacts")
    audit = commands.add_parser("audit", help="record official API and local hardware capability")
    audit.add_argument("--output", type=Path, default=RESULT_DIR / "kumo_capability_audit.json")
    run = commands.add_parser("run", help="run direct, native DARR, or external regime retrieval")
    run.add_argument("--phase", choices=["development", "test"], required=True)
    run.add_argument(
        "--method",
        choices=[
            "DIRECT",
            "NATIVE_DARR",
            "REGIME_UNRESTRICTED",
            "REGIME_CALENDAR_EXACT",
            "REGIME_CALENDAR_FAMILY",
            "REGIME_CALENDAR_AND_PREISSUE",
        ],
        required=True,
    )
    run.add_argument("--seq-len", type=int, default=512)
    run.add_argument("--alpha", type=float, default=0.25)
    run.add_argument("--k", type=int, default=8)
    run.add_argument("--temperature", type=float, default=0.05)
    run.add_argument("--context-stride", type=int, default=4)
    run.add_argument("--native-archive-max-rows", type=int, default=8192)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run.add_argument("--max-origins", type=int)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--runtime-output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="evaluate a completed Kumo prediction file")
    evaluate.add_argument("--phase", choices=["development", "test"], required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    freeze = commands.add_parser("freeze", help="freeze a final configuration from development-only results")
    freeze.add_argument("--selection-table", type=Path, required=True)
    diagnose = commands.add_parser("diagnose", help="plot issuance-safe retrieved examples for one completed origin")
    diagnose.add_argument("--day", required=True)
    diagnose.add_argument(
        "--strategy",
        choices=["unrestricted", "calendar_exact", "calendar_family", "calendar_and_preissue"],
        required=True,
    )
    diagnose.add_argument("--seq-len", type=int, default=512)
    diagnose.add_argument("--k", type=int, default=8)
    diagnose.add_argument("--direct-predictions", type=Path, required=True)
    diagnose.add_argument("--enhanced-predictions", type=Path, required=True)
    diagnose.add_argument("--output-dir", type=Path, default=RESULT_DIR / "diagnostics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_weights(), indent=2))
    elif args.command == "audit":
        print(json.dumps(audit_environment(args.output), indent=2))
    elif args.command == "run":
        run_command(args)
    elif args.command == "evaluate":
        outputs = write_evaluation(args.predictions, args.output_dir, phase=args.phase)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    elif args.command == "freeze":
        print(json.dumps(freeze_command(args.selection_table), indent=2))
    elif args.command == "diagnose":
        outputs = diagnose_command(args)
        print(json.dumps([str(path) for path in outputs], indent=2))


if __name__ == "__main__":
    main()
