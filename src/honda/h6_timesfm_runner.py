from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator

from src.honda.h6_foundation_data import ROOT, load_origins, result_frame, timesfm_inputs, write_runtime


MODEL_DIR = ROOT / "artifacts/foundation_models/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "test"], required=True)
    parser.add_argument("--context-days", type=int, required=True)
    parser.add_argument("--mode", choices=["U", "C0", "C1"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-origins", type=int)
    args = parser.parse_args()

    if args.phase == "test":
        frozen = ROOT / "config/honda_timesfm_zero_shot_frozen.json"
        if not frozen.exists():
            raise PermissionError("TimesFM TEST inference forbidden before frozen configuration")
        config = json.loads(frozen.read_text(encoding="utf-8"))
        if config["selected_context_days"] != args.context_days or config["selected_mode"] != args.mode:
            raise PermissionError("TimesFM TEST invocation differs from frozen configuration")

    torch.set_grad_enabled(False)
    torch.manual_seed(20261029)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    years = [2021, 2022] if args.phase == "development" else [2023]
    origins = load_origins(years, args.context_days, args.max_origins)
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    forecaster = TimesFM3Evaluator(
        ModelConfig(checkpoint_path=str(MODEL_DIR), per_core_batch_size=args.batch_size, device=args.device)
    )
    load_seconds = time.perf_counter() - started

    predictions: list[np.ndarray | None] = [None] * len(origins)
    infer_started = time.perf_counter()
    for horizon in sorted({x.target_horizon for x in origins}):
        indices = [i for i, x in enumerate(origins) if x.target_horizon == horizon]
        for start in range(0, len(indices), args.batch_size):
            chosen = indices[start : start + args.batch_size]
            batch_origins = [origins[i] for i in chosen]
            contexts, past_only, past_future = timesfm_inputs(batch_origins, args.mode)
            outputs = list(
                forecaster.predict_batch(
                    contexts=contexts,
                    horizon=horizon,
                    past_only_covariates=past_only,
                    past_future_covariates=past_future,
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                    make_positive=False,
                    sort_quantiles=True,
                    use_znorm=False,
                    padding_mode="none",
                )
            )
            for idx, output in zip(chosen, outputs):
                predictions[idx] = np.asarray(output.forecast, dtype=float)
            peak_rss = max(peak_rss, process.memory_info().rss)
    inference_seconds = time.perf_counter() - infer_started
    if any(x is None for x in predictions):
        raise RuntimeError("Missing TimesFM predictions")
    frame = result_frame(origins, predictions, "TIMESFM3", args.mode, args.context_days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False, compression="zstd")
    write_runtime(
        args.runtime_output,
        {
            "model": "TIMESFM3",
            "model_id": "google/timesfm-3.0-pytorch",
            "revision": REVISION,
            "package": "timesfm",
            "package_version": importlib.metadata.version("timesfm"),
            "checkpoint_sha256": sha256(MODEL_DIR / "model.safetensors"),
            "phase": args.phase,
            "mode": args.mode,
            "context_days": args.context_days,
            "forecast_origins": len(origins),
            "prediction_rows": len(frame),
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "seconds_per_origin": inference_seconds / len(origins),
            "peak_rss_bytes": peak_rss,
            "device": args.device,
            "dtype": "float32",
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "point_forecast": "official forecast output (median)",
            "native_missing_policy": "official historical-only linear interpolation within context",
            "gradient_enabled": torch.is_grad_enabled(),
        },
    )


if __name__ == "__main__":
    main()
