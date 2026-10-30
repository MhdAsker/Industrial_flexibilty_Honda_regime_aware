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
from chronos import Chronos2Pipeline

from src.honda.h6_foundation_data import ROOT, chronos_inputs, load_origins, result_frame, write_runtime


MODEL_DIR = ROOT / "artifacts/foundation_models/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "test"], required=True)
    parser.add_argument("--context-days", type=int, required=True)
    parser.add_argument("--mode", choices=["U", "C0", "C1"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-origins", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.phase == "test":
        frozen = ROOT / "config/honda_chronos2_zero_shot_frozen.json"
        if not frozen.exists():
            raise PermissionError("Chronos TEST inference forbidden before frozen configuration")
        config = json.loads(frozen.read_text(encoding="utf-8"))
        if config["selected_context_days"] != args.context_days or config["selected_mode"] != args.mode:
            raise PermissionError("Chronos TEST invocation differs from frozen configuration")

    torch.set_grad_enabled(False)
    torch.manual_seed(20261029)
    years = [2021, 2022] if args.phase == "development" else [2023]
    origins = load_origins(years, args.context_days, args.max_origins)
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats()
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_DIR, device_map=args.device, dtype=dtype)
    load_seconds = time.perf_counter() - started
    quantiles = list(pipeline.quantiles)
    median_index = quantiles.index(0.5)

    predictions: list[np.ndarray | None] = [None] * len(origins)
    infer_started = time.perf_counter()
    for horizon in sorted({x.target_horizon for x in origins}):
        indices = [i for i, x in enumerate(origins) if x.target_horizon == horizon]
        for start in range(0, len(indices), args.batch_size):
            chosen = indices[start : start + args.batch_size]
            batch_origins = [origins[i] for i in chosen]
            outputs = pipeline.predict(
                chronos_inputs(batch_origins, args.mode),
                prediction_length=horizon,
                context_length=args.context_days * 96,
                batch_size=args.batch_size,
                cross_learning=False,
            )
            for idx, output in zip(chosen, outputs):
                predictions[idx] = output[0, median_index, :].detach().cpu().numpy().astype(float)
            peak_rss = max(peak_rss, process.memory_info().rss)
    inference_seconds = time.perf_counter() - infer_started
    if any(x is None for x in predictions):
        raise RuntimeError("Missing Chronos predictions")
    frame = result_frame(origins, predictions, "CHRONOS2", args.mode, args.context_days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False, compression="zstd")
    write_runtime(
        args.runtime_output,
        {
            "model": "CHRONOS2",
            "model_id": "amazon/chronos-2",
            "revision": REVISION,
            "package": "chronos-forecasting",
            "package_version": importlib.metadata.version("chronos-forecasting"),
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
            "dtype": str(dtype).replace("torch.", ""),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "point_forecast": "official quantile 0.5 (median)",
            "native_missing_policy": "Chronos-2 NaN context mask",
            "gradient_enabled": torch.is_grad_enabled(),
        },
    )


if __name__ == "__main__":
    main()
