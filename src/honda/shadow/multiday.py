from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResidualBlock:
    block_id: str
    issue_time_utc: str
    timestamps_utc: tuple[str, ...]
    residual_kw: tuple[float, ...]
    descriptor: tuple[float, ...]


def build_multiday_residual_blocks(
    frame: pd.DataFrame,
    *,
    block_hours: float = 72,
    stride_hours: float = 24,
) -> tuple[ResidualBlock, ...]:
    required = {"timestamp_utc", "available_at_utc", "actual_kw", "forecast_kw"}
    if missing := required - set(frame):
        raise ValueError(f"Residual-block frame is missing: {sorted(missing)}")
    ordered = frame.copy().sort_values("timestamp_utc").reset_index(drop=True)
    stamps = pd.DatetimeIndex(pd.to_datetime(ordered.timestamp_utc, utc=True))
    available = pd.DatetimeIndex(pd.to_datetime(ordered.available_at_utc, utc=True))
    if len(stamps) < 2 or not stamps.is_unique:
        raise ValueError("Residual-block timestamps must be unique")
    step = pd.Series(stamps).diff().dropna().median()
    intervals = round(block_hours / (step.total_seconds() / 3600))
    stride = round(stride_hours / (step.total_seconds() / 3600))
    if intervals < 2 or stride < 1:
        raise ValueError("Residual block duration or stride is invalid")
    residual = ordered.actual_kw.to_numpy(float) - ordered.forecast_kw.to_numpy(float)
    blocks = []
    for start in range(0, len(ordered) - intervals + 1, stride):
        stop = start + intervals
        block_stamps = stamps[start:stop]
        physical_hours = (block_stamps[-1] - block_stamps[0]).total_seconds() / 3600 + step.total_seconds() / 3600
        if not np.isclose(physical_hours, block_hours):
            continue
        values = residual[start:stop]
        issue = available[start:stop].max()
        descriptor = (
            float(np.mean(values)), float(np.std(values)), float(np.max(values)),
            float(np.min(values)), float(np.mean(np.abs(np.diff(values)))),
        )
        blocks.append(ResidualBlock(
            block_id=f"{block_stamps[0].isoformat()}__{block_stamps[-1].isoformat()}",
            issue_time_utc=issue.isoformat(),
            timestamps_utc=tuple(stamp.isoformat() for stamp in block_stamps),
            residual_kw=tuple(float(value) for value in values),
            descriptor=descriptor,
        ))
    return tuple(blocks)


def select_conditional_blocks(
    blocks: tuple[ResidualBlock, ...],
    target_descriptor: tuple[float, ...],
    issue_time_utc: str,
    count: int,
    bandwidth: float = 1.0,
) -> tuple[ResidualBlock, ...]:
    issue = pd.Timestamp(issue_time_utc)
    candidates = [block for block in blocks if pd.Timestamp(block.issue_time_utc) <= issue]
    if count < 1 or len(candidates) < count or bandwidth <= 0:
        raise ValueError("Insufficient causal residual blocks or invalid selection settings")
    features = np.asarray([block.descriptor for block in candidates])
    target = np.asarray(target_descriptor, dtype=float)
    scale = np.nanstd(features, axis=0); scale[scale < 1e-9] = 1
    distance = np.sqrt(np.mean(((features - target) / scale) ** 2, axis=1))
    order = np.argsort(distance / bandwidth, kind="mergesort")[:count]
    return tuple(candidates[index] for index in order)


def construct_information_tree(
    scenarios_kw: np.ndarray,
    timestamps_utc: tuple[str, ...],
    update_times_utc: tuple[str, ...],
    *,
    branch_bins: int = 2,
) -> np.ndarray:
    """Create cumulative node IDs; branches occur only at declared information updates."""
    scenarios = np.asarray(scenarios_kw, dtype=float)
    stamps = pd.DatetimeIndex(pd.to_datetime(timestamps_utc, utc=True))
    updates = pd.DatetimeIndex(pd.to_datetime(update_times_utc, utc=True))
    if scenarios.ndim != 2 or scenarios.shape[1] != len(stamps) or branch_bins < 2:
        raise ValueError("Scenario-tree dimensions or branch count are invalid")
    if not stamps.is_monotonic_increasing or not updates.is_monotonic_increasing:
        raise ValueError("Scenario and update timestamps must be increasing")
    nodes = np.full(scenarios.shape, "root", dtype=object)
    paths = ["root"] * len(scenarios)
    for update_number, update in enumerate(updates):
        position = int(stamps.searchsorted(update, side="left"))
        if position >= len(stamps):
            continue
        signal = scenarios[:, position]
        edges = np.quantile(signal, np.linspace(0, 1, branch_bins + 1)[1:-1])
        branches = np.searchsorted(edges, signal, side="right")
        for scenario, branch in enumerate(branches):
            paths[scenario] = f"{paths[scenario]}/u{update_number}b{int(branch)}"
            nodes[scenario, position:] = paths[scenario]
    return nodes.astype(str)
