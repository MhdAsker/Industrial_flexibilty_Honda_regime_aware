from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_adapter import HondaKumoOrigin


def write_retrieval_diagnostic(
    query: HondaKumoOrigin,
    selected: pd.DataFrame,
    *,
    direct_full_path: np.ndarray,
    retrieval_full_path: np.ndarray,
    actual_full_path: np.ndarray | None,
    output_dir: Path,
) -> tuple[Path, Path]:
    if selected.empty:
        raise ValueError("Cannot diagnose an empty retrieval selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_day = query.target_day_local.replace("/", "-")
    metadata_path = output_dir / f"retrieval_{safe_day}.csv"
    metadata = selected.drop(columns=["candidate"]).copy()
    metadata.to_csv(metadata_path, index=False, float_format="%.10g", lineterminator="\n")

    figure_path = output_dir / f"retrieval_{safe_day}.png"
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    query_profile = query.input_frame.grid_exchange_kw_signed.to_numpy(float)
    axes[0].plot(query_profile, color="black", linewidth=2, label="query history")
    for row in selected.head(5).itertuples():
        axes[0].plot(row.candidate.input_values, alpha=0.55, label=f"{row.candidate_target_day_local} d={row.similarity_distance:.3f}")
    axes[0].set_title("Issuance-safe query and retrieved historical profiles")
    axes[0].set_ylabel("signed exchange (kW)")
    axes[0].legend(fontsize=7, ncol=2)

    for row in selected.head(5).itertuples():
        axes[1].plot(
            row.candidate.continuation_values,
            alpha=0.7,
            label=f"{row.candidate_target_day_local} d={row.similarity_distance:.3f}",
        )
    axes[1].set_title("Observed historical continuations available before query issuance")
    axes[1].set_ylabel("signed exchange (kW)")
    axes[1].legend(fontsize=7, ncol=2)

    axes[2].plot(direct_full_path, label="direct Kumo")
    axes[2].plot(retrieval_full_path, label="retrieval enhanced")
    if actual_full_path is not None:
        axes[2].plot(actual_full_path, color="black", linewidth=2, label="observed")
    axes[2].set_title("Complete issue-to-delivery path; similarity is not causal evidence")
    axes[2].set_ylabel("signed exchange (kW)")
    axes[2].set_xlabel("15-minute step from issue time")
    axes[2].legend()
    fig.savefig(figure_path, dpi=150, metadata={"Software": "Honda Kumo study"})
    plt.close(fig)
    return metadata_path, figure_path
