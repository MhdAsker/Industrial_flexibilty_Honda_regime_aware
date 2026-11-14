from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_adapter import FREQ, HondaKumoOrigin
from .regimes import PreissueThresholds, calendar_family, preissue_features, preissue_regime


@dataclass(frozen=True)
class RetrievalCandidate:
    candidate_id: str
    issue_utc: pd.Timestamp
    target_day_local: str
    input_timestamps: pd.DatetimeIndex
    continuation_timestamps: pd.DatetimeIndex
    input_values: np.ndarray
    continuation_values: np.ndarray
    calendar_context: str
    calendar_family: str
    preissue_regime: str

    @property
    def available_at_utc(self) -> pd.Timestamp:
        return self.continuation_timestamps[-1] + FREQ


def validate_candidate(candidate: RetrievalCandidate, query_issue_utc: pd.Timestamp) -> None:
    query_issue = pd.Timestamp(query_issue_utc).tz_convert("UTC")
    if len(candidate.input_values) != len(candidate.input_timestamps):
        raise ValueError("Retrieval input metadata length mismatch")
    if len(candidate.continuation_values) != len(candidate.continuation_timestamps):
        raise ValueError("Retrieval continuation metadata length mismatch")
    if not np.isfinite(candidate.input_values).all() or not np.isfinite(candidate.continuation_values).all():
        raise ValueError("Retrieval candidates must use complete OBSERVED_ORIGINAL values")
    if candidate.input_timestamps[-1] + FREQ != candidate.issue_utc:
        raise ValueError("Candidate input must end exactly at its historical issue boundary")
    if candidate.continuation_timestamps[0] != candidate.issue_utc:
        raise ValueError("Candidate continuation must begin at its historical issue time")
    if candidate.available_at_utc > query_issue:
        raise ValueError("Retrieval continuation was not completely observable at query issuance")
    if candidate.continuation_timestamps[-1] >= query_issue:
        raise ValueError("Retrieval candidate overlaps the unknown query period")


def build_retrieval_candidates(
    observed_series: pd.Series,
    protocol_index: pd.DataFrame,
    *,
    seq_len: int,
    continuation_steps: int,
    query_issue_utc: pd.Timestamp,
    thresholds: PreissueThresholds,
) -> list[RetrievalCandidate]:
    series = observed_series.sort_index()
    if series.index.tz is None:
        raise ValueError("Observed series index must be timezone-aware")
    candidates: list[RetrievalCandidate] = []
    day_metadata = protocol_index.groupby("target_day_local", sort=True).first()
    for target_day, row in day_metadata.iterrows():
        issue = pd.Timestamp(row.forecast_issue_time_utc).tz_convert("UTC")
        input_times = pd.date_range(end=issue - FREQ, periods=seq_len, freq=FREQ, tz="UTC")
        continuation_times = pd.date_range(issue, periods=continuation_steps, freq=FREQ, tz="UTC")
        input_values = series.reindex(input_times).to_numpy(float)
        continuation = series.reindex(continuation_times).to_numpy(float)
        calendar_context = str(row.holiday_context)
        if not np.isfinite(input_values).all() or not np.isfinite(continuation).all():
            continue
        features = preissue_features(input_values)
        candidate = RetrievalCandidate(
            candidate_id=f"{target_day}@{issue.isoformat()}",
            issue_utc=issue,
            target_day_local=str(target_day),
            input_timestamps=input_times,
            continuation_timestamps=continuation_times,
            input_values=input_values,
            continuation_values=continuation,
            calendar_context=calendar_context,
            calendar_family=calendar_family(calendar_context),
            preissue_regime=preissue_regime(features, thresholds),
        )
        try:
            validate_candidate(candidate, query_issue_utc)
        except ValueError:
            continue
        candidates.append(candidate)
    return candidates


def normalized_profile_distance(query: np.ndarray, candidate: np.ndarray) -> float:
    q = np.asarray(query, dtype=float)
    c = np.asarray(candidate, dtype=float)
    if q.shape != c.shape or q.ndim != 1 or not np.isfinite(q).all() or not np.isfinite(c).all():
        raise ValueError("Similarity requires finite, equal-length one-dimensional profiles")
    q_scale = max(float(np.std(q)), 1e-6)
    c_scale = max(float(np.std(c)), 1e-6)
    qn = (q - float(np.mean(q))) / q_scale
    cn = (c - float(np.mean(c))) / c_scale
    return float(np.sqrt(np.mean((qn - cn) ** 2)))


def select_candidates(
    query: HondaKumoOrigin,
    candidates: list[RetrievalCandidate],
    *,
    query_calendar_context: str,
    query_preissue_regime: str,
    strategy: str,
    k: int,
) -> pd.DataFrame:
    if strategy not in {"unrestricted", "calendar_exact", "calendar_family", "calendar_and_preissue"}:
        raise ValueError(f"Unsupported retrieval strategy: {strategy}")
    if k < 1:
        raise ValueError("k must be positive")
    query_values = query.input_frame.grid_exchange_kw_signed.to_numpy(float)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        validate_candidate(candidate, query.issue_utc)
        eligible = True
        if strategy == "calendar_exact":
            eligible = candidate.calendar_context == query_calendar_context
        elif strategy == "calendar_family":
            eligible = candidate.calendar_family == calendar_family(query_calendar_context)
        elif strategy == "calendar_and_preissue":
            eligible = (
                candidate.calendar_family == calendar_family(query_calendar_context)
                and candidate.preissue_regime == query_preissue_regime
            )
        if eligible:
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_issue_utc": candidate.issue_utc,
                    "candidate_target_day_local": candidate.target_day_local,
                    "calendar_context": candidate.calendar_context,
                    "calendar_family": candidate.calendar_family,
                    "preissue_regime": candidate.preissue_regime,
                    "similarity_distance": normalized_profile_distance(query_values, candidate.input_values),
                    "available_at_utc": candidate.available_at_utc,
                    "candidate": candidate,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["candidate_id", "similarity_distance", "candidate"])
    return pd.DataFrame(rows).sort_values(["similarity_distance", "candidate_issue_utc"]).head(k).reset_index(drop=True)


def retrieval_forecast(selected: pd.DataFrame, *, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    if selected.empty:
        raise ValueError("No eligible retrieval candidates")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    distance = selected.similarity_distance.to_numpy(float)
    logits = -distance / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    continuations = np.stack([c.continuation_values for c in selected.candidate])
    return np.sum(continuations * weights[:, None], axis=0), weights


def longest_complete_archive(
    observed_series: pd.Series,
    *,
    query_issue_utc: pd.Timestamp,
    seq_len: int,
    continuation_steps: int,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Return one continuous, fully observed archive safe for native DARR.

    Kumo's public API accepts one continuous context frame and offers no window
    eligibility mask. Restricting it to a complete run prevents its internal
    zero fill from fabricating retrieved continuations.
    """

    cutoff = pd.Timestamp(query_issue_utc).tz_convert("UTC") - FREQ
    eligible = observed_series.loc[observed_series.index <= cutoff].copy()
    complete = eligible.notna()
    groups = complete.ne(complete.shift(fill_value=False)).cumsum()
    runs: list[pd.Series] = []
    for _, run in eligible[complete].groupby(groups[complete]):
        expected = pd.date_range(run.index[0], run.index[-1], freq=FREQ, tz="UTC")
        if len(expected) == len(run) and len(run) >= seq_len + continuation_steps:
            runs.append(run)
    if not runs:
        raise ValueError("No complete historical run is long enough for native DARR")
    chosen = max(runs, key=len)
    if max_rows is not None and len(chosen) > max_rows:
        chosen = chosen.iloc[-max_rows:]
    if chosen.index[-1] + FREQ > pd.Timestamp(query_issue_utc).tz_convert("UTC"):
        raise ValueError("Native DARR archive extends beyond query issuance")
    return pd.DataFrame(
        {
            "timestamp": chosen.index.tz_localize(None),
            "grid_exchange_kw_signed": chosen.to_numpy(float),
        }
    )
