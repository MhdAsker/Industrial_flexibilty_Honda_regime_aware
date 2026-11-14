from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.honda.kumo.retrieval import RetrievalCandidate, validate_candidate


def candidate(continuation_end: str) -> RetrievalCandidate:
    issue = pd.Timestamp("2022-12-30T12:00:00Z")
    input_times = pd.date_range(end=issue - pd.Timedelta(minutes=15), periods=8, freq="15min", tz="UTC")
    continuation_times = pd.date_range(end=pd.Timestamp(continuation_end), periods=4, freq="15min", tz="UTC")
    return RetrievalCandidate(
        candidate_id="fixture",
        issue_utc=issue,
        target_day_local="2022-12-31",
        input_timestamps=input_times,
        continuation_timestamps=continuation_times,
        input_values=np.arange(8, dtype=float),
        continuation_values=np.arange(4, dtype=float),
        calendar_context="NORMAL_WEEKDAY",
        calendar_family="WORKDAY",
        preissue_regime="TYPICAL_LOAD",
    )


def test_future_continuation_is_rejected():
    query_issue = pd.Timestamp("2022-12-30T12:30:00Z")
    with pytest.raises(ValueError, match="not completely observable|overlaps"):
        validate_candidate(candidate("2022-12-30T12:45:00Z"), query_issue)


def test_candidate_input_must_end_at_historical_issue_boundary():
    item = candidate("2022-12-30T12:45:00Z")
    shifted = RetrievalCandidate(**{**item.__dict__, "issue_utc": item.issue_utc + pd.Timedelta(minutes=15)})
    with pytest.raises(ValueError, match="end exactly"):
        validate_candidate(shifted, pd.Timestamp("2023-01-01T12:00:00Z"))


def test_nonfinite_retrieval_continuation_is_rejected():
    item = candidate("2022-12-30T12:45:00Z")
    bad = item.continuation_values.copy()
    bad[-1] = np.nan
    injected = RetrievalCandidate(**{**item.__dict__, "continuation_values": bad})
    with pytest.raises(ValueError, match="complete OBSERVED_ORIGINAL"):
        validate_candidate(injected, pd.Timestamp("2023-01-01T12:00:00Z"))
