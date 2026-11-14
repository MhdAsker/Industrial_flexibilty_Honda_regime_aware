from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.honda.kumo.data_adapter import HondaKumoOrigin
from src.honda.kumo.retrieval import (
    RetrievalCandidate,
    longest_complete_archive,
    retrieval_forecast,
    select_candidates,
)
from src.honda.kumo.regimes import calendar_family


def query() -> HondaKumoOrigin:
    issue = pd.Timestamp("2023-01-01T12:00:00Z")
    history = pd.date_range(end=issue - pd.Timedelta(minutes=15), periods=8, freq="15min", tz="UTC")
    path = pd.date_range(issue, periods=6, freq="15min", tz="UTC")
    return HondaKumoOrigin(
        target_day_local="2023-01-02",
        issue_utc=issue,
        input_frame=pd.DataFrame({"timestamp": history.tz_localize(None), "grid_exchange_kw_signed": np.arange(8.0)}),
        forecast_path_timestamps=path,
        target_timestamps=path[2:],
        target_positions=np.arange(2, 6),
        observed_fraction=1.0,
        imputed_history_count=0,
    )


def candidate(identifier: str, offset: float, family: str, regime: str) -> RetrievalCandidate:
    issue = pd.Timestamp("2022-01-01T12:00:00Z") + pd.Timedelta(days=int(offset))
    history = pd.date_range(end=issue - pd.Timedelta(minutes=15), periods=8, freq="15min", tz="UTC")
    continuation = pd.date_range(issue, periods=6, freq="15min", tz="UTC")
    return RetrievalCandidate(
        candidate_id=identifier,
        issue_utc=issue,
        target_day_local=str(issue.date()),
        input_timestamps=history,
        continuation_timestamps=continuation,
        input_values=np.arange(8.0) + offset,
        continuation_values=np.full(6, offset),
        calendar_context="NORMAL_WEEKDAY" if family == "WORKDAY" else "WEEKEND",
        calendar_family=family,
        preissue_regime=regime,
    )


def test_regime_filter_is_external_explicit_and_auditable():
    selected = select_candidates(
        query(),
        [candidate("a", 1, "WORKDAY", "TYPICAL_LOAD"), candidate("b", 2, "NONWORKDAY", "LOW_LOAD")],
        query_calendar_context="NORMAL_WEEKDAY",
        query_preissue_regime="TYPICAL_LOAD",
        strategy="calendar_and_preissue",
        k=8,
    )
    assert selected.candidate_id.tolist() == ["a"]
    assert selected.available_at_utc.iloc[0] < query().issue_utc


def test_combined_holiday_labels_map_to_a_known_issuance_safe_family():
    assert calendar_family("WEEKEND;PUBLIC_HOLIDAY;DAY_AFTER_PUBLIC_HOLIDAY") == "NONWORKDAY"
    assert calendar_family("NORMAL_WEEKDAY;DAY_BEFORE_PUBLIC_HOLIDAY") == "WORKDAY_ADJACENT_HOLIDAY"
    assert calendar_family("NORMAL_WEEKDAY") == "WORKDAY"


def test_retrieval_only_control_and_weights_are_well_defined():
    selected = select_candidates(
        query(),
        [candidate("a", 1, "WORKDAY", "TYPICAL_LOAD"), candidate("b", 2, "WORKDAY", "TYPICAL_LOAD")],
        query_calendar_context="NORMAL_WEEKDAY",
        query_preissue_regime="TYPICAL_LOAD",
        strategy="unrestricted",
        k=2,
    )
    forecast, weights = retrieval_forecast(selected, temperature=0.05)
    assert forecast.shape == (6,)
    assert weights.sum() == pytest.approx(1.0)
    assert np.isfinite(forecast).all()


def test_native_darr_archive_is_one_continuous_complete_run():
    index = pd.date_range("2022-01-01", periods=40, freq="15min", tz="UTC")
    values = pd.Series(np.arange(40.0), index=index)
    values.iloc[12:15] = np.nan
    archive = longest_complete_archive(
        values,
        query_issue_utc=pd.Timestamp("2023-01-01T00:00:00Z"),
        seq_len=8,
        continuation_steps=4,
    )
    assert archive.grid_exchange_kw_signed.notna().all()
    assert archive.timestamp.diff().dropna().eq(pd.Timedelta(minutes=15)).all()


def test_native_darr_archive_fails_when_no_complete_window_exists():
    index = pd.date_range("2022-01-01", periods=10, freq="15min", tz="UTC")
    with pytest.raises(ValueError, match="No complete historical run"):
        longest_complete_archive(
            pd.Series(np.arange(10.0), index=index),
            query_issue_utc=pd.Timestamp("2023-01-01T00:00:00Z"),
            seq_len=8,
            continuation_steps=4,
        )
