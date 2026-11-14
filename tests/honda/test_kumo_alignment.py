from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.honda.kumo.data_adapter import build_origin


TZ = "Europe/Berlin"


def synthetic_origin(day: str, seq_len: int = 512):
    local_start = pd.Timestamp(day, tz=TZ)
    local_end = local_start + pd.DateOffset(days=1)
    target_times = pd.date_range(local_start, local_end, freq="15min", inclusive="left").tz_convert("UTC")
    issue_local = local_start - pd.Timedelta(days=1) + pd.Timedelta(hours=13)
    issue = issue_local.tz_convert("UTC")
    target = pd.DataFrame(
        {
            "timestamp_utc": target_times,
            "target_day_local": day,
            "forecast_issue_time_utc": issue,
        }
    )
    history_index = pd.date_range(end=issue - pd.Timedelta(minutes=15), periods=seq_len, freq="15min", tz="UTC")
    observed = pd.Series(np.arange(seq_len, dtype=float), index=history_index)
    return build_origin(target, observed, seq_len=seq_len)


@pytest.mark.parametrize(
    ("day", "target_steps", "full_steps"),
    [("2023-02-15", 96, 140), ("2023-03-26", 92, 136), ("2023-10-29", 100, 144)],
)
def test_complete_issue_to_delivery_path_handles_physical_dst(day, target_steps, full_steps):
    origin = synthetic_origin(day)
    assert len(origin.target_timestamps) == target_steps
    assert origin.forecast_horizon == full_steps
    assert origin.forecast_path_timestamps[0] == origin.issue_utc
    assert origin.forecast_path_timestamps[-1] == origin.target_timestamps[-1]
    assert origin.target_positions[0] == 44
    selected = origin.select_target(np.arange(full_steps, dtype=float))
    assert selected[0] == 44
    assert len(selected) == target_steps


def test_does_not_naively_treat_first_96_post_issue_steps_as_delivery_day():
    origin = synthetic_origin("2023-02-15")
    prediction = np.arange(origin.forecast_horizon, dtype=float)
    assert origin.select_target(prediction)[0] == 44
    assert origin.select_target(prediction)[-1] == 139
    assert not np.array_equal(origin.select_target(prediction), prediction[:96])
