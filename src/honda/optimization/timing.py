from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class InformationTimeline:
    dataset_id: str
    forecast_issue_time_local: str
    price_publication_evidence: str
    scheduling_decision_rule: str
    timing_classification: str
    allowed_measurements: str = "only measurements completed by frozen forecast issuance"


def validate_price_timing(
    forecast_issue_time: str,
    decision_time: str,
    publication_time: str | None,
    classification: str,
) -> None:
    issue = pd.Timestamp(forecast_issue_time)
    decision = pd.Timestamp(decision_time)
    if issue.tzinfo is None or decision.tzinfo is None:
        raise ValueError("Timing values must be timezone-aware")
    if decision < issue:
        raise ValueError("Scheduling decision cannot precede forecast issuance")
    if classification == "OPERATIONAL_TIMING_VERIFIED":
        if publication_time is None:
            raise ValueError("Verified operational timing requires a publication timestamp")
        published = pd.Timestamp(publication_time)
        if published.tzinfo is None or decision < published:
            raise ValueError("Scheduling decision precedes price publication")
    elif classification not in {"EX_POST_PRICE_AWARE", "UNSUPPORTED_TIMING"}:
        raise ValueError("Unknown timing classification")


def observation_available(interval_end_utc: str, forecast_issue_time_utc: str) -> bool:
    end = pd.Timestamp(interval_end_utc)
    issue = pd.Timestamp(forecast_issue_time_utc)
    if end.tzinfo is None or issue.tzinfo is None:
        raise ValueError("Availability timestamps must be timezone-aware")
    return bool(end <= issue)
