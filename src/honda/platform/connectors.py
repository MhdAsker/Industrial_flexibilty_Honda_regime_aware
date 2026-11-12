from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class ConnectorBatch:
    connector_id: str
    connector_type: str
    observed_at_utc: str
    available_at_utc: str
    retrieved_at_utc: str
    records: tuple[dict[str, Any], ...]
    source_cursor: str | None = None

    def validate(self, issue_time_utc: str, maximum_age_seconds: int) -> None:
        observed = pd.Timestamp(self.observed_at_utc)
        available = pd.Timestamp(self.available_at_utc)
        retrieved = pd.Timestamp(self.retrieved_at_utc)
        issue = pd.Timestamp(issue_time_utc)
        if any(value.tzinfo is None for value in (observed, available, retrieved, issue)):
            raise ValueError("Connector timestamps must be timezone-aware")
        if available > retrieved:
            raise ValueError("Source availability cannot follow retrieval")
        if available > issue or retrieved > issue:
            raise ValueError("Snapshot contains information unavailable at issue time")
        age = (issue - observed).total_seconds()
        if age < 0:
            raise ValueError("Observed timestamp lies in the future")
        if age > maximum_age_seconds:
            raise ValueError(f"Connector data are stale by {age:.0f} seconds")
        if not self.records:
            raise ValueError("Connector returned no records")

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), default=str)
        return sha256(payload.encode("utf-8")).hexdigest()


class LiveConnector(Protocol):
    connector_id: str
    connector_type: str
    maximum_age_seconds: int

    def fetch(self, issue_time_utc: str) -> ConnectorBatch:
        """Fetch only data known by the supplied issue time."""
        ...


@dataclass
class InMemoryConnector:
    """Deterministic adapter used for integration tests and offline replay only."""

    connector_id: str
    connector_type: str
    maximum_age_seconds: int
    batch: ConnectorBatch

    def fetch(self, issue_time_utc: str) -> ConnectorBatch:
        self.batch.validate(issue_time_utc, self.maximum_age_seconds)
        return self.batch


def capture_connector_snapshot(connectors: tuple[LiveConnector, ...], issue_time_utc: str) -> dict[str, Any]:
    if not connectors:
        raise ValueError("At least one connector is required")
    batches: dict[str, Any] = {}
    for connector in connectors:
        if connector.connector_id in batches:
            raise ValueError("Connector IDs must be unique inside a snapshot")
        batch = connector.fetch(issue_time_utc)
        if batch.connector_id != connector.connector_id or batch.connector_type != connector.connector_type:
            raise ValueError("Connector batch identity does not match its registration")
        batch.validate(issue_time_utc, connector.maximum_age_seconds)
        batches[connector.connector_id] = {"sha256": batch.digest(), "batch": batch.payload()}
    return {"issue_time_utc": issue_time_utc, "connectors": batches}
