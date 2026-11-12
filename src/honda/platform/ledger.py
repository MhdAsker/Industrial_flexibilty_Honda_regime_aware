from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

TARGET_SEMANTICS = {"SIGNED_GRID_EXCHANGE", "NONNEGATIVE_DEMAND", "OTHER_APPROVED"}
LABEL_STATUSES = {"PENDING", "OBSERVED", "UNAVAILABLE"}


@dataclass(frozen=True)
class PredictionLedgerRecord:
    tenant_id: str
    site_id: str
    issue_id: str
    target_timestamp_utc: str
    model_id: str
    prediction_id: str
    information_cutoff_utc: str
    target_semantics: str
    prediction_kw: float
    model_sha256: str
    feature_sha256: str
    code_sha256: str
    environment_sha256: str
    snapshot_sha256: str
    calibration_version: str
    scenario_id: str | None = None
    label_kw: float | None = None
    operator_action: str | None = None
    decision_outcome: str | None = None
    label_status: str = "PENDING"

    def validate(self) -> None:
        if not all((self.tenant_id, self.site_id, self.issue_id, self.model_id, self.prediction_id, self.calibration_version)):
            raise ValueError("Prediction-ledger identity is incomplete")
        target = pd.Timestamp(self.target_timestamp_utc)
        cutoff = pd.Timestamp(self.information_cutoff_utc)
        if target.tzinfo is None or cutoff.tzinfo is None or cutoff > target:
            raise ValueError("Prediction-ledger timestamps are invalid")
        if self.target_semantics not in TARGET_SEMANTICS or self.label_status not in LABEL_STATUSES:
            raise ValueError("Prediction-ledger semantics or label status is unsupported")
        digests = (
            self.model_sha256, self.feature_sha256, self.code_sha256,
            self.environment_sha256, self.snapshot_sha256,
        )
        if any(len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value) for value in digests):
            raise ValueError("Prediction-ledger hashes must be SHA-256 digests")
        if not np.isfinite(self.prediction_kw):
            raise ValueError("Prediction-ledger prediction must be finite")
        if self.label_status == "OBSERVED" and (self.label_kw is None or not np.isfinite(self.label_kw)):
            raise ValueError("Observed labels require a finite value")
        if self.label_status != "OBSERVED" and self.label_kw is not None:
            raise ValueError("Only observed ledger rows may contain a label")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def paired_ledger_comparison(
    champion: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare models only on identical site/issue/interval snapshots."""
    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        return row["site_id"], row["issue_id"], row["target_timestamp_utc"]

    left = {key(row): row for row in champion if row["label_status"] == "OBSERVED"}
    right = {key(row): row for row in challenger if row["label_status"] == "OBSERVED"}
    if not left or set(left) != set(right):
        raise ValueError("Champion and challenger do not have identical observed ledger support")
    records = []
    for identity in sorted(left):
        champion_row = left[identity]; challenger_row = right[identity]
        if champion_row["snapshot_sha256"] != challenger_row["snapshot_sha256"]:
            raise ValueError("Champion and challenger did not run on the same immutable snapshot")
        if champion_row["target_semantics"] != challenger_row["target_semantics"]:
            raise ValueError("Champion and challenger target semantics differ")
        if not np.isclose(champion_row["label_kw"], challenger_row["label_kw"]):
            raise ValueError("Champion and challenger labels differ")
        actual = float(champion_row["label_kw"])
        records.append({
            "site_id": identity[0],
            "champion_error": abs(actual - float(champion_row["prediction_kw"])),
            "challenger_error": abs(actual - float(challenger_row["prediction_kw"])),
        })
    frame = pd.DataFrame(records)
    sites = frame.groupby("site_id")[["champion_error", "challenger_error"]].mean()
    sites["relative_improvement"] = (sites.champion_error - sites.challenger_error) / sites.champion_error
    return {
        "rows": len(frame),
        "sites": len(sites),
        "macro_relative_improvement": float(sites.relative_improvement.mean()),
        "worst_site_relative_improvement": float(sites.relative_improvement.min()),
        "per_site": sites.reset_index().to_dict("records"),
        "support_contract": "IDENTICAL_SITE_ISSUE_INTERVAL_SNAPSHOT",
    }
