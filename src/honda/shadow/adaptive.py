from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


def _hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdaptiveCalibrationState:
    """Immutable online bias and adaptive-conformal state for one issue sequence."""

    partner_id: str
    model_id: str
    sequence: int
    target_miscoverage: float
    adaptive_miscoverage: float
    conformal_learning_rate: float
    bias_kw: float
    bias_learning_rate: float
    absolute_scores_kw: tuple[float, ...]
    maximum_scores: int
    effective_at_utc: str
    previous_state_hash: str | None = None
    schema_version: str = "shadow.adaptive-calibration.v1"

    @classmethod
    def initial(
        cls,
        partner_id: str,
        model_id: str,
        effective_at_utc: str,
        *,
        target_miscoverage: float = 0.1,
        conformal_learning_rate: float = 0.01,
        bias_learning_rate: float = 0.05,
        maximum_scores: int = 2_000,
    ) -> AdaptiveCalibrationState:
        result = cls(
            partner_id, model_id, 0, target_miscoverage, target_miscoverage,
            conformal_learning_rate, 0.0, bias_learning_rate, (), maximum_scores,
            effective_at_utc,
        )
        result.validate()
        return result

    def validate(self) -> None:
        stamp = pd.Timestamp(self.effective_at_utc)
        if stamp.tzinfo is None:
            raise ValueError("Adaptive state effective time must be timezone-aware")
        if not self.partner_id or not self.model_id or self.sequence < 0:
            raise ValueError("Adaptive state identity is invalid")
        if not 0 < self.target_miscoverage < 1 or not 0 < self.adaptive_miscoverage < 1:
            raise ValueError("Adaptive miscoverage levels must lie inside (0, 1)")
        if self.conformal_learning_rate <= 0 or not 0 < self.bias_learning_rate <= 1:
            raise ValueError("Adaptive learning rates are invalid")
        if self.maximum_scores < 20 or len(self.absolute_scores_kw) > self.maximum_scores:
            raise ValueError("Adaptive score-window configuration is invalid")
        if not np.isfinite((self.bias_kw, *self.absolute_scores_kw)).all():
            raise ValueError("Adaptive state contains nonfinite values")

    @property
    def state_hash(self) -> str:
        return _hash(asdict(self))

    def radius_kw(self) -> float | None:
        if len(self.absolute_scores_kw) < 20:
            return None
        level = float(np.clip(1 - self.adaptive_miscoverage, 0, 1))
        return float(np.quantile(self.absolute_scores_kw, level, method="higher"))

    def calibrate(self, point_forecast_kw: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        point = np.asarray(point_forecast_kw, dtype=float) + self.bias_kw
        radius = self.radius_kw()
        if radius is None:
            return point, None, None
        return point, point - radius, point + radius

    def update(
        self,
        raw_prediction_kw: np.ndarray,
        realized_kw: np.ndarray,
        effective_at_utc: str,
    ) -> AdaptiveCalibrationState:
        self.validate()
        predicted = np.asarray(raw_prediction_kw, dtype=float)
        realized = np.asarray(realized_kw, dtype=float)
        if predicted.shape != realized.shape or predicted.size == 0:
            raise ValueError("Adaptive update predictions and realizations must share a nonempty shape")
        if not np.isfinite(predicted).all() or not np.isfinite(realized).all():
            raise ValueError("Adaptive update contains nonfinite observations")
        corrected = predicted + self.bias_kw
        scores = np.abs(realized - corrected)
        radius = self.radius_kw()
        error_rate = self.target_miscoverage if radius is None else float(np.mean(scores > radius))
        next_alpha = float(np.clip(
            self.adaptive_miscoverage
            + self.conformal_learning_rate * (self.target_miscoverage - error_rate),
            0.001,
            0.999,
        ))
        mean_residual = float(np.mean(realized - predicted))
        next_bias = (1 - self.bias_learning_rate) * self.bias_kw + self.bias_learning_rate * mean_residual
        history = (*self.absolute_scores_kw, *(float(value) for value in scores))[-self.maximum_scores:]
        result = replace(
            self,
            sequence=self.sequence + 1,
            adaptive_miscoverage=next_alpha,
            bias_kw=float(next_bias),
            absolute_scores_kw=tuple(history),
            effective_at_utc=effective_at_utc,
            previous_state_hash=self.state_hash,
        )
        result.validate()
        return result


def write_state_once(path: str | Path, state: AdaptiveCalibrationState) -> None:
    state.validate()
    target = Path(path)
    payload = json.dumps({**asdict(state), "state_hash": state.state_hash}, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"Adaptive calibration state is immutable: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(payload, encoding="utf-8", newline="\n")


def load_state(path: str | Path) -> AdaptiveCalibrationState:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = value.pop("state_hash")
    state = AdaptiveCalibrationState(**value)
    state.validate()
    if state.state_hash != expected:
        raise ValueError("Adaptive calibration state hash does not match its content")
    return state
