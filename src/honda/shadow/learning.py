from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgeTransitionModel:
    coefficients: np.ndarray
    intercept: float
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        recovery_residual: np.ndarray,
        feature_names: tuple[str, ...],
        ridge: float = 1.0,
    ) -> RidgeTransitionModel:
        x = np.asarray(features, dtype=float); y = np.asarray(recovery_residual, dtype=float)
        if x.ndim != 2 or y.shape != (len(x),) or x.shape[1] != len(feature_names) or ridge < 0:
            raise ValueError("Transition-model training data are invalid")
        design = np.column_stack((np.ones(len(x)), x))
        penalty = np.eye(design.shape[1]) * ridge; penalty[0, 0] = 0
        fitted = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return cls(fitted[1:], float(fitted[0]), feature_names)

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.coefficients):
            raise ValueError("Transition-model prediction features are invalid")
        return self.intercept + x @ self.coefficients


@dataclass(frozen=True)
class OperatorAcceptanceModel:
    coefficients: np.ndarray
    intercept: float
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        accepted: np.ndarray,
        feature_names: tuple[str, ...],
        *,
        learning_rate: float = 0.05,
        iterations: int = 1_000,
        l2: float = 0.01,
    ) -> OperatorAcceptanceModel:
        x = np.asarray(features, dtype=float); y = np.asarray(accepted, dtype=float)
        if x.ndim != 2 or y.shape != (len(x),) or x.shape[1] != len(feature_names):
            raise ValueError("Operator-model training data are invalid")
        if len(np.unique(y)) < 2 or not np.isin(y, (0, 1)).all():
            raise ValueError("Operator model requires both accepted and non-accepted records")
        mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-9] = 1
        z = (x - mean) / scale
        weights = np.zeros(x.shape[1]); intercept = 0.0
        for _ in range(iterations):
            probability = 1 / (1 + np.exp(-np.clip(intercept + z @ weights, -30, 30)))
            error = probability - y
            intercept -= learning_rate * float(np.mean(error))
            weights -= learning_rate * (z.T @ error / len(z) + l2 * weights)
        raw_weights = weights / scale
        raw_intercept = intercept - float(mean @ raw_weights)
        return cls(raw_weights, raw_intercept, feature_names)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        score = self.intercept + np.asarray(features, dtype=float) @ self.coefficients
        return 1 / (1 + np.exp(-np.clip(score, -30, 30)))


@dataclass(frozen=True)
class OperatorModificationModel:
    coefficients: np.ndarray
    intercept: np.ndarray
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        context_features: np.ndarray,
        schedule_modification_kw: np.ndarray,
        feature_names: tuple[str, ...],
        ridge: float = 1.0,
    ) -> OperatorModificationModel:
        x = np.asarray(context_features, dtype=float)
        y = np.asarray(schedule_modification_kw, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or x.shape[1] != len(feature_names):
            raise ValueError("Operator-modification training data are invalid")
        design = np.column_stack((np.ones(len(x)), x))
        penalty = np.eye(design.shape[1]) * ridge; penalty[0, 0] = 0
        fitted = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return cls(fitted[1:], fitted[0], feature_names)

    def predict(self, context_features: np.ndarray) -> np.ndarray:
        x = np.asarray(context_features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.coefficients):
            raise ValueError("Operator-modification prediction features are invalid")
        return self.intercept + x @ self.coefficients


def infer_nonnegative_preferences(
    chosen_features: np.ndarray,
    alternative_features: np.ndarray,
    *,
    ridge: float = 1.0,
) -> np.ndarray:
    """Estimate normalized cost weights from chosen-minus-alternative feature differences."""
    chosen = np.asarray(chosen_features, dtype=float)
    alternatives = np.asarray(alternative_features, dtype=float)
    if chosen.shape != alternatives.shape or chosen.ndim != 2 or len(chosen) < 2:
        raise ValueError("Inverse-optimization observations are invalid")
    difference = alternatives - chosen
    target = np.ones(len(difference))
    weights = np.linalg.solve(difference.T @ difference + ridge * np.eye(difference.shape[1]), difference.T @ target)
    weights = np.maximum(weights, 0)
    if np.isclose(weights.sum(), 0):
        raise ValueError("Operational preferences are not identifiable from these choices")
    return weights / weights.sum()


def value_of_information_ablation(full_cost: np.ndarray, ablated_costs: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    full = np.asarray(full_cost, dtype=float)
    if full.ndim != 1 or not np.isfinite(full).all():
        raise ValueError("Full-information costs are invalid")
    result = {}
    for source, values in ablated_costs.items():
        ablated = np.asarray(values, dtype=float)
        if ablated.shape != full.shape or not np.isfinite(ablated).all():
            raise ValueError(f"Ablation costs are invalid for {source}")
        delta = ablated - full
        result[source] = {
            "mean_value_eur": float(np.mean(delta)),
            "median_value_eur": float(np.median(delta)),
            "positive_day_fraction": float(np.mean(delta > 0)),
        }
    return result
