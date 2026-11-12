from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError("Install the forecasting extra: pip install -e '.[forecasting]'") from exc
    return LGBMClassifier, LGBMRegressor


DEFAULT_PARAMS = {
    "n_estimators": 300,
    "num_leaves": 15,
    "learning_rate": 0.03,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "reg_lambda": 0.1,
    "random_state": 20261029,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 4,
    "verbosity": -1,
}


@dataclass
class RegimePredictions:
    prediction_kw: np.ndarray
    direct_kw: np.ndarray
    conditional_kw: np.ndarray
    export_probability: np.ndarray
    import_magnitude_kw: np.ndarray
    export_magnitude_kw: np.ndarray


class RegimeAwareMixture:
    """Signed-exchange hurdle model with a development-selected soft blend."""

    def __init__(self, params: dict[str, Any] | None = None, minimum_regime_rows: int = 100) -> None:
        classifier, regressor = _lightgbm()
        common = DEFAULT_PARAMS | (params or {})
        self.classifier = classifier(**common)
        self.direct = regressor(objective="regression_l1", **common)
        self.import_magnitude = regressor(objective="regression_l1", **common)
        self.export_magnitude = regressor(objective="regression_l1", **common)
        self.minimum_regime_rows = minimum_regime_rows
        self.blend_weight = 0.5
        self.fitted = False

    def fit(self, train_x, train_y, validation_x, validation_y) -> RegimeAwareMixture:
        train_y = np.asarray(train_y, dtype=float)
        validation_y = np.asarray(validation_y, dtype=float)
        export = train_y < 0
        if export.sum() < self.minimum_regime_rows or (~export).sum() < self.minimum_regime_rows:
            raise ValueError("Both import and export regimes need enough training rows")
        self.classifier.fit(train_x, export.astype(int))
        self.direct.fit(train_x, train_y)
        self.import_magnitude.fit(train_x[~export], train_y[~export])
        self.export_magnitude.fit(train_x[export], -train_y[export])
        direct, conditional, _, _, _ = self._components(validation_x)
        candidates = np.linspace(0, 1, 21)
        losses = [np.mean(np.abs(validation_y - (weight * conditional + (1 - weight) * direct))) for weight in candidates]
        self.blend_weight = float(candidates[int(np.argmin(losses))])
        self.fitted = True
        return self

    def refit_components(self, x, y) -> RegimeAwareMixture:
        """Refit component learners while preserving a frozen development blend."""
        y = np.asarray(y, dtype=float)
        export = y < 0
        if export.sum() < self.minimum_regime_rows or (~export).sum() < self.minimum_regime_rows:
            raise ValueError("Both import and export regimes need enough refit rows")
        self.classifier.fit(x, export.astype(int))
        self.direct.fit(x, y)
        self.import_magnitude.fit(x[~export], y[~export])
        self.export_magnitude.fit(x[export], -y[export])
        self.fitted = True
        return self

    def _components(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        direct = np.asarray(self.direct.predict(x), dtype=float)
        probability = np.asarray(self.classifier.predict_proba(x)[:, 1], dtype=float)
        import_magnitude = np.maximum(np.asarray(self.import_magnitude.predict(x), dtype=float), 0)
        export_magnitude = np.maximum(np.asarray(self.export_magnitude.predict(x), dtype=float), 0)
        conditional = (1 - probability) * import_magnitude - probability * export_magnitude
        return direct, conditional, probability, import_magnitude, export_magnitude

    def predict_components(self, x) -> RegimePredictions:
        if not self.fitted:
            raise RuntimeError("Regime-aware mixture has not been fitted")
        direct, conditional, probability, import_magnitude, export_magnitude = self._components(x)
        prediction = self.blend_weight * conditional + (1 - self.blend_weight) * direct
        return RegimePredictions(prediction, direct, conditional, probability, import_magnitude, export_magnitude)

    def predict(self, x) -> np.ndarray:
        return self.predict_components(x).prediction_kw


class QuantileLightGBM:
    """Independent quantile learners with deterministic non-crossing projection."""

    def __init__(self, quantiles: tuple[float, ...], params: dict[str, Any] | None = None) -> None:
        if not quantiles or tuple(sorted(quantiles)) != quantiles or len(set(quantiles)) != len(quantiles):
            raise ValueError("Quantiles must be unique and strictly ordered")
        if any(not 0 < value < 1 for value in quantiles):
            raise ValueError("Quantiles must lie in (0,1)")
        _, regressor = _lightgbm()
        common = DEFAULT_PARAMS | (params or {})
        self.quantiles = quantiles
        self.models = {value: regressor(objective="quantile", alpha=value, **common) for value in quantiles}
        self.fitted = False

    def fit(self, x, y) -> QuantileLightGBM:
        for model in self.models.values():
            model.fit(x, y)
        self.fitted = True
        return self

    def predict_raw(self, x) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Quantile model has not been fitted")
        return np.column_stack([self.models[value].predict(x) for value in self.quantiles])

    def predict(self, x) -> np.ndarray:
        return np.sort(self.predict_raw(x), axis=1)


@dataclass
class ConformalGroup:
    key: tuple[str, ...]
    correction_kw: float
    rows: int


class GroupedIntervalCalibrator:
    """Development-only symmetric CQR correction with global shrinkage fallback."""

    def __init__(self, coverage: float = 0.9, minimum_group_rows: int = 200, shrinkage_rows: float = 500) -> None:
        if not 0 < coverage < 1:
            raise ValueError("Coverage must lie in (0,1)")
        self.coverage = coverage
        self.minimum_group_rows = minimum_group_rows
        self.shrinkage_rows = shrinkage_rows
        self.global_correction = 0.0
        self.groups: dict[tuple[str, ...], ConformalGroup] = {}

    def fit(self, actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, group_keys: list[tuple[str, ...]]) -> GroupedIntervalCalibrator:
        actual = np.asarray(actual, dtype=float); lower = np.asarray(lower, dtype=float); upper = np.asarray(upper, dtype=float)
        if not (len(actual) == len(lower) == len(upper) == len(group_keys)):
            raise ValueError("Calibration arrays and group keys must align")
        score = np.maximum.reduce([lower - actual, actual - upper, np.zeros(len(actual))])
        self.global_correction = self._finite_quantile(score)
        for key in sorted(set(group_keys)):
            mask = np.asarray([value == key for value in group_keys])
            rows = int(mask.sum())
            if rows < self.minimum_group_rows:
                continue
            local = self._finite_quantile(score[mask])
            weight = rows / (rows + self.shrinkage_rows)
            correction = weight * local + (1 - weight) * self.global_correction
            self.groups[key] = ConformalGroup(key, float(correction), rows)
        return self

    def _finite_quantile(self, scores: np.ndarray) -> float:
        rows = len(scores)
        level = min(np.ceil((rows + 1) * self.coverage) / rows, 1.0)
        return float(np.quantile(scores, level, method="higher"))

    def transform(self, lower: np.ndarray, upper: np.ndarray, group_keys: list[tuple[str, ...]]) -> tuple[np.ndarray, np.ndarray]:
        if len(lower) != len(upper) or len(lower) != len(group_keys):
            raise ValueError("Interval arrays and group keys must align")
        correction = np.asarray([self.groups.get(key, ConformalGroup(key, self.global_correction, 0)).correction_kw for key in group_keys])
        return np.asarray(lower) - correction, np.asarray(upper) + correction
