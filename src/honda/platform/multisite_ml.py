from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ledger import TARGET_SEMANTICS

MODEL_HIERARCHY = (
    "LOCAL_LIGHTGBM",
    "POOLED_LIGHTGBM_SITE_ID_NORMALIZED",
    "GLOBAL_PLUS_LOCAL_RESIDUAL_CALIBRATION",
    "HIERARCHICAL_PROBABILISTIC_PARTIAL_POOLING",
    "COMPACT_NEURAL_GLOBAL_SITE_EMBEDDINGS",
    "JOINT_CROSS_SITE_SCENARIOS",
)


@dataclass(frozen=True)
class SiteEvaluation:
    site_id: str
    target_semantics: str
    rows: int
    mae: float
    bias: float
    coverage: float | None = None


def validate_multisite_frame(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> None:
    required = {"site_id", "issue_time_utc", "target_timestamp_utc", "target_semantics", "actual_kw", *feature_columns}
    if missing := required - set(frame):
        raise ValueError(f"Multi-site frame is missing columns: {sorted(missing)}")
    if not set(frame.target_semantics).issubset(TARGET_SEMANTICS):
        raise ValueError("Multi-site frame contains unapproved target semantics")
    for _site, group in frame.groupby("site_id"):
        if group.target_semantics.nunique() != 1:
            raise ValueError("A site cannot mix physical target semantics")
        if group.target_semantics.iloc[0] == "NONNEGATIVE_DEMAND" and (group.actual_kw < 0).any():
            raise ValueError("Nonnegative-demand targets cannot contain signed exchange values")


def rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    minimum_train_origins: int,
    evaluation_origins: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    issues = np.sort(pd.to_datetime(frame.issue_time_utc, utc=True).unique())
    if minimum_train_origins < 1 or evaluation_origins < 1 or len(issues) <= minimum_train_origins:
        raise ValueError("Rolling-origin split configuration is unsupported by the data")
    splits = []
    for start in range(minimum_train_origins, len(issues), evaluation_origins):
        test_issues = issues[start:start + evaluation_origins]
        train = pd.to_datetime(frame.issue_time_utc, utc=True).isin(issues[:start]).to_numpy()
        test = pd.to_datetime(frame.issue_time_utc, utc=True).isin(test_issues).to_numpy()
        if train.any() and test.any():
            splits.append((np.flatnonzero(train), np.flatnonzero(test)))
    return tuple(splits)


def leave_one_site_out_splits(frame: pd.DataFrame) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    splits = []
    for site in sorted(frame.site_id.unique()):
        test = frame.site_id.eq(site).to_numpy()
        semantics = frame.loc[test, "target_semantics"].iloc[0]
        train = (~test) & frame.target_semantics.eq(semantics).to_numpy()
        if train.any():
            splits.append((str(site), np.flatnonzero(train), np.flatnonzero(test)))
    return tuple(splits)


def normalize_features_by_site(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, list[float]]]]:
    train_values = np.empty((len(train), len(feature_columns)))
    evaluation_values = np.empty((len(evaluation), len(feature_columns)))
    state = {}
    global_mean = train[list(feature_columns)].mean().to_numpy(float)
    global_scale = train[list(feature_columns)].std().to_numpy(float)
    global_scale[(~np.isfinite(global_scale)) | (global_scale < 1e-9)] = 1
    for site in sorted(set(train.site_id) | set(evaluation.site_id)):
        train_mask = train.site_id.eq(site).to_numpy(); evaluation_mask = evaluation.site_id.eq(site).to_numpy()
        if train_mask.any():
            mean = train.loc[train_mask, list(feature_columns)].mean().to_numpy(float)
            scale = train.loc[train_mask, list(feature_columns)].std().to_numpy(float)
            scale[(~np.isfinite(scale)) | (scale < 1e-9)] = 1
        else:
            mean, scale = global_mean, global_scale
        train_values[train_mask] = (train.loc[train_mask, list(feature_columns)].to_numpy(float) - mean) / scale
        evaluation_values[evaluation_mask] = (evaluation.loc[evaluation_mask, list(feature_columns)].to_numpy(float) - mean) / scale
        state[str(site)] = {"mean": mean.tolist(), "scale": scale.tolist(), "cold_start": not train_mask.any()}
    return train_values, evaluation_values, state


@dataclass(frozen=True)
class PartialPoolingResidualAdapter:
    global_bias: float
    site_bias: dict[str, float]
    pooling_strength: float

    @classmethod
    def fit(
        cls,
        site_ids: np.ndarray,
        actual_kw: np.ndarray,
        global_prediction_kw: np.ndarray,
        pooling_strength: float = 100.0,
    ) -> PartialPoolingResidualAdapter:
        sites = np.asarray(site_ids, dtype=str); actual = np.asarray(actual_kw, dtype=float)
        prediction = np.asarray(global_prediction_kw, dtype=float)
        if actual.shape != prediction.shape or actual.shape != sites.shape or pooling_strength < 0:
            raise ValueError("Partial-pooling adapter inputs are invalid")
        residual = actual - prediction; global_bias = float(np.mean(residual)); local = {}
        for site in np.unique(sites):
            values = residual[sites == site]; weight = len(values) / (len(values) + pooling_strength)
            local[str(site)] = float(weight * np.mean(values) + (1 - weight) * global_bias)
        return cls(global_bias, local, pooling_strength)

    def predict(self, site_ids: np.ndarray, global_prediction_kw: np.ndarray) -> np.ndarray:
        sites = np.asarray(site_ids, dtype=str); prediction = np.asarray(global_prediction_kw, dtype=float)
        if sites.shape != prediction.shape:
            raise ValueError("Partial-pooling prediction inputs are invalid")
        return prediction + np.asarray([self.site_bias.get(site, self.global_bias) for site in sites])


def evaluate_sites(
    site_ids: np.ndarray,
    semantics: np.ndarray,
    actual_kw: np.ndarray,
    prediction_kw: np.ndarray,
) -> tuple[SiteEvaluation, ...]:
    frame = pd.DataFrame({"site": site_ids, "semantics": semantics, "actual": actual_kw, "prediction": prediction_kw})
    result = []
    for site, group in frame.groupby("site"):
        if group.semantics.nunique() != 1:
            raise ValueError("Site evaluation mixes target semantics")
        residual = group.prediction.to_numpy(float) - group.actual.to_numpy(float)
        result.append(SiteEvaluation(str(site), str(group.semantics.iloc[0]), len(group), float(np.mean(np.abs(residual))), float(np.mean(residual))))
    return tuple(result)


def advanced_model_readiness(
    *,
    comparable_approved_sites: int,
    seasonal_days: int,
    pooled_rows: int,
    minimum_sites: int = 10,
    minimum_days: int = 365,
    minimum_rows: int = 100_000,
) -> dict[str, object]:
    checks = {
        "comparable_sites": comparable_approved_sites >= minimum_sites,
        "seasonal_evidence": seasonal_days >= minimum_days,
        "pooled_sample": pooled_rows >= minimum_rows,
    }
    return {
        "status": "ELIGIBLE_AS_CHALLENGER" if all(checks.values()) else "DEFER_ADVANCED_GENERATIVE_MODELS",
        "checks": checks,
        "models": ("TACTIS", "NORMALIZING_FLOW", "DIFFUSION", "GRAPH_NEURAL_NETWORK"),
        "claim_boundary": "READINESS_GATE_NOT_PERFORMANCE_EVIDENCE",
    }


def hierarchy_execution_plan(comparable_approved_sites: int, seasonal_days: int) -> list[dict[str, object]]:
    classical_ready = comparable_approved_sites >= 3 and seasonal_days >= 365
    return [
        {
            "order": index + 1,
            "model": model,
            "status": (
                "IMPLEMENTATION_READY_DATA_REQUIRED"
                if index < 4 and not classical_ready
                else "ELIGIBLE_FOR_BENCHMARK"
                if index < 4
                else "DEFER_UNTIL_SCALE_GATE"
                if model == "COMPACT_NEURAL_GLOBAL_SITE_EMBEDDINGS"
                else "IMPLEMENTED_EMPIRICAL_CHALLENGER_DATA_REQUIRED"
            ),
        }
        for index, model in enumerate(MODEL_HIERARCHY)
    ]
