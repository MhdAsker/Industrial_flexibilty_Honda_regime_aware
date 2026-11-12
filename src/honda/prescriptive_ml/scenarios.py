from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd


def _rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join([str(seed), *(str(value) for value in parts)]).encode()
    return np.random.default_rng(int.from_bytes(sha256(payload).digest()[:8], "little"))


@dataclass
class ConditionalScenarioResult:
    scenarios: np.ndarray
    weights: np.ndarray
    donor_indices: np.ndarray
    effective_donor_count: float
    metadata: dict[str, object]


def conditional_donor_weights(
    target_features: np.ndarray,
    donor_features: np.ndarray,
    *,
    bandwidth: float = 1.0,
    minimum_effective_donors: float = 10.0,
) -> tuple[np.ndarray, float]:
    target = np.asarray(target_features, dtype=float)
    donors = np.asarray(donor_features, dtype=float)
    if donors.ndim != 2 or target.shape != (donors.shape[1],):
        raise ValueError("Target and donor feature dimensions do not align")
    if len(donors) < 2 or bandwidth <= 0:
        raise ValueError("Conditional weighting needs donors and positive bandwidth")
    center = np.nanmedian(donors, axis=0)
    scale = np.nanmedian(np.abs(donors - center), axis=0) * 1.4826
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    standardized = np.nan_to_num((donors - target) / scale, nan=0.0, posinf=10.0, neginf=-10.0)
    distance = np.mean(standardized**2, axis=1)
    current_bandwidth = bandwidth
    for _ in range(20):
        raw = np.exp(-0.5 * distance / current_bandwidth**2)
        weights = raw / raw.sum()
        effective = float(1 / np.sum(weights**2))
        if effective >= min(minimum_effective_donors, len(donors)):
            return weights, effective
        current_bandwidth *= 1.25
    return weights, effective


def empirical_copula_coupling(marginal_samples: np.ndarray, dependence_template: np.ndarray) -> np.ndarray:
    marginals = np.asarray(marginal_samples, dtype=float)
    template = np.asarray(dependence_template, dtype=float)
    if marginals.shape != template.shape or marginals.ndim != 2:
        raise ValueError("ECC samples and dependence template must share a 2D shape")
    result = np.empty_like(marginals)
    for interval in range(marginals.shape[1]):
        order = np.argsort(template[:, interval], kind="mergesort")
        sorted_values = np.sort(marginals[:, interval])
        result[order, interval] = sorted_values
    return result


def conditional_coherent_scenarios(
    point_forecast_kw: np.ndarray,
    target_features: np.ndarray,
    donor_features: np.ndarray,
    donor_residual_paths_kw: np.ndarray,
    donor_available_at_utc: tuple[str, ...],
    target_issue_time_utc: str,
    count: int,
    *,
    marginal_samples_kw: np.ndarray | None = None,
    seed: int = 20261029,
    target_id: str = "target",
    bandwidth: float = 1.0,
    minimum_effective_donors: float = 10.0,
) -> ConditionalScenarioResult:
    point = np.asarray(point_forecast_kw, dtype=float)
    residuals = np.asarray(donor_residual_paths_kw, dtype=float)
    if residuals.ndim != 2 or residuals.shape[1] != len(point):
        raise ValueError("Donor residual paths do not match forecast horizon")
    if len(donor_features) != len(residuals) or count < 1:
        raise ValueError("Donor features and residual paths must align")
    issue = pd.Timestamp(target_issue_time_utc)
    availability = pd.DatetimeIndex(pd.to_datetime(donor_available_at_utc, utc=True))
    if issue.tzinfo is None or len(availability) != len(residuals):
        raise ValueError("Donor availability timestamps must align and be timezone-aware")
    if np.any(availability > issue):
        raise ValueError("Conditional donor set contains future information")
    weights, effective = conditional_donor_weights(
        target_features, donor_features, bandwidth=bandwidth, minimum_effective_donors=minimum_effective_donors,
    )
    rng = _rng(seed, target_id, count, len(residuals))
    donor_indices = rng.choice(len(residuals), size=count, replace=True, p=weights)
    selected_residuals = residuals[donor_indices]
    if marginal_samples_kw is None:
        paths = point[None, :] + selected_residuals
        method = "weighted_residual_trajectory"
    else:
        marginals = np.asarray(marginal_samples_kw, dtype=float)
        if marginals.shape != (count, len(point)):
            raise ValueError("Marginal samples must match requested scenario shape")
        paths = empirical_copula_coupling(marginals, selected_residuals)
        method = "calibrated_marginals_ecc"
    return ConditionalScenarioResult(
        scenarios=paths,
        weights=np.full(count, 1 / count),
        donor_indices=donor_indices,
        effective_donor_count=effective,
        metadata={
            "method": method,
            "target_id": target_id,
            "raw_donor_count": len(residuals),
            "effective_donor_count": effective,
            "bandwidth": bandwidth,
            "target_issue_time_utc": issue.isoformat(),
            "latest_donor_available_at_utc": availability.max().isoformat(),
        },
    )


@dataclass
class ScenarioReduction:
    scenarios: np.ndarray
    weights: np.ndarray
    selected_indices: np.ndarray
    assignment: np.ndarray
    mean_assignment_distance: float


def reduce_scenarios(scenarios: np.ndarray, target_count: int, source_weights: np.ndarray | None = None) -> ScenarioReduction:
    matrix = np.asarray(scenarios, dtype=float)
    if matrix.ndim != 2 or not 1 <= target_count <= len(matrix):
        raise ValueError("Scenario reduction dimensions are invalid")
    weights = np.full(len(matrix), 1 / len(matrix)) if source_weights is None else np.asarray(source_weights, dtype=float)
    if len(weights) != len(matrix) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Source scenario weights must be nonnegative and sum to one")
    distance = np.mean(np.abs(matrix[:, None, :] - matrix[None, :, :]), axis=2)
    selected = [int(np.argmin(distance @ weights))]
    nearest = distance[:, selected[0]].copy()
    while len(selected) < target_count:
        best_index = -1; best_cost = float("inf")
        for candidate in range(len(matrix)):
            if candidate in selected:
                continue
            cost = float(np.dot(weights, np.minimum(nearest, distance[:, candidate])))
            if cost < best_cost - 1e-15:
                best_cost = cost; best_index = candidate
        selected.append(best_index)
        nearest = np.minimum(nearest, distance[:, best_index])
    selected_array = np.asarray(selected, dtype=int)
    assignment_local = np.argmin(distance[:, selected_array], axis=1)
    reduced_weights = np.asarray([weights[assignment_local == index].sum() for index in range(target_count)])
    return ScenarioReduction(
        scenarios=matrix[selected_array], weights=reduced_weights, selected_indices=selected_array,
        assignment=selected_array[assignment_local], mean_assignment_distance=float(np.dot(weights, nearest)),
    )


def energy_score(actual: np.ndarray, scenarios: np.ndarray, weights: np.ndarray | None = None) -> float:
    actual = np.asarray(actual, dtype=float); scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[1] != len(actual):
        raise ValueError("Energy-score shapes do not align")
    weights = np.full(len(scenarios), 1 / len(scenarios)) if weights is None else np.asarray(weights, dtype=float)
    first = np.dot(weights, np.linalg.norm(scenarios - actual, axis=1))
    pairwise = np.linalg.norm(scenarios[:, None, :] - scenarios[None, :, :], axis=2)
    second = 0.5 * float(weights @ pairwise @ weights)
    return float(first - second)
