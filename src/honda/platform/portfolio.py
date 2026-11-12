from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class PortfolioSite:
    site_id: str
    response_capacity_kw: float
    marginal_cost_eur_per_kw: float
    network_group: str

    def validate(self) -> None:
        if not self.site_id or not self.network_group:
            raise ValueError("Portfolio site identity and network group are required")
        if self.response_capacity_kw < 0 or self.marginal_cost_eur_per_kw < 0:
            raise ValueError("Portfolio capacity and marginal cost must be nonnegative")


@dataclass
class PortfolioAllocation:
    status: str
    allocation_kw: dict[str, float]
    total_cost_eur: float | None
    portfolio_variance: float | None
    risk_budget: float | None
    diversification_value: float | None
    binding_network_groups: tuple[str, ...]


def allocate_portfolio(
    sites: tuple[PortfolioSite, ...],
    target_response_kw: float,
    covariance: np.ndarray,
    network_group_limits_kw: dict[str, float],
    risk_aversion: float = 0.0,
    risk_budget: float | None = None,
) -> PortfolioAllocation:
    if not sites or target_response_kw < 0 or risk_aversion < 0:
        raise ValueError("Portfolio sites, target, and risk aversion are invalid")
    for site in sites:
        site.validate()
    if len({site.site_id for site in sites}) != len(sites):
        raise ValueError("Portfolio site IDs must be unique")
    covariance = np.asarray(covariance, dtype=float)
    count = len(sites)
    if covariance.shape != (count, count) or not np.allclose(covariance, covariance.T):
        raise ValueError("Portfolio covariance must be a symmetric site-by-site matrix")
    if np.min(np.linalg.eigvalsh(covariance)) < -1e-8:
        raise ValueError("Portfolio covariance must be positive semidefinite")
    if risk_budget is not None and risk_budget < 0:
        raise ValueError("Risk budget must be nonnegative")
    capacities = np.asarray([site.response_capacity_kw for site in sites])
    costs = np.asarray([site.marginal_cost_eur_per_kw for site in sites])
    if target_response_kw > capacities.sum() + 1e-8:
        return PortfolioAllocation("infeasible capacity", {}, None, None, risk_budget, None, ())
    constraints: list[dict] = [{"type": "eq", "fun": lambda x: np.sum(x) - target_response_kw}]
    for group, limit in network_group_limits_kw.items():
        if limit < 0:
            raise ValueError("Network-group limits must be nonnegative")
        mask = np.asarray([site.network_group == group for site in sites], dtype=float)
        constraints.append({"type": "ineq", "fun": lambda x, mask=mask, limit=limit: limit - np.dot(mask, x)})
    if risk_budget is not None:
        constraints.append({"type": "ineq", "fun": lambda x: risk_budget - float(x @ covariance @ x)})
    initial = capacities * (target_response_kw / capacities.sum()) if capacities.sum() else np.zeros(count)
    # Project a feasible equality start greedily; SLSQP then handles group/risk constraints.
    initial = np.minimum(initial, capacities)
    remainder = target_response_kw - initial.sum()
    for index in np.argsort(costs):
        addition = min(remainder, capacities[index] - initial[index])
        initial[index] += addition; remainder -= addition
    objective = lambda x: float(np.dot(costs, x) + risk_aversion * (x @ covariance @ x))
    solved = minimize(objective, initial, method="SLSQP", bounds=[(0, value) for value in capacities], constraints=constraints, options={"ftol": 1e-10, "maxiter": 1000})
    if not solved.success:
        return PortfolioAllocation(solved.message, {}, None, None, risk_budget, None, ())
    allocation = np.maximum(solved.x, 0)
    variance = float(allocation @ covariance @ allocation)
    standalone_variance = float(np.dot(np.diag(covariance), allocation**2))
    binding = tuple(sorted(
        group for group, limit in network_group_limits_kw.items()
        if np.isclose(sum(allocation[i] for i, site in enumerate(sites) if site.network_group == group), limit, atol=1e-6)
    ))
    return PortfolioAllocation(
        "optimal", {site.site_id: float(allocation[i]) for i, site in enumerate(sites)},
        float(np.dot(costs, allocation)), variance, risk_budget,
        standalone_variance - variance, binding,
    )
