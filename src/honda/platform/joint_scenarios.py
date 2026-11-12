from __future__ import annotations

import numpy as np

from .ledger import TARGET_SEMANTICS


def joint_empirical_copula(
    marginal_samples_kw: dict[str, np.ndarray],
    dependence_templates: dict[str, np.ndarray],
    target_semantics: dict[str, str],
) -> dict[str, np.ndarray]:
    """Couple site-specific marginals without pooling their physical targets."""
    sites = set(marginal_samples_kw)
    if not sites or sites != set(dependence_templates) or sites != set(target_semantics):
        raise ValueError("Joint scenario sites do not align")
    if not set(target_semantics.values()).issubset(TARGET_SEMANTICS):
        raise ValueError("Joint scenarios contain unsupported target semantics")
    shapes = {np.asarray(marginal_samples_kw[site]).shape for site in sites}
    template_shapes = {np.asarray(dependence_templates[site]).shape for site in sites}
    if len(shapes) != 1 or shapes != template_shapes:
        raise ValueError("Joint scenario marginal and template shapes must align")
    scenario_count, horizon = next(iter(shapes))
    if scenario_count < 2 or horizon < 1:
        raise ValueError("Joint scenario dimensions are invalid")
    result = {}
    for site in sorted(sites):
        marginal = np.asarray(marginal_samples_kw[site], dtype=float)
        template = np.asarray(dependence_templates[site], dtype=float)
        coupled = np.empty_like(marginal)
        for interval in range(horizon):
            ranks = np.argsort(np.argsort(template[:, interval], kind="mergesort"), kind="mergesort")
            coupled[:, interval] = np.sort(marginal[:, interval])[ranks]
        if target_semantics[site] == "NONNEGATIVE_DEMAND" and np.any(coupled < 0):
            raise ValueError("Nonnegative-demand site received negative marginal scenarios")
        result[site] = coupled
    return result
