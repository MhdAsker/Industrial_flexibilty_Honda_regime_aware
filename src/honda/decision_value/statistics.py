from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_cvar(values: np.ndarray, alpha: float = .9) -> float:
    values = np.asarray(values, dtype=float)
    threshold = np.quantile(values, alpha)
    tail = values[values >= threshold]
    return float(np.mean(tail)) if len(tail) else np.nan


def lag1(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def bootstrap_mean(values: np.ndarray, *, resamples: int = 2000, seed: int = 20261029) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    return values[indices].mean(axis=1)


def moving_block_bootstrap(values: np.ndarray, *, block_length: int = 7, resamples: int = 2000, seed: int = 20261029) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values); rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    for i in range(resamples):
        starts = rng.integers(0, n, size=blocks)
        index = ((starts[:, None] + offsets) % n).reshape(-1)[:n]
        draws[i] = values[index].mean()
    return draws


def paired_comparison(
    daily: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    *,
    label: str,
    resamples: int = 2000,
    seed: int = 20261029,
    block_length: int = 7,
) -> dict:
    a = daily[daily.policy.eq(policy_a)][["target_day_local", "realized_cost_eur"]].rename(columns={"realized_cost_eur": "a"})
    b = daily[daily.policy.eq(policy_b)][["target_day_local", "realized_cost_eur"]].rename(columns={"realized_cost_eur": "b"})
    merged = a.merge(b, on="target_day_local", how="inner").sort_values("target_day_local")
    merged = merged.dropna()
    difference = (merged.a - merged.b).to_numpy(float)
    ordinary = bootstrap_mean(difference, resamples=resamples, seed=seed)
    block = moving_block_bootstrap(difference, block_length=block_length, resamples=resamples, seed=seed)
    p = min(1.0, 2 * min(float(np.mean(ordinary <= 0)), float(np.mean(ordinary >= 0))))
    trim = np.quantile(np.abs(difference), .99)
    return {
        "comparison": label, "policy_A": policy_a, "policy_B": policy_b,
        "interpretation": "positive difference means policy B has lower realized cost",
        "common_days": len(difference), "mean_cost_A_minus_B_eur": float(np.mean(difference)),
        "median_cost_A_minus_B_eur": float(np.median(difference)),
        "ci95_lower_eur": float(np.quantile(ordinary, .025)), "ci95_upper_eur": float(np.quantile(ordinary, .975)),
        "block_ci95_lower_eur": float(np.quantile(block, .025)), "block_ci95_upper_eur": float(np.quantile(block, .975)),
        "policy_B_daily_win_fraction": float(np.mean(difference > 0)), "lag1_daily_difference": lag1(difference),
        "mean_without_top1pct_abs_outliers_eur": float(np.mean(difference[np.abs(difference) <= trim])),
        "two_sided_bootstrap_p": p, "bootstrap_resamples": resamples, "block_length": block_length, "seed": seed,
    }


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    order = np.argsort(frame.two_sided_bootstrap_p.to_numpy(float))
    adjusted = np.empty(len(frame)); running = 0.0
    m = len(frame)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(frame.iloc[index].two_sided_bootstrap_p))
        running = max(running, value); adjusted[index] = running
    frame["holm_adjusted_p"] = adjusted
    return frame


def summarize_policies(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in daily.groupby("policy", sort=True):
        valid = group[group.solver_status.isin(["optimal", "not_required"]) & group.realized_cost_eur.notna()]
        costs = valid.realized_cost_eur.to_numpy(float); savings = valid.savings_eur.to_numpy(float)
        rows.append({
            "policy": policy, "available_days": group.target_day_local.nunique(), "valid_days": len(valid),
            "infeasible_or_failed_days": int(len(group) - len(valid)), "mean_realized_cost_eur_day": float(np.mean(costs)) if len(costs) else np.nan,
            "mean_savings_eur_day": float(np.mean(savings)) if len(savings) else np.nan,
            "median_savings_eur_day": float(np.median(savings)) if len(savings) else np.nan,
            "p10_savings_eur_day": float(np.quantile(savings, .1)) if len(savings) else np.nan,
            "p90_savings_eur_day": float(np.quantile(savings, .9)) if len(savings) else np.nan,
            "negative_savings_fraction": float(np.mean(savings < 0)) if len(savings) else np.nan,
            "p90_realized_cost_eur": float(np.quantile(costs, .9)) if len(costs) else np.nan,
            "p95_realized_cost_eur": float(np.quantile(costs, .95)) if len(costs) else np.nan,
            "realized_cvar90_eur": empirical_cvar(costs, .9) if len(costs) else np.nan,
            "realized_cvar95_eur": empirical_cvar(costs, .95) if len(costs) else np.nan,
            "mean_peak_import_kw": float(valid.peak_import_kw.mean()) if len(valid) else np.nan,
            "mean_shifted_energy_kwh": float(valid.shifted_energy_kwh.mean()) if len(valid) else np.nan,
            "mean_backlog_utilization": float(valid.backlog_utilization.mean()) if len(valid) else np.nan,
            "realized_import_violation_days": int((valid.import_violation_intervals > 0).sum()) if len(valid) else 0,
            "realized_export_violation_days": int((valid.export_violation_intervals > 0).sum()) if len(valid) else 0,
            "median_runtime_seconds": float(valid.runtime_seconds.median()) if len(valid) else np.nan,
            "p95_runtime_seconds": float(valid.runtime_seconds.quantile(.95)) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)
