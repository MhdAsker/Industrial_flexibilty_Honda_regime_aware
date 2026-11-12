from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from honda.prescriptive_ml.cli import audit
from honda.prescriptive_ml.evaluation import (
    PromotionGates,
    evaluate_point_forecast,
    interval_reliability,
    paired_day_bootstrap,
    promotion_report,
    weighted_interval_score,
)
from honda.prescriptive_ml.forecasting import (
    GroupedIntervalCalibrator,
    QuantileLightGBM,
    RegimeAwareMixture,
)
from honda.prescriptive_ml.rolling import (
    RollingPlan,
    RollingVintage,
    compare_rolling_policies,
    cvar,
    information_value_metrics,
    simulate_execute_first,
)
from honda.prescriptive_ml.scenarios import (
    conditional_coherent_scenarios,
    empirical_copula_coupling,
    energy_score,
    reduce_scenarios,
)

ROOT = Path(__file__).resolve().parents[2]


def test_regime_aware_mixture_fits_soft_gate_on_development_only():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(600, 4))
    y = 30 * x[:, 0] + rng.normal(0, 4, 600)
    model = RegimeAwareMixture(
        {"n_estimators": 25, "min_child_samples": 5, "n_jobs": 1}, minimum_regime_rows=20,
    )
    model.fit(x[:400], y[:400], x[400:500], y[400:500])
    result = model.predict_components(x[500:])
    assert result.prediction_kw.shape == (100,)
    assert np.all((result.export_probability >= 0) & (result.export_probability <= 1))
    assert 0 <= model.blend_weight <= 1


def test_quantile_lightgbm_projection_is_non_crossing():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 3)); y = x[:, 0] + rng.normal(size=300)
    model = QuantileLightGBM((.1, .5, .9), {"n_estimators": 20, "min_child_samples": 5, "n_jobs": 1})
    prediction = model.fit(x[:250], y[:250]).predict(x[250:])
    assert prediction.shape == (50, 3)
    assert np.all(np.diff(prediction, axis=1) >= 0)


def test_grouped_conformal_interval_uses_only_fitted_calibration_scores():
    actual = np.arange(300, dtype=float)
    lower = actual - .2; upper = actual + .2
    lower[::5] += 2; upper[::5] += 2
    groups = [("H1",)] * 150 + [("H2",)] * 150
    calibrator = GroupedIntervalCalibrator(.9, minimum_group_rows=100, shrinkage_rows=50)
    calibrator.fit(actual, lower, upper, groups)
    adjusted_lower, adjusted_upper = calibrator.transform(lower, upper, groups)
    assert np.all(adjusted_lower <= lower)
    assert np.all(adjusted_upper >= upper)
    assert set(calibrator.groups) == {("H1",), ("H2",)}


def test_wis_and_reliability_are_finite_and_directional():
    actual = np.asarray([0.0, 1.0, 2.0])
    quantiles = (.1, .5, .9)
    good = np.asarray([[-1, 0, 1], [0, 1, 2], [1, 2, 3]], dtype=float)
    bad = good + 5
    assert weighted_interval_score(actual, good, quantiles) < weighted_interval_score(actual, bad, quantiles)
    reliability = interval_reliability(actual, good[:, 0], good[:, 2], .8)
    assert reliability["empirical_coverage"] == 1


def test_conditional_coherent_scenarios_reject_future_donor_and_support_ecc():
    point = np.zeros(4)
    donor_features = np.asarray([[0, 0], [1, 0], [3, 2]], dtype=float)
    residuals = np.asarray([[1, 2, 3, 4], [-1, -2, -3, -4], [4, 1, -2, -3]], dtype=float)
    available = ("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00", "2025-01-03T00:00:00+00:00")
    marginals = np.tile(np.arange(12, dtype=float).reshape(3, 4), (2, 1))
    result = conditional_coherent_scenarios(
        point, np.asarray([0, 0]), donor_features, residuals, available,
        "2025-01-04T00:00:00+00:00", 6, marginal_samples_kw=marginals,
        minimum_effective_donors=2,
    )
    assert result.scenarios.shape == (6, 4)
    assert np.isclose(result.weights.sum(), 1)
    with pytest.raises(ValueError, match="future information"):
        conditional_coherent_scenarios(
            point, np.asarray([0, 0]), donor_features, residuals,
            available[:-1] + ("2025-01-05T00:00:00+00:00",),
            "2025-01-04T00:00:00+00:00", 4,
        )


def test_ecc_preserves_each_marginal_and_template_ranks():
    marginal = np.asarray([[1, 10], [2, 20], [3, 30]], dtype=float)
    template = np.asarray([[9, 1], [3, 3], [6, 2]], dtype=float)
    coupled = empirical_copula_coupling(marginal, template)
    assert np.array_equal(np.sort(coupled, axis=0), np.sort(marginal, axis=0))
    for column in range(2):
        assert np.array_equal(np.argsort(coupled[:, column]), np.argsort(template[:, column]))


def test_scenario_reduction_conserves_probability_and_has_finite_score():
    rng = np.random.default_rng(8); scenarios = rng.normal(size=(20, 12)); actual = rng.normal(size=12)
    reduced = reduce_scenarios(scenarios, 5)
    assert reduced.scenarios.shape == (5, 12)
    assert np.isclose(reduced.weights.sum(), 1)
    assert reduced.mean_assignment_distance >= 0
    assert np.isfinite(energy_score(actual, reduced.scenarios, reduced.weights))


def test_execute_first_carries_state_and_never_exposes_realized_future_to_policy():
    stamps = tuple(pd.date_range("2026-01-02", periods=6, freq="h", tz="UTC").astype(str))
    vintages = tuple(
        RollingVintage(
            issue_time_utc=str(pd.Timestamp(stamps[offset]) - pd.Timedelta(minutes=1)),
            timestamps_utc=stamps[offset:], interval_hours=tuple(1.0 for _ in stamps[offset:]),
            information={"known_offset": offset},
        )
        for offset in (0, 2, 4)
    )
    observed_information = []
    def policy(vintage, initial_state):
        observed_information.append((vintage.information, initial_state))
        return RollingPlan(tuple(1.0 for _ in vintage.timestamps_utc), 1, 2, .01, True, {})
    result = simulate_execute_first(vintages, stamps, tuple(1.0 for _ in stamps), policy, initial_state_kwh=0, execute_intervals=2, state_min_kwh=0, state_max_kwh=10)
    assert result.missed_intervals == 0
    assert result.state_kwh[-1] == 6
    assert [value[1] for value in observed_information] == [0, 2, 4]


def test_cvar_and_information_value_metrics_have_expected_signs():
    assert cvar(np.asarray([1, 2, 10]), np.asarray([.8, .1, .1]), .8) >= 5
    values = information_value_metrics(8, 10, 6, 9, 5)
    assert values == {
        "value_of_stochastic_solution_eur": 2,
        "expected_value_of_perfect_information_eur": 2,
        "recourse_value_eur": 1,
        "oracle_regret_eur": 3,
    }


def test_rolling_policy_comparison_uses_common_vintages_and_settlement():
    stamps = tuple(pd.date_range("2026-01-02", periods=2, freq="h", tz="UTC").astype(str))
    vintage = RollingVintage(
        issue_time_utc=str(pd.Timestamp(stamps[0]) - pd.Timedelta(minutes=1)),
        timestamps_utc=stamps,
        interval_hours=(1.0, 1.0),
        information={},
    )

    def policy(value):
        return lambda _vintage, _state: RollingPlan((value, value), value, value, .01, True, {})

    compared = compare_rolling_policies(
        {"deterministic": policy(1.0), "stochastic": policy(2.0)},
        (vintage,), stamps, (1.0, 1.0),
        lambda result: {"energy_kwh": np.nansum(result.executed_schedule_kw)},
        initial_state_kwh=0, execute_intervals=2, state_min_kwh=0, state_max_kwh=10,
    )
    assert [item.policy for item in compared] == ["deterministic", "stochastic"]
    assert [item.settlement["energy_kwh"] for item in compared] == [2.0, 4.0]


def test_promotion_is_fail_closed_when_probabilistic_metrics_are_missing():
    frame = pd.DataFrame({
        "target_day_local": ["A", "A", "B", "B"], "actual_kw": [1, -1, 2, -2],
        "champion": [1, 1, 2, 2], "challenger": [1, -1, 2, -2],
    })
    metrics = evaluate_point_forecast(frame.actual_kw.to_numpy(), frame.challenger.to_numpy())
    bootstrap = paired_day_bootstrap(frame, "challenger", "champion", samples=100)
    report = promotion_report(metrics, bootstrap, gates=PromotionGates(maximum_mae_kw=1, minimum_export_recall=.5))
    assert report["status"] == "DO_NOT_PROMOTE"
    assert report["required_metrics_complete"] is False


def test_repository_audit_reports_exposed_holdout_and_missing_training_sources():
    result = audit()
    assert result["champion_metrics"]["mae_kw"] == pytest.approx(71.51254282)
    assert isinstance(result["training_readiness"]["ready"], bool)
    assert result["claim_boundary"] == "EXPLORATORY_EXPOSED_2023_HOLDOUT_NO_PROMOTION_CLAIM"


def test_frozen_exploratory_results_remain_fail_closed():
    experiment = json.loads((ROOT / "results/prescriptive_ml/experiment_summary.json").read_text(encoding="utf-8"))
    assert experiment["selected_candidate"] == "REGIME_MIXTURE"
    assert all(value["status"] == "DO_NOT_PROMOTE" for value in experiment["promotion_reports"].values())
    scenarios = json.loads((ROOT / "results/prescriptive_ml/scenario_experiment_summary.json").read_text(encoding="utf-8"))
    raw = scenarios["comparison"]["CONDITIONAL_ECC_RAW"]
    assert raw["wis_kw"]["relative_change"] < 0
    assert raw["energy_score_kw"]["relative_change"] < 0
    assert raw["variogram_score"]["relative_change"] < 0
    assert scenarios["comparison"]["CONDITIONAL_ECC_REDUCED"]["coverage_90"]["value"] < .8
    selection = json.loads((ROOT / "results/prescriptive_ml/scenario_development_selection.json").read_text(encoding="utf-8"))
    assert selection["test_period_used_for_selection"] is False
    assert selection["selected_bandwidth"] == 2.0
    assert selection["selected_scenario_count"] == 200
    assert selection["reduction_selected"] is False
