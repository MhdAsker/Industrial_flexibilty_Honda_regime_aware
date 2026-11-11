from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from honda.flexlab.adapters import SiteData, SyntheticSiteAdapter
from honda.flexlab.schemas import ConfigError, ExperimentConfig, SiteConfig
from honda.flexlab.workflow import evaluate_experiment, reproduce_experiment, validate_experiment
from honda.scenarios.calibration import fit_global_trajectory_scale, joint_reliability


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/flexlab/synthetic/experiment.yaml"


def test_example_configuration_and_site_contract_are_valid():
    config, data = validate_experiment(EXAMPLE)
    assert config.schema_version == "flexlab.experiment.v2"
    assert len(data.timestamps_utc) == 96
    assert data.coherent_scenarios_kw.shape == (50, 96)
    assert data.independent_scenarios_kw.shape == (50, 96)


@pytest.mark.parametrize("day,expected", [("2025-03-30", 92), ("2025-10-26", 100)])
def test_synthetic_adapter_preserves_dst_physical_days(day, expected):
    site = SiteConfig("DST", "synthetic", day, "demand_nonnegative", sign_convention="demand_positive")
    data = SyntheticSiteAdapter().load(site)
    assert len(data.timestamps_utc) == expected
    assert len(set(data.timestamps_utc)) == expected


def test_duplicate_utc_timestamp_is_rejected():
    site = SiteConfig("X", "synthetic", "2025-01-15", "demand_nonnegative", sign_convention="demand_positive")
    data = SyntheticSiteAdapter().load(site)
    broken = SiteData(data.dataset_id, data.target_day_local, (data.timestamps_utc[0],) + data.timestamps_utc[:-1], data.actual_kw, data.point_forecast_kw, data.independent_scenarios_kw, data.coherent_scenarios_kw, data.scenario_weights, data.provenance)
    with pytest.raises(ValueError, match="Duplicate UTC"):
        broken.validate(site)


def test_unknown_config_field_is_rejected(tmp_path):
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")); value["site"]["mystery_unit"] = "MW"
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown site fields"):
        ExperimentConfig.load(path)


def test_tariff_horizon_mismatch_is_rejected(tmp_path):
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")); value["tariff"]["import_eur_per_mwh"] = [100.0, 120.0]
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ConfigError, match="tariff length"):
        validate_experiment(path)


def test_verified_timing_cannot_precede_publication(tmp_path):
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")); value["timing"]["decision_time_local"] = "12:30"
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="publication"):
        validate_experiment(path)


def test_end_to_end_bundle_is_complete_and_reproducible(tmp_path):
    output = tmp_path / "bundle"
    summary = evaluate_experiment(EXAMPLE, output)
    expected = {"experiment.yaml", "policy_summary.csv", "schedules.csv", "states.csv", "summary.json", "dashboard.json", "manifest.json"}
    assert expected == {x.name for x in output.iterdir()}
    policies = pd.read_csv(output / "policy_summary.csv")
    assert set(policies.policy) == {"deterministic", "independent", "coherent", "calibrated_coherent"}
    assert policies.solver_status.eq("optimal").all()
    assert policies.terminal_residual_kwh.max() < 1e-7
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == "flexlab.manifest.v2"
    assert manifest["unsupported_claims"] == ["CUSTOMER_SAVINGS", "TRADING_PROFIT", "PLANT_VALIDATION"]
    assert summary["timing_classification"] == "OPERATIONAL_TIMING_VERIFIED"
    dashboard = json.loads((output / "dashboard.json").read_text(encoding="utf-8"))
    assert len(dashboard["trajectory"]["timestamp_utc"]) == 96
    assert set(dashboard["trajectory"]["schedules_kw"]) == set(policies.policy)
    assert reproduce_experiment(EXAMPLE)["passed"] is True


def test_calibrated_policy_requires_development_fitted_scale(tmp_path):
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")); value.pop("coherent_calibration_scale")
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ConfigError, match="requires coherent_calibration_scale"):
        ExperimentConfig.load(path)


def test_global_calibration_improves_underdispersed_ensemble():
    rng = np.random.default_rng(7); days, members, horizon = 80, 50, 96
    point = np.zeros((days, horizon)); actual = rng.normal(0, 2.0, size=(days, horizon))
    scenarios = rng.normal(0, .8, size=(days, members, horizon))
    calibration = fit_global_trajectory_scale(actual, point, scenarios)
    assert calibration.scale > 1
    assert calibration.mean_absolute_coverage_error_after < calibration.mean_absolute_coverage_error_before


def test_joint_reliability_reports_required_dimensions():
    rng = np.random.default_rng(11); actual = rng.normal(size=(12, 96)); scenarios = rng.normal(size=(12, 40, 96))
    result = joint_reliability(actual, scenarios)
    assert set(result) == {"nominal_level", "marginal_interval_coverage", "simultaneous_whole_day_coverage", "daily_energy_coverage", "maximum_ramp_upper_coverage"}
    assert all(0 <= value <= 1 for value in result.values())
