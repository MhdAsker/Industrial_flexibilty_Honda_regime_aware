from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """A human-readable experiment configuration error."""


def _only(data: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown {section} fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class SiteConfig:
    dataset_id: str
    adapter: str
    target_day_local: str
    target_semantics: str
    unit: str = "kW"
    sign_convention: str = "import_positive"
    timezone: str = "Europe/Berlin"
    interval_minutes: int = 15
    seed: int = 20261029

    def validate(self) -> None:
        if self.adapter not in {"synthetic", "honda", "spark"}:
            raise ConfigError("site.adapter must be synthetic, honda, or spark")
        if self.target_semantics not in {"demand_nonnegative", "signed_grid_exchange"}:
            raise ConfigError("Unsupported target semantics")
        if self.unit != "kW" or self.interval_minutes != 15:
            raise ConfigError("FlexLab v2 currently requires kW at 15-minute resolution")
        if self.sign_convention not in {"import_positive", "demand_positive"}:
            raise ConfigError("Unsupported sign convention")
        if self.target_semantics == "signed_grid_exchange" and self.sign_convention != "import_positive":
            raise ConfigError("Signed grid exchange must use import_positive")
        if self.target_semantics == "demand_nonnegative" and self.sign_convention != "demand_positive":
            raise ConfigError("Nonnegative demand must use demand_positive")


@dataclass(frozen=True)
class AssetConfig:
    p_up_kw: float
    p_down_kw: float
    e_advance_kwh: float
    e_backlog_kwh: float
    x_initial_kwh: float = 0.0
    x_terminal_kwh: float = 0.0
    ramp_limit_kw_per_interval: float | None = None
    label: str = "SIMULATED_DEFERRABLE_PROCESS"

    def validate(self) -> None:
        values = (self.p_up_kw, self.p_down_kw, self.e_advance_kwh, self.e_backlog_kwh)
        if min(values) < 0:
            raise ConfigError("Asset power and energy capacities must be nonnegative")
        if self.ramp_limit_kw_per_interval is not None and self.ramp_limit_kw_per_interval < 0:
            raise ConfigError("Asset ramp limit must be nonnegative")


@dataclass(frozen=True)
class GridConfig:
    mode: str = "scenario_wise"
    import_max_kw: float | None = None
    export_max_kw: float | None = 0.0
    chance_quantile: float = 0.95
    soft_violation_penalty_eur_per_kw_interval: float = 0.0
    peak_penalty_eur_per_kw_per_day: float = 0.0

    def validate(self) -> None:
        if self.mode not in {"nominal", "scenario_wise", "chance_approximation", "soft"}:
            raise ConfigError("Unsupported grid constraint mode")
        if not 0 < self.chance_quantile < 1:
            raise ConfigError("chance_quantile must lie in (0, 1)")
        for name in ("import_max_kw", "export_max_kw"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ConfigError(f"{name} must be nonnegative")
        if min(self.soft_violation_penalty_eur_per_kw_interval, self.peak_penalty_eur_per_kw_per_day) < 0:
            raise ConfigError("Grid penalty coefficients must be nonnegative")


@dataclass(frozen=True)
class TariffConfigV2:
    import_eur_per_mwh: float | list[float]
    export_eur_per_mwh: float | list[float] = 0.0
    label: str = "SYNTHETIC_COUNTERFACTUAL_TARIFF"

    @staticmethod
    def _vector(value: float | list[float], horizon: int, name: str) -> list[float]:
        vector = [float(value)] * horizon if isinstance(value, (int, float)) else [float(x) for x in value]
        if len(vector) != horizon:
            raise ConfigError(f"{name} tariff length {len(vector)} does not match horizon {horizon}")
        return vector

    def vectors(self, horizon: int) -> tuple[list[float], list[float]]:
        buy = self._vector(self.import_eur_per_mwh, horizon, "Import")
        sell = self._vector(self.export_eur_per_mwh, horizon, "Export")
        if any(b < s for b, s in zip(buy, sell)):
            raise ConfigError("Import tariff must dominate export remuneration")
        return buy, sell


@dataclass(frozen=True)
class TimingConfig:
    forecast_issue_local: str
    decision_time_local: str
    classification: str
    price_publication_local: str | None = None

    def validate(self) -> None:
        allowed = {"OPERATIONAL_TIMING_VERIFIED", "EX_POST_PRICE_AWARE", "UNSUPPORTED_TIMING"}
        if self.classification not in allowed:
            raise ConfigError("Unknown timing classification")
        if self.classification == "OPERATIONAL_TIMING_VERIFIED" and self.price_publication_local is None:
            raise ConfigError("Verified operational timing requires a price publication time")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    site: SiteConfig
    asset: AssetConfig
    grid: GridConfig
    tariff: TariffConfigV2
    timing: TimingConfig
    policies: tuple[str, ...] = ("deterministic", "independent", "coherent")
    coherent_calibration_scale: float | None = None
    risk_lambda: float = 0.0
    cvar_alpha: float = 0.95
    schema_version: str = "flexlab.experiment.v2"
    supported_claims: tuple[str, ...] = ("SYNTHETIC_COUNTERFACTUAL",)
    unsupported_claims: tuple[str, ...] = ("CUSTOMER_SAVINGS", "PLANT_VALIDATION")
    source_path: Path | None = field(default=None, compare=False, repr=False)

    def validate(self) -> None:
        if self.schema_version != "flexlab.experiment.v2":
            raise ConfigError("Unsupported experiment schema version")
        if not self.experiment_id.strip():
            raise ConfigError("experiment_id cannot be empty")
        self.site.validate(); self.asset.validate(); self.grid.validate(); self.timing.validate()
        allowed = {"deterministic", "independent", "coherent", "calibrated_coherent"}
        if not self.policies or not set(self.policies).issubset(allowed):
            raise ConfigError("Unknown policy in experiment configuration")
        if "calibrated_coherent" in self.policies and self.coherent_calibration_scale is None:
            raise ConfigError("calibrated_coherent requires coherent_calibration_scale fitted on development data")
        if self.coherent_calibration_scale is not None and self.coherent_calibration_scale <= 0:
            raise ConfigError("coherent_calibration_scale must be positive")
        if self.risk_lambda < 0 or not 0 < self.cvar_alpha < 1:
            raise ConfigError("Invalid CVaR settings")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("source_path", None)
        return value

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigError("Experiment YAML must contain a mapping")
        _only(raw, {"schema_version", "experiment_id", "site", "asset", "grid", "tariff", "timing", "policies", "coherent_calibration_scale", "risk_lambda", "cvar_alpha", "supported_claims", "unsupported_claims"}, "experiment")
        section_types = {"site": SiteConfig, "asset": AssetConfig, "grid": GridConfig, "tariff": TariffConfigV2, "timing": TimingConfig}
        sections: dict[str, Any] = {}
        for name, kind in section_types.items():
            data = raw.get(name)
            if not isinstance(data, dict):
                raise ConfigError(f"Missing or invalid {name} section")
            allowed = set(kind.__dataclass_fields__)
            _only(data, allowed, name)
            try:
                sections[name] = kind(**data)
            except TypeError as exc:
                raise ConfigError(f"Invalid {name} section: {exc}") from exc
        config = cls(
            experiment_id=str(raw.get("experiment_id", "")),
            schema_version=str(raw.get("schema_version", "flexlab.experiment.v2")),
            policies=tuple(raw.get("policies", ("deterministic", "independent", "coherent"))),
            coherent_calibration_scale=None if raw.get("coherent_calibration_scale") is None else float(raw["coherent_calibration_scale"]),
            risk_lambda=float(raw.get("risk_lambda", 0.0)), cvar_alpha=float(raw.get("cvar_alpha", 0.95)),
            supported_claims=tuple(raw.get("supported_claims", ("SYNTHETIC_COUNTERFACTUAL",))),
            unsupported_claims=tuple(raw.get("unsupported_claims", ("CUSTOMER_SAVINGS", "PLANT_VALIDATION"))),
            source_path=source, **sections,
        )
        config.validate()
        return config
