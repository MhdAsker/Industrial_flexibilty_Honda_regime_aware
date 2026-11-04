"""Physically explicit industrial-flexibility optimization framework."""

from .contracts import (
    DecisionInput,
    F1Config,
    F2Config,
    F3Config,
    OptimizationResult,
    TariffConfig,
)
from .optimizer import settle_schedule, solve_f1
from .legacy import solve_f0
from .thermal import solve_f2

__all__ = [
    "DecisionInput",
    "F1Config",
    "F2Config",
    "F3Config",
    "OptimizationResult",
    "TariffConfig",
    "settle_schedule",
    "solve_f0",
    "solve_f1",
    "solve_f2",
]
