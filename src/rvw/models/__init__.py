"""Market risk model suite package."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rvw.config import AppConfig
from rvw.models.base import BaseRiskModel, RiskForecast
from rvw.models.ewma import EWMAModel, compute_ewma_covariance
from rvw.models.garch_fhs import GARCHFilteredHistoricalModel
from rvw.models.historical import HistoricalSimulationModel
from rvw.models.monte_carlo import MonteCarloRiskModel
from rvw.models.parametric_normal import ParametricNormalModel
from rvw.models.parametric_t import ParametricStudentTModel

MODEL_REGISTRY = {
    "historical_simulation": HistoricalSimulationModel,
    "parametric_normal": ParametricNormalModel,
    "parametric_t": ParametricStudentTModel,
    "ewma": EWMAModel,
    "garch_fhs": GARCHFilteredHistoricalModel,
    "monte_carlo": MonteCarloRiskModel,
}


def get_model(name: str, params: Optional[Dict[str, Any]] = None) -> BaseRiskModel:
    """Instantiate a risk model by its registered identifier.

    Args:
        name: Name of model ('historical_simulation', 'parametric_normal', etc.).
        params: Optional dictionary of hyperparameter overrides.

    Returns:
        BaseRiskModel: Instantiated model.

    Raises:
        KeyError: If model name is unknown.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](name=name, params=params)


def get_all_models(config: AppConfig) -> Dict[str, BaseRiskModel]:
    """Instantiate all six models configured with application parameters.

    Args:
        config: Application configuration.

    Returns:
        Dict[str, BaseRiskModel]: Map of model identifiers to model instances.
    """
    return {
        "historical_simulation": HistoricalSimulationModel(
            params=config.models.historical_simulation
        ),
        "parametric_normal": ParametricNormalModel(
            params=config.models.parametric_normal
        ),
        "parametric_t": ParametricStudentTModel(
            params=config.models.parametric_t
        ),
        "ewma": EWMAModel(
            params=config.models.ewma
        ),
        "garch_fhs": GARCHFilteredHistoricalModel(
            params=config.models.garch_fhs
        ),
        "monte_carlo": MonteCarloRiskModel(
            params=config.models.monte_carlo
        ),
    }


__all__ = [
    "MODEL_REGISTRY",
    "BaseRiskModel",
    "EWMAModel",
    "GARCHFilteredHistoricalModel",
    "HistoricalSimulationModel",
    "MonteCarloRiskModel",
    "ParametricNormalModel",
    "ParametricStudentTModel",
    "RiskForecast",
    "compute_ewma_covariance",
    "get_all_models",
    "get_model",
]
