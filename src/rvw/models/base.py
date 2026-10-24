"""Base risk model abstraction and standardized forecast data structure."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.portfolio.portfolio import Portfolio


@dataclass
class RiskForecast:
    """Standardized output container for VaR and ES risk forecasts.

    Conventions:
    - Loss convention: VaR and ES are strictly positive numbers representing
      fractional losses of portfolio value. Loss = -Return.
    - Horizon: Exactly 1 trading day.
    """
    model_name: str
    var_forecasts: Dict[float, float]  # e.g. {0.95: 0.015, 0.99: 0.024}
    es_forecasts: Dict[float, float]   # e.g. {0.975: 0.022, 0.99: 0.028}
    metadata: Dict[str, Any] = field(default_factory=dict)
    warning_message: Optional[str] = None
    failed: bool = False

    def get_var(self, alpha: float) -> float:
        """Retrieve VaR at specified confidence level alpha."""
        if alpha not in self.var_forecasts:
            raise KeyError(f"VaR for alpha={alpha} not computed in forecast. Available: {list(self.var_forecasts.keys())}")
        return self.var_forecasts[alpha]

    def get_es(self, alpha: float) -> float:
        """Retrieve Expected Shortfall at specified confidence level alpha."""
        if alpha not in self.es_forecasts:
            raise KeyError(f"ES for alpha={alpha} not computed in forecast. Available: {list(self.es_forecasts.keys())}")
        return self.es_forecasts[alpha]


class BaseRiskModel(ABC):
    """Abstract base class that all market risk forecasting models must implement."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the model with an identifier and parameter dictionary."""
        self.name = name
        self.params = params or {}

    @abstractmethod
    def forecast(
        self,
        returns_window: pd.DataFrame,
        portfolio: Portfolio,
        var_alphas: Optional[List[float]] = None,
        es_alphas: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> RiskForecast:
        """Generate out-of-sample 1-day ahead VaR and ES forecasts.

        Args:
            returns_window: Historical window of asset returns (shape T x N) strictly <= t-1.
            portfolio: Portfolio instance defining asset weights.
            var_alphas: Confidence levels for VaR (defaults to [0.95, 0.99]).
            es_alphas: Confidence levels for ES (defaults to [0.975, 0.99]).
            seed: Optional RNG seed for stochastic models.

        Returns:
            RiskForecast: Standardized risk forecast.
        """
        pass
