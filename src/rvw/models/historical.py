"""Historical Simulation (HS) market risk model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.models.base import BaseRiskModel, RiskForecast
from rvw.portfolio.portfolio import Portfolio, portfolio_returns


class HistoricalSimulationModel(BaseRiskModel):
    """Historical Simulation empirical distribution model.

    Computes 1-day ahead VaR as the empirical quantile of historical portfolio losses
    L_t = -R_{p, t} using documented linear interpolation. Expected Shortfall is
    computed as the conditional tail expectation:
        ES_alpha = mean(L | L >= VaR_alpha)
    """

    def __init__(self, name: str = "historical_simulation", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, params=params)
        self.interpolation = self.params.get("interpolation", "linear")

    def forecast(
        self,
        returns_window: pd.DataFrame,
        portfolio: Portfolio,
        var_alphas: Optional[List[float]] = None,
        es_alphas: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> RiskForecast:
        """Forecast empirical VaR and tail-mean ES from historical returns."""
        if var_alphas is None:
            var_alphas = [0.95, 0.99]
        if es_alphas is None:
            es_alphas = [0.975, 0.99]

        p_ret = portfolio_returns(returns_window, portfolio)
        # Loss convention: Loss = -Return
        losses = -p_ret.values
        n_obs = len(losses)

        if n_obs < 10:
            raise ValueError(f"Historical simulation requires at least 10 observations (got {n_obs}).")

        var_forecasts: Dict[float, float] = {}
        for alpha in var_alphas:
            # Documented quantile interpolation: numpy percentile with linear method
            var_val = float(np.percentile(losses, alpha * 100.0, method=self.interpolation))
            var_forecasts[alpha] = max(0.0, var_val)

        es_forecasts: Dict[float, float] = {}
        for alpha in es_alphas:
            # Empirical ES: mean of losses in the (1-alpha) tail
            var_threshold = float(np.percentile(losses, alpha * 100.0, method=self.interpolation))
            tail_losses = losses[losses >= var_threshold]
            if len(tail_losses) == 0:
                es_val = var_threshold
            else:
                es_val = float(np.mean(tail_losses))
            es_forecasts[alpha] = max(var_forecasts.get(alpha, 0.0), es_val)

        return RiskForecast(
            model_name=self.name,
            var_forecasts=var_forecasts,
            es_forecasts=es_forecasts,
            metadata={
                "window_observations": n_obs,
                "interpolation": self.interpolation,
                "portfolio_name": portfolio.name,
            },
        )
