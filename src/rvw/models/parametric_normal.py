"""Parametric Normal (variance-covariance) market risk model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from rvw.models.base import BaseRiskModel, RiskForecast
from rvw.portfolio.portfolio import Portfolio, portfolio_returns


class ParametricNormalModel(BaseRiskModel):
    """Variance-covariance Parametric Normal model.

    Assumes portfolio returns follow a Gaussian distribution N(mu, sigma^2).
    VaR_alpha = -mu + sigma * z_alpha
    ES_alpha = -mu + sigma * phi(z_alpha) / (1 - alpha)
    where z_alpha = Phi^{-1}(alpha) is the standard normal quantile.
    """

    def __init__(self, name: str = "parametric_normal", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, params=params)
        self.assume_zero_mean = self.params.get("assume_zero_mean", False)

    def forecast(
        self,
        returns_window: pd.DataFrame,
        portfolio: Portfolio,
        var_alphas: Optional[List[float]] = None,
        es_alphas: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> RiskForecast:
        """Forecast analytical normal VaR and ES."""
        if var_alphas is None:
            var_alphas = [0.95, 0.99]
        if es_alphas is None:
            es_alphas = [0.975, 0.99]

        p_ret = portfolio_returns(returns_window, portfolio)
        n_obs = len(p_ret)

        if n_obs < 5:
            raise ValueError(f"Parametric normal model requires at least 5 observations (got {n_obs}).")

        mu = 0.0 if self.assume_zero_mean else float(p_ret.mean())
        sigma = float(p_ret.std(ddof=1))

        if sigma <= 0.0 or np.isnan(sigma):
            sigma = 1e-6

        var_forecasts: Dict[float, float] = {}
        for alpha in var_alphas:
            z_alpha = float(stats.norm.ppf(alpha))
            var_val = -mu + sigma * z_alpha
            var_forecasts[alpha] = max(0.0, var_val)

        es_forecasts: Dict[float, float] = {}
        for alpha in es_alphas:
            z_alpha = float(stats.norm.ppf(alpha))
            pdf_z = float(stats.norm.pdf(z_alpha))
            es_val = -mu + sigma * (pdf_z / (1.0 - alpha))
            es_forecasts[alpha] = max(var_forecasts.get(alpha, 0.0), es_val)

        return RiskForecast(
            model_name=self.name,
            var_forecasts=var_forecasts,
            es_forecasts=es_forecasts,
            metadata={
                "mean": mu,
                "volatility": sigma,
                "window_observations": n_obs,
                "assume_zero_mean": self.assume_zero_mean,
            },
        )
