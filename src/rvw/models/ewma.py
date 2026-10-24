"""Exponentially Weighted Moving Average (EWMA / RiskMetrics) market risk model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from rvw.models.base import BaseRiskModel, RiskForecast
from rvw.portfolio.portfolio import Portfolio


def compute_ewma_covariance(
    returns: np.ndarray,
    lambda_param: float = 0.94,
    initial_cov: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute recursive EWMA covariance matrix through time.

    Sigma_t = lambda * Sigma_{t-1} + (1 - lambda) * r_{t-1} * r_{t-1}^T

    Args:
        returns: Matrix of asset returns of shape (T, N).
        lambda_param: Decay factor (typically 0.94 for daily data).
        initial_cov: Optional starting covariance matrix (N, N). If None, uses sample covariance.

    Returns:
        np.ndarray: Final step covariance matrix of shape (N, N).
    """
    T, N = returns.shape
    if initial_cov is None:
        cov = np.cov(returns, rowvar=False)
        if N == 1:
            cov = np.array([[float(cov)]])
    else:
        cov = initial_cov.copy()

    for t in range(T):
        r_t = returns[t:t + 1, :]  # shape (1, N)
        outer_prod = r_t.T @ r_t   # shape (N, N)
        cov = lambda_param * cov + (1.0 - lambda_param) * outer_prod

    return cov


class EWMAModel(BaseRiskModel):
    """RiskMetrics EWMA Volatility and Covariance Model.

    Applies exponential decay weighting to capture volatility clustering.
    Assumes zero drift and Gaussian innovations:
        VaR_alpha = sigma_p * z_alpha
        ES_alpha = sigma_p * phi(z_alpha) / (1 - alpha)
    where sigma_p = sqrt(w^T * Sigma_t * w).
    """

    def __init__(self, name: str = "ewma", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, params=params)
        self.lambda_param = float(self.params.get("lambda_param", 0.94))

    def forecast(
        self,
        returns_window: pd.DataFrame,
        portfolio: Portfolio,
        var_alphas: Optional[List[float]] = None,
        es_alphas: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> RiskForecast:
        """Forecast EWMA analytical normal VaR and ES."""
        if var_alphas is None:
            var_alphas = [0.95, 0.99]
        if es_alphas is None:
            es_alphas = [0.975, 0.99]

        aligned_returns = returns_window[portfolio.tickers].values
        weights = portfolio.weights[portfolio.tickers].values
        n_obs = len(aligned_returns)

        if n_obs < 10:
            raise ValueError(f"EWMA model requires at least 10 observations (got {n_obs}).")

        final_cov = compute_ewma_covariance(aligned_returns, lambda_param=self.lambda_param)
        port_var = float(weights.T @ final_cov @ weights)
        sigma_p = np.sqrt(max(port_var, 1e-8))

        var_forecasts: Dict[float, float] = {}
        for alpha in var_alphas:
            z_alpha = float(stats.norm.ppf(alpha))
            var_forecasts[alpha] = float(sigma_p * z_alpha)

        es_forecasts: Dict[float, float] = {}
        for alpha in es_alphas:
            z_alpha = float(stats.norm.ppf(alpha))
            pdf_z = float(stats.norm.pdf(z_alpha))
            es_forecasts[alpha] = float(sigma_p * (pdf_z / (1.0 - alpha)))

        return RiskForecast(
            model_name=self.name,
            var_forecasts=var_forecasts,
            es_forecasts=es_forecasts,
            metadata={
                "lambda": self.lambda_param,
                "sigma_p": sigma_p,
                "covariance_matrix": final_cov,
                "window_observations": n_obs,
            },
        )
