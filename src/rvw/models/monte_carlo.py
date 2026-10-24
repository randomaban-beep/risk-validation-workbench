"""Monte Carlo simulation market risk model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.models.base import BaseRiskModel, RiskForecast
from rvw.portfolio.portfolio import Portfolio


class MonteCarloRiskModel(BaseRiskModel):
    """Monte Carlo full revaluation market risk model.

    Generates correlated joint asset return paths via Cholesky decomposition
    of the empirical covariance matrix. Deterministic seed management ensures reproducibility.
    """

    def __init__(self, name: str = "monte_carlo", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, params=params)
        self.num_simulations = int(self.params.get("num_simulations", 25000))
        self.distribution = str(self.params.get("distribution", "normal"))
        self.df_t = float(self.params.get("df_t", 5.0))

    def forecast(
        self,
        returns_window: pd.DataFrame,
        portfolio: Portfolio,
        var_alphas: Optional[List[float]] = None,
        es_alphas: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> RiskForecast:
        """Forecast Monte Carlo VaR and ES using Cholesky factorisation."""
        if var_alphas is None:
            var_alphas = [0.95, 0.99]
        if es_alphas is None:
            es_alphas = [0.975, 0.99]

        aligned_returns = returns_window[portfolio.tickers].values
        weights = portfolio.weights[portfolio.tickers].values
        T, N = aligned_returns.shape

        if T < 10:
            raise ValueError(f"Monte Carlo model requires at least 10 observations (got {T}).")

        rng = np.random.default_rng(seed)

        mu = np.mean(aligned_returns, axis=0)
        cov = np.cov(aligned_returns, rowvar=False)
        if N == 1:
            cov = np.array([[float(cov)]])

        # Ensure positive definite via minimal diagonal ridge if needed
        min_eig = np.min(np.linalg.eigvalsh(cov))
        if min_eig < 1e-8:
            cov = cov + np.eye(N) * (1e-7 - min_eig)

        cholesky = np.linalg.cholesky(cov)

        if self.distribution == "t":
            # Multivariate Student-t draws
            chi2_draws = rng.chisquare(df=self.df_t, size=(self.num_simulations, 1)) / self.df_t
            std_normals = rng.standard_normal(size=(self.num_simulations, N))
            scale_t = np.sqrt((self.df_t - 2.0) / self.df_t)
            sim_returns = mu + (std_normals @ cholesky.T) * (scale_t / np.sqrt(chi2_draws))
        else:
            # Multivariate Normal draws
            std_normals = rng.standard_normal(size=(self.num_simulations, N))
            sim_returns = mu + (std_normals @ cholesky.T)

        sim_port_returns = sim_returns @ weights
        sim_losses = -sim_port_returns

        var_forecasts: Dict[float, float] = {}
        for alpha in var_alphas:
            var_val = float(np.percentile(sim_losses, alpha * 100.0))
            var_forecasts[alpha] = max(0.0, var_val)

        es_forecasts: Dict[float, float] = {}
        for alpha in es_alphas:
            var_th = float(np.percentile(sim_losses, alpha * 100.0))
            tail = sim_losses[sim_losses >= var_th]
            es_val = float(np.mean(tail)) if len(tail) > 0 else var_th
            es_forecasts[alpha] = max(var_forecasts.get(alpha, 0.0), es_val)

        return RiskForecast(
            model_name=self.name,
            var_forecasts=var_forecasts,
            es_forecasts=es_forecasts,
            metadata={
                "num_simulations": self.num_simulations,
                "distribution": self.distribution,
                "seed": seed,
                "window_observations": T,
            },
        )
