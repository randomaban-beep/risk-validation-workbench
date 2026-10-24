"""Filtered Historical Simulation (FHS) via GARCH(1,1)-t innovations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from arch import arch_model

from rvw.exceptions import ModelFitError
from rvw.models.base import BaseRiskModel, RiskForecast
from rvw.portfolio.portfolio import Portfolio, portfolio_returns

logger = logging.getLogger(__name__)


class GARCHFilteredHistoricalModel(BaseRiskModel):
    """Filtered Historical Simulation (FHS) market risk model.

    Fits a GARCH(1,1) model with Student-t innovations to the portfolio return series.
    The standardized residuals z_t = epsilon_t / sigma_t are resampled and scaled
    by the 1-step ahead conditional volatility forecast sigma_{t+1|t}.
    """

    def __init__(self, name: str = "garch_fhs", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, params=params)
        self.p = int(self.params.get("p", 1))
        self.q = int(self.params.get("q", 1))
        self.dist = str(self.params.get("dist", "StudentsT"))
        self.resample_size = int(self.params.get("resample_size", 10000))

    def forecast(
        self,
        returns_window: pd.DataFrame,
        portfolio: Portfolio,
        var_alphas: Optional[List[float]] = None,
        es_alphas: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> RiskForecast:
        """Fit GARCH(1,1)-t and forecast out-of-sample FHS VaR and ES."""
        if var_alphas is None:
            var_alphas = [0.95, 0.99]
        if es_alphas is None:
            es_alphas = [0.975, 0.99]

        p_ret = portfolio_returns(returns_window, portfolio)
        n_obs = len(p_ret)

        if n_obs < 50:
            raise ValueError(f"GARCH FHS requires at least 50 observations for convergence (got {n_obs}).")

        # Scale by 100 for numerical stability during MLE optimization
        scaled_returns = p_ret.values * 100.0

        try:
            am = arch_model(
                scaled_returns,
                mean="Constant",
                vol="Garch",
                p=self.p,
                q=self.q,
                dist=self.dist,
            )
            res = am.fit(disp="off", show_warning=False)

            # Extract standardized residuals: epsilon_t / sigma_t
            std_resid = res.std_resid
            valid_resid = std_resid[~np.isnan(std_resid)]

            # 1-day ahead conditional volatility and mean forecasts
            forecast_res = res.forecast(horizon=1)
            f_var = float(forecast_res.variance.iloc[-1, 0])
            f_mean = float(forecast_res.mean.iloc[-1, 0])

            cond_sigma = np.sqrt(max(f_var, 1e-6)) / 100.0
            cond_mu = f_mean / 100.0

            # Bootstrap / resample standardized innovations
            rng = np.random.default_rng(seed)
            sampled_z = rng.choice(valid_resid, size=self.resample_size, replace=True)

            # Simulated returns and losses: L = -(cond_mu + cond_sigma * z)
            sim_returns = cond_mu + cond_sigma * sampled_z
            sim_losses = -sim_returns

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
                    "cond_sigma": cond_sigma,
                    "cond_mu": cond_mu,
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                    "params": {str(k): float(v) for k, v in res.params.items()},
                    "window_observations": n_obs,
                    "resample_size": self.resample_size,
                },
            )

        except Exception as exc:
            logger.warning("GARCH estimation failed on window (%d obs): %s", n_obs, exc)
            raise ModelFitError(f"GARCH model fit error: {exc}") from exc
