"""Parametric Student-t market risk model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from rvw.models.base import BaseRiskModel, RiskForecast
from rvw.portfolio.portfolio import Portfolio, portfolio_returns


class ParametricStudentTModel(BaseRiskModel):
    """Parametric Student-t distribution risk model.

    Captures fat tails in financial asset returns. Degrees of freedom nu is estimated
    via Maximum Likelihood Estimation (MLE) with configurable lower bound.
    VaR and Expected Shortfall are evaluated using exact closed-form analytical formulas:
        VaR_alpha = -mu + scale * t_nu^{-1}(alpha)
        ES_alpha = -mu + scale * [ f_nu(q) / (1 - alpha) * (nu + q^2) / (nu - 1) ]
    where q = t_nu^{-1}(alpha) and f_nu is the standard Student-t probability density.
    """

    def __init__(self, name: str = "parametric_t", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, params=params)
        self.fit_method = self.params.get("fit_method", "mle")
        self.min_df = float(self.params.get("min_df", 2.1))

    def forecast(
        self,
        returns_window: pd.DataFrame,
        portfolio: Portfolio,
        var_alphas: Optional[List[float]] = None,
        es_alphas: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> RiskForecast:
        """Forecast analytical Student-t VaR and ES."""
        if var_alphas is None:
            var_alphas = [0.95, 0.99]
        if es_alphas is None:
            es_alphas = [0.975, 0.99]

        p_ret = portfolio_returns(returns_window, portfolio)
        n_obs = len(p_ret)

        if n_obs < 15:
            raise ValueError(f"Parametric Student-t model requires at least 15 observations (got {n_obs}).")

        # Fit Student-t distribution
        try:
            nu_est, loc_est, scale_est = stats.t.fit(p_ret.values)
            nu = max(float(nu_est), self.min_df)
            loc = float(loc_est)
            scale = max(float(scale_est), 1e-6)
        except Exception:
            # Fallback to method of moments if MLE fails
            nu = 5.0
            loc = float(p_ret.mean())
            std = float(p_ret.std(ddof=1))
            scale = std * np.sqrt((nu - 2.0) / nu)

        var_forecasts: Dict[float, float] = {}
        for alpha in var_alphas:
            q = float(stats.t.ppf(alpha, df=nu))
            var_val = -loc + scale * q
            var_forecasts[alpha] = max(0.0, var_val)

        es_forecasts: Dict[float, float] = {}
        for alpha in es_alphas:
            q = float(stats.t.ppf(alpha, df=nu))
            pdf_q = float(stats.t.pdf(q, df=nu))
            # Analytical tail expectation for Student-t
            tail_exp = (pdf_q / (1.0 - alpha)) * ((nu + q**2) / (nu - 1.0))
            es_val = -loc + scale * tail_exp
            es_forecasts[alpha] = max(var_forecasts.get(alpha, 0.0), es_val)

        return RiskForecast(
            model_name=self.name,
            var_forecasts=var_forecasts,
            es_forecasts=es_forecasts,
            metadata={
                "df": nu,
                "loc": loc,
                "scale": scale,
                "window_observations": n_obs,
                "fit_method": self.fit_method,
            },
        )
