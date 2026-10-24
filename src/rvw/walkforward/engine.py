"""Strict walk-forward out-of-sample forecasting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.exceptions import LookAheadLeakageError, ModelFitError
from rvw.logging_utils import log_event
from rvw.models.base import BaseRiskModel, RiskForecast
from rvw.models.historical import HistoricalSimulationModel
from rvw.portfolio.portfolio import Portfolio, portfolio_returns

logger = logging.getLogger(__name__)


@dataclass
class ForecastFrame:
    """Out-of-sample walk-forward forecasts, realized returns, and hits."""
    model_name: str
    portfolio_name: str
    window_length: int
    window_type: str
    refit_every: int
    dates: pd.DatetimeIndex
    portfolio_returns: pd.Series
    realized_losses: pd.Series
    var_forecasts: pd.DataFrame
    es_forecasts: pd.DataFrame
    hits: pd.DataFrame
    failed_dates: List[pd.Timestamp] = field(default_factory=list)
    fallback_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Combine realized losses, forecasts, and hits into a unified DataFrame."""
        df = pd.DataFrame(index=self.dates)
        df["Return"] = self.portfolio_returns.values
        df["Loss"] = self.realized_losses.values

        for col in self.var_forecasts.columns:
            df[f"VaR_{col}"] = self.var_forecasts[col].values
        for col in self.es_forecasts.columns:
            df[f"ES_{col}"] = self.es_forecasts[col].values
        for col in self.hits.columns:
            df[f"Hit_{col}"] = self.hits[col].values

        return df

    def summary(self) -> Dict[str, Any]:
        """Summary statistics of walkforward execution."""
        res: Dict[str, Any] = {
            "model_name": self.model_name,
            "portfolio_name": self.portfolio_name,
            "observations": len(self.dates),
            "start_date": str(self.dates.min().date()) if len(self.dates) > 0 else None,
            "end_date": str(self.dates.max().date()) if len(self.dates) > 0 else None,
            "failed_fits": len(self.failed_dates),
            "fallbacks_used": self.fallback_count,
            "hit_counts": {},
            "hit_rates": {},
        }
        for col in self.hits.columns:
            cnt = int(self.hits[col].sum())
            res["hit_counts"][col] = cnt
            res["hit_rates"][col] = float(cnt / len(self.dates)) if len(self.dates) > 0 else 0.0
        return res


def run_walkforward(
    returns: pd.DataFrame,
    portfolio: Portfolio,
    model: BaseRiskModel,
    window_length: int = 500,
    refit_every: int = 1,
    window_type: str = "rolling",
    var_alphas: Optional[List[float]] = None,
    es_alphas: Optional[List[float]] = None,
    fallback_on_failure: bool = True,
    seed: Optional[int] = 42,
) -> ForecastFrame:
    """Execute strict walk-forward out-of-sample risk forecasting.

    Causality Principle:
    The forecast for day t is evaluated strictly on the historical sample
    available through day t-1. Day t returns are NEVER seen during estimation.

    Args:
        returns: Full historical asset returns matrix (T x N).
        portfolio: Portfolio specifying asset weights.
        model: BaseRiskModel instance.
        window_length: Number of historical days in estimation window.
        refit_every: Refit model parameters every N steps (cached in between).
        window_type: 'rolling' (fixed length) or 'expanding'.
        var_alphas: VaR alpha levels (e.g. [0.95, 0.99]).
        es_alphas: ES alpha levels (e.g. [0.975, 0.99]).
        fallback_on_failure: If True, fallback to Historical Simulation on fit errors.
        seed: Random seed for stochastic models.

    Returns:
        ForecastFrame: Aligned out-of-sample forecasts and realized hits.

    Raises:
        ValueError: If sample length is less than window_length + 1.
    """
    if var_alphas is None:
        var_alphas = [0.95, 0.99]
    if es_alphas is None:
        es_alphas = [0.975, 0.99]

    T = len(returns)
    if T <= window_length:
        raise ValueError(
            f"Insufficient return history ({T} days) for window length {window_length}."
        )

    # Compute realized portfolio returns across full timeline
    p_returns = portfolio_returns(returns, portfolio)
    p_losses = -p_returns

    # Out-of-sample forecast dates: from index window_length to T - 1
    forecast_dates = returns.index[window_length:]
    n_forecasts = len(forecast_dates)

    var_cols = [f"{int(a * 100)}" if a.is_integer() else f"{a:.3f}".rstrip("0") for a in var_alphas]
    es_cols = [f"{int(a * 100)}" if a.is_integer() else f"{a:.3f}".rstrip("0") for a in es_alphas]

    var_matrix = np.full((n_forecasts, len(var_alphas)), np.nan)
    es_matrix = np.full((n_forecasts, len(es_alphas)), np.nan)

    failed_dates: List[pd.Timestamp] = []
    fallback_count = 0
    fallback_model = HistoricalSimulationModel() if fallback_on_failure else None

    cached_forecast: Optional[RiskForecast] = None

    for i, t_idx in enumerate(range(window_length, T)):
        current_date = returns.index[t_idx]

        # Information set strictly up to t_idx - 1 (exclusive of t_idx)
        if window_type == "rolling":
            window_slice = returns.iloc[t_idx - window_length : t_idx]
        elif window_type == "expanding":
            window_slice = returns.iloc[0 : t_idx]
        else:
            raise ValueError(f"Unknown window_type '{window_type}'. Must be 'rolling' or 'expanding'.")

        # Refit decision
        need_refit = (i % refit_every == 0) or (cached_forecast is None)

        forecast: Optional[RiskForecast] = None
        if need_refit:
            try:
                # Stochastic models receive derived seed for reproducibility
                step_seed = (seed + i) if seed is not None else None
                forecast = model.forecast(
                    returns_window=window_slice,
                    portfolio=portfolio,
                    var_alphas=var_alphas,
                    es_alphas=es_alphas,
                    seed=step_seed,
                )
                cached_forecast = forecast
            except Exception as exc:
                failed_dates.append(current_date)
                log_event(
                    logger=logger,
                    level=logging.WARNING,
                    event_type="WALKFORWARD_FIT_FAILURE",
                    message=f"Model '{model.name}' fit failed on {current_date.date()}: {exc}",
                    date=str(current_date.date()),
                    model=model.name,
                    error=str(exc),
                )
                if fallback_on_failure and fallback_model is not None:
                    fallback_count += 1
                    try:
                        forecast = fallback_model.forecast(
                            returns_window=window_slice,
                            portfolio=portfolio,
                            var_alphas=var_alphas,
                            es_alphas=es_alphas,
                        )
                        forecast.warning_message = f"Fallback to HS on {current_date.date()} due to: {exc}"
                    except Exception as fb_exc:
                        forecast = None
                else:
                    forecast = None
        else:
            forecast = cached_forecast

        if forecast is not None:
            for j, alpha in enumerate(var_alphas):
                var_matrix[i, j] = forecast.get_var(alpha)
            for j, alpha in enumerate(es_alphas):
                es_matrix[i, j] = forecast.get_es(alpha)

    # Wrap into DataFrames
    var_df = pd.DataFrame(var_matrix, index=forecast_dates, columns=var_cols)
    es_df = pd.DataFrame(es_matrix, index=forecast_dates, columns=es_cols)

    oos_losses = p_losses.loc[forecast_dates]
    oos_returns = p_returns.loc[forecast_dates]

    # Calculate VaR exceptions (hits): Realized Loss > VaR Forecast
    hits_matrix = np.zeros(var_matrix.shape, dtype=bool)
    for j in range(len(var_alphas)):
        var_col_vals = var_matrix[:, j]
        valid_mask = ~np.isnan(var_col_vals)
        hits_matrix[valid_mask, j] = oos_losses.values[valid_mask] > var_col_vals[valid_mask]

    hits_df = pd.DataFrame(hits_matrix, index=forecast_dates, columns=var_cols)

    return ForecastFrame(
        model_name=model.name,
        portfolio_name=portfolio.name,
        window_length=window_length,
        window_type=window_type,
        refit_every=refit_every,
        dates=forecast_dates,
        portfolio_returns=oos_returns,
        realized_losses=oos_losses,
        var_forecasts=var_df,
        es_forecasts=es_df,
        hits=hits_df,
        failed_dates=failed_dates,
        fallback_count=fallback_count,
        metadata={"var_alphas": var_alphas, "es_alphas": es_alphas},
    )
