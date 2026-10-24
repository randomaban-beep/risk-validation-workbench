"""Deterministic synthetic market data generator for the Risk Validation Workbench."""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# Default indicative universe baseline annualized volatilities and drift
ASSET_PROFILES: Dict[str, Dict[str, float]] = {
    "SPY": {"annual_vol": 0.18, "annual_drift": 0.08, "initial_price": 150.0},
    "QQQ": {"annual_vol": 0.22, "annual_drift": 0.11, "initial_price": 45.0},
    "IWM": {"annual_vol": 0.24, "annual_drift": 0.07, "initial_price": 75.0},
    "EFA": {"annual_vol": 0.20, "annual_drift": 0.05, "initial_price": 60.0},
    "EEM": {"annual_vol": 0.25, "annual_drift": 0.06, "initial_price": 38.0},
    "VNQ": {"annual_vol": 0.21, "annual_drift": 0.07, "initial_price": 50.0},
    "TLT": {"annual_vol": 0.14, "annual_drift": 0.03, "initial_price": 90.0},
    "IEF": {"annual_vol": 0.07, "annual_drift": 0.02, "initial_price": 85.0},
    "LQD": {"annual_vol": 0.09, "annual_drift": 0.04, "initial_price": 105.0},
    "HYG": {"annual_vol": 0.12, "annual_drift": 0.05, "initial_price": 80.0},
    "GLD": {"annual_vol": 0.16, "annual_drift": 0.06, "initial_price": 65.0},
    "DBC": {"annual_vol": 0.20, "annual_drift": 0.02, "initial_price": 25.0},
}


def generate_synthetic_prices(
    tickers: Optional[List[str]] = None,
    start_date: str = "2007-06-01",
    end_date: Optional[str] = "2024-06-30",
    seed: int = 777,
) -> pd.DataFrame:
    """Generate realistic synthetic daily adjusted close prices for validation.

    DATASET STATUS: SYNTHETIC.
    Simulated using multivariate Student-t innovations (df=5) with asset class
    correlations, mimicking stylized empirical facts (fat tails, equity-bond hedge).

    Args:
        tickers: List of ticker symbols to generate. Defaults to 12 indicative ETFs.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD). If None, defaults to current UTC date.
        seed: Random seed for deterministic generation.

    Returns:
        pd.DataFrame: Daily prices with DatetimeIndex (business days) and tickers as columns.
                      All column names and attributes indicate SYNTHETIC origin.
    """
    if tickers is None:
        tickers = list(ASSET_PROFILES.keys())

    if end_date is None:
        end_date = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")

    date_index = pd.bdate_range(start=start_date, end=end_date)
    num_days = len(date_index)
    num_assets = len(tickers)

    if num_days <= 1:
        raise ValueError(f"Date range {start_date} to {end_date} produces insufficient business days.")

    rng = np.random.default_rng(seed)

    # Base correlation matrix structure
    corr = np.eye(num_assets)
    equity_tickers = {"SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ"}
    bond_tickers = {"TLT", "IEF", "LQD", "HYG"}

    for i in range(num_assets):
        for j in range(num_assets):
            if i == j:
                continue
            ti, tj = tickers[i], tickers[j]
            if ti in equity_tickers and tj in equity_tickers:
                corr[i, j] = 0.75
            elif ti in bond_tickers and tj in bond_tickers:
                corr[i, j] = 0.70
            elif (ti in equity_tickers and tj in {"TLT", "IEF"}) or (tj in equity_tickers and ti in {"TLT", "IEF"}):
                corr[i, j] = -0.25  # Flight-to-safety negative correlation
            elif (ti in equity_tickers and tj in {"HYG", "LQD"}) or (tj in equity_tickers and ti in {"HYG", "LQD"}):
                corr[i, j] = 0.50   # Credit spread equity correlation
            elif ti == "GLD" or tj == "GLD":
                corr[i, j] = 0.10
            elif ti == "DBC" or tj == "DBC":
                corr[i, j] = 0.25
            else:
                corr[i, j] = 0.20

    # Ensure positive semi-definite correlation
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d_inv = np.diag(1.0 / np.sqrt(np.diag(corr)))
    corr = d_inv @ corr @ d_inv

    cholesky = np.linalg.cholesky(corr)

    # Generate multivariate Student-t innovations (df = 5)
    df_t = 5.0
    chi2_samples = rng.chisquare(df=df_t, size=(num_days - 1, 1)) / df_t
    gaussian_samples = rng.standard_normal(size=(num_days - 1, num_assets))
    t_innovations = (gaussian_samples @ cholesky.T) / np.sqrt(chi2_samples)

    # Scale daily returns
    daily_dt = 1.0 / 252.0
    daily_returns = np.zeros((num_days - 1, num_assets))
    prices = np.zeros((num_days, num_assets))

    for idx, ticker in enumerate(tickers):
        profile = ASSET_PROFILES.get(ticker, {"annual_vol": 0.18, "annual_drift": 0.06, "initial_price": 100.0})
        vol_daily = profile["annual_vol"] * np.sqrt(daily_dt)
        drift_daily = profile["annual_drift"] * daily_dt
        prices[0, idx] = profile["initial_price"]

        # Scale t-distribution variance (variance of standard t with df is df / (df-2))
        t_scale = np.sqrt((df_t - 2.0) / df_t)
        daily_returns[:, idx] = drift_daily + (t_innovations[:, idx] * t_scale * vol_daily)

    # Accumulate prices
    for t in range(1, num_days):
        prices[t] = prices[t - 1] * (1.0 + daily_returns[t - 1])

    df = pd.DataFrame(prices, index=date_index, columns=tickers)
    df.index.name = "Date"
    df.attrs["DATA_STATUS"] = "SYNTHETIC"
    df.attrs["SEED"] = seed
    return df
