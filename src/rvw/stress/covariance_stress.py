"""Correlation and volatility stress testing for covariance matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from rvw.portfolio.portfolio import Portfolio


@dataclass
class CovarianceStressResult:
    """Outcome of covariance and volatility stress testing."""
    scenario_name: str
    portfolio_name: str
    baseline_vol: float
    stressed_vol: float
    vol_increase_pct: float
    baseline_var_99: float
    stressed_var_99: float
    baseline_es_99: float
    stressed_es_99: float
    vol_scale_factor: float
    correlation_shift: float
    disclaimer: str = "Covariance stress scenario analysis (NOT backtesting)."

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "portfolio_name": self.portfolio_name,
            "baseline_vol": round(self.baseline_vol, 4),
            "stressed_vol": round(self.stressed_vol, 4),
            "vol_increase_pct": round(self.vol_increase_pct, 2),
            "baseline_var_99": round(self.baseline_var_99, 4),
            "stressed_var_99": round(self.stressed_var_99, 4),
            "baseline_es_99": round(self.baseline_es_99, 4),
            "stressed_es_99": round(self.stressed_es_99, 4),
            "vol_scale_factor": self.vol_scale_factor,
            "correlation_shift": self.correlation_shift,
            "disclaimer": self.disclaimer,
        }


def stress_covariance_matrix(
    returns: pd.DataFrame,
    portfolio: Portfolio,
    vol_scale_factor: float = 1.5,
    correlation_shift: float = 0.3,
    alpha: float = 0.99,
) -> CovarianceStressResult:
    """Stress asset return covariance matrix by scaling volatilities and shifting correlations.

    Stress formulation:
    1. Volatilities: sigma_stressed = sigma_base * vol_scale_factor
    2. Correlations: rho_stressed_ij = rho_base_ij + correlation_shift * (1.0 - rho_base_ij)
    3. Reconstruct stressed covariance matrix and verify positive semi-definiteness.

    Args:
        returns: Asset return DataFrame.
        portfolio: Target portfolio.
        vol_scale_factor: Multiplier for asset standard deviations (e.g. 1.5 for +50% vol).
        correlation_shift: Shift parameter moving correlations towards +1.0.
        alpha: Confidence level for VaR and ES (default 0.99).

    Returns:
        CovarianceStressResult: Comparison of baseline vs stressed risk metrics.
    """
    aligned = returns[portfolio.tickers].dropna()
    weights = portfolio.weights[portfolio.tickers].values
    N = len(portfolio.tickers)

    base_cov = np.cov(aligned.values, rowvar=False)
    if N == 1:
        base_cov = np.array([[float(base_cov)]])

    base_vols = np.sqrt(np.diag(base_cov))
    d_inv = np.diag(1.0 / np.maximum(base_vols, 1e-8))
    base_corr = d_inv @ base_cov @ d_inv

    # 1. Scale volatilities
    stressed_vols = base_vols * vol_scale_factor

    # 2. Shift correlations towards 1.0 (breakdown in diversification)
    stressed_corr = np.copy(base_corr)
    for i in range(N):
        for j in range(N):
            if i != j:
                stressed_corr[i, j] = base_corr[i, j] + correlation_shift * (1.0 - base_corr[i, j])

    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(stressed_corr)
    eigvals = np.maximum(eigvals, 1e-6)
    stressed_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d_inv_st = np.diag(1.0 / np.sqrt(np.diag(stressed_corr)))
    stressed_corr = d_inv_st @ stressed_corr @ d_inv_st

    # 3. Stressed Covariance
    d_vol = np.diag(stressed_vols)
    stressed_cov = d_vol @ stressed_corr @ d_vol

    # Baseline portfolio variance
    base_port_var = float(weights.T @ base_cov @ weights)
    base_port_vol = np.sqrt(max(base_port_var, 1e-8))

    # Stressed portfolio variance
    stressed_port_var = float(weights.T @ stressed_cov @ weights)
    stressed_port_vol = np.sqrt(max(stressed_port_var, 1e-8))

    # Analytical normal VaR and ES
    z_alpha = float(stats.norm.ppf(alpha))
    es_factor = float(stats.norm.pdf(z_alpha) / (1.0 - alpha))

    base_var = base_port_vol * z_alpha
    stressed_var = stressed_port_vol * z_alpha
    base_es = base_port_vol * es_factor
    stressed_es = stressed_port_vol * es_factor

    vol_pct = (stressed_port_vol - base_port_vol) / base_port_vol * 100.0

    return CovarianceStressResult(
        scenario_name=f"Vol x{vol_scale_factor:.1f}, Corr +{correlation_shift:.2f}",
        portfolio_name=portfolio.name,
        baseline_vol=base_port_vol,
        stressed_vol=stressed_port_vol,
        vol_increase_pct=vol_pct,
        baseline_var_99=base_var,
        stressed_var_99=stressed_var,
        baseline_es_99=base_es,
        stressed_es_99=stressed_es,
        vol_scale_factor=vol_scale_factor,
        correlation_shift=correlation_shift,
    )
