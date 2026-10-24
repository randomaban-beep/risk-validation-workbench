"""Euler component risk decomposition for VaR and Expected Shortfall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from rvw.portfolio.portfolio import Portfolio, portfolio_returns


@dataclass
class EulerAttributionResult:
    """Outcome of Euler component VaR and ES risk decomposition."""
    portfolio_name: str
    alpha: float
    total_var: float
    total_es: float
    portfolio_vol: float
    portfolio_mean: float
    marginal_var: Dict[str, float]
    component_var: Dict[str, float]
    pct_var: Dict[str, float]
    marginal_es: Dict[str, float]
    component_es: Dict[str, float]
    pct_es: Dict[str, float]
    weights: Dict[str, float]
    euler_sum_error_var: float
    euler_sum_error_es: float

    def to_dataframe(self) -> pd.DataFrame:
        """Present asset risk decomposition in a tabular DataFrame."""
        rows = []
        for ticker in self.weights.keys():
            rows.append({
                "Asset": ticker,
                "Weight": self.weights[ticker],
                "Marginal_VaR": self.marginal_var[ticker],
                "Component_VaR": self.component_var[ticker],
                "Pct_VaR": self.pct_var[ticker],
                "Marginal_ES": self.marginal_es[ticker],
                "Component_ES": self.component_es[ticker],
                "Pct_ES": self.pct_es[ticker],
            })
        df = pd.DataFrame(rows).set_index("Asset")
        return df

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "portfolio_name": self.portfolio_name,
            "alpha": self.alpha,
            "total_var": round(self.total_var, 6),
            "total_es": round(self.total_es, 6),
            "portfolio_vol": round(self.portfolio_vol, 6),
            "portfolio_mean": round(self.portfolio_mean, 6),
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "marginal_var": {k: round(v, 6) for k, v in self.marginal_var.items()},
            "component_var": {k: round(v, 6) for k, v in self.component_var.items()},
            "pct_var": {k: round(v, 2) for k, v in self.pct_var.items()},
            "marginal_es": {k: round(v, 6) for k, v in self.marginal_es.items()},
            "component_es": {k: round(v, 6) for k, v in self.component_es.items()},
            "pct_es": {k: round(v, 2) for k, v in self.pct_es.items()},
            "euler_sum_error_var": float(self.euler_sum_error_var),
            "euler_sum_error_es": float(self.euler_sum_error_es),
        }


def compute_euler_parametric_risk(
    returns: pd.DataFrame,
    portfolio: Portfolio,
    alpha: float = 0.99,
    assume_zero_mean: bool = False,
) -> EulerAttributionResult:
    """Compute Euler component VaR and ES risk decomposition.

    Mathematical Formulation:
    Linear homogeneity of degree 1 holds for VaR:
        VaR(w) = sum_{i=1}^N w_i * (d VaR / d w_i)
    Marginal VaR:
        d VaR / d w_i = -mu_i + z_alpha * (Sigma * w)_i / sigma_p
    Component VaR:
        CVaR_i = w_i * (d VaR / d w_i)
    Component ES:
        CES_i = w_i * [ -mu_i + (phi(z_alpha) / (1 - alpha)) * (Sigma * w)_i / sigma_p ]

    Euler Identity:
        sum_i CVaR_i == Total VaR
        sum_i CES_i == Total ES

    Args:
        returns: Asset return DataFrame.
        portfolio: Target portfolio.
        alpha: Confidence level (default 0.99).
        assume_zero_mean: If True, set asset means to 0.

    Returns:
        EulerAttributionResult: Detailed component breakdown and exact sum check.
    """
    aligned = returns[portfolio.tickers].dropna()
    weights = portfolio.weights[portfolio.tickers].values
    N = len(portfolio.tickers)

    cov = np.cov(aligned.values, rowvar=False)
    if N == 1:
        cov = np.array([[float(cov)]])

    mu_vec = np.zeros(N) if assume_zero_mean else np.mean(aligned.values, axis=0)

    # Portfolio metrics
    port_var = float(weights.T @ cov @ weights)
    port_vol = np.sqrt(max(port_var, 1e-8))
    port_mu = float(weights.T @ mu_vec)

    # Normal factors
    z_alpha = float(stats.norm.ppf(alpha))
    es_factor = float(stats.norm.pdf(z_alpha) / (1.0 - alpha))

    # Total VaR and ES
    total_var = -port_mu + port_vol * z_alpha
    total_es = -port_mu + port_vol * es_factor

    # Beta vector: cov @ w / port_vol
    cov_w = cov @ weights
    beta_vec = cov_w / port_vol

    # Marginal VaR and ES: shape (N,)
    m_var = -mu_vec + z_alpha * beta_vec
    m_es = -mu_vec + es_factor * beta_vec

    # Component VaR and ES
    c_var = weights * m_var
    c_es = weights * m_es

    # Percentage contributions
    pct_var = (c_var / total_var) * 100.0 if abs(total_var) > 1e-8 else np.zeros(N)
    pct_es = (c_es / total_es) * 100.0 if abs(total_es) > 1e-8 else np.zeros(N)

    # Verification of Euler additivity
    sum_err_var = float(abs(np.sum(c_var) - total_var))
    sum_err_es = float(abs(np.sum(c_es) - total_es))

    tickers = portfolio.tickers
    return EulerAttributionResult(
        portfolio_name=portfolio.name,
        alpha=alpha,
        total_var=total_var,
        total_es=total_es,
        portfolio_vol=port_vol,
        portfolio_mean=port_mu,
        marginal_var={tickers[i]: float(m_var[i]) for i in range(N)},
        component_var={tickers[i]: float(c_var[i]) for i in range(N)},
        pct_var={tickers[i]: float(pct_var[i]) for i in range(N)},
        marginal_es={tickers[i]: float(m_es[i]) for i in range(N)},
        component_es={tickers[i]: float(c_es[i]) for i in range(N)},
        pct_es={tickers[i]: float(pct_es[i]) for i in range(N)},
        weights={tickers[i]: float(weights[i]) for i in range(N)},
        euler_sum_error_var=sum_err_var,
        euler_sum_error_es=sum_err_es,
    )


def compute_historical_tail_attribution(
    returns: pd.DataFrame,
    portfolio: Portfolio,
    alpha: float = 0.99,
) -> Dict[str, Any]:
    """Compute empirical asset contributions to historical Expected Shortfall.

    On the set of historical days where portfolio loss exceeds VaR,
    each asset's tail loss contribution is:
        Contrib_i = mean( - w_i * R_{i, t} | L_t >= VaR_alpha )
    The sum of asset tail contributions equals the historical Expected Shortfall.

    Args:
        returns: Asset return DataFrame.
        portfolio: Target portfolio.
        alpha: Confidence level.

    Returns:
        Dict[str, Any]: Asset-level tail losses and verification sum check.
    """
    aligned = returns[portfolio.tickers].dropna()
    weights = portfolio.weights[portfolio.tickers].values
    tickers = portfolio.tickers

    port_rets = aligned.values @ weights
    losses = -port_rets

    var_threshold = float(np.percentile(losses, alpha * 100.0))
    tail_mask = losses >= var_threshold
    tail_days_count = int(np.sum(tail_mask))

    if tail_days_count == 0:
        tail_mask = losses == np.max(losses)
        tail_days_count = int(np.sum(tail_mask))

    total_historical_es = float(np.mean(losses[tail_mask]))

    tail_returns_matrix = aligned.values[tail_mask]  # shape (K, N)
    # Asset loss: - R_{i, t}
    asset_tail_losses = -tail_returns_matrix

    # Asset weighted tail contribution: w_i * mean(- R_{i, t})
    contributions: Dict[str, float] = {}
    pct_contributions: Dict[str, float] = {}

    for idx, ticker in enumerate(tickers):
        c_i = float(weights[idx] * np.mean(asset_tail_losses[:, idx]))
        contributions[ticker] = c_i
        pct_contributions[ticker] = (c_i / total_historical_es) * 100.0 if total_historical_es > 0 else 0.0

    sum_contrib = sum(contributions.values())
    sum_err = abs(sum_contrib - total_historical_es)

    return {
        "portfolio_name": portfolio.name,
        "alpha": alpha,
        "historical_var": var_threshold,
        "historical_es": total_historical_es,
        "tail_days_count": tail_days_count,
        "asset_tail_contributions": {k: float(v) for k, v in contributions.items()},
        "pct_tail_contributions": {k: float(v) for k, v in pct_contributions.items()},
        "euler_sum_error": float(sum_err),
    }
