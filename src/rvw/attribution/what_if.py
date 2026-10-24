"""What-if portfolio rebalancing analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.attribution.euler import compute_euler_parametric_risk
from rvw.portfolio.portfolio import Portfolio


@dataclass
class WhatIfResult:
    """Outcome of what-if portfolio rebalancing analysis."""
    current_portfolio_name: str
    target_portfolio_name: str
    alpha: float
    current_var: float
    target_var: float
    delta_var: float
    pct_delta_var: float
    current_es: float
    target_es: float
    delta_es: float
    pct_delta_es: float
    current_vol: float
    target_vol: float
    delta_vol: float
    asset_weight_deltas: Dict[str, float]
    asset_cvar_deltas: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "current_portfolio_name": self.current_portfolio_name,
            "target_portfolio_name": self.target_portfolio_name,
            "alpha": self.alpha,
            "current_var": round(self.current_var, 6),
            "target_var": round(self.target_var, 6),
            "delta_var": round(self.delta_var, 6),
            "pct_delta_var": round(self.pct_delta_var, 2),
            "current_es": round(self.current_es, 6),
            "target_es": round(self.target_es, 6),
            "delta_es": round(self.delta_es, 6),
            "pct_delta_es": round(self.pct_delta_es, 2),
            "current_vol": round(self.current_vol, 6),
            "target_vol": round(self.target_vol, 6),
            "delta_vol": round(self.delta_vol, 6),
            "asset_weight_deltas": {k: round(v, 4) for k, v in self.asset_weight_deltas.items()},
            "asset_cvar_deltas": {k: round(v, 6) for k, v in self.asset_cvar_deltas.items()},
        }


def compute_what_if_rebalance(
    returns: pd.DataFrame,
    current_portfolio: Portfolio,
    target_portfolio: Portfolio,
    alpha: float = 0.99,
) -> WhatIfResult:
    """Evaluate risk changes resulting from a proposed portfolio rebalancing.

    Args:
        returns: Asset return DataFrame.
        current_portfolio: Current asset allocation.
        target_portfolio: Proposed target asset allocation.
        alpha: Confidence level (default 0.99).

    Returns:
        WhatIfResult: Comparison of risk metrics and asset contribution deltas.
    """
    # Combine all assets involved
    all_tickers = sorted(list(set(current_portfolio.tickers) | set(target_portfolio.tickers)))

    # Expand weight vectors to full union universe
    w_curr_series = pd.Series({t: current_portfolio.weights.get(t, 0.0) for t in all_tickers})
    w_tgt_series = pd.Series({t: target_portfolio.weights.get(t, 0.0) for t in all_tickers})

    p_curr = Portfolio(weights=w_curr_series, name=current_portfolio.name)
    p_tgt = Portfolio(weights=w_tgt_series, name=target_portfolio.name)

    res_curr = compute_euler_parametric_risk(returns, p_curr, alpha=alpha)
    res_tgt = compute_euler_parametric_risk(returns, p_tgt, alpha=alpha)

    delta_var = res_tgt.total_var - res_curr.total_var
    pct_delta_var = (delta_var / res_curr.total_var) * 100.0 if res_curr.total_var > 0 else 0.0

    delta_es = res_tgt.total_es - res_curr.total_es
    pct_delta_es = (delta_es / res_curr.total_es) * 100.0 if res_curr.total_es > 0 else 0.0

    delta_vol = res_tgt.portfolio_vol - res_curr.portfolio_vol

    weight_deltas = {}
    cvar_deltas = {}
    for t in all_tickers:
        weight_deltas[t] = float(w_tgt_series[t] - w_curr_series[t])
        cvar_curr = res_curr.component_var.get(t, 0.0)
        cvar_tgt = res_tgt.component_var.get(t, 0.0)
        cvar_deltas[t] = float(cvar_tgt - cvar_curr)

    return WhatIfResult(
        current_portfolio_name=current_portfolio.name,
        target_portfolio_name=target_portfolio.name,
        alpha=alpha,
        current_var=res_curr.total_var,
        target_var=res_tgt.total_var,
        delta_var=delta_var,
        pct_delta_var=pct_delta_var,
        current_es=res_curr.total_es,
        target_es=res_tgt.total_es,
        delta_es=delta_es,
        pct_delta_es=pct_delta_es,
        current_vol=res_curr.portfolio_vol,
        target_vol=res_tgt.portfolio_vol,
        delta_vol=delta_vol,
        asset_weight_deltas=weight_deltas,
        asset_cvar_deltas=cvar_deltas,
    )
