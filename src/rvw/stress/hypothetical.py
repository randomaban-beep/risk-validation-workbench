"""Hypothetical stress testing with duration approximation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.portfolio.portfolio import Portfolio


@dataclass
class HypotheticalScenarioResult:
    """Outcome of hypothetical scenario stress test."""
    scenario_id: str
    scenario_name: str
    portfolio_name: str
    portfolio_loss: float
    portfolio_return: float
    asset_shocks: Dict[str, float]
    asset_loss_contributions: Dict[str, float]
    disclaimer: str = "Illustrative assumptions, not calibrated forecasts. Scenario analysis, NOT backtesting."

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "portfolio_name": self.portfolio_name,
            "portfolio_loss": round(self.portfolio_loss, 4),
            "portfolio_return": round(self.portfolio_return, 4),
            "asset_shocks": {k: round(v, 4) for k, v in self.asset_shocks.items()},
            "asset_loss_contributions": {k: round(v, 4) for k, v in self.asset_loss_contributions.items()},
            "disclaimer": self.disclaimer,
        }


def evaluate_hypothetical_scenario(
    portfolio: Portfolio,
    scenario_id: str,
    scenario_name: str,
    asset_shocks: Dict[str, float],
    yield_shocks_bps: Optional[Dict[str, float]] = None,
    duration_assumptions: Optional[Dict[str, float]] = None,
) -> HypotheticalScenarioResult:
    """Evaluate hypothetical instantaneous multi-asset shock on a portfolio.

    For fixed income assets with yield shocks:
        Shock = - Duration * (delta_y_bps / 10000.0)

    Args:
        portfolio: Target portfolio.
        scenario_id: Identifier key.
        scenario_name: Descriptive title.
        asset_shocks: Direct return shocks per asset (e.g. {'SPY': -0.20}).
        yield_shocks_bps: Optional interest rate shifts in basis points (e.g. {'TLT': 100}).
        duration_assumptions: Modified duration per fixed-income asset in years.

    Returns:
        HypotheticalScenarioResult: Calculated portfolio loss and asset-level contributions.
    """
    total_shocks: Dict[str, float] = dict(asset_shocks)

    if yield_shocks_bps and duration_assumptions:
        for ticker, bps in yield_shocks_bps.items():
            duration = duration_assumptions.get(ticker, 0.0)
            # Duration approximation: delta_P / P = - D * delta_y
            delta_y = bps / 10000.0
            price_shock = -duration * delta_y
            total_shocks[ticker] = price_shock

    # Compute portfolio return: sum(w_i * shock_i)
    port_ret = 0.0
    contributions: Dict[str, float] = {}

    for ticker, weight in portfolio.weights.items():
        shock = total_shocks.get(ticker, 0.0)
        contrib = float(weight * shock)
        port_ret += contrib
        contributions[ticker] = -contrib  # positive loss contribution

    port_loss = -port_ret

    return HypotheticalScenarioResult(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        portfolio_name=portfolio.name,
        portfolio_loss=port_loss,
        portfolio_return=port_ret,
        asset_shocks=total_shocks,
        asset_loss_contributions=contributions,
    )
