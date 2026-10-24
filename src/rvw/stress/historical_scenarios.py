"""Historical scenario replay engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.portfolio.portfolio import Portfolio, portfolio_returns


@dataclass
class HistoricalScenarioResult:
    """Outcome of historical scenario replay."""
    scenario_id: str
    scenario_name: str
    start_date: str
    end_date: str
    portfolio_name: str
    trading_days: int
    cumulative_return: float
    cumulative_loss: float  # -cumulative_return
    max_drawdown: float
    asset_cumulative_returns: Dict[str, float]
    disclaimer: str = "Historical scenario replay analysis (NOT backtesting)."

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "portfolio_name": self.portfolio_name,
            "trading_days": self.trading_days,
            "cumulative_return": round(self.cumulative_return, 4),
            "cumulative_loss": round(self.cumulative_loss, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "asset_cumulative_returns": {k: round(v, 4) for k, v in self.asset_cumulative_returns.items()},
            "disclaimer": self.disclaimer,
        }


def replay_historical_scenario(
    returns: pd.DataFrame,
    portfolio: Portfolio,
    scenario_id: str,
    scenario_name: str,
    start_date: str,
    end_date: str,
) -> HistoricalScenarioResult:
    """Replay actual historical return window through current portfolio weights.

    Computes compounded cumulative portfolio return, cumulative loss, and maximum drawdown.

    Args:
        returns: Full historical asset returns DataFrame.
        portfolio: Target portfolio with asset weights.
        scenario_id: Short identifier (e.g. 'gfc', 'covid_crash').
        scenario_name: Full descriptive title.
        start_date: Inclusive start date (YYYY-MM-DD).
        end_date: Inclusive end date (YYYY-MM-DD).

    Returns:
        HistoricalScenarioResult: Evaluated scenario performance.
    """
    slice_df = returns.loc[start_date:end_date]
    if slice_df.empty:
        raise ValueError(f"No return data available between {start_date} and {end_date}.")

    p_rets = portfolio_returns(slice_df, portfolio)
    n_days = len(p_rets)

    # Compounded return: prod(1 + R) - 1
    cum_ret = float(np.prod(1.0 + p_rets.values) - 1.0)
    cum_loss = -cum_ret

    # Drawdown calculation
    wealth_index = np.cumprod(1.0 + p_rets.values)
    peaks = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index - peaks) / peaks
    max_dd = float(np.abs(np.min(drawdowns)))

    # Asset-level cumulative returns
    asset_cum = {}
    for ticker in portfolio.tickers:
        if ticker in slice_df.columns:
            r_asset = slice_df[ticker].values
            asset_cum[ticker] = float(np.prod(1.0 + r_asset) - 1.0)

    return HistoricalScenarioResult(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        start_date=start_date,
        end_date=end_date,
        portfolio_name=portfolio.name,
        trading_days=n_days,
        cumulative_return=cum_ret,
        cumulative_loss=cum_loss,
        max_drawdown=max_dd,
        asset_cumulative_returns=asset_cum,
    )


def run_all_historical_scenarios(
    returns: pd.DataFrame,
    portfolio: Portfolio,
    scenarios_config: Dict[str, Any],
) -> List[HistoricalScenarioResult]:
    """Execute all configured historical replay scenarios.

    Args:
        returns: Returns DataFrame.
        portfolio: Portfolio instance.
        scenarios_config: Configuration dictionary with historical scenarios.

    Returns:
        List[HistoricalScenarioResult]: Evaluated scenarios.
    """
    results: List[HistoricalScenarioResult] = []
    hist_cfg = scenarios_config.get("historical", {})

    for sc_id, sc_def in hist_cfg.items():
        try:
            res = replay_historical_scenario(
                returns=returns,
                portfolio=portfolio,
                scenario_id=sc_id,
                scenario_name=sc_def.get("name", sc_id),
                start_date=sc_def["start_date"],
                end_date=sc_def["end_date"],
            )
            results.append(res)
        except Exception as exc:
            logger.warning("Could not replay historical scenario '%s': %s", sc_id, exc)

    return results
