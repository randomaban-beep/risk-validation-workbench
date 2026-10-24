"""Portfolio representation, validation, and return aggregation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.exceptions import ConfigurationError


@dataclass
class Portfolio:
    """Represents an investment portfolio with validated asset weights.

    Attributes:
        name: Human-readable portfolio designation.
        weights: Pandas Series indexed by ticker symbols with allocation fractions.
    """
    weights: pd.Series
    name: str = "custom"

    def __post_init__(self) -> None:
        """Validate portfolio asset weights."""
        if not isinstance(self.weights, pd.Series):
            self.weights = pd.Series(self.weights)

        # Drop zero or negligible weights
        self.weights = self.weights[self.weights.abs() > 1e-8].astype(float)

        if self.weights.empty:
            raise ConfigurationError(f"Portfolio '{self.name}' has no active asset weights.")

        # Check weight sum (tolerance 1e-4)
        total_weight = float(self.weights.sum())
        if not np.isclose(total_weight, 1.0, atol=1e-4):
            raise ConfigurationError(
                f"Portfolio '{self.name}' weights must sum to 1.0 (got {total_weight:.6f})."
            )

        # Normalize slightly to ensure exact unity
        self.weights = self.weights / total_weight

    @property
    def tickers(self) -> List[str]:
        """List of active ticker symbols."""
        return list(self.weights.index)

    def to_dict(self) -> Dict[str, float]:
        """Convert weights to dictionary mapping."""
        return {str(k): float(v) for k, v in self.weights.items()}


def create_preset_portfolio(
    preset_name: str,
    preset_dict: Dict[str, Dict[str, float]],
) -> Portfolio:
    """Construct a Portfolio instance from configuration preset definitions.

    Args:
        preset_name: Key of the preset (e.g. 'equal_weight', 'sixty_forty').
        preset_dict: Mapping of preset names to ticker weight dictionaries.

    Returns:
        Portfolio: Instantiated and validated Portfolio object.

    Raises:
        ConfigurationError: If preset_name does not exist in preset_dict.
    """
    if preset_name not in preset_dict:
        available = list(preset_dict.keys())
        raise ConfigurationError(
            f"Unknown portfolio preset '{preset_name}'. Available presets: {available}"
        )

    weight_map = preset_dict[preset_name]
    weights_series = pd.Series(weight_map)
    return Portfolio(weights=weights_series, name=preset_name)


def portfolio_returns(
    returns: pd.DataFrame,
    portfolio: Portfolio,
) -> pd.Series:
    r"""Calculate daily portfolio simple returns assuming daily constant-weight rebalancing.

    R_{p, t} = \sum_{i=1}^N w_i \cdot R_{i, t}

    Args:
        returns: Asset simple returns DataFrame with DatetimeIndex.
        portfolio: Target portfolio holding asset weights.

    Returns:
        pd.Series: Daily portfolio returns series with DatetimeIndex and name.

    Raises:
        ConfigurationError: If any asset required by the portfolio is missing from returns.
    """
    missing_assets = set(portfolio.tickers) - set(returns.columns)
    if missing_assets:
        raise ConfigurationError(
            f"Returns DataFrame is missing required portfolio assets: {sorted(missing_assets)}"
        )

    aligned_returns = returns[portfolio.tickers]
    weights = portfolio.weights[portfolio.tickers].values

    port_ret = aligned_returns.values @ weights
    series = pd.Series(port_ret, index=aligned_returns.index, name=f"Return_{portfolio.name}")
    series.attrs = dict(returns.attrs)
    series.attrs["portfolio_name"] = portfolio.name
    return series
