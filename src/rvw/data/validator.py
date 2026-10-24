"""Data quality validation module for prices and return time series."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.config import DataValidationConfig


@dataclass
class QualityIssue:
    """Representation of an identified data anomaly."""
    category: str
    ticker: Optional[str]
    date: Optional[str]
    message: str
    severity: str  # "ERROR", "WARNING", "INFO"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Structured report produced by validate_prices."""
    is_valid: bool
    total_rows: int
    total_columns: int
    tickers: List[str]
    start_date: str
    end_date: str
    issues: List[QualityIssue] = field(default_factory=list)
    metrics_by_asset: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_summary_df(self) -> pd.DataFrame:
        """Convert per-asset metrics to a summary pandas DataFrame."""
        if not self.metrics_by_asset:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(self.metrics_by_asset, orient="index")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily arithmetic returns from daily adjusted close prices.

    Return convention:
        R_t = (P_t - P_{t-1}) / P_{t-1}

    Args:
        prices: Daily price matrix with DatetimeIndex and asset columns.

    Returns:
        pd.DataFrame: Simple daily returns aligned to dates 1 to T.
    """
    if len(prices) <= 1:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns.attrs = dict(prices.attrs)
    return returns


def validate_prices(
    prices: pd.DataFrame,
    config: Optional[DataValidationConfig] = None,
    expected_start_date: Optional[str] = None,
) -> DataQualityReport:
    """Execute comprehensive market data integrity validation checks.

    Detects and logs:
    1. Missing values (NaNs, null values).
    2. Duplicate dates in index.
    3. Non-monotonic index ordering.
    4. Non-positive prices (P <= 0).
    5. Extreme daily returns (|r| > max_return_jump).
    6. Stale prices (consecutive zero returns >= max_stale_days).
    7. Excessive calendar gaps (> max_calendar_gap_days business days).
    8. History truncation (assets starting after expected_start_date).

    Args:
        prices: Price DataFrame with DatetimeIndex.
        config: DataValidationConfig with threshold settings.
        expected_start_date: Expected start date (YYYY-MM-DD).

    Returns:
        DataQualityReport: Comprehensive structured validation audit report.
    """
    if config is None:
        config = DataValidationConfig()

    issues: List[QualityIssue] = []
    has_critical_error = False

    if prices.empty:
        issues.append(QualityIssue(
            category="EMPTY_DATASET",
            ticker=None,
            date=None,
            message="Provided price dataset is completely empty.",
            severity="ERROR",
        ))
        return DataQualityReport(
            is_valid=False,
            total_rows=0,
            total_columns=0,
            tickers=[],
            start_date="",
            end_date="",
            issues=issues,
            metrics_by_asset={},
        )

    # 1. Duplicate dates
    if prices.index.duplicated().any():
        dup_count = int(prices.index.duplicated().sum())
        issues.append(QualityIssue(
            category="DUPLICATE_DATES",
            ticker=None,
            date=None,
            message=f"Found {dup_count} duplicate date rows in price index.",
            severity="ERROR",
            details={"duplicate_count": dup_count},
        ))
        has_critical_error = True

    # 2. Monotonic index
    if not prices.index.is_monotonic_increasing:
        issues.append(QualityIssue(
            category="NON_MONOTONIC_INDEX",
            ticker=None,
            date=None,
            message="Price index is not strictly monotonically increasing.",
            severity="ERROR",
        ))
        has_critical_error = True

    # 3. Calendar gaps (> max_calendar_gap_days business days)
    if len(prices.index) > 1:
        date_diffs = pd.Series(prices.index).diff().dt.days
        # Weekends typically give 3 days. A business gap of > max_calendar_gap_days business days is > max_calendar_gap_days * 7 / 5
        large_gaps = date_diffs[date_diffs > (config.max_calendar_gap_days * 2)]
        for idx_pos, gap_len in large_gaps.items():
            gap_date = str(prices.index[idx_pos].date())
            issues.append(QualityIssue(
                category="CALENDAR_GAP",
                ticker=None,
                date=gap_date,
                message=f"Calendar gap of {gap_len} calendar days detected prior to {gap_date}.",
                severity="WARNING",
                details={"gap_days": int(gap_len), "date": gap_date},
            ))

    # Compute returns for return-based checks
    returns = compute_returns(prices)

    metrics_by_asset: Dict[str, Dict[str, Any]] = {}
    tickers = list(prices.columns)

    for ticker in tickers:
        series = prices[ticker]
        ret_series = returns[ticker] if ticker in returns.columns else pd.Series(dtype=float)

        nan_count = int(series.isna().sum())
        non_positive_count = int((series <= 0.0).sum())
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()

        first_date_str = str(first_valid.date()) if first_valid is not None else None
        last_date_str = str(last_valid.date()) if last_valid is not None else None

        # Check late start date
        if expected_start_date and first_valid is not None:
            exp_dt = pd.to_datetime(expected_start_date)
            if first_valid > exp_dt + pd.Timedelta(days=14):
                issues.append(QualityIssue(
                    category="LATE_START_DATE",
                    ticker=ticker,
                    date=first_date_str,
                    message=f"History for {ticker} starts at {first_date_str}, later than expected {expected_start_date}.",
                    severity="WARNING",
                    details={"actual_start": first_date_str, "expected_start": expected_start_date},
                ))

        # Check missing values
        if nan_count > 0:
            issues.append(QualityIssue(
                category="MISSING_VALUES",
                ticker=ticker,
                date=None,
                message=f"Asset {ticker} has {nan_count} missing (NaN) price observations.",
                severity="ERROR",
                details={"nan_count": nan_count},
            ))
            has_critical_error = True

        # Check non-positive prices
        if non_positive_count > 0:
            issues.append(QualityIssue(
                category="NON_POSITIVE_PRICE",
                ticker=ticker,
                date=None,
                message=f"Asset {ticker} has {non_positive_count} non-positive (<=0) price values.",
                severity="ERROR",
                details={"non_positive_count": non_positive_count},
            ))
            has_critical_error = True

        # Extreme returns jumps (|r| > max_return_jump)
        extreme_jumps = ret_series[ret_series.abs() > config.max_return_jump]
        extreme_count = len(extreme_jumps)
        if extreme_count > 0:
            for j_date, j_ret in extreme_jumps.items():
                issues.append(QualityIssue(
                    category="EXTREME_RETURN_JUMP",
                    ticker=ticker,
                    date=str(j_date.date()),
                    message=f"Extreme return jump of {j_ret:.2%} on {j_date.date()} exceeds threshold {config.max_return_jump:.2%}.",
                    severity="WARNING",
                    details={"return": float(j_ret), "date": str(j_date.date())},
                ))

        # Stale prices (consecutive zero returns)
        zero_runs = (ret_series == 0.0).astype(int)
        stale_streak = 0
        max_zero_streak = 0
        for val in zero_runs:
            if val == 1:
                stale_streak += 1
                if stale_streak > max_zero_streak:
                    max_zero_streak = stale_streak
            else:
                stale_streak = 0

        if max_zero_streak >= config.max_stale_days:
            issues.append(QualityIssue(
                category="STALE_PRICES",
                ticker=ticker,
                date=None,
                message=f"Asset {ticker} exhibits {max_zero_streak} consecutive zero returns (stale pricing).",
                severity="WARNING",
                details={"max_consecutive_zero_returns": max_zero_streak},
            ))

        # Populate summary metrics
        metrics_by_asset[ticker] = {
            "first_date": first_date_str,
            "last_date": last_date_str,
            "total_obs": int(len(series)),
            "nan_count": nan_count,
            "non_positive_count": non_positive_count,
            "extreme_jumps": extreme_count,
            "max_stale_days": max_zero_streak,
            "mean_price": float(series.mean()) if nan_count < len(series) else 0.0,
            "annualized_vol": float(ret_series.std() * np.sqrt(252)) if len(ret_series) > 1 else 0.0,
        }

    start_str = str(prices.index.min().date())
    end_str = str(prices.index.max().date())

    return DataQualityReport(
        is_valid=not has_critical_error,
        total_rows=len(prices),
        total_columns=len(tickers),
        tickers=tickers,
        start_date=start_str,
        end_date=end_str,
        issues=issues,
        metrics_by_asset=metrics_by_asset,
    )
