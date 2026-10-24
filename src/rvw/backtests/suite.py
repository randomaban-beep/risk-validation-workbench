"""Comprehensive backtesting evaluation suite and comparative matrix generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.backtests.basel import BaselZoneResult, classify_basel_zone
from rvw.backtests.christoffersen import (
    ConditionalCoverageResult,
    IndependenceResult,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
)
from rvw.backtests.kupiec import KupiecResult, kupiec_pof_test
from rvw.backtests.mcneil_frey import McNeilFreyResult, mcneil_frey_es_test
from rvw.walkforward.engine import ForecastFrame, format_alpha_col


@dataclass
class ModelBacktestSummary:
    """Consolidated backtest metrics for a single model at a given alpha level.

    Basel traffic-light fields:
      - ``basel_zone_worst``: worst zone over any rolling 250-day window (standard interpretation).
      - ``basel_zone_latest``: zone of the final 250-day rolling window.
      - ``basel_zone_full_period_scaled``: zone when exceptions and window length are both the full
        test-period length (scaled cutoffs); labelled "full-period scaled, for illustration only".
    """
    model_name: str
    alpha: float
    observations: int
    exceptions: int
    expected_exceptions: float
    exception_rate: float
    kupiec_lr: float
    kupiec_p: float
    independence_lr: float
    independence_p: float
    conditional_coverage_lr: float
    conditional_coverage_p: float
    # Standard Basel: worst and latest rolling-250-day window zones
    basel_zone_worst: str
    basel_zone_latest: str
    # Full-period scaled (non-standard, labelled for illustration only)
    basel_zone_full_period_scaled: str
    es_alpha: float
    mcneil_frey_stat: float
    mcneil_frey_p: float
    mean_var: float
    mean_es: float
    failed_forecast_count: int
    fallback_count: int

    # Backward-compatible alias: worst rolling window is the primary reported zone
    @property
    def basel_zone(self) -> str:
        """Primary Basel zone: worst rolling-250-day window (standard interpretation)."""
        return self.basel_zone_worst

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "model_name": self.model_name,
            "alpha": self.alpha,
            "observations": self.observations,
            "exceptions": self.exceptions,
            "expected_exceptions": round(self.expected_exceptions, 2),
            "exception_rate": round(self.exception_rate, 4),
            "kupiec_lr": round(self.kupiec_lr, 3),
            "kupiec_p": round(self.kupiec_p, 4),
            "independence_lr": round(self.independence_lr, 3),
            "independence_p": round(self.independence_p, 4),
            "conditional_coverage_lr": round(self.conditional_coverage_lr, 3),
            "conditional_coverage_p": round(self.conditional_coverage_p, 4),
            # Standard Basel rolling windows
            "basel_zone": self.basel_zone_worst,          # primary: worst rolling window
            "basel_zone_worst": self.basel_zone_worst,
            "basel_zone_latest": self.basel_zone_latest,
            "basel_zone_full_period_scaled": self.basel_zone_full_period_scaled,  # non-standard, illustration
            "es_alpha": self.es_alpha,
            "mcneil_frey_stat": round(self.mcneil_frey_stat, 3),
            "mcneil_frey_p": round(self.mcneil_frey_p, 4),
            "mean_var": round(self.mean_var, 4),
            "mean_es": round(self.mean_es, 4),
            "failed_forecast_count": self.failed_forecast_count,
            "fallback_count": self.fallback_count,
        }


def evaluate_backtests(
    forecast_frame: ForecastFrame,
    var_alpha: float = 0.99,
    es_alpha: float = 0.99,
    significance_level: float = 0.05,
) -> ModelBacktestSummary:
    """Run full statistical backtesting suite on a ForecastFrame.

    Args:
        forecast_frame: Generated walk-forward forecast container.
        var_alpha: VaR confidence level to test (default 0.99).
        es_alpha: ES confidence level to test (default 0.99).
        significance_level: Alpha threshold for hypothesis rejections.

    Returns:
        ModelBacktestSummary: Comprehensive test metrics.
    """
    col_str = format_alpha_col(var_alpha)
    es_col_str = format_alpha_col(es_alpha)

    if col_str not in forecast_frame.hits.columns:
        raise KeyError(f"VaR alpha {var_alpha} (col '{col_str}') not found in ForecastFrame.")

    hits = forecast_frame.hits[col_str].values
    losses = forecast_frame.realized_losses.values
    var_series = forecast_frame.var_forecasts[col_str].values
    es_series = forecast_frame.es_forecasts[es_col_str].values if es_col_str in forecast_frame.es_forecasts else var_series

    # Filter valid pairs
    valid_mask = ~np.isnan(var_series) & ~np.isnan(losses)
    valid_hits = hits[valid_mask]
    valid_losses = losses[valid_mask]
    valid_var = var_series[valid_mask]
    valid_es = es_series[valid_mask]

    n_obs = len(valid_hits)
    n_exc = int(np.sum(valid_hits))

    # 1. Kupiec POF
    kupiec = kupiec_pof_test(valid_hits, confidence_level=var_alpha, significance_level=significance_level)

    # 2. Christoffersen Independence
    indep = christoffersen_independence_test(valid_hits, significance_level=significance_level)

    # 3. Christoffersen Conditional Coverage
    cc = christoffersen_conditional_coverage_test(valid_hits, confidence_level=var_alpha, significance_level=significance_level)

    # 4. Basel Traffic Light — three flavours
    # (a) Full-period scaled (non-standard — window scales with n_obs): labelled illustration only
    basel_full = classify_basel_zone(exceptions=n_exc, window_length=n_obs, confidence_level=var_alpha)

    # (b) Rolling 250-day windows (standard Basel interpretation)
    ROLLING_WINDOW = 250
    hits_series = pd.Series(valid_hits.astype(int))
    if len(hits_series) >= ROLLING_WINDOW:
        rolling_exc = hits_series.rolling(window=ROLLING_WINDOW).sum().dropna().astype(int)
        zone_series = rolling_exc.apply(
            lambda x: classify_basel_zone(int(x), window_length=ROLLING_WINDOW, confidence_level=var_alpha).zone
        )
        # Worst zone: Red > Yellow > Green
        _zone_rank = {"Green": 0, "Yellow": 1, "Red": 2}
        worst_zone = max(zone_series, key=lambda z: _zone_rank.get(z, 0))
        latest_zone = zone_series.iloc[-1]
    else:
        # Not enough data for even one 250-day window; fall back to full-period scaled
        worst_zone = basel_full.zone
        latest_zone = basel_full.zone

    # 5. McNeil-Frey ES test
    mf = mcneil_frey_es_test(valid_losses, valid_var, valid_es, significance_level=significance_level)

    return ModelBacktestSummary(
        model_name=forecast_frame.model_name,
        alpha=var_alpha,
        observations=n_obs,
        exceptions=n_exc,
        expected_exceptions=kupiec.expected_exceptions,
        exception_rate=kupiec.actual_rate,
        kupiec_lr=kupiec.lr_stat,
        kupiec_p=kupiec.p_value,
        independence_lr=indep.lr_stat,
        independence_p=indep.p_value,
        conditional_coverage_lr=cc.lr_stat,
        conditional_coverage_p=cc.p_value,
        basel_zone_worst=worst_zone,
        basel_zone_latest=latest_zone,
        basel_zone_full_period_scaled=basel_full.zone,
        es_alpha=es_alpha,
        mcneil_frey_stat=mf.t_stat,
        mcneil_frey_p=mf.p_value_one_sided,
        mean_var=float(np.mean(valid_var)) if len(valid_var) > 0 else 0.0,
        mean_es=float(np.mean(valid_es)) if len(valid_es) > 0 else 0.0,
        failed_forecast_count=len(forecast_frame.failed_dates),
        fallback_count=forecast_frame.fallback_count,
    )


def compare_models(
    forecast_frames: List[ForecastFrame],
    var_alpha: float = 0.99,
    es_alpha: float = 0.99,
) -> pd.DataFrame:
    """Generate a side-by-side comparative backtesting table across multiple models.

    Args:
        forecast_frames: List of ForecastFrames (one per model).
        var_alpha: Target VaR confidence level.
        es_alpha: Target ES confidence level.

    Returns:
        pd.DataFrame: Comparative benchmark DataFrame.
    """
    rows = []
    for ff in forecast_frames:
        summary = evaluate_backtests(ff, var_alpha=var_alpha, es_alpha=es_alpha)
        rows.append(summary.to_dict())

    df = pd.DataFrame(rows)
    df.set_index("model_name", inplace=True)
    return df
