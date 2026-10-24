"""McNeil-Frey (2000) Expected Shortfall exceedance-residual backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
from scipy import stats


@dataclass
class McNeilFreyResult:
    """Outcome of McNeil-Frey Expected Shortfall exceedance residual test."""
    test_name: str
    n_violations: int
    mean_excess_loss: float
    std_excess_loss: float
    t_stat: float
    p_value_two_sided: float
    p_value_one_sided: float  # H1: mean > 0 (underestimating tail severity)
    reject_null: bool
    assumptions_doc: str
    significance_level: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            "test_name": self.test_name,
            "n_violations": self.n_violations,
            "mean_excess_loss": float(self.mean_excess_loss),
            "std_excess_loss": float(self.std_excess_loss),
            "t_stat": float(self.t_stat),
            "p_value_two_sided": float(self.p_value_two_sided),
            "p_value_one_sided": float(self.p_value_one_sided),
            "reject_null": bool(self.reject_null),
            "assumptions_doc": self.assumptions_doc,
            "significance_level": self.significance_level,
        }


def mcneil_frey_es_test(
    losses: np.ndarray,
    var_forecasts: np.ndarray,
    es_forecasts: np.ndarray,
    significance_level: float = 0.05,
) -> McNeilFreyResult:
    """Execute McNeil and Frey (2000) test on Expected Shortfall exceedance residuals.

    Assumptions & Methodology:
    Conditioned on an exception occurring (L_t > VaR_t), the exceedance residual
    is defined as e_t = L_t - ES_t. Under the null hypothesis of a correctly specified
    tail distribution, e_t has expectation zero:
        H0: E[L_t - ES_t | L_t > VaR_t] = 0
    against the one-sided alternative H1: E[e_t] > 0 (tail losses systematically exceed ES forecasts).

    Args:
        losses: Realized loss series L_t = -R_{p, t}.
        var_forecasts: Corresponding VaR forecasts.
        es_forecasts: Corresponding ES forecasts.
        significance_level: Alpha threshold for hypothesis rejection (default 0.05).

    Returns:
        McNeilFreyResult: Statistical test outcome.
    """
    assumptions = (
        "McNeil-Frey (2000): Tests H0 that mean exceedance residual E[L - ES | L > VaR] = 0. "
        "Assumes i.i.d. exceedance residuals with finite variance. One-sided p-value tests H1: E[residual] > 0."
    )

    losses_arr = np.asarray(losses, dtype=float)
    var_arr = np.asarray(var_forecasts, dtype=float)
    es_arr = np.asarray(es_forecasts, dtype=float)

    # Filter to exception days
    mask = (losses_arr > var_arr) & ~np.isnan(losses_arr) & ~np.isnan(var_arr) & ~np.isnan(es_arr)
    violations_losses = losses_arr[mask]
    violations_es = es_arr[mask]
    k = len(violations_losses)

    if k < 2:
        return McNeilFreyResult(
            test_name="McNeil-Frey Expected Shortfall Test",
            n_violations=k,
            mean_excess_loss=0.0 if k == 0 else float(violations_losses[0] - violations_es[0]),
            std_excess_loss=0.0,
            t_stat=0.0,
            p_value_two_sided=1.0,
            p_value_one_sided=1.0,
            reject_null=False,
            assumptions_doc=assumptions + " (Inconclusive: fewer than 2 exceptions).",
            significance_level=significance_level,
        )

    residuals = violations_losses - violations_es
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals, ddof=1))

    if std_res <= 1e-8:
        t_stat = 0.0
        p_two = 1.0
        p_one = 1.0
    else:
        se = std_res / np.sqrt(k)
        t_stat = mean_res / se
        # One-sided test (H1: mean > 0)
        p_one = float(stats.t.sf(t_stat, df=k - 1))
        # Two-sided test
        p_two = float(2.0 * stats.t.sf(abs(t_stat), df=k - 1))

    reject = bool(p_one < significance_level)

    return McNeilFreyResult(
        test_name="McNeil-Frey Expected Shortfall Test",
        n_violations=k,
        mean_excess_loss=mean_res,
        std_excess_loss=std_res,
        t_stat=float(t_stat),
        p_value_two_sided=p_two,
        p_value_one_sided=p_one,
        reject_null=reject,
        assumptions_doc=assumptions,
        significance_level=significance_level,
    )
