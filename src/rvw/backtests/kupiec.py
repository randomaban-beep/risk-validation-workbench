"""Kupiec Proportion of Failures (POF) Unconditional Coverage Likelihood Ratio test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from scipy import stats


@dataclass
class KupiecResult:
    """Results from Kupiec POF test."""
    test_name: str
    n_observations: int
    n_exceptions: int
    expected_exceptions: float
    confidence_level: float
    target_rate: float
    actual_rate: float
    lr_stat: float
    p_value: float
    reject_null: bool  # True if p_value < alpha_significance (default 0.05)
    significance_level: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_name": self.test_name,
            "n_observations": self.n_observations,
            "n_exceptions": self.n_exceptions,
            "expected_exceptions": self.expected_exceptions,
            "confidence_level": self.confidence_level,
            "target_rate": self.target_rate,
            "actual_rate": self.actual_rate,
            "lr_stat": float(self.lr_stat),
            "p_value": float(self.p_value),
            "reject_null": bool(self.reject_null),
            "significance_level": self.significance_level,
        }


def kupiec_pof_test(
    hits: np.ndarray | list[bool],
    confidence_level: float = 0.99,
    significance_level: float = 0.05,
) -> KupiecResult:
    """Execute the Kupiec (1995) Proportion of Failures (POF) unconditional coverage test.

    Null Hypothesis H0: Realized exception probability equals p0 = 1 - confidence_level.
    Test statistic:
        LR_uc = -2 * ln( (p0^x * (1 - p0)^(N - x)) / (p_hat^x * (1 - p_hat)^(N - x)) )
        LR_uc ~ Chi2(1)

    Handles edge cases x = 0 and x = N without numerical divergence.

    Args:
        hits: Binary array/series indicating whether realized loss exceeded VaR (1=hit, 0=no hit).
        confidence_level: VaR confidence level (e.g. 0.95 or 0.99).
        significance_level: Alpha threshold for hypothesis rejection (default 0.05).

    Returns:
        KupiecResult: Structured statistical test result.
    """
    arr = np.asarray(hits, dtype=int)
    N = len(arr)
    if N == 0:
        raise ValueError("Cannot perform Kupiec POF test on empty hit sequence.")

    x = int(np.sum(arr))
    p0 = 1.0 - confidence_level
    expected = N * p0
    p_hat = x / N

    # Edge cases
    if x == 0:
        # p_hat = 0: lim x * ln(p_hat) -> 0
        lr = -2.0 * N * np.log(1.0 - p0)
    elif x == N:
        # p_hat = 1: lim (N - x) * ln(1 - p_hat) -> 0
        lr = -2.0 * N * np.log(p0)
    else:
        # Standard Kupiec formula
        term1 = x * np.log(p_hat / p0)
        term2 = (N - x) * np.log((1.0 - p_hat) / (1.0 - p0))
        lr = 2.0 * (term1 + term2)

    lr_stat = max(0.0, float(lr))
    # Chi-squared survival function with 1 degree of freedom
    p_value = float(stats.chi2.sf(lr_stat, df=1))
    reject = bool(p_value < significance_level)

    return KupiecResult(
        test_name="Kupiec Unconditional Coverage (POF)",
        n_observations=N,
        n_exceptions=x,
        expected_exceptions=expected,
        confidence_level=confidence_level,
        target_rate=p0,
        actual_rate=p_hat,
        lr_stat=lr_stat,
        p_value=p_value,
        reject_null=reject,
        significance_level=significance_level,
    )
