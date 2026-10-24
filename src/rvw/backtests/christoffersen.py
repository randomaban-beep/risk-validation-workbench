"""Christoffersen (1998) Independence and Conditional Coverage Likelihood Ratio tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
from scipy import stats

from rvw.backtests.kupiec import kupiec_pof_test


@dataclass
class IndependenceResult:
    """Results from Christoffersen Independence test."""
    test_name: str
    transition_counts: Dict[str, int]
    transition_probabilities: Dict[str, float]
    lr_stat: float
    p_value: float
    reject_null: bool
    significance_level: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_name": self.test_name,
            "transition_counts": self.transition_counts,
            "transition_probabilities": self.transition_probabilities,
            "lr_stat": float(self.lr_stat),
            "p_value": float(self.p_value),
            "reject_null": bool(self.reject_null),
            "significance_level": self.significance_level,
        }


@dataclass
class ConditionalCoverageResult:
    """Results from Christoffersen Conditional Coverage joint test."""
    test_name: str
    kupiec_lr: float
    independence_lr: float
    lr_stat: float
    p_value: float
    reject_null: bool
    degrees_of_freedom: int = 2
    significance_level: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_name": self.test_name,
            "kupiec_lr": float(self.kupiec_lr),
            "independence_lr": float(self.independence_lr),
            "lr_stat": float(self.lr_stat),
            "p_value": float(self.p_value),
            "reject_null": bool(self.reject_null),
            "degrees_of_freedom": self.degrees_of_freedom,
            "significance_level": self.significance_level,
        }


def compute_transition_counts(hits: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute first-order Markov transition counts (T00, T01, T10, T11)."""
    t00 = t01 = t10 = t11 = 0
    for i in range(len(hits) - 1):
        prev, curr = hits[i], hits[i + 1]
        if prev == 0 and curr == 0:
            t00 += 1
        elif prev == 0 and curr == 1:
            t01 += 1
        elif prev == 1 and curr == 0:
            t10 += 1
        elif prev == 1 and curr == 1:
            t11 += 1
    return t00, t01, t10, t11


def christoffersen_independence_test(
    hits: np.ndarray | list[bool],
    significance_level: float = 0.05,
) -> IndependenceResult:
    """Execute Christoffersen (1998) test for exception independence.

    Tests whether hits are serially independent against a first-order Markov alternative.
    Null Hypothesis H0: pi_01 = pi_11 = pi (independence).
    Test statistic:
        LR_ind = -2 * ln( L(H0) / L(H1) ) ~ Chi2(1)

    Args:
        hits: Binary array of hit indicators (1=hit, 0=no hit).
        significance_level: Alpha threshold for hypothesis rejection (default 0.05).

    Returns:
        IndependenceResult: Statistical test outcome.
    """
    arr = np.asarray(hits, dtype=int)
    N = len(arr)
    if N < 2:
        raise ValueError("Independence test requires at least 2 observations.")

    t00, t01, t10, t11 = compute_transition_counts(arr)

    # Edge cases: no transitions or zero exceptions
    if (t01 + t11) == 0 or (t00 + t10) == 0:
        return IndependenceResult(
            test_name="Christoffersen Independence",
            transition_counts={"T00": t00, "T01": t01, "T10": t10, "T11": t11},
            transition_probabilities={"pi01": 0.0, "pi11": 0.0, "pi": 0.0},
            lr_stat=0.0,
            p_value=1.0,
            reject_null=False,
            significance_level=significance_level,
        )

    # Probabilities under H1
    pi01 = t01 / (t00 + t01) if (t00 + t01) > 0 else 0.0
    pi11 = t11 / (t10 + t11) if (t10 + t11) > 0 else 0.0

    # Unconditional probability under H0
    pi = (t01 + t11) / (t00 + t01 + t10 + t11)

    # Log-likelihood under H0
    log_l0 = (t00 + t10) * np.log(1.0 - pi) + (t01 + t11) * np.log(pi)

    # Log-likelihood under H1 (handling 0 * log(0) -> 0 terms)
    log_l1 = 0.0
    if t00 > 0:
        log_l1 += t00 * np.log(1.0 - pi01)
    if t01 > 0:
        log_l1 += t01 * np.log(pi01)
    if t10 > 0:
        log_l1 += t10 * np.log(1.0 - pi11)
    if t11 > 0:
        log_l1 += t11 * np.log(pi11)

    lr_ind = max(0.0, float(-2.0 * (log_l0 - log_l1)))
    p_value = float(stats.chi2.sf(lr_ind, df=1))
    reject = bool(p_value < significance_level)

    return IndependenceResult(
        test_name="Christoffersen Independence",
        transition_counts={"T00": t00, "T01": t01, "T10": t10, "T11": t11},
        transition_probabilities={"pi01": float(pi01), "pi11": float(pi11), "pi": float(pi)},
        lr_stat=lr_ind,
        p_value=p_value,
        reject_null=reject,
        significance_level=significance_level,
    )


def christoffersen_conditional_coverage_test(
    hits: np.ndarray | list[bool],
    confidence_level: float = 0.99,
    significance_level: float = 0.05,
) -> ConditionalCoverageResult:
    """Execute Christoffersen (1998) joint Conditional Coverage test.

    LR_cc = LR_uc + LR_ind ~ Chi2(2)

    Jointly tests both unconditional coverage and exception independence.

    Args:
        hits: Binary array of hits.
        confidence_level: VaR confidence level.
        significance_level: Alpha rejection threshold (default 0.05).

    Returns:
        ConditionalCoverageResult: Statistical test outcome with chi2(2) p-value.
    """
    kupiec = kupiec_pof_test(hits, confidence_level=confidence_level, significance_level=significance_level)
    indep = christoffersen_independence_test(hits, significance_level=significance_level)

    lr_cc = float(kupiec.lr_stat + indep.lr_stat)
    p_value = float(stats.chi2.sf(lr_cc, df=2))
    reject = bool(p_value < significance_level)

    return ConditionalCoverageResult(
        test_name="Christoffersen Conditional Coverage",
        kupiec_lr=kupiec.lr_stat,
        independence_lr=indep.lr_stat,
        lr_stat=lr_cc,
        p_value=p_value,
        reject_null=reject,
        degrees_of_freedom=2,
        significance_level=significance_level,
    )
