"""Unit tests and hand-derived numerical checks for statistical backtests."""

import numpy as np
import pandas as pd
import pytest

from rvw.backtests.basel import classify_basel_zone
from rvw.backtests.christoffersen import (
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    compute_transition_counts,
)
from rvw.backtests.kupiec import kupiec_pof_test
from rvw.backtests.mcneil_frey import mcneil_frey_es_test


def test_kupiec_pof_exact_arithmetic():
    """Verify Kupiec test matches exact hand-calculated reference value:

    N=250, x=5, p=0.01 -> LR ≈ 1.957, p-value ≈ 0.1619.
    """
    # 5 ones and 245 zeros
    hits = np.zeros(250, dtype=int)
    hits[:5] = 1

    res = kupiec_pof_test(hits, confidence_level=0.99)

    assert res.n_observations == 250
    assert res.n_exceptions == 5
    assert np.isclose(res.lr_stat, 1.95669, atol=1e-3)
    assert np.isclose(res.p_value, 0.16186, atol=1e-3)
    assert not res.reject_null


def test_kupiec_edge_cases_zero_and_all():
    """Verify Kupiec POF handles boundary cases x=0 and x=N without crashing or NaN."""
    # Case x=0
    hits_zero = np.zeros(100, dtype=int)
    res_zero = kupiec_pof_test(hits_zero, confidence_level=0.99)
    assert res_zero.n_exceptions == 0
    assert not np.isnan(res_zero.lr_stat)
    assert not np.isnan(res_zero.p_value)

    # Case x=N
    hits_all = np.ones(50, dtype=int)
    res_all = kupiec_pof_test(hits_all, confidence_level=0.99)
    assert res_all.n_exceptions == 50
    assert not np.isnan(res_all.lr_stat)
    assert res_all.reject_null is True


def test_christoffersen_transition_matrix_counting():
    """Verify hand-countable transition matrix:

    Sequence: 0, 0, 1, 1, 0, 1, 0, 0
    Transitions:
    (0->0): 2
    (0->1): 2
    (1->1): 1
    (1->0): 2
    """
    seq = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    t00, t01, t10, t11 = compute_transition_counts(seq)

    assert t00 == 2
    assert t01 == 2
    assert t11 == 1
    assert t10 == 2


def test_christoffersen_independence_clustering_rejection():
    """Verify that evenly spaced hits are not rejected, while severely clustered hits are rejected."""
    # Evenly spaced hits: 0,0,0,0,1, 0,0,0,0,1 ... (no consecutive hits)
    pattern_iid = ([0] * 9 + [1]) * 10
    res_iid = christoffersen_independence_test(pattern_iid)
    assert not res_iid.reject_null

    # Heavily clustered hits: 0...0 then 1,1,1,1,1,1,1,1,1,1
    pattern_clustered = [0] * 90 + [1] * 10
    res_clustered = christoffersen_independence_test(pattern_clustered)
    assert res_clustered.reject_null is True
    assert res_clustered.lr_stat > 3.841  # 5% chi2(1) critical value


def test_conditional_coverage_additivity():
    """Verify LR_cc = LR_uc + LR_ind and chi2(2) degrees of freedom."""
    hits = np.zeros(200, dtype=int)
    hits[10:15] = 1  # 5 clustered hits

    res_cc = christoffersen_conditional_coverage_test(hits, confidence_level=0.99)
    res_uc = kupiec_pof_test(hits, confidence_level=0.99)
    res_ind = christoffersen_independence_test(hits)

    assert np.isclose(res_cc.lr_stat, res_uc.lr_stat + res_ind.lr_stat, atol=1e-8)
    assert res_cc.degrees_of_freedom == 2


def test_basel_traffic_light_boundaries():
    """Verify standard regulatory boundary values over 250 days:

    0-4 -> Green
    5-9 -> Yellow
    10+ -> Red
    """
    assert classify_basel_zone(0).zone == "Green"
    assert classify_basel_zone(4).zone == "Green"
    assert classify_basel_zone(5).zone == "Yellow"
    assert classify_basel_zone(9).zone == "Yellow"
    assert classify_basel_zone(10).zone == "Red"
    assert classify_basel_zone(20).zone == "Red"


def test_mcneil_frey_es_test():
    """Verify McNeil-Frey exceedance residual test detects tail underestimation."""
    losses = np.array([0.01, 0.02, 0.05, 0.06, 0.07, 0.01])
    var = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03])
    # Model underpredicts ES (forecasts 0.035 but realized tail average is 0.06)
    es = np.array([0.035, 0.035, 0.035, 0.035, 0.035, 0.035])

    res = mcneil_frey_es_test(losses, var, es)
    assert res.n_violations == 3  # indices 2, 3, 4
    assert res.mean_excess_loss > 0.0
    assert res.t_stat > 0.0
    assert "McNeil-Frey" in res.test_name
