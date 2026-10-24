"""Statistical and regulatory backtesting suite package."""

from rvw.backtests.basel import BaselZoneResult, classify_basel_zone, compute_rolling_traffic_light
from rvw.backtests.christoffersen import (
    ConditionalCoverageResult,
    IndependenceResult,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
)
from rvw.backtests.kupiec import KupiecResult, kupiec_pof_test
from rvw.backtests.mcneil_frey import McNeilFreyResult, mcneil_frey_es_test
from rvw.backtests.suite import ModelBacktestSummary, compare_models, evaluate_backtests

__all__ = [
    "BaselZoneResult",
    "ConditionalCoverageResult",
    "IndependenceResult",
    "KupiecResult",
    "McNeilFreyResult",
    "ModelBacktestSummary",
    "christoffersen_conditional_coverage_test",
    "christoffersen_independence_test",
    "classify_basel_zone",
    "compare_models",
    "compute_rolling_traffic_light",
    "evaluate_backtests",
    "kupiec_pof_test",
    "mcneil_frey_es_test",
]
