"""Market data ingestion, validation, and caching package."""

from rvw.data.cache import load_cache, save_cache
from rvw.data.fetcher import DataBundle, load_prices
from rvw.data.synthetic import generate_synthetic_prices
from rvw.data.validator import DataQualityReport, QualityIssue, compute_returns, validate_prices

__all__ = [
    "DataBundle",
    "DataQualityReport",
    "QualityIssue",
    "compute_returns",
    "generate_synthetic_prices",
    "load_cache",
    "load_prices",
    "save_cache",
    "validate_prices",
]
