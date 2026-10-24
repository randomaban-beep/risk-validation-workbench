"""Unit tests for data ingestion, caching, and validation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from rvw.config import DataValidationConfig
from rvw.data.cache import load_cache, save_cache
from rvw.data.fetcher import load_prices
from rvw.data.synthetic import generate_synthetic_prices
from rvw.data.validator import compute_returns, validate_prices


def test_synthetic_data_generation_reproducibility():
    """Verify synthetic price generation produces identical results with identical seed."""
    df1 = generate_synthetic_prices(tickers=["SPY", "TLT"], start_date="2020-01-01", end_date="2020-03-31", seed=42)
    df2 = generate_synthetic_prices(tickers=["SPY", "TLT"], start_date="2020-01-01", end_date="2020-03-31", seed=42)
    pd.testing.assert_frame_equal(df1, df2)

    df3 = generate_synthetic_prices(tickers=["SPY", "TLT"], start_date="2020-01-01", end_date="2020-03-31", seed=99)
    assert not df1.equals(df3)
    assert df1.attrs.get("DATA_STATUS") == "SYNTHETIC"


def test_cache_save_and_load(tmp_path: Path):
    """Verify parquet caching and JSON sidecar metadata preservation."""
    df = generate_synthetic_prices(tickers=["SPY", "GLD"], start_date="2022-01-01", end_date="2022-02-28", seed=123)
    cache_file = tmp_path / "test_cache.parquet"

    save_cache(df, cache_file, metadata={"test_run": True})
    assert cache_file.exists()
    sidecar = cache_file.with_suffix(".meta.json")
    assert sidecar.exists()

    loaded_df, meta = load_cache(cache_file)
    pd.testing.assert_frame_equal(df, loaded_df, check_freq=False)
    assert meta["test_run"] is True
    assert meta["is_synthetic"] is True
    assert meta["tickers"] == ["SPY", "GLD"]


def test_compute_returns():
    """Verify simple arithmetic returns computation."""
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"SPY": [100.0, 105.0, 94.5]}, index=dates)
    returns = compute_returns(prices)

    assert len(returns) == 2
    assert np.isclose(returns.iloc[0]["SPY"], 0.05)
    assert np.isclose(returns.iloc[1]["SPY"], -0.10)


def test_validator_detects_nan(sample_prices: pd.DataFrame):
    """Verify that injected NaN values are detected and cause is_valid=False."""
    bad_prices = sample_prices.copy()
    bad_prices.iloc[50, 0] = np.nan

    report = validate_prices(bad_prices)
    assert not report.is_valid
    categories = [i.category for i in report.issues]
    assert "MISSING_VALUES" in categories


def test_validator_detects_duplicate_dates(sample_prices: pd.DataFrame):
    """Verify that duplicate index timestamps are detected."""
    bad_prices = sample_prices.copy()
    new_index = list(bad_prices.index)
    new_index[10] = new_index[9]  # Duplicate date
    bad_prices.index = new_index

    report = validate_prices(bad_prices)
    assert not report.is_valid
    categories = [i.category for i in report.issues]
    assert "DUPLICATE_DATES" in categories


def test_validator_detects_non_positive_price(sample_prices: pd.DataFrame):
    """Verify that prices <= 0 are detected and flagged as errors."""
    bad_prices = sample_prices.copy()
    bad_prices.iloc[20, 1] = -5.0

    report = validate_prices(bad_prices)
    assert not report.is_valid
    categories = [i.category for i in report.issues]
    assert "NON_POSITIVE_PRICE" in categories


def test_validator_detects_extreme_jumps(sample_prices: pd.DataFrame):
    """Verify that a 60% jump is detected as EXTREME_RETURN_JUMP."""
    bad_prices = sample_prices.copy()
    bad_prices.iloc[100, 0] = bad_prices.iloc[99, 0] * 1.60  # +60% jump

    cfg = DataValidationConfig(max_return_jump=0.25)
    report = validate_prices(bad_prices, config=cfg)
    categories = [i.category for i in report.issues]
    assert "EXTREME_RETURN_JUMP" in categories


def test_validator_detects_stale_prices(sample_prices: pd.DataFrame):
    """Verify consecutive zero returns trigger STALE_PRICES warning."""
    bad_prices = sample_prices.copy()
    # Force 7 days of flat prices
    for i in range(150, 157):
        bad_prices.iloc[i, 0] = bad_prices.iloc[149, 0]

    cfg = DataValidationConfig(max_stale_days=5)
    report = validate_prices(bad_prices, config=cfg)
    categories = [i.category for i in report.issues]
    assert "STALE_PRICES" in categories


def test_load_prices_force_synthetic(default_config, tmp_path):
    """Verify load_prices produces valid synthetic data bundle when requested."""
    default_config.paths.cache_dir = str(tmp_path / "cache")
    bundle = load_prices(default_config, refresh=True, force_synthetic=True)

    assert bundle.is_synthetic is True
    assert bundle.source == "SYNTHETIC"
    assert bundle.quality_report.is_valid is True
    assert len(bundle.prices) > 1000
    assert set(bundle.prices.columns) == set(default_config.universe.tickers)
