"""Pytest fixtures and test configuration."""

from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd

from rvw.config import AppConfig, load_config
from rvw.data.cache import load_cache
from rvw.data.validator import compute_returns


@pytest.fixture(scope="session")
def default_config() -> AppConfig:
    """Fixture providing the default application configuration."""
    return load_config("config/default.yaml")


@pytest.fixture(scope="session")
def synthetic_test_config() -> AppConfig:
    """Fixture providing fast synthetic test configuration."""
    return load_config("config/synthetic_test.yaml")


@pytest.fixture(scope="session")
def sample_prices() -> pd.DataFrame:
    """Fixture providing the committed sample price dataframe."""
    sample_path = Path("data/sample/synthetic_prices.parquet")
    df, _ = load_cache(sample_path)
    return df


@pytest.fixture(scope="session")
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Fixture providing sample daily simple returns."""
    return compute_returns(sample_prices)
