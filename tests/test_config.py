"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest

from rvw.config import AppConfig, load_config


def test_load_default_config():
    """Verify loading default configuration file."""
    config = load_config("config/default.yaml")
    assert isinstance(config, AppConfig)
    assert len(config.universe.tickers) == 12
    assert "SPY" in config.universe.tickers
    assert config.risk_conventions.horizon_days == 1
    assert 0.95 in config.risk_conventions.var_alphas
    assert 0.99 in config.risk_conventions.var_alphas
    assert 0.975 in config.risk_conventions.es_alphas
    assert "equal_weight" in config.portfolios
    assert "sixty_forty" in config.portfolios


def test_load_synthetic_test_config():
    """Verify loading synthetic test configuration file."""
    config = load_config("config/synthetic_test.yaml")
    assert isinstance(config, AppConfig)
    assert len(config.universe.tickers) == 5
    assert config.walkforward.window_length == 250
    assert config.seeds.synthetic_seed == 777


def test_config_serialization():
    """Verify configuration object serializes to dictionary and JSON."""
    config = load_config("config/default.yaml")
    d = config.to_dict()
    assert isinstance(d, dict)
    assert "universe" in d
    assert "portfolios" in d

    json_str = config.to_json()
    assert isinstance(json_str, str)
    assert "historical_simulation" in json_str


def test_missing_config_raises_file_not_found():
    """Verify proper exception when config file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config("config/non_existent_config.yaml")
