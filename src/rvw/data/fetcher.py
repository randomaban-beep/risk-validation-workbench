"""Market data ingestion, caching, and resilient network fallback engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from rvw.config import AppConfig
from rvw.data.cache import load_cache, save_cache
from rvw.data.synthetic import generate_synthetic_prices
from rvw.data.validator import DataQualityReport, compute_returns, validate_prices
from rvw.logging_utils import log_event

logger = logging.getLogger(__name__)


@dataclass
class DataBundle:
    """Standardized container for loaded prices, returns, and validation results."""
    prices: pd.DataFrame
    returns: pd.DataFrame
    quality_report: DataQualityReport
    is_synthetic: bool
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Provide a top-level summary of the dataset."""
        return {
            "source": self.source,
            "is_synthetic": self.is_synthetic,
            "tickers": list(self.prices.columns),
            "rows": len(self.prices),
            "start_date": str(self.prices.index.min().date()) if len(self.prices) > 0 else None,
            "end_date": str(self.prices.index.max().date()) if len(self.prices) > 0 else None,
            "data_valid": self.quality_report.is_valid,
            "issue_count": len(self.quality_report.issues),
        }


def download_yfinance_prices(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download daily adjusted close prices via yfinance.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date string (YYYY-MM-DD).
        end_date: Optional end date string.

    Returns:
        pd.DataFrame: Adjusted close prices aligned by trading date.
    """
    import yfinance as yf

    logger.info("Attempting yfinance download for %d tickers: %s", len(tickers), tickers)
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise ValueError("yfinance returned an empty dataset.")

    # When multiple tickers are requested, yfinance returns multi-index columns: (PriceCol, Ticker)
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            prices = data["Close"]
        else:
            prices = data.xs(data.columns.levels[0][0], axis=1)
    else:
        # Single ticker case
        prices = data[["Close"]] if "Close" in data.columns else data

    # Reorder columns to match requested ticker list
    available_cols = [c for c in tickers if c in prices.columns]
    prices = prices[available_cols].dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    return prices


def load_prices(
    config: AppConfig,
    refresh: bool = False,
    force_synthetic: bool = False,
) -> DataBundle:
    """Load asset price time series with automatic caching and network failure fallback.

    Network failure policy:
        If yfinance download fails or cannot reach remote endpoints, a structured warning
        is logged, the dataset is synthesized with asset-class calibrations, all outputs
        are labeled as 'SYNTHETIC', and the exact fetch command is printed.

    Args:
        config: Application configuration.
        refresh: Force re-download / regeneration bypassing local cache.
        force_synthetic: Directly generate synthetic data without attempting remote fetch.

    Returns:
        DataBundle: Populated container with prices, returns, quality report, and provenance.
    """
    cache_dir = Path(config.paths.cache_dir)
    cache_path = cache_dir / "indicative_prices.parquet"

    # Check local cache first
    if cache_path.exists() and not refresh and not force_synthetic:
        try:
            logger.info("Loading prices from local cache: %s", cache_path)
            prices, meta = load_cache(cache_path)
            is_synthetic = meta.get("is_synthetic", False)
            source = meta.get("source", "cache")
            quality = validate_prices(prices, config.data_validation, config.universe.start_date)
            returns = compute_returns(prices)
            return DataBundle(
                prices=prices,
                returns=returns,
                quality_report=quality,
                is_synthetic=is_synthetic,
                source=source,
                metadata=meta,
            )
        except Exception as exc:
            logger.warning("Failed to read cache at %s: %s. Refreshing...", cache_path, exc)

    prices: Optional[pd.DataFrame] = None
    is_synthetic = False
    source = "yfinance"

    if not force_synthetic:
        try:
            prices = download_yfinance_prices(
                tickers=config.universe.tickers,
                start_date=config.universe.start_date,
                end_date=config.universe.end_date,
            )
            # Ensure all tickers present
            missing = set(config.universe.tickers) - set(prices.columns)
            if missing:
                raise ValueError(f"Missing tickers from download: {missing}")
            logger.info("Successfully fetched %d rows from yfinance.", len(prices))
        except Exception as exc:
            log_event(
                logger=logger,
                level=logging.WARNING,
                event_type="NETWORK_FETCH_FAILURE",
                message=f"yfinance download failed: {exc}. Falling back to calibrated SYNTHETIC data.",
                error=str(exc),
            )
            print("\n" + "=" * 80)
            print(">>> WARNING: NETWORK FETCH FAILED. USING SYNTHETIC DATA.")
            print(">>> ALL OUTPUTS WILL BE PROMINENTLY LABELED AS 'SYNTHETIC'.")
            print(">>> TO RETRY FETCH WHEN ONLINE, RUN:")
            print(">>> python -m rvw.cli fetch-data --config config/default.yaml --refresh")
            print("=" * 80 + "\n")
            prices = None

    if prices is None or force_synthetic:
        is_synthetic = True
        source = "SYNTHETIC"
        logger.info("Generating calibrated SYNTHETIC prices (seed=%d)...", config.seeds.synthetic_seed)
        prices = generate_synthetic_prices(
            tickers=config.universe.tickers,
            start_date=config.universe.start_date,
            end_date=config.universe.end_date or "2024-06-30",
            seed=config.seeds.synthetic_seed,
        )

    # Save to cache
    save_cache(
        prices,
        cache_path,
        metadata={
            "is_synthetic": is_synthetic,
            "source": source,
            "seed": config.seeds.synthetic_seed if is_synthetic else None,
        },
    )

    # Run data quality validation
    quality_report = validate_prices(prices, config.data_validation, config.universe.start_date)
    returns = compute_returns(prices)

    log_event(
        logger=logger,
        level=logging.INFO,
        event_type="DATA_INGESTION_COMPLETED",
        message=f"Data ingestion complete. Rows={len(prices)}, Assets={len(prices.columns)}, Synthetic={is_synthetic}",
        is_synthetic=is_synthetic,
        rows=len(prices),
        valid=quality_report.is_valid,
    )

    return DataBundle(
        prices=prices,
        returns=returns,
        quality_report=quality_report,
        is_synthetic=is_synthetic,
        source=source,
        metadata={"cache_path": str(cache_path)},
    )
