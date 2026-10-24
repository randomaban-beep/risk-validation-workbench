"""Parquet caching with structured metadata sidecars."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd


def save_cache(
    df: pd.DataFrame,
    cache_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Save price DataFrame to parquet with an accompanying JSON metadata sidecar.

    Args:
        df: Price dataframe with DatetimeIndex and ticker columns.
        cache_path: Path to target parquet file.
        metadata: Optional dictionary with custom provenance metadata.

    Returns:
        Tuple[Path, Path]: (parquet_path, sidecar_json_path)
    """
    parquet_path = Path(cache_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = parquet_path.with_suffix(".meta.json")

    # Write parquet
    df.to_parquet(parquet_path, engine="pyarrow")

    # Prepare sidecar metadata
    is_synth = df.attrs.get("DATA_STATUS") == "SYNTHETIC" or (metadata and metadata.get("is_synthetic", False))
    meta: Dict[str, Any] = {
        "parquet_file": str(parquet_path.name),
        "created_at_utc": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "tickers": list(df.columns),
        "start_date": str(df.index.min().date()) if len(df) > 0 else None,
        "end_date": str(df.index.max().date()) if len(df) > 0 else None,
        "is_synthetic": bool(is_synth),
        "source": "SYNTHETIC" if is_synth else "yfinance",
        "disclaimer": (
            "Educational prototype, inspired by model-validation practice; "
            "not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation."
        ),
    }
    if metadata:
        meta.update(metadata)

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return parquet_path, sidecar_path


def load_cache(cache_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load cached price DataFrame and its sidecar metadata.

    Args:
        cache_path: Path to target parquet file.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (prices_df, metadata_dict)

    Raises:
        FileNotFoundError: If parquet file does not exist.
    """
    parquet_path = Path(cache_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Cache file not found at {parquet_path.resolve()}")

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    sidecar_path = parquet_path.with_suffix(".meta.json")

    metadata: Dict[str, Any] = {}
    if sidecar_path.exists():
        with open(sidecar_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if metadata.get("is_synthetic", False):
        df.attrs["DATA_STATUS"] = "SYNTHETIC"

    return df, metadata
