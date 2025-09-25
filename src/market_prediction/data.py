"""Data loading utilities for Nifty 50 modelling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency for offline environments
    yf = None  # type: ignore


@dataclass
class MarketDataConfig:
    """Configuration for fetching market data."""

    symbol: str = "^NSEI"
    start: str = "2010-01-01"
    end: Optional[str] = None
    cache_path: Optional[Path] = None


def download_market_data(config: MarketDataConfig) -> pd.DataFrame:
    """Download OHLCV data for the configured index.

    Parameters
    ----------
    config:
        Parameters describing the instrument and time range. When ``cache_path``
        is supplied and the file exists it will be used instead of performing a
        download. This is convenient for environments without network access.

    Returns
    -------
    pandas.DataFrame
        Daily OHLCV data with a ``DatetimeIndex``.
    """

    if config.cache_path and config.cache_path.exists():
        data = pd.read_csv(config.cache_path, parse_dates=["Date"], index_col="Date")
        data.sort_index(inplace=True)
        return data

    if yf is None:
        raise RuntimeError(
            "yfinance is not installed and no cached data file was provided."
        )

    ticker = yf.Ticker(config.symbol)
    data = ticker.history(start=config.start, end=config.end)

    if data.empty:
        raise RuntimeError(
            "Downloaded market data is empty. Verify the symbol and date range."
        )

    data.index.name = "Date"

    if config.cache_path:
        config.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(config.cache_path)

    return data


def load_data_from_csv(path: Path) -> pd.DataFrame:
    """Load market data from a CSV file.

    The function simply wraps :func:`pandas.read_csv` to ensure consistent
    parsing of the ``Date`` column.
    """

    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")
