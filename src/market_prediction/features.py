"""Feature engineering for the Nifty 50 forecasting model."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_feature(frame: pd.Series, window: int, func: str) -> pd.Series:
    return getattr(frame.rolling(window=window, min_periods=window), func)()


def compute_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Compute a set of technical indicators used as model features."""

    prices = data["Close"]
    returns = prices.pct_change()

    features = pd.DataFrame(index=data.index)
    features["return_1d"] = returns
    features["log_return_1d"] = np.log1p(returns)

    for window in (5, 10, 20):
        features[f"sma_{window}"] = _rolling_feature(prices, window, "mean")
        features[f"ema_{window}"] = prices.ewm(span=window, adjust=False).mean()
        features[f"vol_{window}"] = _rolling_feature(returns, window, "std")

    momentum = prices.diff()
    up = momentum.clip(lower=0)
    down = -momentum.clip(upper=0)
    rs = _rolling_feature(up, 14, "mean") / _rolling_feature(down, 14, "mean")
    features["rsi_14"] = 100 - (100 / (1 + rs))

    features["bb_high"] = _rolling_feature(prices, 20, "mean") + 2 * _rolling_feature(returns, 20, "std")
    features["bb_low"] = _rolling_feature(prices, 20, "mean") - 2 * _rolling_feature(returns, 20, "std")

    volume = data.get("Volume")
    if volume is not None:
        features["volume_change"] = volume.pct_change()
        features["volume_sma_20"] = _rolling_feature(volume, 20, "mean")

    features["target_return"] = returns.shift(-1)

    features.dropna(inplace=True)
    return features
