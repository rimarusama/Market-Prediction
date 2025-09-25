"""Command line entry-point for training a Nifty 50 prediction model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .data import MarketDataConfig, download_market_data
from .features import compute_technical_indicators
from .model import forecast_next_return, train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="^NSEI", help="Ticker symbol to download.")
    parser.add_argument(
        "--start", default="2010-01-01", help="Start date for historical data."
    )
    parser.add_argument("--end", default=None, help="Optional end date for data")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional path to cache the downloaded CSV data.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts.joblib"),
        help="Path where the trained model artifacts will be persisted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MarketDataConfig(
        symbol=args.symbol, start=args.start, end=args.end, cache_path=args.cache
    )

    raw_data = download_market_data(config)
    features = compute_technical_indicators(raw_data)

    training_result = train_random_forest(features)

    latest_features = features.iloc[-1].drop("target_return")
    next_return_prediction = forecast_next_return(
        training_result.artifacts, latest_features
    )

    joblib.dump(training_result, args.model_out)

    print("Model trained successfully!")
    print(f"Test MAE: {training_result.test_mae:.6f}")
    print(f"Test R^2: {training_result.test_r2:.4f}")
    print(f"Next-day return prediction: {next_return_prediction:.6f}")


if __name__ == "__main__":
    main()
