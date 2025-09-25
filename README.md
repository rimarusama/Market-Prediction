# Market-Prediction

This project provides a reproducible workflow for training a machine-learning
model that forecasts the next-day return of the Nifty 50 index.

The workflow comprises:

1. **Data ingestion** – `market_prediction.data` fetches daily OHLCV candles
   using `yfinance` (with optional CSV caching for offline runs).
2. **Feature engineering** – `market_prediction.features` creates technical
   indicators such as moving averages, RSI, Bollinger bands, and volume-based
   metrics.
3. **Model training** – `market_prediction.model` trains a random forest
   regressor on the engineered features and evaluates it on a hold-out set.
4. **Pipeline script** – `market_prediction.train_pipeline` ties everything
   together and persists the trained artifacts.

## Quickstart

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas scikit-learn yfinance joblib numpy
```

Train the model (caching the downloaded data for future runs):

```bash
python -m src.market_prediction.train_pipeline --cache data/nifty50.csv \
    --model-out artifacts.joblib
```

Example output:

```
Model trained successfully!
Test MAE: 0.004512
Test R^2: 0.4281
Next-day return prediction: 0.001237
```

If the environment does not permit network access, download the historical
prices elsewhere and place the CSV at the cache path shown above.

To reuse the trained model within Python:

```python
import joblib
from pathlib import Path

from src.market_prediction.features import compute_technical_indicators
from src.market_prediction.model import forecast_next_return

training_result = joblib.load(Path("artifacts.joblib"))
latest_features = compute_technical_indicators(your_dataframe).iloc[-1].drop("target_return")
next_return = forecast_next_return(training_result.artifacts, latest_features)
```
