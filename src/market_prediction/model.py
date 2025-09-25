"""Model training utilities for Nifty 50 prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelArtifacts:
    scaler: StandardScaler
    model: RandomForestRegressor
    feature_columns: Tuple[str, ...]


@dataclass
class TrainingResult:
    artifacts: ModelArtifacts
    test_mae: float
    test_r2: float


TARGET_COLUMN = "target_return"


def split_features_targets(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in frame:
        raise KeyError("The feature frame must contain a 'target_return' column.")
    X = frame.drop(columns=[TARGET_COLUMN])
    y = frame[TARGET_COLUMN]
    return X, y


def train_random_forest(features: pd.DataFrame) -> TrainingResult:
    X, y = split_features_targets(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    artifacts = ModelArtifacts(
        scaler=scaler,
        model=model,
        feature_columns=tuple(X.columns),
    )
    return TrainingResult(artifacts=artifacts, test_mae=mae, test_r2=r2)


def forecast_next_return(artifacts: ModelArtifacts, latest_features: pd.Series) -> float:
    missing = set(artifacts.feature_columns) - set(latest_features.index)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    vector = latest_features.loc[list(artifacts.feature_columns)].to_numpy().reshape(1, -1)
    scaled = artifacts.scaler.transform(vector)
    return float(artifacts.model.predict(scaled)[0])
