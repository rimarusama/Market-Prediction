import pytest

pandas = pytest.importorskip("pandas")

from src.market_prediction.model import split_features_targets, train_random_forest


pd = pandas


def make_dummy_features():
    index = pd.date_range("2020-01-01", periods=200)
    data = {
        "feature_a": range(200),
        "feature_b": range(200, 400),
        "target_return": [i * 0.001 for i in range(200)],
    }
    return pd.DataFrame(data, index=index)


def test_split_features_targets():
    frame = make_dummy_features()
    X, y = split_features_targets(frame)
    assert "target_return" not in X.columns
    assert len(y) == len(frame)


def test_train_random_forest_runs():
    frame = make_dummy_features()
    result = train_random_forest(frame)
    assert result.test_mae >= 0
    assert -1 <= result.test_r2 <= 1
