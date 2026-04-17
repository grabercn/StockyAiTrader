"""Tests for core.model — LightGBM training and prediction."""

import numpy as np
from core.features import engineer_features, INTRADAY_FEATURES
from core.labeling import triple_barrier_label
from core.model import train_lgbm, predict_lgbm


class TestTraining:
    def _prepare(self, dummy_ohlcv):
        data = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        data["vader_sentiment"] = 0.1
        data["finbert_sentiment"] = 0.05
        data["Label"] = triple_barrier_label(data)
        data = data.dropna(subset=[c for c in INTRADAY_FEATURES if c in data.columns])
        return data

    def test_train_returns_model(self, dummy_ohlcv):
        data = self._prepare(dummy_ohlcv)
        model, features = train_lgbm(data, INTRADAY_FEATURES, "TEST_UNIT")
        assert model is not None
        assert len(features) > 0

    def test_train_with_insufficient_data(self, dummy_ohlcv):
        data = self._prepare(dummy_ohlcv).head(5)  # Too few rows
        model, features = train_lgbm(data, INTRADAY_FEATURES, "TEST_SMALL")
        assert model is None

    def test_predict_shapes(self, dummy_ohlcv):
        data = self._prepare(dummy_ohlcv)
        model, features = train_lgbm(data, INTRADAY_FEATURES, "TEST_PRED")
        assert model is not None

        actions, confs, probs = predict_lgbm(model, data, features)
        assert len(actions) == len(data)
        assert len(confs) == len(data)
        assert probs.shape == (len(data), 3)

    def test_predictions_valid(self, dummy_ohlcv):
        data = self._prepare(dummy_ohlcv)
        model, features = train_lgbm(data, INTRADAY_FEATURES, "TEST_VALID")
        actions, confs, probs = predict_lgbm(model, data, features)

        # Actions should be 0, 1, or 2
        assert set(actions).issubset({0, 1, 2})
        # Confidences should be 0-1
        assert (confs >= 0).all() and (confs <= 1).all()
        # Probabilities should sum to ~1
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.01)
