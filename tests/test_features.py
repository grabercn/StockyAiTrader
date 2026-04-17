"""Tests for core.features — technical indicator calculation."""

import numpy as np
from core.features import engineer_features, INTRADAY_FEATURES, LONGTERM_FEATURES


class TestIntradayFeatures:
    def test_produces_all_columns(self, dummy_ohlcv):
        result = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        # Check key features exist (excluding sentiment — added by data pipeline)
        for col in ["log_return", "pct_change", "vwap", "price_vs_vwap",
                     "bb_position", "bb_width", "macd_hist", "RSI_14", "RSI_2",
                     "ema_cross", "obv_slope", "ATRr_14", "EMA_9", "EMA_21"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_inf_values(self, dummy_ohlcv):
        result = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        for col in INTRADAY_FEATURES:
            if col in result.columns and col not in ("vader_sentiment", "finbert_sentiment"):
                assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_vwap_reasonable(self, dummy_ohlcv):
        result = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        # VWAP should be in the same ballpark as Close
        assert abs(result["vwap"].iloc[-1] - result["Close"].iloc[-1]) < 20

    def test_rsi_bounded(self, dummy_ohlcv):
        result = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        valid_rsi = result["RSI_14"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_bb_position_bounded(self, dummy_ohlcv):
        result = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        valid = result["bb_position"].dropna()
        # Should mostly be 0-1, but can exceed during breakouts
        assert valid.median() > 0 and valid.median() < 1


class TestLongtermFeatures:
    def test_produces_sma_cross(self, dummy_ohlcv_daily):
        result = engineer_features(dummy_ohlcv_daily.copy(), mode="longterm")
        assert "sma_cross" in result.columns
        assert "SMA_50" in result.columns
        assert "SMA_200" in result.columns

    def test_day_of_week(self, dummy_ohlcv_daily):
        result = engineer_features(dummy_ohlcv_daily.copy(), mode="longterm")
        assert "day_of_week" in result.columns
        valid = result["day_of_week"].dropna()
        assert valid.min() >= 0 and valid.max() <= 1
