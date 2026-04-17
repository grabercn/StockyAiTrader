"""Tests for core.labeling — triple barrier label generation."""

import numpy as np
import pandas as pd
from core.labeling import triple_barrier_label, BUY, SELL, HOLD


class TestTripleBarrier:
    def test_output_length(self, dummy_ohlcv):
        from core.features import engineer_features
        data = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        labels = triple_barrier_label(data)
        assert len(labels) == len(data)

    def test_valid_labels(self, dummy_ohlcv):
        from core.features import engineer_features
        data = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        labels = triple_barrier_label(data)
        assert set(labels).issubset({SELL, HOLD, BUY})

    def test_not_all_same(self, dummy_ohlcv):
        from core.features import engineer_features
        data = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        labels = triple_barrier_label(data)
        assert len(set(labels)) > 1, "All labels are the same — likely a bug"

    def test_wider_barriers_more_holds(self, dummy_ohlcv):
        from core.features import engineer_features
        data = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        tight = triple_barrier_label(data, atr_tp=1.0, atr_sl=0.5, max_bars=20)
        wide = triple_barrier_label(data, atr_tp=5.0, atr_sl=4.0, max_bars=20)
        tight_holds = (tight == HOLD).sum()
        wide_holds = (wide == HOLD).sum()
        assert wide_holds >= tight_holds
