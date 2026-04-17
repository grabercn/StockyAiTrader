"""
Integration tests — verify full end-to-end pipelines work together.

These test multiple modules cooperating, not individual units.
"""

import numpy as np
import pandas as pd
from core.features import engineer_features, INTRADAY_FEATURES, LONGTERM_FEATURES
from core.labeling import triple_barrier_label, LABEL_NAMES
from core.model import train_lgbm, predict_lgbm
from core.risk import RiskManager
from core.signals import write_signal, read_signal
from core.logger import log_decision, get_today_logs
from core.data import get_all_features
from addons import get_all_addons, get_addon_features, gather_features


class TestFullIntradayPipeline:
    """Test the complete intraday pipeline: data -> features -> train -> predict -> risk -> signal."""

    def test_end_to_end(self, dummy_ohlcv):
        # 1. Engineer features
        data = engineer_features(dummy_ohlcv.copy(), mode="intraday")
        assert "RSI_14" in data.columns
        assert "vwap" in data.columns

        # 2. Add sentiment placeholders
        data["vader_sentiment"] = 0.1
        data["finbert_sentiment"] = 0.05

        # 3. Label
        data["Label"] = triple_barrier_label(data)
        assert set(data["Label"].unique()).issubset({0, 1, 2})

        # 4. Clean for training
        available = [c for c in INTRADAY_FEATURES if c in data.columns]
        data = data.dropna(subset=available)
        assert len(data) >= 30

        # 5. Train model
        model, features = train_lgbm(data, INTRADAY_FEATURES, "INTEG_TEST")
        assert model is not None

        # 6. Predict
        actions, confs, probs = predict_lgbm(model, data, features)
        assert len(actions) == len(data)
        assert probs.shape[1] == 3

        # 7. Risk management
        rm = RiskManager(portfolio_value=100000)
        price = float(data["Close"].iloc[-1])
        atr = float(data["ATRr_14"].iloc[-1])
        size = rm.position_size(price, atr)
        assert size >= 0

        sl = rm.stop_loss(price, atr, "buy")
        tp = rm.take_profit(price, atr, "buy")
        assert sl < price < tp

        # 8. Write and read signal
        action = LABEL_NAMES[actions[-1]]
        sig = write_signal("INTEG", action, float(confs[-1]), price, size, sl, tp, atr)
        read_back = read_signal()
        assert read_back["ticker"] == "INTEG"
        assert read_back["action"] == action

        # 9. Log the decision
        entry = log_decision("INTEG", action, float(confs[-1]), price, size, sl, tp, atr,
                             [float(p) for p in probs[-1]], reasoning="Integration test")
        assert entry["type"] == "decision"


class TestFullLongtermPipeline:
    """Test the long-term pipeline with daily data."""

    def test_end_to_end(self, dummy_ohlcv_daily):
        data = engineer_features(dummy_ohlcv_daily.copy(), mode="longterm")
        assert "SMA_50" in data.columns

        data["vader_sentiment"] = 0.0
        data["finbert_sentiment"] = 0.0
        data["Label"] = triple_barrier_label(data, atr_tp=3.0, atr_sl=2.0, max_bars=30)

        available = [c for c in LONGTERM_FEATURES if c in data.columns]
        data = data.dropna(subset=available)
        assert len(data) >= 50

        model, features = train_lgbm(data, LONGTERM_FEATURES, "INTEG_LONG", prefix="lgbm_long", min_samples=50)
        assert model is not None

        actions, confs, probs = predict_lgbm(model, data, features)
        assert len(actions) == len(data)


class TestAddonIntegration:
    """Test that addon features integrate correctly into the feature pipeline."""

    def test_addon_features_extend_core(self):
        all_f = get_all_features("intraday")
        core_f = INTRADAY_FEATURES
        addon_f = get_addon_features()
        assert len(all_f) == len(core_f) + len(addon_f)

    def test_addon_features_are_numeric(self, dummy_ohlcv):
        result = gather_features("TEST", dummy_ohlcv)
        for k, v in result.items():
            assert isinstance(v, (int, float, np.integer, np.floating)), \
                f"Addon feature {k} is {type(v)}, expected numeric"

    def test_all_active_addons_return_features(self, dummy_ohlcv):
        active = [a for a in get_all_addons() if a.available and a.enabled]
        for addon in active:
            result = addon._module.get_features("TEST", dummy_ohlcv)
            assert isinstance(result, dict), f"{addon.name} didn't return dict"
            for feat_name in addon.features:
                assert feat_name in result, f"{addon.name} missing feature {feat_name}"


class TestProfileIntegration:
    """Test that profile changes actually affect the addon system."""

    def test_minimal_disables_addons(self):
        from core.profiles import apply_profile
        apply_profile("Minimal")
        active = [a for a in get_all_addons() if a.enabled]
        assert len(active) == 0, "Minimal should disable all addons"

        # Restore default
        apply_profile("Balanced")

    def test_features_change_with_profile(self):
        from core.profiles import apply_profile

        apply_profile("Max")
        max_features = get_all_features("intraday")

        apply_profile("Minimal")
        min_features = get_all_features("intraday")

        assert len(max_features) >= len(min_features)

        # Restore
        apply_profile("Balanced")
