"""Tests for core.intelligent_trader — adaptive trading engine."""

import numpy as np
import pandas as pd
from core.intelligent_trader import (
    compute_urgency, adaptive_interval, adaptive_period,
    AGGRESSIVITY_PROFILES, get_aggressivity, get_aggressivity_names,
    IntelligentStock,
)


class TestUrgency:
    def _make_data(self, volatility=0.01, trend=0.0, volume_spike=1.0):
        """Create dummy data with controllable characteristics."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2024-01-01 09:30", periods=n, freq="5min")
        base = 100 + np.arange(n) * trend
        noise = np.random.randn(n) * volatility * 100
        close = base + noise
        volume = np.random.randint(10000, 50000, n).astype(float)
        volume[-1] *= volume_spike  # Spike last bar

        data = pd.DataFrame({
            "Open": close - 0.1, "High": close + abs(noise) * 0.5,
            "Low": close - abs(noise) * 0.5, "Close": close, "Volume": volume,
        }, index=dates)

        # Add indicators the urgency function needs
        from core.features import engineer_features
        data = engineer_features(data, mode="intraday")
        return data

    def test_calm_market_low_urgency(self):
        data = self._make_data(volatility=0.002, trend=0.0, volume_spike=1.0)
        urgency = compute_urgency(data)
        assert urgency < 0.5, f"Calm market should have low urgency, got {urgency}"

    def test_volatile_market_high_urgency(self):
        data = self._make_data(volatility=0.05, trend=0.1, volume_spike=3.0)
        urgency = compute_urgency(data)
        assert urgency > 0.3, f"Volatile market should have higher urgency, got {urgency}"

    def test_urgency_bounded(self):
        data = self._make_data(volatility=0.1, trend=0.5, volume_spike=10.0)
        urgency = compute_urgency(data)
        assert 0.0 <= urgency <= 1.0

    def test_empty_data(self):
        urgency = compute_urgency(pd.DataFrame())
        assert urgency == 0.5  # Default for unknown


class TestAdaptiveInterval:
    def test_high_urgency_short_interval(self):
        interval = adaptive_interval(0.9)
        assert interval <= 120  # 2 min or less

    def test_low_urgency_long_interval(self):
        interval = adaptive_interval(0.1)
        assert interval >= 600  # 10 min or more

    def test_profile_bias_increases(self):
        base = adaptive_interval(0.5)
        aggressive = adaptive_interval(0.5, profile_bias=0.2)
        assert aggressive <= base  # More urgent = shorter interval

    def test_profile_bias_decreases(self):
        base = adaptive_interval(0.5)
        chill = adaptive_interval(0.5, profile_bias=-0.2)
        assert chill >= base  # Less urgent = longer interval


class TestAdaptivePeriod:
    def test_high_urgency_short_period(self):
        period = adaptive_period(0.8)
        assert period == "2d"

    def test_low_urgency_long_period(self):
        period = adaptive_period(0.1)
        assert period == "5d"


class TestAggressivityProfiles:
    def test_all_profiles_exist(self):
        names = get_aggressivity_names()
        assert "Chill" in names
        assert "Default" in names
        assert "Aggressive" in names
        assert "YOLO" in names

    def test_profiles_have_required_fields(self):
        required = ["min_confidence", "size_multiplier", "atr_stop_mult",
                     "atr_profit_mult", "urgency_bias", "max_trades_per_day", "description"]
        for name in get_aggressivity_names():
            prof = get_aggressivity(name)
            for field in required:
                assert field in prof, f"{name} missing {field}"

    def test_chill_more_conservative_than_yolo(self):
        chill = get_aggressivity("Chill")
        yolo = get_aggressivity("YOLO")
        assert chill["min_confidence"] > yolo["min_confidence"]
        assert chill["size_multiplier"] < yolo["size_multiplier"]
        assert chill["max_trades_per_day"] < yolo["max_trades_per_day"]

    def test_unknown_profile_returns_default(self):
        prof = get_aggressivity("nonexistent_xyz")
        default = get_aggressivity("Default")
        assert prof == default


class TestIntelligentStock:
    def test_creation(self):
        stock = IntelligentStock(ticker="AAPL")
        assert stock.ticker == "AAPL"
        assert stock.mode == "intelligent"
        assert stock.last_signal == "HOLD"

    def test_defaults(self):
        stock = IntelligentStock(ticker="TEST", aggressivity="Aggressive")
        assert stock.aggressivity == "Aggressive"
        assert stock.trades_today == 0
