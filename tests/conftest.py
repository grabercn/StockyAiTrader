"""
Shared test fixtures for StockyAiTrader test suite.

Provides reusable dummy data, temporary directories, and mock objects
so individual test files stay clean and focused.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add StockyApps to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "StockyApps"))


@pytest.fixture
def dummy_ohlcv():
    """Generate 100 bars of realistic-looking OHLCV data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-06-01 09:30", periods=n, freq="5min")
    close = 150 + np.random.randn(n).cumsum() * 0.5
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(10000, 500000, n).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
    }, index=dates)


@pytest.fixture
def dummy_ohlcv_daily():
    """Generate 250 daily bars (roughly 1 year)."""
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="B")  # Business days
    close = 150 + np.random.randn(n).cumsum() * 1.0
    high = close + np.abs(np.random.randn(n)) * 1.0
    low = close - np.abs(np.random.randn(n)) * 1.0
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000000, 50000000, n).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
    }, index=dates)


@pytest.fixture
def tmp_log_dir(tmp_path):
    """Provide a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return str(log_dir)
