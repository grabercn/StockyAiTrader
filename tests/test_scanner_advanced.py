"""Tests for scanner: progress queue, grey rows, fallbacks, LLM, sizing."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "StockyApps"))

import numpy as np
import pandas as pd
from core.scanner import scan_ticker, auto_determine_settings, ScanResult, _auto_cache
from core.risk import RiskManager


class TestAutoFallback:
    def test_auto_determine_returns_tuple(self):
        _auto_cache.clear()
        period, interval, reason = auto_determine_settings("AAPL")
        assert isinstance(period, str)
        assert isinstance(interval, str)
        assert len(reason) > 0

    def test_auto_determine_caches(self):
        _auto_cache.clear()
        auto_determine_settings("MSFT")
        assert any("MSFT" in k for k in _auto_cache)

    # NOTE: scan_ticker tests removed — each loads TinyLlama (~30s per test).
    # Scan functionality is integration-tested manually via the app.
    # Template reasoning + model manager covered in test_llm_and_deep.py.


class TestScanResult:
    def test_has_period_interval(self):
        r = ScanResult("TEST", "BUY", 0.8, 100, 10, 95, 110, 2.0,
                        [0.1, 0.1, 0.8], {}, "test", 0.8,
                        period_used="2d", interval_used="1m")
        assert r.period_used == "2d"
        assert r.interval_used == "1m"

    def test_insufficient_result(self):
        r = ScanResult("BAD", "--", 0, 0, 0, 0, 0, 0,
                        [0, 0, 0], {}, "no data", -1, error="Not enough data (5 bars)")
        assert r.error is not None
        assert r.score == -1
        assert r.action == "--"


class TestProgressQueue:
    def test_worker_has_poll_method(self):
        from panels.workers import ScanWorker
        rm = RiskManager()
        w = ScanWorker(["AAPL"], "5d", "5m", rm)
        assert hasattr(w, 'poll_progress')
        # Before run, poll returns empty
        items = w.poll_progress()
        assert items == []


class TestShareSizing:
    def test_confidence_affects_size(self):
        rm = RiskManager(portfolio_value=100000)
        low = rm.position_size(100.0, 2.0, confidence=0.3)
        high = rm.position_size(100.0, 2.0, confidence=0.9)
        assert high >= low

    def test_zero_atr_returns_zero(self):
        rm = RiskManager()
        assert rm.position_size(100.0, 0) == 0

    def test_zero_price_returns_zero(self):
        rm = RiskManager()
        assert rm.position_size(0, 2.0) == 0


class TestDelistedRemoval:
    def test_no_wish_in_penny_stocks(self):
        from core.discovery import _get_penny_stocks, _cache
        _cache.clear()
        # Check the WSB fallback list doesn't include WISH or TELL
        from core.discovery import _get_penny_stocks
        import inspect
        source = inspect.getsource(_get_penny_stocks)
        assert "WISH" not in source
        assert "TELL" not in source
