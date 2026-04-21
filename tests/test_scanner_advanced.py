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

    def test_scan_ticker_returns_result(self):
        rm = RiskManager()
        r = scan_ticker("AAPL", "5d", "5m", rm)
        assert isinstance(r, ScanResult)
        assert r.ticker == "AAPL"
        assert r.action in ("BUY", "SELL", "HOLD")

    def test_scan_ticker_auto_mode(self):
        rm = RiskManager()
        r = scan_ticker("MSFT", "5d", "5m", rm, auto_settings=True)
        assert isinstance(r, ScanResult)
        assert r.period_used in ("2d", "3d", "5d")
        assert r.interval_used in ("1m", "5m", "15m", "30m")

    def test_insufficient_data_returns_grey_result(self):
        """A ticker with no data should return score=-1 and error."""
        rm = RiskManager()
        r = scan_ticker("ZZZZZNOTREAL", "1d", "1m", rm)
        assert r.error is not None
        assert r.score <= 0


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
