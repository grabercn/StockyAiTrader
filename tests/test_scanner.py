"""Tests for core.scanner — multi-stock scanning and ranking."""

from core.scanner import ScanResult, DEFAULT_TICKERS, TECH_TICKERS, ETF_TICKERS


class TestScanResult:
    def test_dataclass_creation(self):
        r = ScanResult(
            ticker="AAPL", action="BUY", confidence=0.85, price=150.0,
            position_size=10, stop_loss=147.0, take_profit=155.0, atr=2.0,
            probs=[0.05, 0.10, 0.85], feature_importances={"RSI_14": 500},
            reasoning="Test reason", score=0.8,
        )
        assert r.ticker == "AAPL"
        assert r.action == "BUY"
        assert r.score == 0.8
        assert r.error is None

    def test_error_result(self):
        r = ScanResult(
            ticker="BAD", action="HOLD", confidence=0, price=0,
            position_size=0, stop_loss=0, take_profit=0, atr=0,
            probs=[0, 1, 0], feature_importances={},
            reasoning="Error", score=0, error="Not enough data",
        )
        assert r.error is not None


class TestTickerLists:
    def test_default_has_entries(self):
        assert len(DEFAULT_TICKERS) >= 20

    def test_tech_has_entries(self):
        assert len(TECH_TICKERS) >= 8

    def test_etf_has_entries(self):
        assert len(ETF_TICKERS) >= 5

    def test_no_duplicates(self):
        assert len(DEFAULT_TICKERS) == len(set(DEFAULT_TICKERS))
        assert len(TECH_TICKERS) == len(set(TECH_TICKERS))
        assert len(ETF_TICKERS) == len(set(ETF_TICKERS))

    def test_all_uppercase(self):
        for t in DEFAULT_TICKERS + TECH_TICKERS + ETF_TICKERS:
            assert t == t.upper(), f"{t} should be uppercase"
