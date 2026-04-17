"""Tests for core.logger — structured decision logging."""

import json
from core.logger import log_decision, log_event, log_trade_execution, get_today_logs


class TestLogger:
    def test_log_decision(self):
        entry = log_decision("AAPL", "BUY", 0.9, 150.0, 10, 147.0, 155.0, 2.0,
                             [0.05, 0.05, 0.9], reasoning="Test")
        assert entry["type"] == "decision"
        assert entry["ticker"] == "AAPL"
        assert entry["action"] == "BUY"

    def test_log_event(self):
        entry = log_event("test", "Unit test event")
        assert entry["type"] == "event"
        assert entry["message"] == "Unit test event"

    def test_log_execution(self):
        entry = log_trade_execution("TSLA", "buy", 5, "market", "abc123")
        assert entry["type"] == "execution"
        assert entry["order_id"] == "abc123"

    def test_read_today_logs(self):
        log_event("test", "Readback test")
        logs = get_today_logs(10)
        assert len(logs) > 0
        assert logs[0]["type"] in ("event", "decision", "execution", "scan")

    def test_feature_importances_logged(self):
        entry = log_decision("SPY", "HOLD", 0.5, 450.0, 0, 0, 0, 5.0,
                             [0.2, 0.6, 0.2],
                             feature_importances={"RSI_14": 500, "ema_cross": 300},
                             reasoning="Feature importance test")
        assert "feature_importances" in entry
        assert entry["feature_importances"]["RSI_14"] == 500
