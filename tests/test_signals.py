"""Tests for core.signals — signal file I/O."""

import os
import json
from core.signals import write_signal, read_signal, _SIGNAL_FILE


class TestSignals:
    def test_write_and_read(self):
        sig = write_signal("AAPL", "BUY", 0.85, 150.0, 10, 147.0, 155.0, 2.0)
        assert sig["ticker"] == "AAPL"
        assert sig["action"] == "BUY"

        read_back = read_signal()
        assert read_back is not None
        assert read_back["ticker"] == "AAPL"
        assert read_back["confidence"] == 0.85

    def test_read_missing_file(self, tmp_path, monkeypatch):
        import core.signals as sig_mod
        monkeypatch.setattr(sig_mod, "_SIGNAL_FILE", str(tmp_path / "nonexistent.json"))
        assert sig_mod.read_signal() is None

    def test_signal_has_timestamp(self):
        sig = write_signal("TSLA", "SELL", 0.7, 200.0, 5, 203.0, 195.0, 3.0)
        assert "timestamp" in sig
