"""Tests for core.event_bus — inter-panel signal communication."""

import sys
import pytest

# Qt signals need a QApplication — skip if display not available
try:
    from PyQt5.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt not available (headless)")
class TestEventBus:
    def test_log_entry_signal(self):
        from core.event_bus import EventBus
        bus = EventBus()
        received = []
        bus.log_entry.connect(lambda msg, lvl: received.append((msg, lvl)))
        bus.log_entry.emit("test", "info")
        assert len(received) == 1
        assert received[0] == ("test", "info")

    def test_ticker_selected_signal(self):
        from core.event_bus import EventBus
        bus = EventBus()
        received = []
        bus.ticker_selected.connect(lambda t: received.append(t))
        bus.ticker_selected.emit("AAPL")
        assert received == ["AAPL"]

    def test_multiple_subscribers(self):
        from core.event_bus import EventBus
        bus = EventBus()
        a, b = [], []
        bus.log_entry.connect(lambda msg, _: a.append(msg))
        bus.log_entry.connect(lambda msg, _: b.append(msg))
        bus.log_entry.emit("hello", "info")
        assert a == ["hello"]
        assert b == ["hello"]
