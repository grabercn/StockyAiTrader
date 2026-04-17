"""Tests for core.broker — Alpaca API wrapper (no live calls, just structure)."""

from core.broker import AlpacaBroker


class TestBrokerInit:
    def test_creates_with_keys(self):
        broker = AlpacaBroker("fake_key", "fake_secret")
        assert broker.base_url.startswith("https://paper-api")
        assert broker.headers["APCA-API-KEY-ID"] == "fake_key"

    def test_paper_mode_default(self):
        broker = AlpacaBroker("k", "s", paper=True)
        assert "paper" in broker.base_url

    def test_live_mode(self):
        broker = AlpacaBroker("k", "s", paper=False)
        assert "paper" not in broker.base_url

    def test_bad_key_returns_error(self):
        broker = AlpacaBroker("bad", "bad")
        result = broker.get_account()
        assert "error" in result
