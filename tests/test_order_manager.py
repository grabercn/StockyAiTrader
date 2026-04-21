"""Tests for core.order_manager and core.llm_reasoner."""

from core.order_manager import OrderManager, TrackedOrder
from core.llm_reasoner import generate_reasoning, _template_reasoning, is_available
from core.risk import RiskManager


class TestTrackedOrder:
    def test_creation(self):
        o = TrackedOrder(
            order_id="abc123", symbol="AAPL", side="buy", qty=10,
            order_type="market", status="new", submitted_at="2024-01-01T10:00:00",
        )
        assert o.symbol == "AAPL"
        assert o.status == "new"
        assert o.filled_qty == 0

    def test_track_from_alpaca_data(self):
        om = OrderManager(broker=None)
        om.track_order({
            "id": "test123", "symbol": "TSLA", "side": "buy",
            "qty": "5", "type": "market", "status": "accepted",
            "submitted_at": "2024-01-01T10:00:00Z",
        })
        orders = om.get_recent_orders()
        assert len(orders) == 1
        assert orders[0].symbol == "TSLA"

    def test_get_active_orders(self):
        om = OrderManager(broker=None)
        om.track_order({"id": "1", "symbol": "A", "side": "buy", "qty": "1",
                        "type": "market", "status": "new", "submitted_at": ""})
        om.track_order({"id": "2", "symbol": "B", "side": "sell", "qty": "1",
                        "type": "market", "status": "filled", "submitted_at": ""})
        active = om.get_active_orders()
        assert len(active) == 1
        assert active[0].symbol == "A"


class TestLLMReasoner:
    def test_template_fallback(self):
        """Template reasoning always works, even without model."""
        result = _template_reasoning(
            "AAPL", "BUY", 0.85, 150.0, 2.5,
            [0.05, 0.10, 0.85], "low", "bullish", "RSI_14, ema_cross"
        )
        assert "BUY" in result
        assert "AAPL" in result
        assert "150" in result

    # NOTE: generate_reasoning test removed — loads TinyLlama (~30s).
    # Covered by test_llm_and_deep.py::TestLLMReasoner::test_generate_reasoning_returns_string

    def test_is_available_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)


class TestSmartPositionSizing:
    def test_basic(self):
        rm = RiskManager(portfolio_value=100000)
        size = rm.position_size(150.0, 2.5)
        assert size > 0

    def test_buying_power_cap(self):
        rm = RiskManager(portfolio_value=100000)
        full = rm.position_size(150.0, 2.5)
        limited = rm.position_size(150.0, 2.5, buying_power=500)
        assert limited <= full

    def test_high_confidence_more_shares(self):
        rm = RiskManager(portfolio_value=1000000)
        low = rm.position_size(50.0, 1.0, confidence=0.3)
        high = rm.position_size(50.0, 1.0, confidence=0.9)
        assert high >= low

    def test_many_positions_reduces(self):
        rm = RiskManager(portfolio_value=1000000)
        few = rm.position_size(50.0, 1.0, existing_positions=1)
        many = rm.position_size(50.0, 1.0, existing_positions=8)
        assert many <= few

    def test_aggressivity_multiplier(self):
        rm = RiskManager(portfolio_value=100000)
        normal = rm.position_size(100.0, 2.0, aggressivity_mult=1.0)
        yolo = rm.position_size(100.0, 2.0, aggressivity_mult=2.0)
        assert yolo >= normal
