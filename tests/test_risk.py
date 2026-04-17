"""Tests for core.risk — position sizing and risk management."""

from core.risk import RiskManager


class TestPositionSizing:
    def test_basic_sizing(self):
        rm = RiskManager(portfolio_value=100000)
        size = rm.position_size(150.0, 2.5)
        assert size > 0
        # Risk = 2% of 100k = $2000, stop = 2.5 * 1.5 = $3.75
        # Expected ~533 shares, capped at 10% = 66 shares
        assert size <= 66  # 10% cap: 100000 * 0.10 / 150 = 66

    def test_zero_atr(self):
        rm = RiskManager()
        assert rm.position_size(150.0, 0) == 0

    def test_zero_price(self):
        rm = RiskManager()
        assert rm.position_size(0, 2.5) == 0

    def test_volatile_stock_fewer_shares(self):
        # Use low price + big portfolio so the 10% cap doesn't mask ATR differences
        rm = RiskManager(portfolio_value=10000000)
        calm = rm.position_size(10.0, 0.1)    # Low volatility
        wild = rm.position_size(10.0, 5.0)    # High volatility
        assert wild < calm  # More volatile = fewer shares


class TestStopLoss:
    def test_buy_stop_below_price(self):
        rm = RiskManager()
        sl = rm.stop_loss(150.0, 2.0, "buy")
        assert sl < 150.0

    def test_sell_stop_above_price(self):
        rm = RiskManager()
        sl = rm.stop_loss(150.0, 2.0, "sell")
        assert sl > 150.0


class TestTakeProfit:
    def test_buy_tp_above_price(self):
        rm = RiskManager()
        tp = rm.take_profit(150.0, 2.0, "buy")
        assert tp > 150.0

    def test_reward_exceeds_risk(self):
        rm = RiskManager()
        sl = rm.stop_loss(150.0, 2.0, "buy")
        tp = rm.take_profit(150.0, 2.0, "buy")
        risk = 150.0 - sl
        reward = tp - 150.0
        assert reward > risk  # Reward-to-risk > 1


class TestCanTrade:
    def test_default_allows(self):
        rm = RiskManager()
        can, reason = rm.can_trade()
        assert can is True

    def test_max_positions_blocks(self):
        rm = RiskManager(max_positions=2)
        rm.open_positions = 2
        can, reason = rm.can_trade()
        assert can is False
        assert "positions" in reason.lower()

    def test_drawdown_blocks(self):
        rm = RiskManager(portfolio_value=100000, max_daily_drawdown=0.05)
        rm.daily_pnl = -5001  # Over 5% of 100k
        can, reason = rm.can_trade()
        assert can is False
        assert "drawdown" in reason.lower()
