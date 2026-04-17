"""
Risk management for automated trading.

Enforces position sizing and loss limits to prevent catastrophic drawdowns.
Even the best model will have losing streaks — risk management keeps you alive.

Key rules:
- Never risk more than 2% of portfolio on a single trade
- Stop trading for the day if down 5% (prevents tilt/revenge trading)
- Maximum 5 simultaneous positions (diversification)
- Stop-loss and take-profit based on ATR (adapts to volatility)
"""

from datetime import datetime


class RiskManager:
    """
    Calculates position sizes and enforces trading limits.

    Usage:
        rm = RiskManager(portfolio_value=100000)
        can, reason = rm.can_trade()
        size = rm.position_size(price=150.0, atr=2.5)
        sl = rm.stop_loss(price=150.0, atr=2.5, side="buy")
        tp = rm.take_profit(price=150.0, atr=2.5, side="buy")
    """

    def __init__(
        self,
        portfolio_value=100_000,
        max_risk_per_trade=0.02,    # 2% of portfolio per trade
        max_daily_drawdown=0.05,    # 5% daily loss limit
        max_positions=5,            # Max simultaneous positions
        atr_stop_mult=1.5,          # Stop-loss = 1.5x ATR from entry
        atr_profit_mult=2.5,        # Take-profit = 2.5x ATR from entry (1.67:1 reward-to-risk)
    ):
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_positions = max_positions
        self.atr_stop_mult = atr_stop_mult
        self.atr_profit_mult = atr_profit_mult

        # Daily tracking — resets at midnight
        self.daily_pnl = 0.0
        self.open_positions = 0
        self._last_reset = datetime.now().date()

    def _reset_if_new_day(self):
        """Reset daily P&L counter at the start of each trading day."""
        today = datetime.now().date()
        if today != self._last_reset:
            self.daily_pnl = 0.0
            self._last_reset = today

    def can_trade(self):
        """
        Check if a new trade is allowed under current risk limits.

        Returns:
            (allowed: bool, reason: str)
        """
        self._reset_if_new_day()

        max_loss = self.portfolio_value * self.max_daily_drawdown
        if self.daily_pnl <= -max_loss:
            return False, f"Daily drawdown limit (${max_loss:,.0f}) reached"

        if self.open_positions >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"

        return True, "OK"

    def position_size(self, price, atr):
        """
        Calculate how many shares to buy based on ATR volatility.

        Logic: risk_amount / stop_distance = shares
        This means wider stops (volatile stocks) → fewer shares,
        and tighter stops (calm stocks) → more shares.

        Also capped at 10% of portfolio per position to prevent concentration.
        """
        if atr <= 0 or price <= 0:
            return 0

        risk_amount = self.portfolio_value * self.max_risk_per_trade
        stop_distance = atr * self.atr_stop_mult
        shares = int(risk_amount / stop_distance)

        # Never put more than 10% of portfolio in one position
        max_shares = int((self.portfolio_value * 0.10) / price)
        return min(shares, max_shares)

    def stop_loss(self, price, atr, side="buy"):
        """Calculate stop-loss price based on ATR distance."""
        distance = atr * self.atr_stop_mult
        return price - distance if side == "buy" else price + distance

    def take_profit(self, price, atr, side="buy"):
        """Calculate take-profit price based on ATR distance."""
        distance = atr * self.atr_profit_mult
        return price + distance if side == "buy" else price - distance
