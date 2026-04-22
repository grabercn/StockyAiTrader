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
        atr_stop_mult=0.9,          # Stop-loss = 0.9x ATR from entry (grid search on 5139 decisions)
        atr_profit_mult=5.5,        # Take-profit = 5.5x ATR from entry (grid search: +19.9% return)
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

    def position_size(self, price, atr, buying_power=None, existing_positions=0,
                       confidence=0.5, aggressivity_mult=1.0):
        """
        Smart position sizing based on ATR, buying power, confidence, and portfolio state.

        Factors in:
        - ATR volatility (core: wider stops → fewer shares)
        - Available buying power (never exceed what you can afford)
        - Confidence (higher confidence → allow slightly larger position)
        - Existing positions (more positions → smaller each to diversify)
        - Aggressivity multiplier (from profile: Chill=0.5x, YOLO=2x)
        - 10% portfolio cap per position

        Returns number of shares (0 if conditions aren't met).
        """
        if atr <= 0 or price <= 0:
            return 0

        # Base: ATR-based sizing
        risk_amount = self.portfolio_value * self.max_risk_per_trade
        stop_distance = atr * self.atr_stop_mult
        shares = int(risk_amount / stop_distance)

        # Portfolio concentration cap (10% per position)
        max_shares = int((self.portfolio_value * 0.10) / price)
        shares = min(shares, max_shares)

        # Buying power cap (can't buy more than you can afford)
        if buying_power is not None and buying_power > 0:
            affordable = int(buying_power * 0.25 / price)  # Use max 25% of buying power per trade
            shares = min(shares, affordable)

        # Confidence scaling: high confidence = up to 1.3x, low = down to 0.7x
        conf_mult = 0.7 + (confidence * 0.6)  # 0% → 0.7x, 100% → 1.3x
        shares = int(shares * conf_mult)

        # Diversification: reduce size when already holding many positions
        if existing_positions >= 3:
            diversity_mult = max(0.5, 1.0 - (existing_positions - 2) * 0.1)
            shares = int(shares * diversity_mult)

        # Aggressivity profile multiplier
        shares = max(1, int(shares * aggressivity_mult))

        return shares

    def stop_loss(self, price, atr, side="buy"):
        """Calculate stop-loss price based on ATR distance."""
        distance = atr * self.atr_stop_mult
        return price - distance if side == "buy" else price + distance

    def take_profit(self, price, atr, side="buy"):
        """Calculate take-profit price based on ATR distance."""
        distance = atr * self.atr_profit_mult
        return price + distance if side == "buy" else price - distance
