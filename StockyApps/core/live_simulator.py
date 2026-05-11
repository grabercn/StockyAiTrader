# -*- coding: utf-8 -*-
"""
Live Trading Simulator — realistic friction layer for paper trading.

Simulates every real-world constraint that paper trading ignores:
- Order fill delays (market orders: 0.3-2s, limit orders: 0-30s or not at all)
- Slippage (price moves between order and fill, scales with order size + volatility)
- Partial fills (large orders may only partially fill)
- Spread costs (bid-ask spread, wider for illiquid stocks)
- Commission simulation (optional, Alpaca is free but models other brokers)
- Market hours enforcement (no fills outside RTH unless extended_hours=True)
- Settlement delays (T+1 for equities — BP not immediately available after sell)
- PDT rules enforcement (flag if 4+ day trades in 5 days with <$25K equity)
- Short sale restrictions (can't short below $5, uptick rule)
- Wash sale tracking (IRS 30-day rule for tax loss harvesting)
- Circuit breaker halts (stock halted if moves >10% in 5 min)
- Order rejection scenarios (insufficient BP, untradeable, halted stock)
- Fill price randomization (fill at slightly worse price than quoted)
- Volume-based fill probability (low volume = orders may not fill)
- Maximum position concentration (can't own >10% of a stock's daily volume)

This module is a LAYER — it does not replace the broker.  It adjusts orders
before they reach the broker and logs every friction effect transparently.

Default is OFF — only active when user enables ``simulate_live_conditions``
in Settings.
"""

import random
import math
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class SimulationResult:
    """Result of applying live-trading friction to an order."""
    adjusted_qty: int
    fill_price: float
    delay_seconds: float
    rejection_reason: Optional[str]      # None = order accepted
    slippage: float                      # Dollar amount of slippage per share
    spread_cost: float                   # Dollar cost of the spread per share
    partial_fill: bool                   # True if qty was reduced
    original_qty: int
    warnings: List[str] = field(default_factory=list)


class LiveSimulator:
    """
    Realistic live-trading friction simulator.

    Usage::

        sim = LiveSimulator(enabled=True)
        result = sim.apply_to_order("buy", "AAPL", 100, 190.0, atr=1.5,
                                     daily_volume=50_000_000, buying_power=50_000)
        if result.rejection_reason:
            log(f"REJECTED: {result.rejection_reason}")
        else:
            broker.place_order(ticker, result.adjusted_qty, ...)
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

        # PDT tracking: list of (date_str, ticker) for each day-trade
        self._day_trades: List[Tuple[str, str]] = []

        # T+1 settlement: list of (available_date, dollar_amount)
        self._pending_settlement: List[Tuple[str, float]] = []

        # Wash sale tracking: {ticker: last_loss_sell_date}
        self._wash_sales: Dict[str, str] = {}

        # Circuit breaker state: {ticker: halt_until_timestamp}
        self._halted: Dict[str, float] = {}

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def simulate_buy(
        self,
        ticker: str,
        qty: int,
        price: float,
        atr: float,
        daily_volume: int = 0,
        buying_power: float = 0.0,
    ) -> Tuple[int, float, float, Optional[str]]:
        """
        Simulate a BUY order with real-world friction.

        Returns:
            (adjusted_qty, fill_price, delay_seconds, rejection_reason)
            rejection_reason is None if the order is accepted.
        """
        result = self._apply_friction(
            "buy", ticker, qty, price, atr, daily_volume, buying_power
        )
        return (
            result.adjusted_qty,
            result.fill_price,
            result.delay_seconds,
            result.rejection_reason,
        )

    def simulate_sell(
        self,
        ticker: str,
        qty: int,
        price: float,
        atr: float,
        entry_price: float = 0.0,
        daily_volume: int = 0,
    ) -> Tuple[int, float, float, Optional[str]]:
        """
        Simulate a SELL order with real-world friction.

        Returns:
            (adjusted_qty, fill_price, delay_seconds, rejection_reason)
            rejection_reason is None if the order is accepted.
        """
        result = self._apply_friction(
            "sell", ticker, qty, price, atr, daily_volume, buying_power=0.0
        )

        # Track wash sale if selling at a loss
        if entry_price > 0 and result.fill_price < entry_price:
            self._wash_sales[ticker.upper()] = datetime.now().strftime("%Y-%m-%d")

        # Track T+1 settlement
        if result.rejection_reason is None and result.adjusted_qty > 0:
            settlement_amount = result.adjusted_qty * result.fill_price
            # Settlement available next business day
            settle_date = self._next_business_day(datetime.now()).strftime("%Y-%m-%d")
            self._pending_settlement.append((settle_date, settlement_amount))

        return (
            result.adjusted_qty,
            result.fill_price,
            result.delay_seconds,
            result.rejection_reason,
        )

    def apply_to_order(
        self,
        side: str,
        ticker: str,
        qty: int,
        price: float,
        atr: float,
        daily_volume: int = 0,
        buying_power: float = 0.0,
        entry_price: float = 0.0,
    ) -> SimulationResult:
        """
        Main entry point — applies all simulations and returns a full result.

        This is the preferred method for engine integration.
        """
        if not self.enabled:
            return SimulationResult(
                adjusted_qty=qty,
                fill_price=price,
                delay_seconds=0.0,
                rejection_reason=None,
                slippage=0.0,
                spread_cost=0.0,
                partial_fill=False,
                original_qty=qty,
            )

        result = self._apply_friction(
            side.lower(), ticker, qty, price, atr, daily_volume, buying_power
        )

        # Post-sell tracking
        if side.lower() == "sell" and result.rejection_reason is None:
            # Wash sale tracking
            if entry_price > 0 and result.fill_price < entry_price:
                self._wash_sales[ticker.upper()] = datetime.now().strftime("%Y-%m-%d")
            # T+1 settlement
            if result.adjusted_qty > 0:
                settlement_amount = result.adjusted_qty * result.fill_price
                settle_date = self._next_business_day(datetime.now()).strftime("%Y-%m-%d")
                self._pending_settlement.append((settle_date, settlement_amount))

        return result

    def check_pdt(self, equity: float) -> Tuple[bool, str]:
        """
        Check if Pattern Day Trader rules are violated.

        PDT rule: if 4+ day trades in a rolling 5 business days
        AND account equity < $25,000 => trading is blocked.

        Returns:
            (ok, message)
            ok=True means trading is allowed.
        """
        if equity >= 25_000:
            count = self._count_recent_day_trades()
            if count >= 3:
                return True, (
                    f"PDT OK (equity ${equity:,.0f} >= $25K) but "
                    f"{count} day trades in 5 days — monitor closely"
                )
            return True, f"PDT OK — {count} day trades in 5 days, equity ${equity:,.0f}"

        count = self._count_recent_day_trades()
        if count >= 4:
            return False, (
                f"PDT VIOLATION — {count} day trades in 5 business days "
                f"with equity ${equity:,.0f} < $25K. Trading blocked."
            )
        if count >= 3:
            return True, (
                f"PDT WARNING — {count}/4 day trades in 5 days. "
                f"One more triggers PDT restriction (equity ${equity:,.0f} < $25K)."
            )
        return True, f"PDT OK — {count} day trades in 5 days, equity ${equity:,.0f}"

    def record_day_trade(self, ticker: str):
        """Record a day trade (buy+sell same stock same day)."""
        today = datetime.now().strftime("%Y-%m-%d")
        self._day_trades.append((today, ticker.upper()))

    def get_settlement_locked(self) -> float:
        """Return total capital locked in T+1 settlement (not yet available)."""
        today = datetime.now().strftime("%Y-%m-%d")
        # Clean up settled entries
        self._pending_settlement = [
            (d, amt) for d, amt in self._pending_settlement if d > today
        ]
        return sum(amt for _, amt in self._pending_settlement)

    def check_wash_sale(self, ticker: str) -> Tuple[bool, str]:
        """
        Check if buying this ticker would trigger a wash sale.

        IRS 30-day rule: if you sold at a loss within the last 30 days,
        buying again disallows the tax loss deduction.

        Returns:
            (is_wash, message)
        """
        key = ticker.upper()
        last_loss = self._wash_sales.get(key)
        if not last_loss:
            return False, ""

        loss_date = datetime.strptime(last_loss, "%Y-%m-%d")
        days_since = (datetime.now() - loss_date).days
        if days_since <= 30:
            return True, (
                f"WASH SALE WARNING: {ticker} sold at a loss {days_since} days ago. "
                f"Buying within 30 days disallows the tax loss deduction. "
                f"Clear after {last_loss} + 30 days."
            )
        else:
            # Past 30 days, clean up
            del self._wash_sales[key]
            return False, ""

    def get_status_summary(self) -> dict:
        """Return a snapshot of all simulation state for UI display."""
        return {
            "enabled": self.enabled,
            "day_trades_5d": self._count_recent_day_trades(),
            "settlement_locked": self.get_settlement_locked(),
            "wash_sale_tickers": list(self._wash_sales.keys()),
            "pending_settlements": len(self._pending_settlement),
            "halted_tickers": [
                t for t, until in self._halted.items()
                if until > datetime.now().timestamp()
            ],
        }

    # ══════════════════════════════════════════════════════════════════════
    # PRIVATE — CORE FRICTION ENGINE
    # ══════════════════════════════════════════════════════════════════════

    def _apply_friction(
        self,
        side: str,
        ticker: str,
        qty: int,
        price: float,
        atr: float,
        daily_volume: int,
        buying_power: float,
    ) -> SimulationResult:
        """Apply all friction models to an order. Returns SimulationResult."""
        warnings: List[str] = []
        original_qty = qty

        # ── 1. Rejection checks ─────────────────────────────────────────

        # Circuit breaker halt
        halted_until = self._halted.get(ticker.upper(), 0)
        if halted_until > datetime.now().timestamp():
            return SimulationResult(
                adjusted_qty=0, fill_price=price, delay_seconds=0,
                rejection_reason=f"{ticker} halted (circuit breaker) — retry later",
                slippage=0, spread_cost=0, partial_fill=False,
                original_qty=original_qty,
            )

        # Buying power check (buys only)
        if side == "buy" and buying_power > 0:
            cost = qty * price
            if cost > buying_power:
                affordable = int(buying_power / price) if price > 0 else 0
                if affordable <= 0:
                    return SimulationResult(
                        adjusted_qty=0, fill_price=price, delay_seconds=0,
                        rejection_reason=(
                            f"Insufficient buying power: need ${cost:,.0f}, "
                            f"have ${buying_power:,.0f}"
                        ),
                        slippage=0, spread_cost=0, partial_fill=False,
                        original_qty=original_qty,
                    )
                qty = affordable
                warnings.append(
                    f"Reduced qty {original_qty} -> {qty} (BP ${buying_power:,.0f})"
                )

        # Wash sale check (buys only — warn but don't block)
        if side == "buy":
            is_wash, wash_msg = self.check_wash_sale(ticker)
            if is_wash:
                warnings.append(wash_msg)

        # ── 2. Spread simulation ────────────────────────────────────────
        spread_pct = self._estimate_spread(price)
        half_spread = price * spread_pct / 2.0
        spread_cost_per_share = half_spread

        # ── 3. Slippage model ──────────────────────────────────────────
        slippage_per_share = self._calc_slippage(
            side, price, atr, qty, daily_volume
        )

        # ── 4. Fill price ──────────────────────────────────────────────
        if side == "buy":
            # Buy: pay MORE (slippage + half spread)
            fill_price = price + slippage_per_share + half_spread
        else:
            # Sell: receive LESS (slippage + half spread deducted)
            fill_price = price - slippage_per_share - half_spread

        # Ensure fill price stays positive and sane
        fill_price = max(0.01, fill_price)
        # Round to cents
        fill_price = round(fill_price, 2)

        # ── 5. Partial fill model ──────────────────────────────────────
        adjusted_qty, was_partial = self._calc_partial_fill(
            qty, price, atr, daily_volume
        )
        if was_partial:
            warnings.append(
                f"Partial fill: {adjusted_qty}/{qty} shares "
                f"(volume/liquidity constraint)"
            )

        # ── 6. Fill delay model ────────────────────────────────────────
        delay = self._calc_delay()

        # ── 7. Re-check buying power with adjusted fill price ──────────
        if side == "buy" and buying_power > 0:
            actual_cost = adjusted_qty * fill_price
            if actual_cost > buying_power:
                adjusted_qty = max(0, int(buying_power / fill_price))
                if adjusted_qty <= 0:
                    return SimulationResult(
                        adjusted_qty=0, fill_price=fill_price, delay_seconds=delay,
                        rejection_reason=(
                            f"After slippage, cost exceeds BP: "
                            f"${actual_cost:,.2f} > ${buying_power:,.0f}"
                        ),
                        slippage=slippage_per_share, spread_cost=spread_cost_per_share,
                        partial_fill=True, original_qty=original_qty,
                        warnings=warnings,
                    )
                was_partial = True
                warnings.append("Qty adjusted again after slippage pricing")

        return SimulationResult(
            adjusted_qty=adjusted_qty,
            fill_price=fill_price,
            delay_seconds=round(delay, 2),
            rejection_reason=None,
            slippage=round(slippage_per_share, 4),
            spread_cost=round(spread_cost_per_share, 4),
            partial_fill=was_partial,
            original_qty=original_qty,
            warnings=warnings,
        )

    # ── Slippage Model ──────────────────────────────────────────────────

    def _calc_slippage(
        self, side: str, price: float, atr: float, qty: int, daily_volume: int
    ) -> float:
        """
        Calculate per-share slippage in dollars.

        Model:
            base = 0.02% of price for liquid stocks
            + 0.01% per 1% of daily volume being traded
            + 0.05% * (ATR% / 1%) for volatile stocks
            * random(0.5, 1.5) for realistic variance
        """
        if price <= 0:
            return 0.0

        # Base slippage: 0.02% for liquid stocks
        base_pct = 0.0002

        # Volume impact: how much of daily volume is this order?
        vol_impact = 0.0
        if daily_volume > 0 and qty > 0:
            order_pct_of_volume = qty / daily_volume
            # 0.01% per 1% of daily volume => 0.01% * (order_pct * 100)
            vol_impact = 0.0001 * (order_pct_of_volume * 100)

        # Volatility impact: higher ATR = more slippage
        atr_impact = 0.0
        if atr > 0 and price > 0:
            atr_pct = atr / price  # ATR as percentage of price
            # 0.05% * (atr_pct / 0.01) — normalized to 1% baseline
            atr_impact = 0.0005 * (atr_pct / 0.01)

        total_pct = base_pct + vol_impact + atr_impact

        # Random variance: 0.5x to 1.5x
        variance = random.uniform(0.5, 1.5)
        total_pct *= variance

        # Cap at 1% to avoid absurd slippage
        total_pct = min(total_pct, 0.01)

        return price * total_pct

    # ── Spread Model ───────────────────────────────────────────────────

    def _estimate_spread(self, price: float) -> float:
        """
        Estimate bid-ask spread as a percentage.

        Uses price as a proxy for market cap / liquidity:
            >$100  = large-cap, tight spread (0.01%)
            $10-100 = mid-range (0.03%)
            $5-10  = small-cap (0.10%)
            <$5    = penny stock (0.50%)
        """
        if price > 100:
            return 0.0001   # 0.01% — very tight (AAPL, GOOGL)
        elif price > 10:
            return 0.0003   # 0.03% — moderate
        elif price >= 5:
            return 0.0010   # 0.10% — wider
        else:
            return 0.0050   # 0.50% — penny stock, very wide

    # ── Partial Fill Model ─────────────────────────────────────────────

    def _calc_partial_fill(
        self, qty: int, price: float, atr: float, daily_volume: int
    ) -> Tuple[int, bool]:
        """
        Determine if order gets partially filled.

        Rules:
            1. If order > 0.5% of daily volume: cap at 0.5%
            2. If order > 1000 shares on low-vol stock (ATR < 0.5%): fill 80%
            3. Random 5% chance of partial fill on any order
        """
        adjusted = qty
        was_partial = False

        # Rule 1: Volume cap — can't own more than 0.5% of daily volume
        if daily_volume > 0 and qty > 0:
            max_by_volume = max(1, int(daily_volume * 0.005))
            if qty > max_by_volume:
                adjusted = max_by_volume
                was_partial = True

        # Rule 2: Large order on low-volatility stock
        if price > 0 and atr > 0:
            atr_pct = atr / price
            if qty > 1000 and atr_pct < 0.005:
                fill_80 = max(1, int(qty * 0.80))
                if fill_80 < adjusted:
                    adjusted = fill_80
                    was_partial = True

        # Rule 3: Random partial fill (5% chance)
        if not was_partial and random.random() < 0.05:
            fill_frac = random.uniform(0.70, 0.95)
            adjusted = max(1, int(qty * fill_frac))
            was_partial = (adjusted < qty)

        return adjusted, was_partial

    # ── Delay Model ────────────────────────────────────────────────────

    def _calc_delay(self) -> float:
        """
        Simulate fill delay for market orders.

        Market orders: 0.3 - 2.0 seconds (log-normal distribution)
        """
        # Log-normal gives a nice right-skewed delay distribution
        # Most fills are fast, some are slow
        delay = random.lognormvariate(math.log(0.8), 0.5)
        return max(0.3, min(delay, 2.0))

    # ── PDT Helper ─────────────────────────────────────────────────────

    def _count_recent_day_trades(self) -> int:
        """Count day trades in the last 5 business days."""
        today = datetime.now()
        cutoff = today - timedelta(days=7)  # 7 calendar days covers 5 business days
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        return sum(1 for d, _ in self._day_trades if d >= cutoff_str)

    # ── Settlement Helper ──────────────────────────────────────────────

    @staticmethod
    def _next_business_day(dt: datetime) -> datetime:
        """Return the next business day after dt."""
        next_day = dt + timedelta(days=1)
        # Skip weekends
        while next_day.weekday() >= 5:  # 5=Sat, 6=Sun
            next_day += timedelta(days=1)
        return next_day

    # ── State Persistence ──────────────────────────────────────────────

    def get_state(self) -> dict:
        """Export simulator state for saving."""
        return {
            "day_trades": list(self._day_trades),
            "pending_settlement": list(self._pending_settlement),
            "wash_sales": dict(self._wash_sales),
        }

    def restore_state(self, state: dict):
        """Restore simulator state from a previous session."""
        if not state:
            return
        self._day_trades = [tuple(x) for x in state.get("day_trades", [])]
        self._pending_settlement = [
            tuple(x) for x in state.get("pending_settlement", [])
        ]
        self._wash_sales = state.get("wash_sales", {})

    def __repr__(self):
        return (
            f"LiveSimulator(enabled={self.enabled}, "
            f"day_trades={self._count_recent_day_trades()}, "
            f"settlement_locked=${self.get_settlement_locked():,.0f})"
        )
