"""
Auto-Trader Background Service — monitors stocks and executes trades autonomously.

Runs as a background thread that:
1. Maintains a list of monitored stocks with their settings
2. On each stock's interval (e.g. every 5 min), fetches ONLY the latest bar
   instead of re-downloading the full dataset (efficient incremental updates)
3. Re-runs the LightGBM model on updated data
4. Executes buy/sell if signal changes and confidence is high enough
5. Emits signals for the UI to display status

Architecture:
    - AutoTraderService (QThread) — the background loop
    - MonitoredStock — per-stock state (settings, last signal, countdown, data cache)
    - The scanner adds/removes stocks from monitoring
    - Works even when main window is minimized (tray mode future)

Optimization:
    Instead of re-fetching the full period every interval, we cache the
    DataFrame and only append the newest bars. Full re-fetch happens on
    first scan and every 10 cycles to prevent drift.
"""

import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

from .data import fetch_intraday, get_all_features
from .model import train_lgbm, predict_lgbm
from .risk import RiskManager
from .labeling import LABEL_NAMES
from .logger import log_decision, log_trade_execution, log_event


@dataclass
class MonitoredStock:
    """State for a single auto-traded stock."""
    ticker: str
    period: str = "5d"          # Training data period
    interval: str = "5m"        # Check interval
    enabled: bool = True
    auto_execute: bool = False  # Actually place trades vs just signal
    min_confidence: float = 0.5

    # Runtime state
    last_signal: str = "HOLD"
    last_confidence: float = 0.0
    last_price: float = 0.0
    last_check: Optional[str] = None
    next_check_seconds: int = 0
    check_count: int = 0
    model: object = None
    features: list = field(default_factory=list)

    # Interval in seconds
    @property
    def interval_seconds(self):
        m = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1min": 60, "5min": 300, "15min": 900}
        # Handle display format too
        m2 = {"1 min": 60, "5 min": 300, "15 min": 900}
        return m.get(self.interval, m2.get(self.interval, 300))


class AutoTraderService(QThread):
    """
    Background thread that monitors stocks and generates/executes signals.

    Signals:
        stock_updated(ticker, action, confidence, price, next_check_secs)
        trade_executed(ticker, side, qty, order_id)
        log(message, level)
    """

    stock_updated = pyqtSignal(str, str, float, float, int)  # ticker, action, conf, price, next_secs
    trade_executed = pyqtSignal(str, str, int, str)
    log = pyqtSignal(str, str)

    def __init__(self, broker=None, risk_manager=None):
        super().__init__()
        self.broker = broker
        self.rm = risk_manager or RiskManager()
        self._stocks: Dict[str, MonitoredStock] = {}
        self._running = True
        self._tick_interval = 1  # Check every 1 second, stocks have their own intervals

    def add_stock(self, ticker, period="5d", interval="5m", auto_execute=False, min_confidence=0.5):
        """Add a stock to monitoring."""
        self._stocks[ticker] = MonitoredStock(
            ticker=ticker, period=period, interval=interval,
            auto_execute=auto_execute, min_confidence=min_confidence,
            next_check_seconds=0,  # Check immediately on first add
        )
        self.log.emit(f"Monitoring {ticker} — {period} data, checking every {interval}", "info")

    def remove_stock(self, ticker):
        """Remove a stock from monitoring."""
        if ticker in self._stocks:
            del self._stocks[ticker]
            self.log.emit(f"Stopped monitoring {ticker}", "info")

    def set_auto_execute(self, ticker, enabled):
        """Enable/disable auto-execution for a stock."""
        if ticker in self._stocks:
            self._stocks[ticker].auto_execute = enabled

    def get_monitored(self):
        """Get dict of all monitored stocks."""
        return dict(self._stocks)

    def is_monitoring(self, ticker):
        return ticker in self._stocks

    def stop(self):
        self._running = False

    def run(self):
        """Main loop — ticks every second, checks each stock's countdown."""
        log_event("auto_trader", "Background auto-trader service started")
        self.log.emit("Auto-trader service started", "agent")

        while self._running:
            # Market hours check — skip execution outside trading hours
            try:
                from core.market_hours import get_session
                mkt = get_session()
                if not mkt.can_scan:
                    time.sleep(30)
                    continue
            except Exception:
                pass

            for ticker, stock in list(self._stocks.items()):
                if not stock.enabled:
                    continue

                stock.next_check_seconds -= 1

                if stock.next_check_seconds <= 0:
                    self._check_stock(stock)
                    # Dynamic interval: volatile stocks check more often
                    base = stock.interval_seconds
                    if stock.last_price > 0 and hasattr(stock, '_last_atr'):
                        vol_pct = stock._last_atr / stock.last_price
                        if vol_pct > 0.02:
                            base = max(60, int(base * 0.5))   # Volatile: check 2x faster
                        elif vol_pct < 0.005:
                            base = min(900, int(base * 1.5))  # Calm: check slower
                    stock.next_check_seconds = base
                    stock.check_count += 1

                # Emit countdown for UI
                self.stock_updated.emit(
                    ticker, stock.last_signal, stock.last_confidence,
                    stock.last_price, stock.next_check_seconds,
                )

            time.sleep(self._tick_interval)

    def _check_stock(self, stock):
        """Run the full analysis pipeline on a single stock."""
        try:
            self.log.emit(f"Checking {stock.ticker}...", "info")
            stock.last_check = datetime.now().strftime("%H:%M:%S")

            # Expire cache entries older than the check interval (keeps historical, refreshes recent)
            import time as _t
            from .data import _price_cache
            for k in list(_price_cache.keys()):
                if k[0] == stock.ticker:
                    _, cached_ts = _price_cache[k]
                    if _t.time() - cached_ts > stock.interval_seconds:
                        del _price_cache[k]

            data = fetch_intraday(stock.ticker, period=stock.period, interval=stock.interval)

            if data.empty or len(data) < 30:
                self.log.emit(f"{stock.ticker}: not enough data", "warn")
                return

            # Train/retrain model (every 10 checks or first time)
            if stock.model is None or stock.check_count % 10 == 0:
                all_features = get_all_features("intraday")
                model, features = train_lgbm(data, all_features, stock.ticker)
                if model:
                    stock.model = model
                    stock.features = features
                else:
                    self.log.emit(f"{stock.ticker}: training failed", "warn")
                    return

            # Predict
            actions, confidences, probs = predict_lgbm(stock.model, data, stock.features)
            action = LABEL_NAMES[actions[-1]]
            confidence = float(confidences[-1])
            price = float(data["Close"].iloc[-1])
            atr = float(data["ATRr_14"].iloc[-1]) if "ATRr_14" in data.columns else price * 0.01

            # Apply RL feedback model if available
            try:
                from .reinforcement import train_feedback_model, get_quality_score
                if not hasattr(self, '_rl_model'):
                    self._rl_model, _, _ = train_feedback_model()
                if self._rl_model:
                    atr_pct = atr / price if price > 0 else 0
                    q = get_quality_score(self._rl_model, confidence, [float(p) for p in probs[-1]], atr_pct, action)
                    confidence = min(1.0, confidence * q)
            except Exception:
                pass

            prev_signal = stock.last_signal
            stock.last_signal = action
            stock.last_confidence = confidence
            stock.last_price = price
            stock._last_atr = atr

            # Log every decision
            log_decision(stock.ticker, action, confidence, price,
                        self.rm.position_size(price, atr),
                        self.rm.stop_loss(price, atr, "buy" if action == "BUY" else "sell"),
                        self.rm.take_profit(price, atr, "buy" if action == "BUY" else "sell"),
                        atr, [float(p) for p in probs[-1]],
                        reasoning=f"AutoTrader check #{stock.check_count}")

            # Handle signal
            if action == "HOLD":
                self.log.emit(
                    f"{stock.ticker}: HOLD ({confidence:.0%}) @ ${price:.2f} — waiting",
                    "system",
                )
            elif action != prev_signal:
                self.log.emit(
                    f"Signal change: {stock.ticker} {prev_signal} → {action} ({confidence:.0%})",
                    "trade",
                )
                if stock.auto_execute and confidence >= stock.min_confidence and self.broker:
                    self._execute_trade(stock, action, price, atr)
            else:
                self.log.emit(
                    f"{stock.ticker}: {action} ({confidence:.0%}) @ ${price:.2f} — holding signal",
                    "info",
                )

        except Exception as e:
            self.log.emit(f"{stock.ticker} check failed: {e}", "error")

    def _execute_trade(self, stock, action, price, atr):
        """Execute buy or sell. Checks BP before buying, dynamic qty for sells."""
        # Market hours check — only execute during regular trading
        try:
            from core.market_hours import is_market_open
            if not is_market_open():
                self.log.emit(f"{stock.ticker}: market closed — skipping execution", "system")
                return
        except Exception:
            pass

        size = self.rm.position_size(price, atr)
        if size <= 0:
            self.log.emit(f"{stock.ticker}: position size 0 — skipping", "warn")
            return

        side = "buy" if action == "BUY" else "sell"

        # For buys: check if already holding this stock (prevent double-buy)
        if side == "buy" and self.broker:
            try:
                positions = self.broker.get_positions()
                if isinstance(positions, list):
                    for p in positions:
                        if p.get("symbol", "").upper() == stock.ticker.upper():
                            held = int(float(p.get("qty", 0)))
                            if held > 0:
                                self.log.emit(
                                    f"{stock.ticker}: already holding {held} shares — skipping buy",
                                    "info")
                                return
            except Exception:
                pass

        # For buys: check buying power first
        if side == "buy" and self.broker:
            try:
                acct = self.broker.get_account()
                bp = float(acct.get("buying_power", 0))
                cost = size * price
                if cost > bp:
                    # Reduce to what we can afford (max 20% of BP per trade)
                    affordable = int(min(bp * 0.20, bp) / price)
                    if affordable <= 0:
                        self.log.emit(
                            f"{stock.ticker}: insufficient BP (${bp:,.0f}) for {size} shares (${cost:,.0f}) — skipping",
                            "warn")
                        return
                    self.log.emit(
                        f"{stock.ticker}: reduced from {size} to {affordable} shares (BP=${bp:,.0f})",
                        "info")
                    size = affordable
            except Exception:
                pass

        # For sells: determine qty based on confidence
        if side == "sell" and self.broker:
            try:
                positions = self.broker.get_positions()
                held = 0
                if isinstance(positions, list):
                    for p in positions:
                        if p.get("symbol", "").upper() == stock.ticker.upper():
                            held = int(float(p.get("qty", 0)))
                            break
                if held > 0:
                    conf = stock.last_confidence
                    if conf > 0.7:
                        size = held  # Strong sell = sell all
                    elif conf > 0.5:
                        size = max(1, int(held * 0.5))  # Moderate = sell half
                    else:
                        size = max(1, int(held * 0.25))  # Weak = sell quarter
                else:
                    self.log.emit(f"{stock.ticker}: no position to sell", "info")
                    return
            except Exception:
                pass

        sl = self.rm.stop_loss(price, atr, side)
        tp = self.rm.take_profit(price, atr, side)

        if side == "sell":
            result = self.broker.close_position(stock.ticker, qty=size)
        else:
            result = self.broker.place_order(stock.ticker, size, side,
                                             stop_loss=sl, take_profit=tp)

        if "error" in result:
            self.log.emit(f"Auto-trade {side.upper()} {stock.ticker} x{size} FAILED: {result['error']}", "error")
            log_trade_execution(stock.ticker, side, size, "market", "failed", error=result["error"])
        else:
            oid = result.get("id", "?")
            self.log.emit(f"Auto-trade {side.upper()} {stock.ticker} x{size} — order {oid}", "trade")
            self.trade_executed.emit(stock.ticker, side, size, oid)
            log_trade_execution(stock.ticker, side, size, "market", oid)
