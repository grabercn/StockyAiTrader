"""
Intelligent Auto-Trader — adaptive trading engine that dynamically adjusts strategy.

OVERVIEW:
    Unlike the basic auto-trader which uses fixed intervals, this engine
    continuously adapts TWO key parameters based on live market conditions:

    1. CHECK INTERVAL — how often to re-evaluate a stock
    2. DATA PERIOD — how much historical data to train on

HOW IT WORKS:

    The engine uses a feedback loop:

    ┌─────────────────────────────────────────┐
    │  Fetch latest data for stock            │
    │  ↓                                      │
    │  Calculate market conditions:           │
    │    - ATR (volatility)                   │
    │    - Price momentum (5-bar EMA slope)   │
    │    - Volume ratio (vs 20-bar average)   │
    │    - Signal stability (how often it     │
    │      flipped BUY/SELL recently)         │
    │  ↓                                      │
    │  Compute "urgency score" (0-1):         │
    │    high volatility  → higher urgency    │
    │    strong momentum  → higher urgency    │
    │    volume spike     → higher urgency    │
    │    unstable signal  → higher urgency    │
    │  ↓                                      │
    │  Adjust interval:                       │
    │    urgency > 0.7 → check every 1 min    │
    │    urgency > 0.4 → check every 5 min    │
    │    urgency < 0.2 → check every 15 min   │
    │  ↓                                      │
    │  Adjust data period:                    │
    │    volatile → shorter period (2d)       │
    │    calm     → longer period (5d)        │
    │  ↓                                      │
    │  Run LightGBM prediction                │
    │  ↓                                      │
    │  Apply aggressivity profile to decide   │
    │  whether to execute the trade           │
    └─────────────────────────────────────────┘

AGGRESSIVITY PROFILES:
    Each profile adjusts confidence thresholds, position sizing, and how
    eagerly the engine acts on signals.

    Chill:        High confidence required (70%), half position size, wide stops
    Default:      Standard thresholds (50%), normal sizing, normal stops
    Aggressive:   Lower threshold (35%), 1.5x sizing, tighter stops
    YOLO:         Minimal threshold (25%), 2x sizing, tight stops, more trades

    These are multipliers on top of the RiskManager's base calculations.
"""

import time
import math
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional
from PyQt5.QtCore import QThread, pyqtSignal

from .data import fetch_intraday, get_all_features
from .model import train_lgbm, predict_lgbm
from .risk import RiskManager
from .labeling import LABEL_NAMES
from .logger import log_decision, log_event


# ─── Aggressivity Profiles ──────────────────────────────────────────────────

AGGRESSIVITY_PROFILES = {
    "Chill": {
        "description": "Patient trading. High confidence required, smaller positions, wider stops.",
        "min_confidence": 0.70,
        "size_multiplier": 0.5,
        "atr_stop_mult": 2.0,
        "atr_profit_mult": 3.5,
        "urgency_bias": -0.15,
        "max_trades_per_day": 3,
        "use_llm": False,           # Not needed — trades rarely
        "min_hardware": "Minimal",  # Works on any hardware
    },
    "Default": {
        "description": "Balanced approach. Standard confidence, normal sizing, standard risk.",
        "min_confidence": 0.50,
        "size_multiplier": 1.0,
        "atr_stop_mult": 1.5,
        "atr_profit_mult": 2.5,
        "urgency_bias": 0.0,
        "max_trades_per_day": 8,
        "use_llm": True,            # LLM reasoning for trade decisions
        "min_hardware": "Balanced",  # Needs FinBERT + LightGBM
    },
    "Aggressive": {
        "description": "Active trading. Lower thresholds, larger positions, tighter stops.",
        "min_confidence": 0.35,
        "size_multiplier": 1.5,
        "atr_stop_mult": 1.2,
        "atr_profit_mult": 2.0,
        "urgency_bias": 0.10,
        "max_trades_per_day": 15,
        "use_llm": True,
        "min_hardware": "Balanced",
    },
    "YOLO": {
        "description": "Maximum aggression. Minimal thresholds, 2x sizing, tight stops, high frequency.",
        "min_confidence": 0.25,
        "size_multiplier": 2.0,
        "atr_stop_mult": 1.0,
        "atr_profit_mult": 1.5,
        "urgency_bias": 0.25,
        "max_trades_per_day": 30,
        "use_llm": True,
        "min_hardware": "Max",      # Needs all addons for best signals at this risk level
    },
}


# ─── Hardware Profile Compatibility ──────────────────────────────────────────
# Maps which aggressivity profiles are compatible with which hardware profiles.
# If user picks an incompatible combo, we warn them.

_HARDWARE_RANK = {"Minimal": 0, "Light": 1, "Balanced": 2, "Max": 3}


def check_profile_compatibility(aggressivity_name, hardware_name):
    """
    Check if aggressivity + hardware profiles are compatible.

    Returns:
        (compatible: bool, warnings: list[str])
    """
    profile = AGGRESSIVITY_PROFILES.get(aggressivity_name, AGGRESSIVITY_PROFILES["Default"])
    min_hw = profile.get("min_hardware", "Minimal")

    hw_rank = _HARDWARE_RANK.get(hardware_name, 2)
    min_rank = _HARDWARE_RANK.get(min_hw, 0)

    warnings = []

    if hw_rank < min_rank:
        warnings.append(
            f"{aggressivity_name} aggressivity recommends '{min_hw}' hardware profile or higher. "
            f"You're on '{hardware_name}' — some features may be limited."
        )

    # Specific warnings
    if aggressivity_name == "YOLO" and hardware_name in ("Minimal", "Light"):
        warnings.append(
            "YOLO mode on Light/Minimal hardware means fewer addon signals. "
            "High-frequency trading without full data coverage increases risk."
        )

    if profile.get("use_llm") and hardware_name == "Minimal":
        warnings.append(
            "LLM reasoning is recommended for this mode but requires at least "
            "Balanced hardware profile. Trade reasoning will use templates instead."
        )

    return len(warnings) == 0, warnings


def get_aggressivity_names():
    return list(AGGRESSIVITY_PROFILES.keys())


def get_aggressivity(name):
    return AGGRESSIVITY_PROFILES.get(name, AGGRESSIVITY_PROFILES["Default"])


# ─── Market Condition Analysis ───────────────────────────────────────────────

def compute_urgency(data):
    """
    Compute an "urgency score" from 0 (calm, check less often) to 1 (volatile, check NOW).

    Based on four factors:
    - Volatility: ATR relative to price (high = urgent)
    - Momentum: slope of recent price movement (steep = urgent)
    - Volume: current vs average (spike = urgent)
    - Signal instability: price crossing EMAs frequently (choppy = urgent)
    """
    if data.empty or len(data) < 20:
        return 0.5  # Unknown → default

    closes = data["Close"].values
    price = closes[-1]

    # 1. Volatility factor (ATR / price)
    if "ATRr_14" in data.columns:
        atr = data["ATRr_14"].iloc[-1]
        volatility = min(1.0, (atr / price) / 0.03)  # 3% ATR = max urgency
    else:
        volatility = 0.5

    # 2. Momentum factor (absolute slope of last 5 bars)
    if len(closes) >= 5:
        slope = abs(closes[-1] - closes[-5]) / closes[-5]
        momentum = min(1.0, slope / 0.02)  # 2% move in 5 bars = max
    else:
        momentum = 0.5

    # 3. Volume factor (current vs 20-bar average)
    if "Volume" in data.columns and len(data) >= 20:
        avg_vol = data["Volume"].rolling(20).mean().iloc[-1]
        cur_vol = data["Volume"].iloc[-1]
        if avg_vol > 0:
            volume_ratio = min(1.0, (cur_vol / avg_vol - 1.0) / 2.0)  # 3x avg = max
            volume_ratio = max(0.0, volume_ratio)
        else:
            volume_ratio = 0.5
    else:
        volume_ratio = 0.5

    # 4. Signal instability (how many times price crossed EMA-9 in last 20 bars)
    if "EMA_9" in data.columns and len(data) >= 20:
        ema = data["EMA_9"].values[-20:]
        price_window = closes[-20:]
        crossings = sum(1 for i in range(1, len(ema))
                       if (price_window[i] > ema[i]) != (price_window[i-1] > ema[i-1]))
        instability = min(1.0, crossings / 8.0)  # 8+ crossings = max chop
    else:
        instability = 0.5

    # Weighted combination
    urgency = (
        volatility * 0.35 +
        momentum * 0.30 +
        volume_ratio * 0.20 +
        instability * 0.15
    )

    return max(0.0, min(1.0, urgency))


def adaptive_interval(urgency, profile_bias=0.0):
    """
    Map urgency score to a check interval in seconds.

    urgency 0.0 → 900s (15 min) — calm, no rush
    urgency 0.5 → 300s (5 min)  — normal
    urgency 1.0 → 60s  (1 min)  — volatile, check frequently

    profile_bias shifts urgency up (aggressive) or down (chill).
    """
    adjusted = max(0.0, min(1.0, urgency + profile_bias))

    if adjusted >= 0.75:
        return 60       # 1 minute
    elif adjusted >= 0.55:
        return 120      # 2 minutes
    elif adjusted >= 0.35:
        return 300      # 5 minutes
    elif adjusted >= 0.15:
        return 600      # 10 minutes
    else:
        return 900      # 15 minutes


def adaptive_period(urgency):
    """
    Map urgency to a training data period.

    Volatile markets → shorter period (recent data matters more)
    Calm markets → longer period (more training data = better model)
    """
    if urgency >= 0.6:
        return "2d"
    elif urgency >= 0.3:
        return "3d"
    else:
        return "5d"


# ─── Intelligent Trading Service ────────────────────────────────────────────

@dataclass
class IntelligentStock:
    """Per-stock state for intelligent monitoring."""
    ticker: str
    mode: str = "intelligent"   # "intelligent" or "manual"
    manual_period: str = "5d"   # Only used in manual mode
    manual_interval: int = 300  # Only used in manual mode

    # Intelligent state
    urgency: float = 0.5
    current_interval: int = 300
    current_period: str = "5d"
    aggressivity: str = "Default"

    # Runtime
    last_signal: str = "HOLD"
    last_confidence: float = 0.0
    last_price: float = 0.0
    last_check: Optional[str] = None
    next_check_seconds: int = 0
    check_count: int = 0
    trades_today: int = 0
    model: object = None
    features: list = field(default_factory=list)


class IntelligentTraderService(QThread):
    """
    Background thread with adaptive interval and data period.

    In INTELLIGENT mode:
        - Computes urgency from market conditions after each check
        - Adjusts check interval dynamically (1min - 15min)
        - Adjusts training data period (2d - 5d)
        - Applies aggressivity profile to trade decisions

    In MANUAL mode:
        - Uses fixed interval and period set by user

    Emits signals for UI updates.
    """

    stock_updated = pyqtSignal(str, str, float, float, int, float, str)
    # ticker, action, confidence, price, next_secs, urgency, period
    trade_executed = pyqtSignal(str, str, int, str)
    log = pyqtSignal(str, str)

    def __init__(self, broker=None, risk_manager=None):
        super().__init__()
        self.broker = broker
        self.rm = risk_manager or RiskManager()
        self._stocks: Dict[str, IntelligentStock] = {}
        self._running = True

    def add_stock(self, ticker, mode="intelligent", aggressivity="Default",
                  manual_period="5d", manual_interval=300):
        self._stocks[ticker] = IntelligentStock(
            ticker=ticker, mode=mode, aggressivity=aggressivity,
            manual_period=manual_period, manual_interval=manual_interval,
            next_check_seconds=5,  # Check almost immediately
        )
        self.log.emit(f"Monitoring {ticker} ({mode} mode, {aggressivity})", "info")

    def remove_stock(self, ticker):
        self._stocks.pop(ticker, None)
        self.log.emit(f"Stopped monitoring {ticker}", "info")

    def is_monitoring(self, ticker):
        return ticker in self._stocks

    def get_monitored(self):
        return dict(self._stocks)

    def stop(self):
        self._running = False

    def run(self):
        log_event("intelligent_trader", "Intelligent auto-trader started")
        self.log.emit("Intelligent auto-trader started", "system")

        while self._running:
            for ticker, stock in list(self._stocks.items()):
                stock.next_check_seconds -= 1

                if stock.next_check_seconds <= 0:
                    self._check_stock(stock)

                    # Set next interval based on mode
                    if stock.mode == "intelligent":
                        stock.next_check_seconds = stock.current_interval
                    else:
                        stock.next_check_seconds = stock.manual_interval

                # Emit update for UI
                self.stock_updated.emit(
                    ticker, stock.last_signal, stock.last_confidence,
                    stock.last_price, stock.next_check_seconds,
                    stock.urgency, stock.current_period,
                )

            time.sleep(1)

    def _check_stock(self, stock):
        try:
            profile = get_aggressivity(stock.aggressivity)
            stock.last_check = datetime.now().strftime("%H:%M:%S")
            stock.check_count += 1

            # Determine period
            if stock.mode == "intelligent":
                period = stock.current_period  # Will be updated after this check
            else:
                period = stock.manual_period

            # Fetch data
            data = fetch_intraday(stock.ticker, period=period, interval="5m")
            if data.empty or len(data) < 30:
                self.log.emit(f"{stock.ticker}: not enough data", "warn")
                return

            # In intelligent mode, compute urgency and adapt
            if stock.mode == "intelligent":
                stock.urgency = compute_urgency(data)
                stock.current_interval = adaptive_interval(stock.urgency, profile["urgency_bias"])
                stock.current_period = adaptive_period(stock.urgency)

                self.log.emit(
                    f"{stock.ticker}: urgency {stock.urgency:.2f} → "
                    f"interval {stock.current_interval}s, period {stock.current_period}",
                    "system",
                )

            # Train model (every 10 checks or first time)
            if stock.model is None or stock.check_count % 10 == 0:
                features = get_all_features("intraday")
                model, used = train_lgbm(data, features, stock.ticker)
                if model:
                    stock.model = model
                    stock.features = used

            if not stock.model:
                return

            # Predict
            actions, confs, probs = predict_lgbm(stock.model, data, stock.features)
            action = LABEL_NAMES[actions[-1]]
            confidence = float(confs[-1])
            price = float(data["Close"].iloc[-1])
            atr = float(data["ATRr_14"].iloc[-1]) if "ATRr_14" in data.columns else price * 0.01

            prev = stock.last_signal
            stock.last_signal = action
            stock.last_confidence = confidence
            stock.last_price = price

            log_decision(stock.ticker, action, confidence, price,
                        0, 0, 0, atr, [float(p) for p in probs[-1]],
                        reasoning=f"IntelligentTrader #{stock.check_count} urgency={stock.urgency:.2f}")

            # Should we trade?
            if action != prev and action != "HOLD":
                if confidence >= profile["min_confidence"]:
                    if stock.trades_today < profile["max_trades_per_day"]:
                        self.log.emit(
                            f"SIGNAL: {stock.ticker} {prev}→{action} ({confidence:.0%}) — executing",
                            "trade",
                        )
                        self._execute(stock, action, price, atr, profile)
                    else:
                        self.log.emit(f"{stock.ticker}: max daily trades reached ({profile['max_trades_per_day']})", "warn")
                else:
                    self.log.emit(
                        f"{stock.ticker}: {action} ({confidence:.0%}) below threshold ({profile['min_confidence']:.0%})",
                        "info",
                    )

        except Exception as e:
            self.log.emit(f"{stock.ticker} check error: {e}", "error")

    def _execute(self, stock, action, price, atr, profile):
        if not self.broker:
            self.log.emit(f"Cannot execute — broker not connected", "error")
            return

        # Position size adjusted by aggressivity
        base_size = self.rm.position_size(price, atr)
        size = max(1, int(base_size * profile["size_multiplier"]))

        # Stops adjusted by profile
        if action == "BUY":
            sl = price - atr * profile["atr_stop_mult"]
            tp = price + atr * profile["atr_profit_mult"]
            result = self.broker.place_order(stock.ticker, size, "buy", stop_loss=sl, take_profit=tp)
        else:
            result = self.broker.close_position(stock.ticker, qty=size)
            sl = tp = 0

        if "error" in result:
            self.log.emit(f"Trade failed: {action} {stock.ticker} x{size}: {result['error']}", "error")
        else:
            oid = result.get("id", "?")
            stock.trades_today += 1
            self.log.emit(f"Executed: {action} {stock.ticker} x{size} — order {oid}", "trade")
            self.trade_executed.emit(stock.ticker, "buy" if action == "BUY" else "sell", size, oid)
