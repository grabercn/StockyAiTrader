"""
Signal passing between DayTrader and StockExecuter.

DayTrader writes a signal to signal.json when it has a recommendation.
StockExecuter polls that file and executes trades based on it.

This replaces the old MQTT approach — simpler, no external broker dependency,
and signals stay local (not broadcast to a public server).
"""

import os
import json
from datetime import datetime

_SIGNAL_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "signal.json")


def write_signal(ticker, action, confidence, price, position_size,
                 stop_loss, take_profit, atr):
    """
    Write a trading signal for StockExecuter to consume.

    Called by DayTrader after each prediction cycle.
    """
    signal = {
        "ticker": ticker,
        "action": action,              # "BUY", "SELL", or "HOLD"
        "confidence": float(confidence),
        "price": float(price),
        "position_size": int(position_size),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "atr": float(atr),
        "timestamp": datetime.now().isoformat(),
    }
    with open(_SIGNAL_FILE, "w") as f:
        json.dump(signal, f, indent=2)

    return signal


def read_signal():
    """
    Read the latest signal from disk.

    Returns:
        dict with signal data, or None if no signal file exists
    """
    if not os.path.exists(_SIGNAL_FILE):
        return None
    try:
        with open(_SIGNAL_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
