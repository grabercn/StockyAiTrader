# -*- coding: utf-8 -*-
"""
Market Hours — shared module for market session awareness.

Used by: agent engine, auto-trader, scanner, dashboard.
All times US/Eastern.

Sessions:
    CLOSED      (8:00 PM - 4:00 AM)   — No data, fully frozen
    PRE_MARKET  (4:00 AM - 9:30 AM)   — Extended hours, thin volume
    OPEN_AVOID  (9:30 AM - 9:45 AM)   — Opening whipsaw zone
    REGULAR     (9:45 AM - 3:45 PM)   — Prime trading window
    CLOSE_AVOID (3:45 PM - 4:00 PM)   — Closing volatility
    AFTER_HOURS (4:00 PM - 8:00 PM)   — Extended hours, wide spreads
    WEEKEND     (Sat-Sun)             — Fully closed

Usage:
    from core.market_hours import get_session, is_market_open, seconds_to_open

    session = get_session()
    if session.can_trade:
        execute_trade()
    else:
        print(session.note)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class MarketSession:
    """Current market session state."""
    name: str           # CLOSED, PRE_MARKET, OPEN_AVOID, REGULAR, CLOSE_AVOID, AFTER_HOURS, WEEKEND
    can_trade: bool     # True only during REGULAR hours
    can_scan: bool      # True when data is changing (PRE, REGULAR, AFTER)
    wait_seconds: int   # Seconds until next tradeable session (0 if can_trade)
    note: str           # Human-readable description


def get_session():
    """
    Get the current market session.

    Returns:
        MarketSession with trading permissions and wait time.
    """
    try:
        import pytz
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)
    except ImportError:
        return MarketSession("UNKNOWN", True, True, 0, "Cannot determine timezone")
    except Exception:
        return MarketSession("UNKNOWN", True, True, 0, "Market hours check failed")

    # Weekend
    if now.weekday() >= 5:
        days_to_mon = 7 - now.weekday()
        next_open = now.replace(hour=9, minute=45, second=0, microsecond=0) + timedelta(days=days_to_mon)
        wait = int((next_open - now).total_seconds())
        return MarketSession("WEEKEND", False, False, wait, "Weekend — market opens Monday")

    t = now.hour * 60 + now.minute  # Minutes since midnight

    if t < 240:  # Before 4:00 AM
        wait = (240 - t) * 60
        return MarketSession("CLOSED", False, False, wait, "Market closed — pre-market at 4:00 AM ET")

    if t < 570:  # 4:00 AM - 9:30 AM
        return MarketSession("PRE_MARKET", False, True, 0,
                           "Pre-market — data updating, no trades (thin volume)")

    if t < 585:  # 9:30 AM - 9:45 AM
        wait = (585 - t) * 60
        return MarketSession("OPEN_AVOID", False, True, wait,
                           f"Opening whipsaw — trading in {wait // 60}m")

    if t < 945:  # 9:45 AM - 3:45 PM
        return MarketSession("REGULAR", True, True, 0,
                           "Market open — prime trading window")

    if t < 960:  # 3:45 PM - 4:00 PM
        wait = (960 - t) * 60
        return MarketSession("CLOSE_AVOID", False, True, wait,
                           "Market closing — paused (end-of-day volatility)")

    if t < 1200:  # 4:00 PM - 8:00 PM
        return MarketSession("AFTER_HOURS", False, True, 0,
                           "After hours — data updating, no trades (wide spreads)")

    # After 8:00 PM
    if now.weekday() == 4:  # Friday
        next_open = now.replace(hour=9, minute=45, second=0, microsecond=0) + timedelta(days=3)
    else:
        next_open = now.replace(hour=9, minute=45, second=0, microsecond=0) + timedelta(days=1)
    wait = int((next_open - now).total_seconds())
    return MarketSession("CLOSED", False, False, wait, "Market closed — opens tomorrow")


def is_market_open():
    """Quick check: can we execute trades right now?"""
    return get_session().can_trade


def is_data_live():
    """Quick check: is market data actively changing?"""
    return get_session().can_scan


def seconds_to_open():
    """Seconds until next tradeable session. 0 if already open."""
    return get_session().wait_seconds


def get_session_label():
    """Short label for UI display (e.g., 'OPEN', 'PRE-MKT', 'CLOSED')."""
    s = get_session()
    labels = {
        "REGULAR": "OPEN",
        "PRE_MARKET": "PRE-MKT",
        "AFTER_HOURS": "AFTER-HRS",
        "OPEN_AVOID": "OPENING",
        "CLOSE_AVOID": "CLOSING",
        "CLOSED": "CLOSED",
        "WEEKEND": "WEEKEND",
        "UNKNOWN": "???",
    }
    return labels.get(s.name, s.name)
