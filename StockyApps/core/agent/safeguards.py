# -*- coding: utf-8 -*-
"""
Agent Safeguards — protective filters and risk controls.

1. Earnings Avoidance: skip buying stocks within 3 days of earnings
2. Manual Stop-Loss Monitor: sell positions that hit stop level (for non-bracket orders)
3. Trailing Stop: move stop to breakeven at +2x ATR, then to +1x ATR at +3x ATR
4. Sector Diversification: cap holdings at 2 per sector
5. Addon Signal Confirmation: boost/reduce confidence based on sentiment + insider data
"""

import importlib
import pandas as pd
from datetime import datetime


# ── Sector mapping for top stocks ──────────────────────────────────────
SECTOR_MAP = {
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AMD": "Tech", "INTC": "Tech",
    "GOOGL": "Tech", "GOOG": "Tech", "META": "Tech", "CRM": "Tech", "ORCL": "Tech",
    "AVGO": "Tech", "QCOM": "Tech", "MU": "Tech", "MRVL": "Tech", "AMAT": "Tech",
    "ADBE": "Tech", "NOW": "Tech", "UBER": "Tech", "SHOP": "Tech", "SNOW": "Tech",
    "PLTR": "Tech", "CRWD": "Tech", "PANW": "Tech", "NET": "Tech", "DDOG": "Tech",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "NFLX": "Consumer", "DIS": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "MCD": "Consumer", "WMT": "Consumer",
    "COST": "Consumer", "TGT": "Consumer", "HD": "Consumer", "LOW": "Consumer",
    # Finance
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance",
    "V": "Finance", "MA": "Finance", "AXP": "Finance", "BRK.B": "Finance",
    "C": "Finance", "WFC": "Finance", "SOFI": "Finance", "HOOD": "Finance",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "LLY": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "OXY": "Energy", "SLB": "Energy",
    "COP": "Energy", "EOG": "Energy", "MPC": "Energy",
    # Industrial
    "BA": "Industrial", "CAT": "Industrial", "GE": "Industrial", "HON": "Industrial",
    "UPS": "Industrial", "FDX": "Industrial", "DE": "Industrial", "RTX": "Industrial",
    # Telecom
    "T": "Telecom", "VZ": "Telecom", "TMUS": "Telecom", "NOK": "Telecom",
    # EV / Clean Energy
    "RIVN": "EV", "LCID": "EV", "NIO": "EV", "PLUG": "EV", "FCEL": "EV",
    "ENPH": "EV", "SEDG": "EV", "RUN": "EV",
    # Meme / Speculative
    "GME": "Meme", "AMC": "Meme", "BBBY": "Meme", "MARA": "Crypto",
    "COIN": "Crypto", "RIOT": "Crypto",
    # Social
    "SNAP": "Social", "PINS": "Social", "RDDT": "Social",
}


def get_sector(ticker):
    """Get sector for a ticker, or 'Other' if unknown."""
    return SECTOR_MAP.get(ticker.upper(), "Other")


def check_earnings_proximity(ticker, max_days=3):
    """
    Check if a stock has earnings within max_days.

    Returns:
        (skip: bool, days: float or None, reason: str)
    """
    try:
        from addons import get_all_addons
        for addon in get_all_addons():
            if addon.module_name == "finnhub_calendar" and addon.available and addon.enabled:
                mod = importlib.import_module("addons.finnhub_calendar")
                feats = mod.get_features(ticker, pd.DataFrame())
                if feats and "days_to_earnings" in feats:
                    days = feats["days_to_earnings"]
                    if days is not None and 0 <= days <= max_days:
                        return True, days, f"earnings in {days:.0f} days"
                    return False, days, ""
                break
    except Exception:
        pass
    return False, None, ""


def check_stop_loss_hits(held_map, agent_stocks, log_fn=None):
    """
    Check if any held positions have hit their stop-loss level.
    For positions without bracket orders (PDT fallback), this is the
    only protection against large drawdowns.

    Returns:
        list of tickers that need emergency sell
    """
    to_sell = []
    for sym, pos in held_map.items():
        stock_info = agent_stocks.get(sym, {})
        stop = stock_info.get("stop_loss")
        if not stop or stop <= 0:
            continue

        current_price = float(pos.get("current_price", 0))
        if current_price <= 0:
            continue

        # Check if price is at or below stop level
        if current_price <= stop:
            entry = float(pos.get("avg_entry_price", 0))
            loss_pct = ((current_price - entry) / entry * 100) if entry > 0 else 0
            if log_fn:
                log_fn(
                    f"  STOP HIT: {sym} @ ${current_price:.2f} <= SL ${stop:.2f} "
                    f"(entry ${entry:.2f}, {loss_pct:+.1f}%)", "warn")
            to_sell.append(sym)
    return to_sell


def update_trailing_stops(held_map, agent_stocks, log_fn=None):
    """
    Update stop-loss levels for positions that have moved favorably.

    Rules:
    - Price up 2x ATR from entry → move stop to breakeven
    - Price up 3x ATR from entry → move stop to entry + 1x ATR
    - Price up 4x ATR from entry → move stop to entry + 2x ATR

    Only moves stops UP, never down.
    """
    updates = 0
    for sym, pos in held_map.items():
        stock_info = agent_stocks.get(sym, {})
        entry = stock_info.get("entry_price", 0)
        current_stop = stock_info.get("stop_loss", 0)
        current_price = float(pos.get("current_price", 0))

        if entry <= 0 or current_price <= 0 or current_stop <= 0:
            continue

        # Estimate ATR from the stop distance at entry
        original_sl_dist = entry - current_stop
        if original_sl_dist <= 0:
            continue
        atr_est = original_sl_dist / 0.9  # Reverse the 0.9x stop mult

        gain = current_price - entry
        gain_atr = gain / atr_est if atr_est > 0 else 0

        new_stop = current_stop  # Default: don't change

        if gain_atr >= 4:
            new_stop = entry + (2 * atr_est)  # Lock in 2x ATR profit
        elif gain_atr >= 3:
            new_stop = entry + (1 * atr_est)  # Lock in 1x ATR profit
        elif gain_atr >= 2:
            new_stop = entry  # Move to breakeven

        # Only move stop UP, never down
        if new_stop > current_stop:
            agent_stocks[sym]["stop_loss"] = round(new_stop, 2)
            updates += 1
            if log_fn:
                log_fn(
                    f"  Trailing stop: {sym} SL ${current_stop:.2f} -> ${new_stop:.2f} "
                    f"(+{gain_atr:.1f}x ATR, locking profit)", "agent")

    return updates


def check_sector_limit(ticker, held_map, max_per_sector=2):
    """
    Check if buying this ticker would exceed the sector concentration limit.

    Returns:
        (allowed: bool, sector: str, count: int, reason: str)
    """
    sector = get_sector(ticker)
    if sector == "Other":
        return True, sector, 0, ""

    # Count how many stocks we already hold in this sector
    count = 0
    for sym in held_map:
        if get_sector(sym) == sector:
            count += 1

    if count >= max_per_sector:
        return False, sector, count, f"already holding {count} {sector} stocks (max {max_per_sector})"
    return True, sector, count, ""


def get_addon_sentiment(ticker):
    """
    Get sentiment signals from StockTwits + SEC Insider addons.

    Returns:
        (sentiment_score: float, detail: str)
        Score: -1.0 (very bearish) to +1.0 (very bullish), 0.0 = neutral/unavailable
    """
    score = 0.0
    details = []
    empty = pd.DataFrame()

    # StockTwits bull ratio
    try:
        from addons import get_all_addons
        for addon in get_all_addons():
            if addon.module_name == "stocktwits" and addon.available and addon.enabled:
                mod = importlib.import_module("addons.stocktwits")
                feats = mod.get_features(ticker, empty)
                if feats:
                    bull = feats.get("stocktwits_bull_ratio", 0.5)
                    vol = feats.get("stocktwits_volume", 0)
                    if vol > 0:
                        # 0.5 = neutral, >0.7 = bullish, <0.3 = bearish
                        sentiment = (bull - 0.5) * 2  # Normalize to -1 to +1
                        score += sentiment * 0.5  # Weight 50%
                        details.append(f"StockTwits {bull:.0%} bull")
                break
    except Exception:
        pass

    # SEC Insider trades
    try:
        from addons import get_all_addons
        for addon in get_all_addons():
            if addon.module_name == "insider_trades" and addon.available and addon.enabled:
                mod = importlib.import_module("addons.insider_trades")
                feats = mod.get_features(ticker, empty)
                if feats:
                    net = feats.get("insider_net_signal", 0)
                    # net: positive = more buys, negative = more sells
                    if net != 0:
                        score += max(-0.5, min(0.5, net * 0.25))  # Weight 25%, cap ±0.5
                        details.append(f"Insider net={net:+.0f}")
                break
    except Exception:
        pass

    detail = ", ".join(details) if details else "no sentiment data"
    return round(max(-1.0, min(1.0, score)), 2), detail
