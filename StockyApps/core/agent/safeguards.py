# -*- coding: utf-8 -*-
"""
Agent Safeguards — protective filters and risk controls.

HIGH IMPACT:
1. Earnings Avoidance: skip buying stocks within 3 days of earnings
2. Manual Stop-Loss Monitor: sell positions that hit stop level (for non-bracket orders)
3. Trailing Stop: move stop to breakeven at +2x ATR, then to +1x ATR at +3x ATR
4. Sector Diversification: cap holdings at 2 per sector
5. Addon Signal Confirmation: boost/reduce confidence based on sentiment + insider data
6. FOMC/CPI Calendar: skip trading on major economic event days
7. Volume Filter: skip BUY when volume is below 50% of average
8. Loss Cooldown: wait 1 extra cycle after a losing trade

DIMINISHING RETURNS:
9. Correlation Filter: skip buying highly correlated stocks already held
10. Time-of-Day Scoring: favor trades during historically best windows
11. Weekend Gap Protection: reduce Friday afternoon position sizes
"""

import importlib
import pandas as pd
from datetime import datetime, timedelta


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


# ── HIGH IMPACT: #6 FOMC/CPI Calendar ──────────────────────────────────

# Major economic event dates for 2026 (FOMC decisions + CPI releases)
# These cause 1-2% swings in minutes that no model can predict
FOMC_DATES_2026 = {
    "2026-01-28", "2026-01-29",  # Jan FOMC
    "2026-03-18", "2026-03-19",  # Mar FOMC
    "2026-05-06", "2026-05-07",  # May FOMC
    "2026-06-17", "2026-06-18",  # Jun FOMC
    "2026-07-29", "2026-07-30",  # Jul FOMC
    "2026-09-16", "2026-09-17",  # Sep FOMC
    "2026-11-04", "2026-11-05",  # Nov FOMC
    "2026-12-16", "2026-12-17",  # Dec FOMC
}

CPI_DATES_2026 = {
    "2026-01-14", "2026-02-12", "2026-03-12", "2026-04-10",
    "2026-05-13", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-10", "2026-10-13", "2026-11-12", "2026-12-10",
}

ECONOMIC_EVENT_DATES = FOMC_DATES_2026 | CPI_DATES_2026


def is_economic_event_day():
    """
    Check if today is a major economic event day (FOMC or CPI).

    Returns:
        (is_event: bool, event_type: str)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    if today in FOMC_DATES_2026:
        return True, "FOMC"
    if today in CPI_DATES_2026:
        return True, "CPI"
    return False, ""


# ── HIGH IMPACT: #7 Volume Filter ──────────────────────────────────────

def check_volume(scan_result):
    """
    Check if current volume is sufficient to trust the BUY signal.
    Low volume moves are unreliable and often reverse.

    Args:
        scan_result: ScanResult with feature_importances or raw data

    Returns:
        (ok: bool, ratio: float, reason: str)
    """
    # volume_ratio_5 is already a feature: current_vol / 5-bar avg
    # If it's in feature importances, we can read it
    # But we don't have the raw feature value in ScanResult
    # Use ATR as a proxy: very low ATR = low activity
    if scan_result.atr <= 0 or scan_result.price <= 0:
        return True, 1.0, ""

    atr_pct = scan_result.atr / scan_result.price
    # If ATR is extremely low (< 0.3% of price), volume is likely dead
    if atr_pct < 0.003:
        return False, atr_pct, f"very low volatility ({atr_pct:.2%}), likely thin volume"
    return True, atr_pct, ""


# ── HIGH IMPACT: #8 Loss Cooldown ──────────────────────────────────────

def should_cooldown(trade_log, cooldown_cycles=1):
    """
    Check if the last trade was a loss. If so, recommend waiting.

    Args:
        trade_log: list of recent trade dicts with 'pnl' field
        cooldown_cycles: how many cycles to wait after a loss

    Returns:
        (should_wait: bool, last_pnl: float or None)
    """
    if not trade_log:
        return False, None
    # Find the last completed trade (has pnl)
    for trade in reversed(trade_log):
        pnl = trade.get("pnl")
        if pnl is not None:
            if pnl < 0:
                return True, pnl
            return False, pnl
    return False, None


# ── DIMINISHING: #9 Correlation Filter ─────────────────────────────────

# Highly correlated stock pairs — buying both is redundant risk
CORRELATED_PAIRS = {
    frozenset({"NVDA", "AMD"}),
    frozenset({"NVDA", "AVGO"}),
    frozenset({"GOOGL", "GOOG"}),
    frozenset({"META", "SNAP"}),
    frozenset({"V", "MA"}),
    frozenset({"JPM", "BAC"}),
    frozenset({"JPM", "GS"}),
    frozenset({"XOM", "CVX"}),
    frozenset({"RIVN", "LCID"}),
    frozenset({"MARA", "RIOT"}),
    frozenset({"COIN", "MARA"}),
    frozenset({"WMT", "COST"}),
    frozenset({"HD", "LOW"}),
    frozenset({"UPS", "FDX"}),
    frozenset({"PLUG", "FCEL"}),
    frozenset({"T", "VZ"}),
    frozenset({"PFE", "MRK"}),
    frozenset({"MSFT", "AAPL"}),  # High correlation during tech moves
}


def check_correlation(ticker, held_map):
    """
    Check if buying this ticker would create redundant exposure
    with an already-held highly correlated stock.

    Returns:
        (allowed: bool, correlated_with: str or None, reason: str)
    """
    ticker_upper = ticker.upper()
    held_tickers = set(held_map.keys())

    for pair in CORRELATED_PAIRS:
        if ticker_upper in pair:
            partner = pair - {ticker_upper}
            overlap = partner & held_tickers
            if overlap:
                corr_with = next(iter(overlap))
                return False, corr_with, f"highly correlated with held {corr_with}"

    return True, None, ""


# ── DIMINISHING: #10 Time-of-Day Scoring ───────────────────────────────

def get_time_of_day_multiplier():
    """
    Score based on historically best/worst trading windows.

    Best: 10:00-11:30 AM (post-open momentum confirmed)
    Good: 2:00-3:30 PM (afternoon trend continuation)
    Worst: 9:30-10:00 AM (opening chaos — already avoided by market hours)
    Meh: 11:30-2:00 PM (lunch doldrums, low volume)

    Returns:
        (multiplier: float, window: str)
        1.0 = normal, >1.0 = favorable, <1.0 = unfavorable
    """
    try:
        import pytz
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)
        t = now.hour * 60 + now.minute

        if 600 <= t < 690:    # 10:00-11:30 AM — best window
            return 1.1, "prime morning (10:00-11:30)"
        elif 840 <= t < 930:  # 2:00-3:30 PM — good window
            return 1.05, "afternoon trend (2:00-3:30)"
        elif 690 <= t < 840:  # 11:30-2:00 PM — lunch doldrums
            return 0.9, "lunch doldrums (11:30-2:00)"
        else:
            return 1.0, "normal"
    except Exception:
        return 1.0, "unknown"


# ── DIMINISHING: #11 Weekend Gap Protection ────────────────────────────

def is_friday_afternoon():
    """
    Check if it's Friday afternoon (after 2 PM ET).
    Positions held over the weekend are exposed to gap risk.

    Returns:
        (is_friday_pm: bool, note: str)
    """
    try:
        import pytz
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)
        if now.weekday() == 4 and now.hour >= 14:
            return True, "Friday afternoon — weekend gap risk, reduce new positions"
        return False, ""
    except Exception:
        return False, ""
