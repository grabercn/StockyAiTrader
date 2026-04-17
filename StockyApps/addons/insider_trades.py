"""
Addon: SEC Insider Trading Tracker

Monitors insider buying/selling from SEC Form 4 filings via OpenInsider.

Why it matters:
    Insider buying clusters (3+ insiders buying in a week) are one of the
    most statistically significant bullish signals. Studies show insider
    buying outperforms by 7-10% annually. CEOs and CFOs know more about
    their company than any model ever will.

Setup:
    No dependencies beyond `requests` and `pandas` (already installed).
    No API key needed — scrapes public SEC data.
"""

import numpy as np
import requests
from datetime import datetime, timedelta
from io import StringIO

ADDON_NAME = "SEC Insider Trades"
ADDON_DESCRIPTION = "Insider buying/selling from SEC filings (free, no key)"
ADDON_FEATURES = ["insider_buy_count", "insider_sell_count", "insider_net_signal"]
DEPENDENCIES = []  # requests + pandas are core deps
REQUIRES_API_KEY = False
API_KEY_NAME = ""

_cache = {}
_cache_ttl = 3600  # 1 hour


def check_available():
    return True, "Ready"


def get_features(ticker, data):
    """
    Fetch recent insider trades for a ticker.

    Returns:
        insider_buy_count:  Number of insider buys in last 30 days
        insider_sell_count: Number of insider sells in last 30 days
        insider_net_signal: +1 (net buying), -1 (net selling), 0 (neutral/unknown)
    """
    import time
    now = time.time()

    if ticker in _cache and (now - _cache[ticker].get("_time", 0)) < _cache_ttl:
        cached = _cache[ticker]
        return {k: v for k, v in cached.items() if k != "_time"}

    try:
        # Use Finnhub if available (cleaner API), otherwise fallback
        result = _fetch_via_sec_edgar(ticker)
        _cache[ticker] = {**result, "_time": now}
        return result

    except Exception as e:
        print(f"Insider trades addon error: {e}")
        return _default()


def _fetch_via_sec_edgar(ticker):
    """Fetch insider transactions from SEC EDGAR full-text search."""
    try:
        # SEC EDGAR XBRL API for insider transactions
        headers = {"User-Agent": "StockyAiTrader research@example.com"}
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=%22{ticker}%22&forms=4&dateRange=custom"
            f"&startdt={(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}"
            f"&enddt={datetime.now().strftime('%Y-%m-%d')}"
        )

        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            total_hits = data.get("hits", {}).get("total", {}).get("value", 0)

            # Rough heuristic: more filings = more activity
            # We can't easily distinguish buy vs sell from search alone,
            # so we use the simpler OpenInsider approach
            return _fetch_via_openinsider(ticker)

        return _default()

    except Exception:
        return _fetch_via_openinsider(ticker)


def _fetch_via_openinsider(ticker):
    """Scrape OpenInsider for insider buy/sell counts."""
    try:
        import pandas as pd

        url = f"http://openinsider.com/screener?s={ticker}&o=&pl=&ph=&ll=&lh=&fd=30&fdr=&td=0&tdr=&feession=&cession=&sig=&session=buy"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "StockyAiTrader/2.0"})

        if resp.status_code != 200:
            return _default()

        # Count "Purchase" vs "Sale" in the response
        text = resp.text.lower()
        buys = text.count(" - purchase")
        sells = text.count(" - sale")

        if buys + sells == 0:
            return _default()

        net = 1.0 if buys > sells else (-1.0 if sells > buys else 0.0)

        return {
            "insider_buy_count": float(buys),
            "insider_sell_count": float(sells),
            "insider_net_signal": net,
        }

    except Exception:
        return _default()


def _default():
    return {
        "insider_buy_count": 0.0,
        "insider_sell_count": 0.0,
        "insider_net_signal": 0.0,
    }
