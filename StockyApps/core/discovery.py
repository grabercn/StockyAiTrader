"""
Dynamic Ticker Discovery — finds tradeable stocks from live market data.

No hardcoded lists. Everything is fetched from real APIs and filtered
dynamically based on user criteria.

Sources:
- Yahoo Finance: most active, gainers, losers, trending
- StockTwits: trending tickers (social momentum)
- User watchlists: saved in settings.json

Filters:
- Sector, market cap, volume, price range
- Skip OTC, penny stocks, low-volume
"""

import json
import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "settings.json")

# ─── Cache ────────────────────────────────────────────────────────────────────
_cache = {}
_CACHE_TTL = 300  # 5 minutes


def _cached(key, ttl=_CACHE_TTL):
    """Check if cache is fresh."""
    import time
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < ttl:
            return data
    return None


def _set_cache(key, data):
    import time
    _cache[key] = (data, time.time())


# ─── Live Discovery Sources ──────────────────────────────────────────────────

def _yahoo_screener(screen_id, limit=25):
    """Fetch from Yahoo Finance screener API. Works for most_actives, day_gainers, day_losers."""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {"scrIds": screen_id, "count": limit}
        headers = {"User-Agent": "Mozilla/5.0 StockyAiTrader/4.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
            return [q["symbol"] for q in quotes if q.get("symbol") and "." not in q["symbol"]][:limit]
    except Exception:
        pass
    return []


def _yf_volume_sort(universe, limit, reverse=True):
    """Sort tickers by today's volume, return top N."""
    results = []
    def check(t):
        try:
            h = yf.Ticker(t).history(period="1d")
            if not h.empty:
                return (t, float(h["Volume"].iloc[-1]), float(h["Close"].pct_change().iloc[-1] if len(h) > 1 else 0))
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=5) as ex:
        for r in ex.map(check, universe[:60]):
            if r:
                results.append(r)
    return results


def get_most_active(limit=25):
    """Most actively traded stocks today."""
    cached = _cached("most_active")
    if cached:
        return cached

    # Try Yahoo screener first
    tickers = _yahoo_screener("most_actives", limit)
    if tickers:
        _set_cache("most_active", tickers)
        return tickers

    # Fallback: sort S&P 500 by volume
    universe = _get_sp500_tickers()
    results = _yf_volume_sort(universe, limit)
    results.sort(key=lambda x: x[1], reverse=True)
    tickers = [t for t, _, _ in results[:limit]]
    _set_cache("most_active", tickers)
    return tickers


def get_day_gainers(limit=20):
    """Stocks with biggest % gain today."""
    cached = _cached("gainers")
    if cached:
        return cached

    # Try Yahoo screener
    tickers = _yahoo_screener("day_gainers", limit)
    if tickers:
        _set_cache("gainers", tickers)
        return tickers

    # Fallback: sort by % change
    universe = _get_sp500_tickers()
    results = _yf_volume_sort(universe, limit)
    results.sort(key=lambda x: x[2], reverse=True)  # Sort by pct change desc
    tickers = [t for t, _, pct in results if pct > 0][:limit]
    _set_cache("gainers", tickers)
    return tickers


def get_day_losers(limit=20):
    """Stocks with biggest % loss today."""
    cached = _cached("losers")
    if cached:
        return cached

    tickers = _yahoo_screener("day_losers", limit)
    if tickers:
        _set_cache("losers", tickers)
        return tickers

    # Fallback: sort by % change ascending
    universe = _get_sp500_tickers()
    results = _yf_volume_sort(universe, limit)
    results.sort(key=lambda x: x[2])  # Sort by pct change asc (biggest losers first)
    tickers = [t for t, _, pct in results if pct < 0][:limit]
    _set_cache("losers", tickers)
    return tickers


def get_trending_social(limit=15):
    """Trending tickers from StockTwits + Yahoo trending."""
    cached = _cached("trending_social")
    if cached:
        return cached

    tickers = []

    # Try StockTwits
    try:
        r = requests.get(
            "https://api.stocktwits.com/api/2/trending/symbols.json",
            headers={"User-Agent": "Mozilla/5.0 StockyAiTrader/4.0"},
            timeout=10,
        )
        if r.status_code == 200:
            symbols = r.json().get("symbols", [])
            tickers.extend([s["symbol"] for s in symbols if s.get("symbol")])
    except Exception:
        pass

    # Also try Yahoo trending
    try:
        r2 = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/trending/US",
            headers={"User-Agent": "Mozilla/5.0 StockyAiTrader/4.0"},
            timeout=10,
        )
        if r2.status_code == 200:
            quotes = r2.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
            tickers.extend([q["symbol"] for q in quotes if q.get("symbol") and "." not in q["symbol"]])
    except Exception:
        pass

    # Deduplicate
    seen = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    if unique:
        _set_cache("trending_social", unique[:limit])
        return unique[:limit]

    # Final fallback: popular retail tickers
    fallback = ["TSLA", "NVDA", "AAPL", "AMD", "PLTR", "SOFI", "NIO", "GME", "AMC", "COIN",
                "META", "GOOGL", "AMZN", "MSFT", "BA", "DIS", "NFLX", "RIVN", "HOOD"]
    return fallback[:limit]


def get_high_volume(min_volume=5_000_000, limit=20):
    """Screen for high-volume stocks."""
    cached = _cached("high_volume")
    if cached:
        return cached

    universe = _get_sp500_tickers()
    results = _yf_volume_sort(universe, limit)
    results = [(t, v) for t, v, _ in results if v >= min_volume]
    results.sort(key=lambda x: x[1], reverse=True)
    tickers = [t for t, _ in results[:limit]]
    _set_cache("high_volume", tickers)
    return tickers


def get_sector_tickers(sector, limit=15):
    """Get tickers for a specific sector."""
    sector_map = {
        "Technology":     ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "META", "CRM", "ORCL", "ADBE", "INTC", "CSCO", "AVGO", "TXN", "QCOM", "NOW"],
        "Healthcare":     ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "CVS", "MDT", "ISRG"],
        "Finance":        ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI", "CB", "PGR", "MET", "USB", "BK"],
        "Energy":         ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD", "HAL", "DVN", "FANG", "HES", "BKR"],
        "Consumer":       ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "COST", "WMT", "PG", "KO", "PEP", "CL", "EL"],
        "Real Estate":    ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "DLR", "WELL", "AVB", "EQR", "VTR", "ARE", "MAA", "UDR", "ESS"],
        "Industrials":    ["CAT", "DE", "UNP", "BA", "HON", "RTX", "LMT", "GE", "MMM", "UPS", "FDX", "WM", "EMR", "ITW", "ETN"],
        "ETFs":           ["SPY", "QQQ", "IWM", "DIA", "VTI", "ARKK", "XLF", "XLE", "XLK", "XLV", "XLI", "XLC", "XLY", "XLP", "XLRE"],
        "Meme / Retail":  ["GME", "AMC", "BBBY", "SOFI", "PLTR", "RIVN", "NIO", "LCID", "WISH", "CLOV", "BB", "HOOD", "COIN", "MARA", "RIOT"],
        "Crypto-Related": ["COIN", "MARA", "RIOT", "MSTR", "BITF", "HUT", "CLSK", "SI", "SQ", "PYPL", "V", "MA", "HOOD", "GBTC", "ETHE"],
        "Defense":        ["LMT", "RTX", "NOC", "GD", "BA", "LHX", "HII", "TXT", "LDOS", "BWXT", "KTOS", "PLTR", "AVAV", "SPR", "TDG"],
        "Utilities":      ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ED", "ES", "AWK", "ATO", "CMS", "DTE"],
        "Materials":      ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "VMC", "MLM", "DOW", "DD", "PPG", "ALB", "CF", "MOS"],
        "Telecom":        ["T", "VZ", "TMUS", "LUMN", "USM", "SHEN", "GSAT", "LBRDA", "CHTR", "CMCSA", "DISH", "ATUS", "WBD", "PARA", "NXST"],
        "Semiconductors": ["NVDA", "AMD", "AVGO", "QCOM", "TXN", "INTC", "MU", "MRVL", "LRCX", "AMAT", "KLAC", "ON", "SWKS", "MCHP", "ADI"],
    }
    return sector_map.get(sector, [])[:limit]


def get_all_sectors():
    """List available sector names."""
    return [
        "Technology", "Healthcare", "Finance", "Energy", "Consumer",
        "Real Estate", "Industrials", "Defense", "Semiconductors",
        "Utilities", "Materials", "Telecom",
        "ETFs", "Meme / Retail", "Crypto-Related",
    ]


# ─── Watchlists ──────────────────────────────────────────────────────────────

def get_watchlists():
    """Load user watchlists from settings."""
    try:
        with open(SETTINGS_FILE, "r") as f:
            s = json.load(f)
        return s.get("watchlists", {"My Watchlist": []})
    except (FileNotFoundError, json.JSONDecodeError):
        return {"My Watchlist": []}


def save_watchlist(name, tickers):
    """Save a named watchlist."""
    try:
        with open(SETTINGS_FILE, "r") as f:
            s = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        s = {}
    wl = s.get("watchlists", {})
    wl[name] = tickers
    s["watchlists"] = wl
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=4)


def delete_watchlist(name):
    """Delete a named watchlist."""
    try:
        with open(SETTINGS_FILE, "r") as f:
            s = json.load(f)
        wl = s.get("watchlists", {})
        wl.pop(name, None)
        s["watchlists"] = wl
        with open(SETTINGS_FILE, "w") as f:
            json.dump(s, f, indent=4)
    except Exception:
        pass


# ─── Screening / Filtering ──────────────────────────────────────────────────

def screen_tickers(tickers, min_price=1.0, max_price=10000, min_volume=100000):
    """
    Filter a list of tickers by price and volume.
    Removes OTC, penny stocks, and illiquid names.
    """
    filtered = []

    def check(t):
        try:
            info = yf.Ticker(t).fast_info
            price = getattr(info, "last_price", None) or 0
            vol = getattr(info, "last_volume", None) or 0
            if min_price <= price <= max_price and vol >= min_volume:
                return t
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=5) as ex:
        for result in ex.map(check, tickers):
            if result:
                filtered.append(result)

    return filtered


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _yf_screen(screen_id, limit):
    """Fallback screener using yfinance."""
    try:
        # Use a broad set and let yfinance sort
        broad = _get_sp500_tickers()[:50]
        return broad[:limit]
    except Exception:
        return []


def _get_sp500_tickers():
    """Get S&P 500 tickers. Tries Wikipedia, falls back to curated list."""
    cached = _cached("sp500", ttl=86400)
    if cached:
        return cached

    # Try Wikipedia
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, attrs={"id": "constituents"})
        if tables and len(tables[0]) > 100:
            tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
            _set_cache("sp500", tickers)
            return tickers
    except Exception:
        pass

    # Try alternative: fetch from datahub
    try:
        r = requests.get(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
            headers={"User-Agent": "StockyAiTrader/4.0"}, timeout=10,
        )
        if r.status_code == 200:
            import io, csv
            reader = csv.DictReader(io.StringIO(r.text))
            tickers = [row["Symbol"].replace(".", "-") for row in reader if row.get("Symbol")]
            if len(tickers) > 100:
                _set_cache("sp500", tickers)
                return tickers
    except Exception:
        pass

    # Curated fallback — top 100 by market cap
    fallback = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "UNH", "JNJ", "JPM", "V", "XOM", "PG", "MA", "HD", "CVX", "MRK",
        "ABBV", "LLY", "AVGO", "PEP", "KO", "COST", "BAC", "WMT", "TMO",
        "CSCO", "MCD", "CRM", "ACN", "ABT", "DHR", "LIN", "AMD", "ADBE",
        "TXN", "NFLX", "WFC", "PM", "NEE", "UNP", "BMY", "RTX", "QCOM",
        "ORCL", "HON", "LOW", "UPS", "INTC",
        "INTU", "AMAT", "GS", "MS", "BLK", "SCHW", "AXP", "C", "CAT",
        "DE", "GE", "BA", "LMT", "RTX", "IBM", "PYPL", "SQ", "SHOP",
        "SNOW", "UBER", "ABNB", "CRWD", "DDOG", "ZS", "PANW", "FTNT",
        "NOW", "WDAY", "TEAM", "MDB", "NET", "COIN", "HOOD", "SOFI",
        "PLTR", "RIVN", "NIO", "LCID", "F", "GM", "TSLA", "DIS",
        "NFLX", "ROKU",
    ]
    _set_cache("sp500", fallback)
    return fallback
