"""
Addon: FRED Macro Indicators (Federal Reserve Economic Data)

Pulls key economic indicators that drive market regime:
- VIX (fear gauge) — high VIX = volatile, low VIX = calm
- 10Y-2Y Treasury spread — inverted = recession risk
- Federal Funds Rate — rising = hawkish = bearish for stocks

Why it matters:
    Your model currently trades blind to macro conditions.
    A stock might look bullish on technicals, but if VIX is spiking
    and the yield curve is inverting, the whole market is about to sell off.

Setup:
    pip install fredapi
    Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html
    Add to settings.json: "fred_api_key": "YOUR_KEY"
"""

import os
import json
import numpy as np

ADDON_NAME = "FRED Macro Indicators"
ADDON_DESCRIPTION = "VIX, yield curve, fed rate from Federal Reserve (free)"
ADDON_FEATURES = ["vix_level", "vix_change", "yield_curve", "fed_rate"]
DEPENDENCIES = ["fredapi"]
REQUIRES_API_KEY = True
API_KEY_NAME = "fred_api_key"

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "settings.json")

# Cache to avoid hitting the API every prediction cycle
_cache = {}
_cache_ttl = 3600  # 1 hour — macro data doesn't change fast


def check_available():
    try:
        import fredapi
        return True, "Ready (needs API key in Settings)"
    except ImportError:
        return False, "Install: pip install fredapi"


def _get_api_key():
    try:
        with open(_SETTINGS_PATH, "r") as f:
            return json.load(f).get(API_KEY_NAME, "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def get_features(ticker, data):
    """
    Fetch macro indicators from FRED.

    Returns:
        vix_level:    Current VIX value (typically 10-40)
        vix_change:   VIX percent change from previous day
        yield_curve:  10Y-2Y Treasury spread (negative = inverted = danger)
        fed_rate:     Current Federal Funds Rate
    """
    import time
    now = time.time()

    # Return cached data if fresh enough
    if _cache and (now - _cache.get("_time", 0)) < _cache_ttl:
        return {k: v for k, v in _cache.items() if k != "_time"}

    api_key = _get_api_key()
    if not api_key:
        return _default()

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)

        # VIX — CBOE Volatility Index
        vix = fred.get_series("VIXCLS", observation_start="2024-01-01")
        vix = vix.dropna()
        vix_level = float(vix.iloc[-1]) if len(vix) > 0 else 20.0
        vix_prev = float(vix.iloc[-2]) if len(vix) > 1 else vix_level
        vix_change = (vix_level - vix_prev) / vix_prev if vix_prev > 0 else 0.0

        # Yield curve — 10Y minus 2Y Treasury (T10Y2Y)
        spread = fred.get_series("T10Y2Y", observation_start="2024-01-01")
        spread = spread.dropna()
        yield_curve = float(spread.iloc[-1]) if len(spread) > 0 else 0.0

        # Federal Funds Rate
        ffr = fred.get_series("FEDFUNDS", observation_start="2024-01-01")
        ffr = ffr.dropna()
        fed_rate = float(ffr.iloc[-1]) if len(ffr) > 0 else 5.0

        result = {
            "vix_level": vix_level,
            "vix_change": vix_change,
            "yield_curve": yield_curve,
            "fed_rate": fed_rate,
        }

        # Cache it
        _cache.update(result)
        _cache["_time"] = now

        return result

    except Exception as e:
        print(f"FRED addon error: {e}")
        return _default()


def _default():
    return {"vix_level": 20.0, "vix_change": 0.0, "yield_curve": 0.0, "fed_rate": 5.0}
