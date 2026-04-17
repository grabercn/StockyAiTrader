"""
Addon: Finnhub Earnings & Economic Calendar

Tracks upcoming earnings dates and major economic events.

Why it matters:
    Stocks move 5-15% on earnings. Trading a stock without knowing
    earnings is tomorrow is gambling, not trading. This addon adds
    a "days_to_earnings" feature so the model can learn to:
    - Avoid entering positions right before earnings
    - Size down when earnings is close
    - Recognize post-earnings momentum

Setup:
    pip install finnhub-python
    Get a free API key at https://finnhub.io/register (60 calls/min free)
    Add to settings.json: "finnhub_api_key": "YOUR_KEY"
"""

import os
import json
import time
from datetime import datetime, timedelta

ADDON_NAME = "Finnhub Earnings Calendar"
ADDON_DESCRIPTION = "Earnings dates + economic events (free, 60 calls/min)"
ADDON_FEATURES = ["days_to_earnings", "has_earnings_this_week"]
DEPENDENCIES = ["finnhub-python"]
REQUIRES_API_KEY = True
API_KEY_NAME = "finnhub_api_key"

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "settings.json")

# Cache earnings data per ticker (refreshed every 6 hours)
_cache = {}
_cache_ttl = 21600


def check_available():
    try:
        import finnhub
        return True, "Ready (needs API key in Settings)"
    except ImportError:
        return False, "Install: pip install finnhub-python"


def _get_api_key():
    try:
        with open(_SETTINGS_PATH, "r") as f:
            return json.load(f).get(API_KEY_NAME, "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def get_features(ticker, data):
    """
    Check when the next earnings date is for this ticker.

    Returns:
        days_to_earnings:      Days until next earnings (-1 if unknown)
        has_earnings_this_week: 1.0 if earnings within 5 trading days, else 0.0
    """
    now = time.time()
    cache_key = f"{ticker}_earnings"

    # Return cached if fresh
    if cache_key in _cache and (now - _cache[cache_key].get("_time", 0)) < _cache_ttl:
        cached = _cache[cache_key]
        return {k: v for k, v in cached.items() if k != "_time"}

    api_key = _get_api_key()
    if not api_key:
        return _default()

    try:
        import finnhub
        client = finnhub.Client(api_key=api_key)

        # Search for earnings in the next 30 days
        today = datetime.now()
        from_date = today.strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=30)).strftime("%Y-%m-%d")

        earnings = client.earnings_calendar(
            _from=from_date, to=to_date, symbol=ticker
        )

        earnings_list = earnings.get("earningsCalendar", [])

        if earnings_list:
            # Find the nearest upcoming earnings date
            nearest = None
            for e in earnings_list:
                edate = datetime.strptime(e["date"], "%Y-%m-%d")
                if edate >= today:
                    if nearest is None or edate < nearest:
                        nearest = edate

            if nearest:
                days = (nearest - today).days
                result = {
                    "days_to_earnings": float(days),
                    "has_earnings_this_week": 1.0 if days <= 5 else 0.0,
                }
                _cache[cache_key] = {**result, "_time": now}
                return result

        result = _default()
        _cache[cache_key] = {**result, "_time": now}
        return result

    except Exception as e:
        print(f"Finnhub addon error: {e}")
        return _default()


def _default():
    return {"days_to_earnings": -1.0, "has_earnings_this_week": 0.0}
