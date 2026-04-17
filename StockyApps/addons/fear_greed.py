"""
Addon: CNN Fear & Greed Index

Fetches the CNN Fear & Greed Index — a composite of 7 market indicators:
- Market momentum (S&P 500 vs 125-day MA)
- Stock price strength (52-week highs vs lows)
- Stock price breadth (advancing vs declining volume)
- Put/Call ratio
- Market volatility (VIX)
- Safe haven demand (bonds vs stocks)
- Junk bond demand (yield spread)

Why it matters:
    A single number (0-100) that captures overall market sentiment.
    Extreme fear (<25) = contrarian buy. Extreme greed (>75) = caution.

Setup:
    No dependencies beyond `requests` (already installed).
    No API key needed.
"""

import requests
import time

ADDON_NAME = "CNN Fear & Greed Index"
ADDON_DESCRIPTION = "Market-wide fear/greed sentiment (free, no key)"
ADDON_FEATURES = ["fear_greed_index", "fear_greed_category"]
DEPENDENCIES = []
REQUIRES_API_KEY = False
API_KEY_NAME = ""

_cache = {"value": 50.0, "category": 0.0, "_time": 0}
_cache_ttl = 1800  # 30 minutes


def check_available():
    return True, "Ready"


def get_features(ticker, data):
    """
    Fetch the current Fear & Greed Index.

    Returns:
        fear_greed_index:    0-100 (0=extreme fear, 100=extreme greed)
        fear_greed_category: -1 (fear), 0 (neutral), +1 (greed)
    """
    now = time.time()
    if (now - _cache["_time"]) < _cache_ttl:
        return {
            "fear_greed_index": _cache["value"],
            "fear_greed_category": _cache["category"],
        }

    try:
        # CNN's Fear & Greed data endpoint
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "StockyAiTrader/2.0"}
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            data_json = resp.json()
            score = data_json.get("fear_and_greed", {}).get("score", 50)
            score = float(score)

            # Categorize: <40 = fear (-1), 40-60 = neutral (0), >60 = greed (+1)
            if score < 40:
                category = -1.0
            elif score > 60:
                category = 1.0
            else:
                category = 0.0

            _cache["value"] = score / 100.0  # Normalize to 0-1
            _cache["category"] = category
            _cache["_time"] = now

            return {
                "fear_greed_index": score / 100.0,
                "fear_greed_category": category,
            }

    except Exception as e:
        print(f"Fear & Greed addon error: {e}")

    return {"fear_greed_index": 0.5, "fear_greed_category": 0.0}
