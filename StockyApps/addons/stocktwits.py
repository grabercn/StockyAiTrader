"""
Addon: StockTwits Retail Sentiment

Fetches real-time bullish/bearish sentiment from StockTwits — the largest
social platform dedicated to stock trading.

Why it matters for day trading:
- Extreme bullish sentiment (>80%) often signals a local top (contrarian)
- Sudden spikes in message volume precede volatile moves
- Free, no API key needed, no auth required

Setup:
    No dependencies beyond `requests` (already installed).
    No API key needed — public endpoint.
"""

import requests
import numpy as np

ADDON_NAME = "StockTwits Sentiment"
ADDON_DESCRIPTION = "Retail trader sentiment from StockTwits (free, no key)"
ADDON_FEATURES = ["stocktwits_bull_ratio", "stocktwits_volume"]
DEPENDENCIES = []  # requests is already a core dependency
REQUIRES_API_KEY = False
API_KEY_NAME = ""

_API_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"


def check_available():
    """Always available — no special dependencies."""
    return True, "Ready"


def get_features(ticker, data):
    """
    Fetch StockTwits sentiment for a ticker.

    Returns:
        stocktwits_bull_ratio: 0.0-1.0 (fraction of messages that are bullish)
        stocktwits_volume:     Message count (normalized by dividing by 30)
    """
    try:
        resp = requests.get(
            _API_URL.format(ticker=ticker),
            timeout=5,
            headers={"User-Agent": "StockyAiTrader/2.0"},
        )

        if resp.status_code != 200:
            return _default()

        messages = resp.json().get("messages", [])
        if not messages:
            return _default()

        # Count bullish vs bearish messages
        bullish = 0
        bearish = 0
        for msg in messages:
            sentiment = (msg.get("entities") or {}).get("sentiment", {})
            basic = sentiment.get("basic") if sentiment else None
            if basic == "Bullish":
                bullish += 1
            elif basic == "Bearish":
                bearish += 1

        total_sentiment = bullish + bearish
        bull_ratio = bullish / total_sentiment if total_sentiment > 0 else 0.5

        # Message volume — normalize by typical count (30 messages per page)
        volume = len(messages) / 30.0

        return {
            "stocktwits_bull_ratio": float(bull_ratio),
            "stocktwits_volume": float(volume),
        }

    except Exception as e:
        print(f"StockTwits addon error: {e}")
        return _default()


def _default():
    return {"stocktwits_bull_ratio": 0.5, "stocktwits_volume": 0.0}
