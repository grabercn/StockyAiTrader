"""
Addon: Reddit WallStreetBets Tracker

Monitors ticker mention frequency and sentiment on r/wallstreetbets
and r/stocks. Uses Reddit's public JSON API (no auth needed).

Why it matters:
    WSB-driven momentum trades are real. A sudden surge in mentions
    with bullish sentiment can precede 5-15% moves on popular tickers.
    The mention velocity (rate of change) is more useful than raw count.

Setup:
    No extra dependencies — uses `requests` (already installed).
    No API key needed — uses Reddit's public JSON endpoints.
"""

import requests
import time
import numpy as np

ADDON_NAME = "Reddit WSB Mentions"
ADDON_DESCRIPTION = "WallStreetBets mention tracking (free, no key)"
ADDON_FEATURES = ["wsb_mention_count", "wsb_sentiment_ratio"]
DEPENDENCIES = []
REQUIRES_API_KEY = False
API_KEY_NAME = ""

_cache = {}
_cache_ttl = 600  # 10 minutes — Reddit data doesn't change that fast

_SUBREDDITS = ["wallstreetbets", "stocks"]
_HEADERS = {"User-Agent": "StockyAiTrader/2.0 (research)"}


def check_available():
    return True, "Ready"


def get_features(ticker, data):
    """
    Count mentions and estimate sentiment for a ticker on Reddit.

    Returns:
        wsb_mention_count:   Number of mentions in hot posts (0-50+)
        wsb_sentiment_ratio: Rough bullish ratio based on upvote ratio
    """
    now = time.time()
    if ticker in _cache and (now - _cache[ticker].get("_time", 0)) < _cache_ttl:
        cached = _cache[ticker]
        return {k: v for k, v in cached.items() if k != "_time"}

    try:
        total_mentions = 0
        total_upvote_ratio = 0.0
        mention_posts = 0

        for sub in _SUBREDDITS:
            mentions, avg_ratio, count = _search_subreddit(sub, ticker)
            total_mentions += mentions
            total_upvote_ratio += avg_ratio * count
            mention_posts += count

        # Average upvote ratio across posts that mention the ticker
        # Higher ratio = more bullish sentiment
        if mention_posts > 0:
            sentiment = total_upvote_ratio / mention_posts
        else:
            sentiment = 0.5

        result = {
            "wsb_mention_count": float(min(total_mentions, 50)),  # Cap at 50
            "wsb_sentiment_ratio": float(sentiment),
        }

        _cache[ticker] = {**result, "_time": now}
        return result

    except Exception as e:
        print(f"Reddit WSB addon error: {e}")
        return _default()


def _search_subreddit(subreddit, ticker):
    """Search a subreddit's hot posts for ticker mentions."""
    try:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=50"
        resp = requests.get(url, headers=_HEADERS, timeout=10)

        if resp.status_code != 200:
            return 0, 0.5, 0

        posts = resp.json().get("data", {}).get("children", [])

        mentions = 0
        upvote_sum = 0.0
        mention_count = 0

        ticker_upper = ticker.upper()
        # Look for $TICKER or standalone TICKER in titles
        search_terms = [f"${ticker_upper}", f" {ticker_upper} ", f" {ticker_upper},"]

        for post in posts:
            post_data = post.get("data", {})
            title = post_data.get("title", "").upper()

            if any(term in f" {title} " for term in search_terms):
                mentions += 1
                upvote_sum += post_data.get("upvote_ratio", 0.5)
                mention_count += 1

        avg_ratio = upvote_sum / mention_count if mention_count > 0 else 0.5
        return mentions, avg_ratio, mention_count

    except Exception:
        return 0, 0.5, 0


def _default():
    return {"wsb_mention_count": 0.0, "wsb_sentiment_ratio": 0.5}
