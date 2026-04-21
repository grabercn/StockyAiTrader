"""
Data pipeline — fetches stock data and produces ML-ready DataFrames.

Combines:
1. yfinance price data (OHLCV)
2. Technical indicator features (core/features.py)
3. Sentiment scores (core/sentiment.py)
4. Addon features (addons/) — dynamically gathered from active addons
5. Triple barrier labels (core/labeling.py)

The addon system means new data sources are automatically included
in the model when they're enabled, with zero changes to this file.
"""

import yfinance as yf
import pandas as pd

from .features import engineer_features, INTRADAY_FEATURES, LONGTERM_FEATURES
from .sentiment import fetch_news, compute_sentiment
from .labeling import triple_barrier_label


def get_all_features(mode="intraday"):
    """
    Get the full feature list: core features + active addon features.

    This is what the model actually trains/predicts on.
    """
    base = INTRADAY_FEATURES if mode == "intraday" else LONGTERM_FEATURES
    try:
        from addons import get_addon_features
        return base + get_addon_features()
    except ImportError:
        return base


def fetch_intraday(ticker, period="5d", interval="5m"):
    """
    Fetch and prepare intraday data for day trading.

    Args:
        ticker:   Stock symbol (e.g. "AAPL")
        period:   How far back to fetch ("1d", "5d", etc.)
        interval: Bar size ("1m", "5m", "15m", "30m")

    Returns:
        DataFrame with features and labels, ready for model training/prediction.
        Empty DataFrame if no data available.
    """
    data = _fetch_price_data(ticker, period, interval)
    if data.empty:
        return data

    # Engineer intraday-specific features
    data = engineer_features(data, mode="intraday")

    # Add sentiment scores (core)
    _add_sentiment(data, ticker)

    # Add addon features (dynamic — whatever's enabled)
    _add_addon_features(data, ticker)

    # Label using aggressive intraday thresholds
    data["Label"] = triple_barrier_label(data, atr_tp=2.0, atr_sl=1.5, max_bars=20)

    # Drop rows where features couldn't be calculated (start of series)
    all_features = get_all_features("intraday")
    available = [c for c in all_features if c in data.columns]
    data = data.dropna(subset=available)

    return data


def fetch_longterm(ticker, period="1y"):
    """
    Fetch and prepare daily data for long-term analysis.

    Args:
        ticker: Stock symbol
        period: How far back ("3mo", "6mo", "1y", "2y", "5y")

    Returns:
        DataFrame with features and labels for long-term model.
    """
    data = _fetch_price_data(ticker, period, interval="1d")
    if data.empty:
        return data

    data = engineer_features(data, mode="longterm")
    _add_sentiment(data, ticker)
    _add_addon_features(data, ticker)

    # Wider barriers for long-term (more room to breathe)
    data["Label"] = triple_barrier_label(data, atr_tp=3.0, atr_sl=2.0, max_bars=30)

    all_features = get_all_features("longterm")
    available = [c for c in all_features if c in data.columns]
    data = data.dropna(subset=available)

    return data


# ─── Internal helpers ─────────────────────────────────────────────────────────

_price_cache = {}  # {(ticker, period, interval): (data, timestamp)}

def _fetch_price_data(ticker, period, interval):
    """Download OHLCV data from Yahoo Finance with short-term caching."""
    import time
    key = (ticker, period, interval)
    if key in _price_cache:
        data, ts = _price_cache[key]
        if time.time() - ts < 120:  # Cache for 2 minutes
            return data.copy()

    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    if not data.empty:
        _price_cache[key] = (data, time.time())
        # Keep cache small
        if len(_price_cache) > 50:
            oldest = min(_price_cache, key=lambda k: _price_cache[k][1])
            del _price_cache[oldest]
    return data if not data.empty else pd.DataFrame()


_sentiment_cache = {}

def _add_sentiment(data, ticker):
    """Fetch news and add core sentiment scores as columns. Cached per ticker."""
    import time
    if ticker in _sentiment_cache and time.time() - _sentiment_cache[ticker][2] < 300:
        data["vader_sentiment"] = _sentiment_cache[ticker][0]
        data["finbert_sentiment"] = _sentiment_cache[ticker][1]
        return

    headlines = fetch_news(ticker)
    vader_score, finbert_score = compute_sentiment(headlines)
    data["vader_sentiment"] = vader_score
    data["finbert_sentiment"] = finbert_score
    _sentiment_cache[ticker] = (vader_score, finbert_score, time.time())


def _add_addon_features(data, ticker):
    """
    Gather features from all active addons and add them as columns.

    Each addon returns a dict of {name: value_or_series}.
    Scalar values are broadcast to every row. Series values are aligned by index.
    If an addon fails, its features default to 0.0.
    """
    try:
        from addons import gather_features
        addon_data = gather_features(ticker, data)

        for name, value in addon_data.items():
            data[name] = value

    except ImportError:
        # Addons directory not found — that's fine, run without them
        pass
