"""
Addon: SPY / Market Correlation

Tracks how the overall market (SPY) is performing and computes
the correlation between your target stock and the broader market.

Why it matters:
    80% of stocks follow the market. If SPY is tanking, it doesn't
    matter how good your stock's technicals look — it's probably
    going down too. This gives the model market-regime awareness.

Setup:
    No extra dependencies — uses yfinance (already installed).
    No API key needed.
"""

import numpy as np

ADDON_NAME = "SPY Market Correlation"
ADDON_DESCRIPTION = "Market regime + stock-to-SPY correlation (free)"
ADDON_FEATURES = ["spy_return", "spy_return_5", "stock_spy_corr"]
DEPENDENCIES = []  # yfinance is a core dep
REQUIRES_API_KEY = False
API_KEY_NAME = ""

_spy_cache = None
_spy_cache_time = 0
_CACHE_TTL = 300  # 5 minutes


def check_available():
    return True, "Ready"


def get_features(ticker, data):
    """
    Compute market context features.

    Returns:
        spy_return:     SPY's latest bar return (market direction right now)
        spy_return_5:   SPY's 5-bar rolling return (short-term market trend)
        stock_spy_corr: Rolling correlation between this stock and SPY
    """
    import time
    global _spy_cache, _spy_cache_time

    try:
        import yfinance as yf

        # Fetch SPY data (cached for 5 min to avoid hammering yfinance)
        now = time.time()
        if _spy_cache is None or (now - _spy_cache_time) > _CACHE_TTL:
            # Match the same period/interval as the input data
            freq = _infer_interval(data)
            period = "5d" if freq in ("1m", "5m", "15m", "30m") else "1y"
            spy = yf.Ticker("SPY").history(period=period, interval=freq)
            if not spy.empty:
                _spy_cache = spy
                _spy_cache_time = now

        if _spy_cache is None or _spy_cache.empty:
            return _default()

        spy = _spy_cache

        # SPY returns
        spy_returns = spy["Close"].pct_change()
        spy_return = float(spy_returns.iloc[-1]) if len(spy_returns) > 0 else 0.0
        spy_return_5 = float(spy_returns.rolling(5).mean().iloc[-1]) if len(spy_returns) >= 5 else 0.0

        # Correlation between stock and SPY (last 20 bars)
        stock_returns = data["Close"].pct_change()
        min_len = min(len(stock_returns), len(spy_returns), 20)

        if min_len >= 10:
            corr = np.corrcoef(
                stock_returns.iloc[-min_len:].fillna(0).values,
                spy_returns.iloc[-min_len:].fillna(0).values,
            )[0, 1]
            corr = float(corr) if not np.isnan(corr) else 0.5
        else:
            corr = 0.5

        return {
            "spy_return": spy_return,
            "spy_return_5": spy_return_5,
            "stock_spy_corr": corr,
        }

    except Exception as e:
        print(f"SPY correlation addon error: {e}")
        return _default()


def _infer_interval(data):
    """Guess the bar interval from the data index."""
    if len(data) < 2:
        return "5m"
    diff = (data.index[1] - data.index[0]).total_seconds()
    if diff <= 60:
        return "1m"
    elif diff <= 300:
        return "5m"
    elif diff <= 900:
        return "15m"
    elif diff <= 1800:
        return "30m"
    elif diff <= 3600:
        return "1h"
    else:
        return "1d"


def _default():
    return {"spy_return": 0.0, "spy_return_5": 0.0, "stock_spy_corr": 0.5}
