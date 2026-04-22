"""
Feature engineering for stock price data.

Transforms raw OHLCV data into ML-ready features using technical indicators,
price statistics, and temporal patterns. Two feature sets are provided:

- INTRADAY_FEATURES: Fast-moving indicators for day trading (1m-30m bars)
- LONGTERM_FEATURES: Slower indicators for swing/position trading (daily bars)

Uses the `ta` library (pure Python, works on Python 3.10+).
"""

import numpy as np
import pandas as pd
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volume import OnBalanceVolumeIndicator


# ─── Feature column lists ────────────────────────────────────────────────────
# These define which columns the model will use as input.
# Must match what engineer_features() produces.

INTRADAY_FEATURES = [
    "log_return", "pct_change",
    "return_mean_5", "return_mean_10", "return_mean_20",
    "return_std_5", "return_std_10", "return_std_20",
    "volume_ratio_5", "volume_ratio_10", "volume_ratio_20",
    "price_vs_vwap",
    "bb_position", "bb_width", "macd_hist",
    "RSI_14", "RSI_2",
    "ema_cross", "obv_slope", "range_atr_ratio",
    "time_sin", "time_cos",
    "momentum_5", "momentum_10",
    "trend_consistency_5", "trend_consistency_10",
    "volume_direction", "range_position", "candle_body_ratio",
    "vader_sentiment", "finbert_sentiment",
]

LONGTERM_FEATURES = [
    "log_return", "pct_change",
    "return_mean_5", "return_mean_10", "return_mean_20", "return_mean_50",
    "return_std_5", "return_std_10", "return_std_20", "return_std_50",
    "volume_ratio_5", "volume_ratio_10", "volume_ratio_20", "volume_ratio_50",
    "sma_cross",
    "bb_position", "bb_width", "macd_hist",
    "RSI_14", "ema_cross", "obv_slope",
    "day_of_week",
    "vader_sentiment", "finbert_sentiment",
]


# ─── Individual feature builders ─────────────────────────────────────────────
# Each function adds one group of related features to the DataFrame.

def _add_return_features(data, windows):
    """Rolling mean and std of log returns + volume ratios at each window."""
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["pct_change"] = data["Close"].pct_change()

    for w in windows:
        data[f"return_mean_{w}"] = data["log_return"].rolling(w).mean()
        data[f"return_std_{w}"] = data["log_return"].rolling(w).std()
        data[f"volume_ratio_{w}"] = data["Volume"] / data["Volume"].rolling(w).mean()


def _add_vwap(data):
    """Volume-Weighted Average Price — the key intraday level.
    Price above VWAP = bullish, below = bearish."""
    cumulative_vp = (data["Close"] * data["Volume"]).cumsum()
    cumulative_v = data["Volume"].cumsum()
    data["vwap"] = cumulative_vp / cumulative_v
    data["price_vs_vwap"] = (data["Close"] - data["vwap"]) / data["vwap"]


def _add_bollinger(data):
    """Bollinger Bands: position within bands (0=lower, 1=upper) and band width."""
    bb = BollingerBands(data["Close"], window=20, window_dev=2)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    mid = bb.bollinger_mavg()

    # Store raw bands for charting
    data["BBU_20_2.0"] = upper
    data["BBL_20_2.0"] = lower
    data["BBM_20_2.0"] = mid

    # Derived features for the model
    bb_range = upper - lower
    data["bb_position"] = np.where(bb_range > 0, (data["Close"] - lower) / bb_range, 0.5)
    data["bb_width"] = np.where(mid > 0, bb_range / mid, 0.0)


def _add_macd(data):
    """MACD histogram — positive = bullish momentum, negative = bearish."""
    macd = MACD(data["Close"], window_slow=26, window_fast=12, window_sign=9)
    data["macd_hist"] = macd.macd_diff()
    data["macd_hist"] = data["macd_hist"].fillna(0.0)


def _add_rsi(data, lengths=(14,)):
    """RSI at one or more lookback lengths. 14 = standard, 2 = fast for scalping."""
    for length in lengths:
        rsi = RSIIndicator(data["Close"], window=length)
        data[f"RSI_{length}"] = rsi.rsi()


def _add_ema_cross(data):
    """EMA 9/21 crossover — positive = short EMA above long (bullish trend)."""
    ema9 = EMAIndicator(data["Close"], window=9).ema_indicator()
    ema21 = EMAIndicator(data["Close"], window=21).ema_indicator()

    # Store raw EMAs for charting
    data["EMA_9"] = ema9
    data["EMA_21"] = ema21

    # Crossover feature for the model
    data["ema_cross"] = np.where(ema21 > 0, (ema9 - ema21) / ema21, 0.0)


def _add_obv(data):
    """On-Balance Volume slope — rising OBV confirms price trend."""
    obv = OnBalanceVolumeIndicator(data["Close"], data["Volume"])
    data["OBV"] = obv.on_balance_volume()
    data["obv_slope"] = data["OBV"].diff(5)


def _add_atr(data):
    """Average True Range — measures volatility for stops and position sizing."""
    atr = AverageTrueRange(data["High"], data["Low"], data["Close"], window=14)
    data["ATRr_14"] = atr.average_true_range()

    # How wide is today's range vs the average? >1 = wider than normal
    data["range_atr_ratio"] = np.where(
        data["ATRr_14"] > 0,
        (data["High"] - data["Low"]) / data["ATRr_14"],
        1.0,
    )


def _add_time_features(data):
    """Cyclical time encoding for intraday data.
    The market open/close have very different behavior than midday."""
    if hasattr(data.index, "hour"):
        minutes = (data.index.hour - 9) * 60 + data.index.minute - 30
        total = 390  # 6.5 trading hours
        data["time_sin"] = np.sin(2 * np.pi * minutes / total)
        data["time_cos"] = np.cos(2 * np.pi * minutes / total)
    else:
        data["time_sin"] = 0.0
        data["time_cos"] = 0.0


def _add_momentum_features(data):
    """Momentum confirmation features — helps distinguish real moves from noise.
    These directly address BUY signal quality by confirming trend strength."""
    # Momentum slope: is the short-term trend accelerating?
    data["momentum_5"] = data["Close"].pct_change(5)
    data["momentum_10"] = data["Close"].pct_change(10)

    # Trend consistency: what % of last N bars were positive?
    data["trend_consistency_5"] = data["Close"].diff().rolling(5).apply(
        lambda x: (x > 0).sum() / len(x), raw=True)
    data["trend_consistency_10"] = data["Close"].diff().rolling(10).apply(
        lambda x: (x > 0).sum() / len(x), raw=True)

    # Volume confirmation: is volume higher on up-moves?
    up_vol = (data["Volume"] * (data["Close"].diff() > 0).astype(float)).rolling(10).sum()
    dn_vol = (data["Volume"] * (data["Close"].diff() <= 0).astype(float)).rolling(10).sum()
    data["volume_direction"] = np.where(dn_vol > 0, up_vol / dn_vol, 1.0)

    # Price position: where in the recent range is the price?
    high_10 = data["High"].rolling(10).max()
    low_10 = data["Low"].rolling(10).min()
    rng = high_10 - low_10
    data["range_position"] = np.where(rng > 0, (data["Close"] - low_10) / rng, 0.5)

    # Candle body ratio: strong closes vs wicks
    body = abs(data["Close"] - data["Open"])
    full_range = data["High"] - data["Low"]
    data["candle_body_ratio"] = np.where(full_range > 0, body / full_range, 0.5)


def _add_sma_cross(data):
    """SMA 50/200 golden/death cross — the classic long-term trend signal.
    Golden cross (SMA50 > SMA200) is bullish."""
    sma50 = SMAIndicator(data["Close"], window=50).sma_indicator()
    sma200 = SMAIndicator(data["Close"], window=200).sma_indicator()

    data["SMA_50"] = sma50
    data["SMA_200"] = sma200

    data["sma_cross"] = np.where(sma200 > 0, (sma50 - sma200) / sma200, 0.0)


def _add_day_of_week(data):
    """Day of week normalized 0-1 (Mon=0, Fri=1). Fridays often see selling."""
    if hasattr(data.index, "dayofweek"):
        data["day_of_week"] = data.index.dayofweek / 4.0
    else:
        data["day_of_week"] = 0.0


# ─── Public API ──────────────────────────────────────────────────────────────

def engineer_features(data, mode="intraday"):
    """
    Add all technical indicator features to a DataFrame of OHLCV data.

    Args:
        data: DataFrame with columns [Open, High, Low, Close, Volume]
        mode: "intraday" for day trading, "longterm" for swing/position trading

    Returns:
        The same DataFrame with feature columns added in-place.
    """
    # Shared indicators used by both modes
    windows = [5, 10, 20]
    if mode == "longterm":
        windows.append(50)

    _add_return_features(data, windows)
    _add_atr(data)
    _add_bollinger(data)
    _add_macd(data)
    _add_ema_cross(data)
    _add_obv(data)

    # Momentum features for all modes (improves BUY signal quality)
    _add_momentum_features(data)

    # Mode-specific indicators
    if mode == "intraday":
        _add_vwap(data)
        _add_rsi(data, lengths=(14, 2))  # Standard RSI + fast RSI for scalping
        _add_time_features(data)
    else:
        _add_rsi(data, lengths=(14,))
        _add_sma_cross(data)
        _add_day_of_week(data)

    return data
