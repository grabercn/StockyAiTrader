"""
Triple Barrier Labeling for training data.

Instead of naive thresholds ("if price goes up 1%, label BUY"), this method
simulates what would actually happen if you entered a trade at each bar:

  1. Take-profit barrier: price rises by X * ATR → label BUY (2)
  2. Stop-loss barrier:   price falls by Y * ATR → label SELL (0)
  3. Time barrier:        neither hit within N bars → label HOLD (1)

This produces more realistic labels because:
- Thresholds adapt to current volatility via ATR
- Labels reflect achievable trades, not just direction
"""

import numpy as np
import pandas as pd


# Label constants — used throughout the codebase
SELL = 0
HOLD = 1
BUY = 2

LABEL_NAMES = {SELL: "SELL", HOLD: "HOLD", BUY: "BUY"}


def triple_barrier_label(data, atr_tp=1.5, atr_sl=1.5, max_bars=20):
    """
    Label each bar based on which price barrier is hit first.

    Symmetric barriers (TP == SL) produce balanced labels, which is critical
    for BUY accuracy. Previously TP=2.0 / SL=1.5 meant stop-loss was hit
    more often, biasing the model toward SELL and making BUY unreliable.

    Args:
        data:     DataFrame with 'Close' and 'ATRr_14' columns
        atr_tp:   Take-profit distance as multiple of ATR (higher = fewer BUY labels)
        atr_sl:   Stop-loss distance as multiple of ATR (higher = fewer SELL labels)
        max_bars: Max bars to look ahead before defaulting to HOLD

    Returns:
        numpy array of labels (0=SELL, 1=HOLD, 2=BUY), same length as data
    """
    labels = np.full(len(data), HOLD)
    closes = data["Close"].values
    has_atr = "ATRr_14" in data.columns

    for i in range(len(data) - 1):
        # Use ATR if available, otherwise fallback to 1% of price
        if has_atr and not np.isnan(data["ATRr_14"].iloc[i]):
            atr = data["ATRr_14"].iloc[i]
        else:
            atr = closes[i] * 0.01

        take_profit = closes[i] + atr_tp * atr
        stop_loss = closes[i] - atr_sl * atr
        end = min(i + max_bars, len(data))

        # Walk forward bar-by-bar to see which barrier is hit first
        for j in range(i + 1, end):
            if closes[j] >= take_profit:
                labels[i] = BUY
                break
            elif closes[j] <= stop_loss:
                labels[i] = SELL
                break
        # If neither barrier hit, label stays HOLD

    return labels
