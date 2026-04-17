"""
Shared chart styling and plotting helpers.

Keeps all matplotlib formatting in one place so every chart
in the app looks consistent without duplicating style code.
"""

import numpy as np

# ─── Color palette ────────────────────────────────────────────────────────────
BG_DARK = "#1a1a2e"      # Window/figure background
BG_PANEL = "#16213e"      # Axis/panel background
BORDER = "#0f3460"        # Borders and group outlines
GRID = "#444444"          # Grid lines

COLOR_PRICE = "#00ccff"   # Main price line
COLOR_BUY = "#00ff88"     # Buy markers and positive values
COLOR_SELL = "#ff4444"    # Sell markers and negative values
COLOR_HOLD = "#ffaa00"    # Hold / warning
COLOR_VWAP = "#ff8800"    # VWAP line
COLOR_EMA_FAST = "#ff4444"  # EMA 9 (fast)
COLOR_EMA_SLOW = "#00ff88"  # EMA 21 (slow)
COLOR_SMA_50 = "#ff8800"  # SMA 50
COLOR_SMA_200 = "#ff4444" # SMA 200


def style_axis(ax, title=""):
    """
    Apply consistent dark-theme styling to a matplotlib axis.

    Call this after plotting data but before canvas.draw().
    """
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color="white", fontsize=14)
    ax.set_ylabel("Price ($)", color="white")
    ax.tick_params(colors="#888")
    ax.grid(True, alpha=0.15, color=GRID)
    ax.legend(
        loc="upper left", fontsize=8,
        facecolor=BG_PANEL, edgecolor=BORDER, labelcolor="white",
    )


def plot_buy_sell_markers(ax, x_values, close_prices, actions):
    """
    Overlay BUY (green up-arrow) and SELL (red down-arrow) markers on a price chart.

    Args:
        ax:           matplotlib axis to plot on
        x_values:     x-axis positions (usually range(len(data)))
        close_prices: array of closing prices (y-axis positions)
        actions:      array of ints (0=SELL, 1=HOLD, 2=BUY)
    """
    buy_mask = actions == 2
    sell_mask = actions == 0

    if buy_mask.any():
        ax.scatter(
            np.array(x_values)[buy_mask],
            close_prices[buy_mask],
            marker="^", color=COLOR_BUY, s=50, zorder=5, label="BUY",
        )
    if sell_mask.any():
        ax.scatter(
            np.array(x_values)[sell_mask],
            close_prices[sell_mask],
            marker="v", color=COLOR_SELL, s=50, zorder=5, label="SELL",
        )
