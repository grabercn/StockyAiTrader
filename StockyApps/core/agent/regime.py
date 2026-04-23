# -*- coding: utf-8 -*-
"""
Market Regime Detector — classifies current market conditions.

Uses existing addon data (no extra API keys required):
- CNN Fear & Greed Index (0-100)
- SPY correlation + returns
- VIX level (if FRED key is set)

Regimes:
    RISK_ON    — Bullish, low fear, trending up. Normal trading.
    CAUTIOUS   — Mixed signals, moderate fear. Reduce sizing.
    RISK_OFF   — Bearish, high fear, VIX elevated. Minimal trading.
    VOLATILE   — Extreme readings either direction. Tighten everything.

Each regime returns multipliers that modify agent behavior:
    - size_mult:     Position size multiplier (0.25 to 1.2)
    - conf_boost:    Added to min_confidence threshold (0.0 to +0.2)
    - stop_mult:     Stop-loss ATR multiplier adjustment (0.8 to 1.5)
    - profit_mult:   Take-profit ATR multiplier adjustment (0.8 to 1.5)
    - scan_mult:     Cycle wait time multiplier (0.7 to 1.5)

Transparent: logs exactly what data drove the classification.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegimeState:
    """Current market regime with adjustment multipliers."""
    name: str = "NEUTRAL"
    size_mult: float = 1.0       # Position size multiplier
    conf_boost: float = 0.0      # Added to min_confidence threshold
    stop_mult: float = 1.0       # Stop-loss ATR multiplier
    profit_mult: float = 1.0     # Take-profit ATR multiplier
    scan_mult: float = 1.0       # Cycle interval multiplier
    description: str = ""        # Human-readable explanation

    # Raw inputs (for transparency)
    fear_greed: Optional[float] = None
    spy_return: Optional[float] = None
    vix: Optional[float] = None


# Regime definitions with their parameter adjustments
REGIMES = {
    "RISK_ON": {
        "size_mult": 1.2,
        "conf_boost": -0.05,   # Lower threshold = more trades
        "stop_mult": 1.2,      # Wider stops (let winners run)
        "profit_mult": 1.3,    # Bigger targets
        "scan_mult": 0.8,      # Check more often
    },
    "CAUTIOUS": {
        "size_mult": 0.7,
        "conf_boost": 0.05,    # Raise threshold slightly
        "stop_mult": 1.0,
        "profit_mult": 1.0,
        "scan_mult": 1.0,
    },
    "RISK_OFF": {
        "size_mult": 0.4,
        "conf_boost": 0.15,    # Much higher threshold
        "stop_mult": 0.8,      # Tighter stops
        "profit_mult": 0.8,    # Smaller targets (take profits faster)
        "scan_mult": 1.3,      # Check less often
    },
    "VOLATILE": {
        "size_mult": 0.25,
        "conf_boost": 0.20,    # Only highest conviction
        "stop_mult": 1.5,      # Wide stops (avoid whipsaws)
        "profit_mult": 1.5,    # Wide targets (capture big moves)
        "scan_mult": 0.7,      # Check often (fast-moving)
    },
}


def detect_regime(addon_data=None, log_fn=None):
    """
    Detect current market regime from addon signals.

    Args:
        addon_data: dict of addon signals (from get_all_addons)
        log_fn:     callable(msg, level) for transparent logging

    Returns:
        RegimeState with all adjustments
    """
    if not addon_data:
        addon_data = _fetch_addon_data()

    # Extract signals
    fg = addon_data.get("fear_greed_index")          # Could be 0-1 or 0-100 depending on addon
    fg_cat = addon_data.get("fear_greed_category")   # -1, 0, +1
    spy_ret = addon_data.get("spy_return")            # Recent SPY return
    spy_ret5 = addon_data.get("spy_return_5")         # 5-bar SPY return
    vix = addon_data.get("vix_level")                 # VIX index
    vix_chg = addon_data.get("vix_change")            # VIX change

    # Normalize F&G to 0-100 scale (addon may return 0-1)
    if fg is not None and fg <= 1.0:
        fg = fg * 100  # Convert 0-1 to 0-100
    vix_norm = min(1.0, vix / 40.0) if vix is not None else None  # VIX 40+ = extreme

    # Classification logic
    reasons = []

    # Score: positive = risk-on, negative = risk-off
    score = 0.0
    data_points = 0

    if fg is not None:
        data_points += 1
        if fg > 70:
            score += 0.4
            reasons.append(f"F&G={fg:.0f} (greedy)")
        elif fg > 55:
            score += 0.2
            reasons.append(f"F&G={fg:.0f} (bullish)")
        elif fg < 25:
            score -= 0.5
            reasons.append(f"F&G={fg:.0f} (extreme fear)")
        elif fg < 40:
            score -= 0.3
            reasons.append(f"F&G={fg:.0f} (fearful)")
        else:
            reasons.append(f"F&G={fg:.0f} (neutral)")

    if spy_ret is not None:
        data_points += 1
        if spy_ret > 0.005:
            score += 0.2
            reasons.append(f"SPY +{spy_ret:.2%}")
        elif spy_ret < -0.005:
            score -= 0.2
            reasons.append(f"SPY {spy_ret:.2%}")
        else:
            reasons.append(f"SPY flat ({spy_ret:+.2%})")

    if spy_ret5 is not None:
        if spy_ret5 > 0.02:
            score += 0.15
            reasons.append(f"SPY 5-bar +{spy_ret5:.2%}")
        elif spy_ret5 < -0.02:
            score -= 0.15
            reasons.append(f"SPY 5-bar {spy_ret5:.2%}")

    is_volatile = False
    if vix is not None:
        data_points += 1
        if vix > 30:
            is_volatile = True
            score -= 0.3
            reasons.append(f"VIX={vix:.1f} (HIGH)")
        elif vix > 20:
            score -= 0.1
            reasons.append(f"VIX={vix:.1f} (elevated)")
        else:
            score += 0.1
            reasons.append(f"VIX={vix:.1f} (calm)")

    # Check for extreme fear + greed readings
    if fg is not None and (fg < 15 or fg > 85):
        is_volatile = True
        reasons.append("EXTREME sentiment reading")

    # Classify
    if data_points == 0:
        regime_name = "CAUTIOUS"
        reasons.append("No regime data available — defaulting to cautious")
    elif is_volatile:
        regime_name = "VOLATILE"
    elif score > 0.3:
        regime_name = "RISK_ON"
    elif score < -0.2:
        regime_name = "RISK_OFF"
    else:
        regime_name = "CAUTIOUS"

    params = REGIMES[regime_name]
    desc = f"{regime_name}: {', '.join(reasons)}"

    state = RegimeState(
        name=regime_name,
        description=desc,
        fear_greed=fg,
        spy_return=spy_ret,
        vix=vix,
        **params,
    )

    if log_fn:
        log_fn(f"  Regime: {desc}", "agent")
        log_fn(
            f"  Adjustments: size={params['size_mult']:.1f}x, "
            f"conf_boost=+{params['conf_boost']:.0%}, "
            f"stop={params['stop_mult']:.1f}x, "
            f"target={params['profit_mult']:.1f}x", "agent")

    return state


def _fetch_addon_data():
    """Fetch regime-relevant addon data (Fear & Greed, SPY, VIX)."""
    import pandas as pd
    data = {}
    empty_df = pd.DataFrame()  # Addons require (ticker, data) but most don't use the df
    try:
        import importlib
        from addons import get_all_addons
        for addon in get_all_addons():
            if not addon.available or not addon.enabled:
                continue
            if addon.module_name in ("fear_greed", "spy_correlation", "fred_macro"):
                try:
                    mod = importlib.import_module(f"addons.{addon.module_name}")
                    if hasattr(mod, "get_features"):
                        feats = mod.get_features("SPY", empty_df)
                        if isinstance(feats, dict):
                            data.update(feats)
                except Exception:
                    pass
    except Exception:
        pass
    return data
