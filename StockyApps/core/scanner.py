"""
Portfolio Scanner — Scans multiple tickers, scores them, and ranks by opportunity.

Runs the full pipeline (data fetch + features + addons + LightGBM) across
a list of tickers concurrently using ThreadPoolExecutor. Returns ranked
results with confidence scores and risk-adjusted sizing.

This is the brain behind the AutoTrader's portfolio recommendations.
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .data import fetch_intraday, get_all_features
from .model import train_lgbm, predict_lgbm
from .risk import RiskManager
from .labeling import LABEL_NAMES


@dataclass
class ScanResult:
    """Result of scanning a single ticker."""
    ticker: str
    action: str               # "BUY", "SELL", "HOLD"
    confidence: float         # 0-1, probability of predicted class
    price: float
    position_size: int        # ATR-based recommended shares
    stop_loss: float
    take_profit: float
    atr: float
    probs: list               # [sell_prob, hold_prob, buy_prob]
    feature_importances: dict  # Top features driving this decision
    reasoning: str            # Human-readable explanation
    score: float              # Composite ranking score (higher = better opportunity)
    error: Optional[str] = None


# ─── Default scan universe ───────────────────────────────────────────────────
# Popular day-trading tickers with high volume and liquidity
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD",
    "SPY", "QQQ", "NFLX", "BA", "DIS", "PYPL", "SQ", "COIN",
    "SOFI", "PLTR", "NIO", "RIVN", "F", "GM", "JPM", "BAC",
]

MEME_TICKERS = ["GME", "AMC", "BB", "BBBY", "WISH", "CLOV", "SOFI"]

TECH_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "CRM", "ORCL"]

ETF_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "VTI", "ARKK", "XLF", "XLE", "XLK"]


def scan_ticker(ticker, period="5d", interval="5m", risk_manager=None):
    """
    Run the full analysis pipeline on a single ticker.

    Returns a ScanResult with action, confidence, sizing, and reasoning.
    """
    try:
        # Fetch data with all features + addons
        data = fetch_intraday(ticker, period=period, interval=interval)

        if data.empty or len(data) < 30:
            return ScanResult(
                ticker=ticker, action="HOLD", confidence=0, price=0,
                position_size=0, stop_loss=0, take_profit=0, atr=0,
                probs=[0, 1, 0], feature_importances={}, reasoning="Insufficient data",
                score=0, error="Not enough data",
            )

        # Train model on this ticker's data
        all_features = get_all_features("intraday")
        model, used_features = train_lgbm(data, all_features, ticker)

        if model is None:
            return ScanResult(
                ticker=ticker, action="HOLD", confidence=0, price=0,
                position_size=0, stop_loss=0, take_profit=0, atr=0,
                probs=[0, 1, 0], feature_importances={}, reasoning="Training failed",
                score=0, error="Model training failed",
            )

        # Predict
        actions, confidences, probs = predict_lgbm(model, data, used_features)
        last_action = LABEL_NAMES[actions[-1]]
        last_conf = float(confidences[-1])
        last_price = float(data["Close"].iloc[-1])
        last_probs = [float(p) for p in probs[-1]]
        atr = float(data["ATRr_14"].iloc[-1]) if "ATRr_14" in data.columns else last_price * 0.01

        # Risk calculations
        if risk_manager is None:
            risk_manager = RiskManager()

        size = risk_manager.position_size(last_price, atr)
        side = "buy" if last_action == "BUY" else "sell"
        sl = risk_manager.stop_loss(last_price, atr, side)
        tp = risk_manager.take_profit(last_price, atr, side)

        # Feature importance — which features mattered most for this prediction
        importances = {}
        if hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type="gain")
            feat_names = used_features[:len(imp)]
            sorted_idx = np.argsort(imp)[::-1][:8]  # Top 8 features
            for idx in sorted_idx:
                if idx < len(feat_names):
                    importances[feat_names[idx]] = float(imp[idx])

        # Generate reasoning — use LLM if available, fallback to template
        try:
            from .llm_reasoner import generate_reasoning as llm_reason
            reasoning = llm_reason(
                ticker, last_action, last_conf, last_price, atr, last_probs,
                feature_importances=importances,
            )
        except Exception:
            reasoning = _generate_reasoning(
                ticker, last_action, last_conf, last_price, atr, last_probs, importances, data
            )

        # Composite score for ranking (higher = better opportunity)
        # Weights: confidence matters most, BUY/SELL bonus, volume factor
        action_bonus = 1.0 if last_action in ("BUY", "SELL") else 0.0
        score = (last_conf * 0.6) + (action_bonus * 0.3) + (abs(last_probs[2] - last_probs[0]) * 0.1)

        return ScanResult(
            ticker=ticker, action=last_action, confidence=last_conf,
            price=last_price, position_size=size, stop_loss=sl, take_profit=tp,
            atr=atr, probs=last_probs, feature_importances=importances,
            reasoning=reasoning, score=score,
        )

    except Exception as e:
        return ScanResult(
            ticker=ticker, action="HOLD", confidence=0, price=0,
            position_size=0, stop_loss=0, take_profit=0, atr=0,
            probs=[0, 1, 0], feature_importances={},
            reasoning=f"Error: {e}", score=0, error=str(e),
        )


def scan_multiple(tickers, period="5d", interval="5m", risk_manager=None,
                  max_workers=3, progress_callback=None):
    """
    Scan multiple tickers concurrently and return ranked results.

    Args:
        tickers:           List of ticker symbols
        period/interval:   Data parameters
        risk_manager:      Shared RiskManager instance
        max_workers:       Concurrent threads (keep low on laptop — CPU bound)
        progress_callback: Called with (completed_count, total, ticker, result)

    Returns:
        List of ScanResult sorted by score (best opportunities first)
    """
    if risk_manager is None:
        risk_manager = RiskManager()

    results = []
    total = len(tickers)

    # Use ThreadPoolExecutor for concurrent scanning
    # Keep workers low (3) since LightGBM training is CPU-intensive on a laptop
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(scan_ticker, t, period, interval, risk_manager): t
            for t in tickers
        }

        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            try:
                result = future.result(timeout=120)
                results.append(result)
            except Exception as e:
                results.append(ScanResult(
                    ticker=ticker, action="HOLD", confidence=0, price=0,
                    position_size=0, stop_loss=0, take_profit=0, atr=0,
                    probs=[0, 1, 0], feature_importances={},
                    reasoning=f"Timeout/Error: {e}", score=0, error=str(e),
                ))

            if progress_callback:
                progress_callback(i + 1, total, ticker, results[-1])

    # Sort by score descending (best opportunities first)
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _generate_reasoning(ticker, action, confidence, price, atr, probs, importances, data):
    """
    Generate a human-readable explanation of why the model made this decision.
    """
    lines = [f"{action} {ticker} @ ${price:.2f} (confidence: {confidence:.1%})"]

    # Probability breakdown
    lines.append(f"Probabilities — SELL: {probs[0]:.1%}, HOLD: {probs[1]:.1%}, BUY: {probs[2]:.1%}")

    # Top driving features
    if importances:
        top = list(importances.items())[:5]
        feat_str = ", ".join([f"{k} ({v:.0f})" for k, v in top])
        lines.append(f"Key factors: {feat_str}")

    # Price context
    if "vwap" in data.columns:
        vwap = data["vwap"].iloc[-1]
        rel = "above" if price > vwap else "below"
        lines.append(f"Price is {rel} VWAP (${vwap:.2f})")

    if "RSI_14" in data.columns:
        rsi = data["RSI_14"].iloc[-1]
        if rsi > 70:
            lines.append(f"RSI at {rsi:.0f} — overbought")
        elif rsi < 30:
            lines.append(f"RSI at {rsi:.0f} — oversold")
        else:
            lines.append(f"RSI at {rsi:.0f}")

    # Volatility
    lines.append(f"ATR: ${atr:.2f} ({atr/price*100:.1f}% of price)")

    return " | ".join(lines)
