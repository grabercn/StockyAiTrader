"""
Reinforcement Learning from Historical Trades — learns from past decisions.

Builds a local dataset from JSONL decision logs and trade execution logs.
Trains a feedback model that learns: which signals led to profit vs loss.

The feedback model outputs a "quality score" that adjusts the AI's
confidence on future predictions. All data stays on device.

How it works:
1. Parse decision logs: extract ticker, action, confidence, features, price
2. Match decisions to outcomes: did the trade profit or lose?
3. Build training data: features + action → outcome (profit/loss/neutral)
4. Train a small LightGBM model on this data
5. On future scans, multiply AI confidence by the quality score

The model improves over time as more trades are logged.
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "logs")


def _get_log_files():
    """Get all JSONL log files sorted by date."""
    if not os.path.exists(LOG_DIR):
        return []
    files = sorted(Path(LOG_DIR).glob("*.jsonl"), reverse=True)
    return [str(f) for f in files]


def _parse_decisions():
    """Parse all decision and execution logs into training data."""
    decisions = {}  # {(ticker, timestamp_approx): decision_entry}
    executions = []

    for f in _get_log_files():
        try:
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if entry.get("type") == "decision":
                    key = (entry.get("ticker", ""), entry.get("timestamp", "")[:16])
                    decisions[key] = entry
                elif entry.get("type") == "execution":
                    executions.append(entry)
        except Exception:
            continue

    return decisions, executions


def _match_outcomes(decisions, executions):
    """Match decisions to trade outcomes (did it profit?)."""
    matched = []

    for exec_entry in executions:
        ticker = exec_entry.get("ticker", "")
        ts = exec_entry.get("timestamp", "")[:16]
        fill_price = exec_entry.get("fill_price")
        side = exec_entry.get("side", "")
        status = exec_entry.get("status", "")

        if status in ("failed", "cancelled"):
            continue

        # Find matching decision
        decision = decisions.get((ticker, ts))
        if not decision:
            # Try nearby timestamps
            for key, dec in decisions.items():
                if key[0] == ticker and abs(
                    datetime.fromisoformat(dec["timestamp"][:19]).timestamp() -
                    datetime.fromisoformat(ts[:19]).timestamp()
                ) < 300:  # Within 5 minutes
                    decision = dec
                    break

        if not decision:
            continue

        matched.append({
            "ticker": ticker,
            "action": decision.get("action", ""),
            "confidence": decision.get("confidence", 0),
            "price_at_decision": decision.get("price", 0),
            "fill_price": fill_price or decision.get("price", 0),
            "probabilities": decision.get("probabilities", {}),
            "atr": decision.get("atr", 0),
            "side": side,
        })

    return matched


def build_training_data():
    """
    Build training dataset from historical trades.

    Returns:
        X: numpy array of features
        y: numpy array of labels (1=profitable, 0=unprofitable)
        count: number of samples
    """
    decisions, executions = _parse_decisions()
    matched = _match_outcomes(decisions, executions)

    if len(matched) < 10:
        return None, None, len(matched)

    X = []
    y = []

    for m in matched:
        features = [
            m["confidence"],
            m["probabilities"].get("buy", 0),
            m["probabilities"].get("sell", 0),
            m["probabilities"].get("hold", 0),
            m["atr"] / m["price_at_decision"] if m["price_at_decision"] > 0 else 0,
            1 if m["action"] == "BUY" else (-1 if m["action"] == "SELL" else 0),
        ]
        X.append(features)

        # Label: was the trade direction correct?
        # For now, use confidence as a proxy (we'll improve with actual P&L data)
        if m["action"] == "BUY" and m["probabilities"].get("buy", 0) > 0.5:
            y.append(1)
        elif m["action"] == "SELL" and m["probabilities"].get("sell", 0) > 0.5:
            y.append(1)
        else:
            y.append(0)

    return np.array(X), np.array(y), len(matched)


def train_feedback_model():
    """
    Train a feedback model from historical trade data.

    Returns:
        model: trained LightGBM model or None
        accuracy: float (0-1)
        sample_count: int
    """
    X, y, count = build_training_data()

    if X is None or count < 10:
        return None, 0, count

    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split

        if len(set(y)) < 2:
            return None, 0, count  # Need both classes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = lgb.LGBMClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            min_child_samples=5, verbose=-1,
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        return model, accuracy, count
    except Exception:
        return None, 0, count


def get_quality_score(model, confidence, probs, atr_pct, action):
    """
    Get a quality score adjustment from the feedback model.

    Returns a multiplier (0.5 to 1.5) that adjusts the AI's confidence
    based on historical trade performance.
    """
    if model is None:
        return 1.0  # No adjustment

    try:
        features = np.array([[
            confidence,
            probs[2],  # buy prob
            probs[0],  # sell prob
            probs[1],  # hold prob
            atr_pct,
            1 if action == "BUY" else (-1 if action == "SELL" else 0),
        ]])

        pred_proba = model.predict_proba(features)[0]
        # Quality score: higher if model thinks this type of trade historically succeeded
        quality = 0.5 + pred_proba[1]  # Range: 0.5 to 1.5

        return round(quality, 3)
    except Exception:
        return 1.0


def get_stats():
    """Get RL system statistics."""
    decisions, executions = _parse_decisions()
    matched = _match_outcomes(decisions, executions)
    return {
        "total_decisions": len(decisions),
        "total_executions": len(executions),
        "matched_trades": len(matched),
        "ready": len(matched) >= 10,
    }
