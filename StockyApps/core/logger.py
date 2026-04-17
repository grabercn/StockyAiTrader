"""
Structured Trading Logger — saves every decision with full reasoning.

Logs are saved as JSON files in the logs/ directory, organized by date.
Each log entry includes:
- The model's decision and confidence
- All feature values at the time of decision
- Feature importances (what drove the decision)
- Addon data that was active
- Human-readable reasoning
- Outcome tracking (for future model improvement)

Log format is designed to be re-ingested for training better models.
"""

import os
import json
from datetime import datetime
from typing import Optional


# ─── Log directory ────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _today_log_path():
    """One log file per day: logs/2025-01-15.jsonl"""
    _ensure_log_dir()
    return os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.jsonl")


def _session_log_path():
    """One summary log per session: logs/session_2025-01-15_093000.json"""
    _ensure_log_dir()
    return os.path.join(LOG_DIR, f"session_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json")


# ─── Log entry builders ──────────────────────────────────────────────────────

def log_decision(ticker, action, confidence, price, position_size,
                 stop_loss, take_profit, atr, probs,
                 feature_values=None, feature_importances=None,
                 active_addons=None, reasoning="", extra=None):
    """
    Log a single trading decision with full context.

    Appends one JSON line to today's log file.
    Each line is a complete, self-contained record.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "decision",
        "ticker": ticker,
        "action": action,
        "confidence": round(confidence, 4),
        "price": round(price, 4),
        "position_size": position_size,
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "atr": round(atr, 4),
        "probabilities": {
            "sell": round(probs[0], 4),
            "hold": round(probs[1], 4),
            "buy": round(probs[2], 4),
        },
        "reasoning": reasoning,
    }

    # Feature values at decision time (for retraining)
    if feature_values:
        entry["features"] = {k: round(v, 6) if isinstance(v, float) else v
                             for k, v in feature_values.items()}

    # What features drove this decision
    if feature_importances:
        entry["feature_importances"] = {k: round(v, 2)
                                        for k, v in feature_importances.items()}

    # Which addons were active
    if active_addons:
        entry["active_addons"] = active_addons

    # Any extra context
    if extra:
        entry["extra"] = extra

    _append_jsonl(entry)
    return entry


def log_trade_execution(ticker, side, qty, order_type, order_id,
                        fill_price=None, status="submitted", error=None):
    """Log when a trade is actually executed (or fails)."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "execution",
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "order_type": order_type,
        "order_id": order_id,
        "fill_price": fill_price,
        "status": status,
        "error": error,
    }
    _append_jsonl(entry)
    return entry


def log_scan_results(tickers_scanned, results_summary, duration_seconds):
    """Log a portfolio scan session summary."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "scan",
        "tickers_scanned": tickers_scanned,
        "duration_seconds": round(duration_seconds, 1),
        "results": results_summary,
    }
    _append_jsonl(entry)
    return entry


def log_event(event_type, message, data=None):
    """Log a general event (startup, error, config change, etc.)."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "event",
        "event": event_type,
        "message": message,
    }
    if data:
        entry["data"] = data
    _append_jsonl(entry)
    return entry


# ─── Log reading (for the UI log viewer) ─────────────────────────────────────

def get_today_logs(max_entries=200):
    """Read today's log entries (most recent first)."""
    path = _today_log_path()
    return _read_jsonl(path, max_entries)


def get_log_files():
    """List all log files with dates and sizes."""
    _ensure_log_dir()
    files = []
    for f in sorted(os.listdir(LOG_DIR), reverse=True):
        if f.endswith(".jsonl"):
            path = os.path.join(LOG_DIR, f)
            size = os.path.getsize(path)
            date = f.replace(".jsonl", "")
            files.append({"date": date, "file": f, "size_kb": round(size / 1024, 1)})
    return files


def get_log_entries(filename, max_entries=500):
    """Read entries from a specific log file."""
    path = os.path.join(LOG_DIR, filename)
    return _read_jsonl(path, max_entries)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _append_jsonl(entry):
    """Append a single JSON line to today's log file."""
    try:
        path = _today_log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        print(f"Logger error: {e}")


def _read_jsonl(path, max_entries):
    """Read a JSONL file and return entries (most recent first)."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = []
        for line in reversed(lines[-max_entries:]):
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries
    except Exception:
        return []
