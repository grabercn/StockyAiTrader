# -*- coding: utf-8 -*-
"""
Session History — persistent storage for completed agent sessions.

Saves each completed agent session to a JSONL file so users can
review historical performance from the AI dashboard.

Storage: logs/session_history.jsonl  (one JSON object per line)
"""

import os
import json
from datetime import datetime

# Same log directory used by core.logger
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs")
SESSION_FILE = os.path.join(LOG_DIR, "session_history.jsonl")


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _next_session_id():
    """Return the next auto-incremented session ID."""
    sessions = get_all_sessions()
    if not sessions:
        return 1
    return max(s.get("session_id", 0) for s in sessions) + 1


def save_session(engine_state, start_time, end_time, profile_name):
    """
    Save a completed agent session to the JSONL history file.

    Args:
        engine_state: dict with keys from AgentEngine (session_pnl, wins,
            losses, trade_log, trades_today, cycle, bp, regime, ...).
            Can also be the dict returned by engine.get_state().
        start_time: ISO string or datetime of session start.
        end_time:   ISO string or datetime of session end.
        profile_name: aggressivity profile name (e.g. "Default").
    """
    _ensure_dir()

    # Normalize times
    if isinstance(start_time, datetime):
        start_time = start_time.isoformat()
    if isinstance(end_time, datetime):
        end_time = end_time.isoformat()

    # Calculate duration
    try:
        t0 = datetime.fromisoformat(start_time)
        t1 = datetime.fromisoformat(end_time)
        duration_minutes = round((t1 - t0).total_seconds() / 60, 1)
    except Exception:
        duration_minutes = 0.0

    # Extract data from engine state
    trade_log = engine_state.get("trade_log", [])
    wins = engine_state.get("wins", 0)
    losses = engine_state.get("losses", 0)
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0

    # Calculate realized P&L from trade log (only closed trades with pnl)
    realized_pnl = sum(
        t.get("pnl", 0) or 0 for t in trade_log if t.get("pnl") is not None
    )
    total_pnl = engine_state.get("session_pnl", realized_pnl)

    # Starting / ending BP
    starting_bp = engine_state.get("starting_bp", engine_state.get("bp", 0))
    ending_bp = engine_state.get("bp", starting_bp)

    # Regime distribution (from trade log timestamps + regime field)
    regime = engine_state.get("regime", "UNKNOWN")
    regime_distribution = engine_state.get("regime_distribution", {regime: 1.0})

    # Stocks traded
    stocks_traded = sorted(set(
        t.get("ticker", "") for t in trade_log if t.get("ticker")
    ))

    # Best / worst trade
    closed_trades = [t for t in trade_log if t.get("pnl") is not None]
    best_trade = None
    worst_trade = None
    if closed_trades:
        best = max(closed_trades, key=lambda t: t.get("pnl", 0))
        worst = min(closed_trades, key=lambda t: t.get("pnl", 0))
        best_trade = {
            "ticker": best.get("ticker", ""),
            "pnl": best.get("pnl", 0),
            "side": best.get("side", ""),
        }
        worst_trade = {
            "ticker": worst.get("ticker", ""),
            "pnl": worst.get("pnl", 0),
            "side": worst.get("side", ""),
        }

    entry = {
        "session_id": _next_session_id(),
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": duration_minutes,
        "starting_bp": starting_bp,
        "ending_bp": ending_bp,
        "total_pnl": round(total_pnl, 2),
        "realized_pnl": round(realized_pnl, 2),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "trades_executed": engine_state.get("trades_today", len(closed_trades)),
        "cycles_completed": engine_state.get("cycle", 0),
        "regime_distribution": regime_distribution,
        "stocks_traded": stocks_traded,
        "trade_log": trade_log,
        "profile_name": profile_name,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }

    with open(SESSION_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return entry


def get_all_sessions():
    """Load all saved sessions from the JSONL file. Returns list of dicts."""
    if not os.path.exists(SESSION_FILE):
        return []
    sessions = []
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sessions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    return sessions


def get_session(session_id):
    """Get a single session by ID. Returns dict or None."""
    for s in get_all_sessions():
        if s.get("session_id") == session_id:
            return s
    return None


def get_session_count():
    """Return the total number of saved sessions."""
    return len(get_all_sessions())
