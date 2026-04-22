# -*- coding: utf-8 -*-
"""
Post-Trade Reflection — learns verbal rules from trade outcomes.

After each agent cycle, this module:
1. Checks if any held positions hit stop-loss since last check
2. Analyzes the original decision context for that trade
3. Extracts a simple heuristic rule about what went wrong
4. Stores rules in settings (persisted across sessions)
5. Injects active rules into Gemini prompts for future trades

Rules are kept concise (max 10 active) and aged out after 7 days.
"""

import json
import os
from datetime import datetime, timedelta

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "settings.json")
MAX_RULES = 10
RULE_EXPIRY_DAYS = 7


def _load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=4)


def get_active_rules():
    """Get all active reflection rules (not expired)."""
    settings = _load_settings()
    rules = settings.get("reflection_rules", [])
    now = datetime.now()
    active = []
    for r in rules:
        created = datetime.fromisoformat(r.get("created", "2020-01-01"))
        if (now - created).days <= RULE_EXPIRY_DAYS:
            active.append(r)
    return active


def get_rules_for_prompt():
    """Get rules formatted for injection into Gemini prompts."""
    rules = get_active_rules()
    if not rules:
        return ""
    lines = ["LEARNED RULES FROM PAST TRADES (respect these):"]
    for r in rules:
        lines.append(f"  - {r['rule']} (from {r.get('ticker', '?')}, {r.get('created', '')[:10]})")
    return "\n".join(lines)


def check_and_reflect(held_map, agent_stocks, trade_history, log_fn=None):
    """
    Analyze recent trade outcomes and extract lessons.

    Args:
        held_map:       {SYMBOL: position_data} current positions
        agent_stocks:   {ticker: agent_stock_info} from engine
        trade_history:  list of recent trade log entries
        log_fn:         callable(msg, level) for logging

    Returns:
        list of new rules generated this cycle (may be empty)
    """
    new_rules = []

    # Check for positions with significant unrealized losses
    for sym, pos in held_map.items():
        upl_pct = float(pos.get("unrealized_plpc", 0))

        # If a position is down more than 3%, reflect on why we entered
        if upl_pct < -0.03:
            stock_info = agent_stocks.get(sym, {})
            if stock_info.get("_reflected"):
                continue  # Already reflected on this position

            entry_price = float(pos.get("avg_entry_price", 0))
            current = float(pos.get("current_price", 0))
            loss_pct = upl_pct * 100

            # Generate a simple rule based on what we know
            rule = _generate_rule(sym, stock_info, loss_pct)
            if rule:
                new_rules.append(rule)
                if log_fn:
                    log_fn(f"  Reflection: {sym} down {loss_pct:.1f}% — rule: {rule['rule']}", "rl")

                # Mark as reflected so we don't repeat
                if sym in agent_stocks:
                    agent_stocks[sym]["_reflected"] = True

    # Persist new rules
    if new_rules:
        _save_rules(new_rules)

    # Clean expired rules
    _cleanup_expired()

    return new_rules


def _generate_rule(ticker, stock_info, loss_pct):
    """Generate a verbal rule from a losing position."""
    confidence = stock_info.get("confidence", 0)
    signal = stock_info.get("signal", "?")
    checks = stock_info.get("checks", 0)

    # Different rule templates based on the failure mode
    if confidence > 0.9 and loss_pct < -5:
        rule_text = (
            f"Very high confidence ({confidence:.0%}) does not guarantee profit. "
            f"{ticker} dropped {loss_pct:.1f}% despite strong signal. "
            f"Consider reducing max position size for extreme-confidence trades."
        )
    elif checks <= 1 and loss_pct < -3:
        rule_text = (
            f"First-scan entries are risky. {ticker} was bought on first check "
            f"and lost {loss_pct:.1f}%. Wait for confirmation (2+ scans showing same signal)."
        )
    elif loss_pct < -5:
        rule_text = (
            f"Large loss on {ticker} ({loss_pct:.1f}%). "
            f"Tighten stop-loss or reduce position size for similar setups."
        )
    else:
        rule_text = (
            f"{ticker} {signal} at {confidence:.0%} confidence resulted in "
            f"{loss_pct:.1f}% drawdown. Review entry timing."
        )

    return {
        "rule": rule_text,
        "ticker": ticker,
        "loss_pct": loss_pct,
        "confidence": confidence,
        "signal": signal,
        "created": datetime.now().isoformat(),
    }


def _save_rules(new_rules):
    """Append new rules to settings, keeping max limit."""
    settings = _load_settings()
    existing = settings.get("reflection_rules", [])
    existing.extend(new_rules)
    # Keep only the most recent MAX_RULES
    if len(existing) > MAX_RULES:
        existing = existing[-MAX_RULES:]
    settings["reflection_rules"] = existing
    _save_settings(settings)


def _cleanup_expired():
    """Remove rules older than RULE_EXPIRY_DAYS."""
    settings = _load_settings()
    rules = settings.get("reflection_rules", [])
    now = datetime.now()
    kept = []
    for r in rules:
        try:
            created = datetime.fromisoformat(r.get("created", "2020-01-01"))
            if (now - created).days <= RULE_EXPIRY_DAYS:
                kept.append(r)
        except Exception:
            pass
    if len(kept) != len(rules):
        settings["reflection_rules"] = kept
        _save_settings(settings)
