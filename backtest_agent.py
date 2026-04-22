# -*- coding: utf-8 -*-
"""
Historical Backtest — simulate the agent pipeline on past decision data.

Uses JSONL log files to replay decisions and estimate P&L under
different configurations. NOT a full market simulation — uses actual
decision prices from logs as entry/exit points.

Usage:
    python backtest_agent.py
"""

import sys, os, json
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "StockyApps"))

from core.logger import get_log_files, get_log_entries


def load_all_data():
    """Load all decisions and executions from JSONL logs."""
    decisions = []
    executions = []
    for fi in get_log_files():
        for e in get_log_entries(fi["file"], 1000):
            t = e.get("type", "")
            if t == "decision":
                decisions.append(e)
            elif t == "execution":
                executions.append(e)
    # Sort chronologically
    decisions.sort(key=lambda x: x.get("timestamp", ""))
    return decisions, executions


def build_price_series(decisions):
    """Build {ticker: [(timestamp, price, action, confidence, atr), ...]}."""
    series = defaultdict(list)
    for d in decisions:
        ticker = d.get("ticker", "")
        if not ticker or ticker in ("INTEG", "TEST"):
            continue
        ts = d.get("timestamp", "")
        price = d.get("price", 0)
        action = d.get("action", "HOLD")
        conf = d.get("confidence", 0)
        atr = d.get("atr", 0)
        probs = d.get("probabilities", {})
        if price > 0:
            series[ticker].append({
                "ts": ts, "price": price, "action": action,
                "confidence": conf, "atr": atr, "probs": probs,
            })
    return dict(series)


def simulate(decisions, config):
    """
    Simulate trading with given config.

    Config keys:
        min_confidence: float (0-1)
        max_trades_per_day: int
        require_confirmation: bool (need 2+ scans same signal)
        use_regime: bool (skip in bad regimes — simulate with F&G if available)
        position_pct: float (fraction of BP per trade)
        stop_mult: float (ATR stop multiplier)
        profit_mult: float (ATR profit multiplier)
    """
    series = build_price_series(decisions)
    min_conf = config.get("min_confidence", 0.5)
    max_trades = config.get("max_trades_per_day", 8)
    require_confirm = config.get("require_confirmation", False)
    position_pct = config.get("position_pct", 0.20)
    stop_mult = config.get("stop_mult", 1.5)
    profit_mult = config.get("profit_mult", 2.5)

    starting_capital = 50000
    capital = starting_capital
    positions = {}  # {ticker: {qty, entry, stop, target}}
    trades = []
    daily_trades = defaultdict(int)
    prev_signals = {}  # {ticker: last_action} for confirmation

    # Process all decisions chronologically
    all_points = []
    for ticker, points in series.items():
        for p in points:
            all_points.append((p["ts"], ticker, p))
    all_points.sort(key=lambda x: x[0])

    for ts, ticker, point in all_points:
        day = ts[:10]
        price = point["price"]
        action = point["action"]
        conf = point["confidence"]
        atr = point["atr"]

        # Check existing positions for stop/target hits
        if ticker in positions:
            pos = positions[ticker]
            hit_stop = price <= pos["stop"] if pos["side"] == "long" else price >= pos["stop"]
            hit_target = price >= pos["target"] if pos["side"] == "long" else price <= pos["target"]

            if hit_stop:
                pnl = (pos["stop"] - pos["entry"]) * pos["qty"] if pos["side"] == "long" else (pos["entry"] - pos["stop"]) * pos["qty"]
                capital += pos["qty"] * pos["stop"]
                trades.append({"ticker": ticker, "side": pos["side"], "entry": pos["entry"],
                              "exit": pos["stop"], "qty": pos["qty"], "pnl": pnl, "reason": "stop",
                              "ts_entry": pos["ts"], "ts_exit": ts})
                del positions[ticker]
            elif hit_target:
                pnl = (pos["target"] - pos["entry"]) * pos["qty"] if pos["side"] == "long" else (pos["entry"] - pos["target"]) * pos["qty"]
                capital += pos["qty"] * pos["target"]
                trades.append({"ticker": ticker, "side": pos["side"], "entry": pos["entry"],
                              "exit": pos["target"], "qty": pos["qty"], "pnl": pnl, "reason": "target",
                              "ts_entry": pos["ts"], "ts_exit": ts})
                del positions[ticker]

        # Skip if already holding this ticker
        if ticker in positions:
            prev_signals[ticker] = action
            continue

        # Daily trade limit
        if daily_trades[day] >= max_trades:
            continue

        # Confidence filter
        if conf < min_conf:
            prev_signals[ticker] = action
            continue

        # Confirmation filter
        if require_confirm:
            prev = prev_signals.get(ticker)
            if prev != action:
                prev_signals[ticker] = action
                continue

        # Execute trade
        if action == "BUY" and capital > 100:
            spend = capital * position_pct
            qty = max(1, int(spend / price))
            cost = qty * price
            if cost <= capital:
                stop = price - (atr * stop_mult) if atr > 0 else price * 0.97
                target = price + (atr * profit_mult) if atr > 0 else price * 1.05
                positions[ticker] = {
                    "qty": qty, "entry": price, "stop": stop, "target": target,
                    "side": "long", "ts": ts,
                }
                capital -= cost
                daily_trades[day] += 1

        elif action == "SELL" and ticker in positions:
            pos = positions[ticker]
            pnl = (price - pos["entry"]) * pos["qty"]
            capital += pos["qty"] * price
            trades.append({"ticker": ticker, "side": "long", "entry": pos["entry"],
                          "exit": price, "qty": pos["qty"], "pnl": pnl, "reason": "signal",
                          "ts_entry": pos["ts"], "ts_exit": ts})
            del positions[ticker]
            daily_trades[day] += 1

        prev_signals[ticker] = action

    # Close remaining positions at last known price
    for ticker, pos in positions.items():
        last_price = series[ticker][-1]["price"] if ticker in series else pos["entry"]
        pnl = (last_price - pos["entry"]) * pos["qty"]
        capital += pos["qty"] * last_price
        trades.append({"ticker": ticker, "side": "long", "entry": pos["entry"],
                      "exit": last_price, "qty": pos["qty"], "pnl": pnl, "reason": "close",
                      "ts_entry": pos["ts"], "ts_exit": "end"})

    return {
        "starting_capital": starting_capital,
        "final_capital": capital,
        "total_return": (capital - starting_capital) / starting_capital,
        "total_pnl": capital - starting_capital,
        "num_trades": len(trades),
        "wins": sum(1 for t in trades if t["pnl"] > 0),
        "losses": sum(1 for t in trades if t["pnl"] <= 0),
        "win_rate": sum(1 for t in trades if t["pnl"] > 0) / max(1, len(trades)),
        "avg_win": sum(t["pnl"] for t in trades if t["pnl"] > 0) / max(1, sum(1 for t in trades if t["pnl"] > 0)),
        "avg_loss": sum(t["pnl"] for t in trades if t["pnl"] <= 0) / max(1, sum(1 for t in trades if t["pnl"] <= 0)),
        "best_trade": max((t["pnl"] for t in trades), default=0),
        "worst_trade": min((t["pnl"] for t in trades), default=0),
        "exits_by_reason": {r: sum(1 for t in trades if t["reason"] == r) for r in set(t["reason"] for t in trades)},
        "trades": trades,
    }


def main():
    print("=" * 70)
    print("STOCKY SUITE AGENT BACKTEST")
    print("Using historical decision data from JSONL logs")
    print("=" * 70)

    decisions, executions = load_all_data()
    print(f"\nData: {len(decisions)} decisions, {len(executions)} executions")
    print(f"Date range: {decisions[0]['timestamp'][:10]} to {decisions[-1]['timestamp'][:10]}")

    tickers = set(d.get("ticker", "") for d in decisions)
    print(f"Unique tickers: {len(tickers)}")

    # Define test configurations
    configs = {
        "Baseline (50% conf, no filters)": {
            "min_confidence": 0.50, "max_trades_per_day": 8,
            "require_confirmation": False, "position_pct": 0.20,
            "stop_mult": 1.5, "profit_mult": 2.5,
        },
        "Conservative (70% conf)": {
            "min_confidence": 0.70, "max_trades_per_day": 8,
            "require_confirmation": False, "position_pct": 0.15,
            "stop_mult": 1.5, "profit_mult": 2.5,
        },
        "With confirmation (2+ scans)": {
            "min_confidence": 0.50, "max_trades_per_day": 8,
            "require_confirmation": True, "position_pct": 0.20,
            "stop_mult": 1.5, "profit_mult": 2.5,
        },
        "Tight stops (1.0x ATR)": {
            "min_confidence": 0.50, "max_trades_per_day": 8,
            "require_confirmation": False, "position_pct": 0.20,
            "stop_mult": 1.0, "profit_mult": 2.0,
        },
        "Wide targets (3.5x ATR)": {
            "min_confidence": 0.50, "max_trades_per_day": 8,
            "require_confirmation": False, "position_pct": 0.20,
            "stop_mult": 1.5, "profit_mult": 3.5,
        },
        "Small positions (10%)": {
            "min_confidence": 0.50, "max_trades_per_day": 8,
            "require_confirmation": False, "position_pct": 0.10,
            "stop_mult": 1.5, "profit_mult": 2.5,
        },
        "Conservative + Confirmation": {
            "min_confidence": 0.65, "max_trades_per_day": 6,
            "require_confirmation": True, "position_pct": 0.15,
            "stop_mult": 1.2, "profit_mult": 2.5,
        },
        "Aggressive (35% conf, 2.0x ATR TP)": {
            "min_confidence": 0.35, "max_trades_per_day": 15,
            "require_confirmation": False, "position_pct": 0.20,
            "stop_mult": 1.2, "profit_mult": 2.0,
        },
    }

    print(f"\nRunning {len(configs)} strategy configurations...\n")
    print(f"{'Strategy':<40} {'P&L':>10} {'Return':>8} {'Trades':>7} {'WR':>6} {'AvgW':>10} {'AvgL':>10} {'Exits'}")
    print("-" * 120)

    results = {}
    for name, config in configs.items():
        r = simulate(decisions, config)
        results[name] = r
        exits = ", ".join(f"{k}={v}" for k, v in r["exits_by_reason"].items())
        print(f"{name:<40} ${r['total_pnl']:>+9,.0f} {r['total_return']:>+7.1%} {r['num_trades']:>6} "
              f"{r['win_rate']:>5.0%} ${r['avg_win']:>+9,.0f} ${r['avg_loss']:>+9,.0f}  {exits}")

    # Find best strategy
    best = max(results.items(), key=lambda x: x[1]["total_pnl"])
    worst = min(results.items(), key=lambda x: x[1]["total_pnl"])

    print(f"\n{'=' * 70}")
    print(f"BEST:  {best[0]} — ${best[1]['total_pnl']:+,.0f} ({best[1]['total_return']:+.1%})")
    print(f"WORST: {worst[0]} — ${worst[1]['total_pnl']:+,.0f} ({worst[1]['total_return']:+.1%})")

    # Detailed breakdown of best strategy
    b = best[1]
    print(f"\n--- Best Strategy Detail ---")
    print(f"Trades: {b['num_trades']} ({b['wins']}W / {b['losses']}L)")
    print(f"Best trade: ${b['best_trade']:+,.2f}")
    print(f"Worst trade: ${b['worst_trade']:+,.2f}")
    print(f"Risk/Reward: {abs(b['avg_win'])/abs(b['avg_loss']) if b['avg_loss'] else 0:.2f}:1")

    # Per-ticker P&L for best strategy
    ticker_pnl = defaultdict(float)
    for t in b["trades"]:
        ticker_pnl[t["ticker"]] += t["pnl"]
    print(f"\nPer-ticker P&L (best strategy):")
    for ticker, pnl in sorted(ticker_pnl.items(), key=lambda x: -x[1])[:10]:
        print(f"  {ticker:>6}: ${pnl:+,.2f}")
    print("  ...")
    for ticker, pnl in sorted(ticker_pnl.items(), key=lambda x: x[1])[:5]:
        print(f"  {ticker:>6}: ${pnl:+,.2f}")

    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS:")
    # Compare confirmation vs no confirmation
    base_pnl = results["Baseline (50% conf, no filters)"]["total_pnl"]
    conf_pnl = results["With confirmation (2+ scans)"]["total_pnl"]
    cons_pnl = results["Conservative (70% conf)"]["total_pnl"]
    if conf_pnl > base_pnl:
        print(f"  + Confirmation filter IMPROVES P&L by ${conf_pnl - base_pnl:+,.0f}")
    else:
        print(f"  - Confirmation filter HURTS P&L by ${conf_pnl - base_pnl:+,.0f}")
    if cons_pnl > base_pnl:
        print(f"  + Higher confidence (70%) IMPROVES P&L by ${cons_pnl - base_pnl:+,.0f}")
    else:
        print(f"  - Higher confidence (70%) HURTS P&L by ${cons_pnl - base_pnl:+,.0f}")

    tight = results["Tight stops (1.0x ATR)"]["total_pnl"]
    if tight > base_pnl:
        print(f"  + Tighter stops IMPROVE P&L by ${tight - base_pnl:+,.0f}")
    else:
        print(f"  - Tighter stops HURT P&L by ${tight - base_pnl:+,.0f}")

    small = results["Small positions (10%)"]["total_pnl"]
    if small > base_pnl:
        print(f"  + Smaller positions IMPROVE P&L by ${small - base_pnl:+,.0f}")
    else:
        print(f"  - Smaller positions reduce magnitude: ${small:+,.0f} vs ${base_pnl:+,.0f}")


if __name__ == "__main__":
    main()
