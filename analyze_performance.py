"""Deep financial analysis of all historical trading data."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "StockyApps"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict
from core.logger import get_log_files, get_log_entries

decisions = []
executions = []
for fi in get_log_files():
    for e in get_log_entries(fi["file"], 2000):
        t = e.get("type", "")
        if t == "decision" and e.get("ticker") not in ("INTEG", "TEST"):
            decisions.append(e)
        elif t == "execution":
            executions.append(e)

decisions.sort(key=lambda x: x.get("timestamp", ""))
print(f"Data: {len(decisions)} decisions, {len(executions)} executions")
print(f"Days: {decisions[0]['timestamp'][:10]} to {decisions[-1]['timestamp'][:10]}")

series = defaultdict(list)
for d in decisions:
    t = d.get("ticker", "")
    if t and d.get("price", 0) > 0:
        series[t].append(d)

# 1. SIGNAL ACCURACY
print(f"\n{'=' * 50}")
print("SIGNAL ACCURACY (price direction)")
print(f"{'=' * 50}")

by_action = defaultdict(lambda: {"correct": 0, "total": 0})
by_conf = defaultdict(lambda: {"correct": 0, "total": 0})
by_feature = defaultdict(lambda: {"correct": 0, "total": 0})

for ticker, pts in series.items():
    for i in range(len(pts) - 1):
        curr, nxt = pts[i], pts[i + 1]
        action = curr.get("action", "HOLD")
        if action == "HOLD":
            continue
        moved_up = nxt["price"] > curr["price"]
        correct = (action == "BUY" and moved_up) or (action == "SELL" and not moved_up)
        conf = curr.get("confidence", 0)
        by_action[action]["total"] += 1
        if correct:
            by_action[action]["correct"] += 1
        if conf >= 0.7: bucket = "70-100%"
        elif conf >= 0.5: bucket = "50-70%"
        elif conf >= 0.3: bucket = "30-50%"
        else: bucket = "0-30%"
        by_conf[bucket]["total"] += 1
        if correct:
            by_conf[bucket]["correct"] += 1
        imps = curr.get("feature_importances", {})
        if imps:
            top_feat = max(imps, key=imps.get)
            by_feature[top_feat]["total"] += 1
            if correct:
                by_feature[top_feat]["correct"] += 1

for action in ["BUY", "SELL"]:
    d = by_action[action]
    acc = d["correct"] / d["total"] if d["total"] > 0 else 0
    print(f"  {action}: {d['correct']}/{d['total']} ({acc:.1%})")

print(f"\nBy confidence level:")
for bucket in sorted(by_conf.keys()):
    d = by_conf[bucket]
    acc = d["correct"] / d["total"] if d["total"] > 0 else 0
    print(f"  {bucket}: {d['correct']}/{d['total']} ({acc:.1%})")

print(f"\nTop features driving decisions (accuracy):")
feat_sorted = sorted(by_feature.items(), key=lambda x: x[1]["total"], reverse=True)
for feat, d in feat_sorted[:12]:
    acc = d["correct"] / d["total"] if d["total"] > 0 else 0
    print(f"  {feat:25s}: {d['correct']:3d}/{d['total']:3d} ({acc:.1%})")

# 2. TICKER PERFORMANCE
print(f"\n{'=' * 50}")
print("TICKER PERFORMANCE")
print(f"{'=' * 50}")

ticker_stats = defaultdict(lambda: {"correct": 0, "total": 0, "buy_c": 0, "buy_t": 0})
for ticker, pts in series.items():
    for i in range(len(pts) - 1):
        curr, nxt = pts[i], pts[i + 1]
        action = curr.get("action", "HOLD")
        if action == "HOLD":
            continue
        moved_up = nxt["price"] > curr["price"]
        correct = (action == "BUY" and moved_up) or (action == "SELL" and not moved_up)
        ticker_stats[ticker]["total"] += 1
        if correct:
            ticker_stats[ticker]["correct"] += 1
        if action == "BUY":
            ticker_stats[ticker]["buy_t"] += 1
            if correct:
                ticker_stats[ticker]["buy_c"] += 1

qualified = {t: s for t, s in ticker_stats.items() if s["total"] >= 5}
best = sorted(qualified.items(), key=lambda x: x[1]["correct"] / max(1, x[1]["total"]), reverse=True)
print(f"Best (>5 decisions):")
for t, s in best[:8]:
    acc = s["correct"] / s["total"]
    print(f"  {t:6s}: {acc:.0%} ({s['correct']}/{s['total']})")
print(f"Worst:")
for t, s in best[-8:]:
    acc = s["correct"] / s["total"]
    print(f"  {t:6s}: {acc:.0%} ({s['correct']}/{s['total']})")

# 3. OPTIMAL THRESHOLDS
print(f"\n{'=' * 50}")
print("BUY ACCURACY BY CONFIDENCE THRESHOLD")
print(f"{'=' * 50}")

for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    correct = total = 0
    for ticker, pts in series.items():
        for i in range(len(pts) - 1):
            if pts[i].get("action") != "BUY":
                continue
            if pts[i].get("confidence", 0) < thresh:
                continue
            total += 1
            if pts[i + 1]["price"] > pts[i]["price"]:
                correct += 1
    acc = correct / total if total > 0 else 0
    print(f"  >= {thresh:.0%}: {correct}/{total} ({acc:.1%})")

# 4. TIME OF DAY
print(f"\n{'=' * 50}")
print("TIME OF DAY ACCURACY")
print(f"{'=' * 50}")

by_hour = defaultdict(lambda: {"correct": 0, "total": 0})
for ticker, pts in series.items():
    for i in range(len(pts) - 1):
        curr, nxt = pts[i], pts[i + 1]
        action = curr.get("action", "HOLD")
        if action == "HOLD":
            continue
        ts = curr.get("timestamp", "")
        if len(ts) >= 13:
            hour = ts[11:13]
            moved_up = nxt["price"] > curr["price"]
            correct = (action == "BUY" and moved_up) or (action == "SELL" and not moved_up)
            by_hour[hour]["total"] += 1
            if correct:
                by_hour[hour]["correct"] += 1

for hour in sorted(by_hour.keys()):
    d = by_hour[hour]
    acc = d["correct"] / d["total"] if d["total"] > 0 else 0
    bar = "#" * int(acc * 20)
    print(f"  {hour}:00 - {acc:.0%} ({d['correct']}/{d['total']}) {bar}")

# 5. PROBABILITY MARGIN
print(f"\n{'=' * 50}")
print("BUY ACCURACY BY PROBABILITY MARGIN (buy_prob - sell_prob)")
print(f"{'=' * 50}")

for margin in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
    correct = total = 0
    for ticker, pts in series.items():
        for i in range(len(pts) - 1):
            if pts[i].get("action") != "BUY":
                continue
            probs = pts[i].get("probabilities", {})
            bp = probs.get("buy", 0)
            sp = probs.get("sell", 0)
            if (bp - sp) < margin:
                continue
            total += 1
            if pts[i + 1]["price"] > pts[i]["price"]:
                correct += 1
    acc = correct / total if total > 0 else 0
    print(f"  margin >= {margin:.0%}: {correct}/{total} ({acc:.1%})")

# 6. ATR ANALYSIS
print(f"\n{'=' * 50}")
print("BUY ACCURACY BY ATR% (volatility)")
print(f"{'=' * 50}")

for atr_min, atr_max, label in [(0, 0.005, "<0.5%"), (0.005, 0.01, "0.5-1%"), (0.01, 0.02, "1-2%"), (0.02, 0.05, "2-5%"), (0.05, 1.0, ">5%")]:
    correct = total = 0
    for ticker, pts in series.items():
        for i in range(len(pts) - 1):
            if pts[i].get("action") != "BUY":
                continue
            atr = pts[i].get("atr", 0)
            price = pts[i].get("price", 1)
            atr_pct = atr / price if price > 0 else 0
            if atr_pct < atr_min or atr_pct >= atr_max:
                continue
            total += 1
            if pts[i + 1]["price"] > pts[i]["price"]:
                correct += 1
    acc = correct / total if total > 0 else 0
    print(f"  ATR {label:8s}: {correct}/{total} ({acc:.1%})")

# SUMMARY
print(f"\n{'=' * 50}")
print("KEY FINDINGS")
print(f"{'=' * 50}")
