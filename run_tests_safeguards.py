"""Test all 11 safeguards + historical impact analysis."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "StockyApps"))
import warnings; warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from collections import defaultdict
from core.agent.safeguards import *
from core.logger import get_log_files, get_log_entries

passed = failed = 0

def test(name, fn):
    global passed, failed
    print(f"\n[{passed+failed+1}] {name}")
    try:
        fn()
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

# ── Unit Tests ──────────────────────────────────────────────

def t_earnings():
    skip, days, reason = check_earnings_proximity("AAPL", 3)
    print(f"  AAPL: skip={skip}, days={days}")
    # AAPL has earnings in ~6 days, should NOT skip (>3 days)
    assert not skip or days <= 3

def t_stop_monitor():
    held = {"HIT": {"current_price": 95, "avg_entry_price": 100, "qty": 10},
            "OK": {"current_price": 110, "avg_entry_price": 100, "qty": 5}}
    stocks = {"HIT": {"stop_loss": 97}, "OK": {"stop_loss": 95}}
    hits = check_stop_loss_hits(held, stocks)
    assert "HIT" in hits and "OK" not in hits
    print(f"  Hits: {hits}")

def t_trailing():
    held = {"UP": {"current_price": 120, "avg_entry_price": 100, "qty": 5}}
    stocks = {"UP": {"stop_loss": 98.2, "entry_price": 100}}
    # ATR est = (100-98.2)/0.9 = 2.0, gain = 20, gain_atr = 10 -> max trail
    updates = update_trailing_stops(held, stocks)
    assert stocks["UP"]["stop_loss"] > 98.2
    print(f"  SL moved: 98.2 -> {stocks['UP']['stop_loss']}")

def t_sector():
    held = {"AAPL": {}, "NVDA": {}}
    ok, sec, cnt, reason = check_sector_limit("AMD", held, 2)
    assert not ok  # 2 tech already
    ok2, _, _, _ = check_sector_limit("JPM", held, 2)
    assert ok2  # Different sector
    print(f"  AMD blocked (Tech full), JPM allowed (Finance)")

def t_sentiment():
    score, detail = get_addon_sentiment("AAPL")
    print(f"  AAPL: score={score}, detail={detail}")

def t_fomc():
    is_ev, ev_type = is_economic_event_day()
    today = datetime.now().strftime("%Y-%m-%d")
    # Verify the function returns consistent results
    if today in FOMC_DATES_2026:
        assert is_ev and ev_type == "FOMC"
    elif today in CPI_DATES_2026:
        assert is_ev and ev_type == "CPI"
    else:
        assert not is_ev
    print(f"  Today ({today}): event={is_ev} type={ev_type or 'none'}")
    # Verify dates exist
    assert len(FOMC_DATES_2026) >= 14
    assert len(CPI_DATES_2026) >= 12
    print(f"  FOMC dates: {len(FOMC_DATES_2026)}, CPI dates: {len(CPI_DATES_2026)}")

def t_volume():
    class FakeResult:
        def __init__(self, atr, price):
            self.atr = atr; self.price = price
    ok, ratio, reason = check_volume(FakeResult(0.01, 100))  # 0.01% ATR = dead
    assert not ok
    ok2, ratio2, _ = check_volume(FakeResult(2.0, 100))  # 2% ATR = active
    assert ok2
    print(f"  Low vol blocked: {reason}")
    print(f"  Normal vol allowed: {ratio2:.2%}")

def t_cooldown():
    log = [{"ticker": "A", "pnl": -50, "side": "sell"}]
    cool, pnl = should_cooldown(log)
    assert cool and pnl == -50
    log2 = [{"ticker": "B", "pnl": 100, "side": "sell"}]
    cool2, _ = should_cooldown(log2)
    assert not cool2
    cool3, _ = should_cooldown([])
    assert not cool3
    print(f"  After loss: wait={cool}, After win: wait={cool2}, Empty: wait={cool3}")

def t_correlation():
    held = {"NVDA": {}, "JPM": {}}
    ok, corr, reason = check_correlation("AMD", held)
    assert not ok  # NVDA-AMD correlated
    ok2, _, _ = check_correlation("TSLA", held)
    assert ok2  # Not correlated with NVDA or JPM
    ok3, corr3, _ = check_correlation("BAC", held)
    assert not ok3  # JPM-BAC correlated
    print(f"  AMD blocked (corr NVDA), TSLA allowed, BAC blocked (corr JPM)")

def t_timeofday():
    mult, window = get_time_of_day_multiplier()
    assert 0.5 <= mult <= 1.5
    print(f"  Current: {mult:.2f}x ({window})")

def t_friday():
    is_fri, note = is_friday_afternoon()
    print(f"  Friday PM: {is_fri} ({note or 'not Friday'})")

print("=" * 60)
print("SAFEGUARD UNIT TESTS")
print("=" * 60)

test("Earnings Avoidance", t_earnings)
test("Stop-Loss Monitor", t_stop_monitor)
test("Trailing Stop", t_trailing)
test("Sector Limit", t_sector)
test("Addon Sentiment", t_sentiment)
test("FOMC/CPI Calendar", t_fomc)
test("Volume Filter", t_volume)
test("Loss Cooldown", t_cooldown)
test("Correlation Filter", t_correlation)
test("Time-of-Day Scoring", t_timeofday)
test("Friday Protection", t_friday)

# ── Historical Impact Analysis ──────────────────────────────
print(f"\n{'=' * 60}")
print("HISTORICAL IMPACT ANALYSIS")
print(f"{'=' * 60}")

# Load all decisions
decisions = []
for fi in get_log_files():
    for e in get_log_entries(fi["file"], 1000):
        if e.get("type") == "decision" and e.get("ticker") not in ("INTEG", "TEST"):
            decisions.append(e)
decisions.sort(key=lambda x: x.get("timestamp", ""))
print(f"\nAnalyzing {len(decisions)} historical decisions...\n")

# Build price series
series = defaultdict(list)
for d in decisions:
    t = d.get("ticker", "")
    if t and d.get("price", 0) > 0:
        series[t].append(d)

# Simulate each filter's impact on BUY accuracy
buy_decisions = [d for d in decisions if d.get("action") == "BUY"]
print(f"Total BUY decisions: {len(buy_decisions)}")

# Calculate baseline accuracy (no filters)
correct_base = 0
total_base = 0
for ticker, pts in series.items():
    for i in range(len(pts) - 1):
        if pts[i].get("action") != "BUY":
            continue
        total_base += 1
        if pts[i + 1]["price"] > pts[i]["price"]:
            correct_base += 1
base_acc = correct_base / total_base if total_base > 0 else 0
print(f"Baseline BUY accuracy: {correct_base}/{total_base} ({base_acc:.1%})")

# Filter: Volume (skip low ATR)
correct_vol = 0
total_vol = 0
for ticker, pts in series.items():
    for i in range(len(pts) - 1):
        if pts[i].get("action") != "BUY":
            continue
        atr = pts[i].get("atr", 0)
        price = pts[i].get("price", 1)
        if atr / price < 0.003:
            continue  # Would be filtered
        total_vol += 1
        if pts[i + 1]["price"] > pts[i]["price"]:
            correct_vol += 1
vol_acc = correct_vol / total_vol if total_vol > 0 else 0
filtered_out = total_base - total_vol
print(f"\nVolume filter (ATR > 0.3%): {correct_vol}/{total_vol} ({vol_acc:.1%}) — filtered {filtered_out}")

# Filter: Sector limit (max 2 per sector)
sector_counts = defaultdict(int)
correct_sec = 0
total_sec = 0
for ticker, pts in series.items():
    for i in range(len(pts) - 1):
        if pts[i].get("action") != "BUY":
            continue
        sec = get_sector(ticker)
        if sec != "Other" and sector_counts[sec] >= 2:
            continue  # Would be filtered
        sector_counts[sec] += 1
        total_sec += 1
        if pts[i + 1]["price"] > pts[i]["price"]:
            correct_sec += 1
sec_acc = correct_sec / total_sec if total_sec > 0 else 0
print(f"Sector limit (max 2): {correct_sec}/{total_sec} ({sec_acc:.1%})")

# Filter: Correlation (skip correlated pairs)
held_sim = set()
correct_corr = 0
total_corr = 0
for ticker, pts in series.items():
    for i in range(len(pts) - 1):
        if pts[i].get("action") != "BUY":
            continue
        blocked = False
        for pair in CORRELATED_PAIRS:
            if ticker.upper() in pair and (pair - {ticker.upper()}) & held_sim:
                blocked = True
                break
        if blocked:
            continue
        held_sim.add(ticker.upper())
        total_corr += 1
        if pts[i + 1]["price"] > pts[i]["price"]:
            correct_corr += 1
corr_acc = correct_corr / total_corr if total_corr > 0 else 0
print(f"Correlation filter: {correct_corr}/{total_corr} ({corr_acc:.1%})")

# Filter: Confirmation (2+ scans)
confirm_counts = defaultdict(int)
correct_conf = 0
total_conf = 0
for ticker, pts in series.items():
    for i in range(len(pts) - 1):
        if pts[i].get("action") != "BUY":
            confirm_counts[ticker] = 0
            continue
        confirm_counts[ticker] += 1
        if confirm_counts[ticker] < 2:
            continue
        total_conf += 1
        if pts[i + 1]["price"] > pts[i]["price"]:
            correct_conf += 1
conf_acc = correct_conf / total_conf if total_conf > 0 else 0
print(f"Confirmation (2+): {correct_conf}/{total_conf} ({conf_acc:.1%})")

# Combined: all filters
print(f"\n--- Combined Impact ---")
print(f"Baseline:      {base_acc:.1%} ({total_base} trades)")
print(f"+ Volume:      {vol_acc:.1%} ({total_vol} trades, -{filtered_out})")
print(f"+ Sector:      {sec_acc:.1%} ({total_sec} trades)")
print(f"+ Correlation: {corr_acc:.1%} ({total_corr} trades)")
print(f"+ Confirmation:{conf_acc:.1%} ({total_conf} trades)")

best_acc = max(vol_acc, sec_acc, corr_acc, conf_acc)
improvement = (best_acc - base_acc) * 100
print(f"\nBest single filter improvement: +{improvement:.1f} percentage points")

print(f"\n{'=' * 60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
if failed == 0:
    print("ALL SAFEGUARD TESTS PASSED")
print(f"{'=' * 60}")
