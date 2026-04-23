"""End-to-end integration test for all agent subsystems."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "StockyApps"))
import warnings; warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np

passed = 0
failed = 0

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

def t1():
    from core.market_hours import get_session, is_market_open, get_session_label
    s = get_session()
    assert hasattr(s, "can_trade") and hasattr(s, "can_scan")
    print(f"  {s.name} | trade={s.can_trade} | scan={s.can_scan} | {get_session_label()}")

def t2():
    from core.data import fetch_intraday
    from core.features import engineer_features, INTRADAY_FEATURES
    data = fetch_intraday("AAPL", period="5d", interval="5m")
    engineer_features(data, "intraday")
    missing = [f for f in INTRADAY_FEATURES if f not in data.columns]
    assert not missing, f"Missing: {missing}"
    momentum = ["momentum_5", "momentum_10", "trend_consistency_5", "volume_direction", "range_position", "candle_body_ratio"]
    for f in momentum:
        assert f in data.columns, f"{f} missing"
    print(f"  {len(INTRADAY_FEATURES)} features, {len(data)} rows, 0 missing")

def t3():
    from core.data import fetch_intraday
    from core.features import engineer_features
    from core.labeling import triple_barrier_label
    data = fetch_intraday("AAPL", period="5d", interval="5m")
    engineer_features(data, "intraday")
    labels = triple_barrier_label(data, atr_tp=1.5, atr_sl=1.5)
    u, c = np.unique(labels, return_counts=True)
    d = dict(zip(u, c))
    assert d.get(0, 0) > 0 and d.get(2, 0) > 0
    print(f"  SELL={d.get(0,0)} HOLD={d.get(1,0)} BUY={d.get(2,0)} (symmetric)")

def t4():
    from core.data import fetch_intraday
    from core.features import engineer_features, INTRADAY_FEATURES
    from core.labeling import triple_barrier_label
    from core.model import train_lgbm, predict_lgbm
    data = fetch_intraday("NVDA", period="5d", interval="5m")
    engineer_features(data, "intraday")
    data["Label"] = triple_barrier_label(data)
    clean = data.dropna()
    model, feats = train_lgbm(clean, INTRADAY_FEATURES, "TEST_E2E")
    assert model is not None
    actions, confs, probs = predict_lgbm(model, clean, feats)
    print(f"  {len(clean)} rows, S={sum(actions==0)} H={sum(actions==1)} B={sum(actions==2)}, conf={confs.mean():.1%}")

def t5():
    from core.risk import RiskManager
    rm = RiskManager()
    assert rm.atr_stop_mult == 0.9 and rm.atr_profit_mult == 5.5
    sl, tp = rm.stop_loss(100, 2.0), rm.take_profit(100, 2.0)
    assert sl < 100 and tp > 100
    print(f"  SL={rm.atr_stop_mult}x TP={rm.atr_profit_mult}x R:R={rm.atr_profit_mult/rm.atr_stop_mult:.1f}:1 | @100: SL=${sl} TP=${tp}")

def t6():
    from core.reinforcement import get_stats, train_feedback_model, get_quality_score
    stats = get_stats()
    m, acc, cnt = train_feedback_model()
    if m:
        q = get_quality_score(m, 0.7, [0.1, 0.2, 0.7], 0.01, "BUY")
        print(f"  {stats['matched_trades']} matched, {acc:.0%} acc, quality(BUY@70%)={q:.2f}")
    else:
        print(f"  {stats['matched_trades']} matched (need 10+ for model)")

def t7():
    from core.agent.regime import detect_regime
    for exp, dd in [("RISK_ON", {"fear_greed_index":80,"spy_return":0.015,"vix_level":14}),
                    ("CAUTIOUS", {"fear_greed_index":50,"spy_return":0.001}),
                    ("RISK_OFF", {"fear_greed_index":20,"spy_return":-0.02,"vix_level":28}),
                    ("VOLATILE", {"fear_greed_index":10,"spy_return":-0.03,"vix_level":35})]:
        r = detect_regime(dd)
        assert r.name == exp, f"Expected {exp}, got {r.name}"
        print(f"  {exp}: size={r.size_mult}x conf={r.conf_boost:+.0%}")

def t8():
    from core.agent.reflection import check_and_reflect, get_active_rules
    held = {"T9": {"unrealized_plpc": -0.05, "avg_entry_price": 100, "current_price": 95, "qty": 5}}
    stocks = {"T9": {"signal": "BUY", "confidence": 0.9, "checks": 1}}
    rules = check_and_reflect(held, stocks, [])
    print(f"  Generated: {len(rules)}, Active: {len(get_active_rules())}")

def t9():
    from core.agent.engine import AgentEngine
    e = AgentEngine(broker=None, log_fn=lambda m,l: None, settings_fn=dict)
    e._agent_stocks = {"A": {"signal": "BUY", "qty": 10, "mode": "Auto", "confidence": 0.8, "checks": 2}}
    e._trades_today = 3; e._cycle = 5; e._session_pnl = 100; e._wins = 4; e._losses = 1
    state = e.get_state()
    e2 = AgentEngine(broker=None, log_fn=lambda m,l: None, settings_fn=dict)
    e2.restore_state(state)
    assert e2.cycle == 5 and e2.trades_today == 3 and e2.session_pnl == 100
    state["saved_date"] = "2020-01-01"
    e3 = AgentEngine(broker=None, log_fn=lambda m,l: None, settings_fn=dict)
    e3.restore_state(state)
    assert e3.cycle == 0 and e3.trades_today == 0 and e3._wins == 4
    print(f"  Same-day: OK | New-day: reset OK, lifetime wins kept")

def t10():
    from core.agent.engine import AgentEngine
    e = AgentEngine(broker=None, log_fn=lambda m,l: None, settings_fn=dict)
    e._agent_stocks = {
        "KEEP": {"signal": "BUY", "confidence": 0.85, "checks": 2, "qty": 10, "mode": "Auto"},
        "PRUNE": {"signal": "HOLD", "confidence": 0.40, "checks": 5, "qty": 0, "mode": "Scanned"},
        "FRESH": {"signal": "BUY", "confidence": 0.72, "checks": 1, "qty": 0, "mode": "Scanned"},
    }
    pruned = e._prune_stale_stocks({})
    assert pruned == 1 and "KEEP" in e._agent_stocks and "PRUNE" not in e._agent_stocks
    print(f"  Pruned {pruned}, kept: {list(e._agent_stocks.keys())}")

def t11():
    from core.intelligent_trader import get_aggressivity, AGGRESSIVITY_PROFILES
    for name in AGGRESSIVITY_PROFILES:
        p = get_aggressivity(name)
        assert "min_confidence" in p and "size_multiplier" in p
        print(f"  {name:12s}: conf={p['min_confidence']:.0%} size={p['size_multiplier']:.1f}x trades={p['max_trades_per_day']}")

def t12():
    from core.scanner import scan_multiple
    from core.risk import RiskManager
    rm = RiskManager()
    results = scan_multiple(["TSLA"], "5d", "5m", rm, max_workers=1, buying_power=50000)
    for r in results:
        if not r.error:
            print(f"  {r.ticker}: {r.action}@{r.confidence:.0%} ${r.price:.2f} SL=${r.stop_loss:.2f} TP=${r.take_profit:.2f}")
        else:
            print(f"  {r.ticker}: {r.error[:50]}")
    assert len(results) == 1

print("=" * 60)
print("STOCKY SUITE -- FULL END-TO-END TEST")
print("=" * 60)

test("Market Hours", t1)
test("Feature Engineering (31)", t2)
test("Triple Barrier Labels", t3)
test("LightGBM Train+Predict", t4)
test("Risk Manager (0.9x/5.5x)", t5)
test("RL Feedback Model", t6)
test("Regime Detection (4 states)", t7)
test("Post-Trade Reflection", t8)
test("Engine State Save/Restore", t9)
test("Stock Pruning", t10)
test("Aggressivity Profiles", t11)
test("Scanner Integration", t12)

print(f"\n{'=' * 60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'=' * 60}")
sys.exit(1 if failed else 0)
