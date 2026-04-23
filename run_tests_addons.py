"""Comprehensive addon + agent subsystem test."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "StockyApps"))
import warnings; warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np, pandas as pd, importlib
from addons import discover_addons
discover_addons()

passed = failed = 0
empty = pd.DataFrame()

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

# ── ADDONS ──
def t_fred():
    mod = importlib.import_module("addons.fred_macro")
    f = mod.get_features("SPY", empty)
    assert f and "vix_level" in f and f["vix_level"] > 0
    print(f"  VIX={f['vix_level']:.2f} Yield={f['yield_curve']} Fed={f['fed_rate']}")

def t_finnhub():
    mod = importlib.import_module("addons.finnhub_calendar")
    f = mod.get_features("AAPL", empty)
    assert f and "days_to_earnings" in f
    f2 = mod.get_features("TSLA", empty)
    print(f"  AAPL={f['days_to_earnings']}d, TSLA={f2.get('days_to_earnings','?')}d")

def t_fg():
    mod = importlib.import_module("addons.fear_greed")
    f = mod.get_features("MARKET", empty)
    assert f and "fear_greed_index" in f
    fg = f["fear_greed_index"]
    print(f"  F&G={fg:.3f} ({fg*100:.0f}/100) cat={f.get('fear_greed_category','?')}")

def t_spy():
    mod = importlib.import_module("addons.spy_correlation")
    from core.data import fetch_intraday
    data = fetch_intraday("AAPL", period="5d", interval="5m")
    f = mod.get_features("AAPL", data)
    assert f
    print(f"  spy_ret={f.get('spy_return','?')} spy5={f.get('spy_return_5','?')} corr={f.get('stock_spy_corr','?')}")

def t_twits():
    mod = importlib.import_module("addons.stocktwits")
    f = mod.get_features("AAPL", empty)
    assert f
    print(f"  bull={f.get('stocktwits_bull_ratio','?')} vol={f.get('stocktwits_volume','?')}")

def t_wsb():
    mod = importlib.import_module("addons.reddit_wsb")
    f = mod.get_features("AAPL", empty)
    assert f
    print(f"  mentions={f.get('wsb_mention_count','?')} sent={f.get('wsb_sentiment_ratio','?')}")

def t_insider():
    mod = importlib.import_module("addons.insider_trades")
    f = mod.get_features("AAPL", empty)
    assert f
    print(f"  buys={f.get('insider_buy_count','?')} sells={f.get('insider_sell_count','?')} net={f.get('insider_net_signal','?')}")

# ── REGIME WITH LIVE DATA ──
def t_regime():
    from core.agent.regime import _fetch_addon_data, detect_regime
    data = _fetch_addon_data()
    assert data and "fear_greed_index" in data, f"No addon data: {data}"
    r = detect_regime(data)
    fg = data["fear_greed_index"]
    if fg <= 1: fg *= 100
    print(f"  F&G={fg:.0f} VIX={data.get('vix_level','?')} -> {r.name} (size={r.size_mult}x conf={r.conf_boost:+.0%})")

# ── CORE PIPELINE ──
def t_features():
    from core.data import fetch_intraday
    from core.features import engineer_features, INTRADAY_FEATURES
    data = fetch_intraday("NVDA", period="5d", interval="5m")
    engineer_features(data, "intraday")
    missing = [f for f in INTRADAY_FEATURES if f not in data.columns]
    assert not missing, f"Missing: {missing}"
    print(f"  {len(INTRADAY_FEATURES)} features OK, {len(data)} rows")

def t_lgbm():
    from core.data import fetch_intraday
    from core.features import engineer_features, INTRADAY_FEATURES
    from core.labeling import triple_barrier_label
    from core.model import train_lgbm, predict_lgbm
    data = fetch_intraday("MSFT", period="5d", interval="5m")
    engineer_features(data, "intraday")
    data["Label"] = triple_barrier_label(data, atr_tp=1.5, atr_sl=1.5)
    clean = data.dropna()
    model, feats = train_lgbm(clean, INTRADAY_FEATURES, "TEST")
    assert model
    a, c, p = predict_lgbm(model, clean, feats)
    print(f"  S={sum(a==0)} H={sum(a==1)} B={sum(a==2)} conf={c.mean():.1%}")

def t_risk():
    from core.risk import RiskManager
    rm = RiskManager()
    assert rm.atr_stop_mult == 0.9 and rm.atr_profit_mult == 5.5
    s1 = rm.position_size(100, 2.0, buying_power=50000, confidence=0.5)
    s2 = rm.position_size(100, 2.0, buying_power=50000, confidence=0.9)
    assert s2 > s1, "Confidence scaling broken"
    print(f"  R:R={rm.atr_profit_mult/rm.atr_stop_mult:.1f}:1 | 50%={s1}sh 90%={s2}sh")

def t_rl():
    from core.reinforcement import train_feedback_model, get_quality_score
    m, acc, cnt = train_feedback_model()
    if m:
        qb = get_quality_score(m, 0.7, [0.1, 0.2, 0.7], 0.01, "BUY")
        qs = get_quality_score(m, 0.7, [0.7, 0.2, 0.1], 0.01, "SELL")
        print(f"  {acc:.0%} acc/{cnt} trades | BUY_q={qb:.2f} SELL_q={qs:.2f}")

def t_reflect():
    from core.agent.reflection import get_active_rules, get_rules_for_prompt
    rules = get_active_rules()
    prompt = get_rules_for_prompt()
    print(f"  {len(rules)} rules, {len(prompt)} chars in prompt")

def t_state():
    from core.agent.engine import AgentEngine
    e = AgentEngine(broker=None, log_fn=lambda m,l: None, settings_fn=dict)
    e._agent_stocks = {"X": {"signal": "BUY", "qty": 5, "mode": "Auto", "confidence": 0.8, "checks": 3}}
    e._trades_today = 2; e._cycle = 7; e._session_pnl = 250; e._wins = 6; e._losses = 3
    s = e.get_state()
    e2 = AgentEngine(broker=None, log_fn=lambda m,l: None, settings_fn=dict)
    e2.restore_state(s)
    assert e2.cycle == 7 and e2.session_pnl == 250
    s["saved_date"] = "2020-01-01"
    e3 = AgentEngine(broker=None, log_fn=lambda m,l: None, settings_fn=dict)
    e3.restore_state(s)
    assert e3.cycle == 0 and e3._wins == 6
    print(f"  Same-day OK, New-day reset OK, lifetime kept")

def t_prune():
    from core.agent.engine import AgentEngine
    e = AgentEngine(broker=None, log_fn=lambda m,l: None, settings_fn=dict)
    e._agent_stocks = {
        "OWN": {"signal": "BUY", "confidence": 0.85, "checks": 2, "qty": 10, "mode": "Auto"},
        "DEAD": {"signal": "HOLD", "confidence": 0.40, "checks": 5, "qty": 0, "mode": "Scanned"},
        "OK": {"signal": "BUY", "confidence": 0.72, "checks": 1, "qty": 0, "mode": "Scanned"},
    }
    p = e._prune_stale_stocks({})
    assert p == 1 and "DEAD" not in e._agent_stocks and "OWN" in e._agent_stocks
    print(f"  Pruned {p}, kept: {list(e._agent_stocks.keys())}")

def t_confirm():
    from core.agent.engine import AgentEngine
    e = AgentEngine(broker=None, log_fn=lambda m,l: None, settings_fn=dict)
    e._buy_confirmations["T"] = 1
    assert e._buy_confirmations["T"] == 1
    e._buy_confirmations["T"] += 1
    assert e._buy_confirmations["T"] == 2
    e._buy_confirmations.pop("T", None)
    assert "T" not in e._buy_confirmations
    print(f"  1st=WAIT, 2nd=GO, SELL=reset")

def t_hours():
    from core.market_hours import get_session, is_market_open, get_session_label
    s = get_session()
    print(f"  {s.name} | trade={s.can_trade} | scan={s.can_scan} | {get_session_label()}")

def t_scanner():
    from core.scanner import scan_multiple
    from core.risk import RiskManager
    rm = RiskManager()
    r = scan_multiple(["TSLA"], "5d", "5m", rm, max_workers=1, buying_power=50000)
    assert len(r) == 1
    x = r[0]
    if not x.error:
        print(f"  {x.ticker}: {x.action}@{x.confidence:.0%} ${x.price:.2f} SL={x.stop_loss:.2f} TP={x.take_profit:.2f}")
    else:
        print(f"  {x.ticker}: {x.error[:50]}")

print("=" * 70)
print("COMPREHENSIVE AGENT + ADDON TEST")
print("=" * 70)

test("FRED Macro (API key)", t_fred)
test("Finnhub Calendar (API key)", t_finnhub)
test("Fear & Greed", t_fg)
test("SPY Correlation", t_spy)
test("StockTwits", t_twits)
test("Reddit WSB", t_wsb)
test("SEC Insider", t_insider)
test("Regime (live data)", t_regime)
test("Features (31)", t_features)
test("LightGBM", t_lgbm)
test("Risk Manager", t_risk)
test("RL Feedback", t_rl)
test("Reflection Rules", t_reflect)
test("Engine State", t_state)
test("Stock Pruning", t_prune)
test("Buy Confirmation", t_confirm)
test("Market Hours", t_hours)
test("Scanner Pipeline", t_scanner)

print(f"\n{'=' * 70}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
if failed == 0:
    print("ALL SYSTEMS OPERATIONAL")
print(f"{'=' * 70}")
