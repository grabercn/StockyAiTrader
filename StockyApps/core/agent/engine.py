# -*- coding: utf-8 -*-
"""
AgentEngine — autonomous trading pipeline.

7-phase loop:
    1. Gather context (account, positions, BP)
    2. Discover tickers (active, gainers, trending, held)
    3. Scan + RL adjust (LightGBM + feedback model)
    4. Pre-filter (drop untradeable, unheld sells)
    5. Split: sells first, then buys
    6. Execute with Gemini advisory + cycle context
    7. Dynamic wait based on activity

Reusable: no Qt dependencies in the core loop.
All UI communication via callbacks (log_fn, on_tray_update, etc.).
"""

import time
import threading
from datetime import datetime


class AgentEngine:
    """Autonomous trading agent engine — decoupled from UI."""

    def __init__(self, broker, log_fn, settings_fn):
        """
        Args:
            broker:      AlpacaBroker instance
            log_fn:      callable(msg: str, level: str) — emit log messages
            settings_fn: callable() -> dict — load current settings
        """
        self.broker = broker
        self._log = log_fn
        self._settings = settings_fn

        # State
        self._running = False
        self._agent_stocks = {}      # {ticker: {signal, confidence, price, ...}}
        self._countdown = 0
        self._cycle = 0
        self._trades_today = 0
        self._phase = "idle"         # Current pipeline phase for transparency
        self._phase_detail = ""      # Extra detail about current phase
        self._cycle_decisions = []   # Decisions made this cycle
        self._cycle_stats = {}       # Stats for current cycle
        self._tradeable_cache = {}   # {ticker: bool} persists across cycles

        # P&L tracking (persists across cycles within session, resets daily)
        self._session_pnl = 0.0      # Realized P&L this session
        self._trade_log = []         # [{ticker, side, qty, price, pnl, timestamp}, ...]
        self._wins = 0
        self._losses = 0

        # Buy confirmation tracking: require 2+ consecutive BUY signals before executing
        self._buy_confirmations = {}  # {ticker: consecutive_buy_count}

        # Callbacks (set by panel)
        self.on_tray_update = None   # callable(**kwargs)
        self.on_tray_action = None   # callable(text)
        self.on_add_monitor = None   # callable(ticker) — add to auto-service

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def running(self):
        return self._running

    @property
    def agent_stocks(self):
        return self._agent_stocks

    @agent_stocks.setter
    def agent_stocks(self, val):
        self._agent_stocks = dict(val) if val else {}

    @property
    def countdown(self):
        return self._countdown

    @property
    def cycle(self):
        return self._cycle

    @property
    def trades_today(self):
        return self._trades_today

    @property
    def phase(self):
        return self._phase

    @property
    def phase_detail(self):
        return self._phase_detail

    @property
    def cycle_decisions(self):
        return list(self._cycle_decisions)

    @property
    def cycle_stats(self):
        return dict(self._cycle_stats)

    @property
    def session_pnl(self):
        return self._session_pnl

    @property
    def win_rate(self):
        total = self._wins + self._losses
        return self._wins / total if total > 0 else 0.0

    @property
    def trade_log(self):
        return list(self._trade_log)

    # ── State Persistence ─────────────────────────────────────────────

    def get_state(self):
        """Export full resumable state as a dict (for saving on close)."""
        return {
            "agent_stocks": dict(self._agent_stocks),
            "cycle": self._cycle,
            "trades_today": self._trades_today,
            "tradeable_cache": dict(self._tradeable_cache),
            "cycle_decisions": list(self._cycle_decisions),
            "cycle_stats": dict(self._cycle_stats),
            "phase": self._phase,
            "regime": self._regime.name if self._regime else None,
            "regime_detail": self._regime.description if self._regime else None,
            "pdt_restricted": getattr(self, '_last_pdt', False),
            "last_bp": getattr(self, '_last_bp', 0),
            "last_dt_bp": getattr(self, '_last_dt_bp', 0),
            "saved_date": datetime.now().strftime("%Y-%m-%d"),
            "session_pnl": self._session_pnl,
            "trade_log": self._trade_log[-20:],  # Keep last 20 trades
            "wins": self._wins,
            "losses": self._losses,
        }

    def restore_state(self, state):
        """Restore state from a previous session. Resets daily counters if date changed."""
        if not state:
            return

        saved_date = state.get("saved_date", "")
        today = datetime.now().strftime("%Y-%m-%d")
        is_new_day = (saved_date != today)

        # Always restore: stocks, cache, decisions (carry across days)
        self._agent_stocks = state.get("agent_stocks", {})
        self._tradeable_cache = state.get("tradeable_cache", {})
        self._cycle_decisions = state.get("cycle_decisions", [])
        self._cycle_stats = state.get("cycle_stats", {})

        # Always restore cumulative P&L history (carries across days)
        self._trade_log = state.get("trade_log", [])
        self._wins = state.get("wins", 0)
        self._losses = state.get("losses", 0)

        if is_new_day:
            # New day: reset daily counters, keep stocks + cache + cumulative P&L
            self._cycle = 0
            self._trades_today = 0
            self._session_pnl = 0.0  # Daily P&L resets
            total = self._wins + self._losses
            wr = f"{self._wins}/{total} ({self.win_rate:.0%})" if total > 0 else "no trades yet"
            self._log(f"  New trading day (saved {saved_date}) — daily counters reset, lifetime: {wr}", "agent")
        else:
            # Same day: restore everything
            self._cycle = state.get("cycle", 0)
            self._trades_today = state.get("trades_today", 0)
            self._session_pnl = state.get("session_pnl", 0.0)

    # ── Control ─────────────────────────────────────────────────────────

    def start(self):
        """Start the agent loop in a daemon thread."""
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        """Signal the agent to stop after the current phase completes."""
        self._running = False

    def _set_phase(self, phase, detail=""):
        """Update current phase for transparency display."""
        self._phase = phase
        self._phase_detail = detail

    def _tray(self, **kwargs):
        if self.on_tray_update:
            try: self.on_tray_update(**kwargs)
            except: pass

    def _tray_act(self, text):
        if self.on_tray_action:
            try: self.on_tray_action(text)
            except: pass

    # ── Main Loop ───────────────────────────────────────────────────────

    def _run(self):
        from core.intelligent_trader import get_aggressivity
        from core.risk import RiskManager
        from core.profiles import get_optimal_workers
        from core.discovery import get_most_active, get_day_gainers, get_trending_social
        from core.scanner import scan_multiple
        from core.logger import log_event
        from core.agent.regime import detect_regime
        from core.agent.reflection import check_and_reflect, get_rules_for_prompt

        settings = self._settings()
        rm = RiskManager()
        profile_name = settings.get("aggressivity", "Default")
        profile = get_aggressivity(profile_name)
        base_min_conf = profile.get("min_confidence", 0.5)
        max_trades = profile.get("max_trades_per_day", 8)
        cycle = 0
        self._regime = None  # Current regime state (for transparency)

        # RL feedback model
        rl_model = None
        try:
            from core.reinforcement import train_feedback_model, get_quality_score
            rl_model, rl_acc, rl_count = train_feedback_model()
            if rl_model:
                self._log(f"Agent: RL model loaded ({rl_count} trades, {rl_acc:.0%} accuracy)", "rl")
        except Exception:
            pass
        rl_retrain_interval = 5

        self._trades_today = 0
        log_event("agent", f"Agent started — profile: {profile_name}, min_conf: {base_min_conf:.0%}")

        while self._running:
            cycle += 1
            self._cycle = cycle
            wait_secs = 300

            try:
                self._log(f"Agent cycle {cycle}: scanning market...", "agent")
                self._tray(cycle=cycle)

                # ═════════════════════════════════════════════════════════
                # PHASE 1: GATHER CONTEXT
                # ═════════════════════════════════════════════════════════
                self._set_phase("context", "Fetching account and positions...")
                acct, effective_bp, dt_bp, held_map, positions_raw, pdt_restricted = self._gather_context()
                initial_bp = effective_bp

                if self._trades_today >= max_trades:
                    self._log(f"Agent: max trades/day ({max_trades}) reached. Waiting.", "warn")
                    self._set_phase("waiting", "Daily trade limit reached")
                    for _ in range(300):
                        if not self._running: break
                        time.sleep(1)
                    continue

                pdt_tag = " [PDT RESTRICTED]" if pdt_restricted else ""
                self._log(
                    f"Agent: BP=${effective_bp:,.0f}{pdt_tag}, "
                    f"{len(held_map)} positions, profile={profile_name}", "agent")
                if pdt_restricted:
                    self._log("  PDT restriction: buying with GTC only (swing trades, hold overnight)", "warn")

                # ── Regime Detection ──
                self._set_phase("context", "Detecting market regime...")
                regime = detect_regime(log_fn=self._log)
                self._regime = regime
                min_conf = min(0.95, base_min_conf + regime.conf_boost)
                self._log(
                    f"  Min confidence: {base_min_conf:.0%} + {regime.conf_boost:+.0%} "
                    f"(regime) = {min_conf:.0%}", "agent")

                # ── Post-Trade Reflection ──
                if held_map:
                    self._set_phase("context", "Reflecting on positions...")
                    try:
                        new_rules = check_and_reflect(
                            held_map, self._agent_stocks, [], log_fn=self._log)
                    except Exception:
                        pass

                # ═════════════════════════════════════════════════════════
                # PHASE 2: DISCOVER TICKERS
                # ═════════════════════════════════════════════════════════
                self._set_phase("discovery", "Finding stocks to analyze...")
                tickers, sources_used = self._discover_tickers(
                    held_map, settings, get_most_active, get_day_gainers, get_trending_social)

                self._log(
                    f"Agent: scanning {len(tickers)} tickers from {', '.join(sources_used)}", "scan")
                self._tray(scanned=len(tickers))

                if not tickers:
                    self._log("Agent: no tickers, waiting 2 min...", "warn")
                    self._set_phase("waiting", "No tickers found")
                    for _ in range(120):
                        if not self._running: break
                        time.sleep(1)
                    continue

                # ═════════════════════════════════════════════════════════
                # PHASE 3: SCAN + RL ADJUST
                # ═════════════════════════════════════════════════════════
                self._set_phase("scanning", f"Analyzing {len(tickers)} stocks...")
                results = scan_multiple(tickers, "5d", "5m", rm,
                    max_workers=get_optimal_workers(), buying_power=effective_bp)
                self._tray(running=True)

                # Retrain RL every N cycles
                if cycle % rl_retrain_interval == 0 and cycle > 0:
                    try:
                        rl_model, rl_acc, rl_count = train_feedback_model()
                        if rl_model:
                            self._log(f"  RL retrained: {rl_count} trades, {rl_acc:.0%} accuracy", "rl")
                    except: pass

                # Apply RL quality scores
                self._set_phase("rl_adjust", "Applying RL feedback scores...")
                if rl_model:
                    for r in results:
                        if not r.error:
                            try:
                                atr_pct = r.atr / r.price if r.price > 0 else 0
                                q = get_quality_score(rl_model, r.confidence, r.probs, atr_pct, r.action)
                                r.confidence = min(1.0, r.confidence * q)
                            except: pass

                # ═════════════════════════════════════════════════════════
                # PHASE 4: PRE-FILTER
                # ═════════════════════════════════════════════════════════
                self._set_phase("filtering", "Pre-filtering signals...")
                valid = self._pre_filter(results, held_map)

                # ═════════════════════════════════════════════════════════
                # PHASE 5: SPLIT + PREPARE
                # ═════════════════════════════════════════════════════════
                self._set_phase("preparing", "Sorting signals: sells first, then buys...")
                sell_candidates = sorted(
                    [r for r in valid if r.action == "SELL"], key=lambda x: -x.confidence)
                buy_candidates = sorted(
                    [r for r in valid if r.action == "BUY"], key=lambda x: -x.confidence)
                hold_candidates = [r for r in valid if r.action == "HOLD"]

                # Pre-load Gemini + addon data once
                use_gemini, addon_signals = self._load_gemini_context()

                # Reset cycle tracking
                self._cycle_decisions = []
                buys, sells, holds, skipped = 0, 0, 0, 0

                # ═════════════════════════════════════════════════════════
                # PHASE 6A: EXECUTE SELLS (free capital first)
                # ═════════════════════════════════════════════════════════
                self._set_phase("selling", f"Processing {len(sell_candidates)} sell signals...")
                for r in sell_candidates:
                    if not self._running or self._trades_today >= max_trades:
                        break

                    self._buy_confirmations.pop(r.ticker, None)  # SELL breaks BUY confirmation
                    self._apply_gemini(r, use_gemini, addon_signals, held_map, effective_bp, dt_bp, pdt_restricted)
                    self._update_stock_entry(r)

                    if r.confidence < min_conf:
                        skipped += 1
                        self._log(f"    SKIP SELL {r.ticker} — {r.confidence:.0%} below threshold", "decision")
                        continue

                    sells += 1
                    pos = held_map.get(r.ticker.upper())
                    if not pos:
                        continue
                    held = int(float(pos.get("qty", 0)))
                    if held <= 0:
                        continue

                    qty = held if r.confidence > 0.7 else max(1, int(held * 0.5)) if r.confidence > 0.5 else max(1, int(held * 0.25))
                    try:
                        result = self.broker.close_position(r.ticker, qty=qty)
                        if "error" not in result:
                            self._trades_today += 1
                            sell_price = float(pos.get("current_price", r.price))
                            freed = qty * sell_price
                            effective_bp += freed

                            # Track realized P&L
                            entry = float(pos.get("avg_entry_price", r.price))
                            trade_pnl = (sell_price - entry) * qty
                            self._session_pnl += trade_pnl
                            if trade_pnl >= 0:
                                self._wins += 1
                            else:
                                self._losses += 1
                            self._trade_log.append({
                                "ticker": r.ticker, "side": "sell", "qty": qty,
                                "entry": entry, "exit": sell_price,
                                "pnl": round(trade_pnl, 2),
                                "timestamp": datetime.now().isoformat(),
                            })

                            pnl_tag = f" P&L ${trade_pnl:+,.2f}" if entry > 0 else ""
                            wr = f" | Session: ${self._session_pnl:+,.2f} ({self.win_rate:.0%} WR)"
                            self._log(
                                f"    EXECUTE SELL {r.ticker} x{qty}/{held} "
                                f"({r.confidence:.0%}) — freed ${freed:,.0f}{pnl_tag}{wr}", "trade")
                            self._tray(sells=self._trades_today)
                            self._tray_act(f"SELL {r.ticker} x{qty} P&L ${trade_pnl:+,.2f}")
                            self._cycle_decisions.append(f"SOLD {r.ticker} x{qty} (${trade_pnl:+,.0f})")
                            self._agent_stocks[r.ticker]["qty"] = held - qty
                            self._agent_stocks[r.ticker]["mode"] = "Auto"
                            if qty >= held:
                                held_map.pop(r.ticker.upper(), None)
                        else:
                            self._log(f"    SELL {r.ticker} FAILED — {result.get('error','')}", "error")
                            self._tray_act(f"SELL {r.ticker} FAILED")
                            self._tray(last_action="error")
                    except Exception as _e:
                        self._log(f"    SELL {r.ticker} ERROR — {_e}", "error")

                # Refresh BP after sells
                if sells > 0 and self.broker:
                    try:
                        acct = self.broker.get_account()
                        effective_bp = min(
                            float(acct.get("buying_power", 0)),
                            float(acct.get("daytrading_buying_power", 0)) or float(acct.get("buying_power", 0)))
                        self._log(f"  Post-sell BP: ${effective_bp:,.0f}", "agent")
                    except: pass

                # ═════════════════════════════════════════════════════════
                # PHASE 6B: EXECUTE BUYS (with freed capital)
                # ═════════════════════════════════════════════════════════
                self._set_phase("buying", f"Processing {len(buy_candidates)} buy signals...")
                for r in buy_candidates:
                    if not self._running or self._trades_today >= max_trades:
                        break

                    self._apply_gemini(r, use_gemini, addon_signals, held_map, effective_bp, dt_bp, pdt_restricted)
                    self._update_stock_entry(r)

                    if r.confidence < min_conf:
                        skipped += 1
                        self._buy_confirmations.pop(r.ticker, None)  # Reset confirmation
                        self._log(f"    SKIP BUY {r.ticker} — {r.confidence:.0%} below threshold", "decision")
                        continue

                    # Buy confirmation: require 2+ consecutive BUY signals before executing
                    # (backtest showed this improves P&L by +$463 and win rate by +8%)
                    self._buy_confirmations[r.ticker] = self._buy_confirmations.get(r.ticker, 0) + 1
                    confirms = self._buy_confirmations[r.ticker]
                    if confirms < 2:
                        self._log(
                            f"    WAIT BUY {r.ticker} — first scan ({r.confidence:.0%}), "
                            f"need 1 more confirmation", "decision")
                        continue

                    # PDT: skip buying stocks we already hold (adding would create round-trip risk)
                    if pdt_restricted and r.ticker.upper() in held_map:
                        self._log(
                            f"    SKIP BUY {r.ticker} — PDT restricted, already holding "
                            f"(adding shares risks day-trade violation)", "warn")
                        continue

                    buys += 1

                    # Capital rotation if BP too low
                    if effective_bp < 100:
                        rotate_threshold = 0.40 if profile_name == "YOLO" else 0.60 if profile_name == "Aggressive" else 0.80
                        if r.confidence >= rotate_threshold and held_map:
                            effective_bp = self._rotate_capital(
                                r, held_map, effective_bp, profile_name)
                            if effective_bp < 100:
                                continue
                        else:
                            self._log(
                                f"    BUY {r.ticker} — BP ${effective_bp:,.0f} too low, "
                                f"need >{rotate_threshold:.0%} conf ({r.confidence:.0%})", "warn")
                            continue

                    # Size and execute
                    # Regime-aware + confidence-scaled position sizing
                    # (backtest: confidence scaling improved P&L from +2.9% to +9.4%)
                    base_alloc = min(effective_bp * 0.20, initial_bp / max(1, 5))
                    conf_scale = 0.7 + (r.confidence * 0.6)  # 50% conf=1.0x, 100% conf=1.3x
                    max_spend = base_alloc * regime.size_mult * conf_scale
                    qty = max(1, int(max_spend / r.price)) if r.price > 0 else 0
                    cost = qty * r.price

                    if qty <= 0 or cost > effective_bp:
                        self._log(f"    BUY {r.ticker} — cost ${cost:,.0f} > BP ${effective_bp:,.0f}", "warn")
                        continue

                    try:
                        # Apply regime-adjusted stop/take-profit
                        adj_sl = r.stop_loss
                        adj_tp = r.take_profit
                        if r.stop_loss and r.atr > 0:
                            sl_dist = abs(r.price - r.stop_loss)
                            adj_sl = r.price - (sl_dist * regime.stop_mult)
                            tp_dist = abs(r.take_profit - r.price) if r.take_profit else sl_dist * 2
                            adj_tp = r.price + (tp_dist * regime.profit_mult)

                        # Order strategy depends on PDT status
                        if pdt_restricted:
                            # PDT restricted: use GTC limit order slightly above market
                            # to avoid day-trading BP check while getting a fill
                            limit_px = round(r.price * 1.005, 2)  # 0.5% above market
                            result = self.broker.place_order(r.ticker, qty, "buy",
                                order_type="limit", limit_price=limit_px,
                                time_in_force="gtc")
                            used_bracket = False
                        else:
                            # Normal: try bracket order first
                            result = self.broker.place_order(r.ticker, qty, "buy",
                                stop_loss=adj_sl, take_profit=adj_tp)
                            used_bracket = True
                            if "error" in result:
                                result = self.broker.place_order(r.ticker, qty, "buy")
                                used_bracket = False
                        if "error" not in result:
                            self._trades_today += 1
                            effective_bp -= cost
                            bracket_tag = "" if used_bracket else " [no bracket]"
                            self._log(
                                f"    EXECUTE BUY {r.ticker} x{qty} @ ${r.price:.2f} "
                                f"(${cost:,.0f}, {r.confidence:.0%}){bracket_tag}", "trade")
                            if adj_sl and used_bracket:
                                self._log(
                                    f"      SL=${adj_sl:.2f} TP=${adj_tp:.2f} "
                                    f"(R:R {abs(adj_tp-r.price)/abs(r.price-adj_sl):.1f}:1)", "system")
                            self._tray(buys=self._trades_today)
                            self._tray_act(f"BUY {r.ticker} x{qty} @ ${r.price:.2f}")
                            self._cycle_decisions.append(f"BOUGHT {r.ticker} x{qty}")
                            self._trade_log.append({
                                "ticker": r.ticker, "side": "buy", "qty": qty,
                                "entry": r.price, "exit": None, "pnl": None,
                                "timestamp": datetime.now().isoformat(),
                            })
                            self._agent_stocks[r.ticker]["qty"] = qty
                            self._agent_stocks[r.ticker]["mode"] = "Auto"
                            self._agent_stocks[r.ticker]["entry_price"] = r.price
                            self._agent_stocks[r.ticker]["stop_loss"] = adj_sl
                            self._agent_stocks[r.ticker]["take_profit"] = adj_tp
                            self._agent_stocks[r.ticker]["has_bracket"] = used_bracket
                            if self.on_add_monitor:
                                try: self.on_add_monitor(r.ticker)
                                except: pass
                        else:
                            self._log(f"    BUY {r.ticker} FAILED — {result.get('error','')}", "error")
                            self._tray_act(f"BUY {r.ticker} FAILED")
                            self._tray(last_action="error")
                    except Exception as _e:
                        self._log(f"    BUY {r.ticker} ERROR — {_e}", "error")

                # Phase 6C: Update holds
                for r in hold_candidates:
                    self._update_stock_entry(r)
                    holds += 1

                # ═════════════════════════════════════════════════════════
                # PHASE 7: SUMMARY + DYNAMIC TIMING
                # ═════════════════════════════════════════════════════════
                filtered_out = len(results) - len(valid)
                self._cycle_stats = {
                    "buys": buys, "sells": sells, "holds": holds,
                    "skipped": skipped, "filtered": filtered_out,
                    "trades_today": self._trades_today, "bp": effective_bp,
                    "regime": regime.name, "regime_detail": regime.description,
                    "min_conf": min_conf,
                    "session_pnl": self._session_pnl,
                    "win_rate": self.win_rate,
                    "wins": self._wins, "losses": self._losses,
                }
                self._set_phase("complete",
                    f"{buys}B {sells}S {holds}H ({skipped} skipped, {filtered_out} filtered)")

                self._log(
                    f"Cycle {cycle}: {buys}B {sells}S {holds}H "
                    f"({skipped} skipped, {filtered_out} filtered)", "agent")

                wait_secs = int(self._calc_wait(buys, sells, profile_name) * regime.scan_mult)
                self._log(
                    f"Cycle {cycle} done. {self._trades_today}/{max_trades} trades. "
                    f"Next in {wait_secs/60:.1f} min.", "system")

            except Exception as e:
                self._log(f"Agent error: {e}", "error")
                import traceback; traceback.print_exc()
                wait_secs = 300
                self._set_phase("error", str(e))

            # Countdown
            self._set_phase("waiting", f"Next cycle in {wait_secs}s")
            self._countdown = wait_secs
            for _ in range(wait_secs):
                if not self._running:
                    break
                time.sleep(1)
                self._countdown = max(0, self._countdown - 1)

        # Cleanup
        self._countdown = 0
        self._set_phase("idle")
        self._log("Autonomous agent stopped", "agent")
        log_event("agent", f"Agent stopped after {cycle} cycles, {self._trades_today} trades")

    # ── Phase Helpers ───────────────────────────────────────────────────

    def _gather_context(self):
        """Phase 1: Single snapshot of account + positions."""
        acct = {}
        bp = 0
        dt_bp = 0
        held_map = {}
        positions_raw = []

        if self.broker:
            try:
                acct = self.broker.get_account()
                bp = float(acct.get("buying_power", 0))
                dt_bp = float(acct.get("daytrading_buying_power", bp))
            except Exception:
                pass
            try:
                positions_raw = self.broker.get_positions()
                if not isinstance(positions_raw, list):
                    positions_raw = []
                for p in positions_raw:
                    sym = p.get("symbol", "").upper()
                    if sym:
                        held_map[sym] = p
            except Exception:
                pass

        # DT_BP=0 means PDT restricted — cannot do intraday round trips.
        # Regular BP can still be used for swing trades (hold overnight).
        effective_bp = bp
        pdt_restricted = (dt_bp <= 0)

        # Store for state export
        self._last_bp = bp
        self._last_dt_bp = dt_bp
        self._last_pdt = pdt_restricted
        return acct, effective_bp, dt_bp, held_map, positions_raw, pdt_restricted

    def _discover_tickers(self, held_map, settings, get_most_active, get_day_gainers, get_trending_social):
        """Phase 2: Gather tickers from multiple sources."""
        tickers = set()
        sources = []

        try:
            ma = get_most_active(10)
            tickers.update(ma)
            sources.append(f"Active({len(ma)})")
        except: pass
        try:
            g = get_day_gainers(5)
            tickers.update(g)
            sources.append(f"Gainers({len(g)})")
        except: pass
        try:
            t = get_trending_social(5)
            tickers.update(t)
            sources.append(f"Trending({len(t)})")
        except: pass

        # Always include held positions
        if held_map:
            tickers.update(held_map.keys())
            sources.append(f"Held({len(held_map)})")

        # Always include agent-bought stocks
        agent_auto = [t for t, s in self._agent_stocks.items() if s.get("mode") == "Auto"]
        if agent_auto:
            tickers.update(agent_auto)

        return list(tickers)[:25], sources

    def _pre_filter(self, results, held_map):
        """Phase 4: Remove untradeable, unheld sells, errored."""
        valid = []
        for r in results:
            if r.error:
                continue
            if r.action == "SELL" and r.ticker.upper() not in held_map:
                continue
            if r.action == "BUY" and self._tradeable_cache.get(r.ticker) is False:
                continue
            valid.append(r)

        # Batch tradability check for new BUY candidates
        for r in valid:
            if r.action == "BUY" and r.ticker not in self._tradeable_cache:
                try:
                    asset = self.broker._get(f"assets/{r.ticker}")
                    ok = "error" not in asset and asset.get("tradable", False)
                    self._tradeable_cache[r.ticker] = ok
                    if not ok:
                        self._log(f"  {r.ticker}: not tradeable on Alpaca, cached", "warn")
                except Exception:
                    self._tradeable_cache[r.ticker] = True

        return [r for r in valid if not (
            r.action == "BUY" and self._tradeable_cache.get(r.ticker) is False)]

    def _load_gemini_context(self):
        """Load Gemini availability and addon signals."""
        use_gemini = False
        addon_signals = {}
        try:
            from core.gemini_advisor import is_enabled as gemini_enabled
            use_gemini = gemini_enabled()
        except: pass
        if use_gemini:
            try:
                from addons import get_all_addons
                for addon in get_all_addons():
                    if addon.available and addon.enabled:
                        try:
                            feats = addon.get_features("MARKET")
                            if feats:
                                for k, v in list(feats.items())[:3]:
                                    addon_signals[f"{addon.name}_{k}"] = v
                        except: pass
            except: pass
        return use_gemini, addon_signals

    def _apply_gemini(self, r, use_gemini, addon_signals, held_map, effective_bp, dt_bp, pdt_restricted=False):
        """Apply Gemini advisory with full cycle context."""
        if not use_gemini or r.action not in ("BUY", "SELL"):
            return
        try:
            from core.gemini_advisor import get_advisory, apply_advisory

            pos_info = None
            pos = held_map.get(r.ticker.upper())
            if pos:
                pos_info = (
                    f"holding {float(pos.get('qty',0)):.0f} shares "
                    f"@ ${float(pos.get('avg_entry_price',0)):.2f} entry, "
                    f"now ${float(pos.get('current_price',0)):.2f}, "
                    f"P&L ${float(pos.get('unrealized_pl',0)):+,.2f}")
            elif r.ticker in self._agent_stocks:
                s = self._agent_stocks[r.ticker]
                pos_info = f"holding {s.get('qty',0)} shares, signal={s.get('signal','?')}"

            pdt_note = (" PDT RESTRICTED — cannot buy stocks already held (round-trip risk). "
                       "Only new positions allowed.") if pdt_restricted else ""
            # Build rich portfolio context with P&L history
            total_trades = self._wins + self._losses
            wr_str = f", win rate {self.win_rate:.0%} ({self._wins}W/{self._losses}L)" if total_trades > 0 else ""
            pnl_str = f", session P&L ${self._session_pnl:+,.2f}" if self._session_pnl != 0 else ""
            port_ctx = (
                f"BP=${effective_bp:,.0f},{pdt_note} "
                f"{self._trades_today} trades today, "
                f"{len(held_map)} positions held{pnl_str}{wr_str}")
            if self._cycle_decisions:
                port_ctx += f"\nThis cycle so far: {'; '.join(self._cycle_decisions[-5:])}"
            # Add regime context
            if self._regime:
                port_ctx += f"\nMarket regime: {self._regime.name} ({self._regime.description})"
            # Add learned reflection rules
            try:
                from core.agent.reflection import get_rules_for_prompt
                rules_text = get_rules_for_prompt()
                if rules_text:
                    port_ctx += f"\n{rules_text}"
            except: pass

            advisory = get_advisory(
                r.ticker, r.price, r.action, r.confidence, r.probs, r.atr,
                feature_importances=r.feature_importances,
                addon_data=addon_signals if addon_signals else None,
                position_info=pos_info,
                portfolio_context=port_ctx)

            if advisory:
                old_conf = r.confidence
                _, r.confidence = apply_advisory(r.action, r.confidence, advisory)
                conv = int(advisory.get("conviction", 3))
                gem_rec = advisory.get("recommendation", "?")
                gem_weight = conv / 5.0
                agree = "AGREE" if gem_rec == r.action else "DISAGREE"
                self._log(
                    f"  {r.ticker}: Local={r.action}({old_conf:.0%}) "
                    f"Gemini={gem_rec}[{conv}/5,w={gem_weight:.1f}] "
                    f"-> {r.confidence:.0%} ({agree})", "decision")
                self._log(f"    Reasoning: {advisory.get('reasoning','')}", "gemini")
        except: pass

    def _update_stock_entry(self, r):
        """Update agent_stocks dict for a scan result."""
        atr_pct = r.atr / r.price if r.price > 0 and r.atr > 0 else 0.01
        if atr_pct > 0.02:    dyn_interval, dyn_label = 120, "2m"
        elif atr_pct > 0.01:  dyn_interval, dyn_label = 300, "5m"
        elif atr_pct > 0.005: dyn_interval, dyn_label = 600, "10m"
        else:                 dyn_interval, dyn_label = 900, "15m"

        existing = self._agent_stocks.get(r.ticker, {})
        self._agent_stocks[r.ticker] = {
            "signal": r.action, "confidence": r.confidence,
            "price": r.price, "last_check": datetime.now().strftime("%H:%M:%S"),
            "checks": existing.get("checks", 0) + 1,
            "qty": existing.get("qty", 0),
            "mode": existing.get("mode", "Scanned"),
            "interval": dyn_label, "next_secs": dyn_interval,
        }

    def _rotate_capital(self, r, held_map, effective_bp, profile_name):
        """Sell weakest position to fund a buy. Returns updated BP."""
        self._log(f"    BP ${effective_bp:,.0f} low — rotating capital...", "agent")
        try:
            worst_sym, worst_pos = None, None
            worst_pct = 999
            for sym, p in held_map.items():
                if sym == r.ticker.upper():
                    continue
                pct = float(p.get("unrealized_plpc", 0))
                if pct < worst_pct:
                    worst_pct, worst_sym, worst_pos = pct, sym, p

            sell_ok = False
            if profile_name == "YOLO":
                sell_ok = True
            elif profile_name == "Aggressive":
                sell_ok = worst_pct < 0.01
            else:
                sell_ok = worst_pct < -0.005

            if worst_pos and sell_ok:
                w_qty = int(float(worst_pos.get("qty", 0)))
                sell_result = self.broker.close_position(worst_sym, qty=w_qty)
                if "error" not in sell_result:
                    self._trades_today += 1
                    freed = w_qty * float(worst_pos.get("current_price", 0))
                    w_pl = float(worst_pos.get("unrealized_pl", 0))
                    # Track rotation P&L
                    self._session_pnl += w_pl
                    if w_pl >= 0: self._wins += 1
                    else: self._losses += 1
                    self._trade_log.append({
                        "ticker": worst_sym, "side": "rotate_sell", "qty": w_qty,
                        "entry": float(worst_pos.get("avg_entry_price", 0)),
                        "exit": float(worst_pos.get("current_price", 0)),
                        "pnl": round(w_pl, 2),
                        "timestamp": datetime.now().isoformat(),
                    })
                    self._log(
                        f"    ROTATE: Sold {worst_sym} x{w_qty} (P&L ${w_pl:+,.0f}) — freed ${freed:,.0f}", "trade")
                    self._cycle_decisions.append(f"ROTATED {worst_sym} (${w_pl:+,.0f})")
                    held_map.pop(worst_sym, None)
                    time.sleep(1)
                    try:
                        acct = self.broker.get_account()
                        effective_bp = min(
                            float(acct.get("buying_power", 0)),
                            float(acct.get("daytrading_buying_power", 0)) or float(acct.get("buying_power", 0)))
                    except: pass
                    return effective_bp
                else:
                    self._log(f"    ROTATE {worst_sym} failed: {sell_result.get('error','')[:60]}", "error")
            else:
                self._log(f"    BUY {r.ticker} — no rotatable positions ({profile_name})", "warn")
        except Exception as _e:
            self._log(f"    Rotation error: {_e}", "error")
        return effective_bp

    def _calc_wait(self, buys, sells, profile_name):
        """Phase 7: Dynamic wait time based on activity."""
        base = 300
        if self._trades_today > 3:   base = int(base * 0.5)
        elif self._trades_today > 0: base = int(base * 0.75)
        strong = buys + sells
        if strong > 5:               base = int(base * 0.6)
        elif strong == 0:            base = int(base * 1.5)
        if profile_name == "YOLO":         base = int(base * 0.5)
        elif profile_name == "Aggressive": base = int(base * 0.7)
        elif profile_name == "Chill":      base = int(base * 1.5)
        return max(60, min(900, base))
