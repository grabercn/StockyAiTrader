"""AI Trading Dashboard — mission control + autonomous agent."""
import sys, os, json, time, threading
import numpy as np
from datetime import datetime, timedelta
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor
from core.branding import *
from core.branding import chart_colors
from core.event_bus import EventBus
from core.logger import log_event

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "settings.json")
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f: return json.load(f)
    except: return {}
def save_settings(s):
    with open(SETTINGS_FILE, "w") as f: json.dump(s, f, indent=4)


class AIDashboardPanel(QWidget):
    """Mission control + autonomous agent for the AI auto-trader."""

    _auto_svc = None
    _agent_stocks = {}  # {ticker: {signal, confidence, price, last_check, checks}}

    def __init__(self, broker, event_bus):
        super().__init__()
        self.broker = broker
        self.bus = event_bus

        # Restore agent tracked stocks from last session
        settings = load_settings()
        saved_tracked = settings.get("agent_tracked_stocks", {})
        if saved_tracked:
            self._agent_stocks = dict(saved_tracked)

        self._build()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start(5000)
        QTimer.singleShot(2000, self.refresh)

    def _build(self):
        from core.ui.backgrounds import GradientHeader
        from core.widgets import StatCard

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        header = GradientHeader("AI Agent", "Auto-trader mission control + autonomous agent")
        layout.addWidget(header)

        # Status cards
        cards = QHBoxLayout()
        cards.setSpacing(8)
        self.card_monitored = StatCard("Monitored", "0", BRAND_ACCENT)
        self.card_trades = StatCard("Trades Today", "0", BRAND_PRIMARY)
        self.card_signals = StatCard("Signals", "0 B / 0 S", BRAND_SECONDARY)
        self.card_mode = StatCard("Mode", "Idle", TEXT_MUTED)
        self.card_monitored.on_clicked = lambda: self._show_card_detail("monitored")
        self.card_trades.on_clicked = lambda: self._show_card_detail("trades")
        self.card_signals.on_clicked = lambda: self._show_card_detail("signals")
        self.card_mode.on_clicked = lambda: self._show_card_detail("mode")
        for c in [self.card_monitored, self.card_trades, self.card_signals, self.card_mode]:
            cards.addWidget(c)
        layout.addLayout(cards)

        # Autonomous agent controls — compact single row
        agent_row = QHBoxLayout()
        self.agent_start_btn = QPushButton("Start Agent")
        self.agent_start_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 11px; padding: 6px 14px;")
        self.agent_start_btn.clicked.connect(self._toggle_agent)
        agent_row.addWidget(self.agent_start_btn)
        self.agent_status = QLabel("Stopped")
        self.agent_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        agent_row.addWidget(self.agent_status)
        self.agent_countdown = QLabel("")
        self.agent_countdown.setStyleSheet(f"color: {BRAND_ACCENT}; font-size: 10px;")
        agent_row.addWidget(self.agent_countdown)
        agent_row.addStretch()
        layout.addLayout(agent_row)

        # Countdown timer
        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._tick_countdown)
        self._countdown_secs = 0

        # Options — single compact row
        settings = load_settings()
        opts = QHBoxLayout()
        self.manage_manual_cb = QCheckBox("Manage manual stocks")
        self.manage_manual_cb.setChecked(settings.get("manage_manual_stocks", False))
        self.manage_manual_cb.toggled.connect(self._toggle_manage_manual)
        opts.addWidget(self.manage_manual_cb)

        self.gemini_cb = QCheckBox("Gemini Advisor")
        self.gemini_cb.setChecked(settings.get("gemini_enabled", False))
        self.gemini_cb.toggled.connect(self._toggle_gemini)
        opts.addWidget(self.gemini_cb)
        self.gemini_key_input = QLineEdit(settings.get("gemini_api_key", ""))
        self.gemini_key_input.setPlaceholderText("API Key")
        self.gemini_key_input.setEchoMode(QLineEdit.Password)
        self.gemini_key_input.setFixedWidth(140)
        self.gemini_key_input.editingFinished.connect(self._save_gemini_key)
        opts.addWidget(self.gemini_key_input)
        opts.addStretch()
        layout.addLayout(opts)

        # Monitored stocks table with unmonitor buttons
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "Stock", "Mode", "Signal", "Conf", "Price",
            "Last Check", "Next", "Interval", "Checks", ""
        ])
        for c in range(9):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(9, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        # Activity log
        log_label = QLabel("Agent Activity Log")
        log_label.setFont(QFont(FONT_FAMILY, 10, QFont.Bold))
        log_label.setStyleSheet(f"color: {BRAND_PRIMARY};")
        layout.addWidget(log_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont(FONT_MONO, 9))
        self.log_area.setMaximumHeight(140)
        layout.addWidget(self.log_area)

        # Controls
        ctrl = QHBoxLayout()
        self.stop_all_btn = QPushButton("Stop All")
        self.stop_all_btn.setStyleSheet(f"background-color: {COLOR_SELL}; padding: 5px 12px;")
        self.stop_all_btn.clicked.connect(self._stop_all)
        ctrl.addWidget(self.stop_all_btn)
        ctrl.addStretch()
        ctrl.addWidget(QLabel("Auto-refreshes every 5s"))
        layout.addLayout(ctrl)

        self.setLayout(layout)
        self.bus.log_entry.connect(self._on_log)

    def _toggle_gemini(self, checked):
        settings = load_settings()
        settings["gemini_enabled"] = checked
        save_settings(settings)
        self.bus.log_entry.emit(f"Gemini advisor {'enabled' if checked else 'disabled'}", "system")

    def _save_gemini_key(self):
        settings = load_settings()
        settings["gemini_api_key"] = self.gemini_key_input.text()
        save_settings(settings)

    def _toggle_manage_manual(self, checked):
        settings = load_settings()
        settings["manage_manual_stocks"] = checked
        save_settings(settings)
        self.bus.log_entry.emit(
            f"AI {'will' if checked else 'will NOT'} manage manually-bought stocks", "system")

        # If enabled, add all current positions to monitoring
        if checked and self.broker:
            main = self.window()
            if main and hasattr(main, 'scanner'):
                svc_getter = getattr(main.scanner, '_get_auto_service', None)
                if svc_getter:
                    svc = svc_getter()
                    try:
                        positions = self.broker.get_positions()
                        if isinstance(positions, list):
                            added = 0
                            for p in positions:
                                sym = p.get("symbol", "")
                                if sym and not svc.is_monitoring(sym):
                                    svc.add_stock(sym, period="5d", interval="5m",
                                                  auto_execute=True, min_confidence=0.5)
                                    added += 1
                            if added:
                                self.bus.log_entry.emit(
                                    f"Added {added} manual positions to AI monitoring", "trade")
                    except Exception:
                        pass

    def _show_card_detail(self, card_type):
        """Show detail popup when stat card is clicked."""
        main = self.window()
        svc = None
        if main and hasattr(main, 'scanner') and hasattr(main.scanner, '_auto_service'):
            svc = main.scanner._auto_service

        monitored = svc.get_monitored() if svc else {}
        settings = load_settings()

        if card_type == "monitored":
            tickers = list(monitored.keys())
            text = f"Monitoring {len(tickers)} stocks:\n\n" + "\n".join(
                f"  {t}: {s.last_signal} ({s.last_confidence:.0%})" for t, s in monitored.items()
            ) if tickers else "No stocks monitored."
        elif card_type == "trades":
            total = sum(getattr(s, 'check_count', 0) for s in monitored.values())
            text = f"Total checks today: {total}\n\n" + "\n".join(
                f"  {t}: {getattr(s, 'check_count', 0)} checks" for t, s in monitored.items()
            )
        elif card_type == "signals":
            buys = [(t, s) for t, s in monitored.items() if s.last_signal == "BUY"]
            sells = [(t, s) for t, s in monitored.items() if s.last_signal == "SELL"]
            holds = [(t, s) for t, s in monitored.items() if s.last_signal == "HOLD"]
            text = (f"BUY ({len(buys)}): {', '.join(t for t,_ in buys) or 'none'}\n"
                    f"SELL ({len(sells)}): {', '.join(t for t,_ in sells) or 'none'}\n"
                    f"HOLD ({len(holds)}): {', '.join(t for t,_ in holds) or 'none'}")
        elif card_type == "mode":
            aggr = settings.get("aggressivity", "Default")
            from core.intelligent_trader import get_aggressivity
            p = get_aggressivity(aggr)
            text = (f"Aggressivity: {aggr}\n"
                    f"Min Confidence: {p['min_confidence']:.0%}\n"
                    f"Position Size: {p['size_multiplier']:.1f}x\n"
                    f"Max Trades/Day: {p['max_trades_per_day']}\n"
                    f"LLM: {'Yes' if p.get('use_llm') else 'No'}")
        else:
            text = "No data"

        QMessageBox.information(self, card_type.title(), text)

    def _on_log(self, msg, level):
        """Capture all trade/system/auto-related logs."""
        keywords = ["auto", "monitor", "signal", "check", "agent", "trade", "buy", "sell", "hold"]
        if level in ("trade", "system") or any(k in msg.lower() for k in keywords):
            ts = datetime.now().strftime("%H:%M:%S")
            colors = {"trade": BRAND_ACCENT, "system": TEXT_MUTED, "info": BRAND_PRIMARY,
                      "warn": COLOR_HOLD, "error": COLOR_SELL}
            c = colors.get(level, TEXT_SECONDARY)
            self.log_area.append(f'<span style="color:{TEXT_MUTED}">{ts}</span> <span style="color:{c}">{msg}</span>')
            sb = self.log_area.verticalScrollBar()
            sb.setValue(sb.maximum())

    def refresh(self):
        # Decrement countdowns every refresh (5 sec)
        for ticker, info in self._agent_stocks.items():
            ns = info.get("next_secs", 0)
            if ns > 0:
                info["next_secs"] = max(0, ns - 5)

        svc = self._get_svc()
        svc_monitored = svc.get_monitored() if svc else {}

        # Merge auto-service monitored + agent's own tracked stocks
        all_stocks = {}
        for ticker, stock in svc_monitored.items():
            all_stocks[ticker] = {
                "signal": getattr(stock, 'last_signal', 'HOLD'),
                "confidence": getattr(stock, 'last_confidence', 0),
                "price": getattr(stock, 'last_price', 0),
                "last_check": getattr(stock, 'last_check', '--'),
                "next_secs": getattr(stock, 'next_check_seconds', 0),
                "interval": getattr(stock, 'interval', '5m'),
                "checks": getattr(stock, 'check_count', 0),
                "mode": "Auto" if getattr(stock, 'auto_execute', False) else "Signal",
            }
        # Add agent-tracked stocks not already in auto-service
        for ticker, info in self._agent_stocks.items():
            if ticker not in all_stocks:
                all_stocks[ticker] = info

        # Add saved positions (from settings, no broker call — prevents freeze)
        settings_chk = load_settings()
        if settings_chk.get("manage_manual_stocks") or getattr(self, '_agent_running', False):
            saved_positions = settings_chk.get("agent_managed_positions", [])
            for p in saved_positions:
                sym = p.get("symbol", "")
                if sym and sym not in all_stocks:
                    all_stocks[sym] = {
                        "signal": "HOLD", "confidence": 0,
                        "price": float(p.get("current_price", 0)),
                        "last_check": "--", "next_secs": 0,
                        "interval": "managed", "checks": 0,
                        "mode": "Position",
                    }
            # Async refresh positions in background (updates saved data)
            if self.broker and not hasattr(self, '_pos_refreshing'):
                self._pos_refreshing = True
                import threading
                def _refresh_pos():
                    try:
                        positions = self.broker.get_positions()
                        if isinstance(positions, list):
                            settings2 = load_settings()
                            settings2["agent_managed_positions"] = [
                                {"symbol": p.get("symbol",""), "qty": p.get("qty","0"),
                                 "side": p.get("side",""), "current_price": p.get("current_price","0"),
                                 "unrealized_pl": p.get("unrealized_pl","0"),
                                 "avg_entry": p.get("avg_entry_price","0")}
                                for p in positions
                            ]
                            save_settings(settings2)
                    except: pass
                    self._pos_refreshing = False
                threading.Thread(target=_refresh_pos, daemon=True).start()

        if not all_stocks:
            self.card_monitored.set_value("0")
            self.card_mode.set_value("No agent" if not getattr(self, '_agent_running', False) else "Running")
            self.table.setRowCount(0)
            return

        monitored = all_stocks
        self.card_monitored.set_value(str(len(monitored)))

        settings = load_settings()
        aggr = settings.get("aggressivity", "Default")
        is_running = getattr(self, '_agent_running', False)
        self.card_mode.set_value(f"{aggr}" if is_running else aggr)

        buys = sum(1 for s in monitored.values() if s.get("signal") == "BUY")
        sells = sum(1 for s in monitored.values() if s.get("signal") == "SELL")
        holds = sum(1 for s in monitored.values() if s.get("signal") == "HOLD")
        self.card_signals.set_value(f"{buys}B {sells}S {holds}H")

        total_checks = sum(s.get("checks", 0) for s in monitored.values())
        self.card_trades.set_value(str(total_checks))

        # Table
        self.table.setRowCount(len(monitored))
        for i, (ticker, info) in enumerate(monitored.items()):
            signal = info.get("signal", "HOLD")
            conf = info.get("confidence", 0)
            price = info.get("price", 0)
            last_check = info.get("last_check", "--")
            next_secs = info.get("next_secs", 0)
            interval = info.get("interval", "5m")
            checks = info.get("checks", 0)
            mode = info.get("mode", "Auto")

            mins = next_secs // 60 if isinstance(next_secs, (int, float)) else 0
            secs = next_secs % 60 if isinstance(next_secs, (int, float)) else 0

            next_str = f"{mins}m{secs:02d}s" if next_secs > 0 else "scanning..."
            vals = [ticker, mode, signal, f"{conf:.0%}",
                    f"${price:.2f}" if price else "--",
                    str(last_check) if last_check else "--",
                    next_str, interval, str(checks)]

            sig_colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if j == 2:
                    item.setForeground(QColor(sig_colors.get(v, TEXT_MUTED)))
                    item.setFont(QFont(FONT_MONO, 10, QFont.Bold))
                if j == 1:
                    item.setForeground(QColor(BRAND_ACCENT if v == "Auto" else TEXT_MUTED))
                self.table.setItem(i, j, item)

            # Unmonitor button
            btn = QPushButton("Remove")
            btn.setStyleSheet(f"color: {COLOR_SELL}; background: transparent; border: 1px solid {COLOR_SELL}; "
                             f"font-size: 9px; padding: 2px 6px; border-radius: 4px;")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"Stop AI monitoring {ticker} (doesn't sell)")
            btn.clicked.connect(lambda _, t=ticker: self._unmonitor(t))
            self.table.setCellWidget(i, 9, btn)

    def _unmonitor(self, ticker):
        # Remove from auto-service
        svc = self._get_svc()
        if svc and svc.is_monitoring(ticker):
            svc.remove_stock(ticker)
        # Remove from agent's tracked stocks
        if ticker in self._agent_stocks:
            del self._agent_stocks[ticker]
        # Save so it doesn't come back
        settings = load_settings()
        monitored = settings.get("monitored_stocks", {})
        monitored.pop(ticker, None)
        tracked = settings.get("agent_tracked_stocks", {})
        tracked.pop(ticker, None)
        settings["monitored_stocks"] = monitored
        settings["agent_tracked_stocks"] = tracked
        save_settings(settings)
        self.bus.log_entry.emit(f"Removed {ticker} — switched to manual management", "system")
        self.refresh()

    def _stop_all(self):
        svc = self._get_svc()
        if not svc:
            return
        count = len(svc.get_monitored())
        if count == 0:
            return
        confirm = QMessageBox.question(
            self, "Stop All",
            f"Remove AI monitoring from {count} stocks?\nPositions will NOT be sold.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        for t in list(svc.get_monitored().keys()):
            svc.remove_stock(t)
        self.bus.log_entry.emit("All AI monitoring stopped", "system")
        self.refresh()

    def _get_svc(self):
        """Get the auto-trade service — check scanner first, then own reference."""
        main = self.window()
        if main and hasattr(main, 'scanner') and hasattr(main.scanner, '_auto_service'):
            self._auto_svc = main.scanner._auto_service
            return self._auto_svc
        if self._auto_svc:
            return self._auto_svc
        # Create our own if scanner not loaded
        if self.broker:
            from core.auto_trader import AutoTraderService
            from core.risk import RiskManager
            self._auto_svc = AutoTraderService(broker=self.broker, risk_manager=RiskManager())
            self._auto_svc.log.connect(self.bus.log_entry.emit)
            self._auto_svc.start()
            return self._auto_svc
        return None

    def _tick_countdown(self):
        if self._countdown_secs > 0:
            self._countdown_secs -= 1
            mins = self._countdown_secs // 60
            secs = self._countdown_secs % 60
            self.agent_countdown.setText(f"Next cycle: {mins}m {secs}s")
        elif getattr(self, '_agent_running', False):
            self.agent_countdown.setText("Scanning...")
            self.agent_countdown.setStyleSheet(f"color: {BRAND_ACCENT}; font-size: 10px;")
        else:
            self.agent_countdown.setText("")

    def _toggle_agent(self):
        """Start/stop the fully autonomous agent."""
        if hasattr(self, '_agent_running') and self._agent_running:
            # Stopping — save state for potential resume
            self._agent_running = False
            self.agent_start_btn.setText("Start Agent")
            self.agent_start_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 12px; padding: 8px 16px;")
            self.agent_status.setText("Agent: Stopped")
            self.bus.log_entry.emit("Autonomous agent stopped", "system")
            # Mark that we have a resumable session
            settings = load_settings()
            settings["agent_was_running"] = True
            save_settings(settings)
            return

        # Starting — check if there's a previous session to resume
        settings = load_settings()
        monitored = settings.get("monitored_stocks", {})

        if monitored and settings.get("agent_was_running", False):
            reply = QMessageBox.question(
                self, "Resume Previous Session?",
                f"The agent was previously managing {len(monitored)} stocks.\n"
                f"({', '.join(list(monitored.keys())[:6])})\n\n"
                f"Resume previous session?\n"
                f"Yes = Resume with same stocks\n"
                f"No = Start fresh scan",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply == QMessageBox.No:
                settings["agent_was_running"] = False
                save_settings(settings)
        else:
            confirm = QMessageBox.question(
                self, "Start Autonomous Agent",
                "The agent will:\n"
                "• Scan the market for opportunities\n"
                "• Buy/sell based on AI signals\n"
                "• Respect your aggressivity profile + buying power\n\n"
                "All decisions logged transparently.\nStart?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return

        self._agent_running = True
        self.agent_start_btn.setText("Stop Agent")
        self.agent_start_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 12px; padding: 8px 16px;")
        self.agent_status.setText("Agent: Running")
        self._countdown_timer.start(1000)
        self.bus.log_entry.emit("Autonomous agent started", "trade")
        log_event("agent", "Autonomous agent started")

        # Log resume context
        settings = load_settings()
        monitored = settings.get("monitored_stocks", {})
        if monitored:
            self.bus.log_entry.emit(f"Resuming with {len(monitored)} monitored stocks", "system")
            for ticker, info in list(monitored.items())[:5]:
                sig = info.get("last_signal", "pending")
                self.bus.log_entry.emit(f"  {ticker}: {sig} ({info.get('interval', '5m')})", "system")
            if len(monitored) > 5:
                self.bus.log_entry.emit(f"  +{len(monitored)-5} more...", "system")
        if settings.get("gemini_enabled"):
            self.bus.log_entry.emit("Gemini AI Advisor: enabled", "system")

        threading.Thread(target=self._run_agent, daemon=True).start()

    def _run_agent(self):
        """Run the autonomous agent — uses aggressivity profile + RL feedback."""
        from core.scanner import scan_multiple
        from core.risk import RiskManager
        from core.profiles import get_optimal_workers
        from core.discovery import get_most_active, get_day_gainers, get_trending_social
        from core.intelligent_trader import get_aggressivity
        from core.logger import log_event

        rm = RiskManager()
        cycle = 0

        # Load aggressivity profile
        settings = load_settings()
        profile_name = settings.get("aggressivity", "Default")
        profile = get_aggressivity(profile_name)
        min_conf = profile["min_confidence"]
        size_mult = profile["size_multiplier"]
        max_trades = profile["max_trades_per_day"]

        # Try loading RL feedback model
        rl_model = None
        try:
            from core.reinforcement import train_feedback_model, get_quality_score
            rl_model, rl_acc, rl_count = train_feedback_model()
            if rl_model:
                self.bus.log_entry.emit(
                    f"Agent: RL model loaded ({rl_count} trades, {rl_acc:.0%} accuracy)", "system")
        except Exception:
            pass

        trades_today = 0
        log_event("agent", f"Agent started — profile: {profile_name}, min_conf: {min_conf:.0%}")

        while getattr(self, '_agent_running', False):
            cycle += 1
            try:
                self.bus.log_entry.emit(f"Agent cycle {cycle}: scanning market...", "system")
                QTimer.singleShot(0, lambda c=cycle: self.agent_status.setText(
                    f"Agent: Running — cycle {c}"))

                # Get buying power
                bp = 0
                if self.broker:
                    try:
                        acct = self.broker.get_account()
                        bp = float(acct.get("buying_power", 0))
                    except Exception as _e: pass

                if trades_today >= max_trades:
                    self.bus.log_entry.emit(
                        f"Agent: max trades/day ({max_trades}) reached. Waiting.", "warn")
                    for _ in range(300):
                        if not getattr(self, '_agent_running', False): break
                        time.sleep(1)
                    continue

                self.bus.log_entry.emit(f"Agent: BP=${bp:,.0f}, profile={profile_name}", "system")

                # Scan multiple sources for diverse opportunities
                tickers = set()
                sources_used = []
                try:
                    ma = get_most_active(10)
                    tickers.update(ma)
                    sources_used.append(f"Active({len(ma)})")
                except: pass
                try:
                    g = get_day_gainers(5)
                    tickers.update(g)
                    sources_used.append(f"Gainers({len(g)})")
                except: pass
                try:
                    t = get_trending_social(5)
                    tickers.update(t)
                    sources_used.append(f"Trending({len(t)})")
                except: pass
                # Always include stocks we currently hold
                try:
                    positions = self.broker.get_positions()
                    if isinstance(positions, list):
                        held = [p.get("symbol", "") for p in positions if p.get("symbol")]
                        tickers.update(held)
                        if held:
                            sources_used.append(f"Held({len(held)})")
                except: pass

                tickers = list(tickers)[:25]  # Allow more to fit held stocks

                self.bus.log_entry.emit(
                    f"Agent: scanning {len(tickers)} tickers from {', '.join(sources_used)}", "system")

                if not tickers:
                    self.bus.log_entry.emit("Agent: no tickers, waiting 2 min...", "warn")
                    for _ in range(120):
                        if not getattr(self, '_agent_running', False): break
                        time.sleep(1)
                    continue

                results = scan_multiple(tickers, "5d", "5m", rm,
                    max_workers=get_optimal_workers(), buying_power=bp)

                # Pre-load Gemini + addon data once
                use_gemini = False
                addon_signals = {}
                try:
                    from core.gemini_advisor import is_enabled as gemini_enabled, get_advisory, apply_advisory
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

                # Apply RL quality scores
                if rl_model:
                    for r in results:
                        if not r.error:
                            try:
                                atr_pct = r.atr / r.price if r.price > 0 else 0
                                q = get_quality_score(rl_model, r.confidence, r.probs, atr_pct, r.action)
                                r.confidence = min(1.0, r.confidence * q)
                            except: pass

                # ── Per-stock pipeline: Gemini → Decide → Execute → Update table ──
                buys, sells, holds, skipped = 0, 0, 0, 0

                for r in sorted(results, key=lambda x: -x.score):
                    if r.error:
                        continue
                    if not getattr(self, '_agent_running', False):
                        break

                    # 1. Gemini advisory (immediate per stock)
                    if use_gemini and r.action in ("BUY", "SELL"):
                        try:
                            # Build rich position context for Gemini
                            pos_info = None
                            if r.ticker in self._agent_stocks:
                                s = self._agent_stocks[r.ticker]
                                pos_info = (f"holding {s.get('qty',0)} shares, "
                                           f"last_signal={s.get('signal','?')}, "
                                           f"checked {s.get('checks',0)}x")
                            # Get exact broker position details
                            try:
                                bpos = self.broker.get_positions()
                                if isinstance(bpos, list):
                                    for p in bpos:
                                        if p.get("symbol","").upper() == r.ticker.upper():
                                            pos_info = (
                                                f"holding {float(p.get('qty',0)):.0f} shares "
                                                f"@ ${float(p.get('avg_entry_price',0)):.2f} entry, "
                                                f"now ${float(p.get('current_price',0)):.2f}, "
                                                f"P&L ${float(p.get('unrealized_pl',0)):+,.2f}")
                                            break
                            except: pass

                            advisory = get_advisory(
                                r.ticker, r.price, r.action, r.confidence, r.probs, r.atr,
                                feature_importances=r.feature_importances,
                                addon_data=addon_signals if addon_signals else None,
                                position_info=pos_info,
                                portfolio_context=f"BP=${bp:,.0f}, {trades_today}/{max_trades} trades")
                            if advisory:
                                old_conf = r.confidence
                                _, r.confidence = apply_advisory(r.action, r.confidence, advisory)
                                conv = int(advisory.get("conviction", 3))
                                gem_rec = advisory.get("recommendation", "?")
                                gem_weight = conv / 5.0
                                agree = "AGREE" if gem_rec == r.action else "DISAGREE"

                                self.bus.log_entry.emit(
                                    f"  {r.ticker}: Local={r.action}({old_conf:.0%}) "
                                    f"Gemini={gem_rec}[{conv}/5,w={gem_weight:.1f}] "
                                    f"→ {r.confidence:.0%} ({agree})", "info")
                                self.bus.log_entry.emit(
                                    f"    Reasoning: {advisory.get('reasoning','')}", "system")
                        except: pass

                    # 2. Calculate dynamic interval based on ATR volatility
                    atr_pct = r.atr / r.price if r.price > 0 and r.atr > 0 else 0.01
                    if atr_pct > 0.02:
                        dyn_interval = 120   # Volatile: 2 min
                        dyn_label = "2m"
                    elif atr_pct > 0.01:
                        dyn_interval = 300   # Normal: 5 min
                        dyn_label = "5m"
                    elif atr_pct > 0.005:
                        dyn_interval = 600   # Calm: 10 min
                        dyn_label = "10m"
                    else:
                        dyn_interval = 900   # Very calm: 15 min
                        dyn_label = "15m"

                    # Update table immediately with latest signal + dynamic interval
                    existing = self._agent_stocks.get(r.ticker, {})
                    self._agent_stocks[r.ticker] = {
                        "signal": r.action, "confidence": r.confidence,
                        "price": r.price, "last_check": datetime.now().strftime("%H:%M:%S"),
                        "checks": existing.get("checks", 0) + 1,
                        "qty": existing.get("qty", 0),
                        "mode": existing.get("mode", "Scanned"),
                        "interval": dyn_label, "next_secs": dyn_interval,
                    }

                    # 3. Classify and execute immediately
                    if r.action == "HOLD" or r.confidence < min_conf:
                        if r.action == "HOLD":
                            holds += 1
                            self.bus.log_entry.emit(f"    Decision: HOLD {r.ticker} — no action", "system")
                        else:
                            skipped += 1
                            self.bus.log_entry.emit(
                                f"    Decision: SKIP {r.ticker} — {r.confidence:.0%} below {min_conf:.0%} threshold", "system")
                        continue

                    if r.action == "SELL" and trades_today < max_trades and self.broker:
                        sells += 1
                        try:
                            positions = self.broker.get_positions()
                            held = 0
                            if isinstance(positions, list):
                                for p in positions:
                                    if p.get("symbol", "").upper() == r.ticker.upper():
                                        held = int(float(p.get("qty", 0)))
                                        break
                            if held > 0:
                                qty = held if r.confidence > 0.7 else max(1, int(held * 0.5)) if r.confidence > 0.5 else max(1, int(held * 0.25))
                                result = self.broker.close_position(r.ticker, qty=qty)
                                if "error" not in result:
                                    trades_today += 1
                                    self.bus.log_entry.emit(f"    Decision: EXECUTE SELL {r.ticker} x{qty}/{held} ({r.confidence:.0%})", "trade")
                                    self._agent_stocks[r.ticker]["qty"] = held - qty
                                    self._agent_stocks[r.ticker]["mode"] = "Auto"
                                else:
                                    self.bus.log_entry.emit(f"    Decision: SELL {r.ticker} FAILED — {result.get('error','')[:40]}", "error")
                            else:
                                self.bus.log_entry.emit(f"    Decision: SELL {r.ticker} — no position held, skipping", "system")
                        except Exception as _e:
                            self.bus.log_entry.emit(f"    Decision: SELL {r.ticker} ERROR — {_e}", "error")

                    elif r.action == "BUY" and trades_today < max_trades and self.broker:
                        buys += 1
                        try:
                            acct = self.broker.get_account()
                            bp = float(acct.get("buying_power", 0))
                        except: pass
                        if bp >= 100:
                            max_spend = min(bp * 0.20, initial_bp / max(1, 5))
                            try:
                                qty = max(1, int(max_spend / r.price)) if r.price > 0 else 0
                                cost = qty * r.price
                                if qty > 0 and cost <= bp:
                                    result = self.broker.place_order(r.ticker, qty, "buy",
                                        stop_loss=r.stop_loss, take_profit=r.take_profit)
                                    if "error" not in result:
                                        trades_today += 1
                                        bp -= cost
                                        self.bus.log_entry.emit(f"    Decision: EXECUTE BUY {r.ticker} x{qty} @ ${r.price:.2f} (${cost:,.0f}, {r.confidence:.0%})", "trade")
                                        self._agent_stocks[r.ticker]["qty"] = qty
                                        self._agent_stocks[r.ticker]["mode"] = "Auto"
                                        try:
                                            mon_svc = self._get_svc()
                                            if mon_svc and not mon_svc.is_monitoring(r.ticker):
                                                mon_svc.add_stock(r.ticker, period="5d", interval="5m", auto_execute=True, min_confidence=0.5)
                                        except: pass
                                    else:
                                        self.bus.log_entry.emit(f"    Decision: BUY {r.ticker} FAILED — {result.get('error','')[:40]}", "error")
                                else:
                                    self.bus.log_entry.emit(f"    Decision: BUY {r.ticker} — cost ${cost:,.0f} > BP ${bp:,.0f}, skipping", "warn")
                            except Exception as _e:
                                self.bus.log_entry.emit(f"    Decision: BUY {r.ticker} ERROR — {_e}", "error")
                        else:
                            self.bus.log_entry.emit(f"    Decision: BUY {r.ticker} — BP ${bp:,.0f} too low, skipping", "warn")

                self.bus.log_entry.emit(
                    f"Cycle {cycle}: {buys}B {sells}S {holds}H ({skipped} skipped)", "trade")

                self.bus.log_entry.emit(
                    f"Cycle {cycle} done. {trades_today}/{max_trades} trades today. Next in 5 min.", "system")
                self._last_cycle = cycle
                self._last_trades_today = trades_today

            except Exception as e:
                self.bus.log_entry.emit(f"Agent error: {e}", "error")

            # Wait 5 minutes with countdown
            self._countdown_secs = 300
            for _ in range(300):
                if not getattr(self, '_agent_running', False): break
                time.sleep(1)
                self._countdown_secs = max(0, self._countdown_secs - 1)

        self._countdown_secs = 0
        self._countdown_timer.stop()
        QTimer.singleShot(0, lambda: self.agent_countdown.setText(""))
        self.bus.log_entry.emit("Autonomous agent stopped", "system")
        log_event("agent", f"Agent stopped after {cycle} cycles, {trades_today} trades")
