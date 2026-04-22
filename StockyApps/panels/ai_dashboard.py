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
from core.agent import AgentEngine

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

        # Create the agent engine (decoupled from UI)
        self._engine = AgentEngine(
            broker=broker,
            log_fn=lambda msg, level: self.bus.log_entry.emit(msg, level),
            settings_fn=load_settings,
        )
        self._engine.on_tray_update = self._tray_update
        self._engine.on_tray_action = self._tray_action
        self._engine.on_add_monitor = self._add_to_monitor

        # Restore agent tracked stocks from last session
        settings = load_settings()
        saved_tracked = settings.get("agent_tracked_stocks", {})
        if saved_tracked:
            self._engine.agent_stocks = dict(saved_tracked)

        # Keep _agent_stocks as a property alias for backward compat
        self._agent_stocks = self._engine.agent_stocks

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
        # Info button — pipeline transparency
        info_btn = QPushButton("?")
        info_btn.setFixedSize(26, 26)
        info_btn.setToolTip("How the AI Agent works")
        info_btn.setStyleSheet(
            f"background-color: {BRAND_PRIMARY}; color: white; font-weight: bold; "
            f"border-radius: 13px; font-size: 13px;")
        info_btn.clicked.connect(self._show_pipeline_info)
        agent_row.addWidget(info_btn)
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
        """Show rich detail popup with charts when stat card is clicked."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from core.logger import get_log_entries, get_log_files
        from collections import defaultdict

        # Gather all log data across files
        all_decisions = []
        all_executions = []
        all_events = []
        for fi in get_log_files():
            for e in get_log_entries(fi["file"], 500):
                t = e.get("type", "")
                if t == "decision":
                    all_decisions.append(e)
                elif t == "execution":
                    all_executions.append(e)
                elif t == "event":
                    all_events.append(e)

        # Get current engine state
        engine = self._engine
        agent_stocks = engine.agent_stocks
        stats = engine.cycle_stats

        dlg = QDialog(self)
        dlg.setWindowIcon(QApplication.instance().windowIcon())
        dlg.setMinimumSize(700, 500)

        lay = QVBoxLayout()
        cc = chart_colors()

        if card_type == "monitored":
            dlg.setWindowTitle("Monitored Stocks — Detail")

            title = QLabel(f"Tracking {len(agent_stocks)} Stocks")
            title.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
            title.setStyleSheet(f"color: {BRAND_PRIMARY};")
            lay.addWidget(title)

            # Confidence distribution chart
            fig = plt.Figure(figsize=(7, 3), dpi=100, facecolor=cc["fig_bg"])
            ax = fig.add_subplot(111)
            ax.set_facecolor(cc["ax_bg"])

            if agent_stocks:
                tickers = []
                confs = []
                colors_list = []
                for t, info in sorted(agent_stocks.items(), key=lambda x: -x[1].get("confidence", 0)):
                    tickers.append(t)
                    confs.append(info.get("confidence", 0) * 100)
                    sig = info.get("signal", "HOLD")
                    colors_list.append(COLOR_BUY if sig == "BUY" else COLOR_SELL if sig == "SELL" else COLOR_HOLD)

                bars = ax.barh(range(len(tickers)), confs, color=colors_list, alpha=0.85)
                ax.set_yticks(range(len(tickers)))
                ax.set_yticklabels(tickers, fontsize=8, color=cc["text"])
                ax.set_xlabel("Confidence %", fontsize=9, color=cc["muted"])
                ax.invert_yaxis()

                # Add value labels
                for bar, val in zip(bars, confs):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                            f"{val:.0f}%", va="center", fontsize=7, color=cc["text"])

            ax.set_title("Current Signal Confidence by Stock", fontsize=10, color=cc["text"])
            ax.tick_params(colors=cc["muted"], labelsize=7)
            ax.grid(True, alpha=0.1, axis="x")
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(200)
            lay.addWidget(canvas)

            # Stock detail table
            tbl = QTableWidget()
            tbl.setColumnCount(7)
            tbl.setHorizontalHeaderLabels(["Stock", "Signal", "Conf", "Price", "Mode", "Checks", "Interval"])
            for c in range(7):
                tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tbl.setRowCount(len(agent_stocks))
            for i, (t, info) in enumerate(sorted(agent_stocks.items(), key=lambda x: -x[1].get("confidence", 0))):
                sig = info.get("signal", "?")
                sig_color = COLOR_BUY if sig == "BUY" else COLOR_SELL if sig == "SELL" else COLOR_HOLD
                vals = [t, sig, f"{info.get('confidence',0):.0%}", f"${info.get('price',0):.2f}",
                        info.get("mode", "?"), str(info.get("checks", 0)), info.get("interval", "?")]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignCenter)
                    if j == 1:
                        item.setForeground(QColor(sig_color))
                    tbl.setItem(i, j, item)
            lay.addWidget(tbl)

        elif card_type == "trades":
            dlg.setWindowTitle("Trade Activity — Detail")

            # Parse execution timestamps by hour and day
            by_hour = defaultdict(int)
            by_day = defaultdict(int)
            buy_trades = []
            sell_trades = []
            for e in all_executions:
                ts = e.get("timestamp", "")[:19]
                side = e.get("side", "")
                ticker = e.get("ticker", "")
                if len(ts) >= 13:
                    hour = ts[11:13]
                    by_hour[hour] += 1
                if len(ts) >= 10:
                    day = ts[:10]
                    by_day[day] += 1
                if side == "buy":
                    buy_trades.append(e)
                elif side == "sell":
                    sell_trades.append(e)

            title = QLabel(f"{len(all_executions)} Trades Executed ({len(buy_trades)} buys, {len(sell_trades)} sells)")
            title.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
            title.setStyleSheet(f"color: {BRAND_PRIMARY};")
            lay.addWidget(title)

            # Trades by hour chart
            fig = plt.Figure(figsize=(7, 2.5), dpi=100, facecolor=cc["fig_bg"])
            ax = fig.add_subplot(121)
            ax.set_facecolor(cc["ax_bg"])
            if by_hour:
                hours = sorted(by_hour.keys())
                counts = [by_hour[h] for h in hours]
                ax.bar(hours, counts, color=BRAND_ACCENT, alpha=0.8)
                ax.set_title("Trades by Hour", fontsize=9, color=cc["text"])
                ax.tick_params(colors=cc["muted"], labelsize=7)
                ax.set_xlabel("Hour", fontsize=8, color=cc["muted"])

            # Trades by day chart
            ax2 = fig.add_subplot(122)
            ax2.set_facecolor(cc["ax_bg"])
            if by_day:
                days = sorted(by_day.keys())
                day_labels = [d[5:] for d in days]  # MM-DD
                counts = [by_day[d] for d in days]
                ax2.bar(day_labels, counts, color=BRAND_PRIMARY, alpha=0.8)
                ax2.set_title("Trades by Day", fontsize=9, color=cc["text"])
                ax2.tick_params(colors=cc["muted"], labelsize=7)

            fig.tight_layout()
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(180)
            lay.addWidget(canvas)

            # Recent trades table
            recent = sorted(all_executions, key=lambda x: x.get("timestamp", ""), reverse=True)[:20]
            tbl = QTableWidget()
            tbl.setColumnCount(6)
            tbl.setHorizontalHeaderLabels(["Time", "Ticker", "Side", "Qty", "Status", "Error"])
            for c in range(6):
                tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tbl.setRowCount(len(recent))
            for i, e in enumerate(recent):
                ts = e.get("timestamp", "")[:19]
                side = e.get("side", "?")
                sc = COLOR_BUY if side == "buy" else COLOR_SELL
                err = e.get("error", "")
                vals = [ts[11:19] if len(ts) > 11 else ts, e.get("ticker", "?"), side.upper(),
                        str(e.get("qty", "?")), e.get("status", "?"), str(err or "OK")[:30]]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignCenter)
                    if j == 2:
                        item.setForeground(QColor(sc))
                    if j == 5 and err:
                        item.setForeground(QColor(COLOR_SELL))
                    elif j == 5:
                        item.setForeground(QColor(BRAND_ACCENT))
                    tbl.setItem(i, j, item)
            lay.addWidget(tbl)

        elif card_type == "signals":
            dlg.setWindowTitle("Signal Distribution — Detail")

            # Count signals by action and by ticker
            action_counts = defaultdict(int)
            ticker_signals = defaultdict(lambda: {"BUY": 0, "SELL": 0, "HOLD": 0})
            conf_by_action = defaultdict(list)
            hourly_signals = defaultdict(lambda: {"BUY": 0, "SELL": 0, "HOLD": 0})

            for d in all_decisions:
                action = d.get("action", "?")
                ticker = d.get("ticker", "?")
                conf = d.get("confidence", 0)
                ts = d.get("timestamp", "")
                action_counts[action] += 1
                ticker_signals[ticker][action] += 1
                conf_by_action[action].append(conf)
                if len(ts) >= 13:
                    hourly_signals[ts[11:13]][action] += 1

            total = sum(action_counts.values())
            title = QLabel(f"{total} Signals: {action_counts.get('BUY',0)} BUY / "
                          f"{action_counts.get('SELL',0)} SELL / {action_counts.get('HOLD',0)} HOLD")
            title.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
            title.setStyleSheet(f"color: {BRAND_PRIMARY};")
            lay.addWidget(title)

            fig = plt.Figure(figsize=(7, 3), dpi=100, facecolor=cc["fig_bg"])

            # Pie chart: signal distribution
            ax1 = fig.add_subplot(131)
            ax1.set_facecolor(cc["fig_bg"])
            if action_counts:
                labels = list(action_counts.keys())
                sizes = list(action_counts.values())
                pie_colors = [COLOR_BUY if l == "BUY" else COLOR_SELL if l == "SELL" else COLOR_HOLD for l in labels]
                ax1.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.0f%%",
                        textprops={"fontsize": 8, "color": "white"})
                ax1.set_title("Distribution", fontsize=9, color=cc["text"])

            # Confidence box plot by action
            ax2 = fig.add_subplot(132)
            ax2.set_facecolor(cc["ax_bg"])
            if conf_by_action:
                data = []
                labels = []
                bcolors = []
                for action in ["BUY", "SELL", "HOLD"]:
                    if action in conf_by_action:
                        data.append([c * 100 for c in conf_by_action[action]])
                        labels.append(action)
                        bcolors.append(COLOR_BUY if action == "BUY" else COLOR_SELL if action == "SELL" else COLOR_HOLD)
                bp = ax2.boxplot(data, labels=labels, patch_artist=True)
                for patch, color in zip(bp["boxes"], bcolors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                ax2.set_title("Confidence %", fontsize=9, color=cc["text"])
                ax2.tick_params(colors=cc["muted"], labelsize=7)

            # Signals by hour stacked bar
            ax3 = fig.add_subplot(133)
            ax3.set_facecolor(cc["ax_bg"])
            if hourly_signals:
                hours = sorted(hourly_signals.keys())
                buys = [hourly_signals[h]["BUY"] for h in hours]
                sells = [hourly_signals[h]["SELL"] for h in hours]
                holds = [hourly_signals[h]["HOLD"] for h in hours]
                x = range(len(hours))
                ax3.bar(x, buys, color=COLOR_BUY, alpha=0.8, label="BUY")
                ax3.bar(x, sells, bottom=buys, color=COLOR_SELL, alpha=0.8, label="SELL")
                ax3.bar(x, holds, bottom=[b + s for b, s in zip(buys, sells)],
                        color=COLOR_HOLD, alpha=0.8, label="HOLD")
                ax3.set_xticks(x)
                ax3.set_xticklabels(hours, fontsize=6, color=cc["muted"])
                ax3.set_title("By Hour", fontsize=9, color=cc["text"])
                ax3.tick_params(colors=cc["muted"], labelsize=7)
                ax3.legend(fontsize=6)

            fig.tight_layout()
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(220)
            lay.addWidget(canvas)

            # Top tickers table
            top = sorted(ticker_signals.items(), key=lambda x: sum(x[1].values()), reverse=True)[:15]
            tbl = QTableWidget()
            tbl.setColumnCount(5)
            tbl.setHorizontalHeaderLabels(["Ticker", "BUY", "SELL", "HOLD", "Total"])
            for c in range(5):
                tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tbl.setRowCount(len(top))
            for i, (ticker, counts) in enumerate(top):
                total = sum(counts.values())
                vals = [ticker, str(counts["BUY"]), str(counts["SELL"]), str(counts["HOLD"]), str(total)]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignCenter)
                    if j == 1: item.setForeground(QColor(COLOR_BUY))
                    elif j == 2: item.setForeground(QColor(COLOR_SELL))
                    elif j == 3: item.setForeground(QColor(COLOR_HOLD))
                    tbl.setItem(i, j, item)
            lay.addWidget(tbl)

        elif card_type == "mode":
            dlg.setWindowTitle("Agent Mode — Detail")

            settings = load_settings()
            aggr = settings.get("aggressivity", "Default")
            from core.intelligent_trader import get_aggressivity, AGGRESSIVITY_PROFILES
            p = get_aggressivity(aggr)

            title = QLabel(f"Profile: {aggr}")
            title.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
            title.setStyleSheet(f"color: {BRAND_PRIMARY};")
            lay.addWidget(title)

            # Regime + engine state
            regime_name = "Unknown"
            regime_desc = ""
            if engine.cycle_stats:
                regime_name = engine.cycle_stats.get("regime", "?")
                regime_desc = engine.cycle_stats.get("regime_detail", "")

            info_text = (
                f"Aggressivity: {aggr}\n"
                f"Min Confidence: {p['min_confidence']:.0%}\n"
                f"Position Size: {p['size_multiplier']:.1f}x\n"
                f"Max Trades/Day: {p['max_trades_per_day']}\n"
                f"LLM Advisor: {'Enabled' if p.get('use_llm') else 'Disabled'}\n"
                f"Stop Loss: {p.get('atr_stop_mult', 1.5):.1f}x ATR\n"
                f"Take Profit: {p.get('atr_profit_mult', 2.5):.1f}x ATR\n\n"
                f"Current Regime: {regime_name}\n"
                f"{regime_desc}\n\n"
                f"Engine: cycle {engine.cycle}, {engine.trades_today} trades today\n"
                f"Phase: {engine.phase}"
            )
            info = QLabel(info_text)
            info.setWordWrap(True)
            info.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
            lay.addWidget(info)

            # Profile comparison chart
            fig = plt.Figure(figsize=(7, 2.5), dpi=100, facecolor=cc["fig_bg"])
            ax = fig.add_subplot(111)
            ax.set_facecolor(cc["ax_bg"])

            profiles = list(AGGRESSIVITY_PROFILES.keys())
            metrics = ["min_confidence", "size_multiplier", "max_trades_per_day"]
            x = range(len(profiles))
            width = 0.25
            for mi, metric in enumerate(metrics):
                vals = []
                for pn in profiles:
                    pp = AGGRESSIVITY_PROFILES[pn]
                    v = pp.get(metric, 0)
                    if metric == "max_trades_per_day":
                        v = v / 30.0  # Normalize to 0-1
                    vals.append(v)
                offset = (mi - 1) * width
                bar_colors = [BRAND_ACCENT if pn == aggr else cc["muted"] for pn in profiles]
                ax.bar([xi + offset for xi in x], vals, width, label=metric.replace("_", " ").title(),
                       color=bar_colors, alpha=0.7 + (0.3 if mi == 0 else 0))

            ax.set_xticks(x)
            ax.set_xticklabels(profiles, fontsize=9, color=cc["text"])
            ax.set_title("Profile Comparison (active highlighted)", fontsize=10, color=cc["text"])
            ax.tick_params(colors=cc["muted"], labelsize=7)
            ax.legend(fontsize=7)
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(180)
            lay.addWidget(canvas)

            # Reflection rules
            from core.agent.reflection import get_active_rules
            rules = get_active_rules()
            if rules:
                rules_label = QLabel(f"Active Reflection Rules ({len(rules)})")
                rules_label.setFont(QFont(FONT_FAMILY, 10, QFont.Bold))
                rules_label.setStyleSheet(f"color: {BRAND_ACCENT};")
                lay.addWidget(rules_label)
                for r in rules:
                    rl = QLabel(f"  {r['rule']}")
                    rl.setWordWrap(True)
                    rl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")
                    lay.addWidget(rl)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)

        dlg.setLayout(lay)
        dlg.exec_()

    def _on_log(self, msg, level):
        """Capture all trade/system/auto-related logs."""
        relevant_levels = {"trade", "system", "agent", "scan", "decision", "gemini", "rl", "error", "warn", "info"}
        keywords = ["auto", "monitor", "signal", "check", "agent", "trade", "buy", "sell", "hold"]
        if level in relevant_levels or any(k in msg.lower() for k in keywords):
            from core.branding import log_html
            self.log_area.append(log_html(msg, level))
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
        cd = self._engine.countdown
        if cd > 0:
            mins, secs = cd // 60, cd % 60
            self.agent_countdown.setText(f"Next cycle: {mins}m {secs}s")
        elif self._engine.running:
            phase = self._engine.phase
            if phase in ("scanning", "selling", "buying"):
                self.agent_countdown.setText(f"{phase.title()}...")
                self.agent_countdown.setStyleSheet(f"color: {BRAND_ACCENT}; font-size: 10px;")
            else:
                self.agent_countdown.setText("Working...")
        else:
            self.agent_countdown.setText("")

    def _tray_update(self, **kwargs):
        """Push agent state to tray overlay."""
        try:
            main = self.window()
            if main and hasattr(main, '_tray'):
                main._tray.update_agent_state(**kwargs)
        except Exception:
            pass

    def _tray_action(self, text):
        """Add a recent action line to the tray overlay."""
        try:
            main = self.window()
            if main and hasattr(main, '_tray'):
                main._tray.add_action(text)
        except Exception:
            pass

    def _toggle_agent(self):
        """Start/stop the fully autonomous agent."""
        if self._engine.running:
            self._engine.stop()
            self.agent_start_btn.setText("Resume Agent")
            self.agent_start_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 12px; padding: 8px 16px;")
            self.agent_status.setText("Agent: Stopped (session saved)")
            self._agent_stocks = self._engine.agent_stocks
            settings = load_settings()
            settings["agent_was_running"] = True
            save_settings(settings)
            return

        # Check if there's a resumable session
        settings = load_settings()
        monitored = settings.get("monitored_stocks", {})
        tracked = settings.get("agent_tracked_stocks", {})
        has_session = bool(self._engine.agent_stocks) or (
            (monitored or tracked) and settings.get("agent_was_running", False))

        if has_session:
            self._show_resume_dialog()
        else:
            confirm = QMessageBox.question(
                self, "Start Autonomous Agent",
                "The agent will:\n"
                "- Scan the market for opportunities\n"
                "- Buy/sell based on AI + Gemini signals\n"
                "- Rotate capital based on your aggressivity\n\n"
                "All decisions logged transparently.\nStart?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return
            self._start_agent()

    def _show_resume_dialog(self):
        """Show resume/new session dialog with position details."""
        settings = load_settings()
        monitored = settings.get("monitored_stocks", {})
        tracked = settings.get("agent_tracked_stocks", {})
        positions = settings.get("agent_managed_positions", [])

        dlg = QDialog(self)
        dlg.setWindowTitle("Resume AI Agent")
        dlg.setWindowIcon(QApplication.instance().windowIcon())
        dlg.setMinimumSize(500, 350)

        lay = QVBoxLayout()
        title = QLabel("Resume Previous Session?")
        title.setFont(QFont(FONT_FAMILY, 13, QFont.Bold))
        title.setStyleSheet(f"color: {BRAND_PRIMARY};")
        lay.addWidget(title)

        all_stocks = {**tracked}
        for t in monitored:
            if t not in all_stocks:
                all_stocks[t] = monitored[t]
        for t in self._engine.agent_stocks:
            if t not in all_stocks:
                all_stocks[t] = self._engine.agent_stocks[t]

        info = QLabel(f"{len(all_stocks)} stocks tracked, {len(positions)} positions held")
        info.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")
        lay.addWidget(info)

        if positions:
            total_pl = sum(float(p.get("unrealized_pl", 0)) for p in positions)
            total_val = sum(float(p.get("qty", 0)) * float(p.get("current_price", 0)) for p in positions)
            summary = QLabel(f"Value: ${total_val:,.0f}  |  P&L: ${total_pl:+,.2f}")
            summary.setStyleSheet(f"color: {COLOR_PROFIT if total_pl >= 0 else COLOR_LOSS}; font-weight: bold;")
            lay.addWidget(summary)

            tbl = QTableWidget()
            tbl.setColumnCount(4)
            tbl.setHorizontalHeaderLabels(["Stock", "Qty", "Price", "P&L"])
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tbl.setMaximumHeight(150)
            tbl.setRowCount(len(positions))
            for i, p in enumerate(positions):
                vals = [p.get("symbol",""), f"{float(p.get('qty',0)):.0f}",
                        f"${float(p.get('current_price',0)):.2f}",
                        f"${float(p.get('unrealized_pl',0)):+,.2f}"]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignCenter)
                    if j == 3:
                        item.setForeground(QColor(COLOR_PROFIT if float(p.get("unrealized_pl",0)) >= 0 else COLOR_LOSS))
                    tbl.setItem(i, j, item)
            lay.addWidget(tbl)

        note = QLabel("Resume = Continue with saved stocks + signals\nNew Session = Clear saved data, start fresh")
        note.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")
        lay.addWidget(note)

        btn_row = QHBoxLayout()
        resume_btn = QPushButton("Resume Session")
        resume_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 11px; padding: 7px 16px;")
        resume_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(resume_btn)
        new_btn = QPushButton("New Session")
        new_btn.setStyleSheet(f"background-color: {BG_INPUT}; font-size: 11px; padding: 7px 16px;")
        new_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(new_btn)
        lay.addLayout(btn_row)

        dlg.setLayout(lay)
        result = dlg.exec_()

        if result == QDialog.Rejected:
            # New session — clear saved state but keep training data
            settings["agent_was_running"] = False
            settings["agent_tracked_stocks"] = {}
            settings["monitored_stocks"] = {}
            settings["agent_managed_positions"] = []
            save_settings(settings)
            self._engine.agent_stocks = {}
            self._agent_stocks = self._engine.agent_stocks
            try:
                svc = self._get_svc()
                if svc:
                    for t in list(svc._stocks.keys()):
                        svc.remove_stock(t)
            except Exception:
                pass
            self.bus.log_entry.emit("Starting fresh session — saved data cleared", "system")

        # Both resume and new session start the agent
        self._start_agent()

    def _start_agent(self):
        """Start the agent engine — called after dialog decisions are made."""
        self.agent_start_btn.setText("Stop Agent")
        self.agent_start_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 12px; padding: 8px 16px;")
        self.agent_status.setText("Agent: Running")
        self._countdown_timer.start(1000)

        # Restore engine state from previous session
        settings = load_settings()
        engine_state = settings.get("agent_engine_state")
        if engine_state:
            self._engine.restore_state(engine_state)
            prev_cycle = engine_state.get("cycle", 0)
            prev_trades = engine_state.get("trades_today", 0)
            pdt = engine_state.get("pdt_restricted", False)
            bp = engine_state.get("last_bp", 0)
            if prev_cycle > 0:
                self.bus.log_entry.emit(
                    f"Restored session: cycle {prev_cycle}, {prev_trades} trades, "
                    f"BP=${bp:,.0f}{' [PDT]' if pdt else ''}, "
                    f"{len(self._engine.agent_stocks)} stocks tracked", "agent")
            cached = engine_state.get("tradeable_cache", {})
            if cached:
                untradeable = [t for t, ok in cached.items() if not ok]
                if untradeable:
                    self.bus.log_entry.emit(
                        f"  Cached untradeable: {', '.join(untradeable[:5])}", "system")

        # Log resume context
        monitored = settings.get("monitored_stocks", {})
        if monitored:
            self.bus.log_entry.emit(f"Resuming with {len(monitored)} monitored stocks", "system")
            for ticker, info in list(monitored.items())[:5]:
                sig = info.get("last_signal", "pending")
                self.bus.log_entry.emit(f"  {ticker}: {sig} ({info.get('interval', '5m')})", "system")
            if len(monitored) > 5:
                self.bus.log_entry.emit(f"  +{len(monitored)-5} more...", "system")
        if settings.get("gemini_enabled"):
            self.bus.log_entry.emit("Gemini AI Advisor: enabled", "gemini")

        self._engine.start()
        self._agent_stocks = self._engine.agent_stocks

    def _add_to_monitor(self, ticker):
        """Callback for engine: add a stock to the auto-monitoring service."""
        try:
            svc = self._get_svc()
            if svc and not svc.is_monitoring(ticker):
                svc.add_stock(ticker, period="5d", interval="5m", auto_execute=True, min_confidence=0.5)
        except Exception:
            pass

    def _show_pipeline_info(self):
        """Show live pipeline transparency popup with animated step indicators."""
        from core.agent.info import PIPELINE_INFO, get_info_html

        dlg = QDialog(self)
        dlg.setWindowTitle("AI Agent Pipeline")
        dlg.setWindowIcon(QApplication.instance().windowIcon())
        dlg.setMinimumSize(600, 520)

        lay = QVBoxLayout()
        title = QLabel("How the Autonomous Agent Works")
        title.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        title.setStyleSheet(f"color: {BRAND_PRIMARY};")
        lay.addWidget(title)

        # Live status bar
        phase = self._engine.phase if self._engine.running else "idle"
        phase_detail = self._engine.phase_detail if self._engine.running else "Agent not running"
        status_colors = {
            "idle": TEXT_MUTED, "context": BRAND_PRIMARY, "discovery": BRAND_SECONDARY,
            "scanning": "#a78bfa", "rl_adjust": "#f472b6", "filtering": COLOR_HOLD,
            "preparing": BRAND_PRIMARY, "selling": COLOR_SELL, "buying": BRAND_ACCENT,
            "complete": BRAND_ACCENT, "waiting": TEXT_MUTED, "error": COLOR_SELL,
        }
        sc = status_colors.get(phase, TEXT_MUTED)
        status = QLabel(f"Current: {phase.upper()} — {phase_detail}")
        status.setStyleSheet(f"color: {sc}; font-weight: bold; font-size: 11px;")
        lay.addWidget(status)

        # Cycle stats
        if self._engine.running:
            stats = self._engine.cycle_stats
            if stats:
                stats_text = (
                    f"Cycle {self._engine.cycle} | "
                    f"{stats.get('buys',0)}B {stats.get('sells',0)}S {stats.get('holds',0)}H | "
                    f"{stats.get('filtered',0)} filtered | "
                    f"Trades: {stats.get('trades_today',0)} | "
                    f"BP: ${stats.get('bp',0):,.0f}")
                st = QLabel(stats_text)
                st.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")
                lay.addWidget(st)

            decisions = self._engine.cycle_decisions
            if decisions:
                dec = QLabel("Recent: " + " | ".join(decisions[-5:]))
                dec.setStyleSheet(f"color: {BRAND_ACCENT}; font-size: 9px;")
                dec.setWordWrap(True)
                lay.addWidget(dec)

        # Pipeline step indicators
        phases = [
            ("1", "context", "Gather Context"),
            ("2", "discovery", "Discover Tickers"),
            ("3", "scanning", "AI Scan + RL"),
            ("4", "filtering", "Pre-Filter"),
            ("5", "preparing", "Split Signals"),
            ("6A", "selling", "Execute Sells"),
            ("6B", "buying", "Execute Buys"),
            ("7", "waiting", "Dynamic Wait"),
        ]
        steps_row = QHBoxLayout()
        steps_row.setSpacing(2)
        for num, key, label in phases:
            active = (phase == key)
            done = False
            phase_order = [p[1] for p in phases]
            if key in phase_order and phase in phase_order:
                done = phase_order.index(phase) > phase_order.index(key)

            if active:
                bg, fg = BRAND_PRIMARY, "white"
            elif done:
                bg, fg = BRAND_ACCENT, "white"
            else:
                bg, fg = BG_INPUT, TEXT_MUTED

            step = QLabel(f" {num} ")
            step.setAlignment(Qt.AlignCenter)
            step.setToolTip(label)
            step.setStyleSheet(
                f"background: {bg}; color: {fg}; border-radius: 4px; "
                f"padding: 3px 6px; font-size: 9px; font-weight: bold;")
            steps_row.addWidget(step)
        steps_row.addStretch()
        lay.addLayout(steps_row)

        # Scrollable detailed info
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        cl = QVBoxLayout()
        cl.setContentsMargins(4, 4, 4, 4)

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml(get_info_html())
        info_text.setFont(QFont(FONT_FAMILY, 10))
        cl.addWidget(info_text)

        content.setLayout(cl)
        scroll.setWidget(content)
        lay.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)

        dlg.setLayout(lay)

        # Auto-refresh the status while dialog is open
        def _refresh_status():
            if not dlg.isVisible():
                return
            p = self._engine.phase if self._engine.running else "idle"
            pd = self._engine.phase_detail if self._engine.running else "Agent not running"
            c = status_colors.get(p, TEXT_MUTED)
            status.setText(f"Current: {p.upper()} — {pd}")
            status.setStyleSheet(f"color: {c}; font-weight: bold; font-size: 11px;")

            # Update step indicators
            for i, (num, key, label) in enumerate(phases):
                active = (p == key)
                phase_order = [pp[1] for pp in phases]
                done = p in phase_order and key in phase_order and phase_order.index(p) > phase_order.index(key)
                w = steps_row.itemAt(i).widget()
                if active:
                    w.setStyleSheet(f"background: {BRAND_PRIMARY}; color: white; border-radius: 4px; padding: 3px 6px; font-size: 9px; font-weight: bold;")
                elif done:
                    w.setStyleSheet(f"background: {BRAND_ACCENT}; color: white; border-radius: 4px; padding: 3px 6px; font-size: 9px; font-weight: bold;")
                else:
                    w.setStyleSheet(f"background: {BG_INPUT}; color: {TEXT_MUTED}; border-radius: 4px; padding: 3px 6px; font-size: 9px; font-weight: bold;")

        refresh_timer = QTimer(dlg)
        refresh_timer.timeout.connect(_refresh_status)
        refresh_timer.start(500)

        dlg.exec_()

    # Legacy method removed — agent loop now runs in core.agent.engine.AgentEngine
    # The _run_agent method has been replaced by self._engine.start()
    _run_agent = None  # Explicitly removed
    # _agent_running flag replaced by self._engine.running
    @property
    def _agent_running(self):
        return self._engine.running
    # ── End of _agent_running property ──

