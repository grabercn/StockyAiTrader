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

    def __init__(self, broker, event_bus):
        super().__init__()
        self.broker = broker
        self.bus = event_bus
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
        for c in [self.card_monitored, self.card_trades, self.card_signals, self.card_mode]:
            cards.addWidget(c)
        layout.addLayout(cards)

        # Autonomous agent controls
        agent_box = QGroupBox("Autonomous Agent")
        agent_box.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: {BRAND_ACCENT}; }}")
        al = QVBoxLayout()
        al.setSpacing(4)

        agent_desc = QLabel(
            "The autonomous agent runs scans, picks the best stocks, and manages trades automatically. "
            "It respects your aggressivity profile, buying power limits, and HOLD signals."
        )
        agent_desc.setWordWrap(True)
        agent_desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")
        al.addWidget(agent_desc)

        agent_row = QHBoxLayout()
        self.agent_start_btn = QPushButton("Start Agent")
        self.agent_start_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 12px; padding: 8px 16px;")
        self.agent_start_btn.clicked.connect(self._toggle_agent)
        agent_row.addWidget(self.agent_start_btn)

        self.agent_status = QLabel("Agent: Stopped")
        self.agent_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        agent_row.addWidget(self.agent_status, 1)
        al.addLayout(agent_row)

        agent_box.setLayout(al)
        layout.addWidget(agent_box)

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
        main = self.window()
        if not main:
            return

        svc = None
        if hasattr(main, 'scanner') and hasattr(main.scanner, '_auto_service'):
            svc = main.scanner._auto_service

        if not svc:
            self.card_monitored.set_value("0")
            self.card_mode.set_value("No agent")
            self.table.setRowCount(0)
            return

        monitored = svc.get_monitored()
        self.card_monitored.set_value(str(len(monitored)))

        settings = load_settings()
        self.card_mode.set_value(settings.get("aggressivity", "Default"))

        buys = sum(1 for s in monitored.values() if s.last_signal == "BUY")
        sells = sum(1 for s in monitored.values() if s.last_signal == "SELL")
        holds = sum(1 for s in monitored.values() if s.last_signal == "HOLD")
        self.card_signals.set_value(f"{buys}B {sells}S {holds}H")

        total_trades = sum(getattr(s, 'check_count', 0) for s in monitored.values())
        self.card_trades.set_value(str(total_trades))

        # Table with unmonitor buttons
        self.table.setRowCount(len(monitored))
        for i, (ticker, stock) in enumerate(monitored.items()):
            signal = getattr(stock, 'last_signal', 'HOLD')
            conf = getattr(stock, 'last_confidence', 0)
            price = getattr(stock, 'last_price', 0)
            last_check = getattr(stock, 'last_check', '--')
            next_secs = getattr(stock, 'next_check_seconds', 0)
            interval = getattr(stock, 'interval', '5m')
            checks = getattr(stock, 'check_count', 0)
            auto = getattr(stock, 'auto_execute', False)

            mins = next_secs // 60
            secs = next_secs % 60
            mode = "Auto" if auto else "Signal"

            vals = [ticker, mode, signal, f"{conf:.0%}",
                    f"${price:.2f}" if price else "--",
                    str(last_check) if last_check else "--",
                    f"{mins}m{secs}s", interval, str(checks)]

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
        main = self.window()
        if not main or not hasattr(main, 'scanner'):
            return
        svc = getattr(main.scanner, '_auto_service', None)
        if svc and svc.is_monitoring(ticker):
            svc.remove_stock(ticker)
            self.bus.log_entry.emit(f"Removed {ticker} from AI monitoring (position kept)", "system")
            self.refresh()

    def _stop_all(self):
        main = self.window()
        if not main or not hasattr(main, 'scanner'):
            return
        svc = getattr(main.scanner, '_auto_service', None)
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

    def _toggle_agent(self):
        """Start/stop the fully autonomous agent."""
        if hasattr(self, '_agent_running') and self._agent_running:
            self._agent_running = False
            self.agent_start_btn.setText("Start Agent")
            self.agent_start_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 12px; padding: 8px 16px;")
            self.agent_status.setText("Agent: Stopped")
            self.bus.log_entry.emit("Autonomous agent stopped", "system")
            return

        confirm = QMessageBox.question(
            self, "Start Autonomous Agent",
            "The agent will:\n"
            "• Scan the market for opportunities\n"
            "• Buy stocks with strong BUY signals\n"
            "• Sell positions with strong SELL signals\n"
            "• Hold when signals are uncertain\n"
            "• Respect your buying power and aggressivity profile\n\n"
            "All decisions are logged and transparent.\nStart?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        self._agent_running = True
        self.agent_start_btn.setText("Stop Agent")
        self.agent_start_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 12px; padding: 8px 16px;")
        self.agent_status.setText("Agent: Running — scanning...")
        self.bus.log_entry.emit("Autonomous agent started", "trade")
        log_event("agent", "Autonomous agent started")

        threading.Thread(target=self._run_agent, daemon=True).start()

    def _run_agent(self):
        """Run the autonomous agent loop."""
        from core.scanner import scan_multiple
        from core.risk import RiskManager
        from core.profiles import get_optimal_workers
        from core.discovery import get_most_active

        rm = RiskManager()
        cycle = 0

        while getattr(self, '_agent_running', False):
            cycle += 1
            try:
                self.bus.log_entry.emit(f"Agent cycle {cycle}: scanning market...", "system")

                # Get buying power
                bp = 0
                if self.broker:
                    try:
                        acct = self.broker.get_account()
                        bp = float(acct.get("buying_power", 0))
                    except: pass

                # Scan most active stocks
                tickers = get_most_active(15)
                if not tickers:
                    self.bus.log_entry.emit("Agent: no tickers found, waiting...", "warn")
                    time.sleep(60)
                    continue

                results = scan_multiple(tickers, "5d", "5m", rm,
                    max_workers=get_optimal_workers(), buying_power=bp)

                # Filter by signal
                buys = [r for r in results if r.action == "BUY" and r.confidence > 0.5 and not r.error]
                sells = [r for r in results if r.action == "SELL" and r.confidence > 0.5 and not r.error]
                holds = [r for r in results if r.action == "HOLD" and not r.error]

                self.bus.log_entry.emit(
                    f"Agent cycle {cycle}: {len(buys)} BUY, {len(sells)} SELL, {len(holds)} HOLD",
                    "trade",
                )

                # Execute sells first (free up capital)
                if sells and self.broker:
                    for r in sells[:3]:
                        try:
                            result = self.broker.close_position(r.ticker)
                            if "error" not in result:
                                self.bus.log_entry.emit(f"Agent SELL {r.ticker} ({r.confidence:.0%})", "trade")
                        except: pass

                # Execute buys (split remaining BP)
                if buys and self.broker and bp > 100:
                    bp_per = bp / min(len(buys), 5)  # Max 5 buys per cycle
                    for r in sorted(buys, key=lambda x: -x.confidence)[:5]:
                        if not getattr(self, '_agent_running', False):
                            break
                        try:
                            qty = max(1, int(bp_per / r.price)) if r.price > 0 else 0
                            if qty > 0:
                                result = self.broker.place_order(r.ticker, qty, "buy",
                                    stop_loss=r.stop_loss, take_profit=r.take_profit)
                                if "error" not in result:
                                    self.bus.log_entry.emit(
                                        f"Agent BUY {r.ticker} x{qty} ({r.confidence:.0%})", "trade")
                                    bp -= qty * r.price
                        except: pass

                self.bus.log_entry.emit(f"Agent cycle {cycle} complete. Next in 5 min.", "system")

            except Exception as e:
                self.bus.log_entry.emit(f"Agent error: {e}", "error")

            # Wait 5 minutes between cycles
            for _ in range(300):
                if not getattr(self, '_agent_running', False):
                    break
                time.sleep(1)

        self.bus.log_entry.emit("Autonomous agent stopped", "system")
