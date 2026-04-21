"""AI Trading Dashboard — centralized view of all AI-managed stocks."""
import sys, os, json, time
import numpy as np
from datetime import datetime, timedelta
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QLinearGradient, QPen
from core.branding import *
from core.branding import chart_colors
from core.event_bus import EventBus

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "settings.json")
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f: return json.load(f)
    except: return {}


class AIDashboardPanel(QWidget):
    """Mission control for the AI auto-trader — shows all managed stocks and agent activity."""

    def __init__(self, broker, event_bus):
        super().__init__()
        self.broker = broker
        self.bus = event_bus
        self._build()

        # Auto-refresh every 5 seconds
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start(5000)

        # Initial refresh after 2s
        QTimer.singleShot(2000, self.refresh)

    def _build(self):
        from core.ui.backgrounds import GradientHeader
        from core.widgets import StatCard

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        header = GradientHeader("AI Dashboard", "Auto-trader mission control — all managed stocks at a glance")
        layout.addWidget(header)

        # Status cards
        cards = QHBoxLayout()
        cards.setSpacing(8)
        self.card_monitored = StatCard("Monitored", "0", BRAND_ACCENT)
        self.card_trades_today = StatCard("Trades Today", "0", BRAND_PRIMARY)
        self.card_signals = StatCard("Active Signals", "0", BRAND_SECONDARY)
        self.card_mode = StatCard("Mode", "Idle", TEXT_MUTED)
        for c in [self.card_monitored, self.card_trades_today, self.card_signals, self.card_mode]:
            cards.addWidget(c)
        layout.addLayout(cards)

        # Main table — all monitored stocks
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Stock", "Mode", "Signal", "Confidence", "Price",
            "Last Check", "Next Check", "Interval", "Trades"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        # Activity log
        log_label = QLabel("Agent Activity Log")
        log_label.setFont(QFont(FONT_FAMILY, 11, QFont.Bold))
        log_label.setStyleSheet(f"color: {BRAND_PRIMARY};")
        layout.addWidget(log_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont(FONT_MONO, 9))
        self.log_area.setMaximumHeight(150)
        layout.addWidget(self.log_area)

        # Controls
        ctrl_row = QHBoxLayout()
        self.stop_all_btn = QPushButton("Stop All Monitoring")
        self.stop_all_btn.setStyleSheet(f"background-color: {COLOR_SELL}; padding: 6px 12px;")
        self.stop_all_btn.clicked.connect(self._stop_all)
        ctrl_row.addWidget(self.stop_all_btn)
        ctrl_row.addStretch()

        status_label = QLabel("Auto-refreshes every 5s")
        status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")
        ctrl_row.addWidget(status_label)
        layout.addLayout(ctrl_row)

        self.setLayout(layout)

        # Listen for log entries tagged as trade/system
        self.bus.log_entry.connect(self._on_log)

    def _on_log(self, msg, level):
        if level in ("trade", "system") and ("auto" in msg.lower() or "monitor" in msg.lower() or "signal" in msg.lower()):
            ts = datetime.now().strftime("%H:%M:%S")
            colors = {"trade": BRAND_ACCENT, "system": TEXT_MUTED}
            c = colors.get(level, TEXT_SECONDARY)
            self.log_area.append(f'<span style="color:{TEXT_MUTED}">{ts}</span> <span style="color:{c}">{msg}</span>')

    def refresh(self):
        """Update the dashboard with current auto-trade state."""
        main = self.window()
        if not main:
            return

        # Get auto-trade service from scanner
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

        # Get aggressivity
        settings = load_settings()
        aggr = settings.get("aggressivity", "Default")
        self.card_mode.set_value(aggr)

        # Count active signals
        buy_count = sum(1 for s in monitored.values() if s.last_signal == "BUY")
        sell_count = sum(1 for s in monitored.values() if s.last_signal == "SELL")
        self.card_signals.set_value(f"{buy_count}B / {sell_count}S")

        # Count trades today
        total_trades = sum(getattr(s, 'trades_today', 0) for s in monitored.values())
        self.card_trades_today.set_value(str(total_trades))

        # Populate table
        self.table.setRowCount(len(monitored))
        for i, (ticker, stock) in enumerate(monitored.items()):
            signal = getattr(stock, 'last_signal', 'HOLD')
            conf = getattr(stock, 'last_confidence', 0)
            price = getattr(stock, 'last_price', 0)
            last_check = getattr(stock, 'last_check', '--')
            next_secs = getattr(stock, 'next_check_seconds', 0)
            interval = getattr(stock, 'interval', '5m')
            check_count = getattr(stock, 'check_count', 0)
            auto_exec = getattr(stock, 'auto_execute', False)

            mins = next_secs // 60
            secs = next_secs % 60
            mode = "Auto-Execute" if auto_exec else "Signal Only"

            vals = [
                ticker, mode, signal, f"{conf:.0%}",
                f"${price:.2f}" if price else "--",
                str(last_check) if last_check else "--",
                f"{mins}m {secs}s",
                interval,
                str(check_count),
            ]

            signal_colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if j == 2:  # Signal column
                    item.setForeground(QColor(signal_colors.get(v, TEXT_MUTED)))
                    item.setFont(QFont(FONT_MONO, 11, QFont.Bold))
                if j == 1:  # Mode column
                    item.setForeground(QColor(BRAND_ACCENT if "Auto" in v else TEXT_MUTED))
                self.table.setItem(i, j, item)

    def _stop_all(self):
        main = self.window()
        if not main or not hasattr(main, 'scanner'):
            return
        svc = getattr(main.scanner, '_auto_service', None)
        if not svc:
            return

        confirm = QMessageBox.question(
            self, "Stop All Monitoring",
            f"Stop monitoring all {len(svc.get_monitored())} stocks?\n"
            f"Auto-trading will stop. You can restart from the Scanner.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        for ticker in list(svc.get_monitored().keys()):
            svc.remove_stock(ticker)
        self.bus.log_entry.emit("All stock monitoring stopped", "system")
        self.refresh()
