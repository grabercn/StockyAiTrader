"""
StockExecuter — Executes trades on Alpaca based on DayTrader signals.

Polls signal.json every 5 seconds for new signals from DayTrader.
Can auto-execute trades or wait for manual confirmation.

Features:
- Bracket orders with automatic stop-loss and take-profit
- Live positions table with P&L
- Daily portfolio chart
- Confidence threshold filter
- Emergency "close all" button
"""

import sys
import json
import os
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QTextEdit, QDialog, QLineEdit, QPushButton, QFormLayout, QAction,
    QMessageBox, QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from core.broker import AlpacaBroker
from core.signals import read_signal
from core.chart import BG_DARK, COLOR_PRICE, COLOR_BUY, COLOR_SELL
from core.style import APP_STYLESHEET, log_html

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "settings.json")


# ─── Settings Dialog ─────────────────────────────────────────────────────────
class SettingsDialog(QDialog):
    """Dialog for entering Alpaca API credentials."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Settings")
        self.setStyleSheet(APP_STYLESHEET)
        self.setMinimumWidth(400)

        layout = QFormLayout(self)
        self.api_key = QLineEdit()
        self.secret_key = QLineEdit()
        self.secret_key.setEchoMode(QLineEdit.Password)

        layout.addRow("Alpaca API Key:", self.api_key)
        layout.addRow("Alpaca Secret Key:", self.secret_key)

        btns = QHBoxLayout()
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(self._test)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        btns.addWidget(test_btn)
        btns.addWidget(save_btn)
        layout.addRow(btns)

    def load(self, settings):
        self.api_key.setText(settings.get("alpaca_api_key", ""))
        self.secret_key.setText(settings.get("alpaca_secret_key", ""))

    def _test(self):
        broker = AlpacaBroker(self.api_key.text(), self.secret_key.text())
        acct = broker.get_account()
        if "error" in acct:
            QMessageBox.warning(self, "Error", f"Connection failed:\n{acct['error']}")
        else:
            QMessageBox.information(
                self, "Connected",
                f"Portfolio: ${float(acct.get('portfolio_value', 0)):,.2f}\n"
                f"Buying Power: ${float(acct.get('buying_power', 0)):,.2f}",
            )

    def _save(self):
        data = {}
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
        data["alpaca_api_key"] = self.api_key.text()
        data["alpaca_secret_key"] = self.secret_key.text()
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        QMessageBox.information(self, "Saved", "Settings saved.")
        self.accept()


# ─── Main Dashboard ──────────────────────────────────────────────────────────
class ExecuterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stocky Executer — Risk-Managed Trading")
        self.setGeometry(100, 100, 1100, 750)
        self.setStyleSheet(APP_STYLESHEET)

        self.broker = None
        self.settings = self._load_settings()
        self.auto_execute = False
        self.min_confidence = 60
        self._last_signal_time = None

        self._build_ui()
        self._init_broker()
        self._refresh_dashboard()

        # Poll for new signals every 5 seconds
        self._signal_timer = QTimer()
        self._signal_timer.timeout.connect(self._check_signal)
        self._signal_timer.start(5000)

        # Refresh dashboard every 30 seconds
        self._dash_timer = QTimer()
        self._dash_timer.timeout.connect(self._refresh_dashboard)
        self._dash_timer.start(30_000)

    # ── UI Construction ───────────────────────────────────────────────────

    def _build_ui(self):
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Settings")
        edit_action = QAction("Edit API Keys", self)
        edit_action.triggered.connect(self._open_settings)
        settings_menu.addAction(edit_action)

        main = QWidget()
        layout = QHBoxLayout()

        # Left: account + chart + positions
        left = QVBoxLayout()

        # Account stats
        acct_box = QGroupBox("Account")
        acct_layout = QGridLayout()
        self.portfolio_lbl = QLabel("Portfolio: --")
        self.portfolio_lbl.setFont(QFont("Consolas", 14, QFont.Bold))
        self.buying_power_lbl = QLabel("Buying Power: --")
        self.cash_lbl = QLabel("Cash: --")
        self.pnl_lbl = QLabel("Day P&L: --")
        acct_layout.addWidget(self.portfolio_lbl, 0, 0, 1, 2)
        acct_layout.addWidget(self.buying_power_lbl, 1, 0)
        acct_layout.addWidget(self.cash_lbl, 1, 1)
        acct_layout.addWidget(self.pnl_lbl, 2, 0, 1, 2)
        acct_box.setLayout(acct_layout)
        left.addWidget(acct_box)

        # Portfolio chart
        self.figure = plt.Figure(figsize=(6, 3), dpi=100, facecolor=BG_DARK)
        self.canvas = FigureCanvas(self.figure)
        left.addWidget(self.canvas)

        # Positions table
        pos_box = QGroupBox("Open Positions")
        pos_layout = QVBoxLayout()
        self.pos_table = QTableWidget()
        self.pos_table.setColumnCount(6)
        self.pos_table.setHorizontalHeaderLabels(
            ["Symbol", "Qty", "Avg Cost", "Current", "P&L", "P&L %"]
        )
        self.pos_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        pos_layout.addWidget(self.pos_table)

        close_all_btn = QPushButton("CLOSE ALL POSITIONS")
        close_all_btn.setStyleSheet("background-color: #cc0000;")
        close_all_btn.clicked.connect(self._close_all)
        pos_layout.addWidget(close_all_btn)

        pos_box.setLayout(pos_layout)
        left.addWidget(pos_box)

        # Right: signal + controls + log
        right = QVBoxLayout()

        # Signal display
        sig_box = QGroupBox("Latest Signal from DayTrader")
        sig_layout = QVBoxLayout()
        self.signal_lbl = QLabel("No signal yet")
        self.signal_lbl.setFont(QFont("Consolas", 16, QFont.Bold))
        self.signal_lbl.setAlignment(Qt.AlignCenter)
        self.signal_detail = QLabel("")
        self.signal_detail.setFont(QFont("Consolas", 10))
        sig_layout.addWidget(self.signal_lbl)
        sig_layout.addWidget(self.signal_detail)
        sig_box.setLayout(sig_layout)
        right.addWidget(sig_box)

        # Execution controls
        ctrl_box = QGroupBox("Execution Controls")
        ctrl_layout = QGridLayout()

        self.auto_cb = QCheckBox("Auto-Execute Trades")
        self.auto_cb.toggled.connect(lambda v: setattr(self, "auto_execute", v))
        ctrl_layout.addWidget(self.auto_cb, 0, 0)

        ctrl_layout.addWidget(QLabel("Min Confidence %:"), 0, 1)
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(30, 99)
        self.conf_spin.setValue(60)
        self.conf_spin.valueChanged.connect(lambda v: setattr(self, "min_confidence", v))
        ctrl_layout.addWidget(self.conf_spin, 0, 2)

        buy_btn = QPushButton("Manual BUY")
        buy_btn.setStyleSheet("background-color: #006600;")
        buy_btn.clicked.connect(lambda: self._manual("buy"))
        ctrl_layout.addWidget(buy_btn, 1, 0)

        sell_btn = QPushButton("Manual SELL")
        sell_btn.setStyleSheet("background-color: #cc0000;")
        sell_btn.clicked.connect(lambda: self._manual("sell"))
        ctrl_layout.addWidget(sell_btn, 1, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_dashboard)
        ctrl_layout.addWidget(refresh_btn, 1, 2)

        ctrl_box.setLayout(ctrl_layout)
        right.addWidget(ctrl_box)

        # Activity log
        log_box = QGroupBox("Activity Log")
        log_layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)
        log_box.setLayout(log_layout)
        right.addWidget(log_box)

        layout.addLayout(left, 3)
        layout.addLayout(right, 2)
        main.setLayout(layout)
        self.setCentralWidget(main)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log(self, msg, level="info"):
        self.log_box.append(log_html(msg, level))

    def _load_settings(self):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _init_broker(self):
        key = self.settings.get("alpaca_api_key")
        secret = self.settings.get("alpaca_secret_key")
        if key and secret:
            self.broker = AlpacaBroker(key, secret)
            self._log("Alpaca API connected (paper mode).")
        else:
            self._log("API keys not set — go to Settings.", "warn")

    def _open_settings(self):
        dialog = SettingsDialog(self)
        dialog.load(self.settings)
        if dialog.exec_():
            self.settings = self._load_settings()
            self._init_broker()
            self._refresh_dashboard()

    # ── Dashboard ─────────────────────────────────────────────────────────

    def _refresh_dashboard(self):
        if not self.broker:
            return

        acct = self.broker.get_account()
        if "error" in acct:
            self._log(f"Account error: {acct['error']}", "error")
            return

        # Update account stats
        portfolio = float(acct.get("portfolio_value", 0))
        buying = float(acct.get("buying_power", 0))
        cash = float(acct.get("cash", 0))
        equity = float(acct.get("equity", 0))
        last_eq = float(acct.get("last_equity", equity))
        day_pnl = equity - last_eq
        day_pct = (day_pnl / last_eq * 100) if last_eq > 0 else 0

        self.portfolio_lbl.setText(f"Portfolio: ${portfolio:,.2f}")
        self.buying_power_lbl.setText(f"Buying Power: ${buying:,.2f}")
        self.cash_lbl.setText(f"Cash: ${cash:,.2f}")
        pnl_color = COLOR_BUY if day_pnl >= 0 else COLOR_SELL
        self.pnl_lbl.setText(f"Day P&L: ${day_pnl:,.2f} ({day_pct:+.2f}%)")
        self.pnl_lbl.setStyleSheet(f"color: {pnl_color}; font-weight: bold;")

        # Update positions table
        positions = self.broker.get_positions()
        if isinstance(positions, list):
            self.pos_table.setRowCount(len(positions))
            for i, pos in enumerate(positions):
                pnl = float(pos.get("unrealized_pl", 0))
                values = [
                    pos.get("symbol", ""),
                    f"{float(pos.get('qty', 0)):.0f}",
                    f"${float(pos.get('avg_entry_price', 0)):.2f}",
                    f"${float(pos.get('current_price', 0)):.2f}",
                    f"${pnl:,.2f}",
                    f"{float(pos.get('unrealized_plpc', 0)) * 100:+.2f}%",
                ]
                for j, val in enumerate(values):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignCenter)
                    if j >= 4:
                        item.setForeground(QColor(COLOR_BUY if pnl >= 0 else COLOR_SELL))
                    self.pos_table.setItem(i, j, item)

        # Update portfolio chart
        history = self.broker.get_portfolio_history()
        if "error" not in history and history.get("equity"):
            self._plot_portfolio(history)

    def _plot_portfolio(self, history):
        self.figure.clear()
        self.figure.set_facecolor(BG_DARK)
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#16213e")

        equity = history["equity"]
        timestamps = [datetime.fromtimestamp(t) for t in history["timestamp"]]

        ax.plot(timestamps, equity, color=COLOR_PRICE, linewidth=1.5)
        ax.fill_between(timestamps, equity, alpha=0.1, color=COLOR_PRICE)
        ax.set_title("Portfolio (1 Week)", color="white", fontsize=10)
        ax.set_ylabel("$", color="white")
        ax.tick_params(colors="#888", labelsize=8)
        ax.grid(True, alpha=0.15, color="#444")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d %H:%M"))
        self.figure.autofmt_xdate()
        self.figure.tight_layout()
        self.canvas.draw()

    # ── Signal Handling ───────────────────────────────────────────────────

    def _check_signal(self):
        """Poll signal.json for new signals from DayTrader."""
        signal = read_signal()
        if not signal:
            return

        # Skip if we've already processed this signal
        sig_time = signal.get("timestamp", "")
        if sig_time == self._last_signal_time:
            return
        self._last_signal_time = sig_time

        action = signal.get("action", "HOLD")
        ticker = signal.get("ticker", "???")
        conf = signal.get("confidence", 0)
        price = signal.get("price", 0)
        size = signal.get("position_size", 0)
        sl = signal.get("stop_loss", 0)
        tp = signal.get("take_profit", 0)

        # Update signal display
        colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": "#ffaa00"}
        self.signal_lbl.setText(f"{action} {ticker}")
        self.signal_lbl.setStyleSheet(f"color: {colors.get(action, '#666')}; font-size: 20px;")
        self.signal_detail.setText(
            f"Price: ${price:.2f} | Size: {size} | Conf: {conf:.1%}\n"
            f"SL: ${sl:.2f} | TP: ${tp:.2f}"
        )

        self._log(f"Signal: {action} {ticker} x{size} @ ${price:.2f} ({conf:.1%})", "trade")

        # Auto-execute if enabled and confident enough
        if self.auto_execute and action in ("BUY", "SELL") and conf * 100 >= self.min_confidence:
            self._execute(ticker, size, action.lower(), sl, tp)
        elif action == "HOLD":
            self._log("HOLD signal — no action.")

    def _execute(self, ticker, qty, side, sl=None, tp=None):
        if not self.broker:
            self._log("Cannot trade — API not configured.", "error")
            return
        if qty <= 0:
            self._log("Position size is 0 — skipping.", "warn")
            return

        self._log(f"Placing {side.upper()}: {ticker} x{qty} (SL ${sl:.2f}, TP ${tp:.2f})", "trade")
        result = self.broker.place_order(ticker, qty, side, stop_loss=sl, take_profit=tp)

        if "error" in result:
            self._log(f"Order failed: {result['error']}", "error")
        else:
            self._log(f"Order placed — ID: {result.get('id', '?')}", "trade")

        self._refresh_dashboard()

    def _manual(self, side):
        signal = read_signal()
        if not signal:
            self._log("No signal available.", "warn")
            return
        self._execute(signal["ticker"], signal["position_size"], side,
                      signal.get("stop_loss"), signal.get("take_profit"))

    def _close_all(self):
        if not self.broker:
            return
        reply = QMessageBox.question(self, "Confirm", "Close ALL positions?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            result = self.broker.close_all_positions()
            if "error" in result:
                self._log(f"Close all failed: {result['error']}", "error")
            else:
                self._log("All positions closed.", "trade")
            self._refresh_dashboard()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExecuterApp()
    window.show()
    sys.exit(app.exec_())
