"""
DayTrader — Aggressive intraday stock analysis and signal generation.

Uses LightGBM on technical indicators + FinBERT sentiment to produce
BUY/SELL/HOLD signals with ATR-based risk management.

Signals are written to signal.json for StockExecuter to consume.
"""

import sys
import time
import numpy as np
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QProgressBar, QComboBox, QGroupBox, QGridLayout,
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pytz

from core.data import fetch_intraday, get_all_features
from core.model import train_lgbm, predict_lgbm
from core.risk import RiskManager
from core.signals import write_signal
from core.labeling import LABEL_NAMES
from core.chart import (
    style_axis, plot_buy_sell_markers, BG_DARK,
    COLOR_PRICE, COLOR_VWAP, COLOR_EMA_FAST, COLOR_EMA_SLOW,
    COLOR_BUY, COLOR_SELL, COLOR_HOLD,
)
from core.style import APP_STYLESHEET, log_html


# ─── Background training thread ─────────────────────────────────────────────
class TrainingWorker(QThread):
    """Trains LightGBM in a background thread so the UI stays responsive."""
    finished = pyqtSignal(object, list, object)  # (model, features, data)

    def __init__(self, data, ticker):
        super().__init__()
        self.data = data
        self.ticker = ticker

    def run(self):
        model, features = train_lgbm(self.data, get_all_features("intraday"), self.ticker)
        self.finished.emit(model, features, self.data)


# ─── Main Window ─────────────────────────────────────────────────────────────
class DayTraderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stocky Day Trader — LightGBM + FinBERT")
        self.setGeometry(100, 100, 1300, 900)
        self.setStyleSheet(APP_STYLESHEET)

        self.model = None
        self.features = []
        self.risk_manager = RiskManager()
        self.stock_ticker = ""

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # ── Controls ──
        controls = QGroupBox("Trading Controls")
        grid = QGridLayout()

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ticker (e.g. AAPL, TSLA, SPY)")
        grid.addWidget(QLabel("Ticker:"), 0, 0)
        grid.addWidget(self.ticker_input, 0, 1)

        self.period_combo = QComboBox()
        self.period_combo.addItems(["5d", "3d", "2d", "1d"])
        grid.addWidget(QLabel("Training Data:"), 0, 2)
        grid.addWidget(self.period_combo, 0, 3)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1m", "5m", "15m", "30m"])
        grid.addWidget(QLabel("Interval:"), 0, 4)
        grid.addWidget(self.interval_combo, 0, 5)

        self.run_btn = QPushButton("ANALYZE & PREDICT")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 1, 0, 1, 6)

        controls.setLayout(grid)
        layout.addWidget(controls)

        # ── Signal display ──
        signal_box = QGroupBox("Current Signal")
        sig_layout = QHBoxLayout()

        self.signal_label = QLabel("WAITING")
        self.signal_label.setFont(QFont("Consolas", 28, QFont.Bold))
        self.signal_label.setAlignment(Qt.AlignCenter)
        self.signal_label.setStyleSheet("color: #666; padding: 10px;")
        sig_layout.addWidget(self.signal_label)

        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont("Consolas", 11))
        self.stats_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        sig_layout.addWidget(self.stats_label)

        signal_box.setLayout(sig_layout)
        layout.addWidget(signal_box)

        # ── Price chart ──
        self.figure = plt.Figure(figsize=(10, 5), dpi=100, facecolor=BG_DARK)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # ── Activity log ──
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(180)
        layout.addWidget(self.log_box)

        # ── Progress bar ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ── Refresh timer ──
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_run)
        self.tick_timer = QTimer()
        self.tick_timer.timeout.connect(self._update_countdown)
        self.tick_timer.setInterval(1000)
        self.countdown_label = QLabel("")
        layout.addWidget(self.countdown_label)

        self.setLayout(layout)

    # ── Logging helper ────────────────────────────────────────────────────

    def _log(self, msg, level="info"):
        self.log_box.append(log_html(msg, level))

    # ── Countdown display ─────────────────────────────────────────────────

    def _update_countdown(self):
        remaining = self.timer.remainingTime() / 1000
        if remaining > 0:
            self.countdown_label.setText(
                f"Next refresh in {time.strftime('%M:%S', time.gmtime(remaining))}"
            )
        else:
            self.tick_timer.stop()

    # ── Main analysis flow ────────────────────────────────────────────────

    def _on_run(self):
        self.stock_ticker = self.ticker_input.text().strip().upper()
        self.timer.stop()
        self.tick_timer.stop()

        if not self.stock_ticker:
            self._log("Enter a valid ticker.", "warn")
            return

        # Check if market is open (9:30 AM - 4:00 PM ET, weekdays)
        est = pytz.timezone("US/Eastern")
        now = datetime.now(est)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now.weekday() >= 5:
            self._log("Market closed (weekend).", "warn")
            return

        if now < market_open or now >= market_close:
            self._log("Market closed. Timer set for next open.", "warn")
            wait = market_open - now if now < market_open else (market_open + timedelta(days=1)) - now
            self.timer.start(int(wait.total_seconds() * 1000))
            self.tick_timer.start()
            return

        # Fetch data
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self._log(f"Fetching {self.stock_ticker} data...")

        data = fetch_intraday(
            self.stock_ticker,
            period=self.period_combo.currentText(),
            interval=self.interval_combo.currentText(),
        )

        if data.empty or len(data) < 30:
            self._log("Not enough data — try a longer period.", "error")
            self.run_btn.setEnabled(True)
            self.progress.setVisible(False)
            return

        self._log(f"Got {len(data)} bars. Training model...")

        # Train in background thread
        self._worker = TrainingWorker(data, self.stock_ticker)
        self._worker.finished.connect(self._on_training_done)
        self._worker.start()

    def _on_training_done(self, model, features, data):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)

        if model is None:
            self._log("Training failed — not enough usable data.", "error")
            return

        self.model = model
        self.features = features
        self._log(f"Model trained with {len(features)} features.")

        # Run predictions on the full dataset
        actions, confidences, probs = predict_lgbm(model, data, features)

        # Extract latest prediction
        last_action_name = LABEL_NAMES[actions[-1]]
        last_conf = confidences[-1]
        last_price = data["Close"].iloc[-1]
        last_probs = probs[-1]
        atr = data["ATRr_14"].iloc[-1] if "ATRr_14" in data.columns else last_price * 0.01

        # Risk management calculations
        can_trade, reason = self.risk_manager.can_trade()
        size = self.risk_manager.position_size(last_price, atr)
        side = "buy" if last_action_name == "BUY" else "sell"
        sl = self.risk_manager.stop_loss(last_price, atr, side)
        tp = self.risk_manager.take_profit(last_price, atr, side)

        # Update signal display
        colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
        self.signal_label.setText(last_action_name)
        self.signal_label.setStyleSheet(f"color: {colors[last_action_name]}; font-size: 28px;")

        self.stats_label.setText(
            f"Price: ${last_price:.2f}  |  ATR: ${atr:.2f}\n"
            f"Confidence: {last_conf:.1%}\n"
            f"Probs: SELL {last_probs[0]:.1%}  HOLD {last_probs[1]:.1%}  BUY {last_probs[2]:.1%}\n"
            f"Size: {size} shares  |  SL: ${sl:.2f}  |  TP: ${tp:.2f}\n"
            f"Can Trade: {'Yes' if can_trade else f'No — {reason}'}"
        )

        self._log(
            f"SIGNAL: {last_action_name} | Conf: {last_conf:.1%} | "
            f"${last_price:.2f} | {size} shr | SL ${sl:.2f} | TP ${tp:.2f}",
            "trade",
        )

        # Write signal for StockExecuter
        write_signal(self.stock_ticker, last_action_name, last_conf,
                     last_price, size, sl, tp, atr)

        # Update chart
        self._update_chart(data, actions)

        # Schedule next refresh based on interval
        ms_map = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000}
        self.timer.start(ms_map.get(self.interval_combo.currentText(), 300_000))
        self.tick_timer.start()

    # ── Chart ─────────────────────────────────────────────────────────────

    def _update_chart(self, data, actions):
        self.figure.clear()
        self.figure.set_facecolor(BG_DARK)
        ax = self.figure.add_subplot(111)

        x = range(len(data))
        closes = data["Close"].values

        # Price + overlays
        ax.plot(x, closes, color=COLOR_PRICE, linewidth=1.5, label="Price")

        if "vwap" in data.columns:
            ax.plot(x, data["vwap"].values, color=COLOR_VWAP, linewidth=1, alpha=0.7, label="VWAP")
        if "EMA_9" in data.columns:
            ax.plot(x, data["EMA_9"].values, color=COLOR_EMA_FAST, linewidth=0.8, alpha=0.6, label="EMA 9")
        if "EMA_21" in data.columns:
            ax.plot(x, data["EMA_21"].values, color=COLOR_EMA_SLOW, linewidth=0.8, alpha=0.6, label="EMA 21")
        if "BBU_20_2.0" in data.columns:
            ax.fill_between(x, data["BBL_20_2.0"].values, data["BBU_20_2.0"].values, alpha=0.1, color="white")

        # Buy/Sell markers
        plot_buy_sell_markers(ax, x, closes, actions)
        style_axis(ax, f"{self.stock_ticker} — Intraday Analysis")

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DayTraderApp()
    window.show()
    sys.exit(app.exec_())
