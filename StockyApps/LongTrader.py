"""
LongTrader — Long-term stock outlook using daily bars.

Same LightGBM engine as DayTrader but tuned for swing/position trading:
- Uses SMA 50/200 golden cross instead of VWAP
- Wider triple barrier thresholds (more room for trades to play out)
- Longer rolling windows (50-day stats)
"""

import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QProgressBar, QComboBox, QGroupBox, QGridLayout,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from core.data import fetch_longterm, get_all_features
from core.model import train_lgbm, predict_lgbm
from core.labeling import LABEL_NAMES
from core.chart import (
    style_axis, plot_buy_sell_markers, BG_DARK,
    COLOR_PRICE, COLOR_SMA_50, COLOR_SMA_200, COLOR_BUY, COLOR_SELL, COLOR_HOLD,
)
from core.style import APP_STYLESHEET, log_html


# ─── Background training thread ─────────────────────────────────────────────
class TrainingWorker(QThread):
    finished = pyqtSignal(object, list, object)

    def __init__(self, data, ticker):
        super().__init__()
        self.data = data
        self.ticker = ticker

    def run(self):
        model, features = train_lgbm(
            self.data, get_all_features("longterm"), self.ticker, prefix="lgbm_long", min_samples=50
        )
        self.finished.emit(model, features, self.data)


# ─── Main Window ─────────────────────────────────────────────────────────────
class LongTraderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stocky Long Trader — LightGBM")
        self.setGeometry(100, 100, 1300, 900)
        self.setStyleSheet(APP_STYLESHEET)

        self.model = None
        self.features = []
        self.stock_ticker = ""

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # ── Controls ──
        controls = QGroupBox("Long-Term Analysis")
        grid = QGridLayout()

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ticker (e.g. AAPL, VOO, SPY)")
        grid.addWidget(QLabel("Ticker:"), 0, 0)
        grid.addWidget(self.ticker_input, 0, 1)

        self.period_combo = QComboBox()
        self.period_combo.addItems(["1y", "6mo", "3mo", "2y", "5y"])
        grid.addWidget(QLabel("Training Period:"), 0, 2)
        grid.addWidget(self.period_combo, 0, 3)

        self.run_btn = QPushButton("ANALYZE")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 1, 0, 1, 4)

        controls.setLayout(grid)
        layout.addWidget(controls)

        # ── Outlook display ──
        outlook_box = QGroupBox("Outlook")
        sig_layout = QHBoxLayout()

        self.signal_label = QLabel("WAITING")
        self.signal_label.setFont(QFont("Consolas", 28, QFont.Bold))
        self.signal_label.setAlignment(Qt.AlignCenter)
        self.signal_label.setStyleSheet("color: #666;")
        sig_layout.addWidget(self.signal_label)

        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont("Consolas", 11))
        sig_layout.addWidget(self.stats_label)

        outlook_box.setLayout(sig_layout)
        layout.addWidget(outlook_box)

        # ── Chart ──
        self.figure = plt.Figure(figsize=(10, 5), dpi=100, facecolor=BG_DARK)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # ── Log ──
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(150)
        layout.addWidget(self.log_box)

        # ── Progress ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.setLayout(layout)

    def _log(self, msg, level="info"):
        self.log_box.append(log_html(msg, level))

    def _on_run(self):
        self.stock_ticker = self.ticker_input.text().strip().upper()
        if not self.stock_ticker:
            self._log("Enter a valid ticker.", "warn")
            return

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self._log(f"Fetching {self.stock_ticker} data...")

        data = fetch_longterm(self.stock_ticker, period=self.period_combo.currentText())

        if data.empty or len(data) < 50:
            self._log("Not enough data.", "error")
            self.run_btn.setEnabled(True)
            self.progress.setVisible(False)
            return

        self._log(f"Got {len(data)} days. Training...")
        self._worker = TrainingWorker(data, self.stock_ticker)
        self._worker.finished.connect(self._on_training_done)
        self._worker.start()

    def _on_training_done(self, model, features, data):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)

        if model is None:
            self._log("Training failed.", "error")
            return

        self.model = model
        self.features = features

        actions, confidences, probs = predict_lgbm(model, data, features)

        last_action = LABEL_NAMES[actions[-1]]
        last_conf = confidences[-1]
        last_price = data["Close"].iloc[-1]
        last_probs = probs[-1]

        colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
        self.signal_label.setText(last_action)
        self.signal_label.setStyleSheet(f"color: {colors[last_action]};")

        self.stats_label.setText(
            f"Price: ${last_price:.2f}\n"
            f"Confidence: {last_conf:.1%}\n"
            f"SELL {last_probs[0]:.1%} | HOLD {last_probs[1]:.1%} | BUY {last_probs[2]:.1%}"
        )

        self._log(f"Outlook: {last_action} ({last_conf:.1%}) @ ${last_price:.2f}")
        self._update_chart(data, actions)

    def _update_chart(self, data, actions):
        self.figure.clear()
        self.figure.set_facecolor(BG_DARK)
        ax = self.figure.add_subplot(111)

        x = range(len(data))
        closes = data["Close"].values

        ax.plot(x, closes, color=COLOR_PRICE, linewidth=1.5, label="Price")
        if "SMA_50" in data.columns:
            ax.plot(x, data["SMA_50"].values, color=COLOR_SMA_50, linewidth=1, alpha=0.7, label="SMA 50")
        if "SMA_200" in data.columns:
            ax.plot(x, data["SMA_200"].values, color=COLOR_SMA_200, linewidth=1, alpha=0.7, label="SMA 200")

        plot_buy_sell_markers(ax, x, closes, actions)
        style_axis(ax, f"{self.stock_ticker} — Long-Term Analysis")

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LongTraderApp()
    window.show()
    sys.exit(app.exec_())
