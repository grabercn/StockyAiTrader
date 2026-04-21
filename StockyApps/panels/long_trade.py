"""Panel module — auto-extracted from StockySuite.py"""
import sys, os, json, time
import numpy as np
from datetime import datetime, timedelta
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QLinearGradient, QPen
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pytz, yfinance as yf
from core.branding import *
from core.branding import get_stylesheet, detect_system_theme, chart_colors
from core.event_bus import EventBus
from core.risk import RiskManager
from core.broker import AlpacaBroker
from core.scanner import scan_multiple, ScanResult
from core.data import fetch_intraday, fetch_longterm, get_all_features
from core.model import train_lgbm, predict_lgbm
from core.labeling import LABEL_NAMES
from core.logger import log_decision, log_trade_execution, log_scan_results, log_event, get_today_logs, get_log_files, get_log_entries
from core.signals import write_signal
from core.model_manager import MANAGED_MODELS, get_model_status, get_lgbm_models, download_model, delete_model, delete_lgbm_model, delete_all_lgbm_models
from addons import get_all_addons, set_addon_enabled, discover_addons

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "settings.json")
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f: return json.load(f)
    except: return {}
def save_settings(s):
    with open(SETTINGS_FILE, "w") as f: json.dump(s, f, indent=4)

# PANEL: LONG TRADE
# ═════════════════════════════════════════════════════════════════════════════

class LongTradePanel(QWidget):
    """Long-term stock outlook analysis."""

    def __init__(self, event_bus):
        super().__init__()
        self.bus = event_bus
        self._build()

    def _build(self):
        from core.ui.backgrounds import GradientHeader
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        header = GradientHeader("Long Trade", "Long-term stock outlook analysis")
        layout.addWidget(header)

        row = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ticker")
        self.ticker_input.setFixedWidth(120)
        row.addWidget(self.ticker_input)
        self.period_cb = QComboBox()
        self.period_cb.addItems(["1y", "6mo", "3mo", "2y", "5y"])
        row.addWidget(self.period_cb)
        self.run_btn = QPushButton("Analyze")
        self.run_btn.clicked.connect(self._analyze)
        row.addWidget(self.run_btn)
        row.addStretch()
        layout.addLayout(row)

        self.signal_lbl = QLabel("—")
        self.signal_lbl.setFont(QFont(FONT_FAMILY, 24, QFont.Bold))
        self.signal_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.signal_lbl)

        self.stats_lbl = QLabel("")
        self.stats_lbl.setFont(QFont(FONT_MONO, 10))
        layout.addWidget(self.stats_lbl)

        self.figure = plt.Figure(figsize=(8, 4), dpi=100, facecolor=chart_colors()["fig_bg"])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        from core.widgets import DetailedProgressBar
        self.progress = DetailedProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.setLayout(layout)

    def _analyze(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            return
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.reset()
        self.progress.set_progress(10, f"Fetching {ticker} data...", "Daily bars from Yahoo Finance")
        self.progress.add_log(f"Fetching {ticker} {self.period_cb.currentText()}")
        QApplication.processEvents()

        data = fetch_longterm(ticker, self.period_cb.currentText())
        if data.empty or len(data) < 50:
            self.bus.log_entry.emit(f"{ticker}: not enough long-term data", "error")
            self.progress.set_progress(100, "Failed — not enough data")
            self.run_btn.setEnabled(True)
            return

        self.progress.set_progress(40, "Training model...", f"{len(data)} daily bars")
        self.progress.add_log(f"Got {len(data)} bars — training")

        feats = get_all_features("longterm")
        self._worker = TrainWorker(data, feats, ticker, prefix="lgbm_long")
        self._worker.finished.connect(lambda m, f, d: self._on_done(ticker, m, f, d))
        self._worker.start()

    def _on_done(self, ticker, model, features, data):
        self.run_btn.setEnabled(True)
        if model is None:
            self.progress.set_progress(100, "Training failed")
            return

        self.progress.set_progress(80, "Running predictions...", f"{len(features)} features")
        actions, confs, probs = predict_lgbm(model, data, features)
        act = LABEL_NAMES[actions[-1]]
        conf = confs[-1]
        price = data["Close"].iloc[-1]
        p = probs[-1]

        colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
        self.signal_lbl.setText(f"{act} {ticker}")
        self.signal_lbl.setStyleSheet(f"color: {colors[act]};")
        self.stats_lbl.setText(
            f"${price:.2f} | Conf: {conf:.0%} | SELL {p[0]:.0%} HOLD {p[1]:.0%} BUY {p[2]:.0%}"
        )
        log_decision(ticker, act, conf, price, 0, 0, 0, 0, list(p), reasoning="LongTrade analysis")
        self.bus.log_entry.emit(f"Long outlook: {act} {ticker} ({conf:.0%})", "trade")

        self.progress.set_progress(100, f"Done — {act} {ticker}", f"Confidence: {conf:.0%}")
        self.progress.add_log(f"Outlook: {act} @ ${price:.2f} ({conf:.0%})")

        # Chart
        self.figure.clear()
        cc = chart_colors(); self.figure.set_facecolor(cc["fig_bg"])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(cc["ax_bg"])
        x = range(len(data))
        closes = data["Close"].values
        ax.plot(x, closes, color=CHART_PRICE, linewidth=1.5, label="Price")
        if "SMA_50" in data.columns:
            ax.plot(x, data["SMA_50"].values, color=CHART_VWAP, linewidth=1, alpha=0.6, label="SMA50")
        if "SMA_200" in data.columns:
            ax.plot(x, data["SMA_200"].values, color=CHART_EMA_FAST, linewidth=1, alpha=0.6, label="SMA200")
        ax.set_title(f"{ticker} Long-Term", color=cc["text"], fontsize=12)
        ax.tick_params(colors=cc["muted"], labelsize=8)
        ax.grid(True, alpha=0.15, color=cc["grid"])
        ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT_SECONDARY)
        self.figure.tight_layout()
        self.canvas.draw()

        from core.ui.chart_tooltip import ChartTooltip
        self._tooltip = ChartTooltip(self.canvas, ax, list(range(len(closes))), list(closes))


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: LOGS
# ═════════════════════════════════════════════════════════════════════════════

