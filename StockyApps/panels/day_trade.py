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

# PANEL: DAY TRADE
# ═════════════════════════════════════════════════════════════════════════════

class DayTradePanel(QWidget):
    """Single-stock intraday analysis with chart and signals."""

    def __init__(self, broker, risk_manager, event_bus):
        super().__init__()
        self.broker = broker
        self.rm = risk_manager
        self.bus = event_bus
        self.model = None
        self.features = []
        self._build()
        self.bus.ticker_selected.connect(self._set_ticker)

    def _build(self):
        from core.ui.backgrounds import GradientHeader
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        header = GradientHeader("Day Trade", "Single-stock intraday analysis")
        layout.addWidget(header)

        # Controls row
        row = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ticker")
        self.ticker_input.setFixedWidth(120)
        row.addWidget(self.ticker_input)
        self.period_cb = QComboBox()
        self.period_cb.addItems(["5d", "3d", "2d", "1d"])
        row.addWidget(self.period_cb)
        self.interval_cb = QComboBox()
        self.interval_cb.addItems(["5m", "1m", "15m", "30m"])
        row.addWidget(self.interval_cb)
        self.run_btn = QPushButton("Analyze")
        self.run_btn.clicked.connect(self._analyze)
        row.addWidget(self.run_btn)
        row.addStretch()
        layout.addLayout(row)

        # Signal display — premium animated badge
        from core.widgets import SignalBadge, GradientDivider
        sig_row = QHBoxLayout()
        self.signal_badge = SignalBadge()
        sig_row.addWidget(self.signal_badge)
        self.stats_lbl = QLabel("")
        self.stats_lbl.setFont(QFont(FONT_MONO, 10))
        sig_row.addWidget(self.stats_lbl)
        layout.addLayout(sig_row)
        layout.addWidget(GradientDivider())

        # Chart
        self.figure = plt.Figure(figsize=(8, 4), dpi=100, facecolor=chart_colors()["fig_bg"])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Progress
        from core.widgets import DetailedProgressBar
        self.progress = DetailedProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.setLayout(layout)

    def _set_ticker(self, ticker):
        self.ticker_input.setText(ticker)
        self._analyze()

    def _analyze(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            return

        # Market hours check
        est = pytz.timezone("US/Eastern")
        now = datetime.now(est)
        if now.weekday() >= 5 or now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
            self.bus.log_entry.emit(f"Market closed — analyzing {ticker} with latest data", "warn")

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.reset()
        self.progress.set_progress(10, f"Fetching {ticker} data...", "Downloading from Yahoo Finance")
        self.progress.add_log(f"Fetching {ticker} {self.period_cb.currentText()} @ {self.interval_cb.currentText()}")
        self.bus.log_entry.emit(f"Analyzing {ticker}...", "info")
        QApplication.processEvents()

        data = fetch_intraday(ticker, self.period_cb.currentText(), self.interval_cb.currentText())
        if data.empty or len(data) < 30:
            self.bus.log_entry.emit(f"{ticker}: not enough data", "error")
            self.progress.set_progress(100, "Failed — not enough data", "Try a longer period")
            self.run_btn.setEnabled(True)
            return

        self.progress.set_progress(40, "Training LightGBM model...", f"{len(data)} bars, computing features")
        self.progress.add_log(f"Got {len(data)} bars — training model")

        feats = get_all_features("intraday")
        self._worker = TrainWorker(data, feats, ticker)
        self._worker.finished.connect(lambda m, f, d: self._on_done(ticker, m, f, d))
        self._worker.start()

    def _on_done(self, ticker, model, features, data):
        self.run_btn.setEnabled(True)
        if model is None:
            self.progress.set_progress(100, "Training failed", "Not enough usable data")
            self.bus.log_entry.emit(f"{ticker}: training failed", "error")
            return

        self.progress.set_progress(80, "Running predictions...", f"{len(features)} features")
        self.progress.add_log(f"Model trained with {len(features)} features")

        self.model = model
        self.features = features
        actions, confs, probs = predict_lgbm(model, data, features)
        act = LABEL_NAMES[actions[-1]]
        conf = confs[-1]
        price = data["Close"].iloc[-1]
        p = probs[-1]
        atr = data["ATRr_14"].iloc[-1] if "ATRr_14" in data.columns else price * 0.01

        size = self.rm.position_size(price, atr)
        side = "buy" if act == "BUY" else "sell"
        sl = self.rm.stop_loss(price, atr, side)
        tp = self.rm.take_profit(price, atr, side)

        self.signal_badge.set_signal(f"{act} {ticker}", conf)
        self.stats_lbl.setText(
            f"${price:.2f} | Conf: {conf:.0%} | "
            f"SELL {p[0]:.0%} HOLD {p[1]:.0%} BUY {p[2]:.0%}\n"
            f"Size: {size} | SL ${sl:.2f} | TP ${tp:.2f} | ATR ${atr:.2f}"
        )

        write_signal(ticker, act, conf, price, size, sl, tp, atr)
        log_decision(ticker, act, conf, price, size, sl, tp, atr, list(p), reasoning=f"DayTrade analysis")
        self.bus.signal_generated.emit(ticker, act, {"conf": conf, "price": price, "size": size})
        self.bus.log_entry.emit(f"{act} {ticker} @ ${price:.2f} ({conf:.0%})", "trade")

        self.progress.set_progress(100, f"Done — {act} {ticker}", f"Confidence: {conf:.0%}")
        self.progress.add_log(f"Signal: {act} @ ${price:.2f} ({conf:.0%})")

        # Chart
        self.figure.clear()
        cc = chart_colors(); self.figure.set_facecolor(cc["fig_bg"])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(cc["ax_bg"])
        x = range(len(data))
        closes = data["Close"].values
        ax.plot(x, closes, color=CHART_PRICE, linewidth=1.5, label="Price")
        if "vwap" in data.columns:
            ax.plot(x, data["vwap"].values, color=CHART_VWAP, linewidth=1, alpha=0.7, label="VWAP")
        if "EMA_9" in data.columns:
            ax.plot(x, data["EMA_9"].values, color=CHART_EMA_FAST, linewidth=0.7, alpha=0.5, label="EMA9")
        if "EMA_21" in data.columns:
            ax.plot(x, data["EMA_21"].values, color=CHART_EMA_SLOW, linewidth=0.7, alpha=0.5, label="EMA21")
        buy_m = actions == 2
        sell_m = actions == 0
        if buy_m.any():
            ax.scatter(np.where(buy_m)[0], closes[buy_m], marker="^", color=COLOR_BUY, s=40, zorder=5)
        if sell_m.any():
            ax.scatter(np.where(sell_m)[0], closes[sell_m], marker="v", color=COLOR_SELL, s=40, zorder=5)
        ax.set_title(f"{ticker} Intraday", color=cc["text"], fontsize=12)
        ax.tick_params(colors=cc["muted"], labelsize=8)
        ax.grid(True, alpha=0.15, color=cc["grid"])
        ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT_SECONDARY)
        self.figure.tight_layout()
        self.canvas.draw()

        # Hover tooltip
        from core.ui.chart_tooltip import ChartTooltip
        self._tooltip = ChartTooltip(self.canvas, ax, list(range(len(closes))), list(closes))


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: LONG TRADE
# ═════════════════════════════════════════════════════════════════════════════

