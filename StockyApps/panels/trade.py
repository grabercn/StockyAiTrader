"""Panel module — combined Trade panel (Intraday + Long-Term)"""
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
from core.data import fetch_intraday, fetch_longterm, get_all_features
from core.model import train_lgbm, predict_lgbm
from core.labeling import LABEL_NAMES
from core.logger import log_decision, log_event
import threading

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "settings.json")
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f: return json.load(f)
    except: return {}
def save_settings(s):
    with open(SETTINGS_FILE, "w") as f: json.dump(s, f, indent=4)


class TradePanel(QWidget):
    """Combined Intraday + Long-Term trade analysis."""

    _analysis_done = pyqtSignal(str, object, object, list, str)  # ticker, model, data, features, mode

    def __init__(self, broker, risk_manager, event_bus):
        super().__init__()
        self.broker = broker
        self.rm = risk_manager
        self.bus = event_bus
        self.model = None
        self.features = []
        self._analysis_done.connect(self._on_done)
        self._build()

    def _build(self):
        from core.ui.backgrounds import GradientHeader
        from core.widgets import DetailedProgressBar

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        header = GradientHeader("Trade", "Single-stock analysis — intraday or long-term")
        layout.addWidget(header)

        # Controls row
        row = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ticker (e.g. AAPL)")
        self.ticker_input.setFixedWidth(140)
        row.addWidget(self.ticker_input)

        # Mode selector
        row.addWidget(QLabel("Mode:"))
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["Intraday", "Long-Term"])
        self.mode_cb.setFixedWidth(100)
        self.mode_cb.currentTextChanged.connect(self._on_mode_change)
        row.addWidget(self.mode_cb)

        # Period
        row.addWidget(QLabel("Period:"))
        self.period_cb = QComboBox()
        self.period_cb.addItems(["5d", "3d", "2d", "1d"])
        self.period_cb.setFixedWidth(70)
        row.addWidget(self.period_cb)

        # Interval (intraday only)
        self.interval_label = QLabel("Interval:")
        row.addWidget(self.interval_label)
        self.interval_cb = QComboBox()
        self.interval_cb.addItems(["5m", "1m", "15m", "30m"])
        self.interval_cb.setFixedWidth(70)
        row.addWidget(self.interval_cb)

        self.run_btn = QPushButton("Analyze")
        self.run_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; padding: 6px 16px;")
        self.run_btn.clicked.connect(self._analyze)
        row.addWidget(self.run_btn)
        row.addStretch()
        layout.addLayout(row)

        # Progress
        self.progress = DetailedProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Chart
        cc = chart_colors()
        self.figure = plt.Figure(dpi=100, facecolor=cc["fig_bg"])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(250)
        layout.addWidget(self.canvas)

        # Signal display
        self.signal_label = QLabel("")
        self.signal_label.setFont(QFont(FONT_FAMILY, 16, QFont.Bold))
        self.signal_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.signal_label)

        # Stats
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(self.stats_label)

        layout.addStretch()
        self.setLayout(layout)

        # Wire ticker from event bus
        self.bus.ticker_selected.connect(lambda t: (self.ticker_input.setText(t), self._analyze()))

    def _on_mode_change(self, mode):
        if mode == "Intraday":
            self.period_cb.clear()
            self.period_cb.addItems(["5d", "3d", "2d", "1d"])
            self.interval_cb.setVisible(True)
            self.interval_label.setVisible(True)
        else:
            self.period_cb.clear()
            self.period_cb.addItems(["1y", "6mo", "3mo", "2y", "5y"])
            self.interval_cb.setVisible(False)
            self.interval_label.setVisible(False)

    def _analyze(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            self.bus.log_entry.emit("Enter a ticker symbol", "warn")
            return

        mode = self.mode_cb.currentText()
        period = self.period_cb.currentText()
        interval = self.interval_cb.currentText() if mode == "Intraday" else "1d"

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.reset()
        self.progress.set_progress(10, f"Fetching {ticker} ({mode})...", f"{period} @ {interval}")
        self.progress.add_log(f"Analyzing {ticker} — {mode} mode")
        self.bus.log_entry.emit(f"Analyzing {ticker} ({mode})...", "info")

        def _fetch_and_train():
            if mode == "Intraday":
                data = fetch_intraday(ticker, period, interval)
                feat_mode = "intraday"
            else:
                data = fetch_longterm(ticker, period)
                feat_mode = "longterm"

            if data.empty or len(data) < 30:
                self._analysis_done.emit(ticker, None, None, [], mode)
                return

            feats = get_all_features(feat_mode)
            model, used = train_lgbm(data, feats, ticker)
            self._analysis_done.emit(ticker, model, data, used if used else [], mode)

        threading.Thread(target=_fetch_and_train, daemon=True).start()

    def _on_done(self, ticker, model, data, features, mode):
        self.run_btn.setEnabled(True)

        if model is None or data is None:
            self.progress.set_progress(100, "Failed", "Not enough data — try a longer period")
            self.bus.log_entry.emit(f"{ticker}: not enough data", "error")
            return

        self.progress.set_progress(80, "Running predictions...", f"{len(features)} features")
        self.progress.add_log(f"Model trained — {len(features)} features")

        self.model = model
        self.features = features
        actions, confs, probs = predict_lgbm(model, data, features)
        act = LABEL_NAMES[actions[-1]]
        conf = confs[-1]
        price = data["Close"].iloc[-1]
        p = probs[-1]

        colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
        self.signal_label.setText(f"{act}  {ticker}  —  ${price:.2f}  ({conf:.0%})")
        self.signal_label.setStyleSheet(f"color: {colors.get(act, BRAND_PRIMARY)};")

        atr = float(data["ATRr_14"].iloc[-1]) if "ATRr_14" in data.columns else price * 0.01
        sl = self.rm.stop_loss(price, atr, "buy" if act == "BUY" else "sell")
        tp = self.rm.take_profit(price, atr, "buy" if act == "BUY" else "sell")
        size = self.rm.position_size(price, atr, confidence=conf)

        self.stats_label.setText(
            f"SELL {p[0]:.0%}  |  HOLD {p[1]:.0%}  |  BUY {p[2]:.0%}\n"
            f"ATR: ${atr:.2f} ({atr/price*100:.1f}%)  |  "
            f"Position: {size} shares  |  SL: ${sl:.2f}  |  TP: ${tp:.2f}\n"
            f"Mode: {mode}  |  {len(data)} bars analyzed"
        )

        # Chart
        self._plot(ticker, data, mode)

        self.progress.set_progress(100, f"Done — {act} {ticker}", f"{conf:.0%} confidence")
        self.progress.add_log(f"Signal: {act} @ ${price:.2f} ({conf:.0%})")

        log_decision(ticker, act, conf, price, size, sl, tp, atr, [float(x) for x in p],
                     reasoning=f"Trade panel {mode} analysis")

    def _plot(self, ticker, data, mode):
        self.figure.clear()
        cc = chart_colors()
        self.figure.set_facecolor(cc["fig_bg"])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(cc["ax_bg"])

        closes = data["Close"].values
        x = list(range(len(closes)))
        timestamps = data.index

        trending_up = closes[-1] >= closes[0]
        color = COLOR_BUY if trending_up else COLOR_SELL

        ax.plot(x, closes, color=color, linewidth=1.5)
        c_min = min(closes)
        ax.fill_between(x, closes, c_min, alpha=0.08, color=color)
        c_range = max(closes) - c_min if max(closes) != c_min else 1
        ax.set_ylim(c_min - c_range * 0.1, max(closes) + c_range * 0.1)

        # EMAs if available
        if "EMA_9" in data.columns:
            ax.plot(x, data["EMA_9"].values, color=BRAND_ACCENT, linewidth=1, alpha=0.6, label="EMA9")
        if "EMA_21" in data.columns:
            ax.plot(x, data["EMA_21"].values, color=BRAND_SECONDARY, linewidth=1, alpha=0.6, label="EMA21")
        if "SMA_50" in data.columns and mode == "Long-Term":
            ax.plot(x, data["SMA_50"].values, color=CHART_VWAP, linewidth=1, alpha=0.6, label="SMA50")

        ax.set_title(f"{ticker} — {mode}", color=cc["text"], fontsize=10)
        ax.tick_params(colors=cc["muted"], labelsize=7)
        ax.grid(True, alpha=0.15, color=cc["grid"])
        ax.legend(fontsize=7, facecolor=cc["ax_bg"], edgecolor=cc["grid"], labelcolor=cc["text"])

        # X-axis labels
        n = len(x)
        step = max(1, n // 5)
        ticks = list(range(0, n, step))
        if ticks[-1] != n - 1:
            ticks.append(n - 1)
        fmt = "%m/%d %H:%M" if mode == "Intraday" else "%Y/%m/%d"
        ax.set_xticks([x[i] for i in ticks])
        ax.set_xticklabels([timestamps[i].strftime(fmt) for i in ticks], fontsize=6)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.2f}"))
        self.figure.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.14)
        self.canvas.draw()

        from core.ui.chart_tooltip import ChartTooltip
        self._tooltip = ChartTooltip(self.canvas, ax, x, list(closes),
                                      x_labels=list(timestamps))
