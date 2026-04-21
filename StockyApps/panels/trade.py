"""Combined Trade panel — Intraday + Long-Term with Quick Trade sidebar."""
import sys, os, json, time, threading
import numpy as np
from datetime import datetime, timedelta
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QLinearGradient, QPen
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pytz, yfinance as yf
from core.branding import *
from core.branding import chart_colors
from core.event_bus import EventBus
from core.risk import RiskManager
from core.broker import AlpacaBroker
from core.data import fetch_intraday, fetch_longterm, get_all_features
from core.model import train_lgbm, predict_lgbm
from core.labeling import LABEL_NAMES
from core.logger import log_decision

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "settings.json")
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f: return json.load(f)
    except: return {}


class TradePanel(QWidget):
    """Combined Intraday + Long-Term analysis with Quick Trade sidebar."""

    _analysis_done = pyqtSignal(str, object, object, list, str)
    _trade_done = pyqtSignal(object, str, int)  # result, side, qty

    def __init__(self, broker, risk_manager, event_bus):
        super().__init__()
        self.broker = broker
        self.rm = risk_manager
        self.bus = event_bus
        self.model = None
        self.features = []
        self._current_ticker = ""
        self._current_price = 0
        self._analysis_done.connect(self._on_done)
        self._trade_done.connect(self._on_trade_result)
        self._build()

    def _build(self):
        from core.ui.backgrounds import GradientHeader
        from core.widgets import DetailedProgressBar
        from core.ui.icons import StockyIcons

        outer = QVBoxLayout()
        outer.setSpacing(6)
        outer.setContentsMargins(8, 4, 8, 4)

        header = GradientHeader("Trade", "Single-stock analysis + execution")
        outer.addWidget(header)

        # Controls row
        row = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ticker")
        self.ticker_input.setFixedWidth(100)
        row.addWidget(self.ticker_input)

        row.addWidget(QLabel("Mode:"))
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["Intraday", "Long-Term"])
        self.mode_cb.setFixedWidth(90)
        self.mode_cb.currentTextChanged.connect(self._on_mode_change)
        row.addWidget(self.mode_cb)

        self.period_cb = QComboBox()
        self.period_cb.addItems(["5d", "3d", "2d", "1d"])
        self.period_cb.setFixedWidth(55)
        row.addWidget(self.period_cb)

        self.interval_label = QLabel("@")
        row.addWidget(self.interval_label)
        self.interval_cb = QComboBox()
        self.interval_cb.addItems(["5m", "1m", "15m", "30m"])
        self.interval_cb.setFixedWidth(55)
        row.addWidget(self.interval_cb)

        self.run_btn = QPushButton("Analyze")
        self.run_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; padding: 5px 14px;")
        self.run_btn.clicked.connect(self._analyze)
        row.addWidget(self.run_btn)
        row.addStretch()
        outer.addLayout(row)

        self.progress = DetailedProgressBar()
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        # Splitter: chart (left) + trade sidebar (right)
        splitter = QSplitter(Qt.Horizontal)

        # Left: chart + signal
        left = QWidget()
        ll = QVBoxLayout()
        ll.setContentsMargins(0, 0, 0, 0)
        cc = chart_colors()
        self.figure = plt.Figure(dpi=100, facecolor=cc["fig_bg"])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)
        ll.addWidget(self.canvas)

        self.signal_label = QLabel("")
        self.signal_label.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        self.signal_label.setAlignment(Qt.AlignCenter)
        ll.addWidget(self.signal_label)

        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")
        ll.addWidget(self.stats_label)
        left.setLayout(ll)
        splitter.addWidget(left)

        # Right: Quick Trade sidebar
        right = QWidget()
        rl = QVBoxLayout()
        rl.setSpacing(4)
        rl.setContentsMargins(6, 4, 6, 4)

        trade_label = QLabel("Quick Trade")
        trade_label.setFont(QFont(FONT_FAMILY, 11, QFont.Bold))
        trade_label.setStyleSheet(f"color: {BRAND_PRIMARY};")
        rl.addWidget(trade_label)

        self.trade_price_label = QLabel("Price: --")
        self.trade_price_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")
        rl.addWidget(self.trade_price_label)

        # Qty + Order Type
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("Qty"))
        self.trade_qty = QSpinBox()
        self.trade_qty.setRange(1, 100000)
        self.trade_qty.setValue(1)
        self.trade_qty.setFixedWidth(70)
        q_row.addWidget(self.trade_qty)
        self.trade_order_type = QComboBox()
        self.trade_order_type.addItems(["Market", "Limit", "Stop"])
        self.trade_order_type.setFixedWidth(75)
        q_row.addWidget(self.trade_order_type)
        rl.addLayout(q_row)

        # Limit price + TIF
        p_row = QHBoxLayout()
        self.trade_limit_price = QLineEdit()
        self.trade_limit_price.setPlaceholderText("Price")
        self.trade_limit_price.setEnabled(False)
        p_row.addWidget(self.trade_limit_price, 1)
        self.trade_tif = QComboBox()
        self.trade_tif.addItems(["DAY", "GTC"])
        self.trade_tif.setFixedWidth(55)
        p_row.addWidget(self.trade_tif)
        rl.addLayout(p_row)
        self.trade_order_type.currentTextChanged.connect(
            lambda t: self.trade_limit_price.setEnabled(t != "Market"))

        # Estimates
        self.trade_est = QLabel("Cost: --")
        self.trade_est.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")
        rl.addWidget(self.trade_est)
        self.trade_bp = QLabel("BP: --")
        self.trade_bp.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")
        rl.addWidget(self.trade_bp)

        self.trade_qty.valueChanged.connect(self._update_estimate)

        # Buttons
        self.buy_btn = QPushButton("BUY")
        self.buy_btn.setStyleSheet(f"background-color: {COLOR_BUY}; font-size: 11px; padding: 6px;")
        self.buy_btn.clicked.connect(lambda: self._execute_trade("buy"))
        rl.addWidget(self.buy_btn)

        self.sell_btn = QPushButton("SELL")
        self.sell_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 11px; padding: 6px;")
        self.sell_btn.clicked.connect(lambda: self._execute_trade("sell"))
        rl.addWidget(self.sell_btn)

        self.trade_status = QLabel("")
        self.trade_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8px;")
        self.trade_status.setWordWrap(True)
        rl.addWidget(self.trade_status)

        rl.addStretch()
        right.setLayout(rl)
        right.setMaximumWidth(220)
        splitter.addWidget(right)
        splitter.setSizes([600, 200])

        outer.addWidget(splitter)
        self.setLayout(outer)

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

    def _update_estimate(self):
        qty = self.trade_qty.value()
        cost = qty * self._current_price
        self.trade_est.setText(f"Cost: ${cost:,.2f}")
        self.buy_btn.setText(f"BUY {qty} (${cost:,.0f})")
        self.sell_btn.setText(f"SELL {qty}")

    def _analyze(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            self.bus.log_entry.emit("Enter a ticker", "warn")
            return

        mode = self.mode_cb.currentText()
        period = self.period_cb.currentText()
        interval = self.interval_cb.currentText() if mode == "Intraday" else "1d"

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.reset()
        self.progress.set_progress(10, f"Analyzing {ticker} ({mode})...", f"{period} @ {interval}")
        self.bus.log_entry.emit(f"Analyzing {ticker} ({mode})...", "info")

        def _work():
            data = fetch_intraday(ticker, period, interval) if mode == "Intraday" else fetch_longterm(ticker, period)
            if data.empty or len(data) < 30:
                # Try fallback with longer period
                if mode == "Intraday":
                    for fb_period in ["5d", "10d"]:
                        data = fetch_intraday(ticker, fb_period, interval)
                        if not data.empty and len(data) >= 30:
                            break
                if data.empty or len(data) < 30:
                    self._analysis_done.emit(ticker, None, None, [], mode)
                    return
            feats = get_all_features("intraday" if mode == "Intraday" else "longterm")
            model, used = train_lgbm(data, feats, ticker)
            self._analysis_done.emit(ticker, model, data, used or [], mode)

        threading.Thread(target=_work, daemon=True).start()

    def _on_done(self, ticker, model, data, features, mode):
        self.run_btn.setEnabled(True)
        if model is None or data is None:
            self.progress.set_progress(100, "Failed", "Not enough data")
            return

        self.model, self.features = model, features
        actions, confs, probs = predict_lgbm(model, data, features)
        act = LABEL_NAMES[actions[-1]]
        conf = confs[-1]
        price = float(data["Close"].iloc[-1])
        p = probs[-1]
        self._current_ticker = ticker
        self._current_price = price

        colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
        self.signal_label.setText(f"{act}  {ticker}  —  ${price:.2f}  ({conf:.0%})")
        self.signal_label.setStyleSheet(f"color: {colors.get(act, BRAND_PRIMARY)};")

        atr = float(data["ATRr_14"].iloc[-1]) if "ATRr_14" in data.columns else price * 0.01
        sl = self.rm.stop_loss(price, atr, "buy" if act == "BUY" else "sell")
        tp = self.rm.take_profit(price, atr, "buy" if act == "BUY" else "sell")
        size = self.rm.position_size(price, atr, confidence=conf)

        self.stats_label.setText(
            f"SELL {p[0]:.0%} | HOLD {p[1]:.0%} | BUY {p[2]:.0%}\n"
            f"ATR: ${atr:.2f} | Position: {size} | SL: ${sl:.2f} | TP: ${tp:.2f}"
        )

        # Update sidebar
        self.trade_price_label.setText(f"Price: ${price:.2f}")
        self.trade_qty.setValue(size if size > 0 else 1)
        self._update_estimate()

        # Fetch buying power async
        if self.broker:
            def _bp():
                try:
                    acct = self.broker.get_account()
                    bp = float(acct.get("buying_power", 0))
                    QTimer.singleShot(0, lambda: self.trade_bp.setText(f"BP: ${bp:,.2f}"))
                except: pass
            threading.Thread(target=_bp, daemon=True).start()

        self._plot(ticker, data, mode)
        self.progress.set_progress(100, f"{act} {ticker}", f"{conf:.0%}")

        log_decision(ticker, act, conf, price, size, sl, tp, atr, [float(x) for x in p],
                     reasoning=f"Trade panel {mode}")

    def _execute_trade(self, side):
        ticker = self._current_ticker or self.ticker_input.text().strip().upper()
        if not ticker or not self.broker:
            self.bus.log_entry.emit("Enter ticker and connect broker", "warn")
            return
        qty = self.trade_qty.value()
        otype = self.trade_order_type.currentText().lower().replace(" ", "_")
        tif = self.trade_tif.currentText().lower()
        limit_price = None
        if otype != "market":
            try: limit_price = float(self.trade_limit_price.text())
            except:
                self.bus.log_entry.emit("Enter a valid price", "warn")
                return

        self.trade_status.setText(f"Submitting {side.upper()} {ticker} x{qty}...")
        self.bus.log_entry.emit(f"Submitting {side.upper()} {ticker} x{qty}...", "info")

        def _exec():
            if side == "sell" and otype == "market":
                result = self.broker.close_position(ticker, qty=qty)
            else:
                result = self.broker.place_order(ticker, qty, side, order_type=otype,
                    time_in_force=tif, limit_price=limit_price)
            self._trade_done.emit(result, side, qty)

        threading.Thread(target=_exec, daemon=True).start()

    def _on_trade_result(self, result, side, qty):
        ticker = self._current_ticker
        if "error" in result:
            self.trade_status.setText(f"Failed: {result['error'][:60]}")
            self.bus.log_entry.emit(f"{side.upper()} {ticker} failed: {result['error'][:80]}", "error")
        else:
            oid = result.get("id", "?")
            self.trade_status.setText(f"Order {oid[:12]}")
            self.bus.log_entry.emit(f"{side.upper()} {ticker} x{qty} — {oid}", "trade")
            self.bus.positions_changed.emit()

    def _plot(self, ticker, data, mode):
        self.figure.clear()
        cc = chart_colors()
        self.figure.set_facecolor(cc["fig_bg"])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(cc["ax_bg"])
        closes = data["Close"].values
        x = list(range(len(closes)))
        ts = data.index
        color = COLOR_BUY if closes[-1] >= closes[0] else COLOR_SELL
        ax.plot(x, closes, color=color, linewidth=1.5)
        c_min = min(closes)
        ax.fill_between(x, closes, c_min, alpha=0.08, color=color)
        c_range = max(closes) - c_min if max(closes) != c_min else 1
        ax.set_ylim(c_min - c_range * 0.1, max(closes) + c_range * 0.1)
        if "EMA_9" in data.columns:
            ax.plot(x, data["EMA_9"].values, color=BRAND_ACCENT, linewidth=1, alpha=0.6, label="EMA9")
        if "EMA_21" in data.columns:
            ax.plot(x, data["EMA_21"].values, color=BRAND_SECONDARY, linewidth=1, alpha=0.6, label="EMA21")
        ax.set_title(f"{ticker} — {mode}", color=cc["text"], fontsize=10)
        ax.tick_params(colors=cc["muted"], labelsize=6)
        ax.grid(True, alpha=0.15, color=cc["grid"])
        ax.legend(fontsize=7, facecolor=cc["ax_bg"], edgecolor=cc["grid"], labelcolor=cc["text"])
        n = len(x)
        step = max(1, n // 5)
        ticks = list(range(0, n, step))
        if ticks[-1] != n - 1: ticks.append(n - 1)
        fmt = "%m/%d %H:%M" if mode == "Intraday" else "%Y/%m/%d"
        ax.set_xticks([x[i] for i in ticks])
        ax.set_xticklabels([ts[i].strftime(fmt) for i in ticks], fontsize=6)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.2f}"))
        self.figure.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.14)
        self.canvas.draw()
        from core.ui.chart_tooltip import ChartTooltip
        self._tooltip = ChartTooltip(self.canvas, ax, x, list(closes), x_labels=list(ts))
