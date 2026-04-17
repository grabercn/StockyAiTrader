"""
Stocky Suite — Unified AI Trading Dashboard.

A comprehensive trading application combining:
- Dashboard: Portfolio overview, positions, P&L, equity chart
- Scanner: Multi-stock AI scanner with auto-invest
- Day Trade: Single-stock intraday analysis
- Long Trade: Long-term outlook analysis
- Logs: Decision history with reasoning
- Settings: API keys, addon management, model management

All panels share a single broker connection, risk manager, and event bus
for seamless inter-panel communication.
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar, QComboBox,
    QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QTabWidget, QSpinBox, QSplitter, QAction, QMessageBox,
    QAbstractItemView, QFormLayout, QDialog, QStatusBar, QScrollArea,
    QSplashScreen, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QLinearGradient, QPen
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pytz

# Add StockyApps to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "StockyApps"))

from core.branding import *
from core.branding import get_stylesheet, detect_system_theme
from core.event_bus import EventBus
from core.risk import RiskManager
from core.broker import AlpacaBroker
from core.scanner import scan_multiple, DEFAULT_TICKERS, TECH_TICKERS, ETF_TICKERS, ScanResult
from core.data import fetch_intraday, fetch_longterm, get_all_features
from core.model import train_lgbm, predict_lgbm
from core.labeling import LABEL_NAMES
from core.logger import (
    log_decision, log_trade_execution, log_scan_results, log_event,
    get_today_logs, get_log_files, get_log_entries,
)
from core.signals import write_signal
from core.model_manager import (
    MANAGED_MODELS, get_model_status, get_lgbm_models,
    download_model, delete_model, delete_lgbm_model, delete_all_lgbm_models,
)
from addons import get_all_addons, set_addon_enabled, discover_addons

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")
ICON_FILE = os.path.join(os.path.dirname(__file__), "icon.ico")


def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)


# ═════════════════════════════════════════════════════════════════════════════
# WORKER THREADS
# ═════════════════════════════════════════════════════════════════════════════

class ScanWorker(QThread):
    progress = pyqtSignal(int, int, str, str)
    finished = pyqtSignal(list)

    def __init__(self, tickers, period, interval, risk_manager):
        super().__init__()
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.risk_manager = risk_manager

    def run(self):
        def cb(done, total, ticker, result):
            self.progress.emit(done, total, ticker, result.action if result else "...")
        results = scan_multiple(self.tickers, self.period, self.interval,
                                self.risk_manager, max_workers=3, progress_callback=cb)
        self.finished.emit(results)


class TrainWorker(QThread):
    finished = pyqtSignal(object, list, object)

    def __init__(self, data, features, ticker, prefix="lgbm"):
        super().__init__()
        self.data = data
        self.features = features
        self.ticker = ticker
        self.prefix = prefix

    def run(self):
        model, feats = train_lgbm(self.data, self.features, self.ticker, prefix=self.prefix)
        self.finished.emit(model, feats, self.data)


class DownloadWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_info):
        super().__init__()
        self.model_info = model_info

    def run(self):
        download_model(self.model_info, progress_callback=self.progress.emit)
        self.finished.emit()


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

class DashboardPanel(QWidget):
    """Portfolio overview: account stats, positions, equity chart."""

    def __init__(self, broker, event_bus):
        super().__init__()
        self.broker = broker
        self.bus = event_bus
        self._build()
        self.bus.positions_changed.connect(self.refresh)
        self.bus.trade_executed.connect(lambda *_: QTimer.singleShot(2000, self.refresh))

    def _build(self):
        layout = QVBoxLayout()

        # Account stats row
        stats = QGroupBox("Account Overview")
        sg = QGridLayout()
        self.lbl_portfolio = QLabel("--")
        self.lbl_portfolio.setFont(QFont(FONT_FAMILY, 22, QFont.Bold))
        self.lbl_portfolio.setStyleSheet(f"color: {BRAND_PRIMARY};")
        sg.addWidget(QLabel("Portfolio Value"), 0, 0)
        sg.addWidget(self.lbl_portfolio, 1, 0)

        self.lbl_buying = QLabel("--")
        self.lbl_buying.setFont(QFont(FONT_FAMILY, 16))
        sg.addWidget(QLabel("Buying Power"), 0, 1)
        sg.addWidget(self.lbl_buying, 1, 1)

        self.lbl_cash = QLabel("--")
        self.lbl_cash.setFont(QFont(FONT_FAMILY, 16))
        sg.addWidget(QLabel("Cash"), 0, 2)
        sg.addWidget(self.lbl_cash, 1, 2)

        self.lbl_pnl = QLabel("--")
        self.lbl_pnl.setFont(QFont(FONT_FAMILY, 16, QFont.Bold))
        sg.addWidget(QLabel("Day P&L"), 0, 3)
        sg.addWidget(self.lbl_pnl, 1, 3)

        stats.setLayout(sg)
        layout.addWidget(stats)

        # Chart
        self.figure = plt.Figure(figsize=(8, 3), dpi=100, facecolor=BG_DARKEST)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Positions table
        pos_box = QGroupBox("Open Positions")
        pl = QVBoxLayout()
        self.pos_table = QTableWidget()
        self.pos_table.setColumnCount(7)
        self.pos_table.setHorizontalHeaderLabels(
            ["Symbol", "Side", "Qty", "Avg Cost", "Current", "P&L", "P&L %"])
        self.pos_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pos_table.verticalHeader().setVisible(False)
        pl.addWidget(self.pos_table)

        close_btn = QPushButton("Close All Positions")
        close_btn.setStyleSheet(f"background-color: {COLOR_SELL};")
        close_btn.clicked.connect(self._close_all)
        pl.addWidget(close_btn)
        pos_box.setLayout(pl)
        layout.addWidget(pos_box)

        self.setLayout(layout)

    def refresh(self):
        if not self.broker:
            return
        acct = self.broker.get_account()
        if "error" in acct:
            return

        pv = float(acct.get("portfolio_value", 0))
        bp = float(acct.get("buying_power", 0))
        cash = float(acct.get("cash", 0))
        eq = float(acct.get("equity", 0))
        leq = float(acct.get("last_equity", eq))
        pnl = eq - leq
        pct = (pnl / leq * 100) if leq > 0 else 0

        self.lbl_portfolio.setText(f"${pv:,.2f}")
        self.lbl_buying.setText(f"${bp:,.2f}")
        self.lbl_cash.setText(f"${cash:,.2f}")
        c = COLOR_PROFIT if pnl >= 0 else COLOR_LOSS
        self.lbl_pnl.setText(f"${pnl:+,.2f} ({pct:+.2f}%)")
        self.lbl_pnl.setStyleSheet(f"color: {c}; font-weight: bold;")

        # Positions
        positions = self.broker.get_positions()
        if isinstance(positions, list):
            self.pos_table.setRowCount(len(positions))
            for i, p in enumerate(positions):
                unrealized = float(p.get("unrealized_pl", 0))
                vals = [
                    p.get("symbol", ""), p.get("side", ""),
                    f"{float(p.get('qty', 0)):.0f}",
                    f"${float(p.get('avg_entry_price', 0)):.2f}",
                    f"${float(p.get('current_price', 0)):.2f}",
                    f"${unrealized:+,.2f}",
                    f"{float(p.get('unrealized_plpc', 0))*100:+.2f}%",
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignCenter)
                    if j >= 5:
                        item.setForeground(QColor(COLOR_PROFIT if unrealized >= 0 else COLOR_LOSS))
                    self.pos_table.setItem(i, j, item)

        # Chart
        hist = self.broker.get_portfolio_history(period="1W", timeframe="1H")
        if "error" not in hist and hist.get("equity"):
            self._plot(hist)

    def _plot(self, hist):
        self.figure.clear()
        self.figure.set_facecolor(BG_DARKEST)
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(BG_PANEL)
        eq = hist["equity"]
        ts = [datetime.fromtimestamp(t) for t in hist["timestamp"]]
        ax.plot(ts, eq, color=BRAND_PRIMARY, linewidth=1.5)
        ax.fill_between(ts, eq, alpha=0.08, color=BRAND_PRIMARY)
        ax.set_title("Portfolio Equity (1W)", color=TEXT_SECONDARY, fontsize=10)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.grid(True, alpha=0.1, color=BORDER)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d"))
        self.figure.tight_layout()
        self.canvas.draw()

    def _close_all(self):
        if not self.broker:
            return
        if QMessageBox.question(self, "Confirm", "Close ALL positions?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.broker.close_all_positions()
            self.bus.log_entry.emit("All positions closed", "trade")
            self.bus.positions_changed.emit()


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: SCANNER
# ═════════════════════════════════════════════════════════════════════════════

class ScannerPanel(QWidget):
    """Multi-stock scanner with ranked recommendations and auto-invest."""

    def __init__(self, broker, risk_manager, event_bus):
        super().__init__()
        self.broker = broker
        self.rm = risk_manager
        self.bus = event_bus
        self.results = []
        self.selected = set()
        self._build()

    def _build(self):
        layout = QVBoxLayout()

        # Controls
        ctrl = QGroupBox("Scan Universe")
        cg = QGridLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("AAPL, TSLA, NVDA... or use presets")
        cg.addWidget(self.ticker_input, 0, 0, 1, 4)

        for i, (name, ticks) in enumerate([("Top 24", DEFAULT_TICKERS), ("Tech", TECH_TICKERS), ("ETFs", ETF_TICKERS)]):
            b = QPushButton(name)
            b.setStyleSheet(f"font-size: 11px; padding: 5px; background-color: {BG_INPUT};")
            b.clicked.connect(lambda _, t=ticks: self.ticker_input.setText(", ".join(t)))
            cg.addWidget(b, 1, i)

        self.period_cb = QComboBox()
        self.period_cb.addItems(["5d", "3d", "2d", "1d"])
        self.interval_cb = QComboBox()
        self.interval_cb.addItems(["5m", "1m", "15m"])
        cg.addWidget(QLabel("Period:"), 2, 0)
        cg.addWidget(self.period_cb, 2, 1)
        cg.addWidget(QLabel("Interval:"), 2, 2)
        cg.addWidget(self.interval_cb, 2, 3)

        self.scan_btn = QPushButton("SCAN & RANK")
        self.scan_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 15px; padding: 12px;")
        self.scan_btn.clicked.connect(self._start_scan)
        cg.addWidget(self.scan_btn, 3, 0, 1, 4)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        cg.addWidget(self.progress, 4, 0, 1, 3)
        self.prog_lbl = QLabel("")
        cg.addWidget(self.prog_lbl, 4, 3)

        ctrl.setLayout(cg)
        layout.addWidget(ctrl)

        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(
            ["", "Ticker", "Signal", "Conf", "Price", "Shares", "SL/TP", "Score", "Reasoning"])
        for c in range(8):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # Actions
        arow = QHBoxLayout()
        sel_btn = QPushButton("Select Signals")
        sel_btn.clicked.connect(self._select_signals)
        arow.addWidget(sel_btn)
        desel_btn = QPushButton("Deselect All")
        desel_btn.setStyleSheet(f"background-color: {BG_INPUT};")
        desel_btn.clicked.connect(self._deselect)
        arow.addWidget(desel_btn)
        self.invest_btn = QPushButton("AUTO-INVEST SELECTED")
        self.invest_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 14px;")
        self.invest_btn.clicked.connect(self._auto_invest)
        arow.addWidget(self.invest_btn)
        layout.addLayout(arow)

        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {BRAND_PRIMARY};")
        layout.addWidget(self.summary)

        self.setLayout(layout)

    def _start_scan(self):
        text = self.ticker_input.text().strip()
        if not text:
            return
        tickers = [t.strip().upper() for t in text.replace(";", ",").split(",") if t.strip()]
        if not tickers:
            return

        self.scan_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(tickers))
        self.progress.setValue(0)
        self.bus.scan_started.emit(len(tickers))
        self.bus.log_entry.emit(f"Scanning {len(tickers)} tickers...", "info")
        self._t0 = time.time()

        self._worker = ScanWorker(tickers, self.period_cb.currentText(),
                                  self.interval_cb.currentText(), self.rm)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_progress(self, done, total, ticker, action):
        self.progress.setValue(done)
        self.prog_lbl.setText(f"{done}/{total} — {ticker}: {action}")

    def _on_done(self, results):
        self.results = results
        elapsed = time.time() - self._t0
        self.progress.setVisible(False)
        self.scan_btn.setEnabled(True)
        self.selected.clear()

        # Log each decision
        for r in results:
            if not r.error:
                log_decision(r.ticker, r.action, r.confidence, r.price,
                             r.position_size, r.stop_loss, r.take_profit, r.atr,
                             r.probs, feature_importances=r.feature_importances,
                             reasoning=r.reasoning)

        log_scan_results(len(results),
                         [{"ticker": r.ticker, "action": r.action, "score": r.score} for r in results],
                         elapsed)

        # Populate table
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            cb = QCheckBox()
            cb.toggled.connect(lambda chk, t=r.ticker: (self.selected.add(t) if chk else self.selected.discard(t)))
            w = QWidget(); l = QHBoxLayout(w); l.addWidget(cb); l.setAlignment(Qt.AlignCenter); l.setContentsMargins(0,0,0,0)
            self.table.setCellWidget(i, 0, w)

            items = [
                (r.ticker, None), (r.action, {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}.get(r.action)),
                (f"{r.confidence:.0%}", None), (f"${r.price:.2f}" if r.price else "--", None),
                (str(r.position_size) if r.position_size else "--", None),
                (f"${r.stop_loss:.0f}/${r.take_profit:.0f}" if r.stop_loss else "--", None),
                (f"{r.score:.2f}", None), (r.reasoning[:100] if r.reasoning else r.error or "", None),
            ]
            for j, (val, color) in enumerate(items):
                it = QTableWidgetItem(val)
                it.setFlags(Qt.ItemIsEnabled)
                if color:
                    it.setForeground(QColor(color))
                    it.setFont(QFont(FONT_MONO, 11, QFont.Bold))
                if j == 0:
                    it.setFont(QFont(FONT_MONO, 11, QFont.Bold))
                self.table.setItem(i, j+1, it)

        buys = sum(1 for r in results if r.action == "BUY")
        sells = sum(1 for r in results if r.action == "SELL")
        self.summary.setText(f"{buys} BUY | {sells} SELL | {len(results)-buys-sells} HOLD | {elapsed:.1f}s")
        self.bus.scan_completed.emit([{"ticker": r.ticker, "action": r.action} for r in results])
        self.bus.log_entry.emit(f"Scan done: {buys} BUY, {sells} SELL ({elapsed:.1f}s)", "trade")

    def _select_signals(self):
        self.selected.clear()
        for i, r in enumerate(self.results):
            sig = r.action in ("BUY", "SELL") and r.confidence > 0.4
            w = self.table.cellWidget(i, 0)
            if w:
                cb = w.findChild(QCheckBox)
                if cb:
                    cb.setChecked(sig)

    def _deselect(self):
        self.selected.clear()
        for i in range(self.table.rowCount()):
            w = self.table.cellWidget(i, 0)
            if w:
                cb = w.findChild(QCheckBox)
                if cb:
                    cb.setChecked(False)

    def _auto_invest(self):
        if not self.selected:
            self.bus.log_entry.emit("No tickers selected", "warn")
            return
        if not self.broker:
            self.bus.log_entry.emit("Alpaca not configured", "error")
            return

        actionable = [r for r in self.results if r.ticker in self.selected and r.action in ("BUY", "SELL") and r.position_size > 0]
        for r in actionable:
            side = "buy" if r.action == "BUY" else "sell"
            result = self.broker.place_order(r.ticker, r.position_size, side,
                                             stop_loss=r.stop_loss, take_profit=r.take_profit)
            oid = result.get("id", "failed")
            if "error" in result:
                self.bus.log_entry.emit(f"{r.action} {r.ticker} FAILED: {result['error']}", "error")
                log_trade_execution(r.ticker, side, r.position_size, "market", "failed", error=result["error"])
            else:
                self.bus.log_entry.emit(f"{r.action} {r.ticker} x{r.position_size} — order {oid}", "trade")
                self.bus.trade_executed.emit(r.ticker, side, r.position_size, oid)
                log_trade_execution(r.ticker, side, r.position_size, "market", oid)

        self.bus.positions_changed.emit()


# ═════════════════════════════════════════════════════════════════════════════
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
        layout = QVBoxLayout()

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

        # Signal display
        sig_row = QHBoxLayout()
        self.signal_lbl = QLabel("—")
        self.signal_lbl.setFont(QFont(FONT_FAMILY, 26, QFont.Bold))
        self.signal_lbl.setAlignment(Qt.AlignCenter)
        sig_row.addWidget(self.signal_lbl)
        self.stats_lbl = QLabel("")
        self.stats_lbl.setFont(QFont(FONT_MONO, 10))
        sig_row.addWidget(self.stats_lbl)
        layout.addLayout(sig_row)

        # Chart
        self.figure = plt.Figure(figsize=(8, 4), dpi=100, facecolor=BG_DARKEST)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
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
        self.bus.log_entry.emit(f"Analyzing {ticker}...", "info")

        data = fetch_intraday(ticker, self.period_cb.currentText(), self.interval_cb.currentText())
        if data.empty or len(data) < 30:
            self.bus.log_entry.emit(f"{ticker}: not enough data", "error")
            self.run_btn.setEnabled(True)
            self.progress.setVisible(False)
            return

        feats = get_all_features("intraday")
        self._worker = TrainWorker(data, feats, ticker)
        self._worker.finished.connect(lambda m, f, d: self._on_done(ticker, m, f, d))
        self._worker.start()

    def _on_done(self, ticker, model, features, data):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        if model is None:
            self.bus.log_entry.emit(f"{ticker}: training failed", "error")
            return

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

        colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
        self.signal_lbl.setText(f"{act} {ticker}")
        self.signal_lbl.setStyleSheet(f"color: {colors[act]};")
        self.stats_lbl.setText(
            f"${price:.2f} | Conf: {conf:.0%} | "
            f"SELL {p[0]:.0%} HOLD {p[1]:.0%} BUY {p[2]:.0%}\n"
            f"Size: {size} | SL ${sl:.2f} | TP ${tp:.2f} | ATR ${atr:.2f}"
        )

        write_signal(ticker, act, conf, price, size, sl, tp, atr)
        log_decision(ticker, act, conf, price, size, sl, tp, atr, list(p), reasoning=f"DayTrade analysis")
        self.bus.signal_generated.emit(ticker, act, {"conf": conf, "price": price, "size": size})
        self.bus.log_entry.emit(f"{act} {ticker} @ ${price:.2f} ({conf:.0%})", "trade")

        # Chart
        self.figure.clear()
        self.figure.set_facecolor(BG_DARKEST)
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(BG_PANEL)
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
        ax.set_title(f"{ticker} Intraday", color=TEXT_SECONDARY, fontsize=12)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.grid(True, alpha=0.1, color=BORDER)
        ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT_SECONDARY)
        self.figure.tight_layout()
        self.canvas.draw()


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: LONG TRADE
# ═════════════════════════════════════════════════════════════════════════════

class LongTradePanel(QWidget):
    """Long-term stock outlook analysis."""

    def __init__(self, event_bus):
        super().__init__()
        self.bus = event_bus
        self._build()

    def _build(self):
        layout = QVBoxLayout()
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

        self.figure = plt.Figure(figsize=(8, 4), dpi=100, facecolor=BG_DARKEST)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.setLayout(layout)

    def _analyze(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            return
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)

        data = fetch_longterm(ticker, self.period_cb.currentText())
        if data.empty or len(data) < 50:
            self.bus.log_entry.emit(f"{ticker}: not enough long-term data", "error")
            self.run_btn.setEnabled(True)
            self.progress.setVisible(False)
            return

        feats = get_all_features("longterm")
        self._worker = TrainWorker(data, feats, ticker, prefix="lgbm_long")
        self._worker.finished.connect(lambda m, f, d: self._on_done(ticker, m, f, d))
        self._worker.start()

    def _on_done(self, ticker, model, features, data):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        if model is None:
            return

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

        # Chart
        self.figure.clear()
        self.figure.set_facecolor(BG_DARKEST)
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(BG_PANEL)
        x = range(len(data))
        closes = data["Close"].values
        ax.plot(x, closes, color=CHART_PRICE, linewidth=1.5, label="Price")
        if "SMA_50" in data.columns:
            ax.plot(x, data["SMA_50"].values, color=CHART_VWAP, linewidth=1, alpha=0.6, label="SMA50")
        if "SMA_200" in data.columns:
            ax.plot(x, data["SMA_200"].values, color=CHART_EMA_FAST, linewidth=1, alpha=0.6, label="SMA200")
        ax.set_title(f"{ticker} Long-Term", color=TEXT_SECONDARY, fontsize=12)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.grid(True, alpha=0.1, color=BORDER)
        ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT_SECONDARY)
        self.figure.tight_layout()
        self.canvas.draw()


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: LOGS
# ═════════════════════════════════════════════════════════════════════════════

class LogsPanel(QWidget):
    """Decision log viewer with filtering and full reasoning."""

    def __init__(self, event_bus):
        super().__init__()
        self.bus = event_bus
        self._build()
        self.bus.log_entry.connect(self._on_live)

    def _build(self):
        layout = QVBoxLayout()

        # File selector
        row = QHBoxLayout()
        self.file_cb = QComboBox()
        self.file_cb.currentIndexChanged.connect(self._load_file)
        row.addWidget(QLabel("Log:"))
        row.addWidget(self.file_cb, 1)
        ref_btn = QPushButton("Refresh")
        ref_btn.setStyleSheet(f"font-size: 11px; padding: 5px; background-color: {BG_INPUT};")
        ref_btn.clicked.connect(self._refresh_files)
        row.addWidget(ref_btn)
        layout.addLayout(row)

        # Splitter: log viewer + live feed
        splitter = QSplitter(Qt.Vertical)

        # Historical logs
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        splitter.addWidget(self.log_view)

        # Live activity feed
        live_box = QGroupBox("Live Activity")
        ll = QVBoxLayout()
        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        ll.addWidget(self.live_log)
        live_box.setLayout(ll)
        splitter.addWidget(live_box)
        splitter.setSizes([400, 200])

        layout.addWidget(splitter)
        self.setLayout(layout)
        self._refresh_files()

    def _on_live(self, msg, level):
        self.live_log.append(log_html(msg, level))

    def _refresh_files(self):
        self.file_cb.clear()
        for f in get_log_files():
            self.file_cb.addItem(f"{f['date']} ({f['size_kb']:.0f}KB)", f["file"])

    def _load_file(self, idx):
        if idx < 0:
            return
        fn = self.file_cb.itemData(idx)
        if not fn:
            return
        entries = get_log_entries(fn, 500)
        self.log_view.clear()
        for e in entries:
            ts = e.get("timestamp", "")[:19]
            t = e.get("type", "")
            if t == "decision":
                act = e.get("action", "?")
                ticker = e.get("ticker", "?")
                conf = e.get("confidence", 0)
                reason = e.get("reasoning", "")
                ac = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}.get(act, TEXT_MUTED)
                self.log_view.append(
                    f'<span style="color:{TEXT_MUTED}">{ts}</span> '
                    f'<span style="color:{ac};font-weight:bold">{act} {ticker}</span> '
                    f'<span style="color:{TEXT_MUTED}">({conf:.0%})</span>')
                if reason:
                    self.log_view.append(f'<span style="color:{TEXT_MUTED};font-size:10px">  {reason}</span>')
                imps = e.get("feature_importances", {})
                if imps:
                    s = ", ".join([f"{k}:{v:.0f}" for k, v in list(imps.items())[:5]])
                    self.log_view.append(f'<span style="color:{BORDER};font-size:9px">  Features: {s}</span>')
            elif t == "execution":
                side = e.get("side", "?")
                ticker = e.get("ticker", "?")
                err = e.get("error")
                c = COLOR_SELL if err else BRAND_ACCENT
                self.log_view.append(
                    f'<span style="color:{TEXT_MUTED}">{ts}</span> '
                    f'<span style="color:{c}">{side.upper()} {ticker} x{e.get("qty",0)}</span>'
                    + (f' <span style="color:{COLOR_SELL}">{err}</span>' if err else ""))
            elif t == "scan":
                self.log_view.append(
                    f'<span style="color:{TEXT_MUTED}">{ts}</span> '
                    f'<span style="color:{COLOR_HOLD}">SCAN {e.get("tickers_scanned",0)} tickers ({e.get("duration_seconds",0):.1f}s)</span>')
            else:
                self.log_view.append(
                    f'<span style="color:{TEXT_MUTED}">{ts}</span> '
                    f'<span style="color:{TEXT_SECONDARY}">{e.get("message","")}</span>')


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

class SettingsPanel(QWidget):
    """API keys, hardware profiles, addon management, model management."""

    def __init__(self, event_bus):
        super().__init__()
        self.bus = event_bus
        self._dl_worker = None
        self._build()

    def _build(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout()

        # ── Hardware Profiles ──
        profile_box = QGroupBox("Hardware Profile")
        pl = QVBoxLayout()

        # Preset selector row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Active Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.currentTextChanged.connect(self._on_profile_preview)
        preset_row.addWidget(self.profile_combo, 1)

        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet(f"background-color: {BRAND_ACCENT};")
        apply_btn.clicked.connect(self._apply_profile)
        preset_row.addWidget(apply_btn)
        pl.addLayout(preset_row)

        # Profile description (must exist BEFORE _populate_profiles triggers preview)
        self.profile_desc = QLabel("")
        self.profile_desc.setWordWrap(True)
        self.profile_desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FONT_SIZE_SMALL}px; padding: 4px;")
        pl.addWidget(self.profile_desc)

        # Now populate (safe because profile_desc exists)
        self._populate_profiles()

        # Custom profile save row
        custom_row = QHBoxLayout()
        self.custom_name = QLineEdit()
        self.custom_name.setPlaceholderText("Custom profile name")
        custom_row.addWidget(self.custom_name)
        save_prof_btn = QPushButton("Save Current as Profile")
        save_prof_btn.setStyleSheet(f"background-color: {BG_INPUT}; font-size: 11px;")
        save_prof_btn.clicked.connect(self._save_custom_profile)
        custom_row.addWidget(save_prof_btn)
        del_prof_btn = QPushButton("Delete")
        del_prof_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 11px;")
        del_prof_btn.clicked.connect(self._delete_custom_profile)
        custom_row.addWidget(del_prof_btn)
        pl.addLayout(custom_row)

        profile_box.setLayout(pl)
        inner_layout.addWidget(profile_box)

        # Appearance
        appear_box = QGroupBox("Appearance")
        al2 = QHBoxLayout()
        al2.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Auto (System)", "Dark", "Light"])
        # Set current
        settings = load_settings()
        theme = settings.get("theme", "auto")
        idx_map = {"auto": 0, "dark": 1, "light": 2}
        self.theme_combo.setCurrentIndex(idx_map.get(theme, 0))
        self.theme_combo.currentIndexChanged.connect(self._change_theme)
        al2.addWidget(self.theme_combo, 1)
        appear_box.setLayout(al2)
        inner_layout.addWidget(appear_box)

        # API Keys
        keys_box = QGroupBox("API Keys")
        kl = QFormLayout()
        self.inputs = {}
        for key, label in [("alpaca_api_key", "Alpaca API Key"), ("alpaca_secret_key", "Alpaca Secret Key")]:
            inp = QLineEdit(settings.get(key, ""))
            if "secret" in key.lower():
                inp.setEchoMode(QLineEdit.Password)
            kl.addRow(label + ":", inp)
            self.inputs[key] = inp

        # Addon API keys
        for addon in get_all_addons():
            if addon.requires_api_key and addon.api_key_name:
                inp = QLineEdit(settings.get(addon.api_key_name, ""))
                inp.setPlaceholderText(f"For {addon.name}")
                kl.addRow(f"{addon.name}:", inp)
                self.inputs[addon.api_key_name] = inp

        save_btn = QPushButton("Save All Keys")
        save_btn.clicked.connect(self._save_keys)
        kl.addRow(save_btn)
        keys_box.setLayout(kl)
        inner_layout.addWidget(keys_box)

        # Model Manager
        model_box = QGroupBox("AI Models")
        ml = QVBoxLayout()
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels(["Model", "Status", "Size", "Action"])
        self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 4):
            self.model_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setFixedHeight(120)
        ml.addWidget(self.model_table)
        self.dl_status = QLabel("")
        self.dl_status.setStyleSheet(f"color: {BRAND_ACCENT};")
        ml.addWidget(self.dl_status)
        model_box.setLayout(ml)
        inner_layout.addWidget(model_box)

        # Addon Manager
        addon_box = QGroupBox("Addons")
        al = QVBoxLayout()
        self.addon_table = QTableWidget()
        self.addon_table.setColumnCount(5)
        self.addon_table.setHorizontalHeaderLabels(["On", "Addon", "Status", "Features", "Config"])
        self.addon_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.addon_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for c in range(2, 5):
            self.addon_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.addon_table.verticalHeader().setVisible(False)
        al.addWidget(self.addon_table)

        install_btn = QPushButton("Install Missing Dependencies")
        install_btn.setStyleSheet(f"background-color: {BG_INPUT};")
        install_btn.clicked.connect(self._install_deps)
        al.addWidget(install_btn)
        addon_box.setLayout(al)
        inner_layout.addWidget(addon_box)

        inner.setLayout(inner_layout)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self._refresh()

    def _change_theme(self, index):
        theme_map = {0: "auto", 1: "dark", 2: "light"}
        theme = theme_map.get(index, "auto")
        settings = load_settings()
        settings["theme"] = theme
        save_settings(settings)
        # Apply immediately to the main window
        main_window = self.window()
        if main_window:
            main_window.setStyleSheet(get_stylesheet(theme))
            if hasattr(main_window, '_theme'):
                main_window._theme = theme
            if hasattr(main_window, '_apply_scale'):
                main_window._apply_scale()
        self.bus.log_entry.emit(f"Theme changed to: {theme}", "system")

    def _save_keys(self):
        settings = load_settings()
        for key, inp in self.inputs.items():
            settings[key] = inp.text()
        save_settings(settings)
        self.bus.settings_changed.emit(settings)
        self.bus.log_entry.emit("Settings saved", "system")

    def _refresh(self):
        # Models
        self.model_table.setRowCount(len(MANAGED_MODELS))
        for i, m in enumerate(MANAGED_MODELS):
            dl, sz = get_model_status(m)
            self.model_table.setItem(i, 0, QTableWidgetItem(m.name))
            s = QTableWidgetItem("Ready" if dl else "Not downloaded")
            s.setForeground(QColor(STATUS_ACTIVE if dl else STATUS_ERROR))
            self.model_table.setItem(i, 1, s)
            self.model_table.setItem(i, 2, QTableWidgetItem(sz if dl else m.size_estimate))
            if not dl:
                btn = QPushButton("Download")
                btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 10px; padding: 3px;")
                btn.clicked.connect(lambda _, mi=m: self._download(mi))
                self.model_table.setCellWidget(i, 3, btn)
            else:
                btn = QPushButton("Delete")
                btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 10px; padding: 3px;")
                btn.clicked.connect(lambda _, mi=m: self._delete_model(mi))
                self.model_table.setCellWidget(i, 3, btn)

        # Addons
        settings = load_settings()
        states = settings.get("addon_states", {})
        addons = get_all_addons()
        self.addon_table.setRowCount(len(addons))
        for i, a in enumerate(addons):
            if a.module_name in states:
                a.enabled = states[a.module_name]
                set_addon_enabled(a.module_name, a.enabled)

            cb = QCheckBox()
            cb.setChecked(a.enabled and a.available)
            cb.setEnabled(a.available)
            cb.toggled.connect(lambda chk, n=a.module_name: self._toggle_addon(n, chk))
            w = QWidget(); l = QHBoxLayout(w); l.addWidget(cb); l.setAlignment(Qt.AlignCenter); l.setContentsMargins(0,0,0,0)
            self.addon_table.setCellWidget(i, 0, w)

            self.addon_table.setItem(i, 1, QTableWidgetItem(a.name))
            st = QTableWidgetItem("Active" if a.available and a.enabled else a.status)
            st.setForeground(QColor(STATUS_ACTIVE if a.available and a.enabled else STATUS_ERROR if not a.available else STATUS_INACTIVE))
            self.addon_table.setItem(i, 2, st)
            self.addon_table.setItem(i, 3, QTableWidgetItem(f"{len(a.features)}"))
            cfg = "Key set" if a.requires_api_key and settings.get(a.api_key_name) else ("Needs key" if a.requires_api_key else "—")
            c = QTableWidgetItem(cfg)
            c.setForeground(QColor(STATUS_ACTIVE if cfg == "Key set" else (STATUS_ERROR if cfg == "Needs key" else TEXT_MUTED)))
            self.addon_table.setItem(i, 4, c)

    def _toggle_addon(self, name, enabled):
        set_addon_enabled(name, enabled)
        settings = load_settings()
        states = settings.get("addon_states", {})
        states[name] = enabled
        settings["addon_states"] = states
        save_settings(settings)
        self._refresh()

    def _download(self, model_info):
        if self._dl_worker and self._dl_worker.isRunning():
            return
        self.dl_status.setText(f"Downloading {model_info.name}...")
        self._dl_worker = DownloadWorker(model_info)
        self._dl_worker.progress.connect(lambda s: self.dl_status.setText(s))
        self._dl_worker.finished.connect(lambda: (self.dl_status.setText("Done!"), self._refresh()))
        self._dl_worker.start()

    def _delete_model(self, model_info):
        delete_model(model_info)
        self._refresh()

    def _install_deps(self):
        missing = set()
        for a in get_all_addons():
            if not a.available:
                missing.update(a.dependencies)
        if not missing:
            self.dl_status.setText("All deps installed!")
            return
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install"] + list(missing),
                      capture_output=True, timeout=120)
        discover_addons()
        self._refresh()

    # ── Profile methods ───────────────────────────────────────────────────

    def _populate_profiles(self):
        from core.profiles import get_all_profiles, get_active_profile_name
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        profiles = get_all_profiles()
        active = get_active_profile_name()
        for i, name in enumerate(profiles):
            is_custom = profiles[name].get("custom", False)
            label = f"{name} (custom)" if is_custom else name
            self.profile_combo.addItem(label, name)
            if name == active:
                self.profile_combo.setCurrentIndex(i)
        self.profile_combo.blockSignals(False)
        self._on_profile_preview(self.profile_combo.currentText())

    def _on_profile_preview(self, display_text):
        if not hasattr(self, 'profile_desc'):
            return
        try:
            from core.profiles import get_all_profiles
            idx = self.profile_combo.currentIndex()
            if idx < 0:
                return
            name = self.profile_combo.itemData(idx)
            profiles = get_all_profiles()
            profile = profiles.get(name, {})
            desc = profile.get("description", "")
            addons_on = sum(1 for v in profile.get("addons", {}).values() if v)
            addons_total = len(profile.get("addons", {}))
            workers = profile.get("scanner_workers", 3)
            self.profile_desc.setText(
                f"{desc}\n"
                f"Addons: {addons_on}/{addons_total} enabled | Scanner threads: {workers}"
            )
        except Exception:
            pass

    def _apply_profile(self):
        from core.profiles import apply_profile
        idx = self.profile_combo.currentIndex()
        if idx < 0:
            return
        name = self.profile_combo.itemData(idx)
        ok, msg = apply_profile(name)
        self.dl_status.setText(msg)
        if ok:
            self.bus.log_entry.emit(f"Profile applied: {name}", "system")
            self.bus.settings_changed.emit(load_settings())
            discover_addons()
            self._refresh()

    def _save_custom_profile(self):
        from core.profiles import save_custom_profile, get_current_addon_states
        name = self.custom_name.text().strip()
        if not name:
            self.dl_status.setText("Enter a profile name first.")
            return
        states = get_current_addon_states()
        ok, msg = save_custom_profile(name, f"Custom profile: {name}", states)
        self.dl_status.setText(msg)
        if ok:
            self._populate_profiles()
            self.bus.log_entry.emit(f"Custom profile saved: {name}", "system")

    def _delete_custom_profile(self):
        from core.profiles import delete_custom_profile
        idx = self.profile_combo.currentIndex()
        if idx < 0:
            return
        name = self.profile_combo.itemData(idx)
        ok, msg = delete_custom_profile(name)
        self.dl_status.setText(msg)
        if ok:
            self._populate_profiles()


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: TAX REPORTS
# ═════════════════════════════════════════════════════════════════════════════

class TaxPanel(QWidget):
    """Generate IRS Form 8949 / Schedule D tax reports from trade history."""

    def __init__(self, broker, event_bus):
        super().__init__()
        self.broker = broker
        self.bus = event_bus
        self._build()

    def _build(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(
            "Generate IRS Form 8949 data (Sales and Dispositions of Capital Assets).\n"
            "Exports CSV for your accountant or TurboTax import."
        ))

        row = QHBoxLayout()
        row.addWidget(QLabel("Tax Year:"))
        self.year_spin = QSpinBox()
        self.year_spin.setRange(2020, 2030)
        self.year_spin.setValue(datetime.now().year)
        row.addWidget(self.year_spin)

        gen_btn = QPushButton("Generate Report")
        gen_btn.setStyleSheet(f"background-color: {BRAND_ACCENT};")
        gen_btn.clicked.connect(self._generate)
        row.addWidget(gen_btn)
        row.addStretch()
        layout.addLayout(row)

        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setFont(QFont(FONT_MONO, 10))
        layout.addWidget(self.report_view)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {BRAND_ACCENT};")
        layout.addWidget(self.status)
        self.setLayout(layout)

    def _generate(self):
        if not self.broker:
            self.status.setText("Alpaca API not configured.")
            return

        from core.tax_report import generate_form_8949
        self.status.setText("Generating...")
        year = self.year_spin.value()

        result = generate_form_8949(self.broker, year)
        self.report_view.setPlainText(result["text"])

        if result["csv_path"]:
            self.status.setText(f"Saved: {result['csv_path']}")
            self.bus.log_entry.emit(f"Tax report generated: {result['csv_path']}", "system")
        else:
            self.status.setText("Report generated (no trades found).")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: SYSTEM TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestingPanel(QWidget):
    """In-app system diagnostics and test runner."""

    def __init__(self, broker, event_bus):
        super().__init__()
        self.broker = broker
        self.bus = event_bus
        self._build()

    def _build(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Run system diagnostics and unit tests to verify everything works."))

        # Quick diagnostics
        diag_box = QGroupBox("System Diagnostics")
        dg = QVBoxLayout()

        self.diag_output = QTextEdit()
        self.diag_output.setReadOnly(True)
        self.diag_output.setFont(QFont(FONT_MONO, 10))
        self.diag_output.setFixedHeight(250)
        dg.addWidget(self.diag_output)

        dbtn_row = QHBoxLayout()
        run_diag_btn = QPushButton("Run All Diagnostics")
        run_diag_btn.setStyleSheet(f"background-color: {BRAND_ACCENT};")
        run_diag_btn.clicked.connect(self._run_diagnostics)
        dbtn_row.addWidget(run_diag_btn)

        api_btn = QPushButton("Test Alpaca Connection")
        api_btn.clicked.connect(self._test_alpaca)
        dbtn_row.addWidget(api_btn)

        addon_btn = QPushButton("Test Addons")
        addon_btn.clicked.connect(self._test_addons)
        dbtn_row.addWidget(addon_btn)

        model_btn = QPushButton("Test Models")
        model_btn.clicked.connect(self._test_models)
        dbtn_row.addWidget(model_btn)
        dg.addLayout(dbtn_row)
        diag_box.setLayout(dg)
        layout.addWidget(diag_box)

        # Unit test runner
        test_box = QGroupBox("Unit Tests (pytest)")
        tl = QVBoxLayout()
        self.test_output = QTextEdit()
        self.test_output.setReadOnly(True)
        self.test_output.setFont(QFont(FONT_MONO, 10))
        tl.addWidget(self.test_output)

        run_test_btn = QPushButton("Run Full Test Suite")
        run_test_btn.setStyleSheet(f"background-color: {BRAND_PRIMARY};")
        run_test_btn.clicked.connect(self._run_pytest)
        tl.addWidget(run_test_btn)
        test_box.setLayout(tl)
        layout.addWidget(test_box)

        self.setLayout(layout)

    def _d(self, msg, ok=True):
        icon = "PASS" if ok else "FAIL"
        color = BRAND_ACCENT if ok else COLOR_SELL
        self.diag_output.append(f'<span style="color:{color}">[{icon}]</span> {msg}')

    def _run_diagnostics(self):
        self.diag_output.clear()
        self.diag_output.append(f'<b>System Diagnostics — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</b>\n')

        # Python version
        import platform
        self._d(f"Python {platform.python_version()}")

        # Core imports
        try:
            from core.model import train_lgbm
            from core.features import engineer_features
            from core.risk import RiskManager
            from core.scanner import scan_multiple
            from core.logger import log_event
            self._d("Core modules import OK")
        except Exception as e:
            self._d(f"Core import FAILED: {e}", False)

        # Dependencies
        for pkg in ["lightgbm", "ta", "yfinance", "transformers", "torch", "numpy", "pandas"]:
            try:
                __import__(pkg)
                self._d(f"Package: {pkg}")
            except ImportError:
                self._d(f"Package: {pkg} NOT INSTALLED", False)

        # Addons
        try:
            addons = get_all_addons()
            active = [a for a in addons if a.available and a.enabled]
            self._d(f"Addons: {len(active)}/{len(addons)} active")
        except Exception as e:
            self._d(f"Addon system: {e}", False)

        # Models
        for m in MANAGED_MODELS:
            dl, sz = get_model_status(m)
            self._d(f"Model {m.name}: {'downloaded' if dl else 'missing'} ({sz})", dl)

        # Features
        try:
            from core.data import get_all_features
            f = get_all_features("intraday")
            self._d(f"Feature pipeline: {len(f)} features")
        except Exception as e:
            self._d(f"Feature pipeline: {e}", False)

        # Alpaca
        self._test_alpaca()

        self.diag_output.append(f'\n<b>Diagnostics complete.</b>')

    def _test_alpaca(self):
        if not self.broker:
            self._d("Alpaca API: not configured", False)
            return
        acct = self.broker.get_account()
        if "error" in acct:
            self._d(f"Alpaca API: {acct['error']}", False)
        else:
            pv = float(acct.get("portfolio_value", 0))
            self._d(f"Alpaca API: connected (portfolio ${pv:,.2f})")

    def _test_addons(self):
        self.diag_output.clear()
        self.diag_output.append("<b>Addon Tests</b>\n")
        import pandas as pd
        dummy = pd.DataFrame({"Close": [100.0]*10, "Volume": [10000.0]*10},
                             index=pd.date_range("2024-01-01", periods=10, freq="5min"))
        for addon in get_all_addons():
            if addon.available and addon.enabled:
                try:
                    result = addon._module.get_features("AAPL", dummy)
                    self._d(f"{addon.name}: {len(result)} features returned")
                except Exception as e:
                    self._d(f"{addon.name}: {e}", False)
            else:
                self._d(f"{addon.name}: skipped (inactive)", False)

    def _test_models(self):
        self.diag_output.clear()
        self.diag_output.append("<b>Model Tests</b>\n")
        for m in MANAGED_MODELS:
            dl, sz = get_model_status(m)
            self._d(f"{m.name}: {'OK' if dl else 'not downloaded'} ({sz})", dl)

        from core.model_manager import get_lgbm_models
        lgbm = get_lgbm_models()
        self._d(f"Trained LightGBM models: {len(lgbm)}")
        for name, size in lgbm:
            self._d(f"  {name} ({size})")

    def _run_pytest(self):
        self.test_output.clear()
        self.test_output.append("<b>Running pytest...</b>\n")
        try:
            import subprocess
            project_root = os.path.join(os.path.dirname(__file__))
            result = subprocess.run(
                [sys.executable, "run_tests.py", "-v", "--tb=short"],
                capture_output=True, text=True, timeout=120,
                cwd=project_root,
            )
            output = result.stdout + result.stderr
            # Colorize pass/fail
            for line in output.split("\n"):
                if "PASSED" in line:
                    self.test_output.append(f'<span style="color:{BRAND_ACCENT}">{line}</span>')
                elif "FAILED" in line:
                    self.test_output.append(f'<span style="color:{COLOR_SELL}">{line}</span>')
                elif "ERROR" in line:
                    self.test_output.append(f'<span style="color:{COLOR_SELL}">{line}</span>')
                elif "passed" in line or "warning" in line:
                    self.test_output.append(f'<span style="color:{BRAND_PRIMARY}">{line}</span>')
                else:
                    self.test_output.append(f'<span style="color:{TEXT_SECONDARY}">{line}</span>')

        except subprocess.TimeoutExpired:
            self.test_output.append(f'<span style="color:{COLOR_SELL}">Tests timed out after 120s</span>')
        except Exception as e:
            self.test_output.append(f'<span style="color:{COLOR_SELL}">Error running tests: {e}</span>')


# ═════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═════════════════════════════════════════════════════════════════════════════

class StockySuite(QMainWindow):
    """Main application window — unified trading dashboard."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(50, 50, 1500, 950)

        # Load theme from settings or auto-detect from Windows
        settings = load_settings()
        self._theme = settings.get("theme", "auto")
        self.setStyleSheet(get_stylesheet(self._theme))

        if os.path.exists(ICON_FILE):
            self.setWindowIcon(QIcon(ICON_FILE))

        # Shared state
        self.event_bus = EventBus()
        self.risk_manager = RiskManager()
        self.broker = self._init_broker()

        # Build UI
        self._build()

        # Status bar
        self.statusBar().showMessage(f"{APP_NAME} v{APP_VERSION} — {APP_TAGLINE}")
        self.statusBar().setStyleSheet(
            f"background-color: {BG_DARK}; color: {TEXT_MUTED}; "
            f"border-top: 1px solid {BORDER}; padding: 4px;"
        )

        # Refresh dashboard on startup
        if hasattr(self, 'dashboard') and hasattr(self.dashboard, 'refresh'):
            QTimer.singleShot(500, self.dashboard.refresh)

        # Activity feed forwarding
        self.event_bus.log_entry.connect(self._on_log)

        # UI scaling — start larger, allow Ctrl+/- zoom
        self._scale = 1.15  # Start 15% bigger than default
        self._apply_scale()

        # Keyboard shortcuts for zoom
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        QShortcut(QKeySequence("Ctrl+="), self, lambda: self._zoom(0.1))
        QShortcut(QKeySequence("Ctrl++"), self, lambda: self._zoom(0.1))
        QShortcut(QKeySequence("Ctrl+-"), self, lambda: self._zoom(-0.1))
        QShortcut(QKeySequence("Ctrl+0"), self, lambda: self._reset_zoom())

        log_event("startup", f"{APP_NAME} v{APP_VERSION} launched")

    def _init_broker(self):
        settings = load_settings()
        key = settings.get("alpaca_api_key", "")
        secret = settings.get("alpaca_secret_key", "")
        if key and secret:
            return AlpacaBroker(key, secret)
        return None

    def _build(self):
        # Central tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        # Create panels — each wrapped in try/except so one bad panel
        # doesn't kill the entire app
        panels = [
            ("Dashboard",   lambda: DashboardPanel(self.broker, self.event_bus)),
            ("Scanner",     lambda: ScannerPanel(self.broker, self.risk_manager, self.event_bus)),
            ("Day Trade",   lambda: DayTradePanel(self.broker, self.risk_manager, self.event_bus)),
            ("Long Trade",  lambda: LongTradePanel(self.event_bus)),
            ("Logs",        lambda: LogsPanel(self.event_bus)),
            ("Tax Reports", lambda: TaxPanel(self.broker, self.event_bus)),
            ("Testing",     lambda: TestingPanel(self.broker, self.event_bus)),
            ("Settings",    lambda: SettingsPanel(self.event_bus)),
        ]

        for tab_name, factory in panels:
            try:
                panel = factory()
                self.tabs.addTab(panel, tab_name)
                # Store reference for cross-panel access
                attr = tab_name.lower().replace(" ", "_")
                setattr(self, attr, panel)
            except Exception as e:
                # Create a fallback error panel instead of crashing
                error_panel = QWidget()
                error_layout = QVBoxLayout()
                error_lbl = QLabel(f"Failed to load {tab_name}:\n{e}")
                error_lbl.setStyleSheet(f"color: {COLOR_SELL}; padding: 20px;")
                error_lbl.setWordWrap(True)
                error_layout.addWidget(error_lbl)
                error_panel.setLayout(error_layout)
                self.tabs.addTab(error_panel, f"{tab_name} (!)")
                log_event("panel_error", f"{tab_name} failed to load: {e}")
                print(f"[ERROR] Panel '{tab_name}' failed: {e}", flush=True)

        self.setCentralWidget(self.tabs)

        # Reconnect broker on settings change
        self.event_bus.settings_changed.connect(self._on_settings_changed)

    def _on_settings_changed(self, settings):
        key = settings.get("alpaca_api_key", "")
        secret = settings.get("alpaca_secret_key", "")
        if key and secret:
            self.broker = AlpacaBroker(key, secret)
            self.dashboard.broker = self.broker
            self.scanner.broker = self.broker
            self.day_trade.broker = self.broker
            self.tax_panel.broker = self.broker
            self.testing_panel.broker = self.broker
            self.dashboard.refresh()

    def _on_log(self, msg, level):
        # Forward to status bar for quick glance
        self.statusBar().showMessage(f"{datetime.now().strftime('%H:%M:%S')} — {msg}", 10000)

    # ── UI Scaling (Ctrl+/-, Ctrl+0 to reset) ─────────────────────────────

    def _apply_scale(self):
        """Apply the current scale factor to the entire UI via font size."""
        base_size = int(FONT_SIZE_BODY * self._scale)
        theme = getattr(self, '_theme', 'auto')
        base_sheet = get_stylesheet(theme)
        self.setStyleSheet(base_sheet + f"""
            * {{ font-size: {base_size}px; }}
            QTabBar::tab {{
                font-size: {base_size}px;
                padding: {int(8*self._scale)}px {int(18*self._scale)}px;
                min-width: 0px;
            }}
            QTabBar {{ qproperty-expanding: 0; }}
        """)
        self.statusBar().showMessage(
            f"Zoom: {self._scale:.0%}  (Ctrl+/- to adjust, Ctrl+0 to reset)", 3000
        )

    def _zoom(self, delta):
        self._scale = max(0.7, min(2.0, self._scale + delta))
        self._apply_scale()

    def _reset_zoom(self):
        self._scale = 1.0
        self._apply_scale()


# ═════════════════════════════════════════════════════════════════════════════
# SPLASH SCREEN
# ═════════════════════════════════════════════════════════════════════════════

def create_splash_pixmap():
    """Generate a premium splash screen image programmatically."""
    w, h = 580, 360
    pixmap = QPixmap(w, h)
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Background gradient (dark blue -> near black)
    grad = QLinearGradient(0, 0, w, h)
    grad.setColorAt(0, QColor(15, 17, 23))
    grad.setColorAt(0.5, QColor(20, 25, 40))
    grad.setColorAt(1, QColor(10, 12, 18))
    painter.setBrush(grad)
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(0, 0, w, h, 16, 16)

    # Subtle border
    painter.setPen(QPen(QColor(42, 45, 58), 1))
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(1, 1, w-2, h-2, 16, 16)

    # Accent line at top
    accent_grad = QLinearGradient(0, 0, w, 0)
    accent_grad.setColorAt(0, QColor(14, 165, 233, 0))
    accent_grad.setColorAt(0.3, QColor(14, 165, 233, 255))
    accent_grad.setColorAt(0.7, QColor(16, 185, 129, 255))
    accent_grad.setColorAt(1, QColor(16, 185, 129, 0))
    painter.setPen(QPen(accent_grad, 3))
    painter.drawLine(40, 4, w-40, 4)

    # App icon (if exists)
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    if os.path.exists(icon_path):
        icon = QPixmap(icon_path).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap((w - 64) // 2, 40, icon)

    # App name
    painter.setPen(QColor(14, 165, 233))
    painter.setFont(QFont("Segoe UI", 28, QFont.Bold))
    painter.drawText(0, 115, w, 45, Qt.AlignCenter, APP_NAME)

    # Tagline
    painter.setPen(QColor(148, 163, 184))
    painter.setFont(QFont("Segoe UI", 12))
    painter.drawText(0, 160, w, 25, Qt.AlignCenter, APP_TAGLINE)

    # Version
    painter.setPen(QColor(100, 116, 139))
    painter.setFont(QFont("Segoe UI", 10))
    painter.drawText(0, 188, w, 20, Qt.AlignCenter, f"v{APP_VERSION}")

    # Feature list
    features = [
        "LightGBM + FinBERT AI Engine",
        "10 Signal Addons  |  38 Features",
        "Multi-Stock Scanner  |  Auto-Invest",
        "Risk Management  |  Tax Reports",
    ]
    painter.setFont(QFont("Segoe UI", 9))
    for i, feat in enumerate(features):
        painter.setPen(QColor(100, 116, 139))
        painter.drawText(0, 225 + i * 20, w, 18, Qt.AlignCenter, feat)

    # Loading text area (will be updated)
    painter.setPen(QColor(14, 165, 233, 150))
    painter.setFont(QFont("Segoe UI", 9))
    painter.drawText(0, h - 30, w, 20, Qt.AlignCenter, "Loading...")

    # Copyright
    painter.setPen(QColor(64, 74, 91))
    painter.setFont(QFont("Segoe UI", 8))
    painter.drawText(0, h - 16, w, 14, Qt.AlignCenter, f"2024-2026 {APP_AUTHOR}  |  {APP_URL}")

    painter.end()
    return pixmap


# ═════════════════════════════════════════════════════════════════════════════
# ABOUT DIALOG
# ═════════════════════════════════════════════════════════════════════════════

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setFixedSize(420, 350)
        self.setStyleSheet(SUITE_STYLESHEET)

        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Icon
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            icon_lbl = QLabel()
            icon_lbl.setPixmap(QPixmap(icon_path).scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            icon_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(icon_lbl)

        name_lbl = QLabel(APP_NAME)
        name_lbl.setFont(QFont(FONT_FAMILY, 20, QFont.Bold))
        name_lbl.setStyleSheet(f"color: {BRAND_PRIMARY};")
        name_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_lbl)

        ver_lbl = QLabel(f"Version {APP_VERSION}")
        ver_lbl.setAlignment(Qt.AlignCenter)
        ver_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(ver_lbl)

        tag_lbl = QLabel(APP_TAGLINE)
        tag_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(tag_lbl)

        desc = QLabel(
            "A comprehensive AI-powered trading suite featuring\n"
            "LightGBM machine learning, FinBERT sentiment analysis,\n"
            "10 pluggable signal addons, multi-stock scanning,\n"
            "risk management, and automated portfolio investing.\n\n"
            "68 unit tests  |  38 ML features  |  4 hardware profiles"
        )
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(desc)

        link = QLabel(f'<a href="{APP_URL}" style="color: {BRAND_PRIMARY};">{APP_URL}</a>')
        link.setOpenExternalLinks(True)
        link.setAlignment(Qt.AlignCenter)
        layout.addWidget(link)

        copy_lbl = QLabel(f"2024-2026 {APP_AUTHOR}")
        copy_lbl.setAlignment(Qt.AlignCenter)
        copy_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(copy_lbl)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)


# ═════════════════════════════════════════════════════════════════════════════
# LOADING WINDOW (real progress bar, not just splash image)
# ═════════════════════════════════════════════════════════════════════════════

class LoadingWindow(QWidget):
    """Premium boot screen with determinate progress bar showing each module loading."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(540, 380)

        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - 540) // 2, (screen.height() - 380) // 2)

        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)

        # Container with styled background
        container = QWidget()
        container.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_DARKEST};
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
        """)
        inner = QVBoxLayout()
        inner.setContentsMargins(30, 25, 30, 20)
        inner.setSpacing(6)

        # Icon
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            icon_lbl = QLabel()
            icon_lbl.setPixmap(QPixmap(icon_path).scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            icon_lbl.setAlignment(Qt.AlignCenter)
            inner.addWidget(icon_lbl)

        # App name
        name = QLabel(APP_NAME)
        name.setFont(QFont(FONT_FAMILY, 24, QFont.Bold))
        name.setStyleSheet(f"color: {BRAND_PRIMARY}; background: transparent; border: none;")
        name.setAlignment(Qt.AlignCenter)
        inner.addWidget(name)

        # Tagline
        tag = QLabel(APP_TAGLINE)
        tag.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent; border: none; font-size: 11px;")
        tag.setAlignment(Qt.AlignCenter)
        inner.addWidget(tag)

        # Version
        ver = QLabel(f"v{APP_VERSION}")
        ver.setStyleSheet(f"color: {BORDER}; background: transparent; border: none; font-size: 10px;")
        ver.setAlignment(Qt.AlignCenter)
        inner.addWidget(ver)

        inner.addSpacing(15)

        # Status message
        self.status_lbl = QLabel("Initializing...")
        self.status_lbl.setFont(QFont(FONT_MONO, 10))
        self.status_lbl.setStyleSheet(f"color: {BRAND_PRIMARY}; background: transparent; border: none;")
        self.status_lbl.setAlignment(Qt.AlignLeft)
        inner.addWidget(self.status_lbl)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {BG_INPUT};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {BRAND_PRIMARY}, stop:1 {BRAND_ACCENT});
                border-radius: 3px;
            }}
        """)
        inner.addWidget(self.progress)

        # Detail log
        self.detail_lbl = QLabel("")
        self.detail_lbl.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent; border: none; font-size: 9px;")
        self.detail_lbl.setAlignment(Qt.AlignLeft)
        inner.addWidget(self.detail_lbl)

        inner.addStretch()

        # Copyright
        copy_lbl = QLabel(f"2024-2026 {APP_AUTHOR}")
        copy_lbl.setStyleSheet(f"color: {BORDER}; background: transparent; border: none; font-size: 8px;")
        copy_lbl.setAlignment(Qt.AlignCenter)
        inner.addWidget(copy_lbl)

        container.setLayout(inner)
        layout.addWidget(container)
        self.setLayout(layout)

    def set_progress(self, pct, status, detail=""):
        self.progress.setValue(int(pct))
        self.status_lbl.setText(status)
        self.detail_lbl.setText(detail)
        QApplication.processEvents()


def boot_app():
    """Boot sequence with real loading progress."""
    app = QApplication(sys.argv)
    if os.path.exists(ICON_FILE):
        app.setWindowIcon(QIcon(ICON_FILE))

    # Show loading window
    loader = LoadingWindow()
    loader.show()
    app.processEvents()

    def step(pct, msg, detail=""):
        loader.set_progress(pct, msg, detail)
        # Small delay so user can actually read each step
        time.sleep(0.35)

    # Boot steps
    step(5,  "Loading core modules...",       "features, model, risk, broker, scanner")
    step(15, "Checking dependencies...",      "lightgbm, ta, transformers, torch")

    step(25, "Discovering addons...",         "Scanning StockyApps/addons/")
    from addons import discover_addons, get_all_addons
    discover_addons()
    addons = get_all_addons()
    active = [a for a in addons if a.available and a.enabled]
    step(40, f"Addons: {len(active)}/{len(addons)} active", ", ".join(a.name for a in active[:4]) + "...")

    step(50, "Initializing risk manager...",  "ATR sizing | 2% risk | 5% drawdown limit")

    step(60, "Connecting to broker...",       "Alpaca paper trading API")

    from core.profiles import get_active_profile_name
    profile = get_active_profile_name()
    step(70, f"Hardware profile: {profile}",  "Addon configuration loaded")

    step(80, "Building interface...",         "8 panels | event bus | signal routing")
    suite = StockySuite()

    # Add Help > About menu
    help_menu = suite.menuBar().addMenu("Help")
    about_action = QAction(f"About {APP_NAME}", suite)
    about_action.triggered.connect(lambda: AboutDialog(suite).exec_())
    help_menu.addAction(about_action)

    step(90, "Loading log history...",        "Decision logs, trade history")
    step(100, "Ready.",                       f"{APP_NAME} v{APP_VERSION}")
    time.sleep(0.5)

    # Hide loader and show main window
    # IMPORTANT: hide() instead of close() to prevent Qt from exiting the app
    # since close() on the last visible widget can trigger app quit
    loader.hide()
    loader.deleteLater()
    suite.show()
    suite.raise_()
    suite.activateWindow()

    sys.exit(app.exec_())


if __name__ == "__main__":
    boot_app()
