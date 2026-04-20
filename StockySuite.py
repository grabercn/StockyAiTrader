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
from core.branding import get_stylesheet, detect_system_theme, chart_colors
from core.event_bus import EventBus
from core.risk import RiskManager
from core.broker import AlpacaBroker
import yfinance as yf
from core.scanner import scan_multiple, ScanResult
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
        from core.widgets import StatCard, GradientDivider
        from core.ui.backgrounds import GradientHeader
        from core.ui.animations import FadeIn, StaggeredFadeIn

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        # Header
        header = GradientHeader("Dashboard", "Portfolio overview and recent activity")
        layout.addWidget(header)

        # Account stats row — premium stat cards
        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)
        self.card_portfolio = StatCard("Portfolio Value", "--", BRAND_PRIMARY)
        self.card_buying = StatCard("Buying Power", "--", BRAND_SECONDARY)
        self.card_cash = StatCard("Cash", "--", TEXT_SECONDARY)
        self.card_pnl = StatCard("Day P&L", "--", BRAND_ACCENT)
        self._stat_cards = [self.card_portfolio, self.card_buying, self.card_cash, self.card_pnl]
        for card in self._stat_cards:
            cards_row.addWidget(card)
        layout.addLayout(cards_row)

        # Splitter: chart (top, stretchy) + positions/activity (bottom)
        splitter = QSplitter(Qt.Vertical)

        # Chart — takes priority space
        chart_widget = QWidget()
        cl = QVBoxLayout()
        cl.setContentsMargins(0, 4, 0, 0)
        self.figure = plt.Figure(dpi=100, facecolor=chart_colors()["fig_bg"])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(140)
        cl.addWidget(self.canvas)
        chart_widget.setLayout(cl)
        splitter.addWidget(chart_widget)

        # Bottom: positions + activity side by side
        bottom = QWidget()
        bl = QHBoxLayout()
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)

        # Positions
        pos_widget = QWidget()
        pl = QVBoxLayout()
        pl.setContentsMargins(0, 0, 0, 0)
        pos_label = QLabel("Open Positions")
        pos_label.setFont(QFont(FONT_FAMILY, 11, QFont.Bold))
        pos_label.setStyleSheet(f"color: {BRAND_PRIMARY};")
        pl.addWidget(pos_label)

        self.pos_table = QTableWidget()
        self.pos_table.setColumnCount(8)
        self.pos_table.setHorizontalHeaderLabels(
            ["Symbol", "Side", "Qty", "Avg Cost", "Current", "P&L", "P&L %", ""])
        for c in range(7):
            self.pos_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
        self.pos_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.pos_table.verticalHeader().setVisible(False)
        pl.addWidget(self.pos_table)
        pos_widget.setLayout(pl)
        bl.addWidget(pos_widget, 3)

        # Activity feed
        act_widget = QWidget()
        al = QVBoxLayout()
        al.setContentsMargins(0, 0, 0, 0)
        act_label = QLabel("Recent Activity")
        act_label.setFont(QFont(FONT_FAMILY, 11, QFont.Bold))
        act_label.setStyleSheet(f"color: {BRAND_PRIMARY};")
        al.addWidget(act_label)

        self.activity_feed = QTextEdit()
        self.activity_feed.setReadOnly(True)
        al.addWidget(self.activity_feed)
        act_widget.setLayout(al)
        bl.addWidget(act_widget, 2)

        bottom.setLayout(bl)
        splitter.addWidget(bottom)
        splitter.setSizes([300, 200])

        layout.addWidget(splitter)
        self.bus.log_entry.connect(self._on_activity)
        self.setLayout(layout)

        # Auto-refresh timer (every 60 seconds)
        self._refresh_interval = 60
        self._countdown = self._refresh_interval
        self._refresh_countdown_label = QLabel(f"Next refresh: {self._countdown}s")
        self._refresh_countdown_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")
        self._refresh_countdown_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self._refresh_countdown_label)

        self._auto_refresh = QTimer(self)
        self._auto_refresh.timeout.connect(self._tick_refresh)
        self._auto_refresh.start(1000)

        # Wire stat card clicks to detail popups
        self.card_portfolio.on_clicked = lambda: self._show_popup("equity")
        self.card_pnl.on_clicked = lambda: self._show_popup("pnl")
        self.card_buying.on_clicked = lambda: self._show_popup("buying")
        self.card_cash.on_clicked = lambda: self._show_popup("equity")

        # Staggered fade-in animation for stat cards on first show
        QTimer.singleShot(200, lambda: StaggeredFadeIn(self._stat_cards, delay_ms=100, duration=400))

    def _tick_refresh(self):
        """Auto-refresh countdown — refreshes dashboard data every 60 seconds."""
        self._countdown -= 1
        if self._countdown <= 0:
            self._countdown = self._refresh_interval
            self.refresh()
        self._refresh_countdown_label.setText(f"Next refresh: {self._countdown}s")

    def _show_popup(self, kind):
        if not self.broker:
            self.bus.log_entry.emit("Connect Alpaca API first — go to Settings", "warn")
            return
        from core.ui.detail_popup import show_equity_popup, show_pnl_popup, show_buying_power_popup
        if kind == "equity":
            show_equity_popup(self.broker, self)
        elif kind == "pnl":
            show_pnl_popup(self.broker, self)
        elif kind == "buying":
            show_buying_power_popup(self.broker, self)

    def _on_activity(self, msg, level):
        from core.branding import log_html
        self.activity_feed.append(log_html(msg, level))

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

        self.card_portfolio.set_value(f"${pv:,.2f}")
        self.card_buying.set_value(f"${bp:,.2f}")
        self.card_cash.set_value(f"${cash:,.2f}")
        pnl_color = COLOR_PROFIT if pnl >= 0 else COLOR_LOSS
        self.card_pnl.set_value(f"${pnl:+,.2f} ({pct:+.2f}%)", pnl_color)

        # Positions with per-stock sell buttons
        positions = self.broker.get_positions()
        if isinstance(positions, list):
            self.pos_table.setRowCount(len(positions))
            for i, p in enumerate(positions):
                unrealized = float(p.get("unrealized_pl", 0))
                sym = p.get("symbol", "")
                qty = float(p.get("qty", 0))
                vals = [
                    sym, p.get("side", ""),
                    f"{qty:.0f}",
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

                # Sell button per stock
                sell_btn = QPushButton(f"Sell")
                sell_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 10px; padding: 3px 8px;")
                sell_btn.clicked.connect(lambda _, s=sym, q=int(qty): self._sell_position(s, q))
                self.pos_table.setCellWidget(i, 7, sell_btn)

        # Chart
        hist = self.broker.get_portfolio_history(period="1W", timeframe="1H")
        if "error" not in hist and hist.get("equity"):
            self._plot(hist)

    def _plot(self, hist):
        self.figure.clear()
        cc = chart_colors(); self.figure.set_facecolor(cc["fig_bg"])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(cc["ax_bg"])
        eq = [e for e in hist["equity"] if e is not None]
        ts = [datetime.fromtimestamp(t) for t, e in zip(hist["timestamp"], hist["equity"]) if e is not None]
        ax.plot(ts, eq, color=BRAND_PRIMARY, linewidth=1.5)
        ax.fill_between(ts, eq, alpha=0.08, color=BRAND_PRIMARY)
        ax.set_title("Portfolio Equity (1W)", color=cc["text"], fontsize=10)
        ax.tick_params(colors=cc["muted"], labelsize=8)
        ax.grid(True, alpha=0.15, color=cc["grid"])
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d"))
        self.figure.tight_layout()
        self.canvas.draw()

        # Add hover tooltip
        from core.ui.chart_tooltip import ChartTooltip
        self._tooltip = ChartTooltip(self.canvas, ax, ts, eq)

    def _sell_position(self, symbol, max_qty):
        """Sell a specific position with qty picker. Uses close_position endpoint."""
        if not self.broker:
            self.bus.log_entry.emit("Alpaca API not connected", "error")
            return

        from PyQt5.QtWidgets import QInputDialog
        qty, ok = QInputDialog.getInt(
            self, f"Sell {symbol}",
            f"How many shares of {symbol} to sell?\n(You own {max_qty})",
            value=max_qty, min=1, max=max_qty,
        )
        if not ok:
            return

        # Use close_position endpoint (not place_order) — avoids 403 short sell restriction
        result = self.broker.close_position(symbol, qty=qty)
        if "error" in result:
            self.bus.log_entry.emit(f"Sell {symbol} x{qty} failed: {result['error']}", "error")
        else:
            self.bus.log_entry.emit(f"Sold {symbol} x{qty} — order {result.get('id', '?')}", "trade")
            self.bus.positions_changed.emit()
            QTimer.singleShot(2000, self.refresh)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: SCANNER
# ═════════════════════════════════════════════════════════════════════════════

class ScannerPanel(QWidget):
    """Dynamic multi-stock scanner with live discovery, detail panel, and auto-invest."""

    def __init__(self, broker, risk_manager, event_bus):
        super().__init__()
        self.broker = broker
        self.rm = risk_manager
        self.bus = event_bus
        self.results = []
        self.selected = set()
        self._build()

    def _build(self):
        from core.ui.backgrounds import GradientHeader
        from core.widgets import DetailedProgressBar
        from core.discovery import get_all_sectors

        outer = QVBoxLayout()
        outer.setSpacing(8)
        outer.setContentsMargins(8, 6, 8, 6)

        header = GradientHeader("Scanner", "Live market discovery + AI-ranked opportunities")
        outer.addWidget(header)

        # ── Splitter: left (controls+results) / right (detail panel) ──
        splitter = QSplitter(Qt.Horizontal)

        # ── LEFT: Controls + Results ──
        left = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        # Source selector row
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source:"))
        self.source_cb = QComboBox()
        self.source_cb.addItems([
            "Custom Tickers",
            "Most Active Today",
            "Day Gainers",
            "Day Losers",
            "Trending (Social)",
            "High Volume",
        ])
        # Add sectors
        for sector in get_all_sectors():
            self.source_cb.addItem(f"Sector: {sector}")
        # Add watchlists
        self._refresh_watchlist_items()
        self.source_cb.currentIndexChanged.connect(self._on_source_changed)
        source_row.addWidget(self.source_cb, 1)

        # Watchlist management
        self.wl_name = QLineEdit()
        self.wl_name.setPlaceholderText("Watchlist name")
        self.wl_name.setFixedWidth(120)
        source_row.addWidget(self.wl_name)
        save_wl = QPushButton("Save WL")
        save_wl.setStyleSheet(f"font-size: 10px; padding: 5px; background-color: {BG_INPUT};")
        save_wl.clicked.connect(self._save_watchlist)
        source_row.addWidget(save_wl)
        layout.addLayout(source_row)

        # Ticker input (shown for Custom, hidden for live sources)
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter tickers: AAPL, TSLA, NVDA, MSFT...")
        layout.addWidget(self.ticker_input)

        # Scan settings row
        settings_row = QHBoxLayout()
        period_lbl = QLabel("Training Data:")
        period_lbl.setToolTip("How many days of historical price data the AI model trains on.\nMore days = more context but slower. 5d is recommended for day trading.")
        settings_row.addWidget(period_lbl)
        self.period_cb = QComboBox()
        self.period_cb.addItems(["5 days", "3 days", "2 days", "1 day"])
        self.period_cb.setToolTip("Amount of historical data used to train the AI model for each stock")
        settings_row.addWidget(self.period_cb)
        interval_lbl = QLabel("Bar Size:")
        interval_lbl.setToolTip("The time interval for each price bar.\n5min = one data point every 5 minutes.\nSmaller bars = more data points but noisier.")
        settings_row.addWidget(interval_lbl)
        self.interval_cb = QComboBox()
        self.interval_cb.addItems(["5 min", "1 min", "15 min"])
        self.interval_cb.setToolTip("Price bar interval — how frequently data points are sampled")
        settings_row.addWidget(self.interval_cb)

        # Help note
        help_lbl = QLabel("ℹ Training Data = how far back to look. Bar Size = data resolution.")
        help_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")
        settings_row.addWidget(help_lbl)
        settings_row.addStretch()

        self.scan_btn = QPushButton("  SCAN & RANK")
        from core.ui.icons import StockyIcons
        self.scan_btn.setIcon(StockyIcons.get_icon("scan", 18, "white"))
        self.scan_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 13px; padding: 10px 20px;")
        self.scan_btn.clicked.connect(self._start_scan)
        settings_row.addWidget(self.scan_btn)
        layout.addLayout(settings_row)

        # Progress
        self.progress = DetailedProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Results table (added Monitor column)
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(
            ["", "Ticker", "Signal", "Conf", "Price", "Shares", "SL/TP", "Score", "Monitor", "Reasoning"])
        for c in range(9):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(9, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellClicked.connect(self._on_row_clicked)
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
        self.invest_btn = QPushButton("  AUTO-INVEST SELECTED")
        self.invest_btn.setIcon(StockyIcons.get_icon("bolt", 16, "white"))
        self.invest_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 13px;")
        self.invest_btn.clicked.connect(self._auto_invest)
        arow.addWidget(self.invest_btn)
        layout.addLayout(arow)

        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {BRAND_PRIMARY}; font-size: 11px;")
        layout.addWidget(self.summary)

        left.setLayout(layout)
        splitter.addWidget(left)

        # ── RIGHT: Detail Panel (shows when a stock is clicked) ──
        self.detail_panel = QWidget()
        dp_layout = QVBoxLayout()
        dp_layout.setSpacing(6)
        dp_layout.setContentsMargins(8, 4, 8, 4)

        self.detail_title = QLabel("Click a stock to view details")
        self.detail_title.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        self.detail_title.setStyleSheet(f"color: {BRAND_PRIMARY};")
        dp_layout.addWidget(self.detail_title)

        # Mini chart
        self.detail_figure = plt.Figure(figsize=(4, 2.5), dpi=100, facecolor=chart_colors()["fig_bg"])
        self.detail_canvas = FigureCanvas(self.detail_figure)
        self.detail_canvas.setMinimumHeight(150)
        dp_layout.addWidget(self.detail_canvas)

        # Stats grid
        self.detail_stats = QTextEdit()
        self.detail_stats.setReadOnly(True)
        self.detail_stats.setFont(QFont(FONT_MONO, 10))
        dp_layout.addWidget(self.detail_stats)

        # Quick-trade controls
        trade_box = QGroupBox("Quick Trade")
        tbl = QVBoxLayout()
        tbl.setSpacing(4)

        # Quantity row
        qty_row = QHBoxLayout()
        qty_row.addWidget(QLabel("Shares:"))
        self.detail_qty = QSpinBox()
        self.detail_qty.setRange(1, 10000)
        self.detail_qty.setValue(1)
        self.detail_qty.setStyleSheet(f"padding: 4px 8px; min-width: 80px;")
        qty_row.addWidget(self.detail_qty)
        self.detail_qty_label = QLabel("")
        self.detail_qty_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")
        qty_row.addWidget(self.detail_qty_label, 1)
        tbl.addLayout(qty_row)

        # Buy button
        buy_row = QHBoxLayout()
        self.detail_buy_btn = QPushButton("  BUY")
        self.detail_buy_btn.setIcon(StockyIcons.get_icon("arrow_up", 14, "white"))
        self.detail_buy_btn.setStyleSheet(f"background-color: {COLOR_BUY}; font-size: 12px; padding: 8px;")
        self.detail_buy_btn.clicked.connect(lambda: self._quick_trade("buy"))
        buy_row.addWidget(self.detail_buy_btn)
        tbl.addLayout(buy_row)

        # Sell button (hidden until user owns the stock)
        sell_row = QHBoxLayout()
        self.detail_sell_btn = QPushButton("  SELL")
        self.detail_sell_btn.setIcon(StockyIcons.get_icon("arrow_down", 14, "white"))
        self.detail_sell_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 12px; padding: 8px;")
        self.detail_sell_btn.clicked.connect(lambda: self._quick_trade("sell"))
        self.detail_sell_btn.setVisible(False)
        sell_row.addWidget(self.detail_sell_btn)
        self.detail_owned_label = QLabel("")
        self.detail_owned_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        sell_row.addWidget(self.detail_owned_label)
        tbl.addLayout(sell_row)

        # Deep analyze button
        self.detail_analyze_btn = QPushButton("  Deep Analyze")
        self.detail_analyze_btn.setIcon(StockyIcons.get_icon("brain", 14, BRAND_PRIMARY))
        self.detail_analyze_btn.setStyleSheet(f"background-color: {BG_INPUT}; font-size: 11px; padding: 6px;")
        self.detail_analyze_btn.clicked.connect(self._deep_analyze)
        tbl.addWidget(self.detail_analyze_btn)

        trade_box.setLayout(tbl)
        dp_layout.addWidget(trade_box)

        self.detail_panel.setLayout(dp_layout)
        splitter.addWidget(self.detail_panel)
        splitter.setSizes([600, 300])

        outer.addWidget(splitter)
        self.setLayout(outer)
        self._selected_result = None

    # ── Source handling ────────────────────────────────────────────────────

    def _on_source_changed(self, idx):
        text = self.source_cb.currentText()
        # Show/hide ticker input based on source
        is_custom = text == "Custom Tickers"
        self.ticker_input.setVisible(is_custom)
        if not is_custom:
            self.ticker_input.clear()

    def _refresh_watchlist_items(self):
        from core.discovery import get_watchlists
        wls = get_watchlists()
        # Remove old watchlist entries from combo
        # (they're after the sectors)
        while self.source_cb.count() > 0 and self.source_cb.itemText(self.source_cb.count()-1).startswith("Watchlist:"):
            self.source_cb.removeItem(self.source_cb.count()-1)
        for name in wls:
            if wls[name]:  # Only show non-empty watchlists
                self.source_cb.addItem(f"Watchlist: {name}")

    def _save_watchlist(self):
        from core.discovery import save_watchlist
        name = self.wl_name.text().strip()
        if not name:
            self.bus.log_entry.emit("Enter a watchlist name first", "warn")
            return
        tickers = self._get_tickers_from_input()
        if not tickers:
            self.bus.log_entry.emit("No tickers to save — scan first or enter tickers", "warn")
            return
        save_watchlist(name, tickers)
        self._refresh_watchlist_items()
        self.bus.log_entry.emit(f"Watchlist '{name}' saved with {len(tickers)} tickers", "info")

    def _get_tickers_from_source(self):
        """Resolve current source selection to a list of tickers."""
        from core.discovery import (
            get_most_active, get_day_gainers, get_day_losers,
            get_trending_social, get_high_volume, get_sector_tickers,
            get_watchlists,
        )

        source = self.source_cb.currentText()

        if source == "Custom Tickers":
            return self._get_tickers_from_input()
        elif source == "Most Active Today":
            return get_most_active(25)
        elif source == "Day Gainers":
            return get_day_gainers(20)
        elif source == "Day Losers":
            return get_day_losers(20)
        elif source == "Trending (Social)":
            return get_trending_social(15)
        elif source == "High Volume":
            return get_high_volume(5_000_000, 20)
        elif source.startswith("Sector: "):
            sector = source.replace("Sector: ", "")
            return get_sector_tickers(sector, 15)
        elif source.startswith("Watchlist: "):
            wl_name = source.replace("Watchlist: ", "")
            return get_watchlists().get(wl_name, [])
        return []

    def _get_tickers_from_input(self):
        text = self.ticker_input.text().strip()
        if text:
            return [t.strip().upper() for t in text.replace(";", ",").split(",") if t.strip()]
        # If no input but we have results, use those tickers
        if self.results:
            return [r.ticker for r in self.results]
        return []

    # ── Scanning ──────────────────────────────────────────────────────────

    def _start_scan(self):
        source = self.source_cb.currentText()

        if source == "Custom Tickers":
            tickers = self._get_tickers_from_input()
            if not tickers:
                self.bus.log_entry.emit("Enter tickers to scan or choose a live source", "warn")
                return
        else:
            # Fetch from live source
            self.progress.setVisible(True)
            self.progress.reset()
            self.progress.set_progress(5, f"Fetching {source}...", "Discovering tickers from live market data")
            self.progress.add_log(f"Source: {source}")
            QApplication.processEvents()

            tickers = self._get_tickers_from_source()
            if not tickers:
                self.progress.set_progress(100, "No tickers found", f"Source '{source}' returned no results")
                self.bus.log_entry.emit(f"No tickers found from {source}", "warn")
                return

            self.progress.set_progress(10, f"Found {len(tickers)} tickers", ", ".join(tickers[:8]) + ("..." if len(tickers) > 8 else ""))
            self.progress.add_log(f"Discovered {len(tickers)} tickers: {', '.join(tickers[:10])}")
            # Also populate the input box so user can see/edit
            self.ticker_input.setText(", ".join(tickers))
            self.ticker_input.setVisible(True)

        self.scan_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.bus.scan_started.emit(len(tickers))
        self.bus.log_entry.emit(f"Scanning {len(tickers)} tickers from {source}...", "info")
        self._t0 = time.time()

        # Map display text back to API values
        period_map = {"5 days": "5d", "3 days": "3d", "2 days": "2d", "1 day": "1d"}
        interval_map = {"5 min": "5m", "1 min": "1m", "15 min": "15m"}
        period = period_map.get(self.period_cb.currentText(), "5d")
        interval = interval_map.get(self.interval_cb.currentText(), "5m")

        self._worker = ScanWorker(tickers, period, interval, self.rm)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_progress(self, done, total, ticker, action):
        pct = 10 + int(done / total * 85) if total > 0 else 0
        self.progress.set_progress(pct, f"Scanning {ticker}...", f"{done}/{total} complete")
        colors = {"BUY": "#10b981", "SELL": "#ef4444"}
        self.progress.add_log(f"{ticker}: <b style='color:{colors.get(action, '#94a3b8')}'>{action}</b>")

    def _on_done(self, results):
        self.results = results
        elapsed = time.time() - self._t0
        self.progress.set_progress(100, f"Scan complete — {len(results)} stocks analyzed", f"{elapsed:.1f}s")
        self.scan_btn.setEnabled(True)
        self.selected.clear()

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
                (f"{r.score:.2f}", None),
            ]
            for j, (val, color) in enumerate(items):
                it = QTableWidgetItem(val)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if color:
                    it.setForeground(QColor(color))
                    it.setFont(QFont(FONT_MONO, 11, QFont.Bold))
                if j == 0:
                    it.setFont(QFont(FONT_MONO, 11, QFont.Bold))
                self.table.setItem(i, j+1, it)

            # Auto-trade toggle button with icon
            is_monitored = hasattr(self, '_auto_service') and self._auto_service and self._auto_service.is_monitoring(r.ticker)
            monitor_btn = QPushButton()
            if is_monitored:
                monitor_btn.setIcon(StockyIcons.get_icon("robot", 14, BRAND_ACCENT))
                monitor_btn.setToolTip(f"Auto-trading {r.ticker} — click to stop")
                monitor_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}30; border: 1px solid {BRAND_ACCENT}; padding: 3px 6px; border-radius: 4px;")
            else:
                monitor_btn.setIcon(StockyIcons.get_icon("play", 14, TEXT_MUTED))
                monitor_btn.setToolTip(f"Start auto-trading {r.ticker}")
                monitor_btn.setStyleSheet(f"background-color: transparent; border: 1px solid {BORDER}; padding: 3px 6px; border-radius: 4px;")
            monitor_btn.clicked.connect(lambda _, t=r.ticker: self._toggle_auto_trade(t))
            self.table.setCellWidget(i, 8, monitor_btn)

            # Reasoning (last column)
            reason_it = QTableWidgetItem(r.reasoning[:80] if r.reasoning else r.error or "")
            reason_it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 9, reason_it)

        buys = sum(1 for r in results if r.action == "BUY")
        sells = sum(1 for r in results if r.action == "SELL")
        errors = sum(1 for r in results if r.error)
        self.summary.setText(
            f"{buys} BUY  |  {sells} SELL  |  {len(results)-buys-sells-errors} HOLD"
            f"  |  {errors} errors  |  {elapsed:.1f}s"
        )
        self.bus.scan_completed.emit([{"ticker": r.ticker, "action": r.action} for r in results])
        self.bus.log_entry.emit(f"Scan done: {buys} BUY, {sells} SELL in {elapsed:.1f}s", "trade")

    # ── Detail Panel ──────────────────────────────────────────────────────

    def _on_row_clicked(self, row, col):
        if row < 0 or row >= len(self.results):
            return
        r = self.results[row]
        self._selected_result = r
        self._show_detail(r)

    def _show_detail(self, r):
        """Show detailed breakdown for a scanned stock."""
        self.detail_title.setText(f"{r.action}  {r.ticker}  —  ${r.price:.2f}")
        colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
        self.detail_title.setStyleSheet(f"color: {colors.get(r.action, BRAND_PRIMARY)};")

        # Stats
        lines = []
        lines.append(f'<b style="color:{BRAND_PRIMARY}">Signal: {r.action}</b> ({r.confidence:.1%} confidence)')
        lines.append(f'<b>Price:</b> ${r.price:.2f}  |  <b>ATR:</b> ${r.atr:.2f} ({r.atr/r.price*100:.1f}%)')
        lines.append(f'<b>Position:</b> {r.position_size} shares')
        lines.append(f'<b>Stop Loss:</b> ${r.stop_loss:.2f}  |  <b>Take Profit:</b> ${r.take_profit:.2f}')
        lines.append(f'<b>Score:</b> {r.score:.3f}')
        lines.append("")
        lines.append(f'<b style="color:{BRAND_PRIMARY}">Probabilities</b>')
        lines.append(f'  <span style="color:{COLOR_SELL}">SELL: {r.probs[0]:.1%}</span>'
                     f'  <span style="color:{COLOR_HOLD}">HOLD: {r.probs[1]:.1%}</span>'
                     f'  <span style="color:{COLOR_BUY}">BUY: {r.probs[2]:.1%}</span>')

        if r.feature_importances:
            lines.append("")
            lines.append(f'<b style="color:{BRAND_PRIMARY}">Key Drivers</b>')
            for feat, imp in sorted(r.feature_importances.items(), key=lambda x: -x[1])[:8]:
                bar_len = int(min(imp / 100, 20))
                bar = "█" * bar_len
                lines.append(f'  <span style="color:{BRAND_ACCENT}">{bar}</span> {feat}: {imp:.0f}')

        # Trade metadata
        lines.append("")
        lines.append(f'<b style="color:{BRAND_PRIMARY}">Trade Info</b>')
        trade_type = "Intraday (Day Trade)" if r.atr and r.atr / r.price < 0.03 else "Swing Trade"
        interval_txt = self.interval_cb.currentText() if hasattr(self, 'interval_cb') else "5 min"
        period_txt = self.period_cb.currentText() if hasattr(self, 'period_cb') else "5 days"
        lines.append(f'  Type: {trade_type}')
        lines.append(f'  Training Data: {period_txt}  |  Bar Size: {interval_txt}')
        lines.append(f'  Volatility: {"High" if r.atr/r.price > 0.02 else "Low"} ({r.atr/r.price*100:.1f}% ATR)')

        if r.reasoning:
            lines.append("")
            lines.append(f'<b style="color:{BRAND_PRIMARY}">Reasoning</b>')
            for part in r.reasoning.split(" | "):
                lines.append(f'  {part}')

        self.detail_stats.setHtml("<br>".join(lines))

        # Set quantity to AI recommendation
        qty = r.position_size if r.position_size > 0 else 1
        self.detail_qty.setValue(qty)
        self.detail_qty_label.setText(f"AI recommends {r.position_size} shares")

        # Update buy button text with quantity and cost
        cost = qty * r.price if r.price > 0 else 0
        self.detail_buy_btn.setText(f"  BUY {qty} shares (${cost:,.0f})")

        # Check if user owns this stock — show/hide sell button
        owned_qty = 0
        if self.broker:
            try:
                positions = self.broker.get_positions()
                if isinstance(positions, list):
                    for p in positions:
                        if p.get("symbol", "").upper() == r.ticker.upper():
                            owned_qty = int(float(p.get("qty", 0)))
                            break
            except Exception:
                pass

        if owned_qty > 0:
            self.detail_sell_btn.setVisible(True)
            self.detail_sell_btn.setText(f"  SELL {min(qty, owned_qty)} shares")
            self.detail_owned_label.setText(f"You own {owned_qty} shares")
            # Default sell qty to what AI recommends or what user owns, whichever is less
            if r.action == "SELL":
                self.detail_qty.setValue(min(r.position_size, owned_qty))
        else:
            self.detail_sell_btn.setVisible(False)
            self.detail_owned_label.setText("You don't own this stock")

        # Update buttons when user changes quantity
        try:
            self.detail_qty.valueChanged.disconnect()
        except Exception:
            pass
        self.detail_qty.valueChanged.connect(
            lambda v: (
                self.detail_buy_btn.setText(f"  BUY {v} shares (${v * r.price:,.0f})"),
                self.detail_sell_btn.setText(f"  SELL {min(v, owned_qty)} shares") if owned_qty > 0 else None,
            )
        )

        # Chart
        self._draw_detail_chart(r.ticker)

    def _draw_detail_chart(self, ticker):
        """Fetch and draw a quick price chart for the detail panel."""
        self.detail_figure.clear()
        cc = chart_colors(); self.detail_figure.set_facecolor(cc["fig_bg"])
        try:
            data = yf.Ticker(ticker).history(period="1mo", interval="1d")
            if data.empty:
                return
            ax = self.detail_figure.add_subplot(111)
            ax.set_facecolor(cc["ax_bg"])
            closes = data["Close"].values
            x = range(len(closes))

            # Determine color by trend
            trending_up = closes[-1] >= closes[0]
            color = COLOR_BUY if trending_up else COLOR_SELL

            ax.plot(x, closes, color=color, linewidth=1.5)
            ax.fill_between(x, closes, alpha=0.08, color=color)
            ax.set_title(f"{ticker} — 1 Month", color=cc["text"], fontsize=10)
            ax.tick_params(colors=cc["muted"], labelsize=7)
            ax.grid(True, alpha=0.15, color=cc["grid"])
            self.detail_figure.tight_layout()
        except Exception:
            pass
        self.detail_canvas.draw()

    def _quick_trade(self, side):
        if not self._selected_result:
            self.bus.log_entry.emit("Select a stock from the scan results first", "warn")
            return
        if not self.broker:
            self.bus.log_entry.emit("Alpaca API not connected — go to Settings to configure", "error")
            return
        r = self._selected_result
        qty = self.detail_qty.value()
        if qty <= 0:
            self.bus.log_entry.emit(f"Set quantity to at least 1 share", "warn")
            return

        cost = qty * r.price
        confirm = QMessageBox.question(
            self, f"Confirm {side.upper()}",
            f"{side.upper()} {qty} shares of {r.ticker}\n"
            f"Price: ${r.price:.2f}\n"
            f"Total: ${cost:,.2f}\n"
            f"Stop Loss: ${r.stop_loss:.2f}\n"
            f"Take Profit: ${r.take_profit:.2f}\n\n"
            f"Proceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        # Use close_position for sells (avoids 403 short sell restriction)
        # Use place_order for buys
        if side == "sell":
            result = self.broker.close_position(r.ticker, qty=qty)
        else:
            result = self.broker.place_order(r.ticker, qty, side,
                                             stop_loss=r.stop_loss, take_profit=r.take_profit)
        if "error" in result:
            self.bus.log_entry.emit(f"{side.upper()} {r.ticker} x{qty} failed: {result['error']}", "error")
        else:
            self.bus.log_entry.emit(f"{side.upper()} {r.ticker} x{qty} — order {result.get('id','?')}", "trade")
            self.bus.positions_changed.emit()
            QTimer.singleShot(2000, lambda: self._show_detail(r))

    def _deep_analyze(self):
        """Open a full technical report popup for the selected stock."""
        if not self._selected_result:
            self.bus.log_entry.emit("Select a stock first", "warn")
            return

        r = self._selected_result

        # Build a comprehensive report
        report = []
        report.append(f"{'='*60}")
        report.append(f"  DEEP ANALYSIS REPORT — {r.ticker}")
        report.append(f"  Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"{'='*60}")
        report.append("")
        report.append(f"SIGNAL: {r.action}  |  Confidence: {r.confidence:.1%}  |  Score: {r.score:.3f}")
        report.append(f"Price: ${r.price:.2f}  |  ATR: ${r.atr:.2f} ({r.atr/r.price*100:.1f}%)")
        report.append("")
        report.append(f"{'─'*60}")
        report.append("PROBABILITIES")
        report.append(f"  SELL:  {r.probs[0]:.1%}  {'█' * int(r.probs[0]*30)}")
        report.append(f"  HOLD:  {r.probs[1]:.1%}  {'█' * int(r.probs[1]*30)}")
        report.append(f"  BUY:   {r.probs[2]:.1%}  {'█' * int(r.probs[2]*30)}")
        report.append("")
        report.append(f"{'─'*60}")
        report.append("RISK MANAGEMENT")
        report.append(f"  Position Size:  {r.position_size} shares")
        report.append(f"  Stop Loss:      ${r.stop_loss:.2f} ({(r.price-r.stop_loss)/r.price*100:.1f}% from entry)")
        report.append(f"  Take Profit:    ${r.take_profit:.2f} ({(r.take_profit-r.price)/r.price*100:.1f}% from entry)")
        report.append(f"  Risk/Reward:    1:{(r.take_profit-r.price)/(r.price-r.stop_loss):.1f}" if r.stop_loss < r.price else "")
        report.append(f"  Max Loss:       ${r.position_size * (r.price - r.stop_loss):,.2f}")
        report.append(f"  Max Gain:       ${r.position_size * (r.take_profit - r.price):,.2f}")

        if r.feature_importances:
            report.append("")
            report.append(f"{'─'*60}")
            report.append("TOP FEATURE DRIVERS (what the AI focused on)")
            for feat, imp in sorted(r.feature_importances.items(), key=lambda x: -x[1])[:10]:
                bar = "█" * int(min(imp / 50, 30))
                report.append(f"  {bar} {feat}: {imp:.0f}")

        if r.reasoning:
            report.append("")
            report.append(f"{'─'*60}")
            report.append("AI REASONING")
            for part in r.reasoning.split(" | "):
                report.append(f"  • {part}")

        report.append("")
        report.append(f"{'='*60}")

        # Show in a dialog
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Deep Analysis — {r.ticker}")
        dlg.setMinimumSize(600, 500)
        bg = theme.color("bg_base") if hasattr(theme, 'color') else BG_DARKEST
        dlg.setStyleSheet(f"QDialog {{ background-color: {bg}; }}")
        lay = QVBoxLayout()
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFont(QFont(FONT_MONO, 10))
        txt.setPlainText("\n".join(report))
        lay.addWidget(txt)

        # Also send to Day Trade
        btn_row = QHBoxLayout()
        day_btn = QPushButton("Open in Day Trade")
        day_btn.clicked.connect(lambda: (self.bus.ticker_selected.emit(r.ticker), dlg.accept()))
        btn_row.addWidget(day_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        dlg.setLayout(lay)
        dlg.exec_()

    # ── Auto-Trade Monitoring ───────────────────────────────────────────

    def _get_auto_service(self):
        """Get or create the auto-trader service."""
        if not hasattr(self, '_auto_service') or self._auto_service is None:
            from core.auto_trader import AutoTraderService
            self._auto_service = AutoTraderService(broker=self.broker, risk_manager=self.rm)
            self._auto_service.log.connect(self.bus.log_entry.emit)
            self._auto_service.stock_updated.connect(self._on_auto_update)
            self._auto_service.trade_executed.connect(
                lambda t, s, q, o: self.bus.trade_executed.emit(t, s, q, o)
            )
            self._auto_service.start()
        return self._auto_service

    def _toggle_auto_trade(self, ticker):
        """Toggle auto-trade monitoring for a stock."""
        svc = self._get_auto_service()
        period_map = {"5 days": "5d", "3 days": "3d", "2 days": "2d", "1 day": "1d"}
        interval_map = {"5 min": "5m", "1 min": "1m", "15 min": "15m"}
        period = period_map.get(self.period_cb.currentText(), "5d")
        interval = interval_map.get(self.interval_cb.currentText(), "5m")

        if svc.is_monitoring(ticker):
            svc.remove_stock(ticker)
            self.bus.log_entry.emit(f"Stopped auto-trading {ticker}", "info")
        else:
            svc.add_stock(ticker, period=period, interval=interval, auto_execute=True, min_confidence=0.5)
            self.bus.log_entry.emit(
                f"Auto-trading {ticker} — {period} data, checking every {interval}",
                "trade",
            )

        # Refresh table to update button icons
        if self.results:
            self._refresh_monitor_icons()

    def _refresh_monitor_icons(self):
        """Update all monitor toggle buttons to reflect current state."""
        from core.ui.icons import StockyIcons
        svc = self._auto_service if hasattr(self, '_auto_service') and self._auto_service else None
        for i in range(self.table.rowCount()):
            ticker_item = self.table.item(i, 1)
            if not ticker_item:
                continue
            ticker = ticker_item.text()
            btn = self.table.cellWidget(i, 8)
            if not btn:
                continue
            is_mon = svc and svc.is_monitoring(ticker)
            if is_mon:
                btn.setIcon(StockyIcons.get_icon("robot", 14, BRAND_ACCENT))
                btn.setToolTip(f"Auto-trading {ticker} — click to stop")
                btn.setStyleSheet(f"background-color: {BRAND_ACCENT}30; border: 1px solid {BRAND_ACCENT}; padding: 3px 6px; border-radius: 4px;")
            else:
                btn.setIcon(StockyIcons.get_icon("play", 14, TEXT_MUTED))
                btn.setToolTip(f"Start auto-trading {ticker}")
                btn.setStyleSheet(f"background-color: transparent; border: 1px solid {BORDER}; padding: 3px 6px; border-radius: 4px;")

    def _on_auto_update(self, ticker, action, confidence, price, next_secs):
        """Called by auto-trader service when a stock is checked."""
        # Update the table row if this ticker is visible
        for i in range(self.table.rowCount()):
            ticker_item = self.table.item(i, 1)
            if ticker_item and ticker_item.text() == ticker:
                # Update signal
                sig_item = self.table.item(i, 2)
                if sig_item:
                    sig_item.setText(action)
                    colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
                    sig_item.setForeground(QColor(colors.get(action, TEXT_MUTED)))
                # Update confidence
                conf_item = self.table.item(i, 3)
                if conf_item:
                    conf_item.setText(f"{confidence:.0%}")
                # Update price
                price_item = self.table.item(i, 4)
                if price_item:
                    price_item.setText(f"${price:.2f}")
                break

    # ── Selection / Invest ────────────────────────────────────────────────

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
            self.bus.log_entry.emit("Select tickers first — check the boxes next to stocks you want to invest in", "warn")
            return
        if not self.broker:
            self.bus.log_entry.emit("Alpaca API not connected — go to Settings tab and enter your API keys from alpaca.markets", "error")
            return

        actionable = [r for r in self.results if r.ticker in self.selected and r.action in ("BUY", "SELL") and r.position_size > 0]
        if not actionable:
            self.bus.log_entry.emit("No actionable signals in selection — all HOLD or 0 shares", "warn")
            return

        for r in actionable:
            side = "buy" if r.action == "BUY" else "sell"
            if side == "sell":
                result = self.broker.close_position(r.ticker, qty=r.position_size)
            else:
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
        from core.ui.backgrounds import GradientHeader
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        header = GradientHeader("Decision Logs", "Full history with reasoning and feature importances")
        layout.addWidget(header)

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
# PANEL: PORTFOLIO PROFILE
# ═════════════════════════════════════════════════════════════════════════════

class PortfolioPanel(QWidget):
    """Detailed portfolio view: holdings, trade history, watchlist, performance."""

    def __init__(self, broker, event_bus):
        super().__init__()
        self.broker = broker
        self.bus = event_bus
        self._build()
        self.bus.positions_changed.connect(self.refresh)
        self.bus.trade_executed.connect(lambda *_: QTimer.singleShot(2000, self.refresh))
        # Initial data load after a short delay (let UI finish building)
        QTimer.singleShot(1000, self.refresh)

    def _build(self):
        from core.ui.backgrounds import GradientHeader
        from core.widgets import StatCard, GradientDivider
        from core.ui.icons import StockyIcons

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        header = GradientHeader("Portfolio", "Holdings, trade history, and performance")
        layout.addWidget(header)

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self.card_equity = StatCard("Equity", "--", BRAND_PRIMARY)
        self.card_pnl_today = StatCard("Today", "--", BRAND_ACCENT)
        self.card_positions = StatCard("Positions", "--", BRAND_SECONDARY)
        self.card_orders = StatCard("Open Orders", "--", TEXT_SECONDARY)
        stats_row.addWidget(self.card_equity)
        stats_row.addWidget(self.card_pnl_today)
        stats_row.addWidget(self.card_positions)
        stats_row.addWidget(self.card_orders)
        layout.addLayout(stats_row)

        # Wire stat card clicks to detail popups
        self.card_equity.on_clicked = lambda: self._show_popup("equity")
        self.card_pnl_today.on_clicked = lambda: self._show_popup("pnl")

        # Main content — tabs within the tab
        inner_tabs = QTabWidget()
        inner_tabs.setStyleSheet(f"""
            QTabBar::tab {{ padding: 6px 14px; font-size: 11px; }}
        """)

        # ── Holdings Tab ──
        holdings_w = QWidget()
        hl = QVBoxLayout()
        self.holdings_table = QTableWidget()
        self.holdings_table.setColumnCount(10)
        self.holdings_table.setHorizontalHeaderLabels([
            "Symbol", "Side", "Qty", "Avg Cost", "Current", "Market Value", "P&L", "P&L %", "Orders", ""
        ])
        self.holdings_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.holdings_table.verticalHeader().setVisible(False)
        self.holdings_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.holdings_table.cellClicked.connect(self._on_holding_clicked)
        hl.addWidget(self.holdings_table)
        holdings_w.setLayout(hl)
        inner_tabs.addTab(holdings_w, StockyIcons.get_icon("wallet", 16, BRAND_PRIMARY), "Holdings")

        # ── Trade History Tab ──
        history_w = QWidget()
        htl = QVBoxLayout()
        hist_ctrl = QHBoxLayout()
        hist_ctrl.addWidget(QLabel("Show:"))
        self.hist_count = QComboBox()
        self.hist_count.addItems(["Last 20", "Last 50", "Last 100", "All"])
        hist_ctrl.addWidget(self.hist_count)
        refresh_btn = QPushButton("  Refresh")
        refresh_btn.setIcon(StockyIcons.get_icon("scan", 14, "white"))
        refresh_btn.setStyleSheet(f"font-size: 11px; padding: 5px;")
        refresh_btn.clicked.connect(self.refresh)
        hist_ctrl.addWidget(refresh_btn)
        hist_ctrl.addStretch()
        htl.addLayout(hist_ctrl)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels([
            "Symbol", "Side", "Qty", "Price", "Status", "Time", "Order ID"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        htl.addWidget(self.history_table)
        history_w.setLayout(htl)
        inner_tabs.addTab(history_w, StockyIcons.get_icon("log", 16, BRAND_PRIMARY), "Trade History")

        # ── Watchlist Tab ──
        watchlist_w = QWidget()
        wl = QVBoxLayout()
        wl_ctrl = QHBoxLayout()
        self.wl_select = QComboBox()
        self._refresh_wl_combo()
        wl_ctrl.addWidget(QLabel("Watchlist:"))
        wl_ctrl.addWidget(self.wl_select, 1)
        wl_ctrl.addStretch()

        wl_add_input = QLineEdit()
        wl_add_input.setPlaceholderText("Add ticker")
        wl_add_input.setFixedWidth(100)
        wl_ctrl.addWidget(wl_add_input)
        add_btn = QPushButton("Add")
        add_btn.setStyleSheet(f"font-size: 11px; padding: 5px; background-color: {BRAND_ACCENT};")
        add_btn.clicked.connect(lambda: self._add_to_watchlist(wl_add_input.text()))
        wl_ctrl.addWidget(add_btn)
        self._wl_add_input = wl_add_input
        wl.addLayout(wl_ctrl)

        self.wl_table = QTableWidget()
        self.wl_table.setColumnCount(3)
        self.wl_table.setHorizontalHeaderLabels(["Ticker", "Last Price", ""])
        self.wl_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.wl_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.wl_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.wl_table.verticalHeader().setVisible(False)
        wl.addWidget(self.wl_table)
        watchlist_w.setLayout(wl)
        inner_tabs.addTab(watchlist_w, StockyIcons.get_icon("chart_up", 16, BRAND_PRIMARY), "Watchlist")

        # ── Performance Chart Tab ──
        perf_w = QWidget()
        perf_l = QVBoxLayout()
        perf_ctrl = QHBoxLayout()
        perf_ctrl.addWidget(QLabel("Period:"))
        self.perf_period = QComboBox()
        self.perf_period.addItems(["1 Week", "1 Month", "3 Months", "1 Year"])
        self.perf_period.currentIndexChanged.connect(lambda _: self._refresh_perf_chart())
        perf_ctrl.addWidget(self.perf_period)
        perf_ctrl.addStretch()
        perf_l.addLayout(perf_ctrl)

        self.perf_figure = plt.Figure(dpi=100, facecolor=chart_colors()["fig_bg"])
        self.perf_canvas = FigureCanvas(self.perf_figure)
        self.perf_canvas.setMinimumHeight(200)
        perf_l.addWidget(self.perf_canvas)
        perf_w.setLayout(perf_l)
        inner_tabs.addTab(perf_w, StockyIcons.get_icon("dashboard", 16, BRAND_PRIMARY), "Performance")

        layout.addWidget(inner_tabs)
        self.setLayout(layout)

    def _show_popup(self, kind):
        if not self.broker:
            self.bus.log_entry.emit("Connect Alpaca API first — go to Settings", "warn")
            return
        from core.ui.detail_popup import show_equity_popup, show_pnl_popup
        if kind == "equity":
            show_equity_popup(self.broker, self)
        elif kind == "pnl":
            show_pnl_popup(self.broker, self)

    def _sell_holding(self, symbol, max_qty):
        """Sell shares of a specific holding."""
        if not self.broker:
            self.bus.log_entry.emit("Alpaca API not connected", "error")
            return
        from PyQt5.QtWidgets import QInputDialog
        qty, ok = QInputDialog.getInt(
            self, f"Sell {symbol}",
            f"How many shares of {symbol} to sell?\n(You own {max_qty})",
            value=max_qty, min=1, max=max_qty,
        )
        if not ok:
            return
        result = self.broker.close_position(symbol, qty=qty)
        if "error" in result:
            self.bus.log_entry.emit(f"Sell {symbol} x{qty} failed: {result['error']}", "error")
        else:
            self.bus.log_entry.emit(f"Sold {symbol} x{qty}", "trade")
            self.bus.positions_changed.emit()
            QTimer.singleShot(2000, self.refresh)

    def _on_holding_clicked(self, row, col):
        """Open detail popup for a clicked holding."""
        item = self.holdings_table.item(row, 0)
        if not item:
            return
        ticker = item.text()
        if not ticker or not self.broker:
            return

        # Get position data
        positions = self.broker.get_positions()
        pos = None
        if isinstance(positions, list):
            for p in positions:
                if p.get("symbol", "").upper() == ticker.upper():
                    pos = p
                    break

        if not pos:
            return

        qty = float(pos.get("qty", 0))
        avg = float(pos.get("avg_entry_price", 0))
        cur = float(pos.get("current_price", 0))
        pnl = float(pos.get("unrealized_pl", 0))
        pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100
        mv = qty * cur

        # Show a popup with stock detail + sell controls
        from core.ui.detail_popup import DetailPopup
        def fetch(period, tf):
            return self.broker.get_portfolio_history(period=period, timeframe=tf)

        def extract(data):
            # Use yfinance for stock-specific chart
            try:
                period_map = {"1D": "1d", "1W": "5d", "1M": "1mo", "3M": "3mo", "1A": "1y"}
                yf_period = period_map.get(period, "1mo")
                stock_data = yf.Ticker(ticker).history(period=yf_period)
                if not stock_data.empty:
                    ts = list(stock_data.index)
                    vals = list(stock_data["Close"].values)
                    return ts, vals, "$"
            except Exception:
                pass
            return [], [], "$"

        popup = DetailPopup(
            f"{ticker} — {qty:.0f} shares",
            fetch, extract,
            stats={
                "Avg Cost": f"${avg:.2f}",
                "Current": f"${cur:.2f}",
                "Market Value": f"${mv:,.2f}",
                "P&L": f"${pnl:+,.2f}",
                "P&L %": f"{pnl_pct:+.2f}%",
                "Qty": f"{qty:.0f} shares",
            },
            parent=self,
        )
        popup.exec_()

    def refresh(self):
        if not self.broker:
            return

        # Account stats
        acct = self.broker.get_account()
        if "error" not in acct:
            eq = float(acct.get("equity", 0))
            leq = float(acct.get("last_equity", eq))
            pnl = eq - leq
            pct = (pnl / leq * 100) if leq > 0 else 0
            self.card_equity.set_value(f"${eq:,.2f}")
            self.card_pnl_today.set_value(
                f"${pnl:+,.2f} ({pct:+.1f}%)",
                COLOR_PROFIT if pnl >= 0 else COLOR_LOSS,
            )

        # Holdings + pending orders
        positions = self.broker.get_positions()
        open_orders = self.broker.get_orders("open")
        order_counts = {}
        if isinstance(open_orders, list):
            for o in open_orders:
                sym = o.get("symbol", "")
                order_counts[sym] = order_counts.get(sym, 0) + 1

        if isinstance(positions, list):
            self.card_positions.set_value(str(len(positions)))
            self.holdings_table.setRowCount(len(positions))
            for i, p in enumerate(positions):
                sym = p.get("symbol", "")
                qty = float(p.get("qty", 0))
                avg = float(p.get("avg_entry_price", 0))
                cur = float(p.get("current_price", 0))
                mv = qty * cur
                pnl = float(p.get("unrealized_pl", 0))
                pnl_pct = float(p.get("unrealized_plpc", 0)) * 100
                pending = order_counts.get(sym, 0)

                vals = [
                    sym, p.get("side", ""),
                    f"{qty:.0f}", f"${avg:.2f}", f"${cur:.2f}",
                    f"${mv:,.2f}", f"${pnl:+,.2f}", f"{pnl_pct:+.2f}%",
                    f"{pending} pending" if pending > 0 else "—",
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignCenter)
                    if j >= 6 and j <= 7:
                        item.setForeground(QColor(COLOR_PROFIT if pnl >= 0 else COLOR_LOSS))
                    if j == 8 and pending > 0:
                        item.setForeground(QColor(COLOR_HOLD))
                    self.holdings_table.setItem(i, j, item)

                # Sell button per stock
                sell_btn = QPushButton("Sell")
                sell_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 10px; padding: 3px 8px;")
                sell_btn.clicked.connect(lambda _, s=sym, q=int(qty): self._sell_holding(s, q))
                self.holdings_table.setCellWidget(i, 9, sell_btn)
        else:
            self.card_positions.set_value("0")
            self.holdings_table.setRowCount(0)

        # Open orders
        orders = self.broker.get_orders("open")
        if isinstance(orders, list):
            self.card_orders.set_value(str(len(orders)))

        # Trade history with running totals
        limit_map = {"Last 20": 20, "Last 50": 50, "Last 100": 100, "All": 500}
        limit = limit_map.get(self.hist_count.currentText(), 20)
        closed = self.broker.get_orders("closed")
        if isinstance(closed, list):
            closed = closed[:limit]
            # Add cost column to table if not already there
            if self.history_table.columnCount() < 8:
                self.history_table.setColumnCount(8)
                self.history_table.setHorizontalHeaderLabels([
                    "Symbol", "Side", "Qty", "Price", "Total Cost", "Status", "Time", "Order ID"
                ])
            self.history_table.setRowCount(len(closed))
            running_total = 0
            for i, o in enumerate(closed):
                qty = float(o.get("filled_qty", 0))
                price = float(o.get("filled_avg_price", 0))
                side = o.get("side", "")
                cost = qty * price
                # Buys are negative cash flow, sells are positive
                if side == "buy":
                    running_total -= cost
                else:
                    running_total += cost

                vals = [
                    o.get("symbol", ""), side,
                    f"{qty:.0f}", f"${price:.2f}",
                    f"${cost:,.2f}", o.get("status", ""),
                    o.get("filled_at", "")[:16] if o.get("filled_at") else "",
                    o.get("id", "")[:10],
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(str(v))
                    item.setTextAlignment(Qt.AlignCenter)
                    if j == 1:
                        item.setForeground(QColor(COLOR_BUY if v == "buy" else COLOR_SELL))
                    if j == 4:
                        item.setForeground(QColor(COLOR_SELL if side == "buy" else COLOR_BUY))
                    self.history_table.setItem(i, j, item)
        else:
            self.history_table.setRowCount(0)

        self._refresh_perf_chart()
        self._refresh_watchlist()

    def _refresh_perf_chart(self):
        if not self.broker:
            return
        period_map = {"1 Week": ("1W", "1H"), "1 Month": ("1M", "1D"),
                      "3 Months": ("3M", "1D"), "1 Year": ("1A", "1D")}
        p, tf = period_map.get(self.perf_period.currentText(), ("1W", "1H"))
        hist = self.broker.get_portfolio_history(period=p, timeframe=tf)

        cc = chart_colors()
        if "error" in hist or not hist.get("equity"):
            # Show empty state message
            self.perf_figure.clear()
            self.perf_figure.set_facecolor(cc["fig_bg"])
            ax = self.perf_figure.add_subplot(111)
            ax.set_facecolor(cc["ax_bg"])
            ax.text(0.5, 0.5, "No performance data available yet.\nStart trading to see your equity curve.",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color=cc["muted"], style='italic')
            ax.set_xticks([])
            ax.set_yticks([])
            self.perf_figure.tight_layout()
            self.perf_canvas.draw()
            return

        self.perf_figure.clear()
        cc = chart_colors(); self.perf_figure.set_facecolor(cc["fig_bg"])
        ax = self.perf_figure.add_subplot(111)
        ax.set_facecolor(cc["ax_bg"])
        eq = hist["equity"]
        ts = [datetime.fromtimestamp(t) for t in hist["timestamp"]]
        trending_up = eq[-1] >= eq[0] if eq else True
        color = COLOR_BUY if trending_up else COLOR_SELL
        ax.plot(ts, eq, color=color, linewidth=1.5)
        ax.fill_between(ts, eq, alpha=0.08, color=color)
        ax.set_title(f"Portfolio — {self.perf_period.currentText()}", color=cc["text"], fontsize=10)
        ax.tick_params(colors=cc["muted"], labelsize=7)
        ax.grid(True, alpha=0.15, color=cc["grid"])
        self.perf_figure.tight_layout()
        self.perf_canvas.draw()

    def _refresh_wl_combo(self):
        from core.discovery import get_watchlists
        self.wl_select.clear()
        for name in get_watchlists():
            self.wl_select.addItem(name)

    def _refresh_watchlist(self):
        from core.discovery import get_watchlists
        name = self.wl_select.currentText()
        wls = get_watchlists()
        tickers = wls.get(name, [])

        # Update table columns
        self.wl_table.setColumnCount(4)
        self.wl_table.setHorizontalHeaderLabels(["Ticker", "Price", "Change", ""])
        self.wl_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 4):
            self.wl_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)

        self.wl_table.setRowCount(len(tickers))
        for i, t in enumerate(tickers):
            self.wl_table.setItem(i, 0, QTableWidgetItem(t))
            try:
                hist = yf.Ticker(t).history(period="2d")
                if not hist.empty and len(hist) >= 1:
                    price = hist["Close"].iloc[-1]
                    self.wl_table.setItem(i, 1, QTableWidgetItem(f"${price:.2f}"))
                    if len(hist) >= 2:
                        prev = hist["Close"].iloc[-2]
                        chg = price - prev
                        pct = (chg / prev * 100) if prev != 0 else 0
                        chg_item = QTableWidgetItem(f"{'+'if chg>=0 else ''}{chg:.2f} ({pct:+.1f}%)")
                        chg_item.setForeground(QColor(COLOR_BUY if chg >= 0 else COLOR_SELL))
                        self.wl_table.setItem(i, 2, chg_item)
                    else:
                        self.wl_table.setItem(i, 2, QTableWidgetItem("--"))
                else:
                    self.wl_table.setItem(i, 1, QTableWidgetItem("--"))
                    self.wl_table.setItem(i, 2, QTableWidgetItem("--"))
            except Exception:
                self.wl_table.setItem(i, 1, QTableWidgetItem("--"))
                self.wl_table.setItem(i, 2, QTableWidgetItem("--"))

            rm_btn = QPushButton("Remove")
            rm_btn.setStyleSheet(f"font-size: 9px; padding: 2px; background-color: {BG_INPUT};")
            rm_btn.clicked.connect(lambda _, tk=t: self._remove_from_watchlist(tk))
            self.wl_table.setCellWidget(i, 3, rm_btn)

    def _add_to_watchlist(self, ticker):
        from core.discovery import get_watchlists, save_watchlist
        ticker = ticker.strip().upper()
        if not ticker:
            return
        name = self.wl_select.currentText() or "My Watchlist"
        wls = get_watchlists()
        tickers = wls.get(name, [])
        if ticker not in tickers:
            tickers.append(ticker)
            save_watchlist(name, tickers)
            self._refresh_watchlist()
            self.bus.log_entry.emit(f"Added {ticker} to {name}", "info")
        self._wl_add_input.clear()

    def _remove_from_watchlist(self, ticker):
        from core.discovery import get_watchlists, save_watchlist
        name = self.wl_select.currentText()
        wls = get_watchlists()
        tickers = wls.get(name, [])
        if ticker in tickers:
            tickers.remove(ticker)
            save_watchlist(name, tickers)
            self._refresh_watchlist()


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
        from core.ui.backgrounds import GradientHeader
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        header = GradientHeader("Settings", "API keys, profiles, addons, and models")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout()
        settings = load_settings()

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

        # Trading Aggressivity
        aggr_box = QGroupBox("Trading Aggressivity")
        ag_layout = QVBoxLayout()
        ag_row = QHBoxLayout()
        ag_row.addWidget(QLabel("Style:"))
        self.aggr_combo = QComboBox()
        from core.intelligent_trader import AGGRESSIVITY_PROFILES
        for name, prof in AGGRESSIVITY_PROFILES.items():
            self.aggr_combo.addItem(f"{name} — {prof['description'][:50]}", name)
        # Load saved
        saved_aggr = settings.get("aggressivity", "Default")
        idx_map = {n: i for i, n in enumerate(AGGRESSIVITY_PROFILES)}
        self.aggr_combo.setCurrentIndex(idx_map.get(saved_aggr, 1))
        self.aggr_combo.currentIndexChanged.connect(self._change_aggressivity)
        ag_row.addWidget(self.aggr_combo, 1)
        ag_layout.addLayout(ag_row)

        self.aggr_desc = QLabel("")
        self.aggr_desc.setWordWrap(True)
        self.aggr_desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        ag_layout.addWidget(self.aggr_desc)
        self._update_aggr_desc()

        aggr_box.setLayout(ag_layout)
        inner_layout.addWidget(aggr_box)

        # Appearance
        appear_box = QGroupBox("Appearance")
        al2 = QVBoxLayout()

        # Theme
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Auto (System)", "Dark", "Light"])
        theme_val = settings.get("theme", "auto")
        idx_map = {"auto": 0, "dark": 1, "light": 2}
        self.theme_combo.setCurrentIndex(idx_map.get(theme_val, 0))
        self.theme_combo.currentIndexChanged.connect(self._change_theme)
        theme_row.addWidget(self.theme_combo, 1)
        al2.addLayout(theme_row)

        # Zoom slider
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("UI Scale:"))
        self.zoom_slider = QSpinBox()
        self.zoom_slider.setRange(70, 200)
        self.zoom_slider.setSuffix("%")
        self.zoom_slider.setSingleStep(5)
        current_zoom = int(settings.get("ui_zoom", 0.95) * 100)
        self.zoom_slider.setValue(current_zoom)
        self.zoom_slider.valueChanged.connect(self._change_zoom)
        zoom_row.addWidget(self.zoom_slider)
        self.zoom_label = QLabel(f"Current: {current_zoom}%")
        self.zoom_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        zoom_row.addWidget(self.zoom_label)
        al2.addLayout(zoom_row)

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
        self.model_table.setMinimumHeight(120)
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

    def _change_zoom(self, value):
        scale = value / 100.0
        main_window = self.window()
        if main_window and hasattr(main_window, '_scale'):
            main_window._scale = scale
            main_window._apply_scale()
            main_window._save_zoom()
        self.zoom_label.setText(f"Current: {value}%")

    def _change_aggressivity(self, index):
        name = self.aggr_combo.currentData()
        settings = load_settings()
        settings["aggressivity"] = name
        save_settings(settings)
        self._update_aggr_desc()
        self.bus.log_entry.emit(f"Trading aggressivity set to: {name}", "system")

    def _update_aggr_desc(self):
        from core.intelligent_trader import get_aggressivity
        name = self.aggr_combo.currentData() or "Default"
        p = get_aggressivity(name)
        self.aggr_desc.setText(
            f"Min confidence: {p['min_confidence']:.0%}  |  "
            f"Position size: {p['size_multiplier']:.1f}x  |  "
            f"Max trades/day: {p['max_trades_per_day']}  |  "
            f"Stop: {p['atr_stop_mult']:.1f}x ATR  |  "
            f"Target: {p['atr_profit_mult']:.1f}x ATR"
        )

    def _change_theme(self, index):
        theme_map = {0: "auto", 1: "dark", 2: "light"}
        theme_name = theme_map.get(index, "auto")
        settings = load_settings()
        settings["theme"] = theme_name
        save_settings(settings)
        # Refresh the theme provider so all custom widgets update
        from core.ui.theme import theme as theme_provider
        theme_provider.refresh()
        # Apply immediately to the main window
        main_window = self.window()
        if main_window:
            main_window.setStyleSheet(get_stylesheet(theme_name))
            if hasattr(main_window, '_theme'):
                main_window._theme = theme_name
            if hasattr(main_window, '_apply_scale'):
                main_window._apply_scale()
        self.bus.log_entry.emit(f"Theme changed to: {theme_name}", "system")

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
        from core.ui.backgrounds import GradientHeader
        from core.widgets import DetailedProgressBar

        layout = QVBoxLayout()
        layout.setSpacing(6)

        header = GradientHeader("Tax Reports", "IRS Form 8949 — Capital Gains & Losses")
        layout.addWidget(header)

        desc = QLabel(
            "Generate a tax report from your Alpaca trade history. "
            "This produces Form 8949 data (Sales and Dispositions of Capital Assets) "
            "showing each closed trade with proceeds, cost basis, and gain/loss.\n\n"
            "The CSV export can be imported into TurboTax, H&R Block, or handed to your accountant."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(desc)

        row = QHBoxLayout()
        row.addWidget(QLabel("Tax Year:"))
        self.year_spin = QSpinBox()
        self.year_spin.setRange(2020, 2030)
        self.year_spin.setValue(datetime.now().year)
        row.addWidget(self.year_spin)

        gen_btn = QPushButton("  Generate Report")
        gen_btn.setStyleSheet(f"background-color: {BRAND_ACCENT};")
        gen_btn.clicked.connect(self._generate)
        row.addWidget(gen_btn)
        row.addStretch()
        layout.addLayout(row)

        self.progress = DetailedProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setFont(QFont(FONT_MONO, 10))
        layout.addWidget(self.report_view)

        self.setLayout(layout)

    def _generate(self):
        if not self.broker:
            self.progress.setVisible(True)
            self.progress.set_progress(100, "Alpaca API not configured",
                                       "Go to Settings and enter your API keys first")
            return

        from core.tax_report import generate_form_8949

        self.progress.setVisible(True)
        self.progress.reset()
        self.progress.set_progress(20, "Connecting to Alpaca...", "Fetching closed orders")
        self.progress.add_log("Requesting trade history from Alpaca API")
        QApplication.processEvents()

        year = self.year_spin.value()
        result = generate_form_8949(self.broker, year)

        self.progress.set_progress(80, "Formatting report...", "")
        self.progress.add_log(f"Found {result['summary']['total_trades']} trades for {year}")
        QApplication.processEvents()

        self.report_view.setPlainText(result["text"])

        if result["summary"]["total_trades"] == 0:
            self.progress.set_progress(100, "No trades found",
                                       f"No closed trades in {year}. Trade first, then generate.")
            self.progress.add_log(f"No closed positions found for tax year {year}")
        elif result["csv_path"]:
            self.progress.set_progress(100, f"Report saved!",
                                       result["csv_path"])
            self.progress.add_log(f"CSV saved: {result['csv_path']}")
            self.bus.log_entry.emit(f"Tax report generated: {result['csv_path']}", "system")
        else:
            self.progress.set_progress(100, "Report generated", "")


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
        from core.ui.backgrounds import GradientHeader
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 4)

        header = GradientHeader("System Testing", "Diagnostics, addon checks, and unit test runner")
        layout.addWidget(header)

        # Quick diagnostics
        diag_box = QGroupBox("System Diagnostics")
        dg = QVBoxLayout()

        self.diag_output = QTextEdit()
        self.diag_output.setReadOnly(True)
        self.diag_output.setFont(QFont(FONT_MONO, 10))
        self.diag_output.setMinimumHeight(250)
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

class _NotificationBar(QWidget):
    """
    Custom-painted notification bar with:
    - Slow-moving gradient background
    - Market open/closed indicator
    - Icon that pulses briefly on new messages
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(32)
        self.setMaximumHeight(34)
        self._phase = 0.0
        self._msg = f"{APP_NAME} v{APP_VERSION} — Ready"
        self._time = ""
        self._icon_name = "check"
        self._icon_color = BRAND_PRIMARY
        self._text_color = QColor(TEXT_SECONDARY)
        self._icon_pulse = 0.0   # 0-1, decays over time (glow)
        self._icon_bounce = 0.0  # 0-1, decays (vertical bounce)
        self._text_slide = 0.0   # 0-1, decays (text slides in from right)
        self._market_open = False

        # Notification history for bell
        self._notif_history = []  # list of (time, msg, level)
        self._unread_count = 0
        self._bell_pulse = 0.0

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

        # Market status check
        self._check_market()
        self._market_timer = QTimer(self)
        self._market_timer.timeout.connect(self._check_market)
        self._market_timer.start(60000)  # Check every minute

    def _check_market(self):
        try:
            est = pytz.timezone("US/Eastern")
            now = datetime.now(est)
            self._market_open = (
                now.weekday() < 5
                and now.hour >= 9 and (now.hour > 9 or now.minute >= 30)
                and now.hour < 16
            )
        except Exception:
            self._market_open = False

    def _tick(self):
        self._phase += 0.008
        if self._icon_pulse > 0:
            self._icon_pulse = max(0, self._icon_pulse - 0.025)
        if self._icon_bounce > 0:
            self._icon_bounce = max(0, self._icon_bounce - 0.04)
        if self._text_slide > 0:
            self._text_slide = max(0, self._text_slide - 0.05)
        if self._bell_pulse > 0:
            self._bell_pulse = max(0, self._bell_pulse - 0.02)
        self.update()

    def show_message(self, msg, level="info"):
        from core.ui.icons import StockyIcons
        icon_map = {
            "info":   ("check",    BRAND_PRIMARY, TEXT_SECONDARY),
            "trade":  ("bolt",     BRAND_ACCENT,  BRAND_ACCENT),
            "warn":   ("warning",  COLOR_HOLD,    COLOR_HOLD),
            "error":  ("x_mark",   COLOR_SELL,    COLOR_SELL),
            "system": ("settings", TEXT_MUTED,    TEXT_MUTED),
        }
        self._icon_name, self._icon_color, text_hex = icon_map.get(level, ("check", TEXT_MUTED, TEXT_MUTED))
        self._text_color = QColor(text_hex)
        self._msg = msg
        self._time = datetime.now().strftime("%H:%M:%S")
        self._icon_pulse = 1.0
        self._icon_bounce = 1.0
        self._text_slide = 1.0

        # Track in history
        self._notif_history.append((self._time, msg, level))
        self._notif_history = self._notif_history[-50:]  # Keep last 50
        self._unread_count += 1
        self._bell_pulse = 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Animated gradient background — slow dark-to-darker drift
        import math
        shift = (math.sin(self._phase) + 1) / 2
        grad = QLinearGradient(0, 0, w, 0)
        c1 = QColor(22, 25, 35)
        c2 = QColor(28, 32, 45)
        c3 = QColor(18, 20, 30)
        grad.setColorAt(0, c1)
        grad.setColorAt(0.3 + shift * 0.2, c2)
        grad.setColorAt(0.7 - shift * 0.2, c1)
        grad.setColorAt(1, c3)
        painter.fillRect(0, 0, w, h, grad)

        # Top border — subtle thin line
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawLine(0, 0, w, 0)

        x = 14

        # Market status dot
        dot_color = QColor(BRAND_ACCENT) if self._market_open else QColor(COLOR_SELL)
        dot_color.setAlphaF(0.7 + math.sin(self._phase * 3) * 0.15)
        painter.setBrush(dot_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(x), int(h / 2 - 4), 8, 8)
        x += 14

        # Market label
        painter.setPen(QColor(TEXT_MUTED))
        painter.setFont(QFont(FONT_MONO, 8))
        market_text = "OPEN" if self._market_open else "CLOSED"
        painter.drawText(int(x), 0, 50, h, Qt.AlignVCenter, market_text)
        x += 55

        # Separator
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawLine(int(x), 6, int(x), h - 6)
        x += 12

        # Icon with bounce + pulse glow
        from core.ui.icons import StockyIcons
        import math as _m
        icon_px = StockyIcons.get(self._icon_name, 16, self._icon_color)

        # Bounce: icon pops up then settles (elastic ease-out)
        bounce_offset = 0
        if self._icon_bounce > 0:
            # Elastic bounce: goes up, overshoots, settles
            t = 1.0 - self._icon_bounce
            bounce_offset = int(_m.sin(t * _m.pi * 2.5) * (1.0 - t) * 8)

        # Glow behind icon
        if self._icon_pulse > 0:
            glow_c = QColor(self._icon_color)
            glow_c.setAlphaF(self._icon_pulse * 0.35)
            painter.setBrush(glow_c)
            painter.setPen(Qt.NoPen)
            glow_r = 10 + self._icon_pulse * 4
            painter.drawEllipse(int(x + 8 - glow_r), int(h / 2 - glow_r + bounce_offset), int(glow_r * 2), int(glow_r * 2))

        painter.drawPixmap(int(x), int(h / 2 - 8 + bounce_offset), icon_px)
        x += 22

        # Message text — slides in from right
        text_offset = int(self._text_slide * 40)  # Starts 40px to the right, slides to 0
        text_alpha = 1.0 - self._text_slide * 0.6  # Fades in as it slides

        tc = QColor(self._text_color)
        tc.setAlphaF(max(0.3, text_alpha))
        painter.setPen(tc)
        painter.setFont(QFont(FONT_FAMILY, 10))
        painter.drawText(int(x + text_offset), 0, w - x - 70, h, Qt.AlignVCenter, self._msg)

        # Timestamp
        if self._time:
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont(FONT_MONO, 8))
            painter.drawText(w - 90, 0, 50, h, Qt.AlignVCenter | Qt.AlignRight, self._time)

        # Bell icon with unread badge
        bell_x = w - 32
        bell_px = StockyIcons.get("bell", 14, BRAND_PRIMARY if self._unread_count > 0 else TEXT_MUTED)

        # Bell pulse glow
        if self._bell_pulse > 0:
            glow = QColor(BRAND_PRIMARY)
            glow.setAlphaF(self._bell_pulse * 0.3)
            painter.setBrush(glow)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(bell_x - 1), int(h/2 - 9), 18, 18)

        painter.drawPixmap(int(bell_x), int(h/2 - 7), bell_px)

        # Unread count badge
        if self._unread_count > 0:
            painter.setBrush(QColor(COLOR_SELL))
            painter.setPen(Qt.NoPen)
            badge_r = 7
            painter.drawEllipse(int(bell_x + 8), int(h/2 - 10), badge_r * 2, badge_r * 2)
            painter.setPen(QColor("white"))
            painter.setFont(QFont(FONT_FAMILY, 7, QFont.Bold))
            count_text = str(self._unread_count) if self._unread_count < 100 else "99+"
            painter.drawText(int(bell_x + 8), int(h/2 - 10), badge_r * 2, badge_r * 2, Qt.AlignCenter, count_text)

        painter.end()

    def mousePressEvent(self, event):
        """Click on bell area opens notification overlay."""
        bell_x = self.width() - 32
        if event.x() >= bell_x - 5:
            self._show_notification_overlay()
        super().mousePressEvent(event)

    def _show_notification_overlay(self):
        """Show overlay with notification history, clear unread count."""
        self._unread_count = 0
        self.update()

        if not self._notif_history:
            return

        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit
        dlg = QDialog(self.window())
        dlg.setWindowTitle("Notifications")
        dlg.setMinimumSize(450, 350)
        bg = theme.color("bg_base") if hasattr(theme, 'color') else BG_DARKEST
        dlg.setStyleSheet(f"QDialog {{ background-color: {bg}; }}")

        lay = QVBoxLayout()
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFont(QFont(FONT_MONO, 10))

        from core.branding import log_html as _lh
        for ts, msg, level in reversed(self._notif_history):
            colors = {"info": BRAND_PRIMARY, "trade": BRAND_ACCENT, "warn": COLOR_HOLD, "error": COLOR_SELL, "system": TEXT_MUTED}
            c = colors.get(level, TEXT_SECONDARY)
            txt.append(f'<span style="color:{TEXT_MUTED}">{ts}</span> <span style="color:{c}">{msg}</span>')

        lay.addWidget(txt)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(lambda: (self._notif_history.clear(), dlg.accept()))
        lay.addWidget(clear_btn)

        dlg.setLayout(lay)
        dlg.exec_()


class StockySuite(QMainWindow):
    """Main application window — unified trading dashboard."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        # Size relative to screen (85% width, 85% height)
        try:
            screen = QApplication.primaryScreen().availableGeometry()
            w = int(screen.width() * 0.85)
            h = int(screen.height() * 0.85)
            x = (screen.width() - w) // 2
            y = (screen.height() - h) // 2
            self.setGeometry(x, y, w, h)
        except Exception:
            self.setGeometry(50, 50, 1200, 800)

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

        # Notification bar — animated gradient background + market status
        self._notif_bar = _NotificationBar(self)
        sb = self.statusBar()
        sb.addPermanentWidget(self._notif_bar, 1)
        sb.setStyleSheet("QStatusBar { border: none; padding: 0; margin: 0; } QStatusBar::item { border: none; }")
        sb.setSizeGripEnabled(False)

        # Refresh dashboard on startup
        if hasattr(self, 'dashboard') and hasattr(self.dashboard, 'refresh'):
            QTimer.singleShot(500, self.dashboard.refresh)

        # Activity feed forwarding
        self.event_bus.log_entry.connect(self._on_log)

        # UI scaling — load saved zoom or auto-detect from screen resolution
        settings = load_settings()
        if "ui_zoom" in settings:
            self._scale = settings["ui_zoom"]
        else:
            self._scale = self._detect_ideal_scale()
        self._apply_scale()

        # Zoom is now controlled via Settings tab slider (removed broken Ctrl+/- shortcuts)

        # System tray agent — minimize to tray on close, toast notifications
        from core.tray_agent import TrayAgent
        self._tray = TrayAgent(self)
        self._tray.setup()

        # Send toast notifications on trade events
        self.event_bus.trade_executed.connect(
            lambda t, s, q, o: self._tray.send_notification(
                f"Trade Executed: {s.upper()} {t}",
                f"{s.upper()} {q} shares of {t}\nOrder: {o}",
                "trade",
            )
        )

        log_event("startup", f"{APP_NAME} v{APP_VERSION} launched")

    def closeEvent(self, event):
        """Minimize to tray instead of quitting."""
        if self._tray and self._tray.tray and self._tray.tray.isVisible():
            event.ignore()
            self.hide()
            self._tray.send_notification(
                "Stocky Suite",
                "Running in background. Double-click tray icon to restore.",
                "info",
            )
        else:
            event.accept()

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
        from core.ui.icons import StockyIcons

        # Tab definitions: (name, icon_key, factory)
        panels = [
            ("Dashboard",   "dashboard", lambda: DashboardPanel(self.broker, self.event_bus)),
            ("Scanner",     "scan",      lambda: ScannerPanel(self.broker, self.risk_manager, self.event_bus)),
            ("Portfolio",   "wallet",    lambda: PortfolioPanel(self.broker, self.event_bus)),
            ("Day Trade",   "bolt",      lambda: DayTradePanel(self.broker, self.risk_manager, self.event_bus)),
            ("Long Trade",  "chart_up",  lambda: LongTradePanel(self.event_bus)),
            ("Logs",        "log",       lambda: LogsPanel(self.event_bus)),
            ("Tax Reports", "tax",       lambda: TaxPanel(self.broker, self.event_bus)),
            ("Testing",     "test",      lambda: TestingPanel(self.broker, self.event_bus)),
            ("Settings",    "settings",  lambda: SettingsPanel(self.event_bus)),
        ]

        for tab_name, icon_key, factory in panels:
            try:
                panel = factory()
                icon = StockyIcons.get_icon(icon_key, 22, BRAND_PRIMARY)
                self.tabs.addTab(panel, icon, tab_name)
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
        """Update the notification bar."""
        self._notif_bar.show_message(msg, level)

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
        # Only show zoom notification if scale is not default
        if hasattr(self, '_notif_bar') and abs(self._scale - 1.0) > 0.01:
            self._notif_bar.show_message(f"UI Scale: {self._scale:.0%}", "system")

    @staticmethod
    def _detect_ideal_scale():
        """Auto-detect ideal UI scale. Qt high DPI handles the heavy lifting now,
        so we just add a small bump for comfort."""
        try:
            screen = QApplication.primaryScreen()
            ratio = screen.devicePixelRatio()
            # Qt high DPI scaling is active — ratio > 1 means it's already scaling
            # Just add a slight comfort bump
            if ratio >= 1.5:
                return 0.95  # Qt already scaled 1.5x — slight reduction for chart room
            elif ratio > 1.0:
                return 1.0
            else:
                w = screen.geometry().width()
                if w >= 2560:
                    return 1.3
                elif w >= 1920:
                    return 1.15
                else:
                    return 1.05
        except Exception:
            return 1.15

    def _zoom(self, delta):
        self._scale = max(0.7, min(2.0, round(self._scale + delta, 2)))
        self._apply_scale()
        self._save_zoom()

    def _reset_zoom(self):
        self._scale = 1.30
        self._apply_scale()
        self._save_zoom()

    def _save_zoom(self):
        settings = load_settings()
        settings["ui_zoom"] = self._scale
        save_settings(settings)


# ═════════════════════════════════════════════════════════════════════════════
# ABOUT DIALOG (boot screen + setup wizard in core/ui/)
# ═════════════════════════════════════════════════════════════════════════════

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setFixedSize(420, 350)
        self.setStyleSheet(SUITE_STYLESHEET)

        layout = QVBoxLayout()
        layout.setSpacing(6)

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


def boot_app():
    """Boot sequence with premium animated loading screen."""
    # Enable Qt high DPI scaling BEFORE QApplication is created
    # This is critical for 2560x1600 / 144 DPI screens
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    if os.path.exists(ICON_FILE):
        app.setWindowIcon(QIcon(ICON_FILE))

    # Premium boot screen with animated orbs and gradient bar
    from core.ui.boot_screen import BootScreen
    boot = BootScreen()
    boot.show()
    app.processEvents()

    def step(pct, msg, detail=""):
        boot.step(pct, msg, detail)
        time.sleep(0.4)

    step(5,  "Loading core modules...",       "features · model · risk · broker · scanner")
    step(15, "Checking dependencies...",      "lightgbm · ta · transformers · torch")

    step(25, "Discovering addons...",         "Scanning StockyApps/addons/")
    from addons import discover_addons, get_all_addons
    discover_addons()
    addons = get_all_addons()
    active = [a for a in addons if a.available and a.enabled]
    step(40, f"Loaded {len(active)} addons",  " · ".join(a.name for a in active[:5]))

    step(50, "Initializing risk engine...",   "ATR sizing · 2% risk · 5% drawdown limit")
    step(60, "Connecting to broker...",       "Alpaca paper trading API")

    from core.profiles import get_active_profile_name
    step(70, f"Profile: {get_active_profile_name()}", "Hardware preset loaded")

    step(80, "Building interface...",         "8 panels · event bus · signal routing")
    suite = StockySuite()

    step(90, "Loading log history...",        "Decision logs · trade history")

    # ── Menu Bar ──
    from core.ui.icons import StockyIcons
    from core.ui.setup_wizard import needs_setup, SetupWizard

    view_menu = suite.menuBar().addMenu("View")
    for i, (name, icon_key) in enumerate([
        ("Dashboard", "dashboard"), ("Scanner", "scan"), ("Portfolio", "wallet"),
        ("Day Trade", "bolt"), ("Long Trade", "chart_up"), ("Logs", "log"),
        ("Tax Reports", "tax"), ("Testing", "test"), ("Settings", "settings"),
    ]):
        action = QAction(StockyIcons.get_icon(icon_key, 16, BRAND_PRIMARY), name, suite)
        idx = i
        action.triggered.connect(lambda _, x=idx: suite.tabs.setCurrentIndex(x))
        action.setShortcut(f"Ctrl+{i+1}")
        view_menu.addAction(action)

    tools_menu = suite.menuBar().addMenu("Tools")
    rerun = QAction("Run Setup Wizard...", suite)
    rerun.triggered.connect(lambda: SetupWizard(suite).exec_())
    tools_menu.addAction(rerun)

    help_menu = suite.menuBar().addMenu("Help")
    about_action = QAction(f"About {APP_NAME}", suite)
    about_action.triggered.connect(lambda: AboutDialog(suite).exec_())
    help_menu.addAction(about_action)

    # Setup wizard on first boot
    step(95, "Checking first-run setup...", "")
    if needs_setup():
        step(100, "Welcome!", "Launching setup wizard...")
        time.sleep(0.3)
        boot.hide()
        wizard = SetupWizard()
        wizard.setStyleSheet(get_stylesheet("auto"))
        wizard.exec_()
        suite.broker = suite._init_broker()
        for attr in ("dashboard", "scanner", "portfolio", "day_trade", "tax_reports", "testing"):
            p = getattr(suite, attr, None)
            if p and hasattr(p, "broker"):
                p.broker = suite.broker
    else:
        step(100, "Ready.", f"{APP_NAME} v{APP_VERSION}")
        time.sleep(0.5)

    # Fade out boot screen, show main window
    boot.finish()
    suite.show()
    suite.raise_()
    suite.activateWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    boot_app()
