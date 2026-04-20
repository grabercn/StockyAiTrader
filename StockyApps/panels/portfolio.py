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

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f: return json.load(f)
    except: return {}
def save_settings(s):
    with open(SETTINGS_FILE, "w") as f: json.dump(s, f, indent=4)

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

        # ── Open Orders Tab ──
        orders_w = QWidget()
        ol = QVBoxLayout()
        ol.setContentsMargins(0, 4, 0, 0)
        self.open_orders_table = QTableWidget()
        self.open_orders_table.setColumnCount(8)
        self.open_orders_table.setHorizontalHeaderLabels(
            ["Symbol", "Side", "Qty", "Type", "Price", "Status", "Submitted", ""])
        for c in range(7):
            self.open_orders_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
        self.open_orders_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.open_orders_table.verticalHeader().setVisible(False)
        ol.addWidget(self.open_orders_table)
        orders_w.setLayout(ol)
        inner_tabs.addTab(orders_w, StockyIcons.get_icon("refresh", 16, BRAND_PRIMARY), "Open Orders")

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

    def _cancel_order(self, order_id):
        if not self.broker:
            return
        result = self.broker.cancel_order(order_id)
        if "error" in result:
            self.bus.log_entry.emit(f"Cancel failed: {result['error']}", "error")
        else:
            self.bus.log_entry.emit(f"Order cancelled: {order_id[:12]}", "trade")
            QTimer.singleShot(1000, self.refresh)

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
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
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

        # Open orders — populate table with cancel buttons
        orders = self.broker.get_orders("open")
        if isinstance(orders, list):
            self.card_orders.set_value(str(len(orders)))
            self.open_orders_table.setRowCount(len(orders))
            for i, o in enumerate(orders):
                price_str = "market"
                if o.get("type") == "limit" and o.get("limit_price"):
                    price_str = f"${float(o['limit_price']):.2f}"
                elif o.get("type") == "stop" and o.get("stop_price"):
                    price_str = f"${float(o['stop_price']):.2f}"
                vals = [
                    o.get("symbol", ""), o.get("side", ""),
                    o.get("qty", "0"), o.get("type", ""),
                    price_str, o.get("status", ""),
                    (o.get("submitted_at", "") or "")[:16],
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(str(v))
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    if j == 1:
                        item.setForeground(QColor(COLOR_BUY if v == "buy" else COLOR_SELL))
                    self.open_orders_table.setItem(i, j, item)
                # Cancel button
                oid = o.get("id", "")
                cancel_btn = QPushButton("Cancel")
                cancel_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 9px; padding: 2px 6px;")
                cancel_btn.clicked.connect(lambda _, _id=oid: self._cancel_order(_id))
                self.open_orders_table.setCellWidget(i, 7, cancel_btn)
        else:
            self.card_orders.set_value("0")
            self.open_orders_table.setRowCount(0)

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
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
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

        from core.ui.chart_tooltip import ChartTooltip
        self._perf_tooltip = ChartTooltip(self.perf_canvas, ax, ts, eq)

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

