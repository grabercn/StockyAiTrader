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

# PANEL: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

class DashboardPanel(QWidget):
    """Portfolio overview: account stats, positions, equity chart."""

    _data_ready = pyqtSignal(object, object, object, object)  # acct, positions, orders, hist

    def __init__(self, broker, event_bus):
        super().__init__()
        self.broker = broker
        self.bus = event_bus
        self._data_ready.connect(self._apply_refresh)
        self._build()
        self.bus.positions_changed.connect(self.refresh)
        self.bus.trade_executed.connect(lambda *_: self.refresh())
        self.bus.trade_executed.connect(lambda *_: QTimer.singleShot(3000, self.refresh))

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
        self.pos_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        pl.addWidget(self.pos_table)
        pos_widget.setLayout(pl)
        bl.addWidget(pos_widget, 3)

        # Right side: orders + activity stacked
        right = QWidget()
        rl = QVBoxLayout()
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        # Active orders
        orders_label = QLabel("Active Orders")
        orders_label.setFont(QFont(FONT_FAMILY, 11, QFont.Bold))
        orders_label.setStyleSheet(f"color: {BRAND_PRIMARY};")
        rl.addWidget(orders_label)

        self.orders_table = QTableWidget()
        self.orders_table.setColumnCount(7)
        self.orders_table.setHorizontalHeaderLabels(["Symbol", "Side", "Qty", "Type", "Price", "Status", ""])
        for c in range(6):
            self.orders_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
        self.orders_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.orders_table.verticalHeader().setVisible(False)
        self.orders_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.orders_table.setMaximumHeight(120)
        rl.addWidget(self.orders_table)

        # Activity feed
        act_label = QLabel("Recent Activity")
        act_label.setFont(QFont(FONT_FAMILY, 11, QFont.Bold))
        act_label.setStyleSheet(f"color: {BRAND_PRIMARY};")
        rl.addWidget(act_label)

        self.activity_feed = QTextEdit()
        self.activity_feed.setReadOnly(True)
        rl.addWidget(self.activity_feed)
        right.setLayout(rl)
        bl.addWidget(right, 2)

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
        # Run API calls in background thread
        import threading
        threading.Thread(target=self._refresh_data, daemon=True).start()

    def _refresh_data(self):
        """Fetch all data from Alpaca in background thread."""
        try:
            acct = self.broker.get_account()
            if "error" in acct:
                return
            positions = self.broker.get_positions()
            orders = self.broker.get_orders("open")
            hist = self.broker.get_portfolio_history(period="1W", timeframe="1H")
            # Update UI on main thread via signal (QTimer doesn't work from threads)
            self._data_ready.emit(acct, positions, orders, hist)
        except Exception:
            pass

    def _apply_refresh(self, acct, positions, orders, hist):
        """Apply fetched data to UI (main thread)."""

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
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    if j >= 5:
                        item.setForeground(QColor(COLOR_PROFIT if unrealized >= 0 else COLOR_LOSS))
                    self.pos_table.setItem(i, j, item)

                # Sell button per stock
                sell_btn = QPushButton(f"Sell")
                sell_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 10px; padding: 3px 8px;")
                sell_btn.clicked.connect(lambda _, s=sym, q=int(qty): self._sell_position(s, q))
                self.pos_table.setCellWidget(i, 7, sell_btn)

        # Active orders (pending/new/accepted only — filled orders go to history)
        open_orders = orders
        if isinstance(open_orders, list):
            # Filter to only truly pending orders
            pending = [o for o in open_orders if o.get("status") in
                       ("new", "accepted", "pending_new", "partially_filled", "held")]
            self.orders_table.setRowCount(len(pending))
            open_orders = pending
        if isinstance(open_orders, list) and open_orders:
            pass  # Continue to populate
        elif isinstance(open_orders, list):
            self.orders_table.setRowCount(0)
        if isinstance(open_orders, list):
            self.orders_table.setRowCount(len(open_orders))
            for i, o in enumerate(open_orders):
                # Show limit price if limit order, else "market"
                price_str = "market"
                if o.get("type") == "limit" and o.get("limit_price"):
                    price_str = f"${float(o['limit_price']):.2f}"
                elif o.get("type") == "stop" and o.get("stop_price"):
                    price_str = f"${float(o['stop_price']):.2f}"

                vals = [
                    o.get("symbol", ""), o.get("side", ""),
                    o.get("qty", "0"), o.get("type", ""),
                    price_str, o.get("status", ""),
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(str(v))
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    if j == 1:
                        item.setForeground(QColor(COLOR_BUY if v == "buy" else COLOR_SELL))
                    self.orders_table.setItem(i, j, item)

                # Cancel button
                oid = o.get("id", "")
                cancel_btn = QPushButton("Cancel")
                cancel_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 9px; padding: 2px 6px;")
                cancel_btn.clicked.connect(lambda _, _id=oid: self._cancel_order(_id))
                self.orders_table.setCellWidget(i, 6, cancel_btn)
        else:
            self.orders_table.setRowCount(0)

        # Chart (already fetched)
        if "error" not in hist and hist.get("equity"):
            self._plot(hist)

    def _plot(self, hist):
        try:
            self.figure.clear()
            cc = chart_colors()
            self.figure.set_facecolor(cc["fig_bg"])
            ax = self.figure.add_subplot(111)
            ax.set_facecolor(cc["ax_bg"])

            # Filter out None values (weekends/closed hours)
            eq = [e for e in hist["equity"] if e is not None]
            ts = [datetime.fromtimestamp(t) for t, e in zip(hist["timestamp"], hist["equity"]) if e is not None]

            if not eq or not ts:
                ax.text(0.5, 0.5, "No equity data available", ha="center", va="center",
                        color=cc["muted"], fontsize=12, transform=ax.transAxes)
                self.canvas.draw()
                return

            # Use integer indices for even spacing (no time gaps from weekends)
            x = list(range(len(eq)))
            color = COLOR_PROFIT if eq[-1] >= eq[0] else COLOR_LOSS

            ax.plot(x, eq, color=color, linewidth=2)
            eq_min = min(eq)
            ax.fill_between(x, eq, eq_min, alpha=0.08, color=color)

            # Y-axis: zoom to data range with padding
            eq_range = max(eq) - eq_min if max(eq) != eq_min else 1
            padding = eq_range * 0.15
            ax.set_ylim(eq_min - padding, max(eq) + padding)

            # Current value annotation
            ax.annotate(f"${eq[-1]:,.2f}", xy=(x[-1], eq[-1]),
                        fontsize=9, fontweight="bold", color=color,
                        xytext=(-70, 10), textcoords="offset points")

            ax.set_title("Portfolio Equity (1W)", color=cc["text"], fontsize=10)
            ax.tick_params(colors=cc["muted"], labelsize=7)
            ax.grid(True, alpha=0.15, color=cc["grid"])

            # X-axis: evenly spaced date labels
            n = len(x)
            n_ticks = min(5, n)
            step = max(1, n // n_ticks)
            tick_idx = list(range(0, n, step))
            if tick_idx[-1] != n - 1:
                tick_idx.append(n - 1)
            ax.set_xticks([x[i] for i in tick_idx])
            ax.set_xticklabels([ts[i].strftime("%m/%d %H:%M") for i in tick_idx],
                               fontsize=6, color=cc["muted"])

            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(
                lambda v, _: f"${v:,.0f}"
            ))

            self.figure.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.14)
            self.canvas.draw()

            from core.ui.chart_tooltip import ChartTooltip
            self._tooltip = ChartTooltip(self.canvas, ax, x, eq, x_labels=ts)
        except Exception as e:
            self.bus.log_entry.emit(f"Chart error: {e}", "error")

    def _cancel_order(self, order_id):
        """Cancel a pending order and log it."""
        if not self.broker:
            return
        # Get order details before cancelling for the log
        orders = self.broker.get_orders("open")
        ticker, side, qty = "", "", 0
        if isinstance(orders, list):
            for o in orders:
                if o.get("id") == order_id:
                    ticker = o.get("symbol", "")
                    side = o.get("side", "")
                    qty = o.get("qty", 0)
                    break

        result = self.broker.cancel_order(order_id)
        if "error" in result:
            self.bus.log_entry.emit(f"Cancel failed: {result['error']}", "error")
        else:
            self.bus.log_entry.emit(f"Order cancelled: {ticker} {side} x{qty}", "trade")
            from core.logger import log_cancellation
            log_cancellation(ticker, order_id, side, qty)
            QTimer.singleShot(1000, self.refresh)

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

