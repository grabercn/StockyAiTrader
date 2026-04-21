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

        from PyQt5.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        outer = QVBoxLayout()
        outer.setSpacing(6)
        outer.setContentsMargins(8, 4, 8, 4)

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
        self.period_cb.addItems(["Auto (Smart)", "5 days", "3 days", "2 days", "1 day"])
        self.period_cb.setToolTip("Amount of historical data used to train the AI model for each stock")
        settings_row.addWidget(self.period_cb)
        interval_lbl = QLabel("Bar Size:")
        interval_lbl.setToolTip("The time interval for each price bar.\n5min = one data point every 5 minutes.\nSmaller bars = more data points but noisier.")
        settings_row.addWidget(interval_lbl)
        self.interval_cb = QComboBox()
        self.interval_cb.addItems(["Auto (Smart)", "5 min", "1 min", "15 min"])
        self.interval_cb.setToolTip("Auto = AI picks best interval per stock based on volatility. Or pick fixed.")
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
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
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

        # Order type: Market vs Limit
        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Order:"))
        self.order_type_cb = QComboBox()
        self.order_type_cb.addItems(["Market", "Limit"])
        self.order_type_cb.setToolTip("Market = instant at current price. Limit = set a price ceiling (buy) or floor (sell).")
        order_row.addWidget(self.order_type_cb)
        order_row.addWidget(QLabel("Price:"))
        self.limit_price_input = QLineEdit()
        self.limit_price_input.setPlaceholderText("e.g. 150.00")
        self.limit_price_input.setStyleSheet(f"padding: 4px; min-width: 80px;")
        self.limit_price_input.setEnabled(False)
        order_row.addWidget(self.limit_price_input)
        self.order_type_cb.currentTextChanged.connect(
            lambda t: self.limit_price_input.setEnabled(t == "Limit")
        )
        tbl.addLayout(order_row)

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
        period_map = {"Auto (Smart)": "5d", "5 days": "5d", "3 days": "3d", "2 days": "2d", "1 day": "1d"}
        interval_map = {"Auto (Smart)": "5m", "5 min": "5m", "1 min": "1m", "15 min": "15m"}
        period = period_map.get(self.period_cb.currentText(), "5d")
        interval = interval_map.get(self.interval_cb.currentText(), "5m")

        # Auto mode: either dropdown set to "Auto (Smart)" triggers per-stock optimization
        is_auto = (self.period_cb.currentText() == "Auto (Smart)" or
                   self.interval_cb.currentText() == "Auto (Smart)")

        if is_auto:
            self.progress.add_log(
                "Smart mode: AI auto-picks training period + bar interval per stock "
                "based on volatility and volume characteristics"
            )

        # Estimate time — use saved average if available, else default 8s/stock
        from core.profiles import get_optimal_workers
        settings = load_settings()
        avg_per_stock = settings.get("scan_avg_seconds", 8.0)
        workers = get_optimal_workers()
        est_seconds = max(5, int(len(tickers) * avg_per_stock / workers))
        self._scan_total = len(tickers)
        self._scan_est = est_seconds
        self.progress.set_progress(5, f"Starting scan...",
            f"0/{len(tickers)} — est. ~{est_seconds}s")
        self.progress.add_log(f"Estimated time: ~{est_seconds}s for {len(tickers)} stocks")

        from panels.workers import ScanWorker as _SW
        self._worker = _SW(tickers, period, interval, self.rm, auto_settings=is_auto)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

        # Poll worker progress from main thread (cross-thread signals unreliable)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_progress)
        self._poll_timer.start(250)  # Check 4x per second

    def _poll_progress(self):
        """Poll worker for progress updates (runs on main thread via QTimer)."""
        if not hasattr(self, '_worker') or not self._worker:
            return
        items = self._worker.poll_progress()
        for done, total, ticker, detail in items:
            self._on_progress(done, total, ticker, detail)

    def _on_progress(self, done, total, ticker, detail):
        try:
            pct = 10 + int(done / total * 85) if total > 0 else 0
            elapsed = time.time() - getattr(self, '_t0', time.time())
            est = getattr(self, '_scan_est', 60)

            # Live ETA
            if done > 0:
                per_stock = elapsed / done
                from core.profiles import get_optimal_workers
                remaining = (total - done) * per_stock / get_optimal_workers()
                eta = max(0, int(remaining))
                eta_str = f"{eta}s left" if eta < 120 else f"{eta//60}m {eta%60}s left"
            else:
                eta_str = "calculating..."

            elapsed_int = int(elapsed)
            if elapsed_int > est:
                detail_text = f"{done}/{total} | {eta_str} | {elapsed_int}s (over est. {est}s)"
            else:
                detail_text = f"{done}/{total} | {eta_str} | {elapsed_int}s / est. {est}s"

            self.progress.set_progress(pct, f"Scanning {ticker}...", detail_text)

            action = detail.split(" ")[0] if detail else "..."
            colors = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#f59e0b"}
            color = colors.get(action, "#94a3b8")
            self.progress.add_log(f"{ticker}: <b style='color:{color}'>{detail}</b>")
            QApplication.processEvents()
        except Exception as e:
            print(f"[PROGRESS ERROR] {e}", flush=True)

    def _on_done(self, results):
        # Stop polling, drain any remaining progress
        if hasattr(self, '_poll_timer'):
            self._poll_timer.stop()
            self._poll_progress()  # Drain final items
        self.results = results
        elapsed = time.time() - self._t0
        est = getattr(self, '_scan_est', 0)
        speed = "faster" if elapsed < est else "slower"
        self.progress.set_progress(100,
            f"Scan complete — {len(results)} stocks in {elapsed:.1f}s",
            f"Est. was {est}s ({speed} than expected)"
        )

        # Save timing for better future estimates (rolling average)
        if len(results) > 0:
            from core.profiles import get_optimal_workers
            per_stock = elapsed / len(results) * get_optimal_workers()
            settings = load_settings()
            old_avg = settings.get("scan_avg_seconds", 8.0)
            # Weighted average: 70% new, 30% old
            settings["scan_avg_seconds"] = round(per_stock * 0.7 + old_avg * 0.3, 2)
            save_settings(settings)
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

            # Show period/interval in SL/TP column when auto mode picked different settings
            p_used = getattr(r, 'period_used', '5d')
            i_used = getattr(r, 'interval_used', '5m')
            settings_str = f"{p_used}/{i_used}" if (p_used != "5d" or i_used != "5m") else ""

            is_insufficient = bool(r.error)
            grey = "#8899aa"  # Lighter grey that's visible on both dark and light backgrounds

            if is_insufficient:
                vals = [r.ticker, "--", "--", f"${r.price:.2f}" if r.price else "--",
                        "--", "--", r.error[:30] if r.error else "no data"]
                for j, val in enumerate(vals):
                    it = QTableWidgetItem(str(val))
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    it.setForeground(QColor(grey))
                    if j == 0:
                        it.setFont(QFont(FONT_MONO, 11, QFont.Bold))
                    self.table.setItem(i, j + 1, it)
            else:
                signal_color = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}.get(r.action, TEXT_SECONDARY)
                vals = [
                    (r.ticker, TEXT_SECONDARY, True),
                    (r.action, signal_color, True),
                    (f"{r.confidence:.0%}", TEXT_SECONDARY, False),
                    (f"${r.price:.2f}" if r.price else "--", TEXT_SECONDARY, False),
                    (str(r.position_size) if r.position_size else "--", TEXT_SECONDARY, False),
                    (f"${r.stop_loss:.0f}/${r.take_profit:.0f}" if r.stop_loss else "--", TEXT_SECONDARY, False),
                    (f"{r.score:.2f}" + (f" [{settings_str}]" if settings_str else ""), TEXT_SECONDARY, False),
                ]
                for j, (val, color, bold) in enumerate(vals):
                    it = QTableWidgetItem(str(val))
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    it.setForeground(QColor(color))
                    if bold:
                        it.setFont(QFont(FONT_MONO, 11, QFont.Bold))
                    self.table.setItem(i, j + 1, it)

            # Auto-trade toggle — minimal icon button
            from core.ui.icons import StockyIcons as _Icons
            is_monitored = hasattr(self, '_auto_service') and self._auto_service and self._auto_service.is_monitoring(r.ticker)
            if not is_insufficient:
                monitor_btn = QPushButton()
                monitor_btn.setFixedSize(28, 28)
                monitor_btn.setCursor(Qt.PointingHandCursor)
                if is_monitored:
                    monitor_btn.setIcon(_Icons.get_icon("robot", 16, BRAND_ACCENT))
                    monitor_btn.setToolTip(f"Monitoring {r.ticker} — click to stop")
                    monitor_btn.setStyleSheet(f"background: {BRAND_ACCENT}25; border: none; border-radius: 14px;")
                else:
                    monitor_btn.setIcon(_Icons.get_icon("play", 16, TEXT_MUTED))
                    monitor_btn.setToolTip(f"Auto-trade {r.ticker}")
                    monitor_btn.setStyleSheet(f"background: transparent; border: none; border-radius: 14px;")
                monitor_btn.clicked.connect(lambda _, t=r.ticker: self._toggle_auto_trade(t))
                # Center the button in the cell
                w = QWidget()
                l = QHBoxLayout(w)
                l.addWidget(monitor_btn)
                l.setAlignment(Qt.AlignCenter)
                l.setContentsMargins(0, 0, 0, 0)
                self.table.setCellWidget(i, 8, w)

            # Reasoning (last column)
            if is_insufficient:
                reason_it = QTableWidgetItem(r.error[:60] if r.error else "No data")
                reason_it.setForeground(QColor(grey))
            else:
                reason_it = QTableWidgetItem(r.reasoning[:80] if r.reasoning else "")
            reason_it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            reason_it.setToolTip(r.reasoning or r.error or "")
            self.table.setItem(i, 9, reason_it)

        buys = sum(1 for r in results if r.action == "BUY")
        sells = sum(1 for r in results if r.action == "SELL")
        errors = sum(1 for r in results if r.error)
        holds = len(results) - buys - sells - errors
        self.summary.setText(
            f"{buys} BUY  |  {sells} SELL  |  {holds} HOLD  |  {errors} failed  |  {elapsed:.1f}s"
        )

        # Log why stocks failed so user knows
        if errors > 0:
            failed = [r for r in results if r.error]
            reasons = {}
            for r in failed:
                reason = r.error or "Unknown"
                if "Not enough data" in reason or "not enough" in reason.lower():
                    reasons["Not enough data"] = reasons.get("Not enough data", 0) + 1
                elif "training" in reason.lower():
                    reasons["Model training failed"] = reasons.get("Model training failed", 0) + 1
                else:
                    reasons[reason[:40]] = reasons.get(reason[:40], 0) + 1

            reason_str = ", ".join(f"{v}x {k}" for k, v in reasons.items())
            self.bus.log_entry.emit(
                f"{errors}/{len(results)} stocks failed: {reason_str}. "
                f"Try a longer training period (5d) or different bar size.",
                "warn",
            )
            self.progress.add_log(f"<b style='color:{COLOR_HOLD}'>{errors} failed:</b> {reason_str}")

        self.bus.scan_completed.emit([{"ticker": r.ticker, "action": r.action} for r in results])
        self.bus.log_entry.emit(f"Scan done: {buys} BUY, {sells} SELL, {errors} failed in {elapsed:.1f}s", "trade")

    # ── Detail Panel ──────────────────────────────────────────────────────

    def _on_row_clicked(self, row, col):
        if row < 0 or row >= len(self.results):
            return
        r = self.results[row]
        self._selected_result = r
        self._show_detail(r)

    def _show_detail(self, r):
        """Show detailed breakdown — heavy data loads in background thread."""
        self.detail_title.setText(f"{r.action}  {r.ticker}")
        colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
        self.detail_title.setStyleSheet(f"color: {colors.get(r.action, BRAND_PRIMARY)};")

        # Set qty and buttons immediately (no network needed)
        qty = r.position_size if r.position_size > 0 else 1
        self.detail_qty.setValue(qty)
        self.detail_qty_label.setText(f"AI recommends {r.position_size} shares")
        cost = qty * r.price if r.price > 0 else 0
        self.detail_buy_btn.setText(f"  BUY {qty} shares (${cost:,.0f})")
        self.detail_sell_btn.setVisible(False)
        self.detail_owned_label.setText("")

        # Show instant signal data + loading indicators for async sections
        loading = f'<span style="color:{TEXT_MUTED}"><i>loading...</i></span>'
        instant_html = (
            f'<b style="color:{BRAND_ACCENT}">Loading details for {r.ticker}...</b><br><br>'
            f'<b style="color:{BRAND_PRIMARY}">Signal: {r.action}</b> ({r.confidence:.1%} confidence, score {r.score:.3f})<br>'
            f'<b>Price:</b> ${r.price:.2f}  |  <b>ATR:</b> ${r.atr:.2f} ({r.atr/r.price*100:.1f}%)<br>'
            f'<b>Position:</b> {r.position_size} shares  |  <b>SL:</b> ${r.stop_loss:.2f}  |  <b>TP:</b> ${r.take_profit:.2f}<br><br>'
            f'<b style="color:{BRAND_PRIMARY}">Company Info:</b> {loading}<br>'
            f'<b style="color:{BRAND_ACCENT}">AI Analysis:</b> {loading}<br>'
            f'<b style="color:{BRAND_PRIMARY}">Addon Signals:</b> {loading}<br>'
        )
        self.detail_stats.setHtml(instant_html)

        # Progressive async loading
        import threading
        if not hasattr(self, '_stock_info_cache'):
            self._stock_info_cache = {}

        def _load_progressive():
            if self._selected_result is not r:
                return

            # Phase 1: Stock info (1-3s)
            if r.ticker not in self._stock_info_cache:
                try:
                    info = yf.Ticker(r.ticker).info
                    self._stock_info_cache[r.ticker] = {
                        "name": info.get("shortName", r.ticker),
                        "desc": info.get("longBusinessSummary", ""),
                        "sector": info.get("sector", ""),
                        "industry": info.get("industry", ""),
                        "cap": f"${info.get('marketCap',0)/1e9:.1f}B" if info.get('marketCap',0) > 1e9 else f"${info.get('marketCap',0)/1e6:.0f}M" if info.get('marketCap',0) > 1e6 else "",
                    }
                except Exception:
                    self._stock_info_cache[r.ticker] = {}

            if self._selected_result is not r:
                return

            # Phase 2: Build main detail HTML (instant, uses cached info)
            html = self._build_detail_html(r)
            QTimer.singleShot(0, lambda: self._apply_detail(r, html))

            if self._selected_result is not r:
                return

            # Phase 3: LLM reasoning (5-30s on first load)
            try:
                from core.llm_reasoner import generate_reasoning
                llm_text = generate_reasoning(
                    r.ticker, r.action, r.confidence, r.price, r.atr, r.probs,
                    feature_importances=r.feature_importances,
                )
                if llm_text and self._selected_result is r:
                    updated = html.replace(
                        '<i style="color:' + TEXT_MUTED + '">Loading AI reasoning...</i>',
                        "<br>".join(f"  {p}" for p in llm_text.split(" | "))
                    )
                    QTimer.singleShot(0, lambda: self._apply_detail(r, updated))
                    # Also update the reasoning column in the table
                    QTimer.singleShot(0, lambda: self._update_reasoning_column(r.ticker, llm_text))
            except Exception:
                if self._selected_result is r:
                    updated = html.replace(
                        '<i style="color:' + TEXT_MUTED + '">Loading AI reasoning...</i>',
                        f'<span style="color:{TEXT_MUTED}">AI model unavailable</span>'
                    )
                    QTimer.singleShot(0, lambda: self._apply_detail(r, updated))

        threading.Thread(target=_load_progressive, daemon=True).start()
        QTimer.singleShot(0, lambda: self._draw_detail_chart(r.ticker))

    def _apply_detail(self, r, html):
        """Apply the loaded detail HTML and update buttons (main thread)."""
        if self._selected_result is not r:
            return  # User clicked a different stock
        self.detail_stats.setHtml(html)

        # Check ownership
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

        qty = self.detail_qty.value()
        if owned_qty > 0:
            self.detail_sell_btn.setVisible(True)
            self.detail_sell_btn.setText(f"  SELL {min(qty, owned_qty)} shares")
            self.detail_owned_label.setText(f"You own {owned_qty} shares")
            if r.action == "SELL":
                self.detail_qty.setValue(min(r.position_size, owned_qty))
        else:
            self.detail_sell_btn.setVisible(False)
            self.detail_owned_label.setText("You don't own this stock")

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

    def _build_detail_html(self, r):
        """Build the detail HTML string — can run on any thread."""
        lines = []
        info = self._stock_info_cache.get(r.ticker, {})
        if info.get("name"):
            lines.append(f'<b style="color:{BRAND_PRIMARY};font-size:13px">{info["name"]}</b>')
            meta = []
            if info.get("sector"):
                meta.append(info["sector"])
            if info.get("industry"):
                meta.append(info["industry"])
            if info.get("cap"):
                meta.append(info["cap"])
            if meta:
                lines.append(f'<span style="color:{TEXT_MUTED}">{" · ".join(meta)}</span>')
            if info.get("desc"):
                short = info["desc"][:200] + ("..." if len(info["desc"]) > 200 else "")
                lines.append(f'<span style="color:{TEXT_SECONDARY};font-size:10px">{short}</span>')
            lines.append("")

        # ── Your Position + Pending Orders ──
        if self.broker:
            try:
                positions = self.broker.get_positions()
                orders = self.broker.get_orders("open")
                pos_qty = 0
                pos_pnl = 0.0
                if isinstance(positions, list):
                    for p in positions:
                        if p.get("symbol", "").upper() == r.ticker.upper():
                            pos_qty = int(float(p.get("qty", 0)))
                            pos_pnl = float(p.get("unrealized_pl", 0))
                            break

                pending = []
                if isinstance(orders, list):
                    for o in orders:
                        if o.get("symbol", "").upper() == r.ticker.upper():
                            side = o.get("side", "")
                            oqty = o.get("qty", "0")
                            otype = o.get("type", "market")
                            status = o.get("status", "")
                            price_s = ""
                            if otype == "limit" and o.get("limit_price"):
                                price_s = f" @ ${float(o['limit_price']):.2f}"
                            pending.append(f"{side.upper()} {oqty}{price_s} [{status}]")

                if pos_qty > 0 or pending:
                    lines.append(f'<b style="color:{BRAND_ACCENT}">Your Position</b>')
                    if pos_qty > 0:
                        pnl_color = COLOR_PROFIT if pos_pnl >= 0 else COLOR_LOSS
                        lines.append(f'  Holding: {pos_qty} shares  |  <span style="color:{pnl_color}">P&L: ${pos_pnl:+,.2f}</span>')
                    if pending:
                        for p in pending:
                            lines.append(f'  Pending: {p}')
                    lines.append("")
            except Exception:
                pass

        # ── LLM Analysis (loaded separately, appended async) ──
        lines.append(f'<b style="color:{BRAND_ACCENT}">AI Analysis</b>')
        lines.append(f'  <i style="color:{TEXT_MUTED}">Loading AI reasoning...</i>')
        lines.append("")

        # ── Signal Overview ──
        lines.append(f'<b style="color:{BRAND_PRIMARY}">Signal: {r.action}</b> ({r.confidence:.1%} confidence, score {r.score:.3f})')
        lines.append(f'<b>Price:</b> ${r.price:.2f}  |  <b>ATR:</b> ${r.atr:.2f} ({r.atr/r.price*100:.1f}%)')
        lines.append(f'<b>Position:</b> {r.position_size} shares  |  <b>SL:</b> ${r.stop_loss:.2f}  |  <b>TP:</b> ${r.take_profit:.2f}')
        lines.append("")

        # ── Share Quantity Reasoning ──
        if r.position_size > 0 and r.price > 0 and r.atr > 0:
            lines.append(f'<b style="color:{BRAND_PRIMARY}">Why {r.position_size} Shares?</b>')
            cost = r.position_size * r.price
            risk_per_share = abs(r.price - r.stop_loss) if r.stop_loss else r.atr
            total_risk = r.position_size * risk_per_share
            lines.append(f'  Total cost: ${cost:,.2f}')
            lines.append(f'  Risk per share: ${risk_per_share:.2f} (ATR-based stop distance)')
            lines.append(f'  Total risk: ${total_risk:,.2f} if stop-loss triggers')
            if r.confidence > 0.7:
                lines.append(f'  Sizing boosted: high confidence ({r.confidence:.0%}) allows larger position')
            elif r.confidence < 0.4:
                lines.append(f'  Sizing reduced: low confidence ({r.confidence:.0%}) limits exposure')
            vol_label = "volatile" if r.atr / r.price > 0.02 else "calm"
            lines.append(f'  Stock is {vol_label} — {"fewer shares to manage risk" if vol_label == "volatile" else "more shares since risk is lower"}')
            lines.append("")

        # ── Trend-Based Sell Recommendation ──
        if r.action == "BUY" and r.take_profit > 0 and r.price > 0:
            gain_pct = (r.take_profit - r.price) / r.price * 100
            lines.append(f'<b style="color:{BRAND_PRIMARY}">Sell Recommendation</b>')
            lines.append(f'  Target exit: ${r.take_profit:.2f} (+{gain_pct:.1f}% from entry)')
            lines.append(f'  Stop loss: ${r.stop_loss:.2f} (-{abs(r.price - r.stop_loss) / r.price * 100:.1f}%)')
            if r.probs[2] > 0.7:
                lines.append(f'  Trend: <b style="color:{COLOR_BUY}">Strong bullish</b> — consider holding past target')
            elif r.probs[2] > 0.5:
                lines.append(f'  Trend: Moderate bullish — sell at target')
            else:
                lines.append(f'  Trend: Weak — sell quickly if it hits target')
            lines.append(f'  <i style="color:{BRAND_ACCENT}">Set a limit sell at ${r.take_profit:.2f} using Quick Trade above</i>')
            lines.append("")

        # ── Probabilities ──
        lines.append(f'<b style="color:{BRAND_PRIMARY}">Probabilities</b>')
        for label, prob, color in [("SELL", r.probs[0], COLOR_SELL), ("HOLD", r.probs[1], COLOR_HOLD), ("BUY", r.probs[2], COLOR_BUY)]:
            bar_w = int(prob * 20)
            bar = "█" * bar_w + "░" * (20 - bar_w)
            lines.append(f'  <span style="color:{color}">{label} {prob:.0%}</span> <span style="color:{TEXT_MUTED}">{bar}</span>')

        # ── Key Indicators ──
        if r.feature_importances:
            lines.append("")
            lines.append(f'<b style="color:{BRAND_PRIMARY}">Key Indicators</b>')
            max_imp = max(r.feature_importances.values()) if r.feature_importances else 1
            for feat, imp in sorted(r.feature_importances.items(), key=lambda x: -x[1])[:8]:
                bar_len = int((imp / max_imp) * 15)
                bar = "█" * bar_len
                lines.append(f'  <span style="color:{BRAND_ACCENT}">{bar}</span> {feat}: {imp:.0f}')

        # ── Addon Signals ──
        # Collect active addon contributions for this stock
        lines.append("")
        lines.append(f'<b style="color:{BRAND_PRIMARY}">Addon Signals</b>')
        try:
            from addons import get_all_addons
            active_addons = [a for a in get_all_addons() if a.available and a.enabled]
            if active_addons:
                for addon in active_addons:
                    try:
                        features = addon.get_features(r.ticker)
                        if features and isinstance(features, dict):
                            # Show which features this addon contributed
                            feat_strs = []
                            for k, v in list(features.items())[:3]:
                                if isinstance(v, float):
                                    feat_strs.append(f"{k}: {v:.2f}")
                                else:
                                    feat_strs.append(f"{k}: {v}")
                            if feat_strs:
                                lines.append(f'  <b>{addon.name}:</b> {" · ".join(feat_strs)}')
                    except Exception:
                        pass
                if len(active_addons) == 0 or all(not a.enabled for a in active_addons):
                    lines.append(f'  <span style="color:{TEXT_MUTED}">No addons active — enable in Settings > Hardware Profile</span>')
            else:
                lines.append(f'  <span style="color:{TEXT_MUTED}">No addons active</span>')
            lines.append(f'  <span style="color:{TEXT_MUTED}">Add custom addons: drop a .py file in StockyApps/addons/</span>')
        except Exception:
            lines.append(f'  <span style="color:{TEXT_MUTED}">Addon data unavailable</span>')

        # ── Advanced Statistics ──
        lines.append("")
        lines.append(f'<b style="color:{BRAND_PRIMARY}">Advanced Statistics</b>')
        vol = "High" if r.atr / r.price > 0.02 else "Moderate" if r.atr / r.price > 0.01 else "Low"
        direction = "Bullish" if r.probs[2] > r.probs[0] else "Bearish"
        strength = "Strong" if r.confidence > 0.7 else "Moderate" if r.confidence > 0.4 else "Weak"
        if r.stop_loss and r.stop_loss < r.price:
            rr = (r.take_profit - r.price) / (r.price - r.stop_loss)
            max_loss = r.position_size * (r.price - r.stop_loss)
            max_gain = r.position_size * (r.take_profit - r.price)
            lines.append(f'  Risk/Reward: <b>1:{rr:.1f}</b>  |  Max Loss: ${max_loss:,.0f}  |  Max Gain: ${max_gain:,.0f}')
        lines.append(f'  Volatility: {vol}  |  Direction: {direction}  |  Strength: {strength}')

        period_used = getattr(r, 'period_used', '5d')
        interval_used = getattr(r, 'interval_used', '5m')
        trade_type = "Intraday" if r.atr and r.atr / r.price < 0.03 else "Swing"
        lines.append(f'  Type: {trade_type}  |  Data: {period_used}/{interval_used}')
        if period_used != "5d" or interval_used != "5m":
            lines.append(f'  <i style="color:{BRAND_ACCENT}">AI auto-selected settings for this stock</i>')

        # ── AI Reasoning ──
        if r.reasoning:
            lines.append("")
            lines.append(f'<b style="color:{BRAND_PRIMARY}">AI Reasoning</b>')
            for part in r.reasoning.split(" | "):
                lines.append(f'  {part}')

        return "<br>".join(lines)

    def _update_reasoning_column(self, ticker, llm_text):
        """Update the reasoning column in the results table when LLM finishes."""
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 1)  # Ticker column
            if item and item.text() == ticker:
                short = llm_text.split(" | ")[0][:80] if llm_text else ""
                reason_item = self.table.item(i, 9)  # Reasoning column
                if reason_item:
                    reason_item.setText(short)
                    reason_item.setToolTip(llm_text)
                break

    def _draw_detail_chart(self, ticker):
        """Draw price chart using the scan's period/interval for the selected stock."""
        self.detail_figure.clear()
        cc = chart_colors(); self.detail_figure.set_facecolor(cc["fig_bg"])
        try:
            r = self._selected_result
            period = getattr(r, 'period_used', '5d') if r else '5d'
            interval = getattr(r, 'interval_used', '5m') if r else '5m'

            data = yf.Ticker(ticker).history(period=period, interval=interval)
            if data.empty or len(data) < 2:
                data = yf.Ticker(ticker).history(period="1mo", interval="1d")
                period, interval = "1mo", "1d"
            if data.empty:
                return

            ax = self.detail_figure.add_subplot(111)
            ax.set_facecolor(cc["ax_bg"])
            closes = data["Close"].values
            timestamps = data.index

            trending_up = closes[-1] >= closes[0]
            color = COLOR_BUY if trending_up else COLOR_SELL

            ax.plot(timestamps, closes, color=color, linewidth=1.5)
            ax.fill_between(timestamps, closes, alpha=0.08, color=color)

            ax.annotate(f"${closes[-1]:.2f}", xy=(timestamps[-1], closes[-1]),
                       fontsize=8, fontweight="bold", color=color,
                       xytext=(-50, 8), textcoords="offset points")

            ax.set_title(f"{ticker} — {period} @ {interval}", color=cc["text"], fontsize=10)
            ax.tick_params(colors=cc["muted"], labelsize=6)
            ax.grid(True, alpha=0.15, color=cc["grid"])

            # Format x-axis based on interval
            n = len(timestamps)
            if n > 8:
                step = max(1, n // 6)
                tick_idx = list(range(0, n, step))
            else:
                tick_idx = list(range(n))

            # Pick format based on data range
            if interval in ("1m", "5m", "15m", "30m"):
                fmt = "%m/%d %H:%M"
            else:
                fmt = "%m/%d"

            ax.set_xticks([timestamps[i] for i in tick_idx])
            ax.set_xticklabels([timestamps[i].strftime(fmt) for i in tick_idx],
                               rotation=30, ha="right", fontsize=6)

            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
            self.detail_figure.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.22)
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

        # Determine order type and limit price
        is_limit = self.order_type_cb.currentText() == "Limit"
        limit_price = None
        if is_limit:
            try:
                limit_price = float(self.limit_price_input.text())
            except (ValueError, TypeError):
                self.bus.log_entry.emit("Enter a valid limit price", "warn")
                return

        order_type = "limit" if is_limit else "market"
        price_str = f"${limit_price:.2f} (limit)" if limit_price else f"${r.price:.2f} (market)"
        cost = qty * (limit_price or r.price)

        confirm = QMessageBox.question(
            self, f"Confirm {side.upper()}",
            f"{side.upper()} {qty} shares of {r.ticker}\n"
            f"Order Type: {order_type.upper()}\n"
            f"Price: {price_str}\n"
            f"Estimated Total: ${cost:,.2f}\n"
            + (f"Stop Loss: ${r.stop_loss:.2f}\n"
               f"Take Profit: ${r.take_profit:.2f}\n" if not is_limit else "")
            + f"\nProceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        # Execute trade in background thread to not freeze UI
        import threading
        self.bus.log_entry.emit(f"Submitting {side.upper()} {r.ticker} x{qty}...", "info")

        def _execute():
            if side == "sell" and not is_limit:
                result = self.broker.close_position(r.ticker, qty=qty)
            else:
                result = self.broker.place_order(
                    r.ticker, qty, side, order_type=order_type,
                    stop_loss=r.stop_loss if not is_limit else None,
                    take_profit=r.take_profit if not is_limit else None,
                    limit_price=limit_price,
                )
            # Update UI on main thread
            QTimer.singleShot(0, lambda: self._on_trade_result(result, side, r, qty))

        threading.Thread(target=_execute, daemon=True).start()

    def _on_trade_result(self, result, side, r, qty):
        if "error" in result:
            self.bus.log_entry.emit(f"{side.upper()} {r.ticker} x{qty} failed: {result['error']}", "error")
        else:
            self.bus.log_entry.emit(f"{side.upper()} {r.ticker} x{qty} — order {result.get('id','?')}", "trade")
            self.bus.positions_changed.emit()
            QTimer.singleShot(2000, lambda: self._show_detail(r))

    def _deep_analyze(self):
        """Run deep analysis in background thread with ETA."""
        if not self._selected_result:
            self.bus.log_entry.emit("Select a stock first", "warn")
            return

        r = self._selected_result
        settings = load_settings()
        est = int(settings.get("deep_analyze_avg_seconds", 60))

        self.bus.log_entry.emit(f"Deep analyzing {r.ticker} (est. ~{est}s)...", "info")
        self.progress.setVisible(True)
        self.progress.reset()
        self.progress.set_progress(5, f"Deep analyzing {r.ticker}...",
                                    f"Estimated time: ~{est}s")
        self.progress.add_log(f"Starting deep analysis for {r.ticker}")
        self.detail_analyze_btn.setEnabled(False)
        self._deep_t0 = time.time()
        self._deep_est = est

        from panels.workers import _DeepAnalyzeWorker as _DAW
        self._deep_worker = _DAW(r)
        self._deep_worker.finished_signal.connect(self._on_deep_done)
        self._deep_worker.start()

        self._deep_poll = QTimer(self)
        self._deep_poll.timeout.connect(self._poll_deep_progress)
        self._deep_poll.start(200)

    def _poll_deep_progress(self):
        if not hasattr(self, '_deep_worker') or not self._deep_worker:
            return
        elapsed = int(time.time() - getattr(self, '_deep_t0', time.time()))
        est = getattr(self, '_deep_est', 60)

        for pct, msg, detail in self._deep_worker.poll_progress():
            if elapsed > est:
                eta_str = f"{elapsed}s (over est. {est}s)"
            else:
                remaining = max(0, est - elapsed)
                eta_str = f"{elapsed}s / est. {est}s | {remaining}s left"
            self.progress.set_progress(pct, msg, eta_str)
            self.progress.add_log(msg)
            QApplication.processEvents()

    def _on_deep_done(self, report_text, ticker):
        """Open the report popup when deep analysis completes."""
        if hasattr(self, '_deep_poll'):
            self._deep_poll.stop()
            self._poll_deep_progress()
        self.detail_analyze_btn.setEnabled(True)

        elapsed = time.time() - getattr(self, '_deep_t0', time.time())
        est = getattr(self, '_deep_est', 60)
        speed = "faster" if elapsed < est else "slower"
        self.progress.set_progress(100, f"Deep analysis complete — {elapsed:.0f}s",
                                    f"Est. was {est}s ({speed} than expected)")
        self.bus.log_entry.emit(f"Deep analysis complete for {ticker} in {elapsed:.0f}s", "trade")

        # Save timing for better future estimates (rolling average)
        settings = load_settings()
        old_avg = settings.get("deep_analyze_avg_seconds", 60)
        settings["deep_analyze_avg_seconds"] = round(elapsed * 0.7 + old_avg * 0.3, 1)
        save_settings(settings)

        from core.ui.theme import theme as _theme
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Deep Analysis - {ticker}")
        dlg.setWindowIcon(QApplication.instance().windowIcon())
        dlg.setMinimumSize(650, 550)
        dlg.setStyleSheet(f"QDialog {{ background-color: {_theme.color('bg_base')}; }}")

        lay = QVBoxLayout()
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFont(QFont(FONT_MONO, 10))
        txt.setPlainText(report_text)
        lay.addWidget(txt)

        btn_row = QHBoxLayout()
        day_btn = QPushButton("Open in Day Trade")
        day_btn.clicked.connect(lambda: (self.bus.ticker_selected.emit(ticker), dlg.accept()))
        btn_row.addWidget(day_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        dlg.setLayout(lay)
        dlg.exec_()

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

