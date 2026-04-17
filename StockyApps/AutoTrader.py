"""
AutoTrader — Multi-stock portfolio scanner and auto-investment tool.

Scans a universe of tickers, ranks them by opportunity, and lets you
select which ones to auto-invest in. Designed to be simple enough for
a Robinhood/Webull user but with advanced data underneath.

Features:
- Scan 20+ tickers concurrently (3 threads to stay laptop-friendly)
- Ranked recommendations with confidence scores
- One-click auto-invest on selected tickers
- Comprehensive log viewer with reasoning for every decision
- Portfolio allocation view
"""

import sys
import os
import time
import json
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QProgressBar, QComboBox, QGroupBox, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QTabWidget,
    QSpinBox, QSplitter, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor

from core.scanner import scan_multiple, DEFAULT_TICKERS, TECH_TICKERS, ETF_TICKERS, ScanResult
from core.risk import RiskManager
from core.broker import AlpacaBroker
from core.logger import (
    log_decision, log_trade_execution, log_scan_results, log_event,
    get_today_logs, get_log_files, get_log_entries,
)
from core.signals import write_signal
from core.style import APP_STYLESHEET, log_html
from core.chart import COLOR_BUY, COLOR_SELL, COLOR_HOLD, BG_DARK

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "settings.json")


# ─── Scan Worker (background thread) ────────────────────────────────────────
class ScanWorker(QThread):
    """Runs portfolio scan in background so UI stays responsive."""
    progress = pyqtSignal(int, int, str, str)   # completed, total, ticker, action
    finished = pyqtSignal(list)                  # list of ScanResult

    def __init__(self, tickers, period, interval, risk_manager):
        super().__init__()
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.risk_manager = risk_manager

    def run(self):
        def on_progress(completed, total, ticker, result):
            action = result.action if result else "..."
            self.progress.emit(completed, total, ticker, action)

        results = scan_multiple(
            self.tickers, self.period, self.interval,
            self.risk_manager, max_workers=3,
            progress_callback=on_progress,
        )
        self.finished.emit(results)


# ─── Main Application ───────────────────────────────────────────────────────
class AutoTraderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stocky AutoTrader — Multi-Stock Portfolio Manager")
        self.setGeometry(50, 50, 1400, 900)
        self.setStyleSheet(APP_STYLESHEET)

        self.risk_manager = RiskManager()
        self.broker = None
        self.scan_results = []
        self.selected_tickers = set()
        self._init_broker()

        self._build_ui()
        log_event("startup", "AutoTrader launched")

    def _init_broker(self):
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
            key = settings.get("alpaca_api_key", "")
            secret = settings.get("alpaca_secret_key", "")
            if key and secret:
                self.broker = AlpacaBroker(key, secret)
        except Exception:
            pass

    def _build_ui(self):
        layout = QVBoxLayout()

        # ── Top bar: controls ──
        top = QGroupBox("Scanner Controls")
        top_layout = QGridLayout()

        # Ticker input
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("AAPL, TSLA, NVDA... (or use presets below)")
        top_layout.addWidget(QLabel("Tickers:"), 0, 0)
        top_layout.addWidget(self.ticker_input, 0, 1, 1, 3)

        # Preset buttons
        preset_layout = QHBoxLayout()
        for name, tickers in [("Top 24", DEFAULT_TICKERS), ("Tech", TECH_TICKERS), ("ETFs", ETF_TICKERS)]:
            btn = QPushButton(name)
            btn.setStyleSheet("font-size: 11px; padding: 6px;")
            btn.clicked.connect(lambda _, t=tickers: self.ticker_input.setText(", ".join(t)))
            preset_layout.addWidget(btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("font-size: 11px; padding: 6px;")
        clear_btn.clicked.connect(lambda: self.ticker_input.clear())
        preset_layout.addWidget(clear_btn)
        top_layout.addLayout(preset_layout, 1, 0, 1, 4)

        # Period / Interval
        self.period_combo = QComboBox()
        self.period_combo.addItems(["5d", "3d", "2d", "1d"])
        top_layout.addWidget(QLabel("Data:"), 2, 0)
        top_layout.addWidget(self.period_combo, 2, 1)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["5m", "1m", "15m", "30m"])
        top_layout.addWidget(QLabel("Interval:"), 2, 2)
        top_layout.addWidget(self.interval_combo, 2, 3)

        # Scan button
        self.scan_btn = QPushButton("SCAN & RANK")
        self.scan_btn.setStyleSheet("font-size: 16px; padding: 12px; background-color: #006600;")
        self.scan_btn.clicked.connect(self._start_scan)
        top_layout.addWidget(self.scan_btn, 3, 0, 1, 4)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        top_layout.addWidget(self.progress_bar, 4, 0, 1, 3)
        self.progress_label = QLabel("")
        top_layout.addWidget(self.progress_label, 4, 3)

        top.setLayout(top_layout)
        layout.addWidget(top)

        # ── Tabs: Results / Log Viewer ──
        self.tabs = QTabWidget()

        # --- Results tab ---
        results_widget = QWidget()
        results_layout = QVBoxLayout()

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            "", "Ticker", "Signal", "Confidence", "Price",
            "Shares", "SL / TP", "Score", "Reasoning"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(8, QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        results_layout.addWidget(self.results_table)

        # Action buttons
        action_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All BUY/SELL")
        self.select_all_btn.clicked.connect(self._select_all_signals)
        action_layout.addWidget(self.select_all_btn)

        self.deselect_btn = QPushButton("Deselect All")
        self.deselect_btn.clicked.connect(self._deselect_all)
        action_layout.addWidget(self.deselect_btn)

        self.invest_btn = QPushButton("AUTO-INVEST SELECTED")
        self.invest_btn.setStyleSheet("background-color: #006600; font-size: 14px; padding: 10px;")
        self.invest_btn.clicked.connect(self._auto_invest)
        action_layout.addWidget(self.invest_btn)

        results_layout.addLayout(action_layout)

        # Summary label
        self.summary_label = QLabel("")
        self.summary_label.setFont(QFont("Consolas", 11))
        self.summary_label.setStyleSheet("color: #00ccff; padding: 5px;")
        results_layout.addWidget(self.summary_label)

        results_widget.setLayout(results_layout)
        self.tabs.addTab(results_widget, "Scan Results")

        # --- Log Viewer tab ---
        log_widget = QWidget()
        log_layout = QVBoxLayout()

        # Log file selector
        log_top = QHBoxLayout()
        self.log_file_combo = QComboBox()
        self.log_file_combo.currentTextChanged.connect(self._load_log_file)
        log_top.addWidget(QLabel("Log File:"))
        log_top.addWidget(self.log_file_combo, 1)
        refresh_log_btn = QPushButton("Refresh")
        refresh_log_btn.setStyleSheet("font-size: 11px; padding: 6px;")
        refresh_log_btn.clicked.connect(self._refresh_log_files)
        log_top.addWidget(refresh_log_btn)
        log_layout.addLayout(log_top)

        # Log content
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.log_viewer)

        log_widget.setLayout(log_layout)
        self.tabs.addTab(log_widget, "Decision Logs")

        # --- Live Activity tab ---
        activity_widget = QWidget()
        activity_layout = QVBoxLayout()
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        activity_layout.addWidget(self.activity_log)
        activity_widget.setLayout(activity_layout)
        self.tabs.addTab(activity_widget, "Live Activity")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

        # Load log files on startup
        self._refresh_log_files()

    # ── Logging helper ────────────────────────────────────────────────────

    def _log(self, msg, level="info"):
        self.activity_log.append(log_html(msg, level))

    # ── Scanning ──────────────────────────────────────────────────────────

    def _start_scan(self):
        # Parse tickers from input
        text = self.ticker_input.text().strip()
        if not text:
            self._log("Enter tickers or use a preset.", "warn")
            return

        tickers = [t.strip().upper() for t in text.replace(";", ",").split(",") if t.strip()]
        if not tickers:
            self._log("No valid tickers.", "warn")
            return

        self._log(f"Starting scan of {len(tickers)} tickers...", "info")
        log_event("scan_start", f"Scanning {len(tickers)} tickers", {"tickers": tickers})

        self.scan_btn.setEnabled(False)
        self.invest_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(tickers))
        self.progress_bar.setValue(0)
        self.scan_results = []
        self.selected_tickers.clear()
        self._scan_start_time = time.time()

        self._worker = ScanWorker(
            tickers, self.period_combo.currentText(),
            self.interval_combo.currentText(), self.risk_manager,
        )
        self._worker.progress.connect(self._on_scan_progress)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    def _on_scan_progress(self, completed, total, ticker, action):
        self.progress_bar.setValue(completed)
        colors = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}
        color = colors.get(action, "white")
        self.progress_label.setText(f"{completed}/{total} — {ticker}: {action}")
        self._log(f"Scanned {ticker}: {action}", "trade" if action != "HOLD" else "info")

    def _on_scan_finished(self, results):
        self.scan_results = results
        elapsed = time.time() - self._scan_start_time
        self.progress_bar.setVisible(False)
        self.scan_btn.setEnabled(True)
        self.invest_btn.setEnabled(True)

        self._log(f"Scan complete: {len(results)} tickers in {elapsed:.1f}s", "info")

        # Log scan summary
        summary = [{"ticker": r.ticker, "action": r.action, "confidence": r.confidence,
                     "score": r.score} for r in results]
        log_scan_results(len(results), summary, elapsed)

        # Log each decision with full reasoning
        for r in results:
            if r.error:
                continue
            try:
                from addons import get_active_addons
                active = [a.name for a in get_active_addons()]
            except Exception:
                active = []

            log_decision(
                ticker=r.ticker, action=r.action, confidence=r.confidence,
                price=r.price, position_size=r.position_size,
                stop_loss=r.stop_loss, take_profit=r.take_profit,
                atr=r.atr, probs=r.probs,
                feature_importances=r.feature_importances,
                active_addons=active, reasoning=r.reasoning,
            )

        # Populate results table
        self._populate_results(results)

        # Summary stats
        buys = sum(1 for r in results if r.action == "BUY")
        sells = sum(1 for r in results if r.action == "SELL")
        holds = sum(1 for r in results if r.action == "HOLD")
        errors = sum(1 for r in results if r.error)
        self.summary_label.setText(
            f"Results: {buys} BUY | {sells} SELL | {holds} HOLD | "
            f"{errors} errors | Scanned in {elapsed:.1f}s"
        )

        # Refresh log viewer
        self._refresh_log_files()

    def _populate_results(self, results):
        self.results_table.setRowCount(len(results))

        for i, r in enumerate(results):
            # Checkbox
            cb = QCheckBox()
            cb.setChecked(False)
            cb.toggled.connect(lambda checked, t=r.ticker: self._toggle_ticker(t, checked))
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.results_table.setCellWidget(i, 0, cb_widget)

            # Ticker
            ticker_item = QTableWidgetItem(r.ticker)
            ticker_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            ticker_item.setFont(QFont("Consolas", 11, QFont.Bold))
            self.results_table.setItem(i, 1, ticker_item)

            # Signal
            signal_item = QTableWidgetItem(r.action)
            signal_item.setFlags(Qt.ItemIsEnabled)
            colors = {"BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD}
            signal_item.setForeground(QColor(colors.get(r.action, "#666")))
            signal_item.setFont(QFont("Consolas", 11, QFont.Bold))
            self.results_table.setItem(i, 2, signal_item)

            # Confidence
            conf_item = QTableWidgetItem(f"{r.confidence:.1%}")
            conf_item.setFlags(Qt.ItemIsEnabled)
            self.results_table.setItem(i, 3, conf_item)

            # Price
            price_item = QTableWidgetItem(f"${r.price:.2f}" if r.price > 0 else "--")
            price_item.setFlags(Qt.ItemIsEnabled)
            self.results_table.setItem(i, 4, price_item)

            # Shares
            shares_item = QTableWidgetItem(str(r.position_size) if r.position_size > 0 else "--")
            shares_item.setFlags(Qt.ItemIsEnabled)
            self.results_table.setItem(i, 5, shares_item)

            # SL / TP
            if r.stop_loss > 0:
                sltp = f"${r.stop_loss:.2f} / ${r.take_profit:.2f}"
            else:
                sltp = "--"
            sltp_item = QTableWidgetItem(sltp)
            sltp_item.setFlags(Qt.ItemIsEnabled)
            self.results_table.setItem(i, 6, sltp_item)

            # Score
            score_item = QTableWidgetItem(f"{r.score:.2f}")
            score_item.setFlags(Qt.ItemIsEnabled)
            self.results_table.setItem(i, 7, score_item)

            # Reasoning
            reason_item = QTableWidgetItem(r.reasoning[:120] if r.reasoning else r.error or "")
            reason_item.setFlags(Qt.ItemIsEnabled)
            reason_item.setToolTip(r.reasoning)
            self.results_table.setItem(i, 8, reason_item)

        self.results_table.resizeRowsToContents()

    # ── Selection ─────────────────────────────────────────────────────────

    def _toggle_ticker(self, ticker, checked):
        if checked:
            self.selected_tickers.add(ticker)
        else:
            self.selected_tickers.discard(ticker)

    def _select_all_signals(self):
        """Select all tickers with BUY or SELL signals."""
        self.selected_tickers.clear()
        for i, r in enumerate(self.scan_results):
            is_signal = r.action in ("BUY", "SELL") and r.confidence > 0.4
            cb_widget = self.results_table.cellWidget(i, 0)
            if cb_widget:
                cb = cb_widget.findChild(QCheckBox)
                if cb:
                    cb.setChecked(is_signal)

    def _deselect_all(self):
        self.selected_tickers.clear()
        for i in range(self.results_table.rowCount()):
            cb_widget = self.results_table.cellWidget(i, 0)
            if cb_widget:
                cb = cb_widget.findChild(QCheckBox)
                if cb:
                    cb.setChecked(False)

    # ── Auto-invest ───────────────────────────────────────────────────────

    def _auto_invest(self):
        if not self.selected_tickers:
            self._log("No tickers selected.", "warn")
            return

        if not self.broker:
            self._log("Alpaca API not configured. Set keys in Settings.", "error")
            return

        selected = [r for r in self.scan_results if r.ticker in self.selected_tickers]
        actionable = [r for r in selected if r.action in ("BUY", "SELL") and r.position_size > 0]

        if not actionable:
            self._log("No actionable signals in selection (all HOLD or 0 shares).", "warn")
            return

        self._log(f"Auto-investing {len(actionable)} positions...", "trade")
        log_event("auto_invest", f"Executing {len(actionable)} trades",
                  {"tickers": [r.ticker for r in actionable]})

        for r in actionable:
            side = "buy" if r.action == "BUY" else "sell"
            self._log(
                f"  {r.action} {r.ticker}: {r.position_size} shares @ ${r.price:.2f} "
                f"(SL ${r.stop_loss:.2f}, TP ${r.take_profit:.2f})",
                "trade",
            )

            # Write signal for StockExecuter compatibility
            write_signal(r.ticker, r.action, r.confidence, r.price,
                        r.position_size, r.stop_loss, r.take_profit, r.atr)

            # Execute via Alpaca
            result = self.broker.place_order(
                r.ticker, r.position_size, side,
                stop_loss=r.stop_loss, take_profit=r.take_profit,
            )

            if "error" in result:
                self._log(f"  FAILED: {result['error']}", "error")
                log_trade_execution(r.ticker, side, r.position_size, "market",
                                   "failed", error=result["error"])
            else:
                order_id = result.get("id", "?")
                self._log(f"  Order placed: {order_id}", "trade")
                log_trade_execution(r.ticker, side, r.position_size, "market", order_id)

    # ── Log Viewer ────────────────────────────────────────────────────────

    def _refresh_log_files(self):
        self.log_file_combo.clear()
        files = get_log_files()
        for f in files:
            self.log_file_combo.addItem(f"{f['date']} ({f['size_kb']:.0f} KB)", f["file"])

    def _load_log_file(self, display_text):
        idx = self.log_file_combo.currentIndex()
        if idx < 0:
            return

        filename = self.log_file_combo.itemData(idx)
        if not filename:
            return

        entries = get_log_entries(filename, max_entries=300)
        self.log_viewer.clear()

        for entry in entries:
            ts = entry.get("timestamp", "")[:19]
            etype = entry.get("type", "?")
            color = {
                "decision": "#00ccff",
                "execution": "#00ff88",
                "scan": "#ffaa00",
                "event": "#b0bec5",
            }.get(etype, "#666")

            if etype == "decision":
                action = entry.get("action", "?")
                ticker = entry.get("ticker", "?")
                conf = entry.get("confidence", 0)
                reasoning = entry.get("reasoning", "")
                action_color = {"BUY": "#00ff88", "SELL": "#ff4444", "HOLD": "#ffaa00"}.get(action, "#666")

                self.log_viewer.append(
                    f'<span style="color:#666">{ts}</span> '
                    f'<span style="color:{action_color}; font-weight:bold">{action} {ticker}</span> '
                    f'<span style="color:#888">({conf:.1%})</span>'
                )
                if reasoning:
                    self.log_viewer.append(
                        f'<span style="color:#555; font-size:10px">  {reasoning}</span>'
                    )

                # Show feature importances if present
                imps = entry.get("feature_importances", {})
                if imps:
                    imp_str = ", ".join([f"{k}: {v:.0f}" for k, v in list(imps.items())[:5]])
                    self.log_viewer.append(
                        f'<span style="color:#444; font-size:10px">  Key features: {imp_str}</span>'
                    )

            elif etype == "execution":
                side = entry.get("side", "?")
                ticker = entry.get("ticker", "?")
                qty = entry.get("qty", 0)
                status = entry.get("status", "?")
                error = entry.get("error", "")
                exec_color = "#00ff88" if not error else "#ff4444"
                self.log_viewer.append(
                    f'<span style="color:#666">{ts}</span> '
                    f'<span style="color:{exec_color}">{side.upper()} {ticker} x{qty} [{status}]</span>'
                    + (f' <span style="color:#ff4444">{error}</span>' if error else "")
                )

            elif etype == "scan":
                count = entry.get("tickers_scanned", 0)
                duration = entry.get("duration_seconds", 0)
                self.log_viewer.append(
                    f'<span style="color:#666">{ts}</span> '
                    f'<span style="color:#ffaa00">SCAN: {count} tickers in {duration:.1f}s</span>'
                )

            else:
                msg = entry.get("message", "")
                self.log_viewer.append(
                    f'<span style="color:#666">{ts}</span> '
                    f'<span style="color:{color}">{msg}</span>'
                )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoTraderApp()
    window.show()
    sys.exit(app.exec_())
