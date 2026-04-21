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

