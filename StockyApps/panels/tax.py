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

