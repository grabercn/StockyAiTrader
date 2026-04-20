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

