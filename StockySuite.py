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
    QSplashScreen, QGraphicsDropShadowEffect, QGraphicsOpacityEffect,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
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


# ═════════════════════════════════════════════════════════════════════════════
# PANELS — each extracted to StockyApps/panels/ for modularity
# ═════════════════════════════════════════════════════════════════════════════
from panels.workers import ScanWorker, TrainWorker, DownloadWorker, _DeepAnalyzeWorker
from panels.dashboard import DashboardPanel
from panels.scanner import ScannerPanel
from panels.day_trade import DayTradePanel
from panels.long_trade import LongTradePanel
from panels.logs import LogsPanel
from panels.portfolio import PortfolioPanel
from panels.settings_panel import SettingsPanel
from panels.tax import TaxPanel
from panels.testing import TestingPanel
from panels.notification_bar import _NotificationBar


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

        # Forward signal updates to tray for live monitoring display
        self.event_bus.signal_generated.connect(
            lambda t, a, d: self._tray.update_stock(t, a, d.get("conf", 0), d.get("price", 0), 0)
        )

        log_event("startup", f"{APP_NAME} v{APP_VERSION} launched")

        # Warn if addons are disabled by profile
        QTimer.singleShot(2000, self._check_profile_warnings)
        QTimer.singleShot(5000, self._check_for_updates)

    def _check_for_updates(self):
        """Check GitHub for newer version in background."""
        import threading
        def _check():
            try:
                from core.updater import check_for_update
                has_update, latest, download_url, release_url = check_for_update()
                if has_update:
                    url = download_url or release_url
                    self._update_url = url
                    # Show message and set click URL atomically via main thread
                    QTimer.singleShot(0, lambda: self._show_update_notification(latest, url))
                    log_event("update", f"New version v{latest} available at {url}")

                    # Also send a toast notification with the link
                    if hasattr(self, '_tray') and self._tray:
                        self._tray.send_notification(
                            f"Stocky Suite v{latest} Available",
                            f"Current: v{APP_VERSION} → New: v{latest}\nClick to download.",
                            "info",
                        )
            except Exception:
                pass
        threading.Thread(target=_check, daemon=True).start()

    def _show_update_notification(self, latest, url):
        """Show update notification on main thread with clickable URL."""
        if hasattr(self, '_notif_bar'):
            self._notif_bar.show_message(
                f"Update available: v{APP_VERSION} → v{latest} — click to download",
                "warn",
                click_url=url,
            )
        log_event("update", f"New version v{latest} available at {url}")

    def _check_profile_warnings(self):
        """Warn user if their profiles have issues."""
        try:
            # Check addon coverage
            from addons import get_all_addons
            all_addons = get_all_addons()
            inactive = [a for a in all_addons if a.available and not a.enabled]
            unavailable = [a for a in all_addons if not a.available]

            if inactive:
                names = ", ".join(a.name for a in inactive[:3])
                extra = f" +{len(inactive)-3} more" if len(inactive) > 3 else ""
                self.event_bus.log_entry.emit(
                    f"{len(inactive)} addons disabled ({names}{extra}) — "
                    f"check Settings > Hardware Profile",
                    "warn",
                )
            if unavailable:
                names = ", ".join(a.name for a in unavailable[:2])
                self.event_bus.log_entry.emit(
                    f"{len(unavailable)} addons need install ({names}) — "
                    f"Settings > Addons",
                    "system",
                )

            # Check aggressivity + hardware compatibility
            from core.intelligent_trader import check_profile_compatibility
            from core.profiles import get_active_profile_name
            settings = load_settings()
            aggr = settings.get("aggressivity", "Default")
            hw = get_active_profile_name()
            compatible, warnings = check_profile_compatibility(aggr, hw)
            for w in warnings:
                self.event_bus.log_entry.emit(w, "warn")
        except Exception:
            pass

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
                return 0.9   # Qt already handles DPI scaling — 90% is the sweet spot
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
    # Set Windows AppUserModelID so taskbar shows our icon instead of Python's
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("grabercn.stockysuite.4")
    except Exception:
        pass

    # Enable Qt high DPI scaling BEFORE QApplication is created
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

    # Dissolve boot screen into particles, then show main window with fade-in
    boot.finish()

    # Give particles time to animate before showing main window
    import time as _time
    for _ in range(30):
        _time.sleep(0.03)
        app.processEvents()

    # Main window fade-in
    effect = QGraphicsOpacityEffect(suite)
    suite.setGraphicsEffect(effect)
    effect.setOpacity(0.0)
    suite.show()
    suite.raise_()
    suite.activateWindow()

    fade_in = QPropertyAnimation(effect, b"opacity")
    fade_in.setDuration(600)
    fade_in.setStartValue(0.0)
    fade_in.setEndValue(1.0)
    fade_in.setEasingCurve(QEasingCurve.OutCubic)
    fade_in.finished.connect(lambda: suite.setGraphicsEffect(None))  # Remove effect after done
    fade_in.start()
    suite._entrance_anim = fade_in  # prevent GC

    sys.exit(app.exec_())


if __name__ == "__main__":
    boot_app()
