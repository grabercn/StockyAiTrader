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
from panels.trade import TradePanel
from panels.ai_dashboard import AIDashboardPanel
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
            # Dashboard data is preloaded during splash — only refresh if needed later
            QTimer.singleShot(3000, self.dashboard.refresh)

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
        """Only warn about addons the current profile expects but are missing."""
        try:
            from addons import get_all_addons
            from core.profiles import PRESETS, get_active_profile_name

            profile_name = get_active_profile_name()
            profile = PRESETS.get(profile_name, {})
            profile_addons = profile.get("addons", {})

            # Only check addons that the profile has enabled
            all_addons = get_all_addons()
            needs_install = []
            needs_key = []
            settings = load_settings()

            for a in all_addons:
                wanted = profile_addons.get(a.module_name, False)
                if not wanted:
                    continue  # Profile doesn't use this addon — no warning

                if not a.available:
                    needs_install.append(a.name)
                elif a.requires_api_key and not settings.get(a.api_key_name):
                    needs_key.append(a.name)

            if needs_install:
                names = ", ".join(needs_install[:3])
                self.event_bus.log_entry.emit(
                    f"{len(needs_install)} addons need install ({names}) — Settings > Addons",
                    "warn",
                )
            if needs_key:
                names = ", ".join(needs_key[:3])
                self.event_bus.log_entry.emit(
                    f"{len(needs_key)} addons need API keys ({names}) — Settings > API Keys",
                    "warn",
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

    def _save_agent_state(self):
        """Save COMPLETE agent state for seamless resume."""
        settings = load_settings()

        # Agent running state
        ai_agent = getattr(self, 'ai_agent', None)
        agent_was_running = ai_agent and getattr(ai_agent, '_agent_running', False)
        settings["agent_was_running"] = agent_was_running

        # Save agent's own tracked stocks with full metadata
        if ai_agent and hasattr(ai_agent, '_agent_stocks'):
            settings["agent_tracked_stocks"] = dict(ai_agent._agent_stocks)

        # Save auto-trader monitored stocks
        monitored = {}
        # Check scanner's service
        scanner = getattr(self, 'scanner', None)
        if scanner and hasattr(scanner, '_auto_service') and scanner._auto_service:
            for ticker, stock in scanner._auto_service.get_monitored().items():
                monitored[ticker] = {
                    "period": stock.period, "interval": stock.interval,
                    "last_signal": stock.last_signal,
                    "last_confidence": stock.last_confidence,
                    "last_price": stock.last_price,
                    "last_check": stock.last_check,
                    "check_count": stock.check_count,
                    "auto_execute": stock.auto_execute,
                }
        # Also check AI dashboard's service
        if ai_agent and hasattr(ai_agent, '_auto_svc') and ai_agent._auto_svc:
            for ticker, stock in ai_agent._auto_svc.get_monitored().items():
                if ticker not in monitored:
                    monitored[ticker] = {
                        "period": stock.period, "interval": stock.interval,
                        "last_signal": stock.last_signal,
                        "last_confidence": stock.last_confidence,
                        "last_price": stock.last_price,
                        "last_check": stock.last_check,
                        "check_count": stock.check_count,
                        "auto_execute": stock.auto_execute,
                    }
        settings["monitored_stocks"] = monitored

        # Save ALL current positions with full detail
        if self.broker:
            try:
                positions = self.broker.get_positions()
                if isinstance(positions, list):
                    managed = []
                    for p in positions:
                        managed.append({
                            "symbol": p.get("symbol", ""),
                            "qty": p.get("qty", "0"),
                            "side": p.get("side", ""),
                            "avg_entry": p.get("avg_entry_price", "0"),
                            "current_price": p.get("current_price", "0"),
                            "unrealized_pl": p.get("unrealized_pl", "0"),
                            "market_value": p.get("market_value", "0"),
                        })
                    settings["agent_managed_positions"] = managed
            except Exception:
                pass

        # Save agent cycle metadata
        if ai_agent:
            settings["agent_cycle_count"] = getattr(ai_agent, '_last_cycle', 0)
            settings["agent_trades_today"] = getattr(ai_agent, '_last_trades_today', 0)

        save_settings(settings)

    def closeEvent(self, event):
        """Save state, then minimize to tray or quit."""
        self._save_agent_state()
        settings = load_settings()
        quit_on_close = settings.get("quit_on_close", False)

        if quit_on_close or not (self._tray and self._tray.tray and self._tray.tray.isVisible()):
            if self._tray and self._tray.tray:
                self._tray.tray.hide()
            event.accept()
        else:
            # Reverse particle animation — edges collapse to center
            event.ignore()
            from core.ui.window_collapse import WindowCollapse
            snapshot = self.grab()
            geo = self.geometry()
            self.hide()

            def _on_collapse_done():
                self._tray.send_notification(
                    "Stocky Suite",
                    "Running in background. Double-click tray icon to restore.",
                    "info",
                )

            overlay = WindowCollapse(snapshot, geo, on_done=_on_collapse_done)
            overlay.start()
            self._close_overlay = overlay

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

        # Lazy-load panels — only Dashboard builds on boot, others on first click
        from core.ui.icons import StockyIcons

        self._panel_factories = [
            ("Dashboard",    "dashboard", lambda: DashboardPanel(self.broker, self.event_bus)),
            ("AI Agent",     "robot",     lambda: AIDashboardPanel(self.broker, self.event_bus)),
            ("Scanner",      "scan",      lambda: ScannerPanel(self.broker, self.risk_manager, self.event_bus)),
            ("Portfolio",    "wallet",    lambda: PortfolioPanel(self.broker, self.event_bus)),
            ("Trade",        "bolt",      lambda: TradePanel(self.broker, self.risk_manager, self.event_bus)),
            ("Logs",         "log",       lambda: LogsPanel(self.event_bus)),
            ("Tax Reports",  "tax",       lambda: TaxPanel(self.broker, self.event_bus)),
            ("Testing",      "test",      lambda: TestingPanel(self.broker, self.event_bus)),
            ("Settings",     "settings",  lambda: SettingsPanel(self.event_bus)),
        ]
        self._loaded_tabs = {}

        for i, (tab_name, icon_key, factory) in enumerate(self._panel_factories):
            icon = StockyIcons.get_icon(icon_key, 22, BRAND_PRIMARY)
            if i == 0:
                # Build Dashboard immediately
                try:
                    panel = factory()
                    self.tabs.addTab(panel, icon, tab_name)
                    attr = tab_name.lower().replace(" ", "_")
                    setattr(self, attr, panel)
                    self._loaded_tabs[i] = True
                except Exception as e:
                    self._add_error_tab(tab_name, e)
                    self._loaded_tabs[i] = True
            else:
                # Placeholder — builds on first click
                placeholder = QWidget()
                pl = QVBoxLayout()
                loading_lbl = QLabel(f"Loading {tab_name}...")
                loading_lbl.setAlignment(Qt.AlignCenter)
                loading_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 14px;")
                pl.addWidget(loading_lbl)
                placeholder.setLayout(pl)
                self.tabs.addTab(placeholder, icon, tab_name)

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

    def _on_tab_changed(self, index):
        """Lazy-load panel on first tab click."""
        if index in self._loaded_tabs:
            return
        self._loaded_tabs[index] = True

        if index >= len(self._panel_factories):
            return

        tab_name, icon_key, factory = self._panel_factories[index]
        try:
            panel = factory()
            self.tabs.removeTab(index)
            from core.ui.icons import StockyIcons
            icon = StockyIcons.get_icon(icon_key, 22, BRAND_PRIMARY)
            self.tabs.insertTab(index, panel, icon, tab_name)
            self.tabs.setCurrentIndex(index)
            attr = tab_name.lower().replace(" ", "_")
            setattr(self, attr, panel)
        except Exception as e:
            self._add_error_tab(tab_name, e, index)

    def _add_error_tab(self, tab_name, error, index=None):
        error_panel = QWidget()
        error_layout = QVBoxLayout()
        error_lbl = QLabel(f"Failed to load {tab_name}:\n{error}")
        error_lbl.setStyleSheet(f"color: {COLOR_SELL}; padding: 20px;")
        error_lbl.setWordWrap(True)
        error_layout.addWidget(error_lbl)
        error_panel.setLayout(error_layout)
        if index is not None:
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, error_panel, f"{tab_name} (!)")
        else:
            self.tabs.addTab(error_panel, f"{tab_name} (!)")
        log_event("panel_error", f"{tab_name} failed to load: {error}")
        print(f"[ERROR] Panel '{tab_name}' failed: {error}", flush=True)

        # Reconnect broker on settings change
        self.event_bus.settings_changed.connect(self._on_settings_changed)

    def _on_settings_changed(self, settings):
        key = settings.get("alpaca_api_key", "")
        secret = settings.get("alpaca_secret_key", "")
        if key and secret:
            self.broker = AlpacaBroker(key, secret)
            # Update broker on all panels that have one
            for attr in ("dashboard", "scanner", "trade", "portfolio", "ai_agent",
                         "tax_panel", "testing_panel"):
                p = getattr(self, attr, None)
                if p and hasattr(p, "broker"):
                    p.broker = self.broker
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
            f"186+ tests  |  38 ML features  |  {len(get_all_addons())} addons  |  RL feedback"
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

    # Preload heavy imports in background while splash animates
    import threading
    _preload_done = threading.Event()
    _preload_data = {}

    def _preload():
        """Load heavy modules and data while splash is showing."""
        try:
            # Pre-import heavy libraries so they're cached
            import lightgbm  # noqa
            import pandas  # noqa
            import numpy  # noqa
            import yfinance  # noqa
            _preload_data["modules"] = True
        except Exception:
            pass
        try:
            # Pre-fetch broker data so dashboard loads instantly
            settings = load_settings()
            key = settings.get("alpaca_api_key", "")
            secret = settings.get("alpaca_secret_key", "")
            if key and secret:
                from core.broker import AlpacaBroker
                broker = AlpacaBroker(key, secret)
                _preload_data["acct"] = broker.get_account()
                _preload_data["positions"] = broker.get_positions()
                _preload_data["orders"] = broker.get_orders("open")
                _preload_data["hist"] = broker.get_portfolio_history(period="1W", timeframe="1H")
        except Exception:
            pass
        _preload_done.set()

    preload_thread = threading.Thread(target=_preload, daemon=True)
    preload_thread.start()

    step(15, "Preloading data...",            "Fetching broker data in background")

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

    step(80, "Building interface...",         "9 panels · event bus · signal routing")
    suite = StockySuite()

    # Wait for preload but keep splash responsive
    step(88, "Waiting for data...", "Almost ready")
    for _ in range(50):  # Up to 5 seconds, non-blocking
        if _preload_done.is_set():
            break
        time.sleep(0.1)
        app.processEvents()

    # Inject preloaded data into dashboard (async — don't block)
    def _inject_preload():
        if _preload_data.get("acct") and hasattr(suite, 'dashboard'):
            try:
                suite.dashboard._apply_refresh(
                    _preload_data.get("acct", {}),
                    _preload_data.get("positions", []),
                    _preload_data.get("orders", []),
                    _preload_data.get("hist", {}),
                )
            except Exception:
                pass
    QTimer.singleShot(500, _inject_preload)

    step(90, "Loading log history...",        "Decision logs · trade history")

    # ── Menu Bar ──
    from core.ui.icons import StockyIcons
    from core.ui.setup_wizard import needs_setup, SetupWizard

    view_menu = suite.menuBar().addMenu("View")
    for i, (name, icon_key) in enumerate([
        ("Dashboard", "dashboard"), ("AI Agent", "robot"), ("Scanner", "scan"),
        ("Portfolio", "wallet"), ("Trade", "bolt"), ("Logs", "log"),
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
        for attr in ("dashboard", "scanner", "portfolio", "trade", "ai_agent", "tax_reports", "testing"):
            p = getattr(suite, attr, None)
            if p and hasattr(p, "broker"):
                p.broker = suite.broker
    else:
        step(100, "Ready.", f"{APP_NAME} v{APP_VERSION}")
        time.sleep(0.5)

    # Dissolve boot screen into particles
    boot.finish()

    # Let boot dissolve animate
    import time as _time
    for _ in range(40):
        _time.sleep(0.03)
        app.processEvents()

    print("[BOOT] Dissolve done, starting reveal...", flush=True)

    # Particle convergence reveal
    from core.ui.window_reveal import WindowReveal

    print("[BOOT] Preparing main window...", flush=True)
    effect = QGraphicsOpacityEffect(suite)
    suite.setGraphicsEffect(effect)
    effect.setOpacity(0.0)
    suite.show()

    app.processEvents()
    for _ in range(10):
        _time.sleep(0.05)
        app.processEvents()
    print("[BOOT] Window shown, starting fade-in...", flush=True)

    def _fade_in_app():
        print("[BOOT] Fading in app...", flush=True)
        fade_in = QPropertyAnimation(effect, b"opacity")
        fade_in.setDuration(600)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.OutCubic)
        fade_in.finished.connect(lambda: suite.setGraphicsEffect(None))
        fade_in.finished.connect(lambda: QTimer.singleShot(1000, _check_agent_resume))
        fade_in.start()
        suite._entrance_anim = fade_in

    def _check_agent_resume():
        """Ask to resume with a simple message box."""
        print("[BOOT] Checking agent resume...", flush=True)
        try:
            settings = load_settings()
            monitored = settings.get("monitored_stocks", {})
            managed_positions = settings.get("agent_managed_positions", [])

            if not monitored and not managed_positions:
                return

            # Simple, safe message box
            stock_list = ", ".join(list(monitored.keys())[:8])
            if len(monitored) > 8:
                stock_list += f" +{len(monitored)-8} more"

            msg = f"Resume AI Trading?\n\n"
            msg += f"Monitoring: {len(monitored)} stocks\n"
            msg += f"({stock_list})\n"
            if managed_positions:
                total_pl = sum(float(p.get("unrealized_pl", 0)) for p in managed_positions)
                msg += f"\nPositions: {len(managed_positions)} stocks (P&L: ${total_pl:+,.2f})\n"
            msg += f"\nResume = Start AI agent\nNo = Manual management"

            reply = QMessageBox.question(suite, "Resume AI Trading", msg,
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                suite.event_bus.log_entry.emit(
                    f"Resuming AI — {len(monitored)} stocks", "trade")
                for ticker in monitored:
                    suite.event_bus.log_entry.emit(f"Restored {ticker}: will check shortly", "system")
                suite.event_bus.log_entry.emit("Starting autonomous agent...", "system")
                suite.tabs.setCurrentIndex(1)
                def _start_agent_delayed():
                    ai = getattr(suite, 'ai_agent', None)
                    if ai and hasattr(ai, '_toggle_agent'):
                        ai._toggle_agent()
                QTimer.singleShot(1500, _start_agent_delayed)
            else:
                settings["agent_was_running"] = False
                settings["agent_managed_positions"] = []
                save_settings(settings)
                suite.event_bus.log_entry.emit("Manual mode — stocks kept, AI not started", "system")
        except Exception as e:
            print(f"[RESUME ERROR] {e}", flush=True)

    # Start the reveal animation, fall back to direct fade-in
    try:
        reveal = WindowReveal(suite, on_done=_fade_in_app)
        reveal.start()
        suite._reveal = reveal
    except Exception:
        print("[BOOT] Reveal failed, direct fade-in", flush=True)
        _fade_in_app()

    suite.raise_()
    suite.activateWindow()

    sys.exit(app.exec_())


if __name__ == "__main__":
    boot_app()
