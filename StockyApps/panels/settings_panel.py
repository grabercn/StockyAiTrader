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

        # Zoom slider — 90% real = 100% display (offset by 10)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("UI Scale:"))
        self.zoom_slider = QSpinBox()
        self.zoom_slider.setRange(70, 200)
        self.zoom_slider.setSuffix("%")
        self.zoom_slider.setSingleStep(5)
        real_zoom = settings.get("ui_zoom", 0.9)
        self.zoom_slider.setValue(int(real_zoom * 100))
        self.zoom_slider.valueChanged.connect(self._change_zoom)
        zoom_row.addWidget(self.zoom_slider)
        al2.addLayout(zoom_row)

        # Close behavior
        close_row = QHBoxLayout()
        close_row.addWidget(QLabel("On close:"))
        self.close_combo = QComboBox()
        self.close_combo.addItems(["Minimize to tray (recommended)", "Quit application"])
        self.close_combo.setCurrentIndex(0 if not settings.get("quit_on_close", False) else 1)
        self.close_combo.currentIndexChanged.connect(self._change_close_behavior)
        close_row.addWidget(self.close_combo, 1)
        al2.addLayout(close_row)

        self.close_note = QLabel("")
        self.close_note.setWordWrap(True)
        self.close_note.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")
        self._update_close_note()
        al2.addWidget(self.close_note)

        appear_box.setLayout(al2)
        inner_layout.addWidget(appear_box)

        # API Keys — with placeholder formats, info icons, signup links
        keys_box = QGroupBox("API Keys")
        kl = QVBoxLayout()

        # Key definitions: (settings_key, label, placeholder, signup_url, instructions)
        key_defs = [
            ("alpaca_api_key", "Alpaca API Key", "PKXXXXXXXXXXXXXXXXXX",
             "https://app.alpaca.markets/signup",
             "1. Go to alpaca.markets and create a free account\n"
             "2. Click 'Paper Trading' in the left sidebar\n"
             "3. Click 'View' under API Keys\n"
             "4. Click 'Regenerate' to get a new key\n"
             "5. Copy the API Key ID (starts with PK...)"),
            ("alpaca_secret_key", "Alpaca Secret Key", "••••••••••••••••••••",
             "https://app.alpaca.markets/signup",
             "Same page as the API Key above.\n"
             "The Secret Key is shown once when you regenerate.\n"
             "Copy it immediately — you can't view it again."),
            ("fred_api_key", "FRED API Key", "abcdef1234567890abcdef1234567890",
             "https://fred.stlouisfed.org/docs/api/api_key.html",
             "1. Go to fred.stlouisfed.org\n"
             "2. Create a free account\n"
             "3. Go to 'My Account' > 'API Keys'\n"
             "4. Click 'Request API Key'\n"
             "5. Copy the 32-character key"),
            ("finnhub_api_key", "Finnhub API Key", "cxxxxxxxxxxxxxxxxxxxxxxxxxx",
             "https://finnhub.io/register",
             "1. Go to finnhub.io and create a free account\n"
             "2. Your API key is shown on the dashboard\n"
             "3. Copy the key (starts with 'c...')"),
            ("timegpt_api_key", "Nixtla TimeGPT Key", "nixtla-tok-XXXXXXXXXXXX",
             "https://dashboard.nixtla.io/",
             "1. Go to dashboard.nixtla.io\n"
             "2. Sign up for a free account\n"
             "3. Go to 'API Keys' tab\n"
             "4. Create a new key\n"
             "5. Copy it (starts with 'nixtla-tok-')"),
        ]

        self.inputs = {}
        for key, label, placeholder, url, instructions in key_defs:
            row = QHBoxLayout()
            lbl = QLabel(f"{label}:")
            lbl.setMinimumWidth(130)
            row.addWidget(lbl)

            inp = QLineEdit(settings.get(key, ""))
            inp.setPlaceholderText(placeholder)
            if "secret" in key.lower():
                inp.setEchoMode(QLineEdit.Password)
            row.addWidget(inp, 1)
            self.inputs[key] = inp

            # Info button — opens popup with instructions
            info_btn = QPushButton("?")
            info_btn.setFixedSize(24, 24)
            info_btn.setToolTip(f"How to get your {label}")
            info_btn.setStyleSheet(f"background-color: {BRAND_PRIMARY}; color: white; font-weight: bold; border-radius: 12px; font-size: 12px;")
            info_btn.clicked.connect(lambda _, l=label, u=url, inst=instructions: self._show_key_help(l, u, inst))
            row.addWidget(info_btn)

            kl.addLayout(row)

        # Addon API keys (dynamic)
        for addon in get_all_addons():
            if addon.requires_api_key and addon.api_key_name:
                if addon.api_key_name not in self.inputs:  # Don't duplicate
                    row = QHBoxLayout()
                    row.addWidget(QLabel(f"{addon.name}:"))
                    inp = QLineEdit(settings.get(addon.api_key_name, ""))
                    inp.setPlaceholderText(f"API key for {addon.name}")
                    row.addWidget(inp, 1)
                    self.inputs[addon.api_key_name] = inp
                    kl.addLayout(row)

        save_btn = QPushButton("Save All Keys")
        save_btn.setStyleSheet(f"background-color: {BRAND_PRIMARY};")
        save_btn.clicked.connect(self._save_keys)
        kl.addWidget(save_btn)
        keys_box.setLayout(kl)
        inner_layout.addWidget(keys_box)

        # Model Manager
        model_box = QGroupBox("AI Models")
        ml = QVBoxLayout()
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(5)
        self.model_table.setHorizontalHeaderLabels(["Model", "Status", "Size", "Profiles", "Action"])
        self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.model_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        for c in [1, 2, 4]:
            self.model_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
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
        self.addon_table.setColumnCount(7)
        self.addon_table.setHorizontalHeaderLabels(["On", "Addon", "Status", "Profiles", "Features", "Config", ""])
        self.addon_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.addon_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.addon_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        for c in [2, 4, 5, 6]:
            self.addon_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.addon_table.verticalHeader().setVisible(False)
        self.addon_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        al.addWidget(self.addon_table)
        addon_box.setLayout(al)
        inner_layout.addWidget(addon_box)

        inner.setLayout(inner_layout)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self._refresh()

        # Auto-refresh every 30 seconds to pick up model/addon changes
        self._auto_refresh = QTimer(self)
        self._auto_refresh.timeout.connect(self._refresh)
        self._auto_refresh.start(30000)

    def _change_close_behavior(self, index):
        quit_on_close = index == 1
        if quit_on_close:
            confirm = QMessageBox.warning(
                self, "Quit on Close",
                "Are you sure? This means:\n\n"
                "- Auto-trading will STOP when you close the window\n"
                "- Background stock monitoring will end\n"
                "- No tray icon — app fully exits\n\n"
                "Note: Stocky Suite does not run any unnecessary background\n"
                "processes. Only active auto-trade monitoring runs in the tray.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                self.close_combo.setCurrentIndex(0)
                return

        settings = load_settings()
        settings["quit_on_close"] = quit_on_close
        save_settings(settings)
        self._update_close_note()
        self.bus.log_entry.emit(
            f"Close behavior: {'quit' if quit_on_close else 'minimize to tray'}",
            "system",
        )

    def _update_close_note(self):
        if self.close_combo.currentIndex() == 0:
            self.close_note.setText(
                "App minimizes to system tray when you click X. Auto-trading and "
                "monitoring continue in the background. No unnecessary processes run."
            )
        else:
            self.close_note.setText(
                "App will fully quit when you click X. All background tasks including "
                "auto-trading and stock monitoring will stop."
            )
            self.close_note.setStyleSheet(f"color: {COLOR_HOLD}; font-size: 9px;")

    def _change_zoom(self, value):
        scale = value / 100.0
        main_window = self.window()
        if main_window and hasattr(main_window, '_scale'):
            main_window._scale = scale
            main_window._apply_scale()
            main_window._save_zoom()

    def _change_aggressivity(self, index):
        name = self.aggr_combo.currentData()
        settings = load_settings()
        settings["aggressivity"] = name
        save_settings(settings)
        self._update_aggr_desc()
        self.bus.log_entry.emit(f"Trading aggressivity set to: {name}", "system")

        # Check compatibility with current hardware profile
        from core.intelligent_trader import check_profile_compatibility
        from core.profiles import get_active_profile_name
        hw = get_active_profile_name()
        compatible, warnings = check_profile_compatibility(name, hw)
        for w in warnings:
            self.bus.log_entry.emit(w, "warn")

    def _update_aggr_desc(self):
        from core.intelligent_trader import get_aggressivity
        name = self.aggr_combo.currentData() or "Default"
        p = get_aggressivity(name)
        llm = "Yes" if p.get("use_llm") else "No"
        min_hw = p.get("min_hardware", "Minimal")
        self.aggr_desc.setText(
            f"Confidence: {p['min_confidence']:.0%}  |  "
            f"Size: {p['size_multiplier']:.1f}x  |  "
            f"Max trades: {p['max_trades_per_day']}/day  |  "
            f"Stop: {p['atr_stop_mult']:.1f}x  |  "
            f"Target: {p['atr_profit_mult']:.1f}x  |  "
            f"LLM: {llm}  |  "
            f"Needs: {min_hw}+ hardware"
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
        from core.profiles import PRESETS, get_active_profile_name
        active_profile = get_active_profile_name()

        # Model → profile mapping
        model_profiles = {
            "FinBERT": "All profiles",
            "FinBERT-Tone": "Max only",
            "Twitter-RoBERTa": "Max only",
            "DistilGPT2": "Fallback",
            "TinyLlama-Chat": "All profiles",
        }

        # Models
        self.model_table.setRowCount(len(MANAGED_MODELS))
        for i, m in enumerate(MANAGED_MODELS):
            dl, sz = get_model_status(m)

            name_item = QTableWidgetItem(m.name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.model_table.setItem(i, 0, name_item)

            s = QTableWidgetItem("Ready" if dl else "Not downloaded")
            s.setForeground(QColor(STATUS_ACTIVE if dl else STATUS_ERROR))
            s.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.model_table.setItem(i, 1, s)

            sz_item = QTableWidgetItem(sz if dl else m.size_estimate)
            sz_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.model_table.setItem(i, 2, sz_item)

            # Profile column
            prof_text = model_profiles.get(m.name, "All")
            prof_item = QTableWidgetItem(prof_text)
            prof_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            # Dim if current profile doesn't use this model
            is_used = prof_text in ("All profiles", "Fallback") or active_profile == "Max" or \
                      (prof_text == "Default+" and active_profile in ("Balanced", "Max"))
            prof_item.setForeground(QColor(STATUS_ACTIVE if is_used else TEXT_MUTED))
            if not is_used:
                prof_item.setText(f"{prof_text} (off)")
            self.model_table.setItem(i, 3, prof_item)

            if not dl:
                btn = QPushButton("Download")
                btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 10px; padding: 3px;")
                btn.clicked.connect(lambda _, mi=m: self._download(mi))
                self.model_table.setCellWidget(i, 4, btn)
            else:
                btn = QPushButton("Delete")
                btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 10px; padding: 3px;")
                btn.clicked.connect(lambda _, mi=m: self._delete_model(mi))
                self.model_table.setCellWidget(i, 4, btn)

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

            # Profile column — which profiles enable this addon
            profiles_using = []
            for pname, pdata in PRESETS.items():
                if pdata.get("addons", {}).get(a.module_name, False):
                    profiles_using.append(pname)
            if profiles_using:
                prof_str = ", ".join(profiles_using)
            else:
                prof_str = "None"
            prof_item = QTableWidgetItem(prof_str)
            prof_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            # Dim if current profile doesn't include this addon
            is_in_profile = active_profile in profiles_using
            prof_item.setForeground(QColor(STATUS_ACTIVE if is_in_profile else TEXT_MUTED))
            if not is_in_profile and profiles_using:
                prof_item.setText(f"{prof_str} (off)")
            self.addon_table.setItem(i, 3, prof_item)

            self.addon_table.setItem(i, 4, QTableWidgetItem(f"{len(a.features)}"))
            cfg = "Key set" if a.requires_api_key and settings.get(a.api_key_name) else ("Needs key" if a.requires_api_key else "—")
            c = QTableWidgetItem(cfg)
            c.setForeground(QColor(STATUS_ACTIVE if cfg == "Key set" else (STATUS_ERROR if cfg == "Needs key" else TEXT_MUTED)))
            self.addon_table.setItem(i, 5, c)

            # Install link for unavailable addons
            if not a.available and a.dependencies:
                install_link = QLabel(f'<a style="color:{BRAND_ACCENT};text-decoration:underline;cursor:pointer;">Install</a>')
                install_link.setCursor(Qt.PointingHandCursor)
                install_link.setToolTip(f"pip install {' '.join(a.dependencies)}")
                install_link.mousePressEvent = lambda _, addon=a: self._install_addon(addon)
                self.addon_table.setCellWidget(i, 6, install_link)
            elif a.available:
                ok_item = QTableWidgetItem("Ready")
                ok_item.setForeground(QColor(STATUS_ACTIVE))
                ok_item.setFlags(Qt.ItemIsEnabled)
                self.addon_table.setItem(i, 6, ok_item)

    def _show_key_help(self, label, url, instructions):
        """Show popup with instructions on how to get an API key."""
        from core.ui.theme import theme as _theme
        dlg = QDialog(self)
        dlg.setWindowTitle(f"How to get: {label}")
        dlg.setWindowIcon(QApplication.instance().windowIcon())
        dlg.setMinimumSize(450, 300)
        dlg.setStyleSheet(f"QDialog {{ background-color: {_theme.color('bg_base')}; }}")

        lay = QVBoxLayout()
        title = QLabel(f"How to get your {label}")
        title.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        title.setStyleSheet(f"color: {BRAND_PRIMARY};")
        lay.addWidget(title)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFont(QFont(FONT_MONO, 10))
        txt.setPlainText(instructions)
        lay.addWidget(txt)

        link_btn = QPushButton(f"Open Signup Page")
        link_btn.setStyleSheet(f"background-color: {BRAND_PRIMARY}; font-size: 12px; padding: 8px;")
        link_btn.clicked.connect(lambda: __import__('os').startfile(url))
        lay.addWidget(link_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)

        dlg.setLayout(lay)
        dlg.exec_()

    def _install_addon(self, addon):
        """Install addon dependencies via pip with confirmation. Runs in background thread."""
        deps = " ".join(addon.dependencies)
        confirm = QMessageBox.question(
            self, f"Install {addon.name}",
            f"This will install the following packages via pip:\n\n"
            f"  {deps}\n\n"
            f"This may download several hundred MB of data.\n"
            f"Proceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        self.bus.log_entry.emit(f"Installing {addon.name}...", "system")

        import subprocess, threading, re
        def _do_install():
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install", "--progress-bar", "on"] + addon.dependencies,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in iter(proc.stdout.readline, ""):
                    line = line.strip()
                    if not line:
                        continue

                    # Parse download progress: "──── 75.6/100.0 kB 4.1 MB/s"
                    m = re.search(r'([\d.]+)/([\d.]+)\s*(kB|MB|GB)', line)
                    if m:
                        done, total = float(m.group(1)), float(m.group(2))
                        pct = int(done / total * 100) if total > 0 else 0
                        self.bus.log_entry.emit(f"[{addon.name}] Downloading... {pct}%", "system")
                        continue

                    # Show key steps
                    if any(kw in line for kw in ["Collecting", "Downloading", "Installing", "Successfully"]):
                        short = line[:80]
                        self.bus.log_entry.emit(f"[{addon.name}] {short}", "system")

                proc.wait(timeout=300)
                if proc.returncode == 0:
                    self.bus.log_entry.emit(f"{addon.name} installed! Refreshing...", "trade")
                    # Re-check availability on main thread
                    QTimer.singleShot(100, self._rediscover_addons)
                else:
                    self.bus.log_entry.emit(f"{addon.name} install failed (exit code {proc.returncode})", "error")
            except Exception as e:
                self.bus.log_entry.emit(f"{addon.name} install error: {e}", "error")

        threading.Thread(target=_do_install, daemon=True).start()

    def _rediscover_addons(self):
        """Re-import and re-check all addons after pip install."""
        import importlib
        import addons as addons_pkg
        importlib.reload(addons_pkg)
        discover_addons()
        # Force re-check availability
        for a in get_all_addons():
            a.check_available()
        self._refresh()
        self.bus.log_entry.emit(f"Addons refreshed — {sum(1 for a in get_all_addons() if a.available)} available", "system")

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

            # Check compatibility with aggressivity
            from core.intelligent_trader import check_profile_compatibility
            settings = load_settings()
            aggr = settings.get("aggressivity", "Default")
            compatible, warnings = check_profile_compatibility(aggr, name)
            for w in warnings:
                self.bus.log_entry.emit(w, "warn")

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

