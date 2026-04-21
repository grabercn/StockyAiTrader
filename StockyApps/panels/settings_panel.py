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

        # Zoom slider
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("UI Scale:"))
        self.zoom_slider = QSpinBox()
        self.zoom_slider.setRange(70, 200)
        self.zoom_slider.setSuffix("%")
        self.zoom_slider.setSingleStep(5)
        current_zoom = int(settings.get("ui_zoom", 0.95) * 100)
        self.zoom_slider.setValue(current_zoom)
        self.zoom_slider.valueChanged.connect(self._change_zoom)
        zoom_row.addWidget(self.zoom_slider)
        self.zoom_label = QLabel(f"Current: {current_zoom}%")
        self.zoom_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        zoom_row.addWidget(self.zoom_label)
        al2.addLayout(zoom_row)

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
            ("nixtla_api_key", "Nixtla TimeGPT Key", "nixtla-tok-XXXXXXXXXXXX",
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
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels(["Model", "Status", "Size", "Action"])
        self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 4):
            self.model_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.model_table.verticalHeader().setVisible(False)
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
        self.addon_table.setColumnCount(6)
        self.addon_table.setHorizontalHeaderLabels(["On", "Addon", "Status", "Features", "Config", ""])
        self.addon_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.addon_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for c in range(2, 5):
            self.addon_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.addon_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.addon_table.verticalHeader().setVisible(False)
        al.addWidget(self.addon_table)
        addon_box.setLayout(al)
        inner_layout.addWidget(addon_box)

        inner.setLayout(inner_layout)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self._refresh()

    def _change_zoom(self, value):
        scale = value / 100.0
        main_window = self.window()
        if main_window and hasattr(main_window, '_scale'):
            main_window._scale = scale
            main_window._apply_scale()
            main_window._save_zoom()
        self.zoom_label.setText(f"Current: {value}%")

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
        # Models
        self.model_table.setRowCount(len(MANAGED_MODELS))
        for i, m in enumerate(MANAGED_MODELS):
            dl, sz = get_model_status(m)
            self.model_table.setItem(i, 0, QTableWidgetItem(m.name))
            s = QTableWidgetItem("Ready" if dl else "Not downloaded")
            s.setForeground(QColor(STATUS_ACTIVE if dl else STATUS_ERROR))
            self.model_table.setItem(i, 1, s)
            self.model_table.setItem(i, 2, QTableWidgetItem(sz if dl else m.size_estimate))
            if not dl:
                btn = QPushButton("Download")
                btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 10px; padding: 3px;")
                btn.clicked.connect(lambda _, mi=m: self._download(mi))
                self.model_table.setCellWidget(i, 3, btn)
            else:
                btn = QPushButton("Delete")
                btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 10px; padding: 3px;")
                btn.clicked.connect(lambda _, mi=m: self._delete_model(mi))
                self.model_table.setCellWidget(i, 3, btn)

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
            self.addon_table.setItem(i, 3, QTableWidgetItem(f"{len(a.features)}"))
            cfg = "Key set" if a.requires_api_key and settings.get(a.api_key_name) else ("Needs key" if a.requires_api_key else "—")
            c = QTableWidgetItem(cfg)
            c.setForeground(QColor(STATUS_ACTIVE if cfg == "Key set" else (STATUS_ERROR if cfg == "Needs key" else TEXT_MUTED)))
            self.addon_table.setItem(i, 4, c)

            # Install button for unavailable addons
            if not a.available and a.dependencies:
                install_btn = QPushButton("Install")
                install_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; font-size: 9px; padding: 2px 6px;")
                install_btn.setToolTip(f"pip install {' '.join(a.dependencies)}")
                install_btn.clicked.connect(lambda _, addon=a: self._install_addon(addon))
                self.addon_table.setCellWidget(i, 5, install_btn)
            elif a.available:
                ok_item = QTableWidgetItem("Ready")
                ok_item.setForeground(QColor(STATUS_ACTIVE))
                ok_item.setFlags(Qt.ItemIsEnabled)
                self.addon_table.setItem(i, 5, ok_item)

    def _show_key_help(self, label, url, instructions):
        """Show popup with instructions on how to get an API key."""
        from core.ui.theme import theme as _theme
        dlg = QDialog(self)
        dlg.setWindowTitle(f"How to get: {label}")
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
        """Install addon dependencies via pip with confirmation."""
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

        self.bus.log_entry.emit(f"Installing {addon.name} dependencies: {deps}", "system")

        import subprocess, threading
        def _do_install():
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + addon.dependencies,
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    self.bus.log_entry.emit(f"{addon.name} installed successfully! Restart to activate.", "trade")
                    # Re-discover addons
                    discover_addons()
                    QTimer.singleShot(500, self._refresh)
                else:
                    self.bus.log_entry.emit(f"{addon.name} install failed: {result.stderr[:200]}", "error")
            except Exception as e:
                self.bus.log_entry.emit(f"{addon.name} install error: {e}", "error")

        threading.Thread(target=_do_install, daemon=True).start()

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

