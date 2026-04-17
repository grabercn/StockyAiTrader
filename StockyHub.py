"""
StockyHub — Main launcher for StockyAiTrader.

Provides:
- Buttons to launch each trading app
- Settings dialog for API keys (Alpaca + addon keys)
- Model Manager for downloading/deleting ML models
- Addon Manager for enabling/disabling trading signal addons
"""

import sys
import os
import json
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QDialog,
    QLineEdit, QFormLayout, QHBoxLayout, QMessageBox, QWidget, QGroupBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QTabWidget, QCheckBox, QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QFont, QColor

# Add StockyApps to path so we can import core + addons
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "StockyApps"))
from core.model_manager import (
    MANAGED_MODELS, get_model_status, get_lgbm_models,
    download_model, delete_model, delete_lgbm_model, delete_all_lgbm_models,
)
from addons import (
    get_all_addons, set_addon_enabled, discover_addons,
)

STOCKY_APPS_FOLDER = "StockyApps"
SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "alpaca_api_key": "",
    "alpaca_secret_key": "",
    "default_ticker": "AAPL",
    "refresh_rate": 5,
}

if not os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(DEFAULT_SETTINGS, f, indent=4)


def load_settings():
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)


# ─── Stylesheet ──────────────────────────────────────────────────────────────
STYLE = """
    QMainWindow, QDialog { background-color: #1a1a2e; color: #e0e0e0; }
    QLabel { color: #b0bec5; }
    QLineEdit { background-color: #16213e; border: 1px solid #0f3460;
        padding: 6px; color: #e0e0e0; border-radius: 4px; }
    QPushButton { background-color: #0f3460; color: white; font-size: 13px;
        padding: 10px; border-radius: 6px; font-weight: bold; }
    QPushButton:hover { background-color: #1a5276; }
    QPushButton:disabled { background-color: #333; color: #666; }
    QGroupBox { border: 1px solid #0f3460; border-radius: 6px;
        margin-top: 8px; padding-top: 15px; color: #b0bec5; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
    QTableWidget { background-color: #16213e; color: #e0e0e0;
        gridline-color: #0f3460; border: 1px solid #0f3460; font-size: 12px; }
    QHeaderView::section { background-color: #0f3460; color: white;
        padding: 4px; border: none; font-size: 12px; }
    QProgressBar { background-color: #16213e; border: 1px solid #0f3460;
        border-radius: 4px; text-align: center; color: white; font-size: 11px; }
    QProgressBar::chunk { background-color: #00ff88; }
    QTabWidget::pane { border: 1px solid #0f3460; background-color: #1a1a2e; }
    QTabBar::tab { background-color: #16213e; color: #b0bec5; padding: 8px 16px;
        border: 1px solid #0f3460; border-bottom: none; border-radius: 4px 4px 0 0; }
    QTabBar::tab:selected { background-color: #0f3460; color: white; }
    QCheckBox { color: #e0e0e0; spacing: 8px; }
    QCheckBox::indicator { width: 18px; height: 18px; }
    QScrollArea { border: none; background-color: #1a1a2e; }
"""


# ─── Download Worker (background thread) ────────────────────────────────────
class DownloadWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_info):
        super().__init__()
        self.model_info = model_info

    def run(self):
        download_model(self.model_info, progress_callback=self.progress.emit)
        self.finished.emit()


# ─── Settings Dialog ─────────────────────────────────────────────────────────
class SettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setMinimumWidth(450)
        self.setStyleSheet(STYLE)

        layout = QFormLayout()
        self.settings = load_settings()

        # Alpaca keys
        layout.addRow(QLabel("── Alpaca (Trading) ──"))
        self.api_key_input = QLineEdit(self.settings.get("alpaca_api_key", ""))
        self.secret_key_input = QLineEdit(self.settings.get("alpaca_secret_key", ""))
        self.secret_key_input.setEchoMode(QLineEdit.Password)
        layout.addRow("API Key:", self.api_key_input)
        layout.addRow("Secret Key:", self.secret_key_input)

        # General
        layout.addRow(QLabel("── General ──"))
        self.ticker_input = QLineEdit(self.settings.get("default_ticker", ""))
        self.refresh_input = QLineEdit(str(self.settings.get("refresh_rate", 5)))
        layout.addRow("Default Ticker:", self.ticker_input)
        layout.addRow("Refresh Rate (s):", self.refresh_input)

        # Addon API keys — dynamically built from addons that need keys
        addon_keys_needed = [
            a for a in get_all_addons() if a.requires_api_key and a.api_key_name
        ]
        if addon_keys_needed:
            layout.addRow(QLabel("── Addon API Keys ──"))
            self._addon_inputs = {}
            for addon in addon_keys_needed:
                inp = QLineEdit(self.settings.get(addon.api_key_name, ""))
                inp.setPlaceholderText(f"Key for {addon.name}")
                layout.addRow(f"{addon.name}:", inp)
                self._addon_inputs[addon.api_key_name] = inp
        else:
            self._addon_inputs = {}

        btns = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        btns.addWidget(save_btn)
        btns.addWidget(cancel_btn)
        layout.addRow(btns)
        self.setLayout(layout)

    def _save(self):
        try:
            self.settings["alpaca_api_key"] = self.api_key_input.text()
            self.settings["alpaca_secret_key"] = self.secret_key_input.text()
            self.settings["default_ticker"] = self.ticker_input.text()
            self.settings["refresh_rate"] = int(self.refresh_input.text())

            # Save addon API keys
            for key_name, inp in self._addon_inputs.items():
                self.settings[key_name] = inp.text()

            save_settings(self.settings)
            QMessageBox.information(self, "Settings", "Saved successfully.")
            self.close()
        except ValueError:
            QMessageBox.warning(self, "Error", "Refresh rate must be a number.")


# ─── Main Hub Window ─────────────────────────────────────────────────────────
class StockyHub(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stocky Hub v2")
        self.setGeometry(200, 200, 720, 800)
        self.setStyleSheet(STYLE)
        self._download_worker = None

        main_widget = QWidget()
        layout = QVBoxLayout()

        # ── Banner + Title ──
        banner = QLabel()
        if os.path.exists("banner.jpg"):
            banner.setPixmap(QPixmap("banner.jpg").scaledToHeight(70, Qt.SmoothTransformation))
        banner.setAlignment(Qt.AlignCenter)
        layout.addWidget(banner)

        title = QLabel("Stocky AI Trader")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Consolas", 18, QFont.Bold))
        title.setStyleSheet("color: #00ccff; padding: 3px;")
        layout.addWidget(title)

        subtitle = QLabel("LightGBM + FinBERT + Addons | Risk-Managed Trading")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 5px;")
        layout.addWidget(subtitle)

        # ── App Launchers ──
        apps_box = QGroupBox("Trading Apps")
        apps_layout = QVBoxLayout()
        self._add_app(apps_layout, "AutoTrader",
                      "Multi-stock scanner + auto-invest portfolio manager",
                      "AutoTrader.py")
        self._add_app(apps_layout, "Day Trader",
                      "Single-stock intraday analysis — LightGBM + FinBERT",
                      "DayTrader.py")
        self._add_app(apps_layout, "Long Trader",
                      "Long-term outlook — SMA/EMA crossover analysis",
                      "LongTrader.py")
        self._add_app(apps_layout, "Stock Executer",
                      "Auto-execute trades via Alpaca with risk management",
                      "StockExecuter.py")
        apps_box.setLayout(apps_layout)
        layout.addWidget(apps_box)

        # ── Tabbed Manager (Models + Addons) ──
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_model_tab(), "Models")
        self.tabs.addTab(self._build_addon_tab(), "Addons")
        layout.addWidget(self.tabs)

        # ── Bottom buttons ──
        btn_row = QHBoxLayout()
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._open_settings)
        refresh_btn = QPushButton("Refresh All")
        refresh_btn.clicked.connect(self._refresh_all)
        btn_row.addWidget(settings_btn)
        btn_row.addWidget(refresh_btn)
        layout.addLayout(btn_row)

        # Status bar
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #00ff88; font-size: 11px;")
        layout.addWidget(self.status_label)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # Initial load
        self._refresh_all()

    # ── App Launchers ─────────────────────────────────────────────────────

    def _add_app(self, parent_layout, name, description, script):
        row = QHBoxLayout()
        btn = QPushButton(name)
        btn.setFixedWidth(180)
        btn.clicked.connect(lambda: self._launch(script))
        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #888; padding-left: 10px;")
        row.addWidget(btn)
        row.addWidget(desc, 1)
        parent_layout.addLayout(row)

    def _launch(self, script_name):
        script_path = os.path.join(STOCKY_APPS_FOLDER, script_name)
        if os.path.exists(script_path):
            try:
                subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.status_label.setText(f"Launched {script_name}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to launch: {e}")
        else:
            QMessageBox.warning(self, "Error", f"{script_name} not found.")

    def _open_settings(self):
        dialog = SettingsDialog()
        dialog.exec_()

    def _refresh_all(self):
        self._refresh_model_tables()
        self._refresh_addon_table()

    # ═══════════════════════════════════════════════════════════════════════
    # MODEL MANAGER TAB
    # ═══════════════════════════════════════════════════════════════════════

    def _build_model_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # HuggingFace models
        layout.addWidget(QLabel("AI Models (HuggingFace):"))
        self.hf_table = QTableWidget()
        self.hf_table.setColumnCount(5)
        self.hf_table.setHorizontalHeaderLabels(["Model", "Status", "Size", "", ""])
        self.hf_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            self.hf_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.hf_table.setFixedHeight(80)
        self.hf_table.verticalHeader().setVisible(False)
        layout.addWidget(self.hf_table)

        # Download progress
        self.dl_progress = QProgressBar()
        self.dl_progress.setRange(0, 0)
        self.dl_progress.setVisible(False)
        layout.addWidget(self.dl_progress)

        # LightGBM models
        layout.addWidget(QLabel("Trained Models (LightGBM):"))
        self.lgbm_table = QTableWidget()
        self.lgbm_table.setColumnCount(3)
        self.lgbm_table.setHorizontalHeaderLabels(["Ticker Model", "Size", ""])
        self.lgbm_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.lgbm_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.lgbm_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.lgbm_table.setFixedHeight(120)
        self.lgbm_table.verticalHeader().setVisible(False)
        layout.addWidget(self.lgbm_table)

        clear_btn = QPushButton("Clear All Trained Models")
        clear_btn.setStyleSheet("background-color: #8B0000; font-size: 11px; padding: 6px;")
        clear_btn.clicked.connect(self._clear_all_lgbm)
        layout.addWidget(clear_btn)

        widget.setLayout(layout)
        return widget

    def _refresh_model_tables(self):
        # HuggingFace models
        self.hf_table.setRowCount(len(MANAGED_MODELS))
        for i, model_info in enumerate(MANAGED_MODELS):
            downloaded, size_str = get_model_status(model_info)

            name_item = QTableWidgetItem(f"{model_info.name}\n{model_info.description}")
            name_item.setFlags(Qt.ItemIsEnabled)
            self.hf_table.setItem(i, 0, name_item)

            status = "Downloaded" if downloaded else "Not Downloaded"
            status_item = QTableWidgetItem(status)
            status_item.setFlags(Qt.ItemIsEnabled)
            status_item.setForeground(QColor("#00ff88" if downloaded else "#ff4444"))
            self.hf_table.setItem(i, 1, status_item)

            size_item = QTableWidgetItem(size_str if downloaded else model_info.size_estimate)
            size_item.setFlags(Qt.ItemIsEnabled)
            self.hf_table.setItem(i, 2, size_item)

            if not downloaded:
                dl_btn = QPushButton("Download")
                dl_btn.setStyleSheet("background-color: #006600; font-size: 11px; padding: 4px;")
                dl_btn.clicked.connect(lambda _, m=model_info: self._download_model(m))
                self.hf_table.setCellWidget(i, 3, dl_btn)
            else:
                ok_item = QTableWidgetItem("Ready")
                ok_item.setFlags(Qt.ItemIsEnabled)
                ok_item.setForeground(QColor("#00ff88"))
                self.hf_table.setItem(i, 3, ok_item)

            if downloaded:
                del_btn = QPushButton("Delete")
                del_btn.setStyleSheet("background-color: #8B0000; font-size: 11px; padding: 4px;")
                del_btn.clicked.connect(lambda _, m=model_info: self._delete_hf_model(m))
                self.hf_table.setCellWidget(i, 4, del_btn)

        self.hf_table.resizeRowsToContents()

        # LightGBM models
        lgbm = get_lgbm_models()
        self.lgbm_table.setRowCount(max(len(lgbm), 1))
        if lgbm:
            for i, (filename, size_str) in enumerate(lgbm):
                name_item = QTableWidgetItem(filename)
                name_item.setFlags(Qt.ItemIsEnabled)
                self.lgbm_table.setItem(i, 0, name_item)

                size_item = QTableWidgetItem(size_str)
                size_item.setFlags(Qt.ItemIsEnabled)
                self.lgbm_table.setItem(i, 1, size_item)

                del_btn = QPushButton("Delete")
                del_btn.setStyleSheet("background-color: #8B0000; font-size: 11px; padding: 4px;")
                del_btn.clicked.connect(lambda _, f=filename: self._delete_lgbm(f))
                self.lgbm_table.setCellWidget(i, 2, del_btn)
        else:
            empty = QTableWidgetItem("No trained models — run Day Trader or Long Trader first")
            empty.setFlags(Qt.ItemIsEnabled)
            empty.setForeground(QColor("#666"))
            self.lgbm_table.setItem(0, 0, empty)

    def _download_model(self, model_info):
        if self._download_worker and self._download_worker.isRunning():
            self.status_label.setText("Download already in progress...")
            return
        self.dl_progress.setVisible(True)
        self.status_label.setText(f"Downloading {model_info.name}...")
        self._download_worker = DownloadWorker(model_info)
        self._download_worker.progress.connect(lambda s: self.status_label.setText(s))
        self._download_worker.finished.connect(self._on_dl_finished)
        self._download_worker.start()

    def _on_dl_finished(self):
        self.dl_progress.setVisible(False)
        self._refresh_model_tables()

    def _delete_hf_model(self, model_info):
        if QMessageBox.question(self, "Confirm", f"Delete {model_info.name}?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            ok, msg = delete_model(model_info)
            self.status_label.setText(msg)
            self._refresh_model_tables()

    def _delete_lgbm(self, filename):
        ok, msg = delete_lgbm_model(filename)
        self.status_label.setText(msg)
        self._refresh_model_tables()

    def _clear_all_lgbm(self):
        if QMessageBox.question(self, "Confirm", "Delete ALL trained models?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            ok, msg = delete_all_lgbm_models()
            self.status_label.setText(msg)
            self._refresh_model_tables()

    # ═══════════════════════════════════════════════════════════════════════
    # ADDON MANAGER TAB
    # ═══════════════════════════════════════════════════════════════════════

    def _build_addon_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel(
            "Addons provide extra data signals to improve trading accuracy.\n"
            "Enable/disable them here. Some need API keys — configure in Settings."
        ))

        # Addon table
        self.addon_table = QTableWidget()
        self.addon_table.setColumnCount(6)
        self.addon_table.setHorizontalHeaderLabels([
            "Enabled", "Addon", "Description", "Status", "Features", "Config"
        ])
        self.addon_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.addon_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.addon_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.addon_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.addon_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.addon_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.addon_table.verticalHeader().setVisible(False)
        layout.addWidget(self.addon_table)

        # Summary
        self.addon_summary = QLabel("")
        self.addon_summary.setStyleSheet("color: #00ccff; font-size: 11px; padding-top: 5px;")
        layout.addWidget(self.addon_summary)

        # Install missing deps button
        install_btn = QPushButton("Install Missing Dependencies (pip)")
        install_btn.setStyleSheet("font-size: 11px; padding: 6px;")
        install_btn.clicked.connect(self._install_missing_deps)
        layout.addWidget(install_btn)

        widget.setLayout(layout)
        return widget

    def _refresh_addon_table(self):
        discover_addons()
        addons = get_all_addons()

        # Load saved enable/disable state from settings
        settings = load_settings()
        addon_states = settings.get("addon_states", {})

        self.addon_table.setRowCount(len(addons))
        active_count = 0
        total_features = 0

        for i, addon in enumerate(addons):
            # Restore saved enable/disable state
            if addon.module_name in addon_states:
                addon.enabled = addon_states[addon.module_name]
                set_addon_enabled(addon.module_name, addon.enabled)

            # Enable checkbox
            cb = QCheckBox()
            cb.setChecked(addon.enabled and addon.available)
            cb.setEnabled(addon.available)
            cb.toggled.connect(lambda checked, name=addon.module_name: self._toggle_addon(name, checked))
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.addon_table.setCellWidget(i, 0, cb_widget)

            # Name
            name_item = QTableWidgetItem(addon.name)
            name_item.setFlags(Qt.ItemIsEnabled)
            name_item.setFont(QFont("Consolas", 10, QFont.Bold))
            self.addon_table.setItem(i, 1, name_item)

            # Description
            desc_item = QTableWidgetItem(addon.description)
            desc_item.setFlags(Qt.ItemIsEnabled)
            self.addon_table.setItem(i, 2, desc_item)

            # Status
            if addon.available and addon.enabled:
                status_text = "Active"
                status_color = "#00ff88"
                active_count += 1
                total_features += len(addon.features)
            elif addon.available and not addon.enabled:
                status_text = "Disabled"
                status_color = "#ffaa00"
            else:
                status_text = addon.status
                status_color = "#ff4444"

            status_item = QTableWidgetItem(status_text)
            status_item.setFlags(Qt.ItemIsEnabled)
            status_item.setForeground(QColor(status_color))
            self.addon_table.setItem(i, 3, status_item)

            # Features count
            feat_item = QTableWidgetItem(f"{len(addon.features)} features")
            feat_item.setFlags(Qt.ItemIsEnabled)
            self.addon_table.setItem(i, 4, feat_item)

            # Config hint
            if addon.requires_api_key:
                has_key = bool(settings.get(addon.api_key_name, ""))
                config_text = "Key set" if has_key else "Needs key"
                config_color = "#00ff88" if has_key else "#ff4444"
            else:
                config_text = "No config"
                config_color = "#666"

            config_item = QTableWidgetItem(config_text)
            config_item.setFlags(Qt.ItemIsEnabled)
            config_item.setForeground(QColor(config_color))
            self.addon_table.setItem(i, 5, config_item)

        self.addon_table.resizeRowsToContents()
        self.addon_summary.setText(
            f"{active_count}/{len(addons)} addons active — "
            f"{total_features} extra features feeding into model"
        )

    def _toggle_addon(self, module_name, enabled):
        """Enable or disable an addon and persist the choice."""
        set_addon_enabled(module_name, enabled)

        # Save state to settings.json
        settings = load_settings()
        states = settings.get("addon_states", {})
        states[module_name] = enabled
        settings["addon_states"] = states
        save_settings(settings)

        self._refresh_addon_table()
        self.status_label.setText(f"Addon {'enabled' if enabled else 'disabled'}: {module_name}")

    def _install_missing_deps(self):
        """Attempt to pip-install missing addon dependencies."""
        addons = get_all_addons()
        missing = set()
        for addon in addons:
            if not addon.available and addon.dependencies:
                missing.update(addon.dependencies)

        if not missing:
            self.status_label.setText("All addon dependencies are installed!")
            return

        self.status_label.setText(f"Installing: {', '.join(missing)}...")
        try:
            import subprocess as sp
            result = sp.run(
                [sys.executable, "-m", "pip", "install"] + list(missing),
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                self.status_label.setText("Dependencies installed! Refreshing...")
                discover_addons()
                self._refresh_addon_table()
            else:
                self.status_label.setText(f"Install failed: {result.stderr[:100]}")
        except Exception as e:
            self.status_label.setText(f"Install error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hub = StockyHub()
    hub.show()
    sys.exit(app.exec_())
