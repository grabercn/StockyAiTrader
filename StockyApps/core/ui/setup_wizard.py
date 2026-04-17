"""
First-Run Setup Wizard — guides new users through complete app configuration.

Shows on first boot (or when settings has no "setup_complete" flag).
Walks through:
1. Welcome + overview
2. Alpaca API key setup (with test connection)
3. Theme selection
4. Hardware profile choice
5. Model downloads
6. Done — ready to trade

Each step is a page in a stacked widget with forward/back navigation.
"""

import os
import json
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QStackedWidget, QWidget, QProgressBar,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon

from ..branding import (
    APP_NAME, APP_VERSION, APP_TAGLINE, APP_URL,
    BRAND_PRIMARY, BRAND_ACCENT, TEXT_MUTED, TEXT_SECONDARY,
    FONT_FAMILY, FONT_MONO,
)
from .theme import theme
from .icons import StockyIcons

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "settings.json")


def _load():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=4)


def needs_setup():
    """Check if the setup wizard should be shown."""
    return not _load().get("setup_complete", False)


class SetupWizard(QDialog):
    """Multi-step setup wizard for first-time users."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} — Setup")
        self.setFixedSize(560, 480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._build()

    def _build(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Pages
        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_welcome())
        self.stack.addWidget(self._page_api_keys())
        self.stack.addWidget(self._page_theme())
        self.stack.addWidget(self._page_profile())
        self.stack.addWidget(self._page_done())
        layout.addWidget(self.stack)

        # Navigation bar
        nav = QWidget()
        nav.setStyleSheet(f"background-color: {theme.color('bg_panel')}; border-top: 1px solid {theme.color('border')};")
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(20, 10, 20, 10)

        # Step indicator
        self.step_label = QLabel("Step 1 of 5")
        self.step_label.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 11px;")
        nav_layout.addWidget(self.step_label)
        nav_layout.addStretch()

        self.back_btn = QPushButton("Back")
        self.back_btn.setStyleSheet(f"background-color: {theme.color('bg_input')}; padding: 8px 20px;")
        self.back_btn.clicked.connect(self._back)
        nav_layout.addWidget(self.back_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.setStyleSheet(f"background-color: {BRAND_PRIMARY}; padding: 8px 20px;")
        self.next_btn.clicked.connect(self._next)
        nav_layout.addWidget(self.next_btn)

        nav.setLayout(nav_layout)
        layout.addWidget(nav)
        self.setLayout(layout)
        self._update_nav()

    def _page(self, title, subtitle=""):
        """Create a styled page container."""
        page = QWidget()
        page.setStyleSheet(f"background-color: {theme.color('bg_base')};")
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 30, 40, 20)
        layout.setSpacing(12)

        t = QLabel(title)
        t.setFont(QFont(FONT_FAMILY, 18, QFont.Bold))
        t.setStyleSheet(f"color: {BRAND_PRIMARY};")
        layout.addWidget(t)

        if subtitle:
            s = QLabel(subtitle)
            s.setWordWrap(True)
            s.setStyleSheet(f"color: {theme.color('text_secondary')}; font-size: 12px;")
            layout.addWidget(s)

        page.setLayout(layout)
        return page, layout

    # ── Page 1: Welcome ──

    def _page_welcome(self):
        page, layout = self._page(
            f"Welcome to {APP_NAME}",
            "Let's get you set up in just a few steps."
        )

        icon_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "icon.png")
        if os.path.exists(icon_path):
            icon = QLabel()
            icon.setPixmap(QPixmap(icon_path).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            icon.setAlignment(Qt.AlignCenter)
            icon.setStyleSheet("padding: 10px; background: transparent;")
            layout.insertWidget(0, icon)

        features = [
            "AI-powered stock analysis with LightGBM + FinBERT",
            "Multi-stock scanner with auto-invest",
            "10 pluggable signal addons",
            "Risk management with ATR-based position sizing",
            "Tax report generation (Form 8949)",
        ]
        for f in features:
            lbl = QLabel(f"  {f}")
            lbl.setStyleSheet(f"color: {theme.color('text_primary')}; font-size: 12px; background: transparent;")
            layout.addWidget(lbl)

        layout.addStretch()
        return page

    # ── Page 2: API Keys ──

    def _page_api_keys(self):
        page, layout = self._page(
            "Broker Connection",
            "Connect your Alpaca account for paper trading. "
            "Get free API keys at alpaca.markets — no real money needed to start."
        )

        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Alpaca API Key")
        layout.addWidget(QLabel("API Key:"))
        layout.addWidget(self.api_key_input)

        self.secret_input = QLineEdit()
        self.secret_input.setPlaceholderText("Alpaca Secret Key")
        self.secret_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(QLabel("Secret Key:"))
        layout.addWidget(self.secret_input)

        test_btn = QPushButton("Test Connection")
        test_btn.setStyleSheet(f"background-color: {BRAND_ACCENT}; padding: 8px;")
        test_btn.clicked.connect(self._test_connection)
        layout.addWidget(test_btn)

        self.api_status = QLabel("")
        self.api_status.setStyleSheet(f"font-size: 11px;")
        layout.addWidget(self.api_status)

        skip = QLabel("You can skip this and configure later in Settings.")
        skip.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 10px; background: transparent;")
        layout.addWidget(skip)

        layout.addStretch()
        return page

    def _test_connection(self):
        import requests
        key = self.api_key_input.text().strip()
        secret = self.secret_input.text().strip()
        if not key or not secret:
            self.api_status.setText("Enter both keys first.")
            self.api_status.setStyleSheet(f"color: {theme.color('sell')};")
            return
        try:
            headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
            r = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers, timeout=10)
            r.raise_for_status()
            acct = r.json()
            pv = float(acct.get("portfolio_value", 0))
            self.api_status.setText(f"Connected! Portfolio: ${pv:,.2f}")
            self.api_status.setStyleSheet(f"color: {BRAND_ACCENT};")
        except Exception as e:
            self.api_status.setText(f"Failed: {e}")
            self.api_status.setStyleSheet(f"color: {theme.color('sell')};")

    # ── Page 3: Theme ──

    def _page_theme(self):
        page, layout = self._page(
            "Appearance",
            "Choose your preferred theme. The app can also match your Windows setting."
        )

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Auto (match Windows)", "Dark", "Light"])
        layout.addWidget(QLabel("Theme:"))
        layout.addWidget(self.theme_combo)

        layout.addStretch()
        return page

    # ── Page 4: Hardware Profile ──

    def _page_profile(self):
        page, layout = self._page(
            "Hardware Profile",
            "Choose how much AI processing to use. You can change this anytime in Settings."
        )

        self.profile_combo = QComboBox()
        from ..profiles import PRESETS
        for name, preset in PRESETS.items():
            self.profile_combo.addItem(f"{name} — {preset['description']}", name)
        self.profile_combo.setCurrentIndex(1)  # Default to Balanced
        layout.addWidget(self.profile_combo)

        layout.addStretch()
        return page

    # ── Page 5: Done ──

    def _page_done(self):
        page, layout = self._page(
            "You're all set!",
            "Your trading suite is ready. You can always adjust settings later."
        )

        tips = [
            "Start with the Scanner tab to find opportunities",
            "Use Day Trade for deep single-stock analysis",
            "Check the Dashboard for your portfolio overview",
            "Run diagnostics in the Testing tab anytime",
        ]
        for tip in tips:
            lbl = QLabel(f"  {tip}")
            lbl.setStyleSheet(f"color: {theme.color('text_primary')}; font-size: 12px; background: transparent;")
            layout.addWidget(lbl)

        layout.addStretch()
        return page

    # ── Navigation ──

    def _update_nav(self):
        idx = self.stack.currentIndex()
        total = self.stack.count()
        self.step_label.setText(f"Step {idx + 1} of {total}")
        self.back_btn.setEnabled(idx > 0)
        self.next_btn.setText("Finish" if idx == total - 1 else "Next")

    def _next(self):
        idx = self.stack.currentIndex()
        if idx == self.stack.count() - 1:
            self._finish()
            return

        # Save current page data before advancing
        if idx == 1:  # API Keys
            settings = _load()
            settings["alpaca_api_key"] = self.api_key_input.text().strip()
            settings["alpaca_secret_key"] = self.secret_input.text().strip()
            _save(settings)

        if idx == 2:  # Theme
            theme_map = {0: "auto", 1: "dark", 2: "light"}
            settings = _load()
            settings["theme"] = theme_map.get(self.theme_combo.currentIndex(), "auto")
            _save(settings)

        if idx == 3:  # Profile
            profile_name = self.profile_combo.currentData()
            from ..profiles import apply_profile
            apply_profile(profile_name)

        self.stack.setCurrentIndex(idx + 1)
        self._update_nav()

    def _back(self):
        idx = self.stack.currentIndex()
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
            self._update_nav()

    def _finish(self):
        settings = _load()
        settings["setup_complete"] = True
        _save(settings)
        self.accept()
