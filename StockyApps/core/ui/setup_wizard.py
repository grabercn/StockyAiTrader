"""
First-Run Setup Wizard — premium branded onboarding experience.

A visually rich, animated wizard that guides new users through
complete app configuration with style and polish.
"""

import os
import json
import math
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QStackedWidget, QWidget, QFrame,
    QGraphicsOpacityEffect, QSpacerItem, QSizePolicy, QApplication,
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QPointF, QSize,
)
from PyQt5.QtGui import (
    QFont, QPixmap, QIcon, QPainter, QColor, QLinearGradient,
    QPen, QBrush, QPainterPath, QRadialGradient,
)

from ..branding import (
    APP_NAME, APP_VERSION, APP_TAGLINE, APP_URL, APP_AUTHOR,
    BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT,
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
    return not _load().get("setup_complete", False)


# ─── Animated Background ────────────────────────────────────────────────────

class _WizardBackground(QWidget):
    """Animated gradient background with floating orbs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        import random
        self._phase = random.uniform(0, 10)  # Start at random phase so orbs are mid-motion
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(35)

    def _tick(self):
        self._phase += 0.02
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Base gradient
        grad = QLinearGradient(0, 0, w, h)
        if theme.is_dark:
            grad.setColorAt(0, QColor(10, 12, 20))
            grad.setColorAt(0.5, QColor(15, 20, 35))
            grad.setColorAt(1, QColor(8, 10, 18))
        else:
            grad.setColorAt(0, QColor(240, 245, 255))
            grad.setColorAt(0.5, QColor(235, 240, 252))
            grad.setColorAt(1, QColor(245, 248, 255))
        painter.fillRect(0, 0, w, h, grad)

        # Floating orbs (animated)
        orbs = [
            (0.2, 0.3, 80, BRAND_PRIMARY, 0.06),
            (0.7, 0.6, 120, BRAND_SECONDARY, 0.04),
            (0.5, 0.8, 60, BRAND_ACCENT, 0.05),
            (0.85, 0.2, 90, BRAND_PRIMARY, 0.03),
        ]
        for bx, by, radius, color_hex, alpha in orbs:
            ox = bx * w + math.sin(self._phase * 1.5 + bx * 10) * 30
            oy = by * h + math.cos(self._phase * 1.2 + by * 8) * 20
            rg = QRadialGradient(QPointF(ox, oy), radius)
            c = QColor(color_hex)
            c.setAlphaF(alpha)
            rg.setColorAt(0, c)
            c.setAlphaF(0)
            rg.setColorAt(1, c)
            painter.setBrush(rg)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(ox, oy), radius, radius)

        # Top accent line
        line_grad = QLinearGradient(0, 0, w, 0)
        line_grad.setColorAt(0, QColor(BRAND_PRIMARY + "00"))
        line_grad.setColorAt(0.3, QColor(BRAND_PRIMARY))
        line_grad.setColorAt(0.7, QColor(BRAND_ACCENT))
        line_grad.setColorAt(1, QColor(BRAND_ACCENT + "00"))
        painter.setPen(QPen(QBrush(line_grad), 3))
        painter.drawLine(0, 0, w, 0)

        painter.end()


# ─── Step Indicator ──────────────────────────────────────────────────────────

class _StepIndicator(QWidget):
    """Animated dot-based step progress indicator."""

    def __init__(self, total_steps=5, parent=None):
        super().__init__(parent)
        self._total = total_steps
        self._current = 0
        self.setFixedHeight(40)

    def set_step(self, step):
        self._current = step
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        dot_r = 6
        spacing = 40
        total_w = (self._total - 1) * spacing
        start_x = (w - total_w) / 2

        for i in range(self._total):
            x = start_x + i * spacing
            y = h / 2

            # Connection line
            if i < self._total - 1:
                line_color = QColor(BRAND_PRIMARY) if i < self._current else QColor(theme.color("border"))
                painter.setPen(QPen(line_color, 2))
                painter.drawLine(int(x + dot_r + 4), int(y), int(x + spacing - dot_r - 4), int(y))

            # Dot
            if i < self._current:
                # Completed — filled brand color
                painter.setBrush(QColor(BRAND_PRIMARY))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(x, y), dot_r, dot_r)
                # Checkmark
                painter.setPen(QPen(QColor("white"), 2, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(int(x - 3), int(y), int(x - 1), int(y + 3))
                painter.drawLine(int(x - 1), int(y + 3), int(x + 4), int(y - 3))
            elif i == self._current:
                # Current — pulsing ring
                painter.setBrush(QColor(BRAND_PRIMARY))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(x, y), dot_r, dot_r)
                # Outer ring
                ring_color = QColor(BRAND_PRIMARY)
                ring_color.setAlphaF(0.3)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(ring_color, 2))
                painter.drawEllipse(QPointF(x, y), dot_r + 4, dot_r + 4)
            else:
                # Future — hollow
                painter.setBrush(QColor(theme.color("bg_input")))
                painter.setPen(QPen(QColor(theme.color("border")), 2))
                painter.drawEllipse(QPointF(x, y), dot_r, dot_r)

        painter.end()


# ─── Feature Card (for welcome page) ────────────────────────────────────────

class _FeatureCard(QFrame):
    """Small branded feature card with icon and text."""

    def __init__(self, icon_name, text, parent=None):
        super().__init__(parent)
        self.setFixedHeight(42)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.color("bg_card")};
                border: 1px solid {theme.color("border")};
                border-radius: 8px;
            }}
            QFrame:hover {{
                border-color: {BRAND_PRIMARY}60;
                background-color: {BRAND_PRIMARY}08;
            }}
        """)
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(StockyIcons.get(icon_name, 18, BRAND_PRIMARY))
        icon_lbl.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(icon_lbl)

        text_lbl = QLabel(text)
        text_lbl.setStyleSheet(f"color: {theme.color('text_primary')}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(text_lbl, 1)

        check_lbl = QLabel()
        check_lbl.setPixmap(StockyIcons.get("check", 14, BRAND_ACCENT))
        check_lbl.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(check_lbl)

        self.setLayout(layout)


# ═════════════════════════════════════════════════════════════════════════════
# SETUP WIZARD
# ═════════════════════════════════════════════════════════════════════════════

class SetupWizard(QDialog):
    """Premium branded multi-step setup wizard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} — Welcome")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        # Qt high DPI handles scaling — use logical pixels
        self._scale = 1.0
        self.setFixedSize(580, 520)

        # Background
        self._bg = _WizardBackground(self)

        # Main layout on top of background
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Step indicator
        self.step_indicator = _StepIndicator(5)
        self.step_indicator.setStyleSheet("background: transparent;")
        layout.addWidget(self.step_indicator)

        # Page stack
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background: transparent;")
        self.stack.addWidget(self._page_welcome())
        self.stack.addWidget(self._page_api_keys())
        self.stack.addWidget(self._page_theme())
        self.stack.addWidget(self._page_profile())
        self.stack.addWidget(self._page_done())
        layout.addWidget(self.stack)

        # Navigation bar
        nav = QWidget()
        nav.setStyleSheet("background: transparent;")
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(30, 8, 30, 16)

        self.step_label = QLabel("Step 1 of 5")
        self.step_label.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 10px;")
        nav_layout.addWidget(self.step_label)
        nav_layout.addStretch()

        self.skip_btn = QPushButton("Skip for now")
        self.skip_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {theme.color('text_muted')};
                border: none; font-size: 11px; padding: 8px 12px; }}
            QPushButton:hover {{ color: {theme.color('text_primary')}; }}
        """)
        self.skip_btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn.clicked.connect(self._skip)
        nav_layout.addWidget(self.skip_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {theme.color('bg_input')}; color: {theme.color('text_secondary')};
                padding: 10px 24px; border-radius: 8px; border: 1px solid {theme.color('border')}; font-weight: 600; }}
            QPushButton:hover {{ background-color: {theme.color('bg_hover')}; }}
        """)
        self.back_btn.clicked.connect(self._back)
        nav_layout.addWidget(self.back_btn)

        self.next_btn = QPushButton("Continue")
        self.next_btn.setStyleSheet(f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {BRAND_PRIMARY}, stop:1 {BRAND_ACCENT}); color: white;
                padding: 10px 28px; border-radius: 8px; border: none; font-weight: 700; font-size: 13px; }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #38bdf8, stop:1 #34d399); }}
        """)
        self.next_btn.clicked.connect(self._next)
        nav_layout.addWidget(self.next_btn)

        nav.setLayout(nav_layout)
        layout.addWidget(nav)

        self.setLayout(layout)
        self._update_nav()

    def resizeEvent(self, event):
        self._bg.resize(self.size())
        super().resizeEvent(event)

    # ── Page helpers ──────────────────────────────────────────────────────

    def _page_container(self):
        page = QWidget()
        page.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 10, 40, 10)
        layout.setSpacing(10)
        return page, layout

    def _title(self, layout, text, icon_name=None):
        s = getattr(self, '_scale', 1.0)
        row = QHBoxLayout()
        if icon_name:
            icon = QLabel()
            icon.setPixmap(StockyIcons.get(icon_name, int(28 * s), BRAND_PRIMARY))
            icon.setStyleSheet("background: transparent;")
            row.addWidget(icon)
        lbl = QLabel(text)
        lbl.setFont(QFont(FONT_FAMILY, int(20 * s), QFont.Bold))
        lbl.setStyleSheet(f"color: {theme.color('text_heading')}; background: transparent;")
        row.addWidget(lbl)
        row.addStretch()
        layout.addLayout(row)

    def _subtitle(self, layout, text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {theme.color('text_secondary')}; font-size: 12px; background: transparent; line-height: 1.5;")
        layout.addWidget(lbl)

    def _input_label(self, layout, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {theme.color('text_secondary')}; font-size: 11px; font-weight: 600; background: transparent; margin-top: 4px;")
        layout.addWidget(lbl)

    # ── Pages ─────────────────────────────────────────────────────────────

    def _page_welcome(self):
        page, layout = self._page_container()

        # Logo
        icon_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "icon.png")
        if os.path.exists(icon_path):
            icon = QLabel()
            icon.setPixmap(QPixmap(icon_path).scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            icon.setAlignment(Qt.AlignCenter)
            icon.setStyleSheet("background: transparent; padding: 5px;")
            layout.addWidget(icon)

        # Title
        name = QLabel(APP_NAME)
        name.setFont(QFont(FONT_FAMILY, 26, QFont.Bold))
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet(f"color: {BRAND_PRIMARY}; background: transparent;")
        layout.addWidget(name)

        tag = QLabel(APP_TAGLINE)
        tag.setAlignment(Qt.AlignCenter)
        tag.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 12px; background: transparent;")
        layout.addWidget(tag)

        layout.addSpacing(12)

        # Feature cards
        features = [
            ("brain",     "LightGBM + FinBERT AI trading engine"),
            ("scan",      "Multi-stock scanner with auto-invest"),
            ("shield",    "ATR risk management with bracket orders"),
            ("bolt",      "10 pluggable signal addons"),
            ("tax",       "Tax report generation (IRS Form 8949)"),
        ]
        for icon_name, text in features:
            layout.addWidget(_FeatureCard(icon_name, text))

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _page_api_keys(self):
        page, layout = self._page_container()
        self._title(layout, "Broker Connection", "wallet")
        self._subtitle(layout, "Connect your Alpaca paper trading account. "
                       "Get free API keys at alpaca.markets — no real money needed.")

        layout.addSpacing(8)

        self._input_label(layout, "API KEY")
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("pk_xxxxxxxxxxxxxxxxxx")
        self.api_key_input.setStyleSheet(f"""
            QLineEdit {{ background-color: {theme.color('bg_card')}; border: 1px solid {theme.color('border')};
                padding: 10px 14px; border-radius: 8px; color: {theme.color('text_primary')}; font-family: {FONT_MONO}; }}
            QLineEdit:focus {{ border-color: {BRAND_PRIMARY}; }}
        """)
        layout.addWidget(self.api_key_input)

        self._input_label(layout, "SECRET KEY")
        self.secret_input = QLineEdit()
        self.secret_input.setPlaceholderText("sk_xxxxxxxxxxxxxxxxxx")
        self.secret_input.setEchoMode(QLineEdit.Password)
        self.secret_input.setStyleSheet(self.api_key_input.styleSheet())
        layout.addWidget(self.secret_input)

        test_btn = QPushButton("  Test Connection")
        test_btn.setIcon(StockyIcons.get_icon("check", 16, "white"))
        test_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {BRAND_ACCENT}; color: white;
                padding: 10px 20px; border-radius: 8px; font-weight: 600; border: none; }}
            QPushButton:hover {{ background-color: #34d399; }}
        """)
        test_btn.setCursor(Qt.PointingHandCursor)
        test_btn.clicked.connect(self._test_connection)
        layout.addWidget(test_btn)

        self.api_status = QLabel("")
        self.api_status.setStyleSheet("font-size: 11px; background: transparent;")
        layout.addWidget(self.api_status)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _test_connection(self):
        import requests
        key = self.api_key_input.text().strip()
        secret = self.secret_input.text().strip()
        if not key or not secret:
            self.api_status.setText("  Enter both keys first.")
            self.api_status.setStyleSheet(f"color: {theme.color('sell')}; font-size: 11px; background: transparent;")
            return
        try:
            self.api_status.setText("  Connecting...")
            self.api_status.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 11px; background: transparent;")
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()

            headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
            r = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers, timeout=10)
            r.raise_for_status()
            pv = float(r.json().get("portfolio_value", 0))
            self.api_status.setText(f"  Connected! Portfolio: ${pv:,.2f}")
            self.api_status.setStyleSheet(f"color: {BRAND_ACCENT}; font-size: 11px; font-weight: 600; background: transparent;")
        except Exception as e:
            self.api_status.setText(f"  Failed: {str(e)[:60]}")
            self.api_status.setStyleSheet(f"color: {theme.color('sell')}; font-size: 11px; background: transparent;")

    def _page_theme(self):
        page, layout = self._page_container()
        self._title(layout, "Appearance", "settings")
        self._subtitle(layout, "Choose your visual style. Auto mode matches your Windows theme setting.")

        layout.addSpacing(12)

        # Theme preview cards
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Auto (match Windows)", "Dark Mode", "Light Mode"])
        self.theme_combo.setStyleSheet(f"""
            QComboBox {{ background-color: {theme.color('bg_card')}; border: 1px solid {theme.color('border')};
                padding: 12px 16px; border-radius: 8px; color: {theme.color('text_primary')}; font-size: 13px; }}
            QComboBox:hover {{ border-color: {BRAND_PRIMARY}; }}
        """)
        layout.addWidget(self.theme_combo)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _page_profile(self):
        page, layout = self._page_container()
        self._title(layout, "Hardware Profile", "shield")
        self._subtitle(layout, "Choose how much AI power to use based on your machine. "
                       "Balanced is recommended for most laptops.")

        layout.addSpacing(8)

        from ..profiles import PRESETS
        self.profile_combo = QComboBox()
        self.profile_combo.setStyleSheet(f"""
            QComboBox {{ background-color: {theme.color('bg_card')}; border: 1px solid {theme.color('border')};
                padding: 12px 16px; border-radius: 8px; color: {theme.color('text_primary')}; font-size: 12px; }}
        """)
        for name, preset in PRESETS.items():
            addons_on = sum(1 for v in preset["addons"].values() if v)
            self.profile_combo.addItem(f"{name}  —  {addons_on} addons, {preset['scanner_workers']} threads", name)
        self.profile_combo.setCurrentIndex(1)  # Balanced
        layout.addWidget(self.profile_combo)

        # Profile descriptions — compact list
        descs = [
            ("Max", "All AI models + addons · 16GB+ RAM"),
            ("Balanced", "Core AI + lightweight addons · Recommended"),
            ("Light", "API-only addons, no heavy models · Fast"),
            ("Minimal", "Core engine only · Fastest possible"),
        ]
        for name, desc in descs:
            row = QHBoxLayout()
            n = QLabel(f"  {name}")
            n.setStyleSheet(f"color: {BRAND_PRIMARY}; font-weight: 700; font-size: 11px; background: transparent;")
            n.setFixedWidth(80)
            row.addWidget(n)
            d = QLabel(desc)
            d.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 10px; background: transparent;")
            row.addWidget(d, 1)
            layout.addLayout(row)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _page_done(self):
        page, layout = self._page_container()

        layout.addSpacing(20)

        # Big checkmark
        check = QLabel()
        check.setPixmap(StockyIcons.get("check", 48, BRAND_ACCENT))
        check.setAlignment(Qt.AlignCenter)
        check.setStyleSheet("background: transparent;")
        layout.addWidget(check)

        done_lbl = QLabel("You're all set!")
        done_lbl.setFont(QFont(FONT_FAMILY, 22, QFont.Bold))
        done_lbl.setAlignment(Qt.AlignCenter)
        done_lbl.setStyleSheet(f"color: {BRAND_ACCENT}; background: transparent;")
        layout.addWidget(done_lbl)

        sub = QLabel("Your trading suite is ready. Here's how to get started:")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {theme.color('text_secondary')}; font-size: 12px; background: transparent;")
        layout.addWidget(sub)

        layout.addSpacing(12)

        tips = [
            ("scan",  "Open the Scanner tab and run your first multi-stock scan"),
            ("bolt",  "Try Day Trade for deep single-stock analysis"),
            ("dashboard", "Check the Dashboard for your portfolio overview"),
            ("test",  "Run diagnostics in Testing to verify everything works"),
        ]
        for icon_name, text in tips:
            layout.addWidget(_FeatureCard(icon_name, text))

        layout.addStretch()

        footer = QLabel(f"{APP_NAME} v{APP_VERSION}  ·  {APP_URL}")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 9px; background: transparent;")
        layout.addWidget(footer)

        page.setLayout(layout)
        return page

    # ── Navigation ────────────────────────────────────────────────────────

    def _update_nav(self):
        idx = self.stack.currentIndex()
        total = self.stack.count()
        self.step_label.setText(f"Step {idx + 1} of {total}")
        self.step_indicator.set_step(idx)
        self.back_btn.setVisible(idx > 0)
        self.skip_btn.setVisible(idx < total - 1)
        if idx == total - 1:
            self.next_btn.setText("Launch Stocky Suite")
            self.next_btn.setStyleSheet(f"""
                QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {BRAND_ACCENT}, stop:1 #34d399); color: white;
                    padding: 12px 32px; border-radius: 8px; border: none; font-weight: 700; font-size: 14px; }}
                QPushButton:hover {{ background: {BRAND_ACCENT}; }}
            """)
        else:
            self.next_btn.setText("Continue")

        # Animate page transition
        page = self.stack.currentWidget()
        if page:
            effect = QGraphicsOpacityEffect(page)
            page.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity")
            anim.setDuration(250)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.start()
            page._fade = anim

    def _next(self):
        idx = self.stack.currentIndex()
        if idx == self.stack.count() - 1:
            self._finish()
            return

        # Save data from current page
        if idx == 1:
            s = _load()
            s["alpaca_api_key"] = self.api_key_input.text().strip()
            s["alpaca_secret_key"] = self.secret_input.text().strip()
            _save(s)
        if idx == 2:
            s = _load()
            s["theme"] = {0: "auto", 1: "dark", 2: "light"}.get(self.theme_combo.currentIndex(), "auto")
            _save(s)
        if idx == 3:
            from ..profiles import apply_profile
            apply_profile(self.profile_combo.currentData())

        self.stack.setCurrentIndex(idx + 1)
        self._update_nav()

    def _back(self):
        idx = self.stack.currentIndex()
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
            self._update_nav()

    def _skip(self):
        """Skip setup — app still works, wizard shows again next launch."""
        self.reject()

    def _finish(self):
        s = _load()
        s["setup_complete"] = True
        _save(s)
        self.accept()
