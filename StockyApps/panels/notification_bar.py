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

# MAIN WINDOW
# ═════════════════════════════════════════════════════════════════════════════

class _NotificationBar(QWidget):
    """
    Custom-painted notification bar with:
    - Slow-moving gradient background
    - Market open/closed indicator
    - Icon that pulses briefly on new messages
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(32)
        self.setMaximumHeight(34)
        self._phase = 0.0
        self._msg = f"{APP_NAME} v{APP_VERSION} — Ready"
        self._time = ""
        self._icon_name = "check"
        self._icon_color = BRAND_PRIMARY
        self._text_color = QColor(TEXT_SECONDARY)
        self._icon_pulse = 0.0   # 0-1, decays over time (glow)
        self._icon_bounce = 0.0  # 0-1, decays (vertical bounce)
        self._text_slide = 0.0   # 0-1, decays (text slides in from right)
        self._market_open = False

        # Notification history for bell
        self._notif_history = []  # list of (time, msg, level)
        self._unread_count = 0
        self._bell_pulse = 0.0
        self._click_url = None

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

        # Market status check
        self._check_market()
        self._market_timer = QTimer(self)
        self._market_timer.timeout.connect(self._check_market)
        self._market_timer.start(60000)  # Check every minute

    def _check_market(self):
        try:
            est = pytz.timezone("US/Eastern")
            now = datetime.now(est)
            self._market_open = (
                now.weekday() < 5
                and now.hour >= 9 and (now.hour > 9 or now.minute >= 30)
                and now.hour < 16
            )
        except Exception:
            self._market_open = False

    def _tick(self):
        self._phase += 0.008
        if self._icon_pulse > 0:
            self._icon_pulse = max(0, self._icon_pulse - 0.025)
        if self._icon_bounce > 0:
            self._icon_bounce = max(0, self._icon_bounce - 0.04)
        if self._text_slide > 0:
            self._text_slide = max(0, self._text_slide - 0.05)
        if self._bell_pulse > 0:
            self._bell_pulse = max(0, self._bell_pulse - 0.02)
        self.update()

    def _reset_bar(self):
        """Reset status bar to default after clearing notifications."""
        from core.branding import APP_NAME, APP_VERSION
        self._msg = f"{APP_NAME} v{APP_VERSION} — Ready"
        self._time = ""
        self._click_url = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_click_url(self, url):
        """Set a URL that opens when the message area is clicked."""
        self._click_url = url
        self.setCursor(Qt.PointingHandCursor if url else Qt.ArrowCursor)
        self.update()

    def show_message(self, msg, level="info", click_url=None):
        from core.ui.icons import StockyIcons
        # Set click URL if provided, otherwise clear unless this is an update message
        if click_url:
            self._click_url = click_url
            self.setCursor(Qt.PointingHandCursor)
        elif "click to download" not in msg:
            self._click_url = None
            self.setCursor(Qt.ArrowCursor)
        icon_map = {
            "info":   ("check",    BRAND_PRIMARY, TEXT_SECONDARY),
            "trade":  ("bolt",     BRAND_ACCENT,  BRAND_ACCENT),
            "warn":   ("warning",  COLOR_HOLD,    COLOR_HOLD),
            "error":  ("x_mark",   COLOR_SELL,    COLOR_SELL),
            "system": ("settings", TEXT_MUTED,    TEXT_MUTED),
        }
        self._icon_name, self._icon_color, text_hex = icon_map.get(level, ("check", TEXT_MUTED, TEXT_MUTED))
        self._text_color = QColor(text_hex)
        self._msg = msg
        self._time = datetime.now().strftime("%H:%M:%S")
        self._icon_pulse = 1.0
        self._icon_bounce = 1.0
        self._text_slide = 1.0

        # Track in history
        self._notif_history.append((self._time, msg, level))
        self._notif_history = self._notif_history[-50:]  # Keep last 50
        self._unread_count += 1
        self._bell_pulse = 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Animated gradient background — slow dark-to-darker drift
        import math
        shift = (math.sin(self._phase) + 1) / 2
        grad = QLinearGradient(0, 0, w, 0)
        c1 = QColor(22, 25, 35)
        c2 = QColor(28, 32, 45)
        c3 = QColor(18, 20, 30)
        grad.setColorAt(0, c1)
        grad.setColorAt(0.3 + shift * 0.2, c2)
        grad.setColorAt(0.7 - shift * 0.2, c1)
        grad.setColorAt(1, c3)
        painter.fillRect(0, 0, w, h, grad)

        # Top border — subtle thin line
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawLine(0, 0, w, 0)

        x = 14

        # Market status dot
        dot_color = QColor(BRAND_ACCENT) if self._market_open else QColor(COLOR_SELL)
        dot_color.setAlphaF(0.7 + math.sin(self._phase * 3) * 0.15)
        painter.setBrush(dot_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(x), int(h / 2 - 4), 8, 8)
        x += 14

        # Market label
        painter.setPen(QColor(TEXT_MUTED))
        painter.setFont(QFont(FONT_MONO, 8))
        market_text = "OPEN" if self._market_open else "CLOSED"
        painter.drawText(int(x), 0, 50, h, Qt.AlignVCenter, market_text)
        x += 55

        # Separator
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawLine(int(x), 6, int(x), h - 6)
        x += 12

        # Icon with bounce + pulse glow
        from core.ui.icons import StockyIcons
        import math as _m
        icon_px = StockyIcons.get(self._icon_name, 16, self._icon_color)

        # Bounce: icon pops up then settles (elastic ease-out)
        bounce_offset = 0
        if self._icon_bounce > 0:
            # Elastic bounce: goes up, overshoots, settles
            t = 1.0 - self._icon_bounce
            bounce_offset = int(_m.sin(t * _m.pi * 2.5) * (1.0 - t) * 8)

        # Glow behind icon
        if self._icon_pulse > 0:
            glow_c = QColor(self._icon_color)
            glow_c.setAlphaF(self._icon_pulse * 0.35)
            painter.setBrush(glow_c)
            painter.setPen(Qt.NoPen)
            glow_r = 10 + self._icon_pulse * 4
            painter.drawEllipse(int(x + 8 - glow_r), int(h / 2 - glow_r + bounce_offset), int(glow_r * 2), int(glow_r * 2))

        painter.drawPixmap(int(x), int(h / 2 - 8 + bounce_offset), icon_px)
        x += 22

        # Message text — slides in from right
        text_offset = int(self._text_slide * 40)  # Starts 40px to the right, slides to 0
        text_alpha = 1.0 - self._text_slide * 0.6  # Fades in as it slides

        tc = QColor(self._text_color)
        tc.setAlphaF(max(0.3, text_alpha))
        painter.setPen(tc)
        painter.setFont(QFont(FONT_FAMILY, 10))
        # If clickable, use underline font to look like a link
        msg_font = QFont(FONT_FAMILY, 10)
        if hasattr(self, '_click_url') and self._click_url:
            msg_font.setUnderline(True)
        painter.setFont(msg_font)
        painter.drawText(int(x + text_offset), 0, w - x - 70, h, Qt.AlignVCenter, self._msg)

        # Timestamp
        if self._time:
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont(FONT_MONO, 8))
            painter.drawText(w - 90, 0, 50, h, Qt.AlignVCenter | Qt.AlignRight, self._time)

        # Bell icon with unread badge
        bell_x = w - 32
        bell_px = StockyIcons.get("bell", 14, BRAND_PRIMARY if self._unread_count > 0 else TEXT_MUTED)

        # Bell pulse glow
        if self._bell_pulse > 0:
            glow = QColor(BRAND_PRIMARY)
            glow.setAlphaF(self._bell_pulse * 0.3)
            painter.setBrush(glow)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(bell_x - 1), int(h/2 - 9), 18, 18)

        painter.drawPixmap(int(bell_x), int(h/2 - 7), bell_px)

        # Unread count badge
        if self._unread_count > 0:
            painter.setBrush(QColor(COLOR_SELL))
            painter.setPen(Qt.NoPen)
            badge_r = 7
            painter.drawEllipse(int(bell_x + 8), int(h/2 - 10), badge_r * 2, badge_r * 2)
            painter.setPen(QColor("white"))
            painter.setFont(QFont(FONT_FAMILY, 7, QFont.Bold))
            count_text = str(self._unread_count) if self._unread_count < 100 else "99+"
            painter.drawText(int(bell_x + 8), int(h/2 - 10), badge_r * 2, badge_r * 2, Qt.AlignCenter, count_text)

        painter.end()

    def mousePressEvent(self, event):
        """Click bell = notification overlay. Click message = open URL if set."""
        bell_x = self.width() - 32
        if event.x() >= bell_x - 5:
            self._show_notification_overlay()
        elif self._click_url:
            import webbrowser, os
            # Use os.startfile on Windows for reliable browser launch
            try:
                os.startfile(self._click_url)
            except Exception:
                webbrowser.open(self._click_url)
            self._click_url = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
        super().mousePressEvent(event)

    def _show_notification_overlay(self):
        """Show polished notification history with filters."""
        self._unread_count = 0
        self.update()

        if not self._notif_history:
            return

        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QComboBox
        from core.ui.theme import theme as _theme
        from core.ui.icons import StockyIcons

        parent = self
        while parent.parent():
            parent = parent.parent()

        dlg = QDialog(parent)
        dlg.setWindowTitle("Notifications")
        dlg.setMinimumSize(500, 400)
        dlg.setStyleSheet(f"QDialog {{ background-color: {_theme.color('bg_base')}; }}")

        lay = QVBoxLayout()

        # Header with count + filter
        hdr = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(StockyIcons.get("bell", 20, BRAND_PRIMARY))
        icon_lbl.setStyleSheet("background: transparent;")
        hdr.addWidget(icon_lbl)
        title = QLabel(f"Notifications ({len(self._notif_history)})")
        title.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        title.setStyleSheet(f"color: {BRAND_PRIMARY}; background: transparent;")
        hdr.addWidget(title)
        hdr.addStretch()
        filter_cb = QComboBox()
        filter_cb.addItems(["All", "Trades", "Warnings", "Errors", "System"])
        hdr.addWidget(filter_cb)
        lay.addLayout(hdr)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFont(QFont(FONT_MONO, 10))

        def populate(filter_type="All"):
            txt.clear()
            level_map = {"Trades": "trade", "Warnings": "warn", "Errors": "error", "System": "system"}
            filt = level_map.get(filter_type)
            colors = {"info": BRAND_PRIMARY, "trade": BRAND_ACCENT, "warn": COLOR_HOLD, "error": COLOR_SELL, "system": TEXT_MUTED}
            for ts, msg, level in reversed(self._notif_history):
                if filt and level != filt:
                    continue
                c = colors.get(level, TEXT_SECONDARY)
                txt.append(
                    f'<span style="color:{TEXT_MUTED}">{ts}</span> '
                    f'<span style="color:{c};font-weight:bold">[{level.upper()}]</span> '
                    f'<span style="color:{_theme.color("text_primary")}">{msg}</span>'
                )

        populate()
        filter_cb.currentTextChanged.connect(populate)
        lay.addWidget(txt)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet(f"background-color: {COLOR_SELL}; font-size: 11px;")
        clear_btn.clicked.connect(lambda: (self._notif_history.clear(), self._reset_bar(), dlg.accept()))
        btn_row.addWidget(clear_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        dlg.setLayout(lay)
        dlg.exec_()


