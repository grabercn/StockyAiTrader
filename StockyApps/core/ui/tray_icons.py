# -*- coding: utf-8 -*-
"""
Dynamic Tray Icons — simple, bold state icons for the system tray.

At tray size (16-24px), detail doesn't work. Instead we generate
clean, high-contrast icons that are instantly readable at any size.

States:
- idle:      Original app icon (unchanged)
- agent:     Blue circle with white bot antenna
- scanning:  Purple circle with white radar arc
- trading:   Green circle with white up-arrow
- error:     Red circle with white X
"""

import os
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QConicalGradient
from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QTimer
import math


ICON_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "icon.ico")
# Generate at multiple sizes so Windows picks the best one
SIZES = [16, 24, 32, 48]


def _make_pixmap(size, bg_color, draw_fn):
    """Create a clean circular icon at the given size."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)

    s = size
    margin = max(1, s // 8)
    r = (s - margin * 2) // 2

    # Filled circle background
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(bg_color))
    p.drawEllipse(QPoint(s // 2, s // 2), r, r)

    # Draw the state symbol
    draw_fn(p, s)

    p.end()
    return pm


def _draw_agent(p, s):
    """Bot/robot — simple antenna + dot."""
    cx, cy = s // 2, s // 2
    w = max(1, s // 8)

    p.setPen(QPen(QColor(255, 255, 255, 230), w, Qt.SolidLine, Qt.RoundCap))
    # Antenna stem
    p.drawLine(cx, cy + s // 6, cx, cy - s // 5)
    # Antenna dot
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(255, 255, 255, 240))
    dot_r = max(2, s // 8)
    p.drawEllipse(QPoint(cx, cy - s // 4), dot_r, dot_r)
    # Eyes — two dots
    eye_y = cy + s // 12
    eye_sp = max(2, s // 5)
    eye_r = max(1, s // 12)
    p.drawEllipse(QPoint(cx - eye_sp // 2, eye_y), eye_r, eye_r)
    p.drawEllipse(QPoint(cx + eye_sp // 2, eye_y), eye_r, eye_r)


def _draw_scanning(p, s, frame=0):
    """Radar sweep — arc + center dot."""
    cx, cy = s // 2, s // 2
    w = max(1, s // 8)
    r = s // 3

    # Rotating arc (90 degrees)
    start_angle = (frame * 45) % 360
    p.setPen(QPen(QColor(255, 255, 255, 220), w, Qt.SolidLine, Qt.RoundCap))
    arc_rect = QRect(cx - r, cy - r, r * 2, r * 2)
    p.drawArc(arc_rect, start_angle * 16, 90 * 16)

    # Center dot
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(255, 255, 255, 240))
    p.drawEllipse(QPoint(cx, cy), max(1, s // 10), max(1, s // 10))


def _draw_trading(p, s):
    """Up arrow — simple upward triangle."""
    cx, cy = s // 2, s // 2
    w = max(2, s // 6)

    p.setPen(QPen(QColor(255, 255, 255, 230), w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    # Arrow shaft
    top = cy - s // 4
    bot = cy + s // 4
    p.drawLine(cx, top, cx, bot)
    # Arrow head
    head = max(2, s // 5)
    p.drawLine(cx, top, cx - head, top + head)
    p.drawLine(cx, top, cx + head, top + head)


def _draw_error(p, s):
    """X mark."""
    cx, cy = s // 2, s // 2
    w = max(2, s // 6)
    d = s // 5

    p.setPen(QPen(QColor(255, 255, 255, 230), w, Qt.SolidLine, Qt.RoundCap))
    p.drawLine(cx - d, cy - d, cx + d, cy + d)
    p.drawLine(cx + d, cy - d, cx - d, cy + d)


def make_icon(state="idle", frame=0):
    """Generate a QIcon for the given state with multiple size variants."""
    if state == "idle":
        if os.path.exists(ICON_FILE):
            return QIcon(ICON_FILE)
        # Fallback
        pm = QPixmap(32, 32)
        pm.fill(QColor("#3b82f6"))
        return QIcon(pm)

    colors = {
        "agent":    "#3b82f6",  # Blue
        "scanning": "#8b5cf6",  # Purple
        "trading":  "#10b981",  # Green
        "error":    "#ef4444",  # Red
    }
    bg = colors.get(state, "#3b82f6")

    draw_fns = {
        "agent":    lambda p, s: _draw_agent(p, s),
        "scanning": lambda p, s: _draw_scanning(p, s, frame),
        "trading":  lambda p, s: _draw_trading(p, s),
        "error":    lambda p, s: _draw_error(p, s),
    }
    draw = draw_fns.get(state, lambda p, s: None)

    icon = QIcon()
    for sz in SIZES:
        icon.addPixmap(_make_pixmap(sz, bg, draw))
    return icon


class TrayIconAnimator:
    """Manages animated tray icon state transitions."""

    def __init__(self, tray_icon):
        self._tray = tray_icon
        self._state = "idle"
        self._frame = 0
        self._base_icon = None

        if tray_icon:
            self._base_icon = tray_icon.icon()

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)

    def set_state(self, state):
        if state == self._state:
            return
        self._state = state
        self._frame = 0

        if state == "scanning":
            self._timer.start(400)
        else:
            self._timer.stop()

        self._apply()

    def _tick(self):
        self._frame += 1
        self._apply()

    def _apply(self):
        if not self._tray:
            return
        if self._state == "idle" and self._base_icon:
            self._tray.setIcon(self._base_icon)
        else:
            self._tray.setIcon(make_icon(self._state, self._frame))
