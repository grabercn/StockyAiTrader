# -*- coding: utf-8 -*-
"""
Dynamic Tray Icons — detailed, colorful state icons for the system tray.

States:
- idle:       Original app icon (unchanged)
- agent:      Blue circle with white robot face
- scanning:   Purple circle with animated radar sweep
- trading:    Green circle with dollar sign
- buying:     Bright green circle with up-arrow + plus
- selling:    Orange circle with down-arrow
- error:      Red circle with warning triangle
- waiting:    Grey-blue circle with clock
- profit:     Gold circle with star (after profitable trade)
- market_closed: Dark grey circle with moon
"""

import os
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QTimer
import math


ICON_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "icon.ico")
SIZES = [16, 24, 32, 48]


def _make_pixmap(size, bg_color, draw_fn, border_color=None):
    """Create a circular icon with optional border ring."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)

    s = size
    margin = max(1, s // 8)
    r = (s - margin * 2) // 2

    # Optional outer ring
    if border_color:
        p.setPen(QPen(QColor(border_color), max(1, s // 12)))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPoint(s // 2, s // 2), r, r)
        r -= max(1, s // 10)

    # Filled circle background
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(bg_color))
    p.drawEllipse(QPoint(s // 2, s // 2), r, r)

    # Draw the state symbol
    draw_fn(p, s)

    p.end()
    return pm


def _draw_agent(p, s):
    """Robot face — antenna + eyes + mouth."""
    cx, cy = s // 2, s // 2
    w = max(1, s // 8)

    p.setPen(QPen(QColor(255, 255, 255, 230), w, Qt.SolidLine, Qt.RoundCap))
    # Antenna stem
    p.drawLine(cx, cy - s // 12, cx, cy - s // 4)
    # Antenna dot (glowing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(100, 255, 100, 200))
    dot_r = max(2, s // 8)
    p.drawEllipse(QPoint(cx, cy - s // 4), dot_r, dot_r)
    # Eyes — two bright dots
    p.setBrush(QColor(255, 255, 255, 240))
    eye_y = cy + s // 20
    eye_sp = max(2, s // 5)
    eye_r = max(1, s // 10)
    p.drawEllipse(QPoint(cx - eye_sp // 2, eye_y), eye_r, eye_r)
    p.drawEllipse(QPoint(cx + eye_sp // 2, eye_y), eye_r, eye_r)
    # Mouth — small horizontal line
    mouth_w = max(1, s // 12)
    p.setPen(QPen(QColor(255, 255, 255, 180), mouth_w, Qt.SolidLine, Qt.RoundCap))
    mouth_y = cy + s // 5
    p.drawLine(cx - eye_sp // 3, mouth_y, cx + eye_sp // 3, mouth_y)


def _draw_scanning(p, s, frame=0):
    """Radar sweep — rotating arc + center dot + blips."""
    cx, cy = s // 2, s // 2
    w = max(1, s // 8)
    r = s // 3

    # Rotating arc (120 degrees, wider sweep)
    start_angle = (frame * 30) % 360
    p.setPen(QPen(QColor(255, 255, 255, 220), w, Qt.SolidLine, Qt.RoundCap))
    arc_rect = QRect(cx - r, cy - r, r * 2, r * 2)
    p.drawArc(arc_rect, start_angle * 16, 120 * 16)

    # Trailing fade arc
    p.setPen(QPen(QColor(255, 255, 255, 80), max(1, w // 2), Qt.SolidLine, Qt.RoundCap))
    p.drawArc(arc_rect, (start_angle - 60) * 16, 60 * 16)

    # Center dot
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(255, 255, 255, 240))
    p.drawEllipse(QPoint(cx, cy), max(1, s // 10), max(1, s // 10))

    # Blip dot (rotates opposite)
    blip_angle = math.radians((frame * 45 + 180) % 360)
    blip_r = r * 0.6
    bx = int(cx + blip_r * math.cos(blip_angle))
    by = int(cy - blip_r * math.sin(blip_angle))
    p.setBrush(QColor(255, 255, 100, 180))
    p.drawEllipse(QPoint(bx, by), max(1, s // 14), max(1, s // 14))


def _draw_trading(p, s):
    """Dollar sign — clear trading symbol."""
    cx, cy = s // 2, s // 2
    w = max(2, s // 6)

    p.setPen(QPen(QColor(255, 255, 255, 240), w, Qt.SolidLine, Qt.RoundCap))
    # Vertical line through $
    p.drawLine(cx, cy - s // 4, cx, cy + s // 4)
    # S curve (simplified as two arcs)
    arc_r = s // 5
    # Top arc (curves right)
    p.drawArc(QRect(cx - arc_r, cy - s // 5, arc_r * 2, arc_r), 0, 180 * 16)
    # Bottom arc (curves left)
    p.drawArc(QRect(cx - arc_r, cy - arc_r // 2, arc_r * 2, arc_r), 180 * 16, 180 * 16)


def _draw_buying(p, s):
    """Up arrow with plus — buying signal."""
    cx, cy = s // 2, s // 2
    w = max(2, s // 6)

    p.setPen(QPen(QColor(255, 255, 255, 240), w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    # Arrow shaft
    top = cy - s // 4
    bot = cy + s // 6
    p.drawLine(cx, top, cx, bot)
    # Arrow head
    head = max(2, s // 5)
    p.drawLine(cx, top, cx - head, top + head)
    p.drawLine(cx, top, cx + head, top + head)
    # Plus sign (small, bottom right)
    plus_x = cx + s // 4
    plus_y = cy + s // 5
    plus_s = max(1, s // 8)
    p.drawLine(plus_x - plus_s, plus_y, plus_x + plus_s, plus_y)
    p.drawLine(plus_x, plus_y - plus_s, plus_x, plus_y + plus_s)


def _draw_selling(p, s):
    """Down arrow — selling signal."""
    cx, cy = s // 2, s // 2
    w = max(2, s // 6)

    p.setPen(QPen(QColor(255, 255, 255, 240), w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    # Arrow shaft
    top = cy - s // 6
    bot = cy + s // 4
    p.drawLine(cx, top, cx, bot)
    # Arrow head
    head = max(2, s // 5)
    p.drawLine(cx, bot, cx - head, bot - head)
    p.drawLine(cx, bot, cx + head, bot - head)


def _draw_error(p, s):
    """Warning triangle with exclamation."""
    cx, cy = s // 2, s // 2
    w = max(2, s // 7)

    p.setPen(QPen(QColor(255, 255, 255, 240), w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    # Triangle
    top = cy - s // 4
    bot = cy + s // 5
    left = cx - s // 4
    right = cx + s // 4
    p.drawLine(cx, top, left, bot)
    p.drawLine(left, bot, right, bot)
    p.drawLine(right, bot, cx, top)
    # Exclamation mark
    p.drawLine(cx, cy - s // 10, cx, cy + s // 16)
    # Dot
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(255, 255, 255, 240))
    p.drawEllipse(QPoint(cx, cy + s // 7), max(1, s // 14), max(1, s // 14))


def _draw_waiting(p, s):
    """Clock face — waiting for next cycle."""
    cx, cy = s // 2, s // 2
    w = max(1, s // 8)
    r = s // 3

    # Clock circle
    p.setPen(QPen(QColor(255, 255, 255, 200), max(1, w // 2)))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPoint(cx, cy), r, r)

    # Clock hands
    p.setPen(QPen(QColor(255, 255, 255, 240), w, Qt.SolidLine, Qt.RoundCap))
    # Hour hand (points to ~10 o'clock)
    p.drawLine(cx, cy, cx - r // 2, cy - r // 2)
    # Minute hand (points to 12)
    p.drawLine(cx, cy, cx, cy - r * 3 // 4)

    # Center dot
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(255, 255, 255, 240))
    p.drawEllipse(QPoint(cx, cy), max(1, s // 14), max(1, s // 14))


def _draw_profit(p, s):
    """Star — profitable trade completed."""
    cx, cy = s // 2, s // 2
    w = max(1, s // 8)
    r_outer = s // 3
    r_inner = s // 6

    p.setPen(Qt.NoPen)
    p.setBrush(QColor(255, 255, 255, 240))

    # 5-pointed star
    points = []
    for i in range(10):
        angle = math.radians(-90 + i * 36)
        r = r_outer if i % 2 == 0 else r_inner
        x = cx + int(r * math.cos(angle))
        y = cy + int(r * math.sin(angle))
        points.append(QPointF(x, y))

    from PyQt5.QtGui import QPolygonF
    p.drawPolygon(QPolygonF(points))


def _draw_market_closed(p, s):
    """Zzz — market is closed (sleep)."""
    cx, cy = s // 2, s // 2
    w = max(1, s // 7)

    p.setPen(QPen(QColor(255, 255, 255, 230), w, Qt.SolidLine, Qt.RoundCap))
    # Big Z
    zw = s // 4
    zt = cy - s // 6
    zb = cy + s // 6
    p.drawLine(cx - zw, zt, cx + zw, zt)  # Top bar
    p.drawLine(cx + zw, zt, cx - zw, zb)  # Diagonal
    p.drawLine(cx - zw, zb, cx + zw, zb)  # Bottom bar


# Color definitions
_COLORS = {
    "agent":         "#3b82f6",  # Blue
    "scanning":      "#8b5cf6",  # Purple
    "trading":       "#10b981",  # Emerald
    "buying":        "#22c55e",  # Bright green
    "selling":       "#f97316",  # Orange
    "error":         "#ef4444",  # Red
    "waiting":       "#64748b",  # Slate
    "profit":        "#eab308",  # Gold
    "market_closed": "#475569",  # Medium slate (visible in tray)
}

_BORDERS = {
    "profit": "#fbbf24",
    "error": "#fca5a5",
}

_DRAW_FNS = {
    "agent":         lambda p, s: _draw_agent(p, s),
    "scanning":      None,  # Handled specially for animation
    "trading":       lambda p, s: _draw_trading(p, s),
    "buying":        lambda p, s: _draw_buying(p, s),
    "selling":       lambda p, s: _draw_selling(p, s),
    "error":         lambda p, s: _draw_error(p, s),
    "waiting":       lambda p, s: _draw_waiting(p, s),
    "profit":        lambda p, s: _draw_profit(p, s),
    "market_closed": lambda p, s: _draw_market_closed(p, s),
}


def make_icon(state="idle", frame=0):
    """Generate a QIcon for the given state with multiple size variants."""
    if state == "idle":
        if os.path.exists(ICON_FILE):
            return QIcon(ICON_FILE)
        pm = QPixmap(32, 32)
        pm.fill(QColor("#3b82f6"))
        return QIcon(pm)

    bg = _COLORS.get(state, "#3b82f6")
    border = _BORDERS.get(state)

    if state == "scanning":
        draw = lambda p, s: _draw_scanning(p, s, frame)
    else:
        draw = _DRAW_FNS.get(state, lambda p, s: None)

    icon = QIcon()
    for sz in SIZES:
        icon.addPixmap(_make_pixmap(sz, bg, draw, border_color=border))
    return icon


class TrayIconAnimator:
    """Manages animated tray icon state transitions with auto-revert."""

    def __init__(self, tray_icon):
        self._tray = tray_icon
        self._state = "idle"
        self._frame = 0
        self._base_icon = None

        if tray_icon:
            self._base_icon = tray_icon.icon()

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)

        # Auto-revert timer for temporary states
        self._revert_timer = QTimer()
        self._revert_timer.setSingleShot(True)
        self._revert_timer.timeout.connect(self._revert)
        self._revert_to = "agent"

    def set_state(self, state, duration_ms=0):
        """
        Set icon state. If duration_ms > 0, auto-reverts to 'agent' after.

        States: idle, agent, scanning, trading, buying, selling,
                error, waiting, profit, market_closed
        """
        if state == self._state and not duration_ms:
            return
        self._state = state
        self._frame = 0

        if state == "scanning":
            self._timer.start(300)  # Faster animation
        else:
            self._timer.stop()

        if duration_ms > 0:
            self._revert_timer.start(duration_ms)

        self._apply()

    def _revert(self):
        """Revert to the base running state."""
        if self._tray:
            self.set_state(self._revert_to)

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
