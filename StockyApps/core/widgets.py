"""
Custom Widget Library — premium UI components for Stocky Suite.

Hand-crafted widgets with animations, gradients, and visual polish
that make the app feel like a professional trading terminal.
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QGraphicsDropShadowEffect, QProgressBar, QFrame,
)
from PyQt5.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, pyqtProperty,
    QTimer, QSize, QRect, QPoint,
)
from PyQt5.QtGui import (
    QPainter, QColor, QLinearGradient, QPen, QFont, QBrush,
    QPainterPath, QRadialGradient,
)
import math


# ─── Animated Stat Card ──────────────────────────────────────────────────────
class StatCard(QFrame):
    """
    A premium stat display card with label, value, and subtle glow.
    Used on the Dashboard for portfolio value, buying power, etc.
    """

    def __init__(self, title="", value="--", accent_color="#0ea5e9", parent=None):
        super().__init__(parent)
        self._accent = QColor(accent_color)
        self._hover = False
        self.setFixedHeight(90)
        self.setMinimumWidth(160)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("background: transparent; border: none;")

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600; background: transparent; border: none;")
        layout.addWidget(self._title_lbl)

        self._value_lbl = QLabel(value)
        self._value_lbl.setStyleSheet(f"color: {accent_color}; font-size: 20px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(self._value_lbl)

        self.setLayout(layout)

    def set_value(self, value, color=None):
        self._value_lbl.setText(value)
        if color:
            self._value_lbl.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold; background: transparent; border: none;")

    def set_title(self, title):
        self._title_lbl.setText(title)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Card background
        r = self.rect().adjusted(1, 1, -1, -1)
        bg = QColor("#1e2130") if not self._hover else QColor("#252836")
        painter.setBrush(bg)
        painter.setPen(QPen(QColor("#2a2d3a"), 1))
        painter.drawRoundedRect(r, 10, 10)

        # Accent bar on left
        accent_rect = QRect(r.x(), r.y() + 8, 3, r.height() - 16)
        painter.setBrush(self._accent)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(accent_rect, 1, 1)

        painter.end()
        super().paintEvent(event)

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()


# ─── Gradient Progress Bar ───────────────────────────────────────────────────
class GradientProgressBar(QWidget):
    """Progress bar with animated gradient fill."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self._max = 100
        self.setFixedHeight(8)

    def setValue(self, val):
        self._value = max(0, min(val, self._max))
        self.update()

    def setRange(self, min_val, max_val):
        self._max = max_val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Track background
        painter.setBrush(QColor("#252836"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, h // 2, h // 2)

        # Filled portion with gradient
        if self._value > 0 and self._max > 0:
            fill_w = int(w * self._value / self._max)
            grad = QLinearGradient(0, 0, fill_w, 0)
            grad.setColorAt(0, QColor("#0ea5e9"))
            grad.setColorAt(1, QColor("#10b981"))
            painter.setBrush(grad)
            painter.drawRoundedRect(0, 0, fill_w, h, h // 2, h // 2)

        painter.end()


# ─── Signal Badge ─────────────────────────────────────────────────────────────
class SignalBadge(QWidget):
    """
    Large, attention-grabbing BUY/SELL/HOLD badge with pulse animation.
    Used on Day Trade and Scanner panels.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._action = "—"
        self._color = QColor("#64748b")
        self._pulse = 0.0
        self._confidence = 0.0
        self.setFixedHeight(70)
        self.setMinimumWidth(200)

        # Pulse animation
        self._pulse_timer = QTimer()
        self._pulse_timer.timeout.connect(self._tick_pulse)
        self._pulse_timer.setInterval(50)
        self._pulse_phase = 0.0

    def set_signal(self, action, confidence=0.0):
        self._action = action
        self._confidence = confidence
        colors = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#f59e0b"}
        self._color = QColor(colors.get(action, "#64748b"))

        if action in ("BUY", "SELL"):
            self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
            self._pulse = 0.0

        self.update()

    def _tick_pulse(self):
        self._pulse_phase += 0.15
        self._pulse = abs(math.sin(self._pulse_phase)) * 0.3
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        r = self.rect().adjusted(4, 4, -4, -4)

        # Glow effect behind badge
        if self._pulse > 0:
            glow_color = QColor(self._color)
            glow_color.setAlphaF(self._pulse * 0.4)
            painter.setBrush(glow_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(r.adjusted(-3, -3, 3, 3), 14, 14)

        # Badge background
        bg_color = QColor(self._color)
        bg_color.setAlphaF(0.15 + self._pulse * 0.1)
        painter.setBrush(bg_color)
        painter.setPen(QPen(self._color, 2))
        painter.drawRoundedRect(r, 12, 12)

        # Action text
        painter.setPen(self._color)
        painter.setFont(QFont("Segoe UI", 22, QFont.Bold))
        painter.drawText(r, Qt.AlignCenter, self._action)

        # Confidence in corner
        if self._confidence > 0:
            painter.setFont(QFont("Segoe UI", 10))
            painter.setPen(QColor(self._color.red(), self._color.green(), self._color.blue(), 180))
            painter.drawText(r.adjusted(0, 0, -10, -5), Qt.AlignRight | Qt.AlignBottom, f"{self._confidence:.0%}")

        painter.end()


# ─── Divider Line ─────────────────────────────────────────────────────────────
class GradientDivider(QWidget):
    """A subtle gradient line divider."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(1)

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0, QColor(42, 45, 58, 0))
        grad.setColorAt(0.3, QColor(14, 165, 233, 100))
        grad.setColorAt(0.7, QColor(16, 185, 129, 100))
        grad.setColorAt(1, QColor(42, 45, 58, 0))
        painter.setPen(QPen(QBrush(grad), 1))
        painter.drawLine(0, 0, w, 0)
        painter.end()


# ─── Section Header ──────────────────────────────────────────────────────────
class SectionHeader(QWidget):
    """A styled section header with accent underline."""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self._text = text

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Text
        painter.setPen(QColor("#e2e8f0"))
        painter.setFont(QFont("Segoe UI", 13, QFont.Bold))
        painter.drawText(0, 0, self.width(), 24, Qt.AlignLeft | Qt.AlignVCenter, self._text)

        # Underline gradient
        w = self.width()
        grad = QLinearGradient(0, 0, w * 0.4, 0)
        grad.setColorAt(0, QColor("#0ea5e9"))
        grad.setColorAt(1, QColor("#0ea5e9", 0))
        painter.setPen(QPen(QBrush(grad), 2))
        painter.drawLine(0, 28, int(w * 0.4), 28)

        painter.end()


# ─── Mini Sparkline ──────────────────────────────────────────────────────────
class Sparkline(QWidget):
    """Tiny inline price chart, used in table cells or stat cards."""

    def __init__(self, data=None, color="#0ea5e9", parent=None):
        super().__init__(parent)
        self._data = data or []
        self._color = QColor(color)
        self.setFixedSize(80, 28)

    def set_data(self, data, color=None):
        self._data = data
        if color:
            self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        if not self._data or len(self._data) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        pad = 2

        mn = min(self._data)
        mx = max(self._data)
        rng = mx - mn if mx != mn else 1

        points = []
        for i, val in enumerate(self._data):
            x = pad + (i / (len(self._data) - 1)) * (w - 2 * pad)
            y = h - pad - ((val - mn) / rng) * (h - 2 * pad)
            points.append((x, y))

        # Draw line
        pen = QPen(self._color, 1.5)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        for i in range(len(points) - 1):
            painter.drawLine(int(points[i][0]), int(points[i][1]),
                           int(points[i+1][0]), int(points[i+1][1]))

        # Fill under line
        path = QPainterPath()
        path.moveTo(points[0][0], h)
        for x, y in points:
            path.lineTo(x, y)
        path.lineTo(points[-1][0], h)
        path.closeSubpath()

        fill_color = QColor(self._color)
        fill_color.setAlphaF(0.08)
        painter.fillPath(path, fill_color)

        painter.end()
