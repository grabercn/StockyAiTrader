"""
Custom Widget Library — premium UI components for Stocky Suite.

Hand-crafted widgets with animations, gradients, and visual polish
that make the app feel like a professional trading terminal.

All widgets use core.ui.theme for colors so they adapt to light/dark mode.
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QGraphicsDropShadowEffect, QProgressBar, QFrame, QTextEdit,
)
from PyQt5.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, pyqtProperty,
    QTimer, QSize, QRect, QPoint,
)
from core.ui.theme import theme
from PyQt5.QtGui import (
    QPainter, QColor, QLinearGradient, QPen, QFont, QBrush,
    QPainterPath, QRadialGradient,
)
import math


# ─── Animated Stat Card ──────────────────────────────────────────────────────
class StatCard(QFrame):
    """
    A premium stat display card with label, value, and subtle glow.
    Click to open a detail popup. Connect on_clicked to provide the popup.
    """

    def __init__(self, title="", value="--", accent_color="#0ea5e9", parent=None):
        super().__init__(parent)
        self._accent = QColor(accent_color)
        self._hover = False
        self.on_clicked = None  # Set to a callable to handle clicks
        self.setMinimumHeight(70)
        self.setMaximumHeight(100)
        self.setMinimumWidth(130)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("background: transparent; border: none;")

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 11px; font-weight: 600; background: transparent; border: none;")
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

        r = self.rect().adjusted(1, 1, -1, -1)
        bg_key = "bg_hover" if self._hover else "bg_card"
        painter.setBrush(theme.qcolor(bg_key))
        painter.setPen(QPen(theme.qcolor("border"), 1))
        painter.drawRoundedRect(r, 10, 10)

        # Accent bar on left
        accent_rect = QRect(r.x(), r.y() + 8, 3, r.height() - 16)
        painter.setBrush(self._accent)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(accent_rect, 1, 1)

        painter.end()
        super().paintEvent(event)

    def mousePressEvent(self, event):
        if self.on_clicked:
            self.on_clicked()
        super().mousePressEvent(event)

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
        painter.setBrush(theme.qcolor("bg_input"))
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
        self.setMinimumHeight(55)
        self.setMaximumHeight(80)
        self.setMinimumWidth(160)

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
class DetailedProgressBar(QWidget):
    """
    A premium progress bar with:
    - Gradient fill
    - Status text
    - Expandable log dropdown showing each step
    - Animated transitions

    Usage:
        bar = DetailedProgressBar()
        bar.set_progress(25, "Scanning AAPL...", "Fetching 5d of 5m data")
        bar.add_log("AAPL: BUY (85%)")
        bar.set_progress(50, "Scanning TSLA...")
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self._max = 100

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Status label
        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {theme.color('primary')}; font-size: 11px; font-weight: 600; background: transparent;")
        layout.addWidget(self._status)

        # The gradient bar itself
        self._bar = _GradientBar()
        self._bar.setFixedHeight(6)
        layout.addWidget(self._bar)

        # Detail label (secondary info)
        self._detail = QLabel("")
        self._detail.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 9px; background: transparent;")
        layout.addWidget(self._detail)

        # Expandable log area (collapsed by default)
        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setFixedHeight(100)
        self._log_area.setVisible(False)
        self._log_area.setStyleSheet(f"""
            QTextEdit {{
                background-color: {theme.color('bg_input')};
                border: 1px solid {theme.color('border')};
                border-radius: 4px;
                font-family: Consolas;
                font-size: 9px;
                color: {theme.color('text_secondary')};
            }}
        """)
        layout.addWidget(self._log_area)

        # Toggle button for log
        self._toggle = QPushButton("Show details")
        self._toggle.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {theme.color('text_muted')};
                font-size: 9px; border: none; padding: 2px; text-align: left;
            }}
            QPushButton:hover {{ color: {theme.color('primary')}; }}
        """)
        self._toggle.setCursor(Qt.PointingHandCursor)
        self._toggle.clicked.connect(self._toggle_log)
        self._toggle.setVisible(False)
        layout.addWidget(self._toggle)

        self.setLayout(layout)

    def set_progress(self, value, status="", detail=""):
        self._value = max(0, min(value, self._max))
        self._bar.set_value(self._value / self._max)
        if status:
            self._status.setText(status)
        if detail:
            self._detail.setText(detail)
        # Force immediate repaint so progress is visible during scans
        self._bar.repaint()
        self._status.repaint()
        self._detail.repaint()

    def add_log(self, msg):
        """Add a line to the detail log."""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        color = theme.color('primary')
        self._log_area.append(f'<span style="color:{theme.color("text_muted")}">{ts}</span> <span style="color:{color}">{msg}</span>')
        sb = self._log_area.verticalScrollBar()
        sb.setValue(sb.maximum())
        self._toggle.setVisible(True)
        if self._log_area.isVisible():
            self._log_area.repaint()

    def set_visible(self, visible):
        super().setVisible(visible)

    def _toggle_log(self):
        showing = self._log_area.isVisible()
        self._log_area.setVisible(not showing)
        self._toggle.setText("Hide details" if not showing else "Show details")

    def reset(self):
        self._value = 0
        self._bar.set_value(0)
        self._status.setText("")
        self._detail.setText("")
        self._log_area.clear()
        self._toggle.setVisible(False)
        self._log_area.setVisible(False)


class _GradientBar(QWidget):
    """Internal: animated gradient bar with shimmer sweep effect."""

    def __init__(self):
        super().__init__()
        self._pct = 0.0
        self._shimmer = 0.0

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def _tick(self):
        self._shimmer = (self._shimmer + 0.02) % 1.0
        if self._pct > 0:
            self.update()

    def set_value(self, pct):
        self._pct = max(0.0, min(1.0, pct))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        # Track
        painter.setBrush(theme.qcolor("bg_input"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, h // 2, h // 2)

        fill_w = int(w * self._pct)
        if fill_w > 2:
            from PyQt5.QtGui import QLinearGradient, QRadialGradient

            # Base gradient
            grad = QLinearGradient(0, 0, fill_w, 0)
            grad.setColorAt(0, QColor("#0ea5e9"))
            grad.setColorAt(0.5, QColor("#6366f1"))
            grad.setColorAt(1, QColor("#10b981"))
            painter.setBrush(grad)
            painter.drawRoundedRect(0, 0, fill_w, h, h // 2, h // 2)

            # Shimmer sweep — bright highlight that moves across the bar
            shimmer_x = self._shimmer * fill_w
            shimmer = QRadialGradient(shimmer_x, h / 2, 30)
            shimmer.setColorAt(0, QColor(255, 255, 255, 60))
            shimmer.setColorAt(1, QColor(255, 255, 255, 0))
            painter.setBrush(shimmer)
            painter.drawRoundedRect(0, 0, fill_w, h, h // 2, h // 2)

            # Glow at leading edge
            if self._pct < 1.0:
                glow = QRadialGradient(fill_w, h / 2, 12)
                gc = QColor("#10b981")
                gc.setAlphaF(0.5)
                glow.setColorAt(0, gc)
                gc.setAlphaF(0)
                glow.setColorAt(1, gc)
                painter.setBrush(glow)
                painter.drawEllipse(int(fill_w - 12), int(-6), 24, int(h + 12))

        painter.end()


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
        painter.setPen(theme.qcolor("text_heading"))
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
