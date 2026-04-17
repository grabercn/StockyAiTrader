"""
Custom Backgrounds — gradient panels, glass effects, and decorative patterns.

Provides widget containers with premium visual treatments:
- GlassPanel: frosted glass effect with subtle border glow
- GradientHeader: header bar with animated gradient
- ParticleBackground: subtle floating particles (decorative)
- PatternPanel: geometric pattern overlay
"""

import math
import random
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import (
    QPainter, QColor, QLinearGradient, QRadialGradient,
    QPen, QBrush, QPainterPath, QFont,
)
from .theme import theme


class GlassPanel(QWidget):
    """
    A container widget with a frosted-glass look.
    Content is added via self.content_layout.

    Features:
    - Semi-transparent gradient background
    - Subtle border with accent glow
    - Rounded corners
    """

    def __init__(self, accent_color="#0ea5e9", border_radius=12, parent=None):
        super().__init__(parent)
        self._accent = QColor(accent_color)
        self._radius = border_radius
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(16, 14, 16, 14)
        self.setLayout(self.content_layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        r = self.rect().adjusted(1, 1, -1, -1)

        # Glass background — adapts to current theme
        bg = theme.qcolor("bg_card")
        grad = QLinearGradient(0, 0, 0, r.height())
        grad.setColorAt(0, QColor(bg.red(), bg.green(), bg.blue(), 220))
        grad.setColorAt(1, QColor(bg.red(), bg.green(), bg.blue(), 240))
        painter.setBrush(grad)

        # Border with accent tint
        border = QColor(self._accent)
        border.setAlphaF(0.2)
        painter.setPen(QPen(border, 1))
        painter.drawRoundedRect(r, self._radius, self._radius)

        # Top highlight line
        highlight = QLinearGradient(r.x(), 0, r.right(), 0)
        highlight.setColorAt(0, QColor(255, 255, 255, 0))
        highlight.setColorAt(0.5, QColor(255, 255, 255, 15))
        highlight.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setPen(QPen(QBrush(highlight), 1))
        painter.drawLine(r.x() + self._radius, r.y(), r.right() - self._radius, r.y())

        painter.end()
        super().paintEvent(event)


class GradientHeader(QWidget):
    """
    A header bar with gradient background and title text.
    Used at the top of panels for visual hierarchy.
    """

    def __init__(self, title="", subtitle="", height=60, parent=None):
        super().__init__(parent)
        self._title = title
        self._subtitle = subtitle
        self.setFixedHeight(height)

    def set_title(self, title, subtitle=""):
        self._title = title
        self._subtitle = subtitle
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Gradient background
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0, QColor(14, 165, 233, 30))
        grad.setColorAt(0.5, QColor(99, 102, 241, 20))
        grad.setColorAt(1, QColor(16, 185, 129, 30))
        painter.fillRect(0, 0, w, h, grad)

        # Bottom accent line
        line_grad = QLinearGradient(0, 0, w, 0)
        line_grad.setColorAt(0, QColor(14, 165, 233, 0))
        line_grad.setColorAt(0.3, QColor(14, 165, 233, 200))
        line_grad.setColorAt(0.7, QColor(16, 185, 129, 200))
        line_grad.setColorAt(1, QColor(16, 185, 129, 0))
        painter.setPen(QPen(QBrush(line_grad), 2))
        painter.drawLine(0, h - 1, w, h - 1)

        # Title text
        painter.setPen(theme.qcolor("text_heading"))
        painter.setFont(QFont("Segoe UI", 15, QFont.Bold))
        y = 12 if self._subtitle else (h - 20) // 2
        painter.drawText(20, y, w - 40, 24, Qt.AlignLeft | Qt.AlignVCenter, self._title)

        # Subtitle
        if self._subtitle:
            painter.setPen(theme.qcolor("text_secondary"))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(20, y + 22, w - 40, 18, Qt.AlignLeft | Qt.AlignVCenter, self._subtitle)

        painter.end()


class ParticleBackground(QWidget):
    """
    Decorative floating particles that drift slowly.
    Used behind content on splash screens or empty-state panels.
    Very lightweight — just draws small circles with low alpha.
    """

    def __init__(self, particle_count=30, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._particles = []

        # Generate particles with random positions and velocities
        for _ in range(particle_count):
            self._particles.append({
                "x": random.random(),      # 0-1 normalized
                "y": random.random(),
                "vx": (random.random() - 0.5) * 0.001,
                "vy": (random.random() - 0.5) * 0.001,
                "size": random.uniform(1.5, 4),
                "alpha": random.uniform(0.03, 0.12),
            })

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)  # 20fps is enough for ambient particles

    def _tick(self):
        for p in self._particles:
            p["x"] = (p["x"] + p["vx"]) % 1.0
            p["y"] = (p["y"] + p["vy"]) % 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        w = self.width()
        h = self.height()

        for p in self._particles:
            color = QColor(14, 165, 233)
            color.setAlphaF(p["alpha"])
            painter.setBrush(color)
            painter.drawEllipse(
                QPointF(p["x"] * w, p["y"] * h),
                p["size"], p["size"],
            )

        painter.end()

    def stop(self):
        self._timer.stop()


class PatternPanel(QWidget):
    """
    A panel with a subtle geometric dot pattern overlay.
    Use as a background behind content for visual texture.
    """

    def __init__(self, spacing=20, dot_size=1.5, parent=None):
        super().__init__(parent)
        self._spacing = spacing
        self._dot_size = dot_size
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.content_layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 8))

        w = self.width()
        h = self.height()
        s = self._spacing

        for x in range(0, w, s):
            for y in range(0, h, s):
                painter.drawEllipse(QPointF(x, y), self._dot_size, self._dot_size)

        painter.end()
        super().paintEvent(event)
