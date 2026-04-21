"""
Premium Boot Screen — branded loading experience with animations.

A full-screen loading window with:
- Animated gradient background with floating orbs
- App branding (icon, name, tagline)
- Gradient progress bar with step descriptions
- Expandable detail log
- Smooth fade-out transition to main app

Usage:
    screen = BootScreen()
    screen.show()
    screen.step(25, "Loading modules...", "core.features, core.model")
    screen.step(100, "Ready.")
    screen.finish()  # fades out
"""

import os
import math
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QApplication,
    QGraphicsOpacityEffect, QTextEdit, QPushButton,
)
from PyQt5.QtCore import (
    Qt, QTimer, QPointF, QPropertyAnimation, QEasingCurve,
)
from PyQt5.QtGui import (
    QPainter, QColor, QLinearGradient, QRadialGradient,
    QPen, QBrush, QFont, QPixmap, QPainterPath,
)

from ..branding import (
    APP_NAME, APP_VERSION, APP_TAGLINE, APP_AUTHOR, APP_URL,
    BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT,
    FONT_FAMILY, FONT_MONO,
)


class _AnimatedBackground(QWidget):
    """Background layer with gradient and animated floating orbs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(35)

    def _tick(self):
        self._phase += 0.025
        self.update()

    def stop(self):
        self._timer.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Deep gradient base
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0, QColor(8, 10, 18))
        grad.setColorAt(0.4, QColor(12, 16, 30))
        grad.setColorAt(0.7, QColor(10, 14, 28))
        grad.setColorAt(1, QColor(6, 8, 14))
        painter.fillRect(0, 0, w, h, grad)

        # Animated orbs
        orbs = [
            (0.15, 0.25, 100, BRAND_PRIMARY,   0.07, 1.3, 0.0),
            (0.75, 0.55, 140, BRAND_SECONDARY,  0.05, 0.9, 2.0),
            (0.45, 0.75, 80,  BRAND_ACCENT,     0.06, 1.1, 4.0),
            (0.88, 0.18, 70,  BRAND_PRIMARY,    0.04, 1.5, 1.5),
            (0.3,  0.5,  110, BRAND_ACCENT,     0.03, 0.7, 3.0),
        ]
        for bx, by, radius, color_hex, alpha, speed, offset in orbs:
            ox = bx * w + math.sin(self._phase * speed + offset) * 40
            oy = by * h + math.cos(self._phase * speed * 0.8 + offset) * 30
            rg = QRadialGradient(QPointF(ox, oy), radius)
            c = QColor(color_hex)
            c.setAlphaF(alpha + math.sin(self._phase * 2 + offset) * 0.015)
            rg.setColorAt(0, c)
            c.setAlphaF(0)
            rg.setColorAt(1, c)
            painter.setBrush(rg)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(ox, oy), radius, radius)

        # Top accent line with animation
        line_phase = self._phase * 0.5
        line_grad = QLinearGradient(0, 0, w, 0)
        shift = (math.sin(line_phase) + 1) / 2 * 0.3
        line_grad.setColorAt(0, QColor(BRAND_PRIMARY + "00"))
        line_grad.setColorAt(0.2 + shift, QColor(BRAND_PRIMARY))
        line_grad.setColorAt(0.5, QColor(BRAND_SECONDARY + "cc"))
        line_grad.setColorAt(0.8 - shift, QColor(BRAND_ACCENT))
        line_grad.setColorAt(1, QColor(BRAND_ACCENT + "00"))
        painter.setPen(QPen(QBrush(line_grad), 3))
        painter.drawLine(0, 2, w, 2)

        # Bottom subtle line
        painter.setPen(QPen(QColor(BRAND_PRIMARY + "15"), 1))
        painter.drawLine(0, h - 1, w, h - 1)

        painter.end()


class _GradientProgressBar(QWidget):
    """Custom-painted gradient progress bar with glow effect."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pct = 0.0
        self._phase = 0.0
        self.setFixedHeight(8)

        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: (setattr(self, '_phase', self._phase + 0.1), self.update()))
        self._timer.start(50)

    def set_value(self, pct):
        self._pct = max(0.0, min(1.0, pct))
        self.update()

    def stop(self):
        self._timer.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Track
        painter.setBrush(QColor(30, 33, 48))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, h // 2, h // 2)

        fill_w = int(w * self._pct)
        if fill_w > 2:
            # Animated gradient fill
            grad = QLinearGradient(0, 0, fill_w, 0)
            shift = (math.sin(self._phase) + 1) / 2
            grad.setColorAt(0, QColor(BRAND_PRIMARY))
            grad.setColorAt(0.5 + shift * 0.2, QColor(BRAND_SECONDARY))
            grad.setColorAt(1, QColor(BRAND_ACCENT))
            painter.setBrush(grad)
            painter.drawRoundedRect(0, 0, fill_w, h, h // 2, h // 2)

            # Glow at the leading edge
            glow = QRadialGradient(QPointF(fill_w, h / 2), 15)
            gc = QColor(BRAND_ACCENT)
            gc.setAlphaF(0.4)
            glow.setColorAt(0, gc)
            gc.setAlphaF(0)
            glow.setColorAt(1, gc)
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(fill_w, h / 2), 15, 15)

        painter.end()


class BootScreen(QWidget):
    """
    Premium branded boot screen with animated background,
    gradient progress bar, and step-by-step loading display.
    """

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Size — Qt high DPI handles scaling, so use logical pixels
        bw, bh = 520, 380
        self.setFixedSize(bw, bh)
        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - bw) // 2, (screen.height() - bh) // 2)
        scale = 1.0  # Qt high DPI does the heavy lifting

        # Animated background
        self._bg = _AnimatedBackground(self)

        # Content on top
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Inner content area
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        inner = QVBoxLayout()
        inner.setContentsMargins(50, 30, 50, 24)
        inner.setSpacing(6)

        # Icon — use splash variant if available, fallback to regular
        icon_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "icon_splash.png")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "icon.png")
        if os.path.exists(icon_path):
            icon_lbl = QLabel()
            icon_lbl.setPixmap(QPixmap(icon_path).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            icon_lbl.setAlignment(Qt.AlignCenter)
            icon_lbl.setStyleSheet("background: transparent;")
            inner.addWidget(icon_lbl)

        # App name
        name = QLabel(APP_NAME)
        name.setFont(QFont(FONT_FAMILY, int(28 * scale), QFont.Bold))
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet(f"color: {BRAND_PRIMARY}; background: transparent; letter-spacing: 2px;")
        inner.addWidget(name)

        # Tagline
        tag = QLabel(APP_TAGLINE)
        tag.setAlignment(Qt.AlignCenter)
        tag.setStyleSheet(f"color: #94a3b8; font-size: {int(11 * scale)}px; background: transparent;")
        inner.addWidget(tag)

        # Version
        ver = QLabel(f"v{APP_VERSION}")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet(f"color: #475569; font-size: {int(9 * scale)}px; background: transparent;")
        inner.addWidget(ver)

        inner.addSpacing(int(30 * scale))

        # Status message
        self._status = QLabel("Initializing...")
        self._status.setFont(QFont(FONT_FAMILY, int(11 * scale), QFont.DemiBold))
        self._status.setStyleSheet(f"color: {BRAND_PRIMARY}; background: transparent;")
        inner.addWidget(self._status)

        # Progress bar
        self._bar = _GradientProgressBar()
        inner.addWidget(self._bar)

        # Detail text
        self._detail = QLabel("")
        self._detail.setStyleSheet(f"color: #64748b; font-size: 9px; background: transparent; font-family: {FONT_MONO};")
        inner.addWidget(self._detail)

        inner.addStretch()

        # Feature tagline at bottom
        features_lbl = QLabel("LightGBM  ·  FinBERT  ·  10 Addons  ·  Risk Management")
        features_lbl.setAlignment(Qt.AlignCenter)
        features_lbl.setStyleSheet(f"color: #334155; font-size: 9px; background: transparent;")
        inner.addWidget(features_lbl)

        # Copyright
        copy_lbl = QLabel(f"© 2024-2026 {APP_AUTHOR}")
        copy_lbl.setAlignment(Qt.AlignCenter)
        copy_lbl.setStyleSheet(f"color: #1e293b; font-size: 8px; background: transparent;")
        inner.addWidget(copy_lbl)

        content.setLayout(inner)
        layout.addWidget(content)
        self.setLayout(layout)

        # Fade-in animation on show
        self._content = content
        effect = QGraphicsOpacityEffect(content)
        content.setGraphicsEffect(effect)
        self._fade_in = QPropertyAnimation(effect, b"opacity")
        self._fade_in.setDuration(800)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.OutCubic)

    def showEvent(self, event):
        super().showEvent(event)
        self._fade_in.start()

    def resizeEvent(self, event):
        self._bg.resize(self.size())
        super().resizeEvent(event)

    def step(self, pct, status, detail=""):
        """Update loading progress."""
        self._bar.set_value(pct / 100.0)
        self._status.setText(status)
        if detail:
            self._detail.setText(detail)
        QApplication.processEvents()

    def finish(self):
        """Dissolve into particles, then clean up."""
        self._bar.stop()

        # Switch to particle dissolve mode
        self._dissolving = True
        self._dissolve_phase = 0.0
        self._particles = []

        # Generate particles from the content area
        import random
        w, h = self.width(), self.height()
        colors = [BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT, "#94a3b8", "#ffffff"]
        for _ in range(80):
            self._particles.append({
                "x": random.uniform(50, w - 50),
                "y": random.uniform(30, h - 30),
                "vx": random.uniform(-3, 3),
                "vy": random.uniform(-5, -1),
                "size": random.uniform(2, 6),
                "color": random.choice(colors),
                "alpha": 1.0,
                "decay": random.uniform(0.015, 0.035),
            })

        # Hide content, keep bg + particles
        self._content.setVisible(False)

        # Dissolve timer
        self._dissolve_timer = QTimer(self)
        self._dissolve_timer.timeout.connect(self._tick_dissolve)
        self._dissolve_timer.start(25)

    def _tick_dissolve(self):
        """Animate particles flying away and fading."""
        self._dissolve_phase += 0.03
        all_dead = True
        for p in self._particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.05  # slight gravity curve
            p["vx"] *= 1.01  # spread out
            p["alpha"] -= p["decay"]
            if p["alpha"] > 0:
                all_dead = False

        self.update()

        if all_dead or self._dissolve_phase > 1.5:
            self._dissolve_timer.stop()
            self._bg.stop()
            self.hide()
            self.deleteLater()

    def paintEvent(self, event):
        if not getattr(self, '_dissolving', False):
            return super().paintEvent(event)

        # Draw background (still animating)
        self._bg.resize(self.size())

        # Draw particles on top
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for p in self._particles:
            if p["alpha"] <= 0:
                continue
            c = QColor(p["color"])
            c.setAlphaF(max(0, p["alpha"]))
            painter.setBrush(c)
            painter.setPen(Qt.NoPen)
            s = p["size"] * (0.5 + p["alpha"] * 0.5)  # shrink as they fade
            painter.drawEllipse(QPointF(p["x"], p["y"]), s, s)

            # Small glow around each particle
            gc = QColor(p["color"])
            gc.setAlphaF(max(0, p["alpha"] * 0.15))
            glow = QRadialGradient(QPointF(p["x"], p["y"]), s * 3)
            glow.setColorAt(0, gc)
            gc.setAlphaF(0)
            glow.setColorAt(1, gc)
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(p["x"], p["y"]), s * 3, s * 3)

        painter.end()
