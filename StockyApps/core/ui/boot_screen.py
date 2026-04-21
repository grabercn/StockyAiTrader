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
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
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
        """Dissolve into a fullscreen particle overlay so particles fly beyond the boot window."""
        self._bg.stop()
        self._bar.stop()
        snapshot = self.grab()
        boot_geo = self.geometry()
        self.hide()
        self.deleteLater()

        # Launch fullscreen dissolve overlay
        overlay = _DissolveOverlay(snapshot, boot_geo)
        overlay.start()
        QApplication.instance()._dissolve_overlay = overlay


class _DissolveOverlay(QWidget):
    """Fullscreen transparent overlay — particles burst outward from the boot screen across the entire screen."""

    def __init__(self, snapshot, boot_geo):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)

        self._snapshot = snapshot
        self._bx = boot_geo.x() - screen.x()
        self._by = boot_geo.y() - screen.y()
        self._bw = boot_geo.width()
        self._bh = boot_geo.height()

        cx = self._bx + self._bw / 2
        cy = self._by + self._bh / 2

        import random
        self._particles = []
        colors = [BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT, "#94a3b8", "#e2e8f0", "#ffffff"]

        for _ in range(140):
            angle = random.uniform(0, 6.28)
            speed = random.uniform(2, 10)
            dist = random.uniform(0, 60)
            self._particles.append({
                "x": cx + math.cos(angle) * dist,
                "y": cy + math.sin(angle) * dist,
                "vx": math.cos(angle) * speed + random.uniform(-1, 1),
                "vy": math.sin(angle) * speed + random.uniform(-1, 1),
                "size": random.uniform(1.5, 7),
                "color": random.choice(colors),
                "alpha": random.uniform(0.7, 1.0),
                "decay": random.uniform(0.01, 0.025),
                "spin": random.uniform(-0.04, 0.04),
                "trail": random.random() > 0.5,
            })

        self._phase = 0.0
        self._snap_alpha = 1.0

    def start(self):
        self.show()
        self.raise_()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(20)

    def _tick(self):
        self._phase += 0.03
        self._snap_alpha = max(0, 1.0 - self._phase * 2.5)

        all_dead = True
        for p in self._particles:
            spin = p.get("spin", 0)
            if spin:
                cs, sn = math.cos(spin), math.sin(spin)
                p["vx"], p["vy"] = p["vx"] * cs - p["vy"] * sn, p["vx"] * sn + p["vy"] * cs
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.02
            p["vx"] *= 1.01
            p["alpha"] -= p["decay"]
            if p["alpha"] > 0:
                all_dead = False

        self.update()

        if all_dead or self._phase > 2.5:
            self._timer.stop()
            self.hide()
            self.deleteLater()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Flash from center at start
        if self._phase < 0.25:
            flash_a = (0.25 - self._phase) / 0.25 * 0.3
            cx = self._bx + self._bw / 2
            cy = self._by + self._bh / 2
            flash = QRadialGradient(QPointF(cx, cy), max(self.width(), self.height()) * 0.5)
            fc = QColor(BRAND_PRIMARY)
            fc.setAlphaF(flash_a)
            flash.setColorAt(0, fc)
            fc2 = QColor(BRAND_PRIMARY)
            fc2.setAlphaF(0)
            flash.setColorAt(1, fc2)
            painter.setBrush(flash)
            painter.setPen(Qt.NoPen)
            painter.drawRect(self.rect())

        # Fading boot screen snapshot at original position
        if self._snap_alpha > 0:
            painter.setOpacity(self._snap_alpha)
            painter.drawPixmap(self._bx, self._by, self._snapshot)
            painter.setOpacity(1.0)

        # Particles flying across the whole screen
        painter.setPen(Qt.NoPen)
        for p in self._particles:
            if p["alpha"] <= 0:
                continue
            a = max(0.0, min(1.0, p["alpha"]))
            s = p["size"] * (0.3 + a * 0.7)

            # Motion trail
            if p.get("trail") and a > 0.2:
                speed = math.sqrt(p["vx"]**2 + p["vy"]**2)
                if speed > 0.5:
                    tc = QColor(p["color"])
                    tc.setAlphaF(a * 0.12)
                    painter.setPen(QPen(tc, max(1, s * 0.3)))
                    nx, ny = -p["vx"] / speed, -p["vy"] / speed
                    tl = speed * 2.5
                    painter.drawLine(QPointF(p["x"], p["y"]), QPointF(p["x"] + nx * tl, p["y"] + ny * tl))
                    painter.setPen(Qt.NoPen)

            # Bright core
            c = QColor(p["color"])
            c.setAlphaF(a)
            painter.setBrush(c)
            painter.drawEllipse(QPointF(p["x"], p["y"]), s, s)

            # Glow halo
            if a > 0.15:
                gc = QColor(p["color"])
                gc.setAlphaF(a * 0.08)
                glow = QRadialGradient(QPointF(p["x"], p["y"]), s * 4)
                glow.setColorAt(0, gc)
                gc2 = QColor(p["color"])
                gc2.setAlphaF(0)
                glow.setColorAt(1, gc2)
                painter.setBrush(glow)
                painter.drawEllipse(QPointF(p["x"], p["y"]), s * 4, s * 4)

        painter.end()
