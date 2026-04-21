"""
Tray Explode Animation — particles burst from tray icon when quitting.

A fullscreen transparent overlay spawns particles at the tray icon position
that explode outward in a firework-like burst, then fade away.
App quits after the animation completes.
"""

import math
import random
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QRadialGradient

from ..branding import BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT


class TrayExplode(QWidget):
    """Fullscreen overlay — firework burst from tray icon position."""

    def __init__(self, cx, cy, on_done=None):
        super().__init__()
        self._on_done = on_done

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)

        # Explosion center (tray icon position in screen coords)
        self._cx = cx - screen.x()
        self._cy = cy - screen.y()

        # Generate burst particles
        self._particles = []
        colors = [BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT, "#ffffff", "#f59e0b", "#ef4444", "#10b981"]

        for _ in range(100):
            angle = random.uniform(0, 6.28)
            speed = random.uniform(3, 15)
            self._particles.append({
                "x": self._cx,
                "y": self._cy,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed - random.uniform(1, 4),  # Bias upward
                "size": random.uniform(2, 7),
                "color": random.choice(colors),
                "alpha": 1.0,
                "decay": random.uniform(0.015, 0.035),
                "spark": random.random() > 0.7,  # 30% are sparkle particles
            })

        self._phase = 0.0

    def start(self):
        self.show()
        self.raise_()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def _tick(self):
        self._phase += 0.03
        all_dead = True

        for p in self._particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.15  # Gravity
            p["vx"] *= 0.99  # Air resistance
            p["alpha"] -= p["decay"]
            if p["alpha"] > 0:
                all_dead = False

        self.update()

        if all_dead or self._phase > 2.0:
            self._timer.stop()
            if self._on_done:
                self._on_done()
            self.hide()
            self.deleteLater()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Initial flash at explosion center
        if self._phase < 0.15:
            flash_a = (0.15 - self._phase) / 0.15 * 0.5
            flash = QRadialGradient(QPointF(self._cx, self._cy), 80)
            fc = QColor("#ffffff")
            fc.setAlphaF(flash_a)
            flash.setColorAt(0, fc)
            fc2 = QColor(BRAND_PRIMARY)
            fc2.setAlphaF(flash_a * 0.3)
            flash.setColorAt(0.5, fc2)
            fc3 = QColor(BRAND_PRIMARY)
            fc3.setAlphaF(0)
            flash.setColorAt(1, fc3)
            painter.setBrush(flash)
            painter.setPen(Qt.NoPen)
            painter.drawRect(self.rect())

        # Particles
        painter.setPen(Qt.NoPen)
        for p in self._particles:
            if p["alpha"] <= 0:
                continue
            a = max(0, min(1, p["alpha"]))
            s = p["size"] * (0.3 + a * 0.7)

            c = QColor(p["color"])
            c.setAlphaF(a)
            painter.setBrush(c)
            painter.drawEllipse(QPointF(p["x"], p["y"]), s, s)

            # Sparkle particles twinkle
            if p.get("spark") and int(self._phase * 20) % 3 == 0:
                sc = QColor("#ffffff")
                sc.setAlphaF(a * 0.8)
                painter.setBrush(sc)
                painter.drawEllipse(QPointF(p["x"], p["y"]), s * 0.5, s * 0.5)

            # Glow
            if a > 0.2:
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
