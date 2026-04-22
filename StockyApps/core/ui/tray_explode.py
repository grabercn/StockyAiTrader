# -*- coding: utf-8 -*-
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
from .anim_config import get_particle_count, get_timer_interval


class TrayExplode(QWidget):
    """Fullscreen overlay -- firework burst from tray icon position."""

    def __init__(self, cx, cy, on_done=None):
        super().__init__()
        self._on_done = on_done

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)

        self._cx = cx - screen.x()
        self._cy = cy - screen.y()

        self._particles = []
        colors = [BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT, "#94a3b8", "#e2e8f0", "#ffffff"]
        num = get_particle_count("tray")

        for _ in range(num):
            angle = random.uniform(0, 6.28)
            speed = random.uniform(3, 14)
            self._particles.append({
                "x": self._cx,
                "y": self._cy,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed - random.uniform(1, 4),
                "size": random.uniform(2, 6),
                "color": random.choice(colors),
                "alpha": 1.0,
                "decay": random.uniform(0.018, 0.04),
                "spark": random.random() > 0.75,
            })

        self._phase = 0.0

    def start(self):
        self.show()
        self.raise_()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(get_timer_interval())

    def _tick(self):
        self._phase += 0.035
        all_dead = True

        for p in self._particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.15
            p["vx"] *= 0.99
            p["alpha"] -= p["decay"]
            if p["alpha"] > 0:
                all_dead = False

        self.update()

        if all_dead or self._phase > 1.8:
            self._timer.stop()
            if self._on_done:
                self._on_done()
            self.hide()
            self.deleteLater()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Initial flash
        if self._phase < 0.12:
            flash_a = (0.12 - self._phase) / 0.12 * 0.4
            flash = QRadialGradient(QPointF(self._cx, self._cy), 60)
            fc = QColor("#ffffff")
            fc.setAlphaF(flash_a)
            flash.setColorAt(0, fc)
            fc2 = QColor(BRAND_PRIMARY)
            fc2.setAlphaF(flash_a * 0.25)
            flash.setColorAt(0.5, fc2)
            fc3 = QColor(BRAND_PRIMARY)
            fc3.setAlphaF(0)
            flash.setColorAt(1, fc3)
            painter.setBrush(flash)
            painter.setPen(Qt.NoPen)
            painter.drawRect(self.rect())

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

            # Sparkle twinkle
            if p.get("spark") and int(self._phase * 20) % 3 == 0:
                sc = QColor("#ffffff")
                sc.setAlphaF(a * 0.7)
                painter.setBrush(sc)
                painter.drawEllipse(QPointF(p["x"], p["y"]), s * 0.4, s * 0.4)

        painter.end()
