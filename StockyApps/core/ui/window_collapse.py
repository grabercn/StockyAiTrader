# -*- coding: utf-8 -*-
"""
Window Collapse Animation — reverse of the window reveal.

When the user minimizes to tray, particles spawn on the window edges
and fly INWARD to the center, spiraling and shrinking into a vanishing
point before disappearing.
"""

import math
import random
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QRadialGradient, QPen, QPixmap

from ..branding import BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT
from .anim_config import get_particle_count, get_timer_interval


class WindowCollapse(QWidget):
    """Fullscreen overlay -- particles fly from window edges to center and vanish."""

    def __init__(self, snapshot, window_geo, on_done=None):
        super().__init__()
        self._on_done = on_done

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)

        self._snapshot = snapshot
        self._wx = window_geo.x() - screen.x()
        self._wy = window_geo.y() - screen.y()
        self._ww = window_geo.width()
        self._wh = window_geo.height()

        self._cx = self._wx + self._ww / 2
        self._cy = self._wy + self._wh / 2

        self._particles = []
        colors = [BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT, "#94a3b8", "#e2e8f0"]
        perimeter = 2 * (self._ww + self._wh)
        num = get_particle_count("collapse")

        for i in range(num):
            t = (i / num) * perimeter
            if t < self._ww:
                sx, sy = self._wx + t, self._wy
            elif t < self._ww + self._wh:
                sx, sy = self._wx + self._ww, self._wy + (t - self._ww)
            elif t < 2 * self._ww + self._wh:
                sx, sy = self._wx + self._ww - (t - self._ww - self._wh), self._wy + self._wh
            else:
                sx, sy = self._wx, self._wy + self._wh - (t - 2 * self._ww - self._wh)

            self._particles.append({
                "x": sx, "y": sy,
                "sx": sx, "sy": sy,
                "size": random.uniform(2, 5),
                "color": random.choice(colors),
                "delay": random.uniform(0, 0.2),
                "spin": random.uniform(0.05, 0.15) * random.choice([-1, 1]),
            })

        self._phase = 0.0
        self._snap_alpha = 1.0

    def start(self):
        self.show()
        self.raise_()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(get_timer_interval())

    def _tick(self):
        self._phase += 0.025
        self._snap_alpha = max(0, 1.0 - self._phase * 3)

        all_done = True
        for p in self._particles:
            if self._phase < p["delay"]:
                all_done = False
                continue

            t = min(1.0, (self._phase - p["delay"]) / 0.7)
            ease = t * t * t

            angle = p["spin"] * t * 8 * math.pi
            radius = (1.0 - ease) * 50

            target_x = self._cx + math.cos(angle) * radius * (1 - ease)
            target_y = self._cy + math.sin(angle) * radius * (1 - ease)

            p["x"] = p["sx"] + (target_x - p["sx"]) * ease
            p["y"] = p["sy"] + (target_y - p["sy"]) * ease
            p["_t"] = t

            if t < 1.0:
                all_done = False

        self.update()

        if all_done or self._phase > 1.5:
            self._timer.stop()
            if self._on_done:
                self._on_done()
            self.hide()
            self.deleteLater()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fading snapshot
        if self._snap_alpha > 0:
            painter.setOpacity(self._snap_alpha)
            painter.drawPixmap(self._wx, self._wy, self._snapshot)
            painter.setOpacity(1.0)

        painter.setPen(Qt.NoPen)
        for p in self._particles:
            if self._phase < p.get("delay", 0):
                continue

            t = p.get("_t", 0)
            alpha = max(0, 1.0 - t * 0.8)
            s = p["size"] * max(0.2, 1.0 - t * 0.9)

            c = QColor(p["color"])
            c.setAlphaF(alpha)
            painter.setBrush(c)
            painter.drawEllipse(QPointF(p["x"], p["y"]), s, s)

            # Motion trail (simplified)
            if alpha > 0.3 and 0.1 < t < 0.8:
                dx = p["x"] - self._cx
                dy = p["y"] - self._cy
                dist = max(1, math.sqrt(dx * dx + dy * dy))
                nx, ny = dx / dist, dy / dist
                trail_len = s * 2.5 * (1 - t)
                tc = QColor(p["color"])
                tc.setAlphaF(alpha * 0.12)
                painter.setPen(QPen(tc, max(1, s * 0.3)))
                painter.drawLine(QPointF(p["x"], p["y"]),
                                QPointF(p["x"] + nx * trail_len, p["y"] + ny * trail_len))
                painter.setPen(Qt.NoPen)

        # Center vanishing point glow
        if self._phase > 0.3:
            intensity = min(1.0, (self._phase - 0.3) * 2)
            r = 25 * (1 - intensity * 0.5)
            vp = QRadialGradient(QPointF(self._cx, self._cy), r)
            vc = QColor(BRAND_PRIMARY)
            vc.setAlphaF(intensity * 0.35)
            vp.setColorAt(0, vc)
            vc2 = QColor(BRAND_PRIMARY)
            vc2.setAlphaF(0)
            vp.setColorAt(1, vc2)
            painter.setBrush(vp)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(self._cx, self._cy), r, r)

        painter.end()
