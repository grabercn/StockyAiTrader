# -*- coding: utf-8 -*-
"""
Window Expand Animation — reverse of WindowCollapse.

When the user reopens from the tray, particles spawn at a center
vanishing point and spiral OUTWARD to form the window edges,
then the window fades in behind them.
"""

import math
import random
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QRadialGradient, QPen

from ..branding import BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT
from .anim_config import get_particle_count, get_timer_interval


class WindowExpand(QWidget):
    """Fullscreen overlay -- particles expand from center to window edges."""

    def __init__(self, window_geo, on_done=None):
        super().__init__()
        self._on_done = on_done

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)

        self._wx = window_geo.x() - screen.x()
        self._wy = window_geo.y() - screen.y()
        self._ww = window_geo.width()
        self._wh = window_geo.height()

        self._cx = self._wx + self._ww / 2
        self._cy = self._wy + self._wh / 2

        self._particles = []
        colors = [BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT, "#94a3b8", "#e2e8f0"]
        perimeter = 2 * (self._ww + self._wh)
        num = get_particle_count("expand")

        for i in range(num):
            t = (i / num) * perimeter
            if t < self._ww:
                tx, ty = self._wx + t, self._wy
            elif t < self._ww + self._wh:
                tx, ty = self._wx + self._ww, self._wy + (t - self._ww)
            elif t < 2 * self._ww + self._wh:
                tx, ty = self._wx + self._ww - (t - self._ww - self._wh), self._wy + self._wh
            else:
                tx, ty = self._wx, self._wy + self._wh - (t - 2 * self._ww - self._wh)

            self._particles.append({
                "x": self._cx, "y": self._cy,
                "tx": tx, "ty": ty,
                "size": random.uniform(2, 5),
                "color": random.choice(colors),
                "delay": random.uniform(0, 0.15),
                "spin": random.uniform(0.05, 0.15) * random.choice([-1, 1]),
                "_t": 0,
            })

        self._phase = 0.0

    def start(self):
        self.show()
        self.raise_()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(get_timer_interval())

    def _tick(self):
        self._phase += 0.03
        all_done = True

        for p in self._particles:
            if self._phase < p["delay"]:
                all_done = False
                continue

            t = min(1.0, (self._phase - p["delay"]) / 0.6)
            ease = 1.0 - (1.0 - t) * (1.0 - t)

            angle = p["spin"] * t * 6 * math.pi
            radius = ease * 25

            p["x"] = self._cx + (p["tx"] - self._cx) * ease + math.cos(angle) * radius * (1 - ease)
            p["y"] = self._cy + (p["ty"] - self._cy) * ease + math.sin(angle) * radius * (1 - ease)
            p["_t"] = t

            if t < 1.0:
                all_done = False

        self.update()

        if all_done or self._phase > 1.2:
            self._timer.stop()
            if self._on_done:
                self._on_done()
            self.hide()
            self.deleteLater()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Center glow (fades as particles leave)
        if self._phase < 0.5:
            intensity = max(0, 1.0 - self._phase * 2)
            vp = QRadialGradient(QPointF(self._cx, self._cy), 35)
            vc = QColor(BRAND_PRIMARY)
            vc.setAlphaF(intensity * 0.35)
            vp.setColorAt(0, vc)
            vc2 = QColor(BRAND_PRIMARY)
            vc2.setAlphaF(0)
            vp.setColorAt(1, vc2)
            painter.setBrush(vp)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(self._cx, self._cy), 35, 35)

        painter.setPen(Qt.NoPen)
        for p in self._particles:
            if self._phase < p.get("delay", 0):
                continue

            t = p.get("_t", 0)
            alpha = min(1.0, t * 1.5)
            s = p["size"] * (0.3 + t * 0.7)

            c = QColor(p["color"])
            c.setAlphaF(alpha)
            painter.setBrush(c)
            painter.drawEllipse(QPointF(p["x"], p["y"]), s, s)

            # Trail toward center (simplified)
            if alpha > 0.3 and 0.1 < t < 0.85:
                dx = self._cx - p["x"]
                dy = self._cy - p["y"]
                dist = max(1, math.sqrt(dx * dx + dy * dy))
                nx, ny = dx / dist, dy / dist
                trail_len = s * 1.5 * (1 - t)
                tc = QColor(p["color"])
                tc.setAlphaF(alpha * 0.1)
                painter.setPen(QPen(tc, max(1, s * 0.3)))
                painter.drawLine(QPointF(p["x"], p["y"]),
                                QPointF(p["x"] + nx * trail_len, p["y"] + ny * trail_len))
                painter.setPen(Qt.NoPen)

        # Window outline as particles arrive
        if self._phase > 0.4:
            outline_alpha = min(0.25, (self._phase - 0.4) * 0.5)
            rc = QColor(BRAND_PRIMARY)
            rc.setAlphaF(outline_alpha)
            painter.setPen(QPen(rc, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._wx, self._wy, self._ww, self._wh)

        painter.end()
