# -*- coding: utf-8 -*-
"""
Window Reveal Animation — particles converge to form the main window outline.

After the boot screen dissolves, this transparent overlay spawns particles
scattered across the screen. They fly inward and settle along the edges
of the main window rectangle, tracing its outline with glowing dots.
As the particles arrive, the main window fades in behind them.

Usage:
    reveal = WindowReveal(main_window)
    reveal.start()  # Call after boot.finish()
"""

import math
import random
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QRadialGradient, QPen

from ..branding import BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT
from .anim_config import get_particle_count, get_timer_interval


class WindowReveal(QWidget):
    """Fullscreen transparent overlay that animates particles into a window shape."""

    def __init__(self, target_window, on_done=None):
        super().__init__()
        self._target = target_window
        self._on_done = on_done

        # Fullscreen transparent overlay
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)

        # Target rectangle in screen coords
        tg = target_window.geometry()
        self._tx = tg.x()
        self._ty = tg.y()
        self._tw = tg.width()
        self._th = tg.height()

        # Generate particles with random start positions and target positions on the window edge
        self._particles = []
        self._phase = 0.0
        colors = [BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT, "#94a3b8", "#e2e8f0"]
        sw, sh = screen.width(), screen.height()

        # Distribute target points along the window rectangle perimeter
        perimeter = 2 * (self._tw + self._th)
        num_particles = get_particle_count("reveal")

        for i in range(num_particles):
            # Target: point on the window edge
            t = (i / num_particles) * perimeter
            if t < self._tw:
                tx, ty = self._tx + t, self._ty
            elif t < self._tw + self._th:
                tx, ty = self._tx + self._tw, self._ty + (t - self._tw)
            elif t < 2 * self._tw + self._th:
                tx, ty = self._tx + self._tw - (t - self._tw - self._th), self._ty + self._th
            else:
                tx, ty = self._tx, self._ty + self._th - (t - 2 * self._tw - self._th)

            # Start: random position far from target
            angle = random.uniform(0, 6.28)
            dist = random.uniform(200, max(sw, sh) * 0.7)
            sx = tx + math.cos(angle) * dist
            sy = ty + math.sin(angle) * dist

            self._particles.append({
                "x": sx, "y": sy,
                "tx": tx, "ty": ty,
                "sx": sx, "sy": sy,
                "size": random.uniform(2, 4),
                "color": random.choice(colors),
                "arrived": False,
                "delay": random.uniform(0, 0.3),
            })

    def start(self):
        self.show()
        self.raise_()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(get_timer_interval())

    def _tick(self):
        self._phase += 0.025
        all_arrived = True

        for p in self._particles:
            if p["arrived"]:
                continue

            if self._phase < p["delay"]:
                all_arrived = False
                continue

            t = min(1.0, (self._phase - p["delay"]) / 0.8)
            ease = t * t * (3 - 2 * t)

            p["x"] = p["sx"] + (p["tx"] - p["sx"]) * ease
            p["y"] = p["sy"] + (p["ty"] - p["sy"]) * ease

            if t >= 1.0:
                p["arrived"] = True
                p["x"] = p["tx"]
                p["y"] = p["ty"]
            else:
                all_arrived = False

        self.update()

        if all_arrived or self._phase > 1.5:
            if not hasattr(self, '_fading_out'):
                self._fading_out = True
                self._fade_phase = 0.0
                if self._on_done:
                    self._on_done()

        if hasattr(self, '_fading_out'):
            self._fade_phase += 0.05
            if self._fade_phase > 1.0:
                self._timer.stop()
                self.hide()
                self.deleteLater()
                return

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        fade = 1.0 - getattr(self, '_fade_phase', 0.0)

        for p in self._particles:
            if self._phase < p.get("delay", 0) and not p["arrived"]:
                continue

            c = QColor(p["color"])
            alpha = fade
            if not p["arrived"]:
                dx = p["x"] - p["tx"]
                dy = p["y"] - p["ty"]
                dist = math.sqrt(dx * dx + dy * dy)
                alpha *= min(1.0, 0.3 + 0.7 * (1.0 - min(dist / 300, 1.0)))

            c.setAlphaF(max(0, min(1, alpha)))
            s = p["size"]

            # Core dot
            painter.setBrush(c)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(p["x"], p["y"]), s, s)

            # Subtle glow (reduced radius)
            if alpha > 0.3:
                gc = QColor(p["color"])
                gc.setAlphaF(max(0, alpha * 0.12))
                glow = QRadialGradient(QPointF(p["x"], p["y"]), s * 3)
                glow.setColorAt(0, gc)
                gc2 = QColor(p["color"])
                gc2.setAlphaF(0)
                glow.setColorAt(1, gc2)
                painter.setBrush(glow)
                painter.drawEllipse(QPointF(p["x"], p["y"]), s * 3, s * 3)

        # Outline glow after arrival
        if hasattr(self, '_fading_out') and fade > 0:
            rc = QColor(BRAND_PRIMARY)
            rc.setAlphaF(fade * 0.3)
            painter.setPen(QPen(rc, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._tx, self._ty, self._tw, self._th)

        painter.end()
