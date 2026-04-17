"""
Animation Library — smooth, premium transitions and effects.

Provides reusable animation functions that can be applied to any QWidget.
All animations use Qt's property animation system for smooth 60fps rendering.

Usage:
    from core.ui.animations import FadeIn, SlideIn, PulseGlow, Shake
    FadeIn(my_widget, duration=400)
    SlideIn(my_widget, direction="left", duration=300)
    PulseGlow(my_widget, color="#0ea5e9", loops=3)
"""

from PyQt5.QtCore import (
    QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup,
    QParallelAnimationGroup, QPoint, QRect, QTimer, pyqtProperty,
)
from PyQt5.QtWidgets import QGraphicsOpacityEffect, QWidget
from PyQt5.QtGui import QColor


def FadeIn(widget, duration=350, start=0.0, end=1.0, ease=QEasingCurve.OutCubic):
    """Fade a widget from transparent to visible."""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity")
    anim.setDuration(duration)
    anim.setStartValue(start)
    anim.setEndValue(end)
    anim.setEasingCurve(ease)
    anim.start()
    # Store reference to prevent garbage collection
    widget._fade_anim = anim
    return anim


def FadeOut(widget, duration=300, callback=None):
    """Fade a widget out, optionally call a function when done."""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity")
    anim.setDuration(duration)
    anim.setStartValue(1.0)
    anim.setEndValue(0.0)
    anim.setEasingCurve(QEasingCurve.InCubic)
    if callback:
        anim.finished.connect(callback)
    anim.start()
    widget._fade_anim = anim
    return anim


def SlideIn(widget, direction="left", distance=50, duration=350):
    """Slide a widget in from a direction (left, right, top, bottom)."""
    pos = widget.pos()
    offsets = {
        "left":   QPoint(pos.x() - distance, pos.y()),
        "right":  QPoint(pos.x() + distance, pos.y()),
        "top":    QPoint(pos.x(), pos.y() - distance),
        "bottom": QPoint(pos.x(), pos.y() + distance),
    }
    start = offsets.get(direction, offsets["left"])

    anim = QPropertyAnimation(widget, b"pos")
    anim.setDuration(duration)
    anim.setStartValue(start)
    anim.setEndValue(pos)
    anim.setEasingCurve(QEasingCurve.OutBack)
    anim.start()
    widget._slide_anim = anim
    return anim


def Shake(widget, intensity=8, duration=400):
    """Shake a widget horizontally — use for errors or invalid input."""
    pos = widget.pos()
    group = QSequentialAnimationGroup(widget)

    steps = 6
    step_dur = duration // steps
    for i in range(steps):
        anim = QPropertyAnimation(widget, b"pos")
        anim.setDuration(step_dur)
        offset = intensity if i % 2 == 0 else -intensity
        # Decay the intensity over time
        decay = 1.0 - (i / steps)
        anim.setEndValue(QPoint(pos.x() + int(offset * decay), pos.y()))
        group.addAnimation(anim)

    # Return to original position
    final = QPropertyAnimation(widget, b"pos")
    final.setDuration(step_dur)
    final.setEndValue(pos)
    group.addAnimation(final)

    group.start()
    widget._shake_anim = group
    return group


def ScaleUp(widget, duration=300):
    """Scale a widget up with a bounce effect using geometry animation."""
    geom = widget.geometry()
    shrunk = QRect(
        geom.x() + geom.width() // 4,
        geom.y() + geom.height() // 4,
        geom.width() // 2,
        geom.height() // 2,
    )
    anim = QPropertyAnimation(widget, b"geometry")
    anim.setDuration(duration)
    anim.setStartValue(shrunk)
    anim.setEndValue(geom)
    anim.setEasingCurve(QEasingCurve.OutBack)
    anim.start()
    widget._scale_anim = anim
    return anim


class PulseEffect:
    """
    Continuously pulse a widget's opacity — draws attention.
    Call stop() to end the pulse.
    """

    def __init__(self, widget, min_opacity=0.6, max_opacity=1.0, duration=1000):
        self.widget = widget
        self.effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(self.effect)

        self.anim = QPropertyAnimation(self.effect, b"opacity")
        self.anim.setDuration(duration)
        self.anim.setStartValue(min_opacity)
        self.anim.setEndValue(max_opacity)
        self.anim.setEasingCurve(QEasingCurve.InOutSine)
        self.anim.setLoopCount(-1)  # Infinite loop
        # Reverse on each cycle
        self.anim.finished.connect(self._reverse)
        self._forward = True

    def start(self):
        self.anim.start()

    def stop(self):
        self.anim.stop()
        self.effect.setOpacity(1.0)

    def _reverse(self):
        if self._forward:
            self.anim.setStartValue(1.0)
            self.anim.setEndValue(0.6)
        else:
            self.anim.setStartValue(0.6)
            self.anim.setEndValue(1.0)
        self._forward = not self._forward


def StaggeredFadeIn(widgets, delay_ms=80, duration=300):
    """Fade in a list of widgets one by one with staggered delay."""
    for i, widget in enumerate(widgets):
        QTimer.singleShot(i * delay_ms, lambda w=widget: FadeIn(w, duration))
