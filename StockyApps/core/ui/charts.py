"""
Custom Chart Widgets — hand-painted financial charts.

These are lightweight QWidget-based charts that don't need matplotlib.
Use them for inline displays, table cells, and dashboards.

- CandlestickChart: OHLC candlestick display
- GaugeWidget: semicircle gauge (confidence, RSI, fear/greed)
- AreaSparkline: filled area mini-chart with gradient
"""

import math
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient,
    QPainterPath, QFont, QConicalGradient,
)


class CandlestickChart(QWidget):
    """
    Compact OHLC candlestick chart widget.
    Green candles = close > open, red = close < open.

    Usage:
        chart = CandlestickChart()
        chart.set_data(ohlc_list)  # list of (open, high, low, close) tuples
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []  # [(open, high, low, close), ...]
        self.setMinimumHeight(120)

    def set_data(self, ohlc_data):
        """Set candle data: list of (open, high, low, close) tuples."""
        self._data = ohlc_data[-60:]  # Show last 60 candles max
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        pad = 8
        n = len(self._data)

        # Price range
        all_highs = [c[1] for c in self._data]
        all_lows = [c[2] for c in self._data]
        price_min = min(all_lows)
        price_max = max(all_highs)
        price_range = price_max - price_min if price_max != price_min else 1

        candle_w = max(2, (w - 2 * pad) / n - 1)
        gap = 1

        def y_pos(price):
            return pad + (1.0 - (price - price_min) / price_range) * (h - 2 * pad)

        for i, (o, hi, lo, c) in enumerate(self._data):
            x = pad + i * (candle_w + gap)
            bullish = c >= o
            color = QColor("#10b981") if bullish else QColor("#ef4444")

            # Wick (high-low line)
            painter.setPen(QPen(color, 1))
            painter.drawLine(
                int(x + candle_w / 2), int(y_pos(hi)),
                int(x + candle_w / 2), int(y_pos(lo)),
            )

            # Body (open-close rectangle)
            body_top = y_pos(max(o, c))
            body_bot = y_pos(min(o, c))
            body_h = max(1, body_bot - body_top)

            painter.setBrush(color if bullish else QColor(color.red(), color.green(), color.blue(), 180))
            painter.setPen(Qt.NoPen)
            painter.drawRect(int(x), int(body_top), int(candle_w), int(body_h))

        painter.end()


class GaugeWidget(QWidget):
    """
    Semicircle gauge displaying a 0-100 value.
    Colors transition from red (0) through yellow (50) to green (100).
    Great for confidence, RSI, fear/greed index.

    Usage:
        gauge = GaugeWidget(label="Confidence")
        gauge.set_value(75)
    """

    def __init__(self, label="", min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self._value = 50
        self._label = label
        self._min = min_val
        self._max = max_val
        self.setFixedSize(140, 90)

    def set_value(self, value):
        self._value = max(self._min, min(self._max, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2
        radius = min(w, h * 2) / 2 - 10

        # Normalize value to 0-1
        norm = (self._value - self._min) / (self._max - self._min) if self._max != self._min else 0.5

        # Background arc (gray track)
        painter.setPen(QPen(QColor("#252836"), 8, Qt.SolidLine, Qt.RoundCap))
        arc_rect = QRectF(cx - radius, h - radius - 5, radius * 2, radius * 2)
        painter.drawArc(arc_rect, 180 * 16, -180 * 16)

        # Value arc (colored)
        # Color: red(0) -> yellow(0.5) -> green(1.0)
        if norm < 0.5:
            r = 239
            g = int(68 + norm * 2 * (158 - 68))
            b = 68
        else:
            r = int(239 - (norm - 0.5) * 2 * (239 - 16))
            g = int(158 + (norm - 0.5) * 2 * (185 - 158))
            b = int(68 + (norm - 0.5) * 2 * (129 - 68))

        color = QColor(r, g, b)
        painter.setPen(QPen(color, 8, Qt.SolidLine, Qt.RoundCap))
        sweep = int(-180 * norm * 16)
        painter.drawArc(arc_rect, 180 * 16, sweep)

        # Value text
        painter.setPen(color)
        painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
        painter.drawText(0, h - 45, w, 30, Qt.AlignCenter, f"{self._value:.0f}")

        # Label
        if self._label:
            painter.setPen(QColor("#94a3b8"))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(0, h - 18, w, 16, Qt.AlignCenter, self._label)

        painter.end()


class AreaSparkline(QWidget):
    """
    Mini area chart with gradient fill — more premium than plain Sparkline.
    Shows trend direction via color (green = up, red = down).

    Usage:
        spark = AreaSparkline()
        spark.set_data([100, 102, 101, 105, 103, 108])
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []
        self.setMinimumSize(100, 40)

    def set_data(self, data):
        self._data = data
        self.update()

    def paintEvent(self, event):
        if not self._data or len(self._data) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        pad = 3

        mn = min(self._data)
        mx = max(self._data)
        rng = mx - mn if mx != mn else 1
        n = len(self._data)

        # Determine trend color
        trending_up = self._data[-1] >= self._data[0]
        color = QColor("#10b981") if trending_up else QColor("#ef4444")

        # Build points
        points = []
        for i, val in enumerate(self._data):
            x = pad + (i / (n - 1)) * (w - 2 * pad)
            y = h - pad - ((val - mn) / rng) * (h - 2 * pad)
            points.append(QPointF(x, y))

        # Fill path
        fill_path = QPainterPath()
        fill_path.moveTo(points[0].x(), h)
        for p in points:
            fill_path.lineTo(p)
        fill_path.lineTo(points[-1].x(), h)
        fill_path.closeSubpath()

        # Gradient fill
        fill_grad = QLinearGradient(0, 0, 0, h)
        fill_color = QColor(color)
        fill_color.setAlphaF(0.25)
        fill_grad.setColorAt(0, fill_color)
        fill_color.setAlphaF(0.02)
        fill_grad.setColorAt(1, fill_color)
        painter.fillPath(fill_path, fill_grad)

        # Line
        pen = QPen(color, 2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        # End dot
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(points[-1], 3, 3)

        painter.end()
