"""
Chart Tooltip — adds hover tooltips to matplotlib charts.

Attach to any FigureCanvas to show data values on mouseover.

Usage:
    tooltip = ChartTooltip(canvas, ax, x_data, y_data, fmt="$,.2f")
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class ChartTooltip:
    """Hover tooltip that shows value at cursor position on a matplotlib chart."""

    def __init__(self, canvas, ax, x_data, y_data, fmt=",.2f", prefix="$", color="#0ea5e9"):
        self.canvas = canvas
        self.ax = ax
        self.x_data = list(x_data)
        self.y_data = list(y_data)
        self.fmt = fmt
        self.prefix = prefix
        self.color = color

        # Create annotation (hidden initially)
        self._annot = ax.annotate(
            "", xy=(0, 0),
            xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1e2130", ec="#0ea5e9", alpha=0.9),
            fontsize=9, color="white",
            arrowprops=dict(arrowstyle="->", color="#0ea5e9", lw=1),
        )
        self._annot.set_visible(False)

        # Vertical line at cursor
        self._vline = ax.axvline(x=0, color=color, alpha=0.3, linewidth=1, linestyle="--")
        self._vline.set_visible(False)

        # Dot at data point
        self._dot, = ax.plot([], [], "o", color=color, markersize=6, zorder=10)
        self._dot.set_visible(False)

        # Connect mouse events
        canvas.mpl_connect("motion_notify_event", self._on_move)
        canvas.mpl_connect("axes_leave_event", self._on_leave)

    def _on_move(self, event):
        if event.inaxes != self.ax or not self.x_data:
            self._hide()
            return

        # Find nearest data point
        try:
            # Handle datetime x-axis
            from matplotlib.dates import num2date, date2num
            x_nums = [date2num(x) if hasattr(x, 'timestamp') else float(i) for i, x in enumerate(self.x_data)]
            cursor_x = event.xdata

            diffs = [abs(xn - cursor_x) for xn in x_nums]
            idx = diffs.index(min(diffs))

            x_val = self.x_data[idx]
            y_val = self.y_data[idx]

            # Format text
            if hasattr(x_val, 'strftime'):
                x_str = x_val.strftime("%m/%d %H:%M")
            else:
                x_str = str(x_val)

            text = f"{x_str}\n{self.prefix}{y_val:{self.fmt}}"

            self._annot.xy = (x_nums[idx], y_val)
            self._annot.set_text(text)
            self._annot.set_visible(True)

            self._vline.set_xdata([x_nums[idx]])
            self._vline.set_visible(True)

            self._dot.set_data([x_nums[idx]], [y_val])
            self._dot.set_visible(True)

            self.canvas.draw_idle()
        except Exception:
            self._hide()

    def _on_leave(self, event):
        self._hide()

    def _hide(self):
        self._annot.set_visible(False)
        self._vline.set_visible(False)
        self._dot.set_visible(False)
        self.canvas.draw_idle()
