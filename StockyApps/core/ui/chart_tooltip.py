"""
Chart Tooltip — adds hover tooltips to matplotlib charts.

Works with both datetime and integer x-axis data.

Usage:
    # Integer x-axis with datetime labels:
    tooltip = ChartTooltip(canvas, ax, x_indices, y_data, x_labels=timestamps)

    # Direct datetime x-axis:
    tooltip = ChartTooltip(canvas, ax, timestamps, y_data)
"""


class ChartTooltip:
    """Hover tooltip that shows value at cursor position on a matplotlib chart."""

    def __init__(self, canvas, ax, x_data, y_data, x_labels=None,
                 fmt=",.2f", prefix="$", color="#0ea5e9"):
        self.canvas = canvas
        self.ax = ax
        self.x_data = list(x_data)      # Plotted x values (ints or datetimes)
        self.y_data = list(y_data)
        self.x_labels = list(x_labels) if x_labels else None  # Optional display labels
        self.fmt = fmt
        self.prefix = prefix

        self._annot = ax.annotate(
            "", xy=(0, 0),
            xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1e2130", ec=color, alpha=0.9),
            fontsize=9, color="white",
            arrowprops=dict(arrowstyle="->", color=color, lw=1),
        )
        self._annot.set_visible(False)

        self._vline = ax.axvline(x=0, color=color, alpha=0.3, linewidth=1, linestyle="--")
        self._vline.set_visible(False)

        self._dot, = ax.plot([], [], "o", color=color, markersize=6, zorder=10)
        self._dot.set_visible(False)

        canvas.mpl_connect("motion_notify_event", self._on_move)
        canvas.mpl_connect("axes_leave_event", self._on_leave)

    def _on_move(self, event):
        if event.inaxes != self.ax or not self.x_data:
            self._hide()
            return

        try:
            cursor_x = event.xdata
            if cursor_x is None:
                self._hide()
                return

            # Convert x_data to floats for comparison
            x_floats = []
            for x in self.x_data:
                if hasattr(x, 'timestamp'):
                    from matplotlib.dates import date2num
                    x_floats.append(date2num(x))
                else:
                    x_floats.append(float(x))

            # Find nearest point
            diffs = [abs(xf - cursor_x) for xf in x_floats]
            idx = diffs.index(min(diffs))

            x_plot = x_floats[idx]
            y_val = self.y_data[idx]

            # Display label
            if self.x_labels and idx < len(self.x_labels):
                label = self.x_labels[idx]
                x_str = label.strftime("%m/%d %H:%M") if hasattr(label, 'strftime') else str(label)
            elif hasattr(self.x_data[idx], 'strftime'):
                x_str = self.x_data[idx].strftime("%m/%d %H:%M")
            else:
                x_str = str(self.x_data[idx])

            text = f"{x_str}\n{self.prefix}{y_val:{self.fmt}}"

            self._annot.xy = (x_plot, y_val)
            self._annot.set_text(text)
            self._annot.set_visible(True)

            self._vline.set_xdata([x_plot])
            self._vline.set_visible(True)

            self._dot.set_data([x_plot], [y_val])
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
