"""
Detail Popup — rich data viewer that opens when clicking stat cards.

Shows a full chart with historical data, live updates, filters,
and key statistics in a polished popup window.
"""

import os
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QWidget, QGridLayout, QApplication,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter, QLinearGradient, QPen, QBrush
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..branding import (
    APP_NAME, BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT,
    COLOR_BUY, COLOR_SELL, COLOR_HOLD, BG_DARKEST, BG_DARK, BG_PANEL,
    BG_INPUT, BORDER, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    FONT_FAMILY, FONT_MONO,
)
from .theme import theme
from .icons import StockyIcons


class _PopupBackground(QWidget):
    """Subtle animated gradient background for popups."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

    def _tick(self):
        self._phase += 0.01
        self.update()

    def paintEvent(self, event):
        import math
        painter = QPainter(self)
        w, h = self.width(), self.height()
        shift = (math.sin(self._phase) + 1) / 2 * 0.15

        if theme.is_dark:
            grad = QLinearGradient(0, 0, w, h)
            grad.setColorAt(0, QColor(15, 17, 23))
            grad.setColorAt(0.4 + shift, QColor(20, 24, 38))
            grad.setColorAt(1, QColor(12, 14, 20))
        else:
            grad = QLinearGradient(0, 0, w, h)
            grad.setColorAt(0, QColor(248, 250, 252))
            grad.setColorAt(0.5, QColor(241, 245, 249))
            grad.setColorAt(1, QColor(248, 250, 252))
        painter.fillRect(0, 0, w, h, grad)

        # Top accent
        line = QLinearGradient(0, 0, w, 0)
        line.setColorAt(0, QColor(BRAND_PRIMARY + "00"))
        line.setColorAt(0.3, QColor(BRAND_PRIMARY))
        line.setColorAt(0.7, QColor(BRAND_ACCENT))
        line.setColorAt(1, QColor(BRAND_ACCENT + "00"))
        painter.setPen(QPen(QBrush(line), 2))
        painter.drawLine(0, 0, w, 0)
        painter.end()


class _MiniStat(QWidget):
    """Small stat display for the popup grid."""

    def __init__(self, label, value="--", color=BRAND_PRIMARY, parent=None):
        super().__init__(parent)
        self._label = label
        self._value = value
        self._color = color
        self.setMinimumHeight(50)

    def set_value(self, value, color=None):
        self._value = value
        if color:
            self._color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Card bg
        painter.setBrush(theme.qcolor("bg_card"))
        painter.setPen(QPen(theme.qcolor("border"), 1))
        painter.drawRoundedRect(2, 2, w - 4, h - 4, 8, 8)

        # Label
        painter.setPen(theme.qcolor("text_muted"))
        painter.setFont(QFont(FONT_FAMILY, 9))
        painter.drawText(10, 6, w - 20, 18, Qt.AlignLeft | Qt.AlignVCenter, self._label)

        # Value
        painter.setPen(QColor(self._color))
        painter.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        painter.drawText(10, 24, w - 20, 22, Qt.AlignLeft | Qt.AlignVCenter, self._value)
        painter.end()


class DetailPopup(QDialog):
    """
    Rich detail popup with chart, stats, and period filters.

    Usage:
        popup = DetailPopup(
            title="Portfolio Value",
            fetch_fn=lambda period: broker.get_portfolio_history(period=period),
            extract_fn=lambda data: (timestamps, values, "$"),
            stats={"High": "$105,234", "Low": "$98,102", ...},
        )
        popup.exec_()
    """

    def __init__(self, title, fetch_fn, extract_fn, stats=None,
                 periods=None, default_period=1, parent=None):
        """
        Args:
            title:          Window title / header text
            fetch_fn:       callable(period_str) -> raw data dict
            extract_fn:     callable(raw_data) -> (timestamps_list, values_list, unit_str)
            stats:          dict of {label: value} for the stat grid
            periods:        list of (display_name, api_period, api_timeframe) tuples
            default_period: index of default period in the list
        """
        super().__init__(parent)
        self.setWindowTitle(f"{title} — {APP_NAME}")
        self.setMinimumSize(700, 500)
        self.resize(800, 550)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._title = title
        self._fetch_fn = fetch_fn
        self._extract_fn = extract_fn
        self._stats_data = stats or {}
        self._periods = periods or [
            ("1 Day", "1D", "15Min"),
            ("1 Week", "1W", "1H"),
            ("1 Month", "1M", "1D"),
            ("3 Months", "3M", "1D"),
            ("1 Year", "1A", "1D"),
        ]

        # Use stylesheet background instead of paintEvent
        # (paintEvent on parent paints OVER child widgets like FigureCanvas)
        bg = theme.color("bg_base")
        self.setStyleSheet(f"""
            QDialog {{ background-color: {bg}; }}
            QLabel {{ background: transparent; }}
        """)
        self._build()

        # Load default period
        self._period_cb.setCurrentIndex(min(default_period, len(self._periods) - 1))
        self._load_data()

        # Auto-refresh every 30 seconds
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._load_data)
        self._refresh_timer.start(30000)

    def _build(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # Header row
        hdr = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(StockyIcons.get("chart_up", 24, BRAND_PRIMARY))
        icon.setStyleSheet("background: transparent;")
        hdr.addWidget(icon)

        title_lbl = QLabel(self._title)
        title_lbl.setFont(QFont(FONT_FAMILY, 18, QFont.Bold))
        title_lbl.setStyleSheet(f"color: {BRAND_PRIMARY}; background: transparent;")
        hdr.addWidget(title_lbl)
        hdr.addStretch()

        # Period selector
        self._period_cb = QComboBox()
        for display, _, _ in self._periods:
            self._period_cb.addItem(display)
        self._period_cb.currentIndexChanged.connect(lambda _: self._load_data())
        self._period_cb.setStyleSheet(f"""
            QComboBox {{ background: {theme.color('bg_input')}; border: 1px solid {theme.color('border')};
                padding: 6px 12px; border-radius: 6px; color: {theme.color('text_primary')}; min-width: 100px; }}
        """)
        hdr.addWidget(self._period_cb)

        refresh_btn = QPushButton()
        refresh_btn.setIcon(StockyIcons.get_icon("scan", 16, BRAND_PRIMARY))
        refresh_btn.setStyleSheet(f"background: {theme.color('bg_input')}; border: 1px solid {theme.color('border')}; padding: 6px; border-radius: 6px;")
        refresh_btn.setToolTip("Refresh data")
        refresh_btn.clicked.connect(self._load_data)
        hdr.addWidget(refresh_btn)

        layout.addLayout(hdr)

        # Current value display
        self._current_lbl = QLabel("--")
        self._current_lbl.setFont(QFont(FONT_FAMILY, 28, QFont.Bold))
        self._current_lbl.setStyleSheet(f"color: {BRAND_PRIMARY}; background: transparent;")
        layout.addWidget(self._current_lbl)

        self._change_lbl = QLabel("")
        self._change_lbl.setFont(QFont(FONT_FAMILY, 12))
        self._change_lbl.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent;")
        layout.addWidget(self._change_lbl)

        # Chart
        self._fig_bg = theme.color("bg_base")
        self._ax_bg = theme.color("bg_card")
        self._figure = plt.Figure(dpi=100, facecolor=self._fig_bg)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(180)
        layout.addWidget(self._canvas, 1)

        # Stats grid
        self._stats_grid = QGridLayout()
        self._stats_grid.setSpacing(6)
        self._stat_widgets = {}
        layout.addLayout(self._stats_grid)

        # Timestamp
        self._ts_label = QLabel("")
        self._ts_label.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 9px; background: transparent;")
        self._ts_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self._ts_label)

        self.setLayout(layout)

    def _load_data(self):
        idx = self._period_cb.currentIndex()
        if idx < 0 or idx >= len(self._periods):
            return

        _, api_period, api_tf = self._periods[idx]

        try:
            raw = self._fetch_fn(api_period, api_tf)
            if not raw or "error" in raw:
                self._current_lbl.setText("No data")
                return

            timestamps, values, unit = self._extract_fn(raw)
            if not values:
                self._current_lbl.setText("No data")
                return

            # Current value
            current = values[-1]
            first = values[0]
            change = current - first
            change_pct = (change / first * 100) if first != 0 else 0

            if unit == "$":
                self._current_lbl.setText(f"${current:,.2f}")
                change_str = f"${change:+,.2f} ({change_pct:+.2f}%)"
            else:
                self._current_lbl.setText(f"{current:,.2f}{unit}")
                change_str = f"{change:+,.2f} ({change_pct:+.2f}%)"

            change_color = COLOR_BUY if change >= 0 else COLOR_SELL
            self._change_lbl.setText(f"{change_str}  vs period start")
            self._change_lbl.setStyleSheet(f"color: {change_color}; background: transparent;")
            self._current_lbl.setStyleSheet(f"color: {change_color}; background: transparent;")

            # Chart
            self._draw_chart(timestamps, values, unit, change >= 0)

            # Stats
            self._update_stats(values, unit, timestamps)

            self._ts_label.setText(f"Updated {datetime.now().strftime('%H:%M:%S')}  •  Auto-refreshes every 30s")

        except Exception as e:
            self._current_lbl.setText(f"Error: {e}")

    def _draw_chart(self, timestamps, values, unit, trending_up):
        self._figure.clear()
        self._figure.set_facecolor(self._fig_bg)
        self._figure.patch.set_facecolor(self._fig_bg)
        panel_bg = self._ax_bg

        ax = self._figure.add_subplot(111)
        ax.set_facecolor(panel_bg)

        color = COLOR_BUY if trending_up else COLOR_SELL

        ax.plot(timestamps, values, color=color, linewidth=2)
        ax.fill_between(timestamps, values, alpha=0.08, color=color)

        # High/low markers
        max_idx = values.index(max(values))
        min_idx = values.index(min(values))
        ax.scatter([timestamps[max_idx]], [values[max_idx]], color=COLOR_BUY, s=40, zorder=5, marker="^")
        ax.scatter([timestamps[min_idx]], [values[min_idx]], color=COLOR_SELL, s=40, zorder=5, marker="v")

        # Latest value marker
        ax.scatter([timestamps[-1]], [values[-1]], color=color, s=50, zorder=5, edgecolors="white", linewidth=1.5)

        text_c = TEXT_SECONDARY if theme.is_dark else "#475569"
        muted_c = TEXT_MUTED if theme.is_dark else "#94a3b8"
        grid_c = BORDER if theme.is_dark else "#e2e8f0"

        ax.set_title(self._title, color=text_c, fontsize=11, pad=8)
        ax.tick_params(colors=muted_c, labelsize=8)
        ax.grid(True, alpha=0.15, color=grid_c)

        if unit == "$":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        if hasattr(timestamps[0], 'strftime'):
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d"))
            self._figure.autofmt_xdate()

        self._figure.tight_layout()
        self._canvas.draw()

    def _update_stats(self, values, unit, timestamps):
        # Clear old stats
        while self._stats_grid.count():
            item = self._stats_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._stat_widgets.clear()

        high = max(values)
        low = min(values)
        avg = sum(values) / len(values)
        current = values[-1]
        first = values[0]
        change = current - first
        vol = max(values) - min(values)  # Range as volatility proxy

        fmt = lambda v: f"${v:,.2f}" if unit == "$" else f"{v:,.2f}"

        stats = [
            ("Current", fmt(current), BRAND_PRIMARY),
            ("High", fmt(high), COLOR_BUY),
            ("Low", fmt(low), COLOR_SELL),
            ("Average", fmt(avg), TEXT_SECONDARY),
            ("Change", f"{'+' if change >= 0 else ''}{fmt(change)}", COLOR_BUY if change >= 0 else COLOR_SELL),
            ("Range", fmt(vol), BRAND_SECONDARY),
            ("Data Points", str(len(values)), TEXT_MUTED),
            ("Period Start", fmt(first), TEXT_MUTED),
        ]

        # Add any extra stats passed in
        for label, value in self._stats_data.items():
            stats.append((label, str(value), TEXT_SECONDARY))

        cols = 4
        for i, (label, value, color) in enumerate(stats):
            w = _MiniStat(label, value, color)
            self._stats_grid.addWidget(w, i // cols, i % cols)
            self._stat_widgets[label] = w


def show_equity_popup(broker, parent=None):
    """Convenience: show portfolio equity detail popup."""
    def fetch(period, tf):
        return broker.get_portfolio_history(period=period, timeframe=tf)

    def extract(data):
        if not data or "error" in data or not data.get("equity"):
            return [], [], "$"
        eq = [e for e in data["equity"] if e is not None]
        ts = [datetime.fromtimestamp(t) for t, e in zip(data["timestamp"], data["equity"]) if e is not None]
        return ts, eq, "$"

    popup = DetailPopup("Portfolio Equity", fetch, extract, parent=parent)
    popup.exec_()


def show_pnl_popup(broker, parent=None):
    """Convenience: show P&L detail popup."""
    def fetch(period, tf):
        return broker.get_portfolio_history(period=period, timeframe=tf)

    def extract(data):
        if not data or "error" in data or not data.get("profit_loss"):
            return [], [], "$"
        pl = [p for p in data["profit_loss"] if p is not None]
        ts = [datetime.fromtimestamp(t) for t, p in zip(data["timestamp"], data["profit_loss"]) if p is not None]
        return ts, pl, "$"

    popup = DetailPopup("Profit & Loss", fetch, extract, parent=parent)
    popup.exec_()


def show_buying_power_popup(broker, parent=None):
    """Convenience: show buying power detail popup."""
    def fetch(period, tf):
        return broker.get_portfolio_history(period=period, timeframe=tf)

    def extract(data):
        if not data or "error" in data or not data.get("equity"):
            return [], [], "$"
        eq = [e for e in data["equity"] if e is not None]
        ts = [datetime.fromtimestamp(t) for t, e in zip(data["timestamp"], data["equity"]) if e is not None]
        return ts, eq, "$"

    popup = DetailPopup("Buying Power History", fetch, extract,
                        stats={"Note": "Buying power = cash + margin"},
                        parent=parent)
    popup.exec_()
