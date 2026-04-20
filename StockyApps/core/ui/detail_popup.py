"""
Detail Popup — clean chart + stats viewer for stat card clicks.

REBUILT from scratch — no custom paintEvent, no background widgets.
Just a styled QDialog with a matplotlib chart that actually works.
"""

import os
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QWidget, QGridLayout, QApplication, QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..branding import (
    APP_NAME, BRAND_PRIMARY, BRAND_ACCENT,
    COLOR_BUY, COLOR_SELL,
    FONT_FAMILY, FONT_MONO, chart_colors,
)
from .theme import theme
from .icons import StockyIcons


class DetailPopup(QDialog):
    """
    Clean detail popup with chart, stats, and period filters.
    No custom painting — just styled widgets that work.
    """

    def __init__(self, title, fetch_fn, extract_fn, stats=None,
                 periods=None, default_period=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title} — {APP_NAME}")
        self.setMinimumSize(700, 500)
        self.resize(800, 550)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        # Style the dialog background via stylesheet (NOT paintEvent)
        bg = theme.color("bg_base")
        self.setStyleSheet(f"QDialog {{ background-color: {bg}; }} QLabel {{ background: transparent; }}")

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

        self._build()

        self._period_cb.setCurrentIndex(min(default_period, len(self._periods) - 1))
        self._load_data()

        # Auto-refresh
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._load_data)
        self._timer.start(30000)

    def _build(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header
        hdr = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(StockyIcons.get("chart_up", 24, BRAND_PRIMARY))
        hdr.addWidget(icon)
        title_lbl = QLabel(self._title)
        title_lbl.setFont(QFont(FONT_FAMILY, 18, QFont.Bold))
        title_lbl.setStyleSheet(f"color: {BRAND_PRIMARY};")
        hdr.addWidget(title_lbl)
        hdr.addStretch()

        self._period_cb = QComboBox()
        for display, _, _ in self._periods:
            self._period_cb.addItem(display)
        self._period_cb.currentIndexChanged.connect(lambda _: self._load_data())
        hdr.addWidget(self._period_cb)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(f"font-size: 11px; padding: 5px;")
        refresh_btn.clicked.connect(self._load_data)
        hdr.addWidget(refresh_btn)
        layout.addLayout(hdr)

        # Current value
        self._current_lbl = QLabel("--")
        self._current_lbl.setFont(QFont(FONT_FAMILY, 26, QFont.Bold))
        self._current_lbl.setStyleSheet(f"color: {BRAND_PRIMARY};")
        layout.addWidget(self._current_lbl)

        self._change_lbl = QLabel("")
        self._change_lbl.setFont(QFont(FONT_FAMILY, 11))
        layout.addWidget(self._change_lbl)

        # Chart — create with proper background from the start
        cc = chart_colors()
        self._cc = cc
        self._figure = plt.Figure(figsize=(8, 3.5), dpi=100)
        self._figure.set_facecolor(cc["fig_bg"])
        self._figure.patch.set_facecolor(cc["fig_bg"])

        self._canvas = FigureCanvas(self._figure)
        # Force the Qt widget background to match via palette
        pal = self._canvas.palette()
        pal.setColor(QPalette.Window, QColor(cc["fig_bg"]))
        pal.setColor(QPalette.Base, QColor(cc["fig_bg"]))
        self._canvas.setPalette(pal)
        self._canvas.setAutoFillBackground(True)
        self._canvas.setMinimumHeight(200)
        layout.addWidget(self._canvas, 1)

        # Stats grid in scroll area
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setMaximumHeight(140)
        stats_scroll.setStyleSheet("QScrollArea { border: none; }")
        stats_inner = QWidget()
        self._stats_grid = QGridLayout()
        self._stats_grid.setSpacing(6)
        stats_inner.setLayout(self._stats_grid)
        stats_scroll.setWidget(stats_inner)
        layout.addWidget(stats_scroll)

        # Timestamp
        self._ts_label = QLabel("")
        self._ts_label.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 9px;")
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
            if not raw or (isinstance(raw, dict) and "error" in raw):
                self._current_lbl.setText("No data available")
                self._current_lbl.setStyleSheet(f"color: {theme.color('text_muted')};")
                return

            timestamps, values, unit = self._extract_fn(raw)
            if not values or len(values) < 2:
                self._current_lbl.setText("Not enough data")
                self._current_lbl.setStyleSheet(f"color: {theme.color('text_muted')};")
                return

            current = values[-1]
            first = values[0]
            change = current - first
            change_pct = (change / first * 100) if first != 0 else 0

            fmt = lambda v: f"${v:,.2f}" if unit == "$" else f"{v:,.2f}{unit}"
            self._current_lbl.setText(fmt(current))

            up = change >= 0
            color = COLOR_BUY if up else COLOR_SELL
            self._current_lbl.setStyleSheet(f"color: {color};")
            self._change_lbl.setText(f"{'+'if up else ''}{fmt(change)} ({change_pct:+.2f}%) vs period start")
            self._change_lbl.setStyleSheet(f"color: {color};")

            self._draw_chart(timestamps, values, unit, up)
            self._update_stats(values, unit)
            self._ts_label.setText(f"Updated {datetime.now().strftime('%H:%M:%S')} · Auto-refreshes every 30s")

        except Exception as e:
            self._current_lbl.setText(f"Error: {str(e)[:50]}")
            self._current_lbl.setStyleSheet(f"color: {COLOR_SELL};")

    def _draw_chart(self, timestamps, values, unit, trending_up):
        cc = self._cc
        self._figure.clear()
        self._figure.set_facecolor(cc["fig_bg"])
        self._figure.patch.set_facecolor(cc["fig_bg"])

        ax = self._figure.add_subplot(111)
        ax.set_facecolor(cc["ax_bg"])

        color = COLOR_BUY if trending_up else COLOR_SELL
        ax.plot(timestamps, values, color=color, linewidth=2)
        ax.fill_between(timestamps, values, alpha=0.08, color=color)

        # High/low/current markers
        max_idx = values.index(max(values))
        min_idx = values.index(min(values))
        ax.scatter([timestamps[max_idx]], [values[max_idx]], color=COLOR_BUY, s=40, zorder=5, marker="^")
        ax.scatter([timestamps[min_idx]], [values[min_idx]], color=COLOR_SELL, s=40, zorder=5, marker="v")
        ax.scatter([timestamps[-1]], [values[-1]], color=color, s=50, zorder=5, edgecolors="white", linewidth=1.5)

        ax.set_title(self._title, color=cc["text"], fontsize=11, pad=8)
        ax.tick_params(colors=cc["muted"], labelsize=8)
        ax.grid(True, alpha=0.15, color=cc["grid"])

        if unit == "$":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        if hasattr(timestamps[0], 'strftime'):
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d"))
            self._figure.autofmt_xdate()

        self._figure.tight_layout()
        self._canvas.draw()

    def _update_stats(self, values, unit):
        # Clear old
        while self._stats_grid.count():
            item = self._stats_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        high = max(values)
        low = min(values)
        avg = sum(values) / len(values)
        current = values[-1]
        first = values[0]
        change = current - first

        fmt = lambda v: f"${v:,.2f}" if unit == "$" else f"{v:,.2f}"

        stats = [
            ("Current", fmt(current), BRAND_PRIMARY),
            ("High", fmt(high), COLOR_BUY),
            ("Low", fmt(low), COLOR_SELL),
            ("Average", fmt(avg), theme.color("text_secondary")),
            ("Change", f"{'+'if change>=0 else ''}{fmt(change)}", COLOR_BUY if change >= 0 else COLOR_SELL),
            ("Range", fmt(high - low), theme.color("text_secondary")),
            ("Data Points", str(len(values)), theme.color("text_muted")),
            ("Period Start", fmt(first), theme.color("text_muted")),
        ]

        for label, value in self._stats_data.items():
            stats.append((label, str(value), theme.color("text_secondary")))

        cols = 4
        for i, (label, value, color) in enumerate(stats):
            cell = QWidget()
            cell.setStyleSheet(f"background-color: {theme.color('bg_card')}; border: 1px solid {theme.color('border')}; border-radius: 6px;")
            cl = QVBoxLayout()
            cl.setContentsMargins(8, 4, 8, 4)
            cl.setSpacing(1)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {theme.color('text_muted')}; font-size: 9px; border: none;")
            cl.addWidget(lbl)
            val = QLabel(value)
            val.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px; border: none;")
            cl.addWidget(val)
            cell.setLayout(cl)
            self._stats_grid.addWidget(cell, i // cols, i % cols)


# ─── Convenience launchers ────────────────────────────────────────────────────

def show_equity_popup(broker, parent=None):
    def fetch(period, tf):
        return broker.get_portfolio_history(period=period, timeframe=tf)

    def extract(data):
        if not data or "error" in data or not data.get("equity"):
            return [], [], "$"
        eq = [e for e in data["equity"] if e is not None]
        ts = [datetime.fromtimestamp(t) for t, e in zip(data["timestamp"], data["equity"]) if e is not None]
        return ts, eq, "$"

    DetailPopup("Portfolio Equity", fetch, extract, parent=parent).exec_()


def show_pnl_popup(broker, parent=None):
    def fetch(period, tf):
        return broker.get_portfolio_history(period=period, timeframe=tf)

    def extract(data):
        if not data or "error" in data or not data.get("profit_loss"):
            return [], [], "$"
        pl = [p for p in data["profit_loss"] if p is not None]
        ts = [datetime.fromtimestamp(t) for t, p in zip(data["timestamp"], data["profit_loss"]) if p is not None]
        return ts, pl, "$"

    DetailPopup("Profit & Loss", fetch, extract, parent=parent).exec_()


def show_buying_power_popup(broker, parent=None):
    def fetch(period, tf):
        return broker.get_portfolio_history(period=period, timeframe=tf)

    def extract(data):
        if not data or "error" in data or not data.get("equity"):
            return [], [], "$"
        eq = [e for e in data["equity"] if e is not None]
        ts = [datetime.fromtimestamp(t) for t, e in zip(data["timestamp"], data["equity"]) if e is not None]
        return ts, eq, "$"

    DetailPopup("Buying Power History", fetch, extract,
                stats={"Note": "Buying power = cash + margin"},
                parent=parent).exec_()
