"""
Premium Table Widget — styled data tables with visual polish.

Provides a QTableWidget subclass with:
- Alternating row colors
- Hover highlighting
- Custom cell renderers for P&L, signals, confidence bars
- Smooth row insertion
"""

from PyQt5.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView, QWidget,
    QHBoxLayout, QLabel, QStyledItemDelegate,
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor, QPainter, QFont, QLinearGradient, QPen, QBrush
from .theme import theme


class PremiumTable(QTableWidget):
    """
    A styled table with alternating rows, hover effects, and
    helper methods for common financial data patterns.
    """

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Custom delegate for cell rendering
        self.setItemDelegate(_PremiumDelegate())

        self._apply_theme()

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QTableWidget {{
                background-color: {theme.color("bg_card")};
                alternate-background-color: {theme.color("table_alt")};
                gridline-color: {theme.color("border")};
                border: 1px solid {theme.color("border")};
                border-radius: 8px;
                outline: none;
                color: {theme.color("text_primary")};
            }}
            QTableWidget::item {{
                padding: 6px 10px;
                border: none;
            }}
            QTableWidget::item:hover {{
                background-color: {theme.color("primary")}20;
            }}
            QTableWidget::item:selected {{
                background-color: {theme.color("primary")}30;
            }}
            QHeaderView::section {{
                background-color: {theme.color("bg_input")};
                color: {theme.color("text_secondary")};
                padding: 8px 10px;
                border: none;
                border-bottom: 2px solid {theme.color("primary")};
                font-weight: 600;
                font-size: 11px;
            }}
        """)

    def add_signal_row(self, values, signal_col=None, pnl_col=None):
        """
        Add a row with automatic color coding.

        Args:
            values:     list of string values
            signal_col: column index containing BUY/SELL/HOLD
            pnl_col:    column index containing P&L value (auto-colors green/red)
        """
        row = self.rowCount()
        self.insertRow(row)

        for col, val in enumerate(values):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignCenter)

            # Color-code signal columns
            if col == signal_col:
                colors = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#f59e0b"}
                color = colors.get(str(val).upper(), "#94a3b8")
                item.setForeground(QColor(color))
                item.setFont(QFont("Segoe UI", 11, QFont.Bold))

            # Color-code P&L columns
            if col == pnl_col:
                try:
                    num = float(val.replace("$", "").replace(",", "").replace("%", "").replace("+", ""))
                    item.setForeground(QColor("#10b981" if num >= 0 else "#ef4444"))
                except (ValueError, AttributeError):
                    pass

            self.setItem(row, col, item)

        return row


class _PremiumDelegate(QStyledItemDelegate):
    """Custom delegate that paints subtle cell borders and rounded selection."""

    def paint(self, painter, option, index):
        # Let the default handle most of the painting
        super().paint(painter, option, index)

        # Add a subtle bottom border to each cell
        painter.setPen(QPen(QColor("#2a2d3a"), 1))
        painter.drawLine(
            option.rect.bottomLeft(),
            option.rect.bottomRight(),
        )


class ConfidenceBar(QWidget):
    """
    A tiny horizontal bar showing confidence level.
    Used inside table cells to visualize probability.
    """

    def __init__(self, value=0.0, parent=None):
        super().__init__(parent)
        self._value = value  # 0.0 - 1.0
        self.setFixedHeight(20)
        self.setMinimumWidth(60)

    def set_value(self, value):
        self._value = max(0.0, min(1.0, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        bar_h = 6
        bar_y = (h - bar_h) // 2

        # Track
        painter.setBrush(theme.qcolor("bg_input"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, bar_y, w, bar_h, 3, 3)

        # Fill
        fill_w = int(w * self._value)
        if fill_w > 0:
            # Green above 60%, yellow 40-60%, red below 40%
            if self._value >= 0.6:
                color = QColor("#10b981")
            elif self._value >= 0.4:
                color = QColor("#f59e0b")
            else:
                color = QColor("#ef4444")

            grad = QLinearGradient(0, 0, fill_w, 0)
            grad.setColorAt(0, color)
            darker = QColor(color)
            darker.setAlphaF(0.7)
            grad.setColorAt(1, darker)

            painter.setBrush(grad)
            painter.drawRoundedRect(0, bar_y, fill_w, bar_h, 3, 3)

        # Value text
        painter.setPen(theme.qcolor("text_primary"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(0, 0, w, h, Qt.AlignCenter, f"{self._value:.0%}")

        painter.end()
