"""
Shared Qt stylesheet for all StockyAiTrader apps.

One place to change the look of every window.
"""

APP_STYLESHEET = """
    QWidget {
        background-color: #1a1a2e;
        color: #e0e0e0;
        font-family: Consolas;
    }
    QLineEdit, QComboBox, QSpinBox {
        background-color: #16213e;
        border: 1px solid #0f3460;
        padding: 6px;
        color: #e0e0e0;
        border-radius: 4px;
    }
    QPushButton {
        background-color: #0f3460;
        color: white;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 14px;
    }
    QPushButton:hover {
        background-color: #1a5276;
    }
    QPushButton:disabled {
        background-color: #333;
        color: #666;
    }
    QTextEdit {
        background-color: #16213e;
        border: 1px solid #0f3460;
        color: #00ff88;
        font-family: Consolas;
        font-size: 12px;
    }
    QGroupBox {
        border: 1px solid #0f3460;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 15px;
        color: #e0e0e0;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    QLabel {
        color: #b0bec5;
    }
    QProgressBar {
        background-color: #16213e;
        border: 1px solid #0f3460;
        border-radius: 4px;
        text-align: center;
        color: white;
    }
    QProgressBar::chunk {
        background-color: #00ff88;
    }
    QTableWidget {
        background-color: #16213e;
        color: #e0e0e0;
        gridline-color: #0f3460;
        border: 1px solid #0f3460;
    }
    QHeaderView::section {
        background-color: #0f3460;
        color: white;
        padding: 4px;
        border: none;
    }
    QCheckBox {
        color: #e0e0e0;
    }
    QMenuBar {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    QMenuBar::item:selected {
        background-color: #0f3460;
    }
"""


def log_html(msg, level="info"):
    """Format a log message as colored HTML for QTextEdit."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    colors = {
        "info": "#00ff88",
        "warn": "#ffaa00",
        "error": "#ff4444",
        "trade": "#00ccff",
    }
    color = colors.get(level, "#e0e0e0")
    return f'<span style="color:{color}">[{timestamp}] {msg}</span>'
