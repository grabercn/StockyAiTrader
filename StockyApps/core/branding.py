"""
Branding constants for Stocky Suite.

Central place for all visual identity: colors, fonts, app metadata.
Import from here instead of hardcoding values.
"""

# ─── App Identity ─────────────────────────────────────────────────────────────
APP_NAME = "Stocky Suite"
APP_VERSION = "2.3.0"
APP_TAGLINE = "AI-Powered Trading Suite"
APP_AUTHOR = "grabercn"
APP_URL = "https://github.com/grabercn/StockyAiTrader"

# ─── Color Palette ────────────────────────────────────────────────────────────
# Primary brand colors
BRAND_PRIMARY = "#0ea5e9"      # Sky blue — trust, technology
BRAND_SECONDARY = "#6366f1"    # Indigo — premium, intelligence
BRAND_ACCENT = "#10b981"       # Emerald — growth, money

# UI backgrounds
BG_DARKEST = "#0f1117"         # Main window background
BG_DARK = "#1a1d28"            # Panel backgrounds
BG_PANEL = "#1e2130"           # Card/group backgrounds
BG_INPUT = "#252836"           # Input fields
BORDER = "#2a2d3a"             # Subtle borders

# Text
TEXT_PRIMARY = "#e2e8f0"       # Main text
TEXT_SECONDARY = "#94a3b8"     # Secondary/muted text
TEXT_MUTED = "#64748b"         # Disabled/hint text

# Trading signals
COLOR_BUY = "#10b981"          # Green — buy/positive
COLOR_SELL = "#ef4444"         # Red — sell/negative
COLOR_HOLD = "#f59e0b"        # Amber — hold/neutral
COLOR_PROFIT = "#10b981"
COLOR_LOSS = "#ef4444"

# Chart colors
CHART_PRICE = "#0ea5e9"       # Price line
CHART_VWAP = "#f59e0b"        # VWAP
CHART_EMA_FAST = "#ef4444"    # Fast EMA
CHART_EMA_SLOW = "#10b981"    # Slow EMA
CHART_GRID = "#1e2130"        # Grid lines

# Status indicators
STATUS_ACTIVE = "#10b981"
STATUS_INACTIVE = "#f59e0b"
STATUS_ERROR = "#ef4444"
STATUS_DISABLED = "#64748b"

# ─── Fonts ────────────────────────────────────────────────────────────────────
FONT_FAMILY = "Segoe UI"       # Windows-native, clean
FONT_MONO = "Consolas"         # Monospace for data/logs
FONT_SIZE_TITLE = 20
FONT_SIZE_HEADING = 14
FONT_SIZE_BODY = 12
FONT_SIZE_SMALL = 10
FONT_SIZE_TINY = 9

# ─── Master Stylesheet ───────────────────────────────────────────────────────
SUITE_STYLESHEET = f"""
    * {{
        font-family: "{FONT_FAMILY}";
        font-size: {FONT_SIZE_BODY}px;
    }}
    QMainWindow, QWidget {{
        background-color: {BG_DARKEST};
        color: {TEXT_PRIMARY};
    }}
    QLabel {{
        color: {TEXT_SECONDARY};
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        padding: 7px 10px;
        color: {TEXT_PRIMARY};
        border-radius: 6px;
        font-size: {FONT_SIZE_BODY}px;
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {BRAND_PRIMARY};
    }}
    QPushButton {{
        background-color: {BRAND_PRIMARY};
        color: white;
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 600;
        font-size: {FONT_SIZE_BODY}px;
        border: none;
    }}
    QPushButton:hover {{
        background-color: #38bdf8;
    }}
    QPushButton:pressed {{
        background-color: #0284c7;
    }}
    QPushButton:disabled {{
        background-color: {BG_INPUT};
        color: {TEXT_MUTED};
    }}
    QGroupBox {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        margin-top: 12px;
        padding: 16px 12px 12px 12px;
        color: {TEXT_SECONDARY};
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 14px;
        padding: 0 6px;
        color: {BRAND_PRIMARY};
    }}
    QTextEdit {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER};
        color: {TEXT_PRIMARY};
        font-family: "{FONT_MONO}";
        font-size: {FONT_SIZE_SMALL}px;
        border-radius: 6px;
        padding: 6px;
    }}
    QTableWidget {{
        background-color: {BG_PANEL};
        color: {TEXT_PRIMARY};
        gridline-color: {BORDER};
        border: 1px solid {BORDER};
        border-radius: 6px;
        font-size: {FONT_SIZE_SMALL}px;
    }}
    QTableWidget::item {{
        padding: 4px 8px;
    }}
    QTableWidget::item:selected {{
        background-color: {BRAND_PRIMARY}40;
    }}
    QHeaderView::section {{
        background-color: {BG_INPUT};
        color: {TEXT_SECONDARY};
        padding: 6px 8px;
        border: none;
        border-bottom: 2px solid {BRAND_PRIMARY};
        font-weight: 600;
        font-size: {FONT_SIZE_SMALL}px;
    }}
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        background-color: {BG_DARKEST};
        border-radius: 0 0 8px 8px;
    }}
    QTabBar::tab {{
        background-color: {BG_DARK};
        color: {TEXT_MUTED};
        padding: 10px 20px;
        border: 1px solid {BORDER};
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        margin-right: 2px;
        font-weight: 600;
    }}
    QTabBar::tab:selected {{
        background-color: {BG_DARKEST};
        color: {BRAND_PRIMARY};
        border-bottom: 2px solid {BRAND_PRIMARY};
    }}
    QTabBar::tab:hover:!selected {{
        color: {TEXT_PRIMARY};
        background-color: {BG_PANEL};
    }}
    QProgressBar {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        text-align: center;
        color: white;
        font-size: {FONT_SIZE_SMALL}px;
        height: 22px;
    }}
    QProgressBar::chunk {{
        background-color: {BRAND_ACCENT};
        border-radius: 5px;
    }}
    QCheckBox {{
        color: {TEXT_PRIMARY};
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid {BORDER};
        background-color: {BG_INPUT};
    }}
    QCheckBox::indicator:checked {{
        background-color: {BRAND_PRIMARY};
        border-color: {BRAND_PRIMARY};
    }}
    QScrollBar:vertical {{
        background-color: {BG_DARKEST};
        width: 10px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background-color: {BORDER};
        border-radius: 5px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {TEXT_MUTED};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QMenuBar {{
        background-color: {BG_DARKEST};
        color: {TEXT_SECONDARY};
        border-bottom: 1px solid {BORDER};
        padding: 2px;
    }}
    QMenuBar::item:selected {{
        background-color: {BRAND_PRIMARY};
        color: white;
        border-radius: 4px;
    }}
    QMenu {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 4px;
    }}
    QMenu::item:selected {{
        background-color: {BRAND_PRIMARY};
        border-radius: 4px;
    }}
    QToolTip {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 6px;
        font-size: {FONT_SIZE_SMALL}px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 30px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 6px;
        selection-background-color: {BRAND_PRIMARY};
        padding: 4px;
        outline: none;
    }}
"""


# ─── Light Theme ──────────────────────────────────────────────────────────────
LIGHT_STYLESHEET = f"""
    * {{
        font-family: "{FONT_FAMILY}";
        font-size: {FONT_SIZE_BODY}px;
    }}
    QMainWindow, QWidget {{
        background-color: #f8fafc;
        color: #1e293b;
    }}
    QLabel {{
        color: #475569;
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 7px 10px;
        color: #1e293b;
        border-radius: 6px;
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {BRAND_PRIMARY};
    }}
    QPushButton {{
        background-color: {BRAND_PRIMARY};
        color: white;
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 600;
        border: none;
    }}
    QPushButton:hover {{
        background-color: #38bdf8;
    }}
    QPushButton:disabled {{
        background-color: #e2e8f0;
        color: #94a3b8;
    }}
    QGroupBox {{
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-top: 12px;
        padding: 16px 12px 12px 12px;
        color: #475569;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 14px;
        padding: 0 6px;
        color: {BRAND_PRIMARY};
    }}
    QTextEdit {{
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        color: #1e293b;
        font-family: "{FONT_MONO}";
        font-size: {FONT_SIZE_SMALL}px;
        border-radius: 6px;
    }}
    QTableWidget {{
        background-color: #ffffff;
        color: #1e293b;
        gridline-color: #e2e8f0;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }}
    QHeaderView::section {{
        background-color: #f1f5f9;
        color: #475569;
        padding: 6px 8px;
        border: none;
        border-bottom: 2px solid {BRAND_PRIMARY};
        font-weight: 600;
    }}
    QTabWidget::pane {{
        border: 1px solid #e2e8f0;
        background-color: #f8fafc;
    }}
    QTabBar::tab {{
        background-color: #f1f5f9;
        color: #64748b;
        padding: 10px 20px;
        border: 1px solid #e2e8f0;
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        margin-right: 2px;
        font-weight: 600;
    }}
    QTabBar::tab:selected {{
        background-color: #f8fafc;
        color: {BRAND_PRIMARY};
        border-bottom: 2px solid {BRAND_PRIMARY};
    }}
    QProgressBar {{
        background-color: #e2e8f0;
        border: none;
        border-radius: 6px;
        text-align: center;
        color: #1e293b;
        height: 22px;
    }}
    QProgressBar::chunk {{
        background-color: {BRAND_ACCENT};
        border-radius: 5px;
    }}
    QCheckBox {{ color: #1e293b; }}
    QCheckBox::indicator {{
        width: 18px; height: 18px; border-radius: 4px;
        border: 2px solid #cbd5e1; background-color: #ffffff;
    }}
    QCheckBox::indicator:checked {{
        background-color: {BRAND_PRIMARY}; border-color: {BRAND_PRIMARY};
    }}
    QScrollBar:vertical {{
        background-color: #f8fafc; width: 10px; border: none;
    }}
    QScrollBar::handle:vertical {{
        background-color: #cbd5e1; border-radius: 5px; min-height: 30px;
    }}
    QMenuBar {{
        background-color: #f8fafc; color: #475569; border-bottom: 1px solid #e2e8f0;
    }}
    QMenuBar::item:selected {{
        background-color: {BRAND_PRIMARY}; color: white; border-radius: 4px;
    }}
    QMenu {{
        background-color: #ffffff; color: #1e293b; border: 1px solid #e2e8f0; border-radius: 6px;
    }}
    QMenu::item:selected {{
        background-color: {BRAND_PRIMARY}; border-radius: 4px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 30px;
    }}
    QComboBox QAbstractItemView {{
        background-color: #ffffff;
        color: #1e293b;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        selection-background-color: {BRAND_PRIMARY};
        padding: 4px;
        outline: none;
    }}
"""


# ─── Theme Detection & Switching ─────────────────────────────────────────────

def detect_system_theme():
    """Detect Windows dark/light mode from registry. Returns 'dark' or 'light'."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        )
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        winreg.CloseKey(key)
        return "light" if value == 1 else "dark"
    except Exception:
        return "dark"  # Default to dark


def get_stylesheet(theme="auto"):
    """Get the stylesheet for a given theme. 'auto' detects from system."""
    if theme == "auto":
        theme = detect_system_theme()
    return LIGHT_STYLESHEET if theme == "light" else SUITE_STYLESHEET


def log_html(msg, level="info"):
    """Format a log message as colored HTML for QTextEdit display."""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {
        "info": BRAND_PRIMARY,
        "warn": COLOR_HOLD,
        "error": COLOR_SELL,
        "trade": BRAND_ACCENT,
        "system": TEXT_MUTED,
    }
    color = colors.get(level, TEXT_SECONDARY)
    return (
        f'<span style="color:{TEXT_MUTED}">{ts}</span> '
        f'<span style="color:{color}">{msg}</span>'
    )
