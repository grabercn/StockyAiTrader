"""
Programmatic SVG Icon Library — no external files needed.

Each icon is generated as a QPixmap from inline SVG strings.
Icons auto-tint to any color, scale to any size, and are cached.

Usage:
    from core.ui.icons import StockyIcons
    icon = StockyIcons.get("chart_up", size=24, color="#10b981")
    label.setPixmap(icon)
"""

from PyQt5.QtGui import QPixmap, QPainter, QColor, QIcon
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtCore import QByteArray, QSize, Qt

# ─── SVG Templates ───────────────────────────────────────────────────────────
# Each SVG uses {color} placeholder for tinting.

_SVGS = {
    "chart_up": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <polyline points="22,6 13.5,14.5 8.5,9.5 2,16" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <polyline points="16,6 22,6 22,12" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "chart_down": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <polyline points="22,18 13.5,9.5 8.5,14.5 2,8" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <polyline points="16,18 22,18 22,12" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "dollar": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <line x1="12" y1="1" x2="12" y2="23" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <path d="M17,5H9.5a3.5,3.5,0,0,0,0,7h5a3.5,3.5,0,0,1,0,7H6" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "shield": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M12,22s8-4,8-10V5L12,2,4,5v7C4,18,12,22,12,22Z" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
        <polyline points="9,12 11,14 15,10" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "scan": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <circle cx="11" cy="11" r="8" fill="none" stroke="{color}" stroke-width="2"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="11" y1="8" x2="11" y2="14" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="8" y1="11" x2="14" y2="11" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
    </svg>""",

    "bolt": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <polygon points="13,2 3,14 12,14 11,22 21,10 12,10 13,2" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
    </svg>""",

    "clock": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="10" fill="none" stroke="{color}" stroke-width="2"/>
        <polyline points="12,6 12,12 16,14" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "settings": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="3" fill="none" stroke="{color}" stroke-width="2"/>
        <path d="M19.4,15a1.65,1.65,0,0,0,.33,1.82l.06.06a2,2,0,1,1-2.83,2.83l-.06-.06a1.65,1.65,0,0,0-1.82-.33,1.65,1.65,0,0,0-1,1.51V21a2,2,0,0,1-4,0v-.09A1.65,1.65,0,0,0,9,19.4a1.65,1.65,0,0,0-1.82.33l-.06.06a2,2,0,1,1-2.83-2.83l.06-.06A1.65,1.65,0,0,0,4.68,15a1.65,1.65,0,0,0-1.51-1H3a2,2,0,0,1,0-4h.09A1.65,1.65,0,0,0,4.6,9a1.65,1.65,0,0,0-.33-1.82l-.06-.06A2,2,0,1,1,7.05,4.27l.06.06A1.65,1.65,0,0,0,9,4.68,1.65,1.65,0,0,0,10,3.17V3a2,2,0,0,1,4,0v.09a1.65,1.65,0,0,0,1,1.51,1.65,1.65,0,0,0,1.82-.33l.06-.06a2,2,0,0,1,2.83,2.83l-.06.06A1.65,1.65,0,0,0,19.4,9a1.65,1.65,0,0,0,1.51,1H21a2,2,0,0,1,0,4h-.09A1.65,1.65,0,0,0,19.4,15Z" fill="none" stroke="{color}" stroke-width="1.5"/>
    </svg>""",

    "log": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M14,2H6A2,2,0,0,0,4,4V20a2,2,0,0,0,2,2H18a2,2,0,0,0,2-4V8Z" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
        <polyline points="14,2 14,8 20,8" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
        <line x1="16" y1="13" x2="8" y2="13" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="16" y1="17" x2="8" y2="17" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
    </svg>""",

    "tax": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <rect x="2" y="5" width="20" height="14" rx="2" fill="none" stroke="{color}" stroke-width="2"/>
        <line x1="2" y1="10" x2="22" y2="10" stroke="{color}" stroke-width="2"/>
        <line x1="12" y1="10" x2="12" y2="19" stroke="{color}" stroke-width="1.5"/>
        <line x1="7" y1="14" x2="7" y2="16" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="17" y1="14" x2="17" y2="16" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
    </svg>""",

    "test": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M9,2v6l-4,8a2,2,0,0,0,2,2H17a2,2,0,0,0,2-2L15,8V2" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
        <line x1="9" y1="2" x2="15" y2="2" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <circle cx="10" cy="14" r="1" fill="{color}"/>
        <circle cx="14" cy="12" r="1" fill="{color}"/>
    </svg>""",

    "dashboard": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <rect x="3" y="3" width="7" height="7" rx="1" fill="none" stroke="{color}" stroke-width="2"/>
        <rect x="14" y="3" width="7" height="4" rx="1" fill="none" stroke="{color}" stroke-width="2"/>
        <rect x="14" y="10" width="7" height="11" rx="1" fill="none" stroke="{color}" stroke-width="2"/>
        <rect x="3" y="13" width="7" height="8" rx="1" fill="none" stroke="{color}" stroke-width="2"/>
    </svg>""",

    "arrow_up": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <polyline points="18,15 12,9 6,15" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "arrow_down": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <polyline points="6,9 12,15 18,9" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "wallet": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M20,7H4A2,2,0,0,1,4,3H18" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
        <rect x="2" y="7" width="20" height="14" rx="2" fill="none" stroke="{color}" stroke-width="2"/>
        <circle cx="17" cy="14" r="1.5" fill="{color}"/>
    </svg>""",

    "brain": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M12,2A7,7,0,0,0,5.2,9.1,5,5,0,0,0,6,19h1V12a5,5,0,0,1,10,0v7h1a5,5,0,0,0,.8-9.9A7,7,0,0,0,12,2Z" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
        <line x1="12" y1="12" x2="12" y2="22" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
    </svg>""",

    "check": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <polyline points="20,6 9,17 4,12" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "x_mark": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <line x1="18" y1="6" x2="6" y2="18" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="6" y1="6" x2="18" y2="18" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
    </svg>""",

    "warning": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M10.29,3.86,1.82,18a2,2,0,0,0,1.71,3H20.47a2,2,0,0,0,1.71-3L13.71,3.86A2,2,0,0,0,10.29,3.86Z" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
        <line x1="12" y1="9" x2="12" y2="13" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <circle cx="12" cy="17" r="1" fill="{color}"/>
    </svg>""",

    "robot": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <rect x="3" y="8" width="18" height="12" rx="2" fill="none" stroke="{color}" stroke-width="2"/>
        <line x1="12" y1="4" x2="12" y2="8" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <circle cx="12" cy="3" r="1.5" fill="{color}"/>
        <circle cx="9" cy="14" r="1.5" fill="{color}"/>
        <circle cx="15" cy="14" r="1.5" fill="{color}"/>
        <line x1="9" y1="17" x2="15" y2="17" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
    </svg>""",

    "hand": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M18,10V6a2,2,0,0,0-4,0V4a2,2,0,0,0-4,0V6a2,2,0,0,0-4,0v7" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M6,13v-1a2,2,0,0,0-4,0v5a8,8,0,0,0,16,0V10a2,2,0,0,0-4,0" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "refresh": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <polyline points="23,4 23,10 17,10" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <polyline points="1,20 1,14 7,14" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M3.51,9a9,9,0,0,1,14.85-3.36L23,10M1,14l4.64,4.36A9,9,0,0,0,20.49,15" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "bell": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M18,8A6,6,0,0,0,6,8c0,7-3,9-3,9H21s-3-2-3-9" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M13.73,21a2,2,0,0,1-3.46,0" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""",

    "play": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <polygon points="5,3 19,12 5,21" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
    </svg>""",

    "pause": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <rect x="6" y="4" width="4" height="16" rx="1" fill="none" stroke="{color}" stroke-width="2"/>
        <rect x="14" y="4" width="4" height="16" rx="1" fill="none" stroke="{color}" stroke-width="2"/>
    </svg>""",
}

# Icon cache: (name, size, color) -> QPixmap
_cache = {}


class StockyIcons:
    """Render SVG icons as QPixmaps at any size and color."""

    @staticmethod
    def get(name, size=20, color="#94a3b8"):
        """
        Get an icon as a QPixmap, rendered at high resolution for crisp display.

        Renders at 2x the requested size internally then scales down with
        smooth filtering, producing crisp icons even on high-DPI screens.
        """
        key = (name, size, color)
        if key in _cache:
            return _cache[key]

        svg_template = _SVGS.get(name)
        if not svg_template:
            return QPixmap(size, size)

        svg_str = svg_template.format(color=color)
        svg_bytes = QByteArray(svg_str.encode("utf-8"))

        renderer = QSvgRenderer(svg_bytes)

        # Render at 2x for crisp high-DPI display
        render_size = size * 2
        pixmap = QPixmap(render_size, render_size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        renderer.render(painter)
        painter.end()

        # Scale down with smooth filtering
        pixmap = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        _cache[key] = pixmap
        return pixmap

    @staticmethod
    def get_icon(name, size=20, color="#94a3b8"):
        """Get an icon as a QIcon (for buttons, menus, etc.)."""
        return QIcon(StockyIcons.get(name, size, color))

    @staticmethod
    def available():
        """List all available icon names."""
        return list(_SVGS.keys())
