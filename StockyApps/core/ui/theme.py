"""
Theme-Aware Color Provider — single source of truth for current theme colors.

All custom widgets should call theme.color("bg_card") instead of hardcoding
"#1e2130". This ensures every component adapts to light/dark mode automatically.

Usage:
    from core.ui.theme import theme
    bg = theme.color("bg_card")
    text = theme.color("text_primary")
"""

import os
import json

_SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "settings.json")

# ─── Color palettes ──────────────────────────────────────────────────────────

_DARK = {
    "bg_base":       "#0f1117",
    "bg_panel":      "#1a1d28",
    "bg_card":       "#1e2130",
    "bg_input":      "#252836",
    "bg_hover":      "#2a2d42",
    "border":        "#2a2d3a",
    "border_accent": "#0ea5e933",
    "text_primary":  "#e2e8f0",
    "text_secondary":"#94a3b8",
    "text_muted":    "#64748b",
    "text_heading":  "#e2e8f0",
    "highlight":     "#ffffff0f",
    "table_alt":     "#222538",
    "shadow":        "#00000060",
}

_LIGHT = {
    "bg_base":       "#f8fafc",
    "bg_panel":      "#ffffff",
    "bg_card":       "#ffffff",
    "bg_input":      "#f1f5f9",
    "bg_hover":      "#e2e8f0",
    "border":        "#e2e8f0",
    "border_accent": "#0ea5e933",
    "text_primary":  "#1e293b",
    "text_secondary":"#475569",
    "text_muted":    "#94a3b8",
    "text_heading":  "#0f172a",
    "highlight":     "#0000000a",
    "table_alt":     "#f8fafc",
    "shadow":        "#00000015",
}

# Brand colors stay the same in both themes
_BRAND = {
    "primary":    "#0ea5e9",
    "secondary":  "#6366f1",
    "accent":     "#10b981",
    "buy":        "#10b981",
    "sell":       "#ef4444",
    "hold":       "#f59e0b",
    "profit":     "#10b981",
    "loss":       "#ef4444",
}


class ThemeProvider:
    """
    Provides the current theme's colors.
    Reads the theme setting from settings.json on first access,
    then caches it. Call refresh() after theme changes.
    """

    def __init__(self):
        self._mode = None  # "dark" or "light"
        self._cache = {}

    def _detect(self):
        """Detect current theme from settings or system."""
        try:
            with open(_SETTINGS_FILE, "r") as f:
                settings = json.load(f)
            theme = settings.get("theme", "auto")
        except (FileNotFoundError, json.JSONDecodeError):
            theme = "auto"

        if theme == "auto":
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
                )
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                winreg.CloseKey(key)
                self._mode = "light" if value == 1 else "dark"
            except Exception:
                self._mode = "dark"
        else:
            self._mode = theme

    @property
    def is_dark(self):
        if self._mode is None:
            self._detect()
        return self._mode == "dark"

    @property
    def is_light(self):
        return not self.is_dark

    def color(self, name):
        """Get a color by name. Returns hex string."""
        if self._mode is None:
            self._detect()

        # Check brand colors first (same in both themes)
        if name in _BRAND:
            return _BRAND[name]

        palette = _DARK if self.is_dark else _LIGHT
        return palette.get(name, "#ff00ff")  # Magenta = missing color (easy to spot)

    def qcolor(self, name):
        """Get a color as QColor object."""
        from PyQt5.QtGui import QColor
        return QColor(self.color(name))

    def refresh(self):
        """Re-detect theme. Call after user changes theme in settings."""
        self._mode = None
        self._cache.clear()


# Singleton — import and use directly
theme = ThemeProvider()
