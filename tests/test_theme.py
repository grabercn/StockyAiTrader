"""Tests for core.ui.theme — theme-aware color provider."""

import sys
import pytest

try:
    from PyQt5.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False


class TestThemeProvider:
    def test_brand_colors_same_in_both(self):
        from core.ui.theme import ThemeProvider, _BRAND
        tp = ThemeProvider()
        for key in _BRAND:
            assert tp.color(key) == _BRAND[key]

    def test_dark_and_light_differ(self):
        from core.ui.theme import _DARK, _LIGHT
        # bg_base should be different between themes
        assert _DARK["bg_base"] != _LIGHT["bg_base"]
        assert _DARK["text_primary"] != _LIGHT["text_primary"]

    def test_missing_color_returns_magenta(self):
        from core.ui.theme import ThemeProvider
        tp = ThemeProvider()
        tp._mode = "dark"
        assert tp.color("nonexistent_color_xyz") == "#ff00ff"

    def test_refresh_resets(self):
        from core.ui.theme import ThemeProvider
        tp = ThemeProvider()
        tp._mode = "dark"
        assert tp.is_dark
        tp.refresh()
        assert tp._mode is None  # Cleared, will re-detect

    @pytest.mark.skipif(not QT_AVAILABLE, reason="Qt not available")
    def test_qcolor_returns_qcolor(self):
        from core.ui.theme import theme
        from PyQt5.QtGui import QColor
        c = theme.qcolor("primary")
        assert isinstance(c, QColor)


class TestSetupWizard:
    def test_needs_setup_flag(self):
        from core.ui.setup_wizard import needs_setup
        # Just verify it returns a bool
        result = needs_setup()
        assert isinstance(result, bool)
