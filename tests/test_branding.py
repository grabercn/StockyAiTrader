"""Tests for core.branding and core.profiles — UI consistency."""

from core.branding import (
    APP_NAME, APP_VERSION, APP_TAGLINE, SUITE_STYLESHEET,
    BRAND_PRIMARY, BRAND_ACCENT, COLOR_BUY, COLOR_SELL, COLOR_HOLD,
    BG_DARKEST, BG_DARK, BG_PANEL, BORDER, TEXT_PRIMARY,
    FONT_FAMILY, FONT_MONO, log_html,
)


class TestBrandingConstants:
    def test_app_identity(self):
        assert APP_NAME == "Stocky Suite"
        assert APP_VERSION  # Not empty
        assert APP_TAGLINE

    def test_colors_are_hex(self):
        for color in [BRAND_PRIMARY, BRAND_ACCENT, COLOR_BUY, COLOR_SELL,
                      COLOR_HOLD, BG_DARKEST, BG_DARK, BG_PANEL, BORDER]:
            assert color.startswith("#"), f"{color} is not a hex color"
            assert len(color) == 7, f"{color} should be #RRGGBB format"

    def test_stylesheet_not_empty(self):
        assert len(SUITE_STYLESHEET) > 500  # Should be substantial
        assert "QMainWindow" in SUITE_STYLESHEET
        assert "QPushButton" in SUITE_STYLESHEET
        assert BRAND_PRIMARY in SUITE_STYLESHEET

    def test_log_html_formatting(self):
        result = log_html("Test message", "info")
        assert "Test message" in result
        assert "color:" in result
        assert "<span" in result

    def test_log_html_levels(self):
        for level in ["info", "warn", "error", "trade", "system"]:
            result = log_html("msg", level)
            assert "msg" in result


class TestFonts:
    def test_font_families_set(self):
        assert FONT_FAMILY  # Not empty
        assert FONT_MONO
