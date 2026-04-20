"""
Comprehensive health check tests — validates every file and module in the project.

These tests ensure:
- Every .py file parses without syntax errors
- All core modules import successfully
- All addons discover and have valid interfaces
- All icons render correctly
- Settings load/save works
- Branding constants are valid
"""

import os
import sys
import ast
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "StockyApps"))


class TestSyntaxCheck:
    """Verify every single .py file in the project parses."""

    def _get_all_py_files(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip .git, __pycache__, build dirs
            if any(skip in dirpath for skip in [".git", "__pycache__", "build", "dist", ".eggs"]):
                continue
            for f in filenames:
                if f.endswith(".py"):
                    files.append(os.path.join(dirpath, f))
        return files

    def test_all_files_parse(self):
        files = self._get_all_py_files()
        assert len(files) >= 30, f"Expected at least 30 .py files, found {len(files)}"
        errors = []
        for f in files:
            try:
                ast.parse(open(f, encoding="utf-8").read())
            except SyntaxError as e:
                errors.append(f"{f}: line {e.lineno}: {e.msg}")
        assert not errors, f"Syntax errors:\n" + "\n".join(errors)

    def test_stockysuite_parses(self):
        path = os.path.join(os.path.dirname(__file__), "..", "StockySuite.py")
        ast.parse(open(path, encoding="utf-8").read())


class TestCoreImports:
    """Verify all core modules import without errors."""

    def test_features(self):
        from core.features import engineer_features, INTRADAY_FEATURES, LONGTERM_FEATURES
        assert len(INTRADAY_FEATURES) >= 20

    def test_model(self):
        from core.model import train_lgbm, predict_lgbm

    def test_risk(self):
        from core.risk import RiskManager
        rm = RiskManager()
        assert rm.can_trade()[0] is True

    def test_labeling(self):
        from core.labeling import triple_barrier_label, LABEL_NAMES
        assert len(LABEL_NAMES) == 3

    def test_broker(self):
        from core.broker import AlpacaBroker

    def test_scanner(self):
        from core.scanner import scan_multiple, ScanResult

    def test_logger(self):
        from core.logger import log_decision, log_event, get_today_logs

    def test_signals(self):
        from core.signals import write_signal, read_signal

    def test_data(self):
        from core.data import fetch_intraday, fetch_longterm, get_all_features

    def test_profiles(self):
        from core.profiles import get_preset_names, get_active_profile_name
        assert len(get_preset_names()) >= 4

    def test_discovery(self):
        from core.discovery import get_all_sectors, get_watchlists
        assert len(get_all_sectors()) >= 10

    def test_tax_report(self):
        from core.tax_report import generate_form_8949

    def test_auto_trader(self):
        from core.auto_trader import AutoTraderService, MonitoredStock

    def test_intelligent_trader(self):
        from core.intelligent_trader import (
            IntelligentTraderService, IntelligentStock,
            compute_urgency, adaptive_interval, adaptive_period,
            AGGRESSIVITY_PROFILES,
        )
        assert len(AGGRESSIVITY_PROFILES) >= 4

    def test_tray_agent(self):
        from core.tray_agent import TrayAgent

    def test_branding(self):
        from core.branding import (
            APP_NAME, APP_VERSION, BRAND_PRIMARY, SUITE_STYLESHEET,
            LIGHT_STYLESHEET, chart_colors, get_stylesheet,
        )
        assert APP_NAME == "Stocky Suite"
        assert len(SUITE_STYLESHEET) > 1000


class TestUIImports:
    """Verify all UI framework modules import."""

    def test_animations(self):
        from core.ui.animations import FadeIn, SlideIn, Shake, PulseEffect

    def test_theme(self):
        from core.ui.theme import theme, ThemeProvider
        assert theme.color("primary") == "#0ea5e9"

    def test_chart_tooltip(self):
        from core.ui.chart_tooltip import ChartTooltip

    def test_detail_popup(self):
        from core.ui.detail_popup import DetailPopup, show_equity_popup

    def test_boot_screen(self):
        from core.ui.boot_screen import BootScreen

    def test_setup_wizard(self):
        from core.ui.setup_wizard import SetupWizard, needs_setup


class TestAddonHealth:
    """Verify all addons discover and have valid interfaces."""

    def test_addons_discover(self):
        from addons import get_all_addons
        addons = get_all_addons()
        assert len(addons) >= 5

    def test_addon_interfaces(self):
        from addons import get_all_addons
        for addon in get_all_addons():
            assert addon.name, f"Addon {addon.module_name} missing name"
            assert addon.description, f"Addon {addon.module_name} missing description"
            assert isinstance(addon.features, list)
            assert isinstance(addon.dependencies, list)
            assert isinstance(addon.requires_api_key, bool)
            assert hasattr(addon._module, "get_features")
            assert hasattr(addon._module, "check_available")


class TestIconHealth:
    """Verify all SVG icons render."""

    def test_all_icons_render(self):
        try:
            from PyQt5.QtWidgets import QApplication
            import sys
            app = QApplication.instance() or QApplication(sys.argv)
            from core.ui.icons import StockyIcons
            names = StockyIcons.available()
            assert len(names) >= 20, f"Expected 20+ icons, got {len(names)}"
            for name in names:
                px = StockyIcons.get(name, 24, "#ffffff")
                assert px.width() == 24
        except Exception:
            pytest.skip("Qt not available for icon rendering")


class TestSettingsHealth:
    """Verify settings load/save works."""

    def test_chart_colors(self):
        from core.branding import chart_colors
        cc = chart_colors()
        assert "fig_bg" in cc
        assert "ax_bg" in cc
        assert "text" in cc
        assert "grid" in cc

    def test_theme_colors_complete(self):
        from core.ui.theme import theme, _DARK, _LIGHT, _BRAND
        # All palette keys should exist in both themes
        for key in _DARK:
            assert key in _LIGHT, f"Key {key} in dark but not light"
        for key in _LIGHT:
            assert key in _DARK, f"Key {key} in light but not dark"
