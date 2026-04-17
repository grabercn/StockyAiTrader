"""Tests for core.ui — custom widget and visual framework."""

import sys
import pytest

try:
    from PyQt5.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt not available")
class TestIcons:
    def test_all_icons_render(self):
        from core.ui.icons import StockyIcons
        for name in StockyIcons.available():
            px = StockyIcons.get(name, 24, "#ffffff")
            assert px.width() == 24, f"Icon {name} wrong width"
            assert px.height() == 24, f"Icon {name} wrong height"

    def test_icon_caching(self):
        from core.ui.icons import StockyIcons
        px1 = StockyIcons.get("chart_up", 20, "#ff0000")
        px2 = StockyIcons.get("chart_up", 20, "#ff0000")
        # Same object from cache
        assert px1 is px2

    def test_icon_color_tinting(self):
        from core.ui.icons import StockyIcons
        px_red = StockyIcons.get("check", 20, "#ff0000")
        px_blue = StockyIcons.get("check", 20, "#0000ff")
        # Different colors = different pixmaps
        assert px_red is not px_blue

    def test_unknown_icon_returns_empty(self):
        from core.ui.icons import StockyIcons
        px = StockyIcons.get("nonexistent_icon_xyz", 16)
        assert px.width() == 16

    def test_available_lists_all(self):
        from core.ui.icons import StockyIcons
        names = StockyIcons.available()
        assert len(names) >= 15
        assert "chart_up" in names
        assert "dashboard" in names
        assert "settings" in names

    def test_get_qicon(self):
        from core.ui.icons import StockyIcons
        from PyQt5.QtGui import QIcon
        icon = StockyIcons.get_icon("dollar", 20)
        assert isinstance(icon, QIcon)


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt not available")
class TestCustomWidgets:
    def test_stat_card_creation(self):
        from core.widgets import StatCard
        card = StatCard("Test", "$100", "#0ea5e9")
        card.set_value("$200", "#10b981")
        card.set_title("Updated")

    def test_signal_badge(self):
        from core.widgets import SignalBadge
        badge = SignalBadge()
        badge.set_signal("BUY", 0.85)
        badge.set_signal("SELL", 0.60)
        badge.set_signal("HOLD", 0.50)

    def test_gradient_divider(self):
        from core.widgets import GradientDivider
        div = GradientDivider()
        assert div.height() == 1

    def test_sparkline(self):
        from core.widgets import Sparkline
        spark = Sparkline([100, 102, 101, 105])
        spark.set_data([50, 55, 53, 60], "#ff0000")


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt not available")
class TestCharts:
    def test_candlestick(self):
        from core.ui.charts import CandlestickChart
        chart = CandlestickChart()
        chart.set_data([(100, 105, 98, 103), (103, 107, 101, 106)])

    def test_gauge(self):
        from core.ui.charts import GaugeWidget
        gauge = GaugeWidget(label="RSI")
        gauge.set_value(75)
        gauge.set_value(0)
        gauge.set_value(100)

    def test_area_sparkline(self):
        from core.ui.charts import AreaSparkline
        spark = AreaSparkline()
        spark.set_data([100, 105, 103, 108, 110])


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt not available")
class TestBackgrounds:
    def test_glass_panel(self):
        from core.ui.backgrounds import GlassPanel
        panel = GlassPanel("#0ea5e9")
        assert panel.content_layout is not None

    def test_gradient_header(self):
        from core.ui.backgrounds import GradientHeader
        header = GradientHeader("Test Title", "Subtitle")
        header.set_title("New Title")

    def test_pattern_panel(self):
        from core.ui.backgrounds import PatternPanel
        panel = PatternPanel(spacing=15)


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt not available")
class TestTables:
    def test_premium_table(self):
        from core.ui.tables import PremiumTable
        table = PremiumTable(["Ticker", "Signal", "Price"])
        table.add_signal_row(["AAPL", "BUY", "$150"], signal_col=1)
        assert table.rowCount() == 1

    def test_confidence_bar(self):
        from core.ui.tables import ConfidenceBar
        bar = ConfidenceBar(0.75)
        bar.set_value(0.5)
        bar.set_value(0.0)
        bar.set_value(1.0)


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt not available")
class TestAnimations:
    def test_imports(self):
        from core.ui.animations import FadeIn, FadeOut, SlideIn, Shake, ScaleUp, PulseEffect, StaggeredFadeIn
        # Just verify they're callable
        assert callable(FadeIn)
        assert callable(SlideIn)
        assert callable(Shake)
