"""
Stocky UI Framework — custom visual system for Stocky Suite.

A modular, extensible UI toolkit providing:
- Animations: fade, slide, pulse, ripple, shake, glow transitions
- Icons: programmatic SVG icon library (no external files needed)
- Backgrounds: gradient panels, particle effects, pattern fills
- Charts: candlestick, gauge, mini sparkline, area charts
- Tables: premium styled tables with hover, sorting indicators
- Widgets: stat cards, signal badges, progress bars, dividers

Usage:
    from core.ui import FadeIn, StockyIcon, GlassPanel, CandlestickChart
"""

from .animations import *
from .icons import StockyIcons
from .backgrounds import GlassPanel, ParticleBackground, GradientHeader
from .charts import CandlestickChart, GaugeWidget, AreaSparkline
from .tables import PremiumTable
