"""Panels package — each panel is a separate module."""
from .workers import ScanWorker
from .workers import TrainWorker
from .workers import DownloadWorker
from .workers import _DeepAnalyzeWorker
from .dashboard import DashboardPanel
from .scanner import ScannerPanel
from .day_trade import DayTradePanel
from .long_trade import LongTradePanel
from .logs import LogsPanel
from .portfolio import PortfolioPanel
from .settings_panel import SettingsPanel
from .tax import TaxPanel
from .testing import TestingPanel
from .notification_bar import _NotificationBar
