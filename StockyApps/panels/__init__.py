"""Panels package — each panel is a separate module."""
from .workers import ScanWorker, TrainWorker, DownloadWorker, _DeepAnalyzeWorker
from .dashboard import DashboardPanel
from .scanner import ScannerPanel
from .trade import TradePanel
from .ai_dashboard import AIDashboardPanel
from .logs import LogsPanel
from .portfolio import PortfolioPanel
from .settings_panel import SettingsPanel
from .tax import TaxPanel
from .testing import TestingPanel
from .notification_bar import _NotificationBar
