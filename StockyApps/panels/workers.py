"""Panel module — auto-extracted from StockySuite.py"""
import sys, os, json, time
import numpy as np
from datetime import datetime, timedelta
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QLinearGradient, QPen
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pytz, yfinance as yf
from core.branding import *
from core.branding import get_stylesheet, detect_system_theme, chart_colors
from core.event_bus import EventBus
from core.risk import RiskManager
from core.broker import AlpacaBroker
from core.scanner import scan_multiple, ScanResult
from core.data import fetch_intraday, fetch_longterm, get_all_features
from core.model import train_lgbm, predict_lgbm
from core.labeling import LABEL_NAMES
from core.logger import log_decision, log_trade_execution, log_scan_results, log_event, get_today_logs, get_log_files, get_log_entries
from core.signals import write_signal
from core.model_manager import MANAGED_MODELS, get_model_status, get_lgbm_models, download_model, delete_model, delete_lgbm_model, delete_all_lgbm_models
from addons import get_all_addons, set_addon_enabled, discover_addons

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "settings.json")
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f: return json.load(f)
    except: return {}
def save_settings(s):
    with open(SETTINGS_FILE, "w") as f: json.dump(s, f, indent=4)

# WORKER THREADS
# ═════════════════════════════════════════════════════════════════════════════

class ScanWorker(QThread):
    progress = pyqtSignal(int, int, str, str)  # done, total, ticker, detail_str
    finished = pyqtSignal(list)

    def __init__(self, tickers, period, interval, risk_manager, auto_settings=False):
        super().__init__()
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.risk_manager = risk_manager
        self.auto_settings = auto_settings

    def run(self):
        def cb(done, total, ticker, result):
            if result and not result.error:
                p = getattr(result, 'period_used', '')
                iv = getattr(result, 'interval_used', '')
                settings = f" [{p}/{iv}]" if self.auto_settings and p else ""
                detail = f"{result.action}{settings}"
            else:
                detail = result.action if result else "..."
            self.progress.emit(done, total, ticker, detail)
        from core.profiles import get_optimal_workers
        results = scan_multiple(self.tickers, self.period, self.interval,
                                self.risk_manager, max_workers=get_optimal_workers(),
                                progress_callback=cb, auto_settings=self.auto_settings)
        self.finished.emit(results)


class TrainWorker(QThread):
    finished = pyqtSignal(object, list, object)

    def __init__(self, data, features, ticker, prefix="lgbm"):
        super().__init__()
        self.data = data
        self.features = features
        self.ticker = ticker
        self.prefix = prefix

    def run(self):
        model, feats = train_lgbm(self.data, self.features, self.ticker, prefix=self.prefix)
        self.finished.emit(model, feats, self.data)


class DownloadWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_info):
        super().__init__()
        self.model_info = model_info

    def run(self):
        download_model(self.model_info, progress_callback=self.progress.emit)
        self.finished.emit()


class _DeepAnalyzeWorker(QThread):
    """Runs deep analysis in background so UI doesn't freeze."""
    progress_update = pyqtSignal(int, str, str)  # pct, status, detail
    finished_signal = pyqtSignal(str, str)        # report_text, ticker

    def __init__(self, scan_result):
        super().__init__()
        self.r = scan_result

    def run(self):
        r = self.r
        report = []

        self.progress_update.emit(10, "Building report header...", r.ticker)
        report.append("=" * 60)
        report.append(f"  DEEP ANALYSIS REPORT: {r.ticker}")
        report.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        report.append(f"VERDICT: {r.action}  |  Confidence: {r.confidence:.1%}  |  Score: {r.score:.3f}")
        report.append(f"Price: ${r.price:.2f}  |  ATR: ${r.atr:.2f} ({r.atr/r.price*100:.1f}%)")

        self.progress_update.emit(20, "Analyzing probabilities...", "")
        report.append("")
        report.append("-" * 60)
        report.append("SIGNAL PROBABILITIES")
        for label, prob, bar_char in [("SELL", r.probs[0], "X"), ("HOLD", r.probs[1], "="), ("BUY", r.probs[2], "#")]:
            bar = bar_char * int(prob * 40)
            report.append(f"  {label:5s}  {prob:6.1%}  [{bar:<40s}]")

        self.progress_update.emit(35, "Calculating risk metrics...", "")
        report.append("")
        report.append("-" * 60)
        report.append("RISK MANAGEMENT")
        report.append(f"  Position Size:    {r.position_size} shares (${r.position_size * r.price:,.2f})")
        report.append(f"  Stop Loss:        ${r.stop_loss:.2f} ({(r.price-r.stop_loss)/r.price*100:.1f}% risk)")
        report.append(f"  Take Profit:      ${r.take_profit:.2f} ({(r.take_profit-r.price)/r.price*100:.1f}% target)")
        if r.stop_loss < r.price:
            rr = (r.take_profit - r.price) / (r.price - r.stop_loss)
            report.append(f"  Risk/Reward:      1:{rr:.1f}")
            report.append(f"  Max Loss:         ${r.position_size * (r.price - r.stop_loss):,.2f}")
            report.append(f"  Max Gain:         ${r.position_size * (r.take_profit - r.price):,.2f}")

        self.progress_update.emit(50, "Analyzing feature importances...", "")
        if r.feature_importances:
            report.append("")
            report.append("-" * 60)
            report.append("TOP FEATURE DRIVERS")
            report.append("(What the AI model weighted most heavily)")
            report.append("")
            max_imp = max(r.feature_importances.values()) if r.feature_importances else 1
            for feat, imp in sorted(r.feature_importances.items(), key=lambda x: -x[1])[:10]:
                bar_len = int((imp / max_imp) * 30)
                report.append(f"  {'#' * bar_len:<30s}  {feat}: {imp:.0f}")

        self.progress_update.emit(65, "Generating LLM reasoning...", "")
        try:
            from core.llm_reasoner import generate_reasoning
            llm_text = generate_reasoning(
                r.ticker, r.action, r.confidence, r.price, r.atr, r.probs,
                feature_importances=r.feature_importances,
            )
            report.append("")
            report.append("-" * 60)
            report.append("AI REASONING")
            for part in llm_text.split(" | "):
                report.append(f"  > {part}")
        except Exception as e:
            report.append(f"\n  (LLM reasoning unavailable: {e})")

        self.progress_update.emit(80, "Computing final scores...", "")
        report.append("")
        report.append("-" * 60)
        report.append("FINAL SCORES")
        report.append(f"  Overall Score:     {r.score:.3f} / 1.000")
        report.append(f"  Confidence:        {r.confidence:.1%}")
        report.append(f"  Signal Strength:   {'STRONG' if r.confidence > 0.7 else 'MODERATE' if r.confidence > 0.4 else 'WEAK'}")
        report.append(f"  Volatility:        {'HIGH' if r.atr/r.price > 0.02 else 'MODERATE' if r.atr/r.price > 0.01 else 'LOW'}")
        report.append(f"  Direction:         {'BULLISH' if r.probs[2] > r.probs[0] else 'BEARISH'}")
        report.append("")
        report.append("=" * 60)

        self.progress_update.emit(95, "Report complete", r.ticker)
        import time; time.sleep(0.3)
        self.finished_signal.emit("\n".join(report), r.ticker)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

