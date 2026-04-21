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

    def __init__(self, tickers, period, interval, risk_manager, auto_settings=False, buying_power=None):
        super().__init__()
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.risk_manager = risk_manager
        self.auto_settings = auto_settings
        self.buying_power = buying_power

    def run(self):
        # Use a thread-safe list for progress updates instead of cross-thread signals
        import threading
        self._progress_queue = []
        self._progress_lock = threading.Lock()

        from core.profiles import get_optimal_workers

        def cb(done, total, ticker, result):
            if result and not result.error:
                p = getattr(result, 'period_used', '')
                iv = getattr(result, 'interval_used', '')
                settings = f" [{p}/{iv}]" if self.auto_settings and p else ""
                detail = f"{result.action}{settings}"
            elif result and result.error:
                detail = f"-- (insufficient data)"
            else:
                detail = "..."
            with self._progress_lock:
                self._progress_queue.append((done, total, ticker, detail))

        results = scan_multiple(self.tickers, self.period, self.interval,
                                self.risk_manager, max_workers=get_optimal_workers(),
                                progress_callback=cb, auto_settings=self.auto_settings,
                                buying_power=self.buying_power)
        self.finished.emit(results)

    def poll_progress(self):
        """Called by main thread timer to drain the progress queue."""
        if not hasattr(self, '_progress_lock'):
            return []
        with self._progress_lock:
            items = list(self._progress_queue)
            self._progress_queue.clear()
        return items


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
    """Comprehensive deep analysis — generates full report with charts, all data, LLM reasoning."""
    progress_update = pyqtSignal(int, str, str)
    finished_signal = pyqtSignal(str, str)  # report_text, ticker

    def __init__(self, scan_result):
        super().__init__()
        self.r = scan_result
        import threading
        self._progress_queue = []
        self._progress_lock = threading.Lock()

    def _emit_progress(self, pct, status, detail=""):
        with self._progress_lock:
            self._progress_queue.append((pct, status, detail))

    def poll_progress(self):
        with self._progress_lock:
            items = list(self._progress_queue)
            self._progress_queue.clear()
        return items

    def run(self):
        import yfinance as yf
        r = self.r
        report = []

        # ── Header ──
        self._emit_progress(5, "Building report header...", r.ticker)
        report.append("=" * 70)
        report.append(f"  DEEP ANALYSIS REPORT: {r.ticker}")
        report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  Estimated analysis time: 1-3 minutes")
        report.append("=" * 70)

        # ── Company Info ──
        self._emit_progress(8, "Fetching company info...", r.ticker)
        try:
            info = yf.Ticker(r.ticker).info
            name = info.get("shortName", r.ticker)
            sector = info.get("sector", "")
            industry = info.get("industry", "")
            cap = info.get("marketCap", 0)
            cap_str = f"${cap/1e9:.1f}B" if cap > 1e9 else f"${cap/1e6:.0f}M"
            desc = info.get("longBusinessSummary", "")[:300]
            report.append(f"\n{name} ({r.ticker})")
            report.append(f"Sector: {sector} | Industry: {industry} | Market Cap: {cap_str}")
            if desc:
                report.append(f"\n{desc}...")
        except Exception:
            report.append(f"\n{r.ticker}")

        # ── Verdict ──
        self._emit_progress(12, "Computing verdict...")
        report.append(f"\n{'=' * 70}")
        report.append(f"VERDICT: {r.action}  |  Confidence: {r.confidence:.1%}  |  Score: {r.score:.3f}")
        report.append(f"Price: ${r.price:.2f}  |  ATR: ${r.atr:.2f} ({r.atr/r.price*100:.1f}%)")
        strength = "STRONG" if r.confidence > 0.7 else "MODERATE" if r.confidence > 0.4 else "WEAK"
        report.append(f"Signal Strength: {strength}  |  Direction: {'BULLISH' if r.probs[2] > r.probs[0] else 'BEARISH'}")

        # ── Probabilities ──
        self._emit_progress(15, "Analyzing signal probabilities...")
        report.append(f"\n{'-' * 70}")
        report.append("SIGNAL PROBABILITIES")
        for label, prob, ch in [("SELL", r.probs[0], "X"), ("HOLD", r.probs[1], "="), ("BUY", r.probs[2], "#")]:
            filled = ch * int(prob * 40)
            empty = "." * (40 - int(prob * 40))
            report.append(f"  {label:5s}  {prob:6.1%}  [{filled}{empty}]")

        # ── Risk Management ──
        self._emit_progress(20, "Calculating risk metrics...")
        report.append(f"\n{'-' * 70}")
        report.append("RISK MANAGEMENT")
        report.append(f"  Position Size:    {r.position_size} shares (${r.position_size * r.price:,.2f})")
        report.append(f"  Stop Loss:        ${r.stop_loss:.2f} ({(r.price-r.stop_loss)/r.price*100:.1f}% below)")
        report.append(f"  Take Profit:      ${r.take_profit:.2f} ({(r.take_profit-r.price)/r.price*100:.1f}% above)")
        if r.stop_loss < r.price and r.take_profit > r.price:
            rr = (r.take_profit - r.price) / (r.price - r.stop_loss)
            max_loss = r.position_size * (r.price - r.stop_loss)
            max_gain = r.position_size * (r.take_profit - r.price)
            report.append(f"  Risk/Reward:      1:{rr:.1f}")
            report.append(f"  Max Loss:         ${max_loss:,.2f}")
            report.append(f"  Max Gain:         ${max_gain:,.2f}")
            report.append(f"  Breakeven needs:  {1/rr*100:.0f}% win rate")

        # ── Price Action Analysis ──
        self._emit_progress(25, "Fetching price history...")
        try:
            period = getattr(r, 'period_used', '5d')
            interval = getattr(r, 'interval_used', '5m')
            data = yf.Ticker(r.ticker).history(period=period, interval=interval)
            if not data.empty and len(data) > 10:
                closes = data["Close"].values
                highs = data["High"].values
                lows = data["Low"].values
                volumes = data["Volume"].values

                report.append(f"\n{'-' * 70}")
                report.append(f"PRICE ACTION ({period} @ {interval}, {len(data)} bars)")
                report.append(f"  Open:     ${closes[0]:.2f}")
                report.append(f"  High:     ${max(highs):.2f}")
                report.append(f"  Low:      ${min(lows):.2f}")
                report.append(f"  Close:    ${closes[-1]:.2f}")
                change = (closes[-1] - closes[0]) / closes[0] * 100
                report.append(f"  Change:   {change:+.2f}%")
                report.append(f"  Avg Vol:  {np.mean(volumes):,.0f}")

                # Price distribution
                import numpy as np
                report.append(f"\n  Price Statistics:")
                report.append(f"    Mean:     ${np.mean(closes):.2f}")
                report.append(f"    Std Dev:  ${np.std(closes):.2f}")
                report.append(f"    Median:   ${np.median(closes):.2f}")

                # Momentum
                if len(closes) >= 20:
                    sma20 = np.mean(closes[-20:])
                    report.append(f"    SMA-20:   ${sma20:.2f} ({'above' if closes[-1] > sma20 else 'below'} current)")
                if len(closes) >= 50:
                    sma50 = np.mean(closes[-50:])
                    report.append(f"    SMA-50:   ${sma50:.2f} ({'above' if closes[-1] > sma50 else 'below'} current)")
        except Exception:
            pass

        # ── Feature Importance ──
        self._emit_progress(35, "Analyzing feature importances...")
        if r.feature_importances:
            report.append(f"\n{'-' * 70}")
            report.append("FEATURE IMPORTANCE (what drove this signal)")
            max_imp = max(r.feature_importances.values())
            for feat, imp in sorted(r.feature_importances.items(), key=lambda x: -x[1])[:12]:
                bar_len = int((imp / max_imp) * 35)
                pct = imp / sum(r.feature_importances.values()) * 100
                report.append(f"  {'#' * bar_len:<35s}  {feat}: {imp:.0f} ({pct:.1f}%)")

        # ── Addon Signals ──
        self._emit_progress(45, "Gathering addon signals...")
        try:
            from addons import get_all_addons
            active = [a for a in get_all_addons() if a.available and a.enabled]
            if active:
                report.append(f"\n{'-' * 70}")
                report.append(f"ADDON SIGNALS ({len(active)} active)")
                for addon in active:
                    try:
                        feats = addon.get_features(r.ticker)
                        if feats and isinstance(feats, dict):
                            report.append(f"\n  {addon.name}:")
                            for k, v in list(feats.items())[:5]:
                                if isinstance(v, float):
                                    report.append(f"    {k}: {v:.4f}")
                                else:
                                    report.append(f"    {k}: {v}")
                    except Exception:
                        pass
        except Exception:
            pass

        # ── LLM Deep Analysis ──
        self._emit_progress(60, "Generating AI deep analysis (this may take a minute)...")
        try:
            from core.llm_reasoner import generate_deep_analysis
            llm_text = generate_deep_analysis(
                r.ticker, r.action, r.confidence, r.price, r.atr, r.probs,
                feature_importances=r.feature_importances,
                position_size=r.position_size,
                stop_loss=r.stop_loss, take_profit=r.take_profit,
            )
            report.append(f"\n{'-' * 70}")
            report.append("AI DEEP ANALYSIS")
            report.append("")
            # Wrap text at 65 chars
            words = llm_text.split()
            line = "  "
            for w in words:
                if len(line) + len(w) > 65:
                    report.append(line)
                    line = "  "
                line += w + " "
            if line.strip():
                report.append(line)
        except Exception as e:
            report.append(f"\n  (AI analysis unavailable: {e})")

        # ── Scan Settings ──
        self._emit_progress(80, "Compiling scan metadata...")
        period_used = getattr(r, 'period_used', '5d')
        interval_used = getattr(r, 'interval_used', '5m')
        report.append(f"\n{'-' * 70}")
        report.append("SCAN SETTINGS")
        report.append(f"  Training Data:  {period_used}")
        report.append(f"  Bar Interval:   {interval_used}")
        vol_label = "HIGH" if r.atr/r.price > 0.02 else "MODERATE" if r.atr/r.price > 0.01 else "LOW"
        report.append(f"  Volatility:     {vol_label} ({r.atr/r.price*100:.1f}% ATR)")
        trade_type = "Intraday" if r.atr/r.price < 0.03 else "Swing"
        report.append(f"  Trade Type:     {trade_type}")

        # ── Final Scores ──
        self._emit_progress(90, "Computing final scores...")
        report.append(f"\n{'=' * 70}")
        report.append("FINAL ASSESSMENT")
        report.append(f"  Overall Score:    {r.score:.3f} / 1.000")
        report.append(f"  Confidence:       {r.confidence:.1%}")
        report.append(f"  Signal:           {r.action} ({strength})")
        if r.action == "BUY" and r.confidence > 0.6:
            rec = "ENTER position with stop-loss protection"
        elif r.action == "SELL" and r.confidence > 0.6:
            rec = "EXIT or SHORT with defined risk"
        else:
            rec = "HOLD — wait for stronger signal"
        report.append(f"  Recommendation:   {rec}")
        report.append(f"\n{'=' * 70}")
        report.append(f"Report generated in background. All data is point-in-time.")

        self._emit_progress(100, "Report complete!", r.ticker)
        import time; time.sleep(0.3)
        self.finished_signal.emit("\n".join(report), r.ticker)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

