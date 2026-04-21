"""
System Tray Agent — keeps the app running in the background.

Features:
- Tray icon with context menu showing live stock monitoring status
- Each monitored stock shows: ticker, signal, confidence, next check
- Animated tooltip cycles through monitored stocks
- Windows toast notifications on trade events
- Quit animation: particles explode from tray icon position
"""

import os
from PyQt5.QtWidgets import QSystemTrayIcon, QMenu, QAction, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer

ICON_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "icon.ico")


class TrayAgent:
    def __init__(self, main_window):
        self.window = main_window
        self.tray = None
        self._notifications_enabled = True
        self._monitored = {}
        self._tooltip_cycle = 0

    def setup(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return False

        self.tray = QSystemTrayIcon(self.window)
        if os.path.exists(ICON_FILE):
            self.tray.setIcon(QIcon(ICON_FILE))
        else:
            self.tray.setIcon(self.window.windowIcon())

        self.tray.setToolTip("Stocky Suite — AI Trading")
        self._menu = QMenu()
        self._rebuild_menu()
        self.tray.setContextMenu(self._menu)
        self.tray.activated.connect(self._on_activated)
        self.tray.show()

        # Tooltip animation timer
        self._tooltip_timer = QTimer()
        self._tooltip_timer.timeout.connect(self._animate_tooltip)
        self._tooltip_timer.start(3000)

        return True

    def _rebuild_menu(self):
        self._menu.clear()

        show_action = QAction("Show Stocky Suite", self.window)
        show_action.triggered.connect(self._show_window)
        self._menu.addAction(show_action)
        self._menu.addSeparator()

        # Monitoring section
        count = len(self._monitored)
        if count > 0:
            header = QAction(f"Monitoring {count} stocks", self.window)
            header.setEnabled(False)
            self._menu.addAction(header)
            self._menu.addSeparator()

            for ticker, info in self._monitored.items():
                signal = info.get("signal", "HOLD")
                conf = info.get("confidence", 0)
                price = info.get("price", 0)
                next_secs = info.get("next_secs", 0)
                interval = info.get("interval", "5m")
                mode = info.get("mode", "manual")

                mins = next_secs // 60
                secs = next_secs % 60
                icon = {"BUY": "^", "SELL": "v", "HOLD": "-"}.get(signal, "?")
                mode_tag = "auto" if mode == "intelligent" else "manual"

                label = f"  {icon} {ticker}  {signal} ({conf:.0%})  ${price:.2f}"
                if next_secs > 0:
                    label += f"  | next: {mins}m{secs}s"
                label += f"  [{mode_tag}, {interval}]"

                action = QAction(label, self.window)
                action.setEnabled(False)
                self._menu.addAction(action)
        else:
            idle = QAction("No stocks monitored", self.window)
            idle.setEnabled(False)
            self._menu.addAction(idle)

        self._menu.addSeparator()

        quit_action = QAction("Quit", self.window)
        quit_action.triggered.connect(self._quit_with_animation)
        self._menu.addAction(quit_action)

    def update_stock(self, ticker, signal, confidence, price, next_secs,
                     interval="5m", mode="manual"):
        self._monitored[ticker] = {
            "signal": signal, "confidence": confidence,
            "price": price, "next_secs": next_secs,
            "interval": interval, "mode": mode,
        }
        self._rebuild_menu()

    def remove_stock(self, ticker):
        self._monitored.pop(ticker, None)
        self._rebuild_menu()

    def update_status(self, monitored_count):
        if not self.tray:
            return
        if monitored_count > 0:
            pass
        else:
            self.tray.setToolTip("Stocky Suite — Idle")

    def _animate_tooltip(self):
        if not self.tray or not self._monitored:
            if self.tray:
                self.tray.setToolTip("Stocky Suite — Idle")
            return

        tickers = list(self._monitored.keys())
        self._tooltip_cycle = (self._tooltip_cycle + 1) % len(tickers)
        ticker = tickers[self._tooltip_cycle]
        info = self._monitored[ticker]

        signal = info.get("signal", "?")
        price = info.get("price", 0)
        next_secs = info.get("next_secs", 0)
        conf = info.get("confidence", 0)
        interval = info.get("interval", "5m")
        mode = info.get("mode", "manual")
        mins = next_secs // 60
        secs = next_secs % 60
        dots = "." * ((self._tooltip_cycle % 3) + 1)

        self.tray.setToolTip(
            f"Stocky Suite — Monitoring {len(tickers)} stocks{dots}\n"
            f"{ticker}: {signal} ({conf:.0%}) @ ${price:.2f}\n"
            f"Next: {mins}m {secs}s | {interval} | {mode}"
        )

    def send_notification(self, title, message, level="info"):
        if not self._notifications_enabled:
            return
        try:
            from winotify import Notification, audio
            toast = Notification(app_id="Stocky Suite", title=title,
                                msg=message, duration="short")
            toast.show()
            return
        except ImportError:
            pass

        if self.tray:
            icon_map = {
                "info": QSystemTrayIcon.Information,
                "trade": QSystemTrayIcon.Information,
                "warn": QSystemTrayIcon.Warning,
                "error": QSystemTrayIcon.Critical,
            }
            self.tray.showMessage(title, message,
                                 icon_map.get(level, QSystemTrayIcon.Information), 5000)

    def _show_window(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        self.window.setWindowState(self.window.windowState() & ~Qt.WindowMinimized)

    def _quit_with_animation(self):
        """Quit with particle explosion from tray icon area."""
        try:
            from core.ui.tray_explode import TrayExplode
            # Get tray icon screen position (bottom-right of screen typically)
            geo = self.tray.geometry()
            if geo.isValid():
                cx, cy = geo.center().x(), geo.center().y()
            else:
                # Fallback: bottom-right corner
                screen = QApplication.primaryScreen().availableGeometry()
                cx = screen.right() - 30
                cy = screen.bottom() - 20

            explode = TrayExplode(cx, cy, on_done=self._final_quit)
            explode.start()
            QApplication.instance()._quit_explode = explode
        except Exception:
            self._final_quit()

    def _final_quit(self):
        if self.tray:
            self.tray.hide()
        QApplication.quit()

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_window()
