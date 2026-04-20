"""
System Tray Agent — keeps the app running in the background when window is closed.

When the user clicks X on the main window, the app minimizes to the system tray
instead of quitting. The auto-trader service continues running in the background.

Features:
- Tray icon with context menu (Show, Quit)
- Tray icon tooltip shows active monitoring count
- Windows toast notifications on trade events
- Double-click tray icon to restore window

Notification sync note:
    Toast notifications use winotify (Windows 10/11 native toasts).
    These are the same notifications that appear in Windows Action Center.
    Future: could add sound, grouping, and action buttons.
"""

import os
from PyQt5.QtWidgets import QSystemTrayIcon, QMenu, QAction, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

ICON_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "icon.ico")


class TrayAgent:
    """
    System tray integration for StockySuite.

    Usage:
        tray = TrayAgent(main_window)
        tray.setup()
        # Now closing the window minimizes to tray instead of quitting
    """

    def __init__(self, main_window):
        self.window = main_window
        self.tray = None
        self._notifications_enabled = True

    def setup(self):
        """Initialize the system tray icon and menu."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return False

        self.tray = QSystemTrayIcon(self.window)

        # Set icon
        if os.path.exists(ICON_FILE):
            self.tray.setIcon(QIcon(ICON_FILE))
        else:
            self.tray.setIcon(self.window.windowIcon())

        self.tray.setToolTip("Stocky Suite — AI Trading")

        # Context menu
        menu = QMenu()

        show_action = QAction("Show Stocky Suite", self.window)
        show_action.triggered.connect(self._show_window)
        menu.addAction(show_action)

        menu.addSeparator()

        # Show monitoring status
        self._status_action = QAction("No stocks monitored", self.window)
        self._status_action.setEnabled(False)
        menu.addAction(self._status_action)

        menu.addSeparator()

        quit_action = QAction("Quit", self.window)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_activated)
        self.tray.show()

        return True

    def update_status(self, monitored_count):
        """Update the tray tooltip and menu with monitoring status."""
        if not self.tray:
            return

        if monitored_count > 0:
            self.tray.setToolTip(f"Stocky Suite — Monitoring {monitored_count} stocks")
            self._status_action.setText(f"Monitoring {monitored_count} stocks")
        else:
            self.tray.setToolTip("Stocky Suite — Idle")
            self._status_action.setText("No stocks monitored")

    def send_notification(self, title, message, level="info"):
        """
        Send a Windows toast notification.
        Also shows as a tray balloon if winotify isn't available.

        Levels: info, trade, warn, error
        """
        if not self._notifications_enabled:
            return

        # Try winotify first (native Windows 10/11 toast)
        try:
            from winotify import Notification, audio
            toast = Notification(
                app_id="Stocky Suite",
                title=title,
                msg=message,
                duration="short",
            )
            if os.path.exists(ICON_FILE.replace(".ico", ".png")):
                toast.set_audio(audio.Default, loop=False)
            toast.show()
            return
        except ImportError:
            pass

        # Fallback: Qt tray balloon
        if self.tray:
            icon_map = {
                "info": QSystemTrayIcon.Information,
                "trade": QSystemTrayIcon.Information,
                "warn": QSystemTrayIcon.Warning,
                "error": QSystemTrayIcon.Critical,
            }
            self.tray.showMessage(title, message, icon_map.get(level, QSystemTrayIcon.Information), 5000)

    def _show_window(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        self.window.setWindowState(self.window.windowState() & ~Qt.WindowMinimized)

    def _quit(self):
        """Actually quit the application."""
        self.tray.hide()
        QApplication.quit()

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_window()
