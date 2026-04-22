# -*- coding: utf-8 -*-
"""
System Tray Agent — keeps the app running in the background.

Features:
- Tray icon with context menu
- Rich left-click overlay: account, positions, agent status, recent actions
- Animated tooltip cycles through monitored stocks
- Windows toast notifications on trade events
- Quit animation: particles explode from tray icon position
"""

import os
import json
import time as _time
from PyQt5.QtWidgets import (
    QSystemTrayIcon, QMenu, QAction, QApplication,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QScrollArea,
)
from PyQt5.QtGui import QIcon, QFont, QColor, QPainter, QPen, QBrush, QLinearGradient
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve

ICON_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "icon.ico")


class TrayAgent:
    def __init__(self, main_window):
        self.window = main_window
        self.tray = None
        self._notifications_enabled = True
        self._monitored = {}
        self._tooltip_cycle = 0
        self._popup = None
        self._icon_animator = None
        # Live state pushed from agent/auto-trader
        self._agent_state = {
            "running": False, "cycle": 0, "scanned": 0,
            "buys": 0, "sells": 0, "last_action": "",
        }
        self._recent_actions = []  # Last 5 trade actions

    def setup(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return False

        self.tray = QSystemTrayIcon(self.window)
        if os.path.exists(ICON_FILE):
            self.tray.setIcon(QIcon(ICON_FILE))
        else:
            self.tray.setIcon(self.window.windowIcon())

        self.tray.setToolTip("Stocky Suite - AI Trading")
        self._menu = QMenu()
        self._rebuild_menu()
        self.tray.setContextMenu(self._menu)  # Right-click menu
        self.tray.activated.connect(self._on_activated)
        self.tray.show()

        # Dynamic icon overlays
        from core.ui.tray_icons import TrayIconAnimator
        self._icon_animator = TrayIconAnimator(self.tray)

        self._tooltip_timer = QTimer()
        self._tooltip_timer.timeout.connect(self._animate_tooltip)
        self._tooltip_timer.start(3000)

        return True

    # ── Agent state updates (called from ai_dashboard / auto_trader) ──

    def update_agent_state(self, **kwargs):
        """Push live agent state. Keys: running, cycle, scanned, buys, sells, last_action."""
        self._agent_state.update(kwargs)

        # Update tray icon + tooltip based on state
        if self._icon_animator:
            if not self._agent_state.get("running"):
                self._icon_animator.set_state("idle")
                self._set_tooltip("Stocky Suite - Idle")
            elif kwargs.get("last_action") == "error":
                self._icon_animator.set_state("error")
                self._set_tooltip("Stocky Suite - Trade failed")
                QTimer.singleShot(5000, lambda: self._icon_animator.set_state("agent")
                                  if self._agent_state.get("running") else None)
            elif "scanned" in kwargs and kwargs["scanned"] > 0:
                self._icon_animator.set_state("scanning")
                self._set_tooltip(f"Stocky Suite - Scanning {kwargs['scanned']} stocks...")
            elif "buys" in kwargs or "sells" in kwargs:
                self._icon_animator.set_state("trading")
                action = self._recent_actions[-1] if self._recent_actions else "Trading"
                self._set_tooltip(f"Stocky Suite - {action}")
                QTimer.singleShot(3000, lambda: self._icon_animator.set_state("agent")
                                  if self._agent_state.get("running") else None)
            elif self._agent_state.get("running"):
                cycle = self._agent_state.get("cycle", 0)
                self._icon_animator.set_state("agent")
                self._set_tooltip(f"Stocky Suite - Agent running (cycle {cycle})")

    def _set_tooltip(self, text):
        """Update tray tooltip text."""
        if self.tray:
            self.tray.setToolTip(text)

    def add_action(self, text):
        """Add a recent trade action line (kept to last 8)."""
        self._recent_actions.append(text)
        if len(self._recent_actions) > 8:
            self._recent_actions = self._recent_actions[-8:]

    # ── Stock monitoring ──

    def update_stock(self, ticker, signal, confidence, price, next_secs,
                     interval="5m", mode="manual"):
        self._monitored[ticker] = {
            "signal": signal, "confidence": confidence,
            "price": price, "next_secs": next_secs,
            "interval": interval, "mode": mode,
        }
        now = _time.time()
        if not hasattr(self, '_last_rebuild') or now - self._last_rebuild > 10:
            self._last_rebuild = now
            self._rebuild_menu()

    def remove_stock(self, ticker):
        self._monitored.pop(ticker, None)
        self._rebuild_menu()

    def update_status(self, monitored_count):
        if not self.tray:
            return
        if monitored_count == 0:
            self.tray.setToolTip("Stocky Suite - Idle")

    # ── Context menu (right-click) ──

    def _rebuild_menu(self):
        self._menu.clear()

        open_action = QAction("Open", self.window)
        open_action.triggered.connect(self._show_window)
        self._menu.addAction(open_action)

        self._menu.addSeparator()

        quit_action = QAction("Quit", self.window)
        quit_action.triggered.connect(self._quit_with_animation)
        self._menu.addAction(quit_action)

    # ── Tooltip ──

    def _animate_tooltip(self):
        """Cycle tooltip through monitored stocks when agent isn't driving updates."""
        if not self.tray:
            return
        # If agent is running, tooltip is driven by update_agent_state
        if self._agent_state.get("running"):
            return
        if not self._monitored:
            self.tray.setToolTip("Stocky Suite - Idle")
            return

        tickers = list(self._monitored.keys())
        self._tooltip_cycle = (self._tooltip_cycle + 1) % len(tickers)
        ticker = tickers[self._tooltip_cycle]
        info = self._monitored[ticker]
        signal = info.get("signal", "?")
        conf = info.get("confidence", 0)
        price = info.get("price", 0)

        self.tray.setToolTip(
            f"Stocky Suite - Watching {len(tickers)} stocks\n"
            f"{ticker}: {signal} ({conf:.0%}) @ ${price:.2f}"
        )

    # ── Notifications ──

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

    # ── Window show/hide ──

    def _show_window(self):
        """Restore window with particle expand animation."""
        from core.ui.anim_config import animations_enabled
        win = self.window

        if not animations_enabled():
            win.show()
            win.raise_()
            win.activateWindow()
            win.setWindowState(win.windowState() & ~Qt.WindowMinimized)
            return

        try:
            from core.ui.window_expand import WindowExpand
            geo = win.geometry()

            effect = QGraphicsOpacityEffect(win)
            win.setGraphicsEffect(effect)
            effect.setOpacity(0.0)
            win.show()
            win.raise_()
            win.activateWindow()
            win.setWindowState(win.windowState() & ~Qt.WindowMinimized)

            def _on_expand_done():
                fade = QPropertyAnimation(effect, b"opacity")
                fade.setDuration(400)
                fade.setStartValue(0.0)
                fade.setEndValue(1.0)
                fade.setEasingCurve(QEasingCurve.OutCubic)
                fade.finished.connect(lambda: win.setGraphicsEffect(None))
                fade.start()
                win._restore_fade = fade

            expand = WindowExpand(geo, on_done=_on_expand_done)
            expand.start()
            win._expand_anim = expand
        except Exception:
            win.show()
            win.raise_()
            win.activateWindow()
            win.setWindowState(win.windowState() & ~Qt.WindowMinimized)

    def _quit_with_animation(self):
        from core.ui.anim_config import animations_enabled
        if not animations_enabled():
            self._final_quit()
            return
        try:
            from core.ui.tray_explode import TrayExplode
            geo = self.tray.geometry()
            if geo.isValid():
                cx, cy = geo.center().x(), geo.center().y()
            else:
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

    # ── Left-click: at-a-glance overlay ──

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            if self._popup and self._popup.isVisible():
                self._popup.close()
            self._show_window()
        elif reason == QSystemTrayIcon.Trigger:
            # Left-click: show custom overlay
            try:
                self._show_glance_popup()
            except Exception as e:
                print(f"[TRAY] Popup error: {e}", flush=True)
                import traceback; traceback.print_exc()

    def _show_glance_popup(self):
        """Custom overlay panel with account, positions, agent status, recent actions."""
        from core.branding import (
            BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT,
            FONT_FAMILY, FONT_MONO, COLOR_PROFIT, COLOR_LOSS,
            COLOR_BUY, COLOR_SELL, COLOR_HOLD,
            TEXT_MUTED, TEXT_SECONDARY, BG_PANEL, BG_INPUT,
        )

        # Toggle off if already open
        if self._popup and self._popup.isVisible():
            self._popup.close()
            self._popup = None
            return

        PANEL_W = 360
        NB = "border: none;"  # no-border shorthand

        popup = QWidget()
        popup.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.Popup)
        popup.setAttribute(Qt.WA_TranslucentBackground)
        popup.setFixedWidth(PANEL_W + 16)

        # Outer container with shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 5)

        container = QFrame()
        container.setGraphicsEffect(shadow)
        container.setStyleSheet(
            f"QFrame#trayPanel {{ background-color: {BG_PANEL}; "
            f"border: 1px solid {BRAND_PRIMARY}55; border-radius: 12px; }}"
        )
        container.setObjectName("trayPanel")

        lay = QVBoxLayout()
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(0)

        # ── Header with accent bar ──
        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        title = QLabel("Stocky Suite")
        title.setFont(QFont(FONT_FAMILY, 13, QFont.Bold))
        title.setStyleSheet(f"color: {BRAND_PRIMARY}; {NB}")
        hdr.addWidget(title)
        hdr.addStretch()

        # Agent status pill
        ag = self._agent_state
        if ag["running"]:
            pill_text = f"Agent: Cycle {ag['cycle']}"
            pill_bg = BRAND_ACCENT
        else:
            pill_text = "Agent: Off"
            pill_bg = TEXT_MUTED
        pill = QLabel(pill_text)
        pill.setFont(QFont(FONT_FAMILY, 8, QFont.Bold))
        pill.setStyleSheet(
            f"color: white; background: {pill_bg}; {NB} "
            f"border-radius: 8px; padding: 2px 8px;"
        )
        hdr.addWidget(pill)
        lay.addLayout(hdr)
        lay.addSpacing(8)

        # ── Account section ──
        acct_data, positions_data = {}, []
        try:
            sf = os.path.join(os.path.dirname(__file__), "..", "..", "settings.json")
            with open(sf, "r") as f:
                settings = json.load(f)
            key, secret = settings.get("alpaca_api_key", ""), settings.get("alpaca_secret_key", "")
            if key and secret:
                from core.broker import AlpacaBroker
                broker = AlpacaBroker(key, secret)
                acct_data = broker.get_account()
                positions_data = broker.get_positions()
                if not isinstance(positions_data, list):
                    positions_data = []
        except Exception:
            pass

        if acct_data and "error" not in acct_data:
            eq = float(acct_data.get("equity", 0))
            leq = float(acct_data.get("last_equity", eq))
            pnl = eq - leq
            pct = (pnl / leq * 100) if leq > 0 else 0
            bp = float(acct_data.get("buying_power", 0))
            pnl_color = COLOR_PROFIT if pnl >= 0 else COLOR_LOSS
            arrow = "+" if pnl >= 0 else ""

            # Big portfolio value
            val = QLabel(f"${eq:,.2f}")
            val.setFont(QFont(FONT_FAMILY, 22, QFont.Bold))
            val.setStyleSheet(f"color: white; {NB}")
            lay.addWidget(val)

            # P&L inline
            pnl_lbl = QLabel(f"{arrow}${pnl:,.2f} ({pct:+.1f}%) today")
            pnl_lbl.setFont(QFont(FONT_FAMILY, 10))
            pnl_lbl.setStyleSheet(f"color: {pnl_color}; {NB}")
            lay.addWidget(pnl_lbl)

            lay.addSpacing(6)

            # Stats row: BP | Positions | Monitored
            stats_row = QHBoxLayout()
            stats_row.setSpacing(0)
            for label, value, color in [
                ("Buying Power", f"${bp:,.0f}", BRAND_SECONDARY),
                ("Positions", str(len(positions_data)), BRAND_ACCENT),
                ("Watching", str(len(self._monitored)), TEXT_SECONDARY),
            ]:
                cell = QVBoxLayout()
                cell.setSpacing(1)
                v = QLabel(value)
                v.setFont(QFont(FONT_FAMILY, 11, QFont.Bold))
                v.setStyleSheet(f"color: {color}; {NB}")
                v.setAlignment(Qt.AlignCenter)
                cell.addWidget(v)
                l = QLabel(label)
                l.setFont(QFont(FONT_FAMILY, 7))
                l.setStyleSheet(f"color: {TEXT_MUTED}; {NB}")
                l.setAlignment(Qt.AlignCenter)
                cell.addWidget(l)
                stats_row.addLayout(cell)
            lay.addLayout(stats_row)
        else:
            no = QLabel("Broker not connected")
            no.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; {NB}")
            lay.addWidget(no)

        # ── Positions ──
        if positions_data:
            lay.addSpacing(8)
            self._add_separator(lay, BRAND_PRIMARY)
            lay.addSpacing(4)

            ph = QLabel(f"POSITIONS ({len(positions_data)})")
            ph.setFont(QFont(FONT_FAMILY, 8, QFont.Bold))
            ph.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 1px; {NB}")
            lay.addWidget(ph)
            lay.addSpacing(4)

            # Sort by unrealized P&L descending (best first)
            sorted_pos = sorted(positions_data, key=lambda p: float(p.get("unrealized_pl", 0)), reverse=True)

            for p in sorted_pos[:8]:
                sym = p.get("symbol", "")
                upl = float(p.get("unrealized_pl", 0))
                upl_pct = float(p.get("unrealized_plpc", 0)) * 100
                cur = float(p.get("current_price", 0))
                qty = float(p.get("qty", 0))
                mkt_val = qty * cur
                pc = COLOR_PROFIT if upl >= 0 else COLOR_LOSS

                row = QHBoxLayout()
                row.setSpacing(0)
                row.setContentsMargins(0, 1, 0, 1)

                # Color bar indicator
                bar = QLabel("")
                bar.setFixedSize(3, 18)
                bar.setStyleSheet(f"background: {pc}; border-radius: 1px; {NB}")
                row.addWidget(bar)
                row.addSpacing(6)

                # Symbol
                sl = QLabel(sym)
                sl.setFont(QFont(FONT_FAMILY, 10, QFont.Bold))
                sl.setStyleSheet(f"color: white; {NB}")
                sl.setFixedWidth(52)
                row.addWidget(sl)

                # Qty + price
                detail = QLabel(f"{qty:.0f} @ ${cur:.2f}")
                detail.setFont(QFont(FONT_MONO, 8))
                detail.setStyleSheet(f"color: {TEXT_MUTED}; {NB}")
                row.addWidget(detail)

                row.addStretch()

                # P&L
                pl = QLabel(f"${upl:+.2f}")
                pl.setFont(QFont(FONT_FAMILY, 9, QFont.Bold))
                pl.setStyleSheet(f"color: {pc}; {NB}")
                pl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                row.addWidget(pl)

                # P&L %
                plp = QLabel(f"{upl_pct:+.1f}%")
                plp.setFont(QFont(FONT_MONO, 8))
                plp.setStyleSheet(f"color: {pc}; {NB}")
                plp.setFixedWidth(48)
                plp.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                row.addWidget(plp)

                lay.addLayout(row)

            if len(positions_data) > 8:
                more = QLabel(f"  +{len(positions_data) - 8} more")
                more.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8px; {NB}")
                lay.addWidget(more)

        # ── Agent Activity ──
        if ag["running"] or self._recent_actions:
            lay.addSpacing(8)
            self._add_separator(lay, BRAND_ACCENT)
            lay.addSpacing(4)

            ah = QLabel("AGENT ACTIVITY")
            ah.setFont(QFont(FONT_FAMILY, 8, QFont.Bold))
            ah.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 1px; {NB}")
            lay.addWidget(ah)
            lay.addSpacing(2)

            if ag["running"]:
                # Agent stats row
                agent_info = QHBoxLayout()
                agent_info.setSpacing(12)
                for label, val, color in [
                    ("Scanned", str(ag.get("scanned", 0)), TEXT_SECONDARY),
                    ("Buys", str(ag.get("buys", 0)), COLOR_BUY),
                    ("Sells", str(ag.get("sells", 0)), COLOR_SELL),
                ]:
                    pair = QHBoxLayout()
                    pair.setSpacing(3)
                    vl = QLabel(val)
                    vl.setFont(QFont(FONT_FAMILY, 10, QFont.Bold))
                    vl.setStyleSheet(f"color: {color}; {NB}")
                    pair.addWidget(vl)
                    ll = QLabel(label)
                    ll.setFont(QFont(FONT_FAMILY, 8))
                    ll.setStyleSheet(f"color: {TEXT_MUTED}; {NB}")
                    pair.addWidget(ll)
                    agent_info.addLayout(pair)
                agent_info.addStretch()
                lay.addLayout(agent_info)
                lay.addSpacing(4)

            # Recent actions
            if self._recent_actions:
                for action_text in self._recent_actions[-5:]:
                    # Color-code by action type
                    if "BUY" in action_text.upper():
                        ac = COLOR_BUY
                    elif "SELL" in action_text.upper():
                        ac = COLOR_SELL
                    elif "ERROR" in action_text.upper() or "FAILED" in action_text.upper():
                        ac = "#ef4444"
                    else:
                        ac = TEXT_MUTED

                    al = QLabel(action_text)
                    al.setFont(QFont(FONT_MONO, 8))
                    al.setStyleSheet(f"color: {ac}; {NB}")
                    al.setWordWrap(True)
                    lay.addWidget(al)

        # ── Monitored stocks (compact) ──
        if self._monitored:
            lay.addSpacing(8)
            self._add_separator(lay, BRAND_SECONDARY)
            lay.addSpacing(4)

            mh = QLabel(f"MONITORING ({len(self._monitored)})")
            mh.setFont(QFont(FONT_FAMILY, 8, QFont.Bold))
            mh.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 1px; {NB}")
            lay.addWidget(mh)
            lay.addSpacing(2)

            for ticker, info in list(self._monitored.items())[:6]:
                signal = info.get("signal", "HOLD")
                conf = info.get("confidence", 0)
                price = info.get("price", 0)
                next_secs = info.get("next_secs", 0)
                sc = {
                    "BUY": COLOR_BUY, "SELL": COLOR_SELL, "HOLD": COLOR_HOLD
                }.get(signal, TEXT_MUTED)

                mr = QHBoxLayout()
                mr.setSpacing(4)
                mr.setContentsMargins(0, 0, 0, 0)

                # Signal dot
                dot = QLabel("*")
                dot.setFont(QFont(FONT_FAMILY, 10, QFont.Bold))
                dot.setStyleSheet(f"color: {sc}; {NB}")
                dot.setFixedWidth(10)
                mr.addWidget(dot)

                tl = QLabel(ticker)
                tl.setFont(QFont(FONT_FAMILY, 9, QFont.Bold))
                tl.setStyleSheet(f"color: white; {NB}")
                tl.setFixedWidth(48)
                mr.addWidget(tl)

                sig = QLabel(f"{signal} {conf:.0%}")
                sig.setFont(QFont(FONT_MONO, 8))
                sig.setStyleSheet(f"color: {sc}; {NB}")
                sig.setFixedWidth(65)
                mr.addWidget(sig)

                pr = QLabel(f"${price:.2f}")
                pr.setFont(QFont(FONT_MONO, 8))
                pr.setStyleSheet(f"color: {TEXT_MUTED}; {NB}")
                mr.addWidget(pr)

                mr.addStretch()

                mins = next_secs // 60
                secs = next_secs % 60
                nxt = QLabel(f"{mins}m{secs:02d}s")
                nxt.setFont(QFont(FONT_MONO, 8))
                nxt.setStyleSheet(f"color: {TEXT_MUTED}; {NB}")
                mr.addWidget(nxt)

                lay.addLayout(mr)

            if len(self._monitored) > 6:
                more = QLabel(f"  +{len(self._monitored) - 6} more")
                more.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8px; {NB}")
                lay.addWidget(more)

        # ── Bottom bar ──
        lay.addSpacing(10)
        self._add_separator(lay, BRAND_PRIMARY)
        lay.addSpacing(6)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        open_btn = QPushButton("Open App")
        open_btn.setStyleSheet(
            f"background: {BRAND_PRIMARY}; color: white; {NB} "
            f"border-radius: 6px; padding: 5px 16px; font-weight: bold; font-size: 10px;"
        )
        open_btn.clicked.connect(lambda: (popup.close(), self._show_window()))
        bottom.addWidget(open_btn)

        bottom.addStretch()

        quit_btn = QPushButton("Quit")
        quit_btn.setStyleSheet(
            f"background: {BG_INPUT}; color: {TEXT_MUTED}; {NB} "
            f"border-radius: 6px; padding: 5px 12px; font-size: 9px;"
        )
        quit_btn.clicked.connect(lambda: (popup.close(), self._quit_with_animation()))
        bottom.addWidget(quit_btn)

        lay.addLayout(bottom)

        container.setLayout(lay)

        outer = QVBoxLayout()
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(container)
        popup.setLayout(outer)
        popup.adjustSize()

        # Position near tray
        geo = self.tray.geometry()
        if geo.isValid():
            cx, cy = geo.center().x(), geo.center().y()
        else:
            screen = QApplication.primaryScreen().availableGeometry()
            cx = screen.right() - 30
            cy = screen.bottom() - 20

        screen = QApplication.primaryScreen().availableGeometry()
        pw, ph = popup.width(), popup.height()
        x = max(screen.x(), min(cx - pw // 2, screen.right() - pw))
        if cy > screen.height() // 2:
            y = cy - ph - 10
        else:
            y = cy + 10
        popup.move(x, y)
        popup.show()

        # Auto-close after 12 seconds
        QTimer.singleShot(12000, lambda: popup.close() if popup.isVisible() else None)
        self._popup = popup

    @staticmethod
    def _add_separator(layout, color):
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {color}33; border: none;")
        layout.addWidget(sep)
