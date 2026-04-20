"""
Order Manager — tracks and displays the lifecycle of all orders.

Integrates with Alpaca's order status system:

Order Lifecycle:
    new → accepted → partially_filled → filled
                   → cancelled
                   → rejected
                   → expired

This module:
1. Polls Alpaca for order status updates
2. Maintains a local queue of active orders
3. Emits signals when orders change status (for UI updates + notifications)
4. Provides the data for the order queue UI widget

We DON'T reinvent Alpaca's queue — we mirror it and add UI/notification on top.
"""

import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
from PyQt5.QtCore import QThread, pyqtSignal, QTimer


@dataclass
class TrackedOrder:
    """Local representation of an Alpaca order."""
    order_id: str
    symbol: str
    side: str           # "buy" or "sell"
    qty: int
    order_type: str     # "market", "limit", etc.
    status: str         # "new", "accepted", "filled", "cancelled", etc.
    submitted_at: str
    filled_at: str = ""
    filled_qty: int = 0
    filled_price: float = 0.0
    stop_price: float = 0.0
    limit_price: float = 0.0
    order_class: str = ""  # "", "bracket", "oco", etc.
    is_leg: bool = False   # True if this is a SL/TP leg of a bracket


class OrderManager(QThread):
    """
    Background thread that polls Alpaca for order status updates.

    Emits:
        order_updated(TrackedOrder) — when any order changes status
        order_filled(symbol, side, qty, price) — when an order fills
        order_failed(symbol, side, reason) — when an order is rejected/cancelled
    """

    order_updated = pyqtSignal(object)          # TrackedOrder
    order_filled = pyqtSignal(str, str, int, float)   # symbol, side, qty, price
    order_failed = pyqtSignal(str, str, str)          # symbol, side, reason
    log = pyqtSignal(str, str)                        # message, level

    def __init__(self, broker):
        super().__init__()
        self.broker = broker
        self._orders: Dict[str, TrackedOrder] = {}
        self._running = True
        self._poll_interval = 5  # Check every 5 seconds

    def get_active_orders(self) -> List[TrackedOrder]:
        """Get all orders that aren't in a terminal state."""
        terminal = {"filled", "cancelled", "expired", "rejected"}
        return [o for o in self._orders.values() if o.status not in terminal]

    def get_recent_orders(self, limit=20) -> List[TrackedOrder]:
        """Get most recent orders including completed ones."""
        all_orders = sorted(self._orders.values(), key=lambda o: o.submitted_at, reverse=True)
        return all_orders[:limit]

    def get_orders_for_symbol(self, symbol) -> List[TrackedOrder]:
        """Get all orders for a specific symbol."""
        return [o for o in self._orders.values() if o.symbol == symbol]

    def track_order(self, order_data):
        """Add an order to tracking from Alpaca's response."""
        if not order_data or "error" in order_data:
            return
        oid = order_data.get("id", "")
        if not oid:
            return

        self._orders[oid] = TrackedOrder(
            order_id=oid,
            symbol=order_data.get("symbol", ""),
            side=order_data.get("side", ""),
            qty=int(float(order_data.get("qty", 0))),
            order_type=order_data.get("type", "market"),
            status=order_data.get("status", "new"),
            submitted_at=order_data.get("submitted_at", datetime.now().isoformat())[:19],
            filled_qty=int(float(order_data.get("filled_qty", 0))),
            filled_price=float(order_data.get("filled_avg_price", 0) or 0),
            stop_price=float(order_data.get("stop_price", 0) or 0),
            limit_price=float(order_data.get("limit_price", 0) or 0),
            order_class=order_data.get("order_class", ""),
        )
        self.log.emit(
            f"Order tracked: {order_data.get('side','').upper()} "
            f"{order_data.get('symbol','')} x{order_data.get('qty',0)} "
            f"[{order_data.get('status','')}]",
            "info",
        )

    def stop(self):
        self._running = False

    def run(self):
        """Poll Alpaca for order status changes."""
        self.log.emit("Order manager started — polling every 5s", "system")

        while self._running:
            try:
                self._poll_orders()
            except Exception as e:
                self.log.emit(f"Order poll error: {e}", "error")
            time.sleep(self._poll_interval)

    def _poll_orders(self):
        """Fetch current orders from Alpaca and update local state."""
        if not self.broker:
            return

        # Get all open orders
        open_orders = self.broker.get_orders("open")
        if not isinstance(open_orders, list):
            return

        # Get recently closed orders (last hour)
        closed_orders = self.broker.get_orders("closed")
        if not isinstance(closed_orders, list):
            closed_orders = []

        all_remote = {o.get("id"): o for o in open_orders + closed_orders[:20] if o.get("id")}

        # Update local orders
        for oid, remote in all_remote.items():
            new_status = remote.get("status", "")
            filled_qty = int(float(remote.get("filled_qty", 0)))
            filled_price = float(remote.get("filled_avg_price", 0) or 0)

            if oid in self._orders:
                old = self._orders[oid]
                if old.status != new_status:
                    # Status changed — emit signals
                    old.status = new_status
                    old.filled_qty = filled_qty
                    old.filled_price = filled_price
                    old.filled_at = remote.get("filled_at", "")[:19] if remote.get("filled_at") else ""

                    self.order_updated.emit(old)

                    if new_status == "filled":
                        self.order_filled.emit(old.symbol, old.side, old.qty, filled_price)
                        self.log.emit(
                            f"Order FILLED: {old.side.upper()} {old.symbol} "
                            f"x{old.qty} @ ${filled_price:.2f}",
                            "trade",
                        )
                    elif new_status in ("cancelled", "rejected", "expired"):
                        self.order_failed.emit(old.symbol, old.side, new_status)
                        self.log.emit(
                            f"Order {new_status.upper()}: {old.side.upper()} {old.symbol} x{old.qty}",
                            "warn",
                        )
            else:
                # New order we haven't seen — track it
                self.track_order(remote)
