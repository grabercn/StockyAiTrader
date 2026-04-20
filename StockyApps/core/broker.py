"""
Alpaca brokerage API wrapper.

Handles all communication with Alpaca's REST API for:
- Account info (portfolio value, buying power)
- Order placement (market orders with bracket stop-loss/take-profit)
- Position management (list, close individual, close all)
- Portfolio history (for charting)

Currently uses paper trading URL. Set paper=False for live trading.
"""

import json
import os
import requests
from datetime import datetime

# Trade history is logged locally for review
_TRADE_LOG = os.path.join(os.path.dirname(__file__), "..", "..", "trade_history.json")


class AlpacaBroker:
    """
    Thin wrapper around Alpaca's REST API.

    All methods return dicts. On error, the dict contains an "error" key.
    This avoids exceptions bubbling up to the UI layer.
    """

    def __init__(self, api_key, secret_key, paper=True):
        base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.base_url = f"{base}/v2"
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

    # ── HTTP helpers ──────────────────────────────────────────────────────

    def _get(self, endpoint, params=None):
        try:
            r = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def _post(self, endpoint, data):
        try:
            r = requests.post(f"{self.base_url}/{endpoint}", headers=self.headers, json=data)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def _delete(self, endpoint):
        try:
            r = requests.delete(f"{self.base_url}/{endpoint}", headers=self.headers)
            r.raise_for_status()
            return r.json() if r.content else {"status": "ok"}
        except requests.RequestException as e:
            return {"error": str(e)}

    # ── Account ───────────────────────────────────────────────────────────

    def get_account(self):
        """Get account details (portfolio value, buying power, cash, etc.)."""
        return self._get("account")

    def get_portfolio_history(self, period="1W", timeframe="1H"):
        """Get equity curve over time for charting."""
        return self._get("account/portfolio/history", {"period": period, "timeframe": timeframe})

    # ── Positions ─────────────────────────────────────────────────────────

    def get_positions(self):
        """List all open positions with current P&L."""
        return self._get("positions")

    def cancel_orders_for_symbol(self, symbol):
        """Cancel all open orders for a specific symbol (frees held shares)."""
        try:
            orders = self.get_orders("open")
            if not isinstance(orders, list):
                return
            for o in orders:
                if o.get("symbol", "").upper() == symbol.upper():
                    oid = o.get("id")
                    if oid:
                        requests.delete(f"{self.base_url}/orders/{oid}", headers=self.headers)
        except Exception:
            pass

    def close_position(self, symbol, qty=None):
        """
        Close/sell a position by symbol.

        Automatically cancels any open orders holding shares first
        (bracket order stop-loss/take-profit legs lock shares and
        cause 403 "insufficient qty available" if not cancelled).
        """
        try:
            # Cancel open orders first — they hold shares hostage
            self.cancel_orders_for_symbol(symbol)

            # Small delay for Alpaca to process cancellations
            import time
            time.sleep(0.5)

            params = {}
            if qty is not None:
                params["qty"] = str(qty)
            r = requests.delete(
                f"{self.base_url}/positions/{symbol}",
                headers=self.headers,
                params=params,
            )
            r.raise_for_status()
            return r.json() if r.content else {"status": "ok"}
        except requests.RequestException as e:
            return {"error": str(e)}

    def close_all_positions(self):
        """Emergency: liquidate everything."""
        return self._delete("positions")

    # ── Orders ────────────────────────────────────────────────────────────

    def get_orders(self, status="open"):
        """List orders by status (open, closed, all)."""
        return self._get("orders", {"status": status})

    def place_order(self, symbol, qty, side, order_type="market",
                    time_in_force="day", stop_loss=None, take_profit=None):
        """
        Place an order, optionally as a bracket order with SL/TP.

        Bracket orders automatically attach a stop-loss and take-profit
        so the exit is managed by the broker, not our code.

        Args:
            symbol:       Ticker (e.g. "AAPL")
            qty:          Number of shares
            side:         "buy" or "sell"
            order_type:   "market", "limit", etc.
            time_in_force: "day" (close at EOD) or "gtc" (good til cancelled)
            stop_loss:    Stop-loss price (triggers bracket order if set with take_profit)
            take_profit:  Take-profit price (triggers bracket order if set with stop_loss)
        """
        order = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }

        # Bracket order: market entry + automatic SL + TP
        if stop_loss and take_profit and order_type == "market":
            order["order_class"] = "bracket"
            order["stop_loss"] = {"stop_price": f"{stop_loss:.2f}"}
            order["take_profit"] = {"limit_price": f"{take_profit:.2f}"}

        result = self._post("orders", order)

        # Log every trade attempt locally
        _log_trade({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol, "side": side, "qty": qty,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "order_id": result.get("id", "failed"),
            "error": result.get("error"),
        })

        return result


def _log_trade(trade_data):
    """Append trade to local JSON log (keeps last 500 entries)."""
    history = []
    if os.path.exists(_TRADE_LOG):
        try:
            with open(_TRADE_LOG, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    history.append(trade_data)
    history = history[-500:]  # Rolling window

    with open(_TRADE_LOG, "w") as f:
        json.dump(history, f, indent=2)
