"""
Event Bus — inter-panel communication for StockySuite.

Panels publish events (e.g. "new signal", "trade executed") and other
panels subscribe to react. This decouples panels from each other —
the Scanner doesn't need to know the Dashboard exists, it just emits
"signal_generated" and whoever is listening picks it up.

Uses Qt signals under the hood for thread safety.
"""

from PyQt5.QtCore import QObject, pyqtSignal


class EventBus(QObject):
    """
    Central event bus shared by all panels in the suite.

    Signals:
        signal_generated:  A new trading signal was produced (ticker, action, data_dict)
        trade_executed:    A trade was placed (ticker, side, qty, order_id)
        scan_started:      Portfolio scan began (ticker_count)
        scan_completed:    Portfolio scan finished (results_list)
        ticker_selected:   User wants to analyze a specific ticker (ticker)
        settings_changed:  Settings were updated (settings_dict)
        log_entry:         New log entry for the activity feed (message, level)
        positions_changed: Open positions changed (refresh dashboard)
    """

    signal_generated = pyqtSignal(str, str, dict)     # ticker, action, full_data
    trade_executed = pyqtSignal(str, str, int, str)    # ticker, side, qty, order_id
    scan_started = pyqtSignal(int)                     # ticker_count
    scan_completed = pyqtSignal(list)                  # list of ScanResult dicts
    ticker_selected = pyqtSignal(str)                  # ticker symbol
    settings_changed = pyqtSignal(dict)                # updated settings
    log_entry = pyqtSignal(str, str)                   # message, level
    positions_changed = pyqtSignal()                   # trigger dashboard refresh
