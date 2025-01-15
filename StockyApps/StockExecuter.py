import sys
import threading
import requests
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout
)
from PyQt5.QtCore import Qt
import paho.mqtt.client as mqtt

# Alpaca API settings
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"
API_KEY = "PK1HNLS989GG2QFSYPQT"  # Replace with your Alpaca API key
SECRET_KEY = "txODF9YjwtaeN7Dh4gwb2UOheqclo5iM9yMIW7s2"  # Replace with your Alpaca secret key

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}

# MQTT broker settings
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "stock-predictions"

# Alpaca API Wrapper
class AlpacaTradingAPI:
    def __init__(self):
        self.base_url = ALPACA_BASE_URL
        self.headers = HEADERS

    def get_account(self):
        """Fetch account details with error handling."""
        try:
            response = requests.get(f"{self.base_url}/account", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching account data: {e}")
            return {"error": "Unable to fetch account data. Please check your API keys or network connection."}

    def place_order(self, symbol, qty, side, order_type="market", time_in_force="gtc"):
        """Place an order on Alpaca."""
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        try:
            response = requests.post(f"{self.base_url}/orders", headers=self.headers, json=order_data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error placing order: {e}")
            return {"error": "Order failed. Please check your API keys or order parameters."}

    def get_portfolio(self):
        """Fetch current portfolio positions with error handling."""
        try:
            response = requests.get(f"{self.base_url}/positions", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching portfolio data: {e}")
            return {"error": "Unable to fetch portfolio data. Please check your API keys or network connection."}

# Create a PyQt5 window to display predictions and statistics
class StockPredictionWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stocky Executer")
        self.setGeometry(100, 100, 700, 500)

        # Create widgets
        self.prediction_label = QLabel("Waiting for predictions...", self)
        self.prediction_label.setWordWrap(True)
        self.prediction_label.setAlignment(Qt.AlignCenter)

        self.log = QTextEdit(self)
        self.log.setReadOnly(True)

        self.stats_label = QLabel("Trading Statistics", self)
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(Qt.AlignLeft)

        # Layout
        prediction_layout = QVBoxLayout()
        prediction_layout.addWidget(self.prediction_label)
        prediction_layout.addWidget(self.log)

        stats_layout = QVBoxLayout()
        stats_layout.addWidget(self.stats_label)

        main_layout = QHBoxLayout()
        main_layout.addLayout(prediction_layout, stretch=2)
        main_layout.addLayout(stats_layout, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Initialize Alpaca API
        self.trading_api = AlpacaTradingAPI()

    def update_display(self, message):
        """Update the prediction label and log."""
        self.prediction_label.setText(message)
        self.log_action(f"Received prediction: {message}")

    def log_action(self, action, error=None):
        """Log an action with error details if any."""
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        if error:
            self.log.append(f"{timestamp} Error: {error}")
        else:
            self.log.append(f"{timestamp} {action}")

    def update_stats(self):
        """Update the statistics label with account and portfolio details."""
        account = self.trading_api.get_account()
        positions = self.trading_api.get_portfolio()

        if "error" in account:
            stats_text = account["error"]
        else:
            stats_text = f"Cash: ${account.get('cash', 'N/A')} Portfolio Value: ${account.get('portfolio_value', 'N/A')}\n"

        if isinstance(positions, dict) and "error" in positions:
            stats_text += f"\n{positions['error']}\n"
        else:
            stats_text += "\nCurrent Positions:\n"
            for position in positions:
                stats_text += f"{position['symbol']}: {position['qty']} shares @ ${position['avg_entry_price']}\n"

        self.stats_label.setText(f"Trading Statistics:\n{stats_text}")

def on_message(client, userdata, msg):
    message = msg.payload.decode()
    try:
        ticker, action, confidence = message.split(' : ')
        confidence = float(confidence)
        display_message = f"Ticker: {ticker}\nAction: {action}\nConfidence: {confidence:.2f}"
        window.update_display(display_message)

        # Fetch portfolio details
        positions = window.trading_api.get_portfolio()
        owned_symbols = {pos['symbol']: pos for pos in positions if 'symbol' in pos}

        if action.lower() == "hold":
            window.log_action(f"Holding {ticker}.")
            return

        if action.lower() == "buy":
            qty = 10  # Mock quantity to trade
            result = window.trading_api.place_order(ticker, qty, "buy")
            window.log_action(f"Bought {qty} {ticker}.", result.get("error"))

        elif action.lower() == "sell":
            if ticker not in owned_symbols:
                window.log_action(f"Cannot sell {ticker}. Shares not owned.")
                return

            # Get current position details
            position = owned_symbols[ticker]
            avg_entry_price = float(position.get('avg_entry_price', 0.0))
            current_price = float(position.get('current_price', 0.0))  # Assume Alpaca API provides 'current_price'

            if avg_entry_price > current_price:
                window.log_action(
                    f"Cannot sell {ticker}. Selling at a loss.\n"
                    f"Purchased at: ${avg_entry_price:.2f}, Current price: ${current_price:.2f}"
                )
                return

            qty = 10  # Mock quantity to trade
            result = window.trading_api.place_order(ticker, qty, "sell")
            window.log_action(f"Sold {qty} {ticker}.", result.get("error"))

        # Update statistics
        window.update_stats()
    except ValueError:
        print("Received message is not in the expected format.")


def on_connect(client, userdata, flags, rc):
    print(f"Connected to broker with result code: {rc}")
    client.subscribe(TOPIC)

# Create the MQTT client and assign callbacks
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Start the PyQt5 application
app = QApplication(sys.argv)
window = StockPredictionWindow()

# Connect to the MQTT broker
client.connect(BROKER, PORT, 60)

# Start the MQTT client loop in a separate thread
def mqtt_loop():
    client.loop_start()

mqtt_thread = threading.Thread(target=mqtt_loop)
mqtt_thread.daemon = True
mqtt_thread.start()

# Show the PyQt5 window
window.show()

# Start the PyQt5 event loop
sys.exit(app.exec_())
