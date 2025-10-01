import sys
import threading
import requests
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout,
    QDialog, QLineEdit, QPushButton, QFormLayout, QMenuBar, QAction, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# MQTT broker settings
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "stock-predictions"

# --- Settings Dialog ---
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.layout = QFormLayout(self)

        self.api_key_input = QLineEdit(self)
        self.secret_key_input = QLineEdit(self)
        self.secret_key_input.setEchoMode(QLineEdit.Password)

        self.layout.addRow("Alpaca API Key:", self.api_key_input)
        self.layout.addRow("Alpaca Secret Key:", self.secret_key_input)

        self.test_button = QPushButton("Test Connection", self)
        self.test_button.clicked.connect(self.test_connection)
        self.layout.addWidget(self.test_button)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_settings)
        self.layout.addWidget(self.save_button)

    def load_settings(self, settings):
        self.api_key_input.setText(settings.get("alpaca_api_key", ""))
        self.secret_key_input.setText(settings.get("alpaca_secret_key", ""))

    def save_settings(self):
        settings = {
            "alpaca_api_key": self.api_key_input.text(),
            "alpaca_secret_key": self.secret_key_input.text()
        }
        try:
            with open("settings.json", "r+") as f:
                data = json.load(f)
                data.update(settings)
                f.seek(0)
                json.dump(data, f, indent=4)
            QMessageBox.information(self, "Success", "Settings saved successfully.")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save settings: {e}")

    def test_connection(self):
        api_key = self.api_key_input.text()
        secret_key = self.secret_key_input.text()
        if not api_key or not secret_key:
            QMessageBox.warning(self, "Error", "API Key and Secret Key cannot be empty.")
            return

        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        }
        try:
            response = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers)
            response.raise_for_status()
            QMessageBox.information(self, "Success", "Alpaca API connection successful!")
        except requests.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to connect to Alpaca API: {e}")

# --- Alpaca API Wrapper ---
class AlpacaTradingAPI:
    def __init__(self, api_key, secret_key):
        self.base_url = "https://paper-api.alpaca.markets/v2"
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        }

    def get_account(self):
        try:
            response = requests.get(f"{self.base_url}/account", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_portfolio_history(self, period='1M', timeframe='1D'):
        params = {'period': period, 'timeframe': timeframe}
        try:
            response = requests.get(f"{self.base_url}/account/portfolio/history", headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def place_order(self, symbol, qty, side, order_type="market", time_in_force="gtc"):
        order_data = {"symbol": symbol, "qty": qty, "side": side, "type": order_type, "time_in_force": time_in_force}
        try:
            response = requests.post(f"{self.base_url}/orders", headers=self.headers, json=order_data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_positions(self):
        try:
            response = requests.get(f"{self.base_url}/positions", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}


# --- Main Application Window ---
class StockExecutionDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stocky Executer - Dashboard")
        self.setGeometry(100, 100, 900, 600)
        self.trading_api = None
        self.settings = self.load_app_settings()

        self.init_ui()
        self.init_api()
        self.update_dashboard()

    def init_ui(self):
        # Menu Bar
        menubar = self.menuBar()
        settings_menu = menubar.addMenu('Settings')
        edit_settings_action = QAction('Edit API Keys', self)
        edit_settings_action.triggered.connect(self.open_settings_dialog)
        settings_menu.addAction(edit_settings_action)

        # Main Layout
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Left side: Graph and Stats
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.stats_label = QLabel("Account Statistics:")
        self.stats_label.setAlignment(Qt.AlignTop)
        left_layout.addWidget(QLabel("Portfolio Performance:"))
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(self.stats_label)

        # Right side: Predictions and Log
        self.prediction_label = QLabel("Waiting for predictions...")
        self.prediction_label.setWordWrap(True)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        right_layout.addWidget(QLabel("Live Predictions:"))
        right_layout.addWidget(self.prediction_label)
        right_layout.addWidget(QLabel("Activity Log:"))
        right_layout.addWidget(self.log)

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def init_api(self):
        api_key = self.settings.get("alpaca_api_key")
        secret_key = self.settings.get("alpaca_secret_key")
        if api_key and secret_key:
            self.trading_api = AlpacaTradingAPI(api_key, secret_key)
        else:
            self.log_action("Alpaca API keys not found. Please configure them in Settings.", error=True)

    def load_app_settings(self):
        try:
            with open("settings.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def open_settings_dialog(self):
        dialog = SettingsDialog(self)
        dialog.load_settings(self.settings)
        if dialog.exec_():
            self.settings = self.load_app_settings()
            self.init_api()
            self.update_dashboard()

    def update_dashboard(self):
        if not self.trading_api:
            return

        # Update stats
        account_data = self.trading_api.get_account()
        if "error" in account_data:
            self.stats_label.setText(f"Account Statistics:\nError: {account_data['error']}")
        else:
            stats_text = f"""
            Portfolio Value: ${account_data.get('portfolio_value', 'N/A')}\n
            Buying Power: ${account_data.get('buying_power', 'N/A')}\n
            Cash: ${account_data.get('cash', 'N/A')}\n
            """
            self.stats_label.setText(f"Account Statistics:\n{stats_text}")

        # Update graph
        history = self.trading_api.get_portfolio_history()
        if "error" not in history and history.get('equity'):
            self.plot_portfolio_history(history)

    def plot_portfolio_history(self, history):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        equity = history['equity']
        timestamps = [datetime.fromtimestamp(t) for t in history['timestamp']]
        ax.plot(timestamps, equity, label='Portfolio Value')
        
        # Plot gains/losses
        profit_loss = history['profit_loss']
        profit_loss_pct = history['profit_loss_pct']
        
        # Create a second y-axis for profit/loss percentage
        ax2 = ax.twinx()
        ax2.plot(timestamps, profit_loss, 'g--', label='Profit/Loss ($)')
        
        ax.set_title('Portfolio Performance')
        ax.set_ylabel('Portfolio Value ($)')
        ax2.set_ylabel('Profit/Loss ($)', color='g')
        # Format date on X-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
        self.figure.autofmt_xdate()

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        self.figure.tight_layout()
        self.canvas.draw()

    def log_action(self, action, error=None):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_message = f"{timestamp} {action}"
        if error:
            log_message = f"{timestamp} Error: {action}"
        self.log.append(log_message)

    def handle_prediction(self, message):
        self.prediction_label.setText(message)
        self.log_action(f"Received prediction: {message}")
        
        if not self.trading_api:
            self.log_action("Cannot execute trade, API not configured.", error=True)
            return

        try:
            ticker, action, confidence_str = message.split(' : ')
            confidence = float(confidence_str)
            action = action.lower()

            if action == "buy":
                result = self.trading_api.place_order(ticker, 10, "buy") # Example qty
                self.log_action(f"Placed BUY order for 10 {ticker}.", result.get("error"))
            elif action == "sell":
                result = self.trading_api.place_order(ticker, 10, "sell") # Example qty
                self.log_action(f"Placed SELL order for 10 {ticker}.", result.get("error"))
            
            self.log_action("Updating dashboard...")
            self.update_dashboard()

        except ValueError:
            self.log_action(f"Could not parse prediction: {message}", error=True)


# --- MQTT Client ---
class MqttClient(QObject):
    message_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            self.client.subscribe(TOPIC)
        else:
            print(f"Failed to connect, return code {rc}\n")

    def on_message(self, client, userdata, msg):
        self.message_received.emit(msg.payload.decode())

    def connect(self):
        self.client.connect(BROKER, PORT, 60)
        thread = threading.Thread(target=self.client.loop_forever)
        thread.daemon = True
        thread.start()


def main():
    app = QApplication(sys.argv)
    window = StockExecutionDashboard()
    
    mqtt_client = MqttClient()
    mqtt_client.message_received.connect(window.handle_prediction)
    mqtt_client.connect()
    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main().client.loop_start()

mqtt_thread = threading.Thread(target=mqtt_loop)
mqtt_thread.daemon = True
mqtt_thread.start()

# Show the PyQt5 window
window.show()

# Start the PyQt5 event loop
sys.exit(app.exec_())
