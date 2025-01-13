import sys
import paho.mqtt.client as mqtt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

# MQTT broker settings
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "stock-predictions"

# Callback when the subscriber receives a message
def on_message(client, userdata, msg):
    # Parse the message payload (format: ticker : action : confidence)
    message = msg.payload.decode()
    try:
        ticker, action, confidence = message.split(' : ')
        display_message = f"Ticker: {ticker}\nAction: {action}\nConfidence: {confidence}"
        # Update the QLabel to display the message
        window.update_display(display_message)
    except ValueError:
        print("Received message is not in the expected format.")

# Callback when the subscriber connects to the broker
def on_connect(client, userdata, flags, rc):
    print(f"Connected to broker with result code: {rc}")
    # Subscribe to the topic
    client.subscribe(TOPIC)

# Create a PyQt5 window to display the predictions
class StockPredictionWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stock Predictions")
        self.setGeometry(100, 100, 400, 200)

        # Create a QLabel widget to display the data
        self.label = QLabel("Waiting for predictions...", self)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)

        # Layout for the window
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_display(self, message):
        """Update the label with the received message."""
        self.label.setText(message)

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
import threading
def mqtt_loop():
    client.loop_start()

# Start the MQTT loop in a background thread
mqtt_thread = threading.Thread(target=mqtt_loop)
mqtt_thread.daemon = True
mqtt_thread.start()

# Show the PyQt5 window
window.show()

# Start the PyQt5 event loop
sys.exit(app.exec_())
