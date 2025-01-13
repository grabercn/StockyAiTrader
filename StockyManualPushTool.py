import sys
import paho.mqtt.client as mqtt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout,
    QHBoxLayout, QWidget, QPushButton, QComboBox, QSpinBox
)

# MQTT broker settings
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "stock-predictions"

class ManualPushTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MQTT Manual Push Tool")
        self.setGeometry(100, 100, 400, 300)

        # Layout and widgets
        layout = QVBoxLayout()

        # Input for ticker
        self.ticker_label = QLabel("Ticker:")
        self.ticker_input = QLineEdit()
        layout.addWidget(self.ticker_label)
        layout.addWidget(self.ticker_input)

        # Dropdown for action
        self.action_label = QLabel("Action:")
        self.action_dropdown = QComboBox()
        self.action_dropdown.addItems(["Buy", "Hold", "Sell"])
        layout.addWidget(self.action_label)
        layout.addWidget(self.action_dropdown)

        # Input for confidence
        self.confidence_label = QLabel("Confidence (%):")
        self.confidence_input = QSpinBox()
        self.confidence_input.setRange(0, 100)
        self.confidence_input.setValue(50)  # Default value
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.confidence_input)

        # Publish button
        self.publish_button = QPushButton("Publish")
        self.publish_button.clicked.connect(self.publish_message)
        layout.addWidget(self.publish_button)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Central widget setup
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # MQTT client setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(BROKER, PORT, 60)

    def publish_message(self):
        """Publish the stock prediction message."""
        ticker = self.ticker_input.text().strip()
        action = self.action_dropdown.currentText()
        confidence = self.confidence_input.value()

        if not ticker:
            self.status_label.setText("Error: Ticker cannot be empty.")
            return

        # Format the message
        message = f"{ticker} : {action} : {confidence}"
        try:
            self.mqtt_client.publish(TOPIC, message)
            self.status_label.setText(f"Message published: {message}")
        except Exception as e:
            self.status_label.setText(f"Failed to publish message: {e}")

# Main application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ManualPushTool()
    window.show()
    sys.exit(app.exec_())
