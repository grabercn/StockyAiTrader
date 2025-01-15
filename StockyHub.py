import sys
import os
import json
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QDialog,
    QLineEdit, QFormLayout, QHBoxLayout, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtGui import QPixmap, QFont

STOCKY_APPS_FOLDER = "StockyApps"
SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "api_key": "your_api_key_here",
    "default_ticker": "AAPL",
    "refresh_rate": 5,
    "theme": "light"
}

if not os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, "w") as file:
        json.dump(DEFAULT_SETTINGS, file, indent=4)

def load_settings():
    with open(SETTINGS_FILE, "r") as file:
        settings = json.load(file)
    # Ensure the theme key exists, default to "light" if missing
    settings.setdefault("theme", "light")
    return settings

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as file:
        json.dump(settings, file, indent=4)

class SettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2f;
                color: white;
            }
            QLineEdit, QPushButton {
                font-size: 14px;
                padding: 6px;
            }
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.layout = QFormLayout()
        self.settings = load_settings()
        self.api_key_input = QLineEdit(self.settings.get("api_key", ""))
        self.ticker_input = QLineEdit(self.settings.get("default_ticker", ""))
        self.refresh_rate_input = QLineEdit(str(self.settings.get("refresh_rate", 5)))
        self.layout.addRow("API Key:", self.api_key_input)
        self.layout.addRow("Default Ticker:", self.ticker_input)
        self.layout.addRow("Refresh Rate (s):", self.refresh_rate_input)

        self.buttons = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)

        self.buttons.addWidget(self.save_button)
        self.buttons.addWidget(self.cancel_button)
        self.layout.addRow(self.buttons)
        self.setLayout(self.layout)

    def save_settings(self):
        try:
            self.settings["api_key"] = self.api_key_input.text()
            self.settings["default_ticker"] = self.ticker_input.text()
            self.settings["refresh_rate"] = int(self.refresh_rate_input.text())
            save_settings(self.settings)
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
            self.close()
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid refresh rate. Must be an integer.")

class StockyHub(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stocky Hub")
        self.setGeometry(200, 200, 600, 500)  # Made the window larger

        self.settings = load_settings()
        self.apply_theme(self.settings["theme"])  # Apply theme

        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d44;
                color: white;
            }
            QLabel {
                font-size: 18px;
            }
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # Main widget and layout
        self.main_widget = QWidget()
        self.layout = QVBoxLayout()

        # Banner Image (comment this out if it causes issues)
        self.banner = QLabel()
        self.banner.setPixmap(QPixmap("banner.jpg").scaledToHeight(100, Qt.SmoothTransformation))
        self.banner.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.banner)

        # Welcome Label
        self.welcome_label = QLabel("Welcome to Stocky Hub")
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.layout.addWidget(self.welcome_label)

        # App Buttons
        self.add_app_button("DayTrader", "A trading app for intraday strategies.", "DayTrader.py")
        self.add_app_button("LongTrader", "For long-term investment strategies.", "LongTrader.py")
        self.add_app_button("StockExecuter", "Executes stock orders automatically.", "StockExecuter.py")

        # Settings Button
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        self.layout.addWidget(self.settings_button, alignment=Qt.AlignCenter)

        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        # Ensure window has a top bar (removes FramelessWindowHint)
        self.setWindowFlags(Qt.Window)  # This enables the default window behavior with a top bar

    def add_app_button(self, app_name, description, script_name):
        layout = QHBoxLayout()
        button = QPushButton(app_name)
        button.clicked.connect(lambda: self.launch_app(script_name))
        label = QLabel(description)
        label.setStyleSheet("font-size: 14px; color: #b0bec5;")
        layout.addWidget(button)
        layout.addWidget(label)
        self.layout.addLayout(layout)

    def launch_app(self, script_name):
        script_path = os.path.join(STOCKY_APPS_FOLDER, script_name)
        if os.path.exists(script_path):
            try:
                # Run the script without feedback
                subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to launch {script_name}. Error: {e}")
        else:
            QMessageBox.warning(self, "Error", f"Script {script_name} not found in {STOCKY_APPS_FOLDER}.")

    def open_settings(self):
        dialog = SettingsDialog()
        dialog.exec_()

    def apply_theme(self, theme):
        if theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2d2d44;
                    color: white;
                }
                QPushButton {
                    background-color: #4caf50;
                    color: white;
                }
            """)
        elif theme == "light":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f5f5f5;
                    color: black;
                }
                QPushButton {
                    background-color: #2196f3;
                    color: white;
                }
            """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    hub = StockyHub()
    hub.show()
    sys.exit(app.exec_())
