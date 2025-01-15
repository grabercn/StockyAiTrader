import sys
import time
import yfinance as yf
import numpy as np
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar, QComboBox
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import feedparser
import paho.mqtt.client as mqtt
import pytz
from datetime import datetime, timedelta

# MQTT broker settings
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "stock-predictions"

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels: BUY, SELL, HOLD

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def fetch_and_prepare_data(stock_ticker, period="1d", interval="5m"):
    stock = yf.Ticker(stock_ticker)
    
    # Fetch historical data with a specific interval
    data = stock.history(period=period, interval=interval)
    
    print(data)

    # Calculate percentage price changes
    data['Price Change (%)'] = data['Close'].pct_change() * 100

    # Label data based on thresholds
    conditions = [
        (data['Price Change (%)'] > 1.5),  # Large positive change -> BUY
        (data['Price Change (%)'] < -1.5), # Large negative change -> SELL
        (abs(data['Price Change (%)']) <= 1.5)  # Small changes -> HOLD
    ]
    labels = [2, 0, 1]  # BUY=2, SELL=0, HOLD=1
    data['Label'] = np.select(conditions, labels)

    # Create text input for AI
    data['Text'] = data.apply(
        lambda row: f"Stock price: {row['Close']:.2f}, volume: {row['Volume']}, percentage change: {row['Price Change (%)']:.2f}.",
        axis=1
    )

    # Filter out rows with NaN labels (e.g., first row with no percentage change)
    data = data.dropna(subset=['Label'])

    return data[['Text', 'Label', 'Close']]

# Function to publish prediction to MQTT
def publish_prediction(stock_ticker, prediction):
    client = mqtt.Client()
    
    # Connect to the MQTT broker
    client.connect(BROKER, PORT, 60)
    
    # Start the loop in the background to handle network operations
    client.loop_start()

    # Publish the prediction
    message = f"{stock_ticker} : {prediction['action']} : {prediction['confidence']:.2f}"
    
    # Publish the message multiple times for redundancy
    client.publish(TOPIC, message)

    print(f"Message published: {message}")

    # Stop the loop once the message is published (optional, depends on use case)
    client.loop_stop()

# Function to fine-tune the model
def train_model(data):
    train_encodings = tokenizer(list(data['Text']), truncation=True, padding=True, max_length=512)
    train_labels = list(data['Label'])
    train_dataset = CustomDataset(train_encodings, train_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()

# Dataset preparation for training
class CustomDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# Use the trained model for predictions
def predict(stock_ticker, period="5d", interval="5m"):
    data = fetch_and_prepare_data(stock_ticker, period, interval)
    predictions = []

    for _, row in data.iterrows():
        inputs = tokenizer(row['Text'], return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits[0].detach().numpy()

        action_idx = logits.argmax()
        action = ['SELL', 'HOLD', 'BUY'][action_idx]
        confidence = np.max(logits)

        predictions.append({
            "text": row['Text'],
            "action": action,
            "logits": logits,
            "confidence": confidence,
            "close_price": row['Close'],
            "description": f"Prediction based on stock price and volume changes.",
            "evaluation": f"Model evaluated the stock's behavior as {action} due to the percentage change in price."
        })
        
    publish_prediction(stock_ticker, predictions[-1])

    return predictions

# Fetch latest news headlines from Google News RSS
def fetch_news(stock_ticker):
    rss_url = f"https://news.google.com/rss/search?q={stock_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    headlines = []
    for entry in feed.entries[:5]:  # Get the top 5 headlines
        headlines.append(entry.title)

    if not headlines:
        headlines = ["No recent news available."]
    
    return headlines

# PyQt5 UI to display graphs and stock recommendations
class StockPredictionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Stocky Day Trader')
        self.setGeometry(100, 100, 1200, 800)

        # Create main layout
        main_layout = QVBoxLayout()

        # Input for custom stock ticker
        self.ticker_input = QLineEdit(self)
        self.ticker_input.setPlaceholderText("Enter stock ticker (e.g., AAPL, GOOGL)")
        
        self.training_period_dropdown = QComboBox(self)
        self.training_period_dropdown.addItems(['1d', '2d','3d', '5d'])
        
        self.prediction_period_dropdown = QComboBox(self)
        self.prediction_period_dropdown.addItems(['1m', '5m', '15m', '30m', '60m']) 
        self.ticker_button = QPushButton('Get Prediction', self)
        self.ticker_button.clicked.connect(self.on_ticker_button_clicked)

        main_layout.addWidget(QLabel('Enter Stock Ticker:'))
        main_layout.addWidget(self.ticker_input)
        main_layout.addWidget(QLabel('Training Period:'))
        main_layout.addWidget(self.training_period_dropdown)
        main_layout.addWidget(QLabel('Data Interval Period:'))
        main_layout.addWidget(self.prediction_period_dropdown)
        main_layout.addWidget(self.ticker_button)

        # Timer for live updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_ticker_button_clicked)

        # Stock recommendation display (BUY, SELL, HOLD)
        self.recommendation_label = QLabel('Recommendation:')
        self.recommendation_label.setStyleSheet('font-size: 24px; color: black;')
        main_layout.addWidget(self.recommendation_label)

        # Graph display
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Log display
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(150)
        main_layout.addWidget(self.log_output)

        # Progress bar for training
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # Indeterminate range
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # display the time left until refresh timer 
        self.timer_label = QLabel('Time until refresh:')
        main_layout.addWidget(self.timer_label)
        
        # Update the timer label every second
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_timer_label)
        self.update_timer.setInterval(1000)  # Update every second (1000 ms)

        # GitHub footer label
        self.github_label = QLabel('<a href="https://github.com/grabercn">Made with ❤️ by Chrismslist</a>')
        self.github_label.setOpenExternalLinks(True)
        main_layout.addWidget(self.github_label)

        # Set layout for main window
        self.setLayout(main_layout)

        # Initialize stock ticker
        self.stock_ticker = ''
        
    def update_timer_label(self):
        # Get the remaining time in milliseconds and convert to seconds
        remaining_time_seconds = self.timer.remainingTime() / 1000  # Convert to seconds
        
        # Convert seconds to hours, minutes, seconds using gmtime
        remaining_time = time.gmtime(max(remaining_time_seconds, 0))  # Ensure it's not negative

        # Format the remaining time as HH:MM:SS
        formatted_time = time.strftime('%H:%M:%S', remaining_time)
        
        # Update the timer label with the formatted time
        self.timer_label.setText(f'Time until refresh: {formatted_time} remaining')

        # Optionally stop the update_timer when the time is up
        if remaining_time_seconds <= 0:
            self.update_timer.stop()

    def on_ticker_button_clicked(self):
        # Get the stock ticker from the text input
        self.stock_ticker = self.ticker_input.text().strip().upper()
        
        # Reset the timer label and stop the timer
        self.timer_label.setText('Time until refresh:')
        self.timer.stop()
        self.update_timer.stop()
        
        if not self.stock_ticker:
            self.log_output.append("Please enter a valid stock ticker.")
            return
        
        ## Check if the stock market is open (between 9:30 AM and 4:00 PM EST) ##
        
        # Define EST timezone
        est = pytz.timezone("US/Eastern")
        current_time = datetime.now(est)

        # Define market open and close times
        market_open_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

        if current_time < market_open_time or current_time >= market_close_time:
            self.log_output.append("The stock market is currently closed. Timer set to next market open.")

            # Calculate time until market opens
            if current_time < market_open_time:
                time_until_open = (market_open_time - current_time).total_seconds()
            else:  # After market close, calculate time until next day's open
                next_day_open = market_open_time + timedelta(days=1)
                time_until_open = (next_day_open - current_time).total_seconds()

            # Convert seconds to milliseconds for the timer
            self.timer.setInterval(int(time_until_open * 1000))
            self.timer.start()
            self.update_timer.start()
            return            

        # Reset the UI
        self.recommendation_label.setText('Recommendation:')
        #self.log_output.clear()

        # Clear the graph
        self.figure.clear()

        # Show progress bar while training
        self.progress_bar.setVisible(True)
        
        # Start training the model
        self.train_and_predict()

    def train_and_predict(self):
        # Fetch historical data for the selected stock and train the model
        data = fetch_and_prepare_data(self.stock_ticker, period=self.training_period_dropdown.currentText(), interval=self.prediction_period_dropdown.currentText())

        # Simulate the training process (this would normally take time)
        # Start the training process in a separate thread
        self.worker = TrainingWorker(data)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.start()

    def on_training_finished(self):
        # After training is complete, hide the progress bar and update the UI
        self.progress_bar.setVisible(False)

        # Make predictions with the newly trained model
        predictions = predict(self.stock_ticker, period=self.training_period_dropdown.currentText(), interval=self.prediction_period_dropdown.currentText())
        
        # Start the countdown timer for the refresh
        # set the timer based on the prediction period dropdown (we need to manually define these values)
        if self.prediction_period_dropdown.currentText() == '1m':
            self.timer.start(60000)
        elif self.prediction_period_dropdown.currentText() == '5m':
            self.timer.start(300000)
        elif self.prediction_period_dropdown.currentText() == '15m':
            self.timer.start(900000)
        elif self.prediction_period_dropdown.currentText() == '30m':
            self.timer.start(1800000)
        elif self.prediction_period_dropdown.currentText() == '60m':
            self.timer.start(360000) # 1 hour
        else:
            self.timer.start(60000)

        # Start the update_timer to update the label
        self.update_timer.start()

        # Update the graph and recommendation label with results
        self.update_graph_and_predictions(predictions)

    def update_stock_data(self):
        # Fetch the latest stock data and update the graph and recommendations
        predictions = predict(self.stock_ticker, period=self.training_period_dropdown.currentText(), interval=self.prediction_period_dropdown.currentText())
        self.update_graph_and_predictions(predictions)

    def update_graph_and_predictions(self, predictions=None):
        # Fetch data and make predictions if needed
        if not predictions:
            predictions = predict(self.stock_ticker, period=self.training_period_dropdown.currentText(), interval=self.prediction_period_dropdown.currentText())
        
        # Fetch historical data for graphing
        data = fetch_and_prepare_data(self.stock_ticker, period=self.training_period_dropdown.currentText(), interval=self.prediction_period_dropdown.currentText())

        # Plot the stock price data on the graph
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(data['Close'], label=f'{self.stock_ticker} Closing Prices', color='blue')
        ax.set_title(f'{self.stock_ticker} Stock Price History')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        self.canvas.draw()

        # Display the latest recommendation
        last_prediction = predictions[-1]
        self.recommendation_label.setText(f'Recommendation: {last_prediction["action"]} ({last_prediction["confidence"]:.2f})')
        self.recommendation_label.setStyleSheet(f'font-size: 24px; color: {"green" if last_prediction["action"] == "BUY" else "red" if last_prediction["action"] == "SELL" else "navy"};')

        # Log the recommendation with time and description
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_output.append(f"{timestamp} | Prediction: {last_prediction['action']} | Confidence: {last_prediction['confidence']:.2f} | Price: ${last_prediction['close_price']:.2f}")

        # Fetch and display news headlines
        #headlines = fetch_news(self.stock_ticker)
        #self.log_output.append("\nTop 5 Headlines:\n" + "\n".join(headlines))

# Worker thread for training model asynchronously
class TrainingWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        # Train the model in the background
        train_model(self.data)
        self.finished.emit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StockPredictionApp()
    window.show()
    sys.exit(app.exec_())
