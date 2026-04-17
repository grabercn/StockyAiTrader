"""
Addon: TimeGPT Price Forecast (Nixtla)

Uses Nixtla's TimeGPT foundation model to generate a short-term price forecast.
The forecast direction and magnitude are added as features for LightGBM.

TimeGPT is pre-trained on 100B+ time points — it catches patterns your local
model might miss because it's seen far more data.

Setup:
    pip install nixtla
    Get a free API key at https://dashboard.nixtla.io/ (1000 free calls)
    Add to settings.json: "timegpt_api_key": "YOUR_KEY"
"""

import os
import json
import numpy as np

ADDON_NAME = "TimeGPT Forecast"
ADDON_DESCRIPTION = "AI price forecast via Nixtla (free tier: 1000 calls)"
ADDON_FEATURES = ["timegpt_direction", "timegpt_magnitude"]
DEPENDENCIES = ["nixtla"]
REQUIRES_API_KEY = True
API_KEY_NAME = "timegpt_api_key"

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "settings.json")


def check_available():
    """Check if nixtla package is installed."""
    try:
        import nixtla
        return True, "Ready (needs API key in Settings)"
    except ImportError:
        return False, "Install: pip install nixtla"


def _get_api_key():
    """Load TimeGPT API key from settings."""
    try:
        with open(_SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        return settings.get(API_KEY_NAME, "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def get_features(ticker, data):
    """
    Get a short-term price forecast from TimeGPT.

    Returns:
        timegpt_direction: +1 (forecast up), -1 (forecast down), 0 (flat/error)
        timegpt_magnitude: Percent change of forecast vs current price
    """
    api_key = _get_api_key()
    if not api_key:
        return {"timegpt_direction": 0.0, "timegpt_magnitude": 0.0}

    try:
        from nixtla import NixtlaClient
        import pandas as pd

        client = NixtlaClient(api_key=api_key)

        # Prepare price series for TimeGPT
        # It expects a DataFrame with 'ds' (datetime) and 'y' (value) columns
        forecast_input = pd.DataFrame({
            "unique_id": ticker,
            "ds": data.index,
            "y": data["Close"].values,
        })

        # Forecast 5 bars ahead
        forecast = client.forecast(
            df=forecast_input,
            h=5,
            freq=pd.infer_freq(data.index) or "5min",
        )

        if forecast is not None and len(forecast) > 0:
            # Compare forecasted price to current price
            current_price = data["Close"].iloc[-1]
            forecast_price = forecast["TimeGPT"].iloc[-1]

            direction = 1.0 if forecast_price > current_price else -1.0
            magnitude = (forecast_price - current_price) / current_price

            return {
                "timegpt_direction": direction,
                "timegpt_magnitude": float(magnitude),
            }

    except Exception as e:
        print(f"TimeGPT addon error: {e}")

    return {"timegpt_direction": 0.0, "timegpt_magnitude": 0.0}
