"""
LightGBM model for stock action classification (BUY / HOLD / SELL).

Why LightGBM over a neural network:
- Much faster to train on CPU (seconds vs minutes)
- Handles tabular/numerical data natively (no text encoding needed)
- Built-in feature importance for debugging
- Less prone to overfitting on small datasets
- Time-series cross-validation prevents lookahead bias

Models are saved per-ticker so retraining only updates one stock's model.
"""

import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


# ─── Model storage ───────────────────────────────────────────────────────────

def get_model_dir():
    """Path to saved model directory (project_root/models/)."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "models")


def _model_path(ticker, prefix="lgbm"):
    """Full path for a ticker's saved model file."""
    model_dir = get_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{prefix}_{ticker}.txt")


# ─── Training ────────────────────────────────────────────────────────────────

# Default LightGBM hyperparameters tuned for stock classification
DEFAULT_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,             # SELL=0, HOLD=1, BUY=2
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,           # Controls model complexity
    "max_depth": 6,             # Prevents overly deep trees
    "min_child_samples": 10,    # Minimum data per leaf (regularization)
    "feature_fraction": 0.8,    # Use 80% of features per tree (reduces overfitting)
    "bagging_fraction": 0.8,    # Use 80% of data per tree
    "bagging_freq": 5,          # Bagging every 5 iterations
    "verbose": -1,              # Suppress training output
    "n_jobs": -1,               # Use all CPU cores
}


def train_lgbm(data, feature_cols, ticker, prefix="lgbm", min_samples=30):
    """
    Train a LightGBM classifier with time-series cross-validation.

    Uses TimeSeriesSplit to avoid training on future data (critical for finance).
    Keeps the best model across 3 folds based on validation accuracy.

    Args:
        data:         DataFrame with feature columns and 'Label' column
        feature_cols: List of column names to use as model input
        ticker:       Stock ticker (used for saving the model file)
        prefix:       Model filename prefix (e.g. "lgbm" or "lgbm_long")
        min_samples:  Minimum rows needed to train (skip if too little data)

    Returns:
        (model, used_features) or (None, []) if training failed
    """
    # Only use features that actually exist in the data
    available = [c for c in feature_cols if c in data.columns]
    X = data[available].values
    y = data["Label"].values.astype(int)

    if len(X) < min_samples:
        return None, []

    # Time-series split: always train on past, validate on future
    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    best_acc = 0.0

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # Early stopping prevents overfitting — stops when validation loss plateaus
        callbacks = [
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(0),  # Suppress per-iteration output
        ]

        model = lgb.train(
            DEFAULT_PARAMS,
            train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            callbacks=callbacks,
        )

        preds = model.predict(X_val).argmax(axis=1)
        acc = accuracy_score(y_val, preds)

        if acc > best_acc:
            best_acc = acc
            best_model = model

    # Save the best model to disk
    if best_model:
        best_model.save_model(_model_path(ticker, prefix))

    return best_model, available


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_lgbm(model, data, feature_cols):
    """
    Run predictions on data using a trained LightGBM model.

    Returns:
        actions:      array of ints (0=SELL, 1=HOLD, 2=BUY)
        confidences:  array of floats (max probability for predicted class)
        probs:        2D array of shape (n_samples, 3) with class probabilities
    """
    available = [c for c in feature_cols if c in data.columns]
    X = data[available].values
    probs = model.predict(X)
    actions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    return actions, confidences, probs
