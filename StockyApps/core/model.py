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
from sklearn.metrics import accuracy_score, f1_score


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
    "learning_rate": 0.03,      # Slower learning = better generalization
    "num_leaves": 20,           # Reduced from 31 to prevent overfitting on small data
    "max_depth": 5,             # Shallower trees = less overfitting
    "min_child_samples": 20,    # Increased from 10 — more data per leaf = stabler predictions
    "feature_fraction": 0.7,    # Use 70% of features per tree (more regularization)
    "bagging_fraction": 0.7,    # Use 70% of data per tree
    "bagging_freq": 3,          # More frequent bagging
    "lambda_l1": 0.1,           # L1 regularization (feature selection pressure)
    "lambda_l2": 1.0,           # L2 regularization (smooth weights)
    "min_gain_to_split": 0.01,  # Don't split unless meaningful gain
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

    # Replace NaN/inf in features with 0 (LightGBM handles missing but not inf)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute class weights to handle imbalance (BUY labels are rare)
    # This gives the model more incentive to get BUY predictions right
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    class_weight_map = {}
    for cls, cnt in zip(classes, counts):
        class_weight_map[int(cls)] = total / (n_classes * cnt)

    # Time-series split: always train on past, validate on future
    # 5 folds gives more validation opportunities than 3
    n_splits = min(5, max(2, len(X) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_model = None
    best_score = 0.0

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Apply class weights as sample weights during training
        sample_weights = np.array([class_weight_map.get(int(label), 1.0) for label in y_train])

        train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # Early stopping prevents overfitting — stops when validation loss plateaus
        callbacks = [
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(0),  # Suppress per-iteration output
        ]

        model = lgb.train(
            DEFAULT_PARAMS,
            train_set,
            num_boost_round=500,   # More rounds (early stopping will cut short if needed)
            valid_sets=[val_set],
            callbacks=callbacks,
        )

        preds = model.predict(X_val).argmax(axis=1)
        # Use weighted F1 score instead of accuracy — penalizes poor BUY predictions more
        score = f1_score(y_val, preds, average="weighted", zero_division=0)

        if score > best_score:
            best_score = score
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
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    probs = model.predict(X)
    actions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    return actions, confidences, probs
