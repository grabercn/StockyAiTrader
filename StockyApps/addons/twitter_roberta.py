"""
Addon: Twitter-RoBERTa Social Sentiment

Uses a RoBERTa model trained on ~124M tweets for sentiment analysis.
Much better at parsing informal social media text (slang, emojis, abbreviations)
than VADER or generic BERT models.

Useful for analyzing StockTwits messages, Reddit posts, and Twitter cashtags
where text like "AAPL to the moon 🚀🚀" needs to be understood.

Setup:
    Model downloads automatically on first use (~500MB).
    No API key needed — runs locally on CPU.
    Manage via Model Manager in StockyHub.
"""

import numpy as np

ADDON_NAME = "Twitter-RoBERTa Sentiment"
ADDON_DESCRIPTION = "Social media sentiment model (500MB, runs on CPU)"
ADDON_FEATURES = ["social_sentiment"]
DEPENDENCIES = ["transformers", "torch"]
REQUIRES_API_KEY = False
API_KEY_NAME = ""

# Lazy-loaded model
_pipeline = None
_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def check_available():
    """Check if transformers and torch are installed."""
    try:
        import transformers
        import torch
        return True, "Ready (model downloads on first use, ~500MB)"
    except ImportError:
        return False, "Install: pip install transformers torch"


def _get_pipeline():
    """Lazy-load the RoBERTa pipeline. Downloads model on first call."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model=_MODEL_ID,
            tokenizer=_MODEL_ID,
            device=-1,  # CPU only
            top_k=None,  # Return all class scores
        )
    return _pipeline


def get_features(ticker, data):
    """
    Analyze StockTwits messages with Twitter-RoBERTa for social sentiment.

    Falls back to analyzing news headlines if StockTwits isn't available.

    Returns:
        social_sentiment: -1.0 (bearish) to +1.0 (bullish)
    """
    try:
        # Try to get StockTwits messages first (raw text, not pre-scored)
        texts = _fetch_social_texts(ticker)

        if not texts:
            # Fallback: use news headlines from core sentiment module
            from core.sentiment import fetch_news
            texts = fetch_news(ticker, max_headlines=10)

        if not texts:
            return {"social_sentiment": 0.0}

        pipe = _get_pipeline()
        results = pipe(texts[:15], truncation=True, max_length=128)

        # Each result is a list of dicts: [{"label": "positive", "score": 0.9}, ...]
        total_score = 0.0
        count = 0
        for result in results:
            # result is a list of label scores
            scores = {r["label"]: r["score"] for r in result}
            # Map labels to numeric: positive=+1, neutral=0, negative=-1
            sentiment = (
                scores.get("positive", 0) * 1.0
                + scores.get("neutral", 0) * 0.0
                + scores.get("negative", 0) * -1.0
            )
            total_score += sentiment
            count += 1

        avg = total_score / count if count > 0 else 0.0
        return {"social_sentiment": float(avg)}

    except Exception as e:
        print(f"Twitter-RoBERTa addon error: {e}")
        return {"social_sentiment": 0.0}


def _fetch_social_texts(ticker):
    """Fetch raw message text from StockTwits for RoBERTa analysis."""
    try:
        import requests
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        resp = requests.get(url, timeout=5, headers={"User-Agent": "StockyAiTrader/2.0"})
        if resp.status_code == 200:
            messages = resp.json().get("messages", [])
            return [m.get("body", "") for m in messages if m.get("body")]
    except Exception:
        pass
    return []
