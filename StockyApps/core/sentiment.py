"""
Sentiment analysis using VADER (fast, rule-based) and FinBERT (financial-domain transformer).

VADER runs instantly on CPU. FinBERT is loaded lazily on first use (~300MB download once).
Both return scores from -1 (bearish) to +1 (bullish).
"""

import numpy as np
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Singleton instances — loaded once, reused across calls
_vader = SentimentIntensityAnalyzer()
_finbert = None


def _get_finbert():
    """Lazy-load FinBERT so startup isn't blocked by model download."""
    global _finbert
    if _finbert is None:
        from transformers import pipeline
        _finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU only — no GPU required
        )
    return _finbert


def fetch_news(ticker, max_headlines=10):
    """
    Fetch recent news headlines for a stock ticker via Google News RSS.

    Returns a list of headline strings (empty list if none found).
    """
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:max_headlines]]


def score_vader(headlines):
    """
    Average VADER compound score across headlines.
    Returns 0.0 if no headlines.
    """
    if not headlines:
        return 0.0
    scores = [_vader.polarity_scores(h)["compound"] for h in headlines]
    return float(np.mean(scores))


def score_finbert(headlines):
    """
    Average FinBERT score across headlines.
    Positive labels add score, negative labels subtract.
    Returns 0.0 on failure or no headlines.
    """
    if not headlines:
        return 0.0
    try:
        finbert = _get_finbert()
        results = finbert(headlines, truncation=True)
        total = 0.0
        for r in results:
            if r["label"] == "positive":
                total += r["score"]
            elif r["label"] == "negative":
                total -= r["score"]
            # "neutral" contributes 0
        return total / len(results)
    except Exception:
        return 0.0


def compute_sentiment(headlines):
    """
    Compute both VADER and FinBERT sentiment scores for a list of headlines.

    Returns:
        (vader_score, finbert_score) — both in range [-1, 1]
    """
    return score_vader(headlines), score_finbert(headlines)
