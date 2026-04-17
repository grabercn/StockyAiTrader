"""
Addon: FinBERT-Tone (Analyst Report Sentiment)

A variant of FinBERT fine-tuned specifically on financial analyst reports
and earnings call transcripts. More nuanced than the base ProsusAI/finbert
at distinguishing tone in formal financial writing.

Use case: Provides a second sentiment signal that can be ensembled with
the core FinBERT. When both agree, confidence is higher.

Setup:
    Model downloads automatically on first use (~420MB).
    No API key needed — runs locally on CPU.
"""

import numpy as np

ADDON_NAME = "FinBERT-Tone (Analyst)"
ADDON_DESCRIPTION = "Analyst report sentiment model (420MB, runs on CPU)"
ADDON_FEATURES = ["finbert_tone"]
DEPENDENCIES = ["transformers", "torch"]
REQUIRES_API_KEY = False
API_KEY_NAME = ""

_pipeline = None
_MODEL_ID = "yiyanghkust/finbert-tone"


def check_available():
    try:
        import transformers
        import torch
        return True, "Ready (model downloads on first use, ~420MB)"
    except ImportError:
        return False, "Install: pip install transformers torch"


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model=_MODEL_ID,
            tokenizer=_MODEL_ID,
            device=-1,
        )
    return _pipeline


def get_features(ticker, data):
    """
    Analyze news headlines with FinBERT-Tone.

    Returns:
        finbert_tone: -1.0 (negative) to +1.0 (positive)
    """
    try:
        from core.sentiment import fetch_news
        headlines = fetch_news(ticker, max_headlines=10)

        if not headlines:
            return {"finbert_tone": 0.0}

        pipe = _get_pipeline()
        results = pipe(headlines, truncation=True, max_length=512)

        total = 0.0
        for r in results:
            label = r["label"].lower()
            if label == "positive":
                total += r["score"]
            elif label == "negative":
                total -= r["score"]

        avg = total / len(results) if results else 0.0
        return {"finbert_tone": float(avg)}

    except Exception as e:
        print(f"FinBERT-Tone addon error: {e}")
        return {"finbert_tone": 0.0}
