"""
LLM Reasoner — generates human-readable trade explanations using a small CPU model.

Uses distilgpt2 (82M params, ~300MB) which runs reasonably on CPU.
The model generates natural-language reasoning for trade decisions based on
structured input (indicators, signal, confidence, price action).

This reasoning is:
1. Displayed in the scanner detail panel and Deep Analyze popup
2. Saved to JSONL decision logs for future model improvement
3. Shown in Windows toast notifications for quick context

The model is lazy-loaded on first use and cached. It's small enough to keep
in memory alongside LightGBM and FinBERT on a 16GB system.

Required for: Default, Aggressive, YOLO profiles
Optional for: Chill profile (can be disabled in Settings)
"""

import os
from datetime import datetime

# Lazy-loaded model
_generator = None
_MODEL_ID = "distilgpt2"


def _get_generator():
    """Lazy-load the text generation pipeline."""
    global _generator
    if _generator is None:
        try:
            from transformers import pipeline
            _generator = pipeline(
                "text-generation",
                model=_MODEL_ID,
                device=-1,  # CPU only
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256,
            )
        except Exception as e:
            print(f"LLM Reasoner: failed to load {_MODEL_ID}: {e}")
            _generator = "failed"
    return _generator if _generator != "failed" else None


def generate_reasoning(ticker, action, confidence, price, atr, probs,
                       feature_importances=None, extra_context=""):
    """
    Generate a human-readable trade reasoning using the LLM.

    Constructs a prompt from structured trade data and lets the model
    complete it with natural language. Falls back to template-based
    reasoning if the model isn't available.

    Args:
        ticker: Stock symbol
        action: "BUY", "SELL", or "HOLD"
        confidence: 0-1 model confidence
        price: Current price
        atr: Average True Range
        probs: [sell_prob, hold_prob, buy_prob]
        feature_importances: dict of {feature: importance_score}
        extra_context: Additional context string

    Returns:
        Human-readable reasoning string
    """
    # Build structured context
    volatility = "high" if atr / price > 0.02 else ("moderate" if atr / price > 0.01 else "low")
    direction = "bullish" if probs[2] > probs[0] else "bearish"

    top_features = ""
    if feature_importances:
        sorted_feats = sorted(feature_importances.items(), key=lambda x: -x[1])[:5]
        top_features = ", ".join(f"{k}" for k, _ in sorted_feats)

    # Try LLM generation
    gen = _get_generator()
    if gen:
        prompt = (
            f"Stock analysis for {ticker} at ${price:.2f}: "
            f"The AI model recommends {action} with {confidence:.0%} confidence. "
            f"Volatility is {volatility}, trend is {direction}. "
            f"Key factors: {top_features}. "
            f"Reasoning:"
        )
        try:
            result = gen(prompt, max_new_tokens=60, num_return_sequences=1)
            generated = result[0]["generated_text"]
            # Extract only the part after "Reasoning:"
            if "Reasoning:" in generated:
                reasoning = generated.split("Reasoning:")[-1].strip()
            else:
                reasoning = generated[len(prompt):].strip()

            # Clean up — take first complete sentence
            for end in [".", "!", "\n"]:
                if end in reasoning:
                    reasoning = reasoning[:reasoning.index(end) + 1]
                    break
            reasoning = reasoning[:200]  # Cap length

            if len(reasoning) > 10:
                return f"{action} {ticker} @ ${price:.2f} — {reasoning}"
        except Exception:
            pass

    # Fallback: template-based reasoning (always works, no model needed)
    return _template_reasoning(ticker, action, confidence, price, atr, probs,
                                volatility, direction, top_features)


def _template_reasoning(ticker, action, confidence, price, atr, probs,
                         volatility, direction, top_features):
    """
    Template-based reasoning fallback.
    Deterministic, always available, structured but still informative.
    """
    lines = [f"{action} {ticker} @ ${price:.2f} ({confidence:.0%} confidence)"]

    # Probability context
    lines.append(f"SELL {probs[0]:.0%} | HOLD {probs[1]:.0%} | BUY {probs[2]:.0%}")

    # Volatility
    lines.append(f"Volatility: {volatility} (ATR ${atr:.2f}, {atr/price*100:.1f}%)")

    # Direction
    if action == "BUY":
        lines.append(f"Trend: {direction} — model sees upside potential")
    elif action == "SELL":
        lines.append(f"Trend: {direction} — model sees downside risk")
    else:
        lines.append(f"Trend: mixed signals — holding position")

    # Top features
    if top_features:
        lines.append(f"Key factors: {top_features}")

    return " | ".join(lines)


def is_available():
    """Check if the LLM model is downloaded and loadable."""
    try:
        from transformers import AutoModelForCausalLM
        # Just check if model files exist in cache, don't load
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{_MODEL_ID}"
        return cache_dir.exists()
    except Exception:
        return False
