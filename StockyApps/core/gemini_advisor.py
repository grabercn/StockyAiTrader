"""
Gemini AI Advisor — advisory layer for the autonomous trading agent.

Uses Google Gemini to analyze stock data and provide reasoning that
heavily skews (but doesn't dictate) trading decisions.

Models (easily changeable):
    PRIMARY_MODEL = "gemini-3.1-flash-preview"
    FALLBACK_MODEL = "gemini-2.5-flash"

The advisor receives:
- Stock ticker, price, signal, confidence, probabilities
- Top feature importances
- Current portfolio context

Returns a structured recommendation with reasoning.
All data stays transparent in the activity log.

Experimental — toggle on/off in AI Agent settings.
"""

import json
import os

# ─── Model Configuration (change these to switch models) ──────────────────
PRIMARY_MODEL = "gemini-2.5-flash-preview-05-20"
FALLBACK_MODEL = "gemini-2.0-flash"

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "settings.json")


def _get_client():
    """Get Gemini client with API key from settings."""
    try:
        with open(SETTINGS_FILE) as f:
            settings = json.load(f)
    except Exception:
        return None, None

    key = settings.get("gemini_api_key", "")
    if not key:
        return None, None

    try:
        from google import genai
        client = genai.Client(api_key=key)
        return client, key
    except ImportError:
        return None, None


def is_enabled():
    """Check if Gemini advisor is enabled in settings."""
    try:
        with open(SETTINGS_FILE) as f:
            settings = json.load(f)
        return settings.get("gemini_enabled", False) and settings.get("gemini_api_key", "")
    except Exception:
        return False


def get_advisory(ticker, price, signal, confidence, probs, atr,
                 feature_importances=None, portfolio_context=""):
    """
    Get Gemini's advisory recommendation for a stock.

    Returns:
        {
            "recommendation": "BUY" | "SELL" | "HOLD",
            "confidence_adjustment": float (-0.3 to +0.3),
            "reasoning": str,
            "model_used": str,
        }
        or None if unavailable.
    """
    client, _ = _get_client()
    if not client:
        return None

    # Build prompt
    top_features = ""
    if feature_importances:
        sorted_f = sorted(feature_importances.items(), key=lambda x: -x[1])[:5]
        top_features = ", ".join(f"{k}: {v:.0f}" for k, v in sorted_f)

    prompt = (
        f"You are a stock trading advisor. Analyze this data and give a brief recommendation.\n\n"
        f"Stock: {ticker} @ ${price:.2f}\n"
        f"AI Signal: {signal} ({confidence:.0%} confidence)\n"
        f"Probabilities: SELL {probs[0]:.0%}, HOLD {probs[1]:.0%}, BUY {probs[2]:.0%}\n"
        f"ATR: ${atr:.2f} ({atr/price*100:.1f}% volatility)\n"
        f"Key factors: {top_features}\n"
        f"{portfolio_context}\n\n"
        f"Respond in this exact JSON format:\n"
        f'{{"recommendation": "BUY/SELL/HOLD", "confidence_adjustment": -0.1 to 0.1, '
        f'"reasoning": "2-3 sentence explanation"}}'
    )

    _last_error = None
    for model_id in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
            )
            text = response.text.strip()

            # Parse JSON from response
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)
            result["model_used"] = model_id
            return result
        except Exception as e:
            _last_error = str(e)
            continue

    # Store last error for logging
    get_advisory._last_error = _last_error
    return None


def apply_advisory(signal, confidence, advisory):
    """
    Apply Gemini's advisory to adjust signal and confidence.

    The advisory heavily skews but doesn't dictate:
    - If Gemini agrees: boost confidence by adjustment
    - If Gemini disagrees: reduce confidence (but don't flip signal)
    - Confidence adjustment capped at ±0.3

    Returns: (adjusted_signal, adjusted_confidence)
    """
    if not advisory:
        return signal, confidence

    adj = float(advisory.get("confidence_adjustment", 0))
    adj = max(-0.3, min(0.3, adj))  # Cap adjustment

    gemini_rec = advisory.get("recommendation", "HOLD")

    if gemini_rec == signal:
        # Agreement — boost confidence
        new_conf = min(1.0, confidence + abs(adj))
    elif gemini_rec == "HOLD":
        # Gemini says wait — slightly reduce
        new_conf = max(0.1, confidence - 0.1)
    else:
        # Disagreement — reduce confidence but don't flip
        new_conf = max(0.1, confidence - abs(adj) - 0.1)

    return signal, round(new_conf, 3)
