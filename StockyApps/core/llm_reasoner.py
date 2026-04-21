"""
LLM Reasoner — generates human-readable trade explanations.

Uses TinyLlama-1.1B-Chat (instruction-tuned, 1.1B params, ~2GB) for
structured trade reasoning. Falls back to template if model unavailable.

The model receives a structured prompt with ALL indicator data and
produces a focused 2-3 sentence analysis. Results are cached per
ticker+action combo to avoid redundant inference.

For Deep Analyze: uses the same model but with a much longer prompt
and more detailed output (full paragraph analysis).

Architecture:
    - Lazy-loaded pipeline, cached after first use
    - Template fallback always available (no model needed)
    - Quality filter rejects nonsensical output
    - Thread-safe (can be called from background threads)
"""

import os
from datetime import datetime

# Lazy-loaded model
_generator = None
_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_FALLBACK_MODEL = "distilgpt2"
_cache = {}  # {(ticker, action): (reasoning, timestamp)}


def _get_generator():
    """Lazy-load the text generation pipeline. Tries TinyLlama first, falls back to distilgpt2."""
    global _generator
    if _generator is not None:
        return _generator if _generator != "failed" else None

    # Only load if model is already downloaded (don't auto-download)
    for model_id in [_MODEL_ID, _FALLBACK_MODEL]:
        try:
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
            if not cache_dir.exists():
                continue  # Skip if not downloaded

            from transformers import pipeline
            _generator = pipeline(
                "text-generation",
                model=model_id,
                device=-1,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=2,
            )
            return _generator
        except Exception:
            continue

    _generator = "failed"
    return None


def generate_reasoning(ticker, action, confidence, price, atr, probs,
                       feature_importances=None, extra_context=""):
    """
    Generate a concise trade reasoning (2-3 sentences).

    Uses structured prompt with indicator data for focused output.
    Cached per ticker+action to avoid redundant inference.
    Falls back to template if model unavailable or output is bad.
    """
    import time

    # Check cache (valid for 5 minutes)
    cache_key = (ticker, action)
    if cache_key in _cache:
        cached, ts = _cache[cache_key]
        if time.time() - ts < 300:
            return cached

    # Build context
    volatility = "high" if atr / price > 0.02 else ("moderate" if atr / price > 0.01 else "low")
    direction = "bullish" if probs[2] > probs[0] else "bearish"
    spread = abs(probs[2] - probs[0])
    conviction = "strong" if spread > 0.4 else ("moderate" if spread > 0.2 else "weak")

    top_features = ""
    if feature_importances:
        sorted_feats = sorted(feature_importances.items(), key=lambda x: -x[1])[:5]
        top_features = ", ".join(f"{k}" for k, _ in sorted_feats)

    # Always build template first (reliable base)
    template = _template_reasoning(ticker, action, confidence, price, atr, probs,
                                    volatility, direction, top_features)

    # Try LLM
    gen = _get_generator()
    if gen:
        prompt = _build_scan_prompt(ticker, action, confidence, price, atr, probs,
                                     volatility, direction, conviction, top_features)
        try:
            result = gen(prompt, max_new_tokens=80, num_return_sequences=1)
            raw = result[0]["generated_text"]

            # Extract response after the prompt
            if prompt in raw:
                generated = raw[len(prompt):].strip()
            elif "Response:" in raw:
                generated = raw.split("Response:")[-1].strip()
            elif "Assistant:" in raw:
                generated = raw.split("Assistant:")[-1].strip()
            else:
                generated = raw.strip()

            # Take first 2 sentences max
            sentences = []
            for s in generated.replace("!", ".").split("."):
                s = s.strip()
                if s and len(s) > 10:
                    sentences.append(s + ".")
                if len(sentences) >= 2:
                    break
            generated = " ".join(sentences)

            # Quality gate
            bad = ["?", "I'm not", "I don't", "which is", "however", "sure",
                   "not sure", "I am", "let me", "sorry", "actually"]
            if (len(generated) > 20 and
                not any(b in generated.lower() for b in bad) and
                generated[0].isupper()):
                reasoning = f"{action} {ticker} @ ${price:.2f} — {generated}"
                _cache[cache_key] = (reasoning, time.time())
                return reasoning
        except Exception:
            pass

    _cache[cache_key] = (template, time.time())
    return template


def _build_scan_prompt(ticker, action, confidence, price, atr, probs,
                        volatility, direction, conviction, top_features):
    """Build a structured prompt for scan-level reasoning."""
    return (
        f"<|system|>You are a stock trading analyst. Give a brief 2-sentence analysis.</s>"
        f"<|user|>"
        f"Stock: {ticker} at ${price:.2f}\n"
        f"Signal: {action} ({confidence:.0%} confidence)\n"
        f"Probabilities: SELL {probs[0]:.0%}, HOLD {probs[1]:.0%}, BUY {probs[2]:.0%}\n"
        f"Volatility: {volatility} (ATR ${atr:.2f})\n"
        f"Trend: {direction}, conviction: {conviction}\n"
        f"Key factors: {top_features}\n"
        f"Give a concise 2-sentence trading analysis.</s>"
        f"<|assistant|>"
    )


def generate_deep_analysis(ticker, action, confidence, price, atr, probs,
                            feature_importances=None, position_size=0,
                            stop_loss=0, take_profit=0):
    """
    Generate a comprehensive analysis for Deep Analyze (longer, more detailed).
    This can take 10-30 seconds — intended for on-demand deep dives.
    """
    gen = _get_generator()
    if not gen:
        return _deep_template(ticker, action, confidence, price, atr, probs,
                              feature_importances, position_size, stop_loss, take_profit)

    volatility = "high" if atr / price > 0.02 else ("moderate" if atr / price > 0.01 else "low")
    direction = "bullish" if probs[2] > probs[0] else "bearish"

    top_features = ""
    if feature_importances:
        sorted_feats = sorted(feature_importances.items(), key=lambda x: -x[1])[:8]
        top_features = "\n".join(f"  - {k}: importance {v:.0f}" for k, v in sorted_feats)

    rr = f"1:{(take_profit - price) / (price - stop_loss):.1f}" if stop_loss < price else "N/A"
    max_loss = position_size * abs(price - stop_loss) if stop_loss else 0
    max_gain = position_size * abs(take_profit - price) if take_profit else 0

    prompt = (
        f"<|system|>You are a senior stock analyst writing a detailed trading report. "
        f"Be specific, data-driven, and actionable.</s>"
        f"<|user|>"
        f"Write a detailed analysis for {ticker} at ${price:.2f}.\n\n"
        f"SIGNAL: {action} with {confidence:.0%} confidence\n"
        f"Probabilities: SELL {probs[0]:.1%}, HOLD {probs[1]:.1%}, BUY {probs[2]:.1%}\n"
        f"Volatility: {volatility} (ATR ${atr:.2f}, {atr/price*100:.1f}%)\n"
        f"Direction: {direction}\n"
        f"Position: {position_size} shares (${position_size * price:,.0f})\n"
        f"Stop Loss: ${stop_loss:.2f} | Take Profit: ${take_profit:.2f}\n"
        f"Risk/Reward: {rr} | Max Loss: ${max_loss:,.0f} | Max Gain: ${max_gain:,.0f}\n\n"
        f"Top indicators driving this signal:\n{top_features}\n\n"
        f"Write 3-4 sentences covering: 1) Why this signal, 2) Key risks, "
        f"3) Entry/exit strategy recommendation.</s>"
        f"<|assistant|>"
    )

    try:
        result = gen(prompt, max_new_tokens=200, num_return_sequences=1)
        raw = result[0]["generated_text"]
        if "<|assistant|>" in raw:
            generated = raw.split("<|assistant|>")[-1].strip()
        else:
            generated = raw[len(prompt):].strip()

        # Clean up
        generated = generated.split("<|")[0].strip()  # Remove any trailing tokens
        generated = generated[:500]  # Cap length

        if len(generated) > 50:
            return generated
    except Exception:
        pass

    return _deep_template(ticker, action, confidence, price, atr, probs,
                          feature_importances, position_size, stop_loss, take_profit)


def _template_reasoning(ticker, action, confidence, price, atr, probs,
                         volatility, direction, top_features):
    """Template-based reasoning fallback — always works, structured."""
    lines = [f"{action} {ticker} @ ${price:.2f} ({confidence:.0%} confidence)"]
    lines.append(f"SELL {probs[0]:.0%} | HOLD {probs[1]:.0%} | BUY {probs[2]:.0%}")
    lines.append(f"Volatility: {volatility} (ATR ${atr:.2f}, {atr/price*100:.1f}%)")

    if action == "BUY":
        lines.append(f"Trend: {direction} — model sees upside potential")
    elif action == "SELL":
        lines.append(f"Trend: {direction} — model sees downside risk")
    else:
        lines.append(f"Trend: mixed signals — holding position")

    if top_features:
        lines.append(f"Key factors: {top_features}")

    return " | ".join(lines)


def _deep_template(ticker, action, confidence, price, atr, probs,
                    feature_importances, position_size, stop_loss, take_profit):
    """Detailed template for deep analysis when LLM unavailable."""
    volatility = "high" if atr / price > 0.02 else ("moderate" if atr / price > 0.01 else "low")
    direction = "bullish" if probs[2] > probs[0] else "bearish"

    analysis = (
        f"The AI model signals {action} for {ticker} with {confidence:.0%} confidence. "
        f"The stock shows {volatility} volatility with ATR at {atr/price*100:.1f}% of price. "
        f"The overall trend is {direction} with buy probability at {probs[2]:.0%} vs "
        f"sell probability at {probs[0]:.0%}. "
    )

    if action == "BUY" and take_profit > price:
        gain_pct = (take_profit - price) / price * 100
        analysis += f"Target upside is {gain_pct:.1f}% to ${take_profit:.2f}. "
    elif action == "SELL" and stop_loss > 0:
        analysis += f"Downside protection at ${stop_loss:.2f}. "

    if feature_importances:
        top = list(sorted(feature_importances.items(), key=lambda x: -x[1]))[:3]
        names = ", ".join(k for k, _ in top)
        analysis += f"Key drivers: {names}."

    return analysis


def get_model_id():
    """Return the model ID currently in use."""
    return _MODEL_ID


def is_available():
    """Check if any LLM model is downloaded."""
    try:
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        for model_id in [_MODEL_ID, _FALLBACK_MODEL]:
            model_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
            if model_dir.exists():
                return True
        return False
    except Exception:
        return False
