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
# ─── Available Gemini Models (verified via API) ──────────────────────
# Latest:
#   gemini-3.1-flash-lite-preview  — newest, fast
#   gemini-3-flash-preview         — gen 3 flash
#   gemini-3.1-pro-preview         — gen 3 pro (slower, best quality)
# Stable:
#   gemini-2.5-flash               — stable flash
#   gemini-2.5-pro                 — stable pro
#   gemini-2.0-flash               — older stable
# ─────────────────────────────────────────────────────────────────────
PRIMARY_MODEL = "gemini-3.1-flash-lite-preview"
FALLBACK_MODEL = "gemini-2.5-flash"

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
                 feature_importances=None, portfolio_context="",
                 addon_data=None, position_info=None, rl_quality=None):
    """
    Get Gemini's advisory with FULL data access (all indicators + addons).

    Returns:
        {
            "recommendation": "BUY" | "SELL" | "HOLD",
            "confidence_adjustment": float (-0.3 to +0.3),
            "conviction": int (1-5, how sure Gemini is),
            "reasoning": str,
            "model_used": str,
        }
        or None if unavailable.
    """
    client, _ = _get_client()
    if not client:
        return None

    # Build comprehensive prompt with ALL available data
    sections = []
    sections.append(f"Stock: {ticker} @ ${price:.2f}")
    sections.append(f"AI Model Signal: {signal} ({confidence:.0%} confidence)")
    sections.append(f"Probabilities: SELL {probs[0]:.1%}, HOLD {probs[1]:.1%}, BUY {probs[2]:.1%}")
    sections.append(f"Volatility: ATR ${atr:.2f} ({atr/price*100:.1f}%)")

    if feature_importances:
        sorted_f = sorted(feature_importances.items(), key=lambda x: -x[1])[:10]
        feat_lines = "\n".join(f"  {k}: importance {v:.0f}" for k, v in sorted_f)
        sections.append(f"Feature Importances (what drove the AI signal):\n{feat_lines}")

    if addon_data:
        addon_lines = "\n".join(f"  {k}: {v}" for k, v in list(addon_data.items())[:15])
        sections.append(f"Addon Signals (sentiment, macro, social):\n{addon_lines}")

    if position_info:
        sections.append(f"Current Position: {position_info}")

    if rl_quality:
        sections.append(f"RL Feedback Quality Score: {rl_quality:.2f} (1.0=neutral, >1=historically good, <1=historically bad)")

    if portfolio_context:
        sections.append(f"Portfolio Context: {portfolio_context}")

    # Add truncated trade history + user profile
    try:
        from .reinforcement import get_stats
        from .logger import get_log_entries
        stats = get_stats()
        sections.append(
            f"Trading History: {stats.get('total_decisions',0)} decisions, "
            f"{stats.get('total_executions',0)} executions, "
            f"{stats.get('matched_trades',0)} matched for learning"
        )

        # Last 5 trades (truncated)
        try:
            entries = get_log_entries(limit=20)
            recent_trades = [e for e in entries if e.get("type") == "execution"
                           and e.get("status") not in ("failed", "cancelled")][:5]
            if recent_trades:
                trade_lines = []
                for t in recent_trades:
                    trade_lines.append(
                        f"  {t.get('side','?').upper()} {t.get('ticker','?')} "
                        f"x{t.get('qty',0)} @ ${t.get('fill_price',0) or '?'}"
                    )
                sections.append(f"Recent Trades:\n" + "\n".join(trade_lines))
        except: pass

        # User profile + aggressivity trading style
        try:
            with open(SETTINGS_FILE) as f:
                user_settings = json.load(f)
            aggr = user_settings.get("aggressivity", "Default")
            manage_manual = user_settings.get("manage_manual_stocks", False)

            style_guide = {
                "Chill": "Conservative — protect capital, only trade on very strong signals. "
                         "Rarely rotate positions. Hold winners, cut losers slowly.",
                "Default": "Balanced — trade on solid signals, occasionally rotate capital "
                           "from underperformers to better opportunities based on data.",
                "Aggressive": "Active — actively seek upside, rotate capital from stale positions "
                              "to higher-momentum opportunities. Sell underperformers to fund winners.",
                "YOLO": "Maximum aggression — constantly seek highest-return opportunities. "
                        "Aggressively sell anything not performing to fund the best picks. "
                        "Maximize capital rotation for maximum upside.",
            }

            sections.append(
                f"User Profile: {aggr} aggressivity\n"
                f"Trading Style: {style_guide.get(aggr, style_guide['Default'])}\n"
                f"{'Also managing manually-bought stocks' if manage_manual else 'AI-managed stocks only'}"
            )
        except: pass
    except: pass

    data_block = "\n".join(sections)

    prompt = (
        f"You are a senior quantitative trading advisor. You have access to the full trading "
        f"system data including history, user profile, positions, and all indicators. "
        f"Match your advice to the user's trading style described below. "
        f"Consider capital rotation — should they sell underperformers to fund better picks? "
        f"Cite specific data points.\n\n"
        f"{data_block}\n\n"
        f"Respond in this exact JSON format:\n"
        f'{{"recommendation": "BUY/SELL/HOLD", "confidence_adjustment": -0.3 to 0.3, '
        f'"conviction": 1 to 5, '
        f'"reasoning": "2-3 sentence analysis citing specific data points"}}'
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
    Multi-vote weighted system: Gemini's vote weight scales with conviction.

    Voting weights:
    - AI Model: base weight = 1.0
    - Gemini: weight = conviction / 5 (0.2 to 1.0)
    - Agreement: combined confidence boost
    - Disagreement: weighted reduction (high conviction Gemini = bigger impact)

    Returns: (adjusted_signal, adjusted_confidence)
    """
    if not advisory:
        return signal, confidence

    adj = float(advisory.get("confidence_adjustment", 0))
    adj = max(-0.3, min(0.3, adj))
    conviction = min(5, max(1, int(advisory.get("conviction", 3))))
    gemini_weight = conviction / 5.0  # 0.2 to 1.0

    gemini_rec = advisory.get("recommendation", "HOLD")

    if gemini_rec == signal:
        # Both agree — boost proportional to Gemini's conviction
        boost = abs(adj) * gemini_weight
        new_conf = min(1.0, confidence + boost)
    elif gemini_rec == "HOLD":
        # Gemini says wait — reduce based on conviction
        reduction = 0.05 * gemini_weight
        new_conf = max(0.1, confidence - reduction)
    else:
        # Disagreement — reduce proportional to conviction
        # High conviction Gemini disagreeing = bigger impact
        reduction = (abs(adj) + 0.1) * gemini_weight
        new_conf = max(0.1, confidence - reduction)

    return signal, round(new_conf, 3)
