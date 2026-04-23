# -*- coding: utf-8 -*-
"""
Pipeline transparency — human-readable descriptions of every phase.

Used by the info popup to show users exactly how the agent works,
what data it uses, and how decisions are made.
"""

PIPELINE_INFO = {
    "overview": {
        "title": "How the Autonomous Agent Works",
        "text": (
            "The agent runs in continuous cycles. Each cycle follows a 7-phase pipeline "
            "that gathers market data, analyzes it with AI models, and makes trading "
            "decisions. Every decision is logged transparently with full reasoning."
        ),
    },
    "phase1_context": {
        "title": "Phase 1: Context + Regime + Safety Checks",
        "text": (
            "At the start of each cycle, the agent runs 5 sequential checks:\n\n"
            "1. ACCOUNT SNAPSHOT — BP, day-trading BP, all positions with P&L\n"
            "2. FOMC/CPI CHECK — halves position sizes on Fed/CPI announcement days\n"
            "3. REGIME DETECTION — classifies market using Fear & Greed (68=bullish), "
            "VIX (18.9=calm), SPY direction. 4 regimes: RISK_ON (1.2x sizing), "
            "CAUTIOUS (0.7x), RISK_OFF (0.4x), VOLATILE (0.25x)\n"
            "4. POST-TRADE REFLECTION — checks for >3% losses, extracts verbal rules, "
            "injects into Gemini prompts (max 10 rules, 7-day expiry)\n"
            "5. SAFETY SYSTEMS:\n"
            "   - Trailing stops: ratchets SL up at +2/3/4x ATR (locks profit)\n"
            "   - Stop-loss monitor: emergency sells if price hits stored SL\n"
            "   - Zombie cleanup: auto-sells positions down >8% with no stop\n\n"
            "All data reused throughout the cycle — no redundant API calls."
        ),
    },
    "phase2_discovery": {
        "title": "Phase 2: Discover Tickers",
        "text": (
            "The agent gathers tickers from multiple sources:\n\n"
            "- Most Active (10) — highest volume stocks today\n"
            "- Day Gainers (5) — biggest percentage movers\n"
            "- Trending Social (5) — trending on StockTwits/social media\n"
            "- Held Positions — stocks you currently own (always monitored)\n"
            "- Agent-Bought — stocks the agent previously purchased\n\n"
            "Combined, deduplicated, limited to 25 tickers per cycle."
        ),
    },
    "phase3_scan": {
        "title": "Phase 3: AI Scan + RL Adjustment",
        "text": (
            "Each ticker is analyzed in parallel using LightGBM:\n\n"
            "1. Fetch 5 days of 5-minute OHLCV data\n"
            "2. Extract 31 technical features including 7 momentum indicators\n"
            "   (RSI, MACD, VWAP, Bollinger, trend consistency, volume direction, etc.)\n"
            "3. Label historical bars using triple-barrier method (ATR-based)\n"
            "4. Train a per-ticker LightGBM model on this data\n"
            "5. Predict: BUY / SELL / HOLD with probability distribution\n\n"
            "Then the RL feedback model adjusts confidence based on historical trade quality:\n"
            "- Multiplier > 1.0 = this pattern historically profited\n"
            "- Multiplier < 1.0 = this pattern historically lost money\n"
            "- RL retrains every 5 cycles with latest trade data"
        ),
    },
    "phase4_filter": {
        "title": "Phase 4: Smart Pre-Filter",
        "text": (
            "Before sending anything to Gemini (which costs time), the agent filters:\n\n"
            "- SELL signals for stocks NOT held — silently dropped (no position to sell)\n"
            "- BUY signals for untradeable stocks — checked via Alpaca and cached\n"
            "- Errored scan results — removed\n\n"
            "This prevents wasted Gemini API calls and noisy log output."
        ),
    },
    "phase5_gemini": {
        "title": "Phase 5: Gemini AI Advisory",
        "text": (
            "Each actionable signal gets a second opinion from Gemini AI:\n\n"
            "Gemini receives ALL available data:\n"
            "- The AI signal with confidence and probability distribution\n"
            "- Top 10 feature importances (what drove the signal)\n"
            "- Your current position in that stock (qty, entry price, P&L)\n"
            "- Addon data (sentiment, macro indicators, social trends)\n"
            "- Portfolio context: buying power, trades today, positions held\n"
            "- Cycle context: what the agent already decided this cycle\n"
            "- Your trading style (aggressivity profile)\n"
            "- RL quality score for this signal type\n\n"
            "Gemini responds with: recommendation, conviction (1-5), reasoning.\n"
            "Multi-vote system: Gemini weight = conviction / 5 (0.2 to 1.0).\n"
            "If both agree → confidence boosted. If disagree → confidence reduced."
        ),
    },
    "safeguards": {
        "title": "Safety Systems (11 Active)",
        "text": (
            "Protective filters running every cycle:\n\n"
            "PHASE 1 (every cycle):\n"
            "1. Trailing Stop — ratchets SL up at +2/3/4x ATR (locks profit)\n"
            "2. Stop-Loss Monitor — emergency sell if price <= stored SL\n"
            "3. Zombie Cleanup — auto-sells positions >8% loss with no stop\n\n"
            "PHASE 6 (before each BUY):\n"
            "4. Buy Confirmation — requires 2+ consecutive BUY scans\n"
            "5. No Double-Buy — skips stocks already held\n"
            "6. Earnings Avoidance — skips if earnings < 3 days (Finnhub)\n"
            "7. Sector Limit — max 2 per sector (93 stocks, 11 sectors)\n"
            "8. Sentiment Boost — StockTwits + insider data adjusts conf ±5%\n"
            "9. Volume Filter — skips BUY when ATR < 0.3% (dead volume)\n"
            "10. Correlation Filter — blocks buying correlated pairs (18 pairs)\n"
            "11. Loss Cooldown — waits 1 cycle after a losing trade\n\n"
            "POSITION SIZING (6 multipliers):\n"
            "base × regime × confidence × time-of-day × FOMC × Friday\n"
            "- Time-of-day: 1.1x morning, 0.9x lunch, 1.05x afternoon\n"
            "- FOMC/CPI day: 0.5x (halved)\n"
            "- Friday after 2 PM: 0.5x (weekend gap risk)"
        ),
    },
    "phase6_execute": {
        "title": "Phase 6: Execute Trades",
        "text": (
            "Trades execute in smart order:\n\n"
            "A. SELLS FIRST — process all sell signals by confidence (highest first).\n"
            "   This frees capital before buying. Sell quantity based on confidence:\n"
            "   - > 70% → sell entire position\n"
            "   - > 50% → sell half\n"
            "   - Otherwise → sell quarter\n"
            "   Realized P&L tracked per trade (entry vs exit).\n\n"
            "B. BUY CONFIRMATION — requires 2+ consecutive BUY scans before executing.\n"
            "   Backtest showed this improves win rate from 42% to 64%.\n"
            "   First-scan BUY signals are logged as WAIT, not executed.\n\n"
            "C. CONFIDENCE-SCALED SIZING — position size scales with confidence.\n"
            "   50% conf = 1.0x base size, 100% conf = 1.3x.\n"
            "   Combined with regime multiplier (RISK_ON=1.2x, VOLATILE=0.25x).\n\n"
            "D. STOP/TARGET — 1.0x ATR stop-loss, 4.5x ATR take-profit (4.5:1 R:R).\n"
            "   Backtest optimized: wider targets let winners run.\n\n"
            "E. PDT AWARENESS — if day-trading BP is zero, uses GTC limit orders\n"
            "   and skips buying stocks already held (avoids round-trip violations).\n\n"
            "F. CAPITAL ROTATION — if BP is too low for a high-confidence BUY,\n"
            "   sells the weakest position to fund it.\n"
            "   Rotation thresholds: YOLO=40%, Aggressive=60%, Default=80%."
        ),
    },
    "phase7_timing": {
        "title": "Phase 7: Dynamic Cycle Timing",
        "text": (
            "The wait between cycles adapts to market conditions:\n\n"
            "Base: 5 minutes\n"
            "Adjustments:\n"
            "- Active trading (3+ trades) → 50% faster\n"
            "- Strong signals found → 40% faster\n"
            "- Quiet market (no signals) → 50% slower\n"
            "- Aggressivity: YOLO=2x faster, Chill=1.5x slower\n\n"
            "Final range: 1 minute (very active) to 15 minutes (quiet)."
        ),
    },
    "rl_system": {
        "title": "Reinforcement Learning Feedback",
        "text": (
            "The RL model learns from YOUR trading history:\n\n"
            "1. Every decision and trade is logged to JSONL files\n"
            "2. Decisions are matched to their outcomes (price direction after trade)\n"
            "3. A LightGBM classifier learns: which confidence + volatility + action\n"
            "   patterns historically led to profitable trades\n"
            "4. Generates a quality multiplier (0.5x to 1.5x) for each new signal\n"
            "5. Retrains every 5 agent cycles with the latest data\n\n"
            "More trading data = better predictions over time."
        ),
    },
    "regime_detection": {
        "title": "Market Regime Detection",
        "text": (
            "The agent classifies current market conditions every cycle:\n\n"
            "RISK_ON — Bullish, low fear, trending up\n"
            "  Size: 1.2x | Confidence: -5% | Wider stops, bigger targets\n\n"
            "CAUTIOUS — Mixed signals, moderate fear\n"
            "  Size: 0.7x | Confidence: +5% | Standard parameters\n\n"
            "RISK_OFF — Bearish, high fear, VIX elevated\n"
            "  Size: 0.4x | Confidence: +15% | Tighter stops, smaller targets\n\n"
            "VOLATILE — Extreme readings either direction\n"
            "  Size: 0.25x | Confidence: +20% | Wide stops (avoid whipsaws)\n\n"
            "Data sources: Fear & Greed (0-100), SPY returns, VIX level.\n"
            "Regime adjustments stack with your aggressivity profile."
        ),
    },
    "reflection_system": {
        "title": "Post-Trade Reflection",
        "text": (
            "After each cycle, the agent reflects on held positions:\n\n"
            "1. Identifies positions with >3% unrealized loss\n"
            "2. Analyzes the original decision context\n"
            "3. Extracts a verbal rule (e.g., 'high confidence does not guarantee profit')\n"
            "4. Stores rules in settings (max 10, expire after 7 days)\n"
            "5. Injects active rules into Gemini prompts\n\n"
            "This creates a self-improving feedback loop where the agent learns "
            "from its mistakes without manual intervention."
        ),
    },
    "risk_management": {
        "title": "Risk Management",
        "text": (
            "Built-in safeguards:\n\n"
            "- 2% max risk per trade (ATR-based position sizing)\n"
            "- Daily trade limit (set by aggressivity profile)\n"
            "- PDT-aware: uses GTC limit orders when day-trading BP is zero\n"
            "- Bracket orders: 1.0x ATR stop-loss, 4.5x ATR take-profit (4.5:1 R:R)\n"
            "- Concentration cap: max 10% of portfolio in one stock\n"
            "- Confidence scaling: higher confidence = proportionally larger position\n"
            "- Buy confirmation: requires 2+ consecutive BUY signals before executing\n"
            "- Tradability cache persists across cycles (no redundant API checks)\n"
            "- Regime adjustments stack with aggressivity profile\n\n"
            "Backtest results (4 days, 739 decisions):\n"
            "  Best config: 70% conf + confirmation + 1.0x/4.5x + scaling\n"
            "  Return: +9.4% | Win rate: 64% | R:R: 38:1 | 14 trades"
        ),
    },
}


def get_info_html():
    """Generate full HTML for the transparency popup."""
    from core.branding import (
        BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT,
        FONT_FAMILY, TEXT_MUTED, TEXT_SECONDARY,
    )

    sections = [
        "overview", "phase1_context", "phase2_discovery", "phase3_scan",
        "phase4_filter", "phase5_gemini", "safeguards", "phase6_execute",
        "phase7_timing", "regime_detection", "reflection_system",
        "rl_system", "risk_management",
    ]

    html = []
    for key in sections:
        info = PIPELINE_INFO[key]
        title = info["title"]
        text = info["text"].replace("\n", "<br>")
        html.append(
            f'<div style="margin-bottom: 16px;">'
            f'<h3 style="color: {BRAND_PRIMARY}; margin: 0 0 4px 0; '
            f'font-family: {FONT_FAMILY};">{title}</h3>'
            f'<p style="color: {TEXT_SECONDARY}; margin: 0; line-height: 1.5; '
            f'font-size: 11px;">{text}</p>'
            f'</div>'
        )

    return "\n".join(html)
