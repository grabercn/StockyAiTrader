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
        "title": "Phase 1: Gather Context + Regime Detection",
        "text": (
            "At the start of each cycle, the agent takes a single snapshot of your account:\n\n"
            "- Portfolio value, cash, and equity\n"
            "- Buying power AND day-trading buying power (uses the lower to avoid rejections)\n"
            "- All current positions with P&L, entry price, and current price\n\n"
            "Then it detects the current MARKET REGIME using addon data:\n"
            "- CNN Fear & Greed Index (0-100)\n"
            "- SPY direction and correlation\n"
            "- VIX level (if FRED key is set)\n\n"
            "Regimes: RISK_ON (normal trading), CAUTIOUS (reduce sizing), "
            "RISK_OFF (minimal trading), VOLATILE (only highest conviction).\n"
            "Each regime adjusts: position size, confidence threshold, stop/target distances, "
            "and cycle frequency.\n\n"
            "Finally, it runs POST-TRADE REFLECTION on held positions:\n"
            "- Checks for positions with significant losses (>3%)\n"
            "- Extracts verbal rules about what went wrong\n"
            "- Rules are injected into Gemini prompts to prevent repeating mistakes\n"
            "- Rules expire after 7 days (max 10 active)"
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
            "2. Extract 23 technical features (RSI, MACD, VWAP, Bollinger, etc.)\n"
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
        "phase4_filter", "phase5_gemini", "phase6_execute", "phase7_timing",
        "regime_detection", "reflection_system", "rl_system", "risk_management",
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
