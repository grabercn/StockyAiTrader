# -*- coding: utf-8 -*-
"""
Autonomous Trading Agent — modular pipeline engine.

Architecture:
    AgentEngine    — 7-phase trading loop (context → discover → scan → filter → sell → buy → wait)
    TradeExecutor  — Buy/sell/rotate execution with bracket fallback
    PIPELINE_INFO  — Human-readable pipeline descriptions for transparency

Usage:
    from core.agent import AgentEngine

    engine = AgentEngine(broker, log_fn=bus.log_entry.emit, settings_fn=load_settings)
    engine.on_tray_update = my_callback
    engine.agent_stocks = saved_stocks
    engine.start()
    ...
    engine.stop()
"""

from .engine import AgentEngine
from .info import PIPELINE_INFO, get_info_html
from .regime import RegimeState, detect_regime
from .reflection import get_active_rules, get_rules_for_prompt

__all__ = [
    "AgentEngine", "PIPELINE_INFO", "get_info_html",
    "RegimeState", "detect_regime",
    "get_active_rules", "get_rules_for_prompt",
]
