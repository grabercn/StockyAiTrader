# -*- coding: utf-8 -*-
"""
Animation configuration — centralizes particle counts and enables/disables
animations based on the active hardware profile.

Profiles at Light or Minimal disable all particle animations entirely.
Balanced uses reduced particle counts. Max uses full counts.
"""

import os
import json

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "settings.json")

# Particle counts per profile tier
_COUNTS = {
    "full":    {"reveal": 100, "collapse": 80, "expand": 80, "tray": 60, "dissolve": 90},
    "reduced": {"reveal": 50,  "collapse": 40, "expand": 40, "tray": 30, "dissolve": 45},
}

# FPS targets per tier (timer interval in ms)
_FPS = {
    "full":    16,   # ~60fps
    "reduced": 22,   # ~45fps
}


def _get_profile():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f).get("active_profile", "Balanced")
    except Exception:
        return "Balanced"


def animations_enabled():
    """Return True if particle animations should run (profile >= Balanced)."""
    return _get_profile() not in ("Light", "Minimal")


def get_particle_count(anim_type):
    """Get particle count for an animation type based on profile.
    anim_type: 'reveal', 'collapse', 'expand', 'tray', 'dissolve'
    """
    tier = "full" if _get_profile() == "Max" else "reduced"
    return _COUNTS[tier].get(anim_type, 60)


def get_timer_interval():
    """Get timer interval (ms) based on profile."""
    tier = "full" if _get_profile() == "Max" else "reduced"
    return _FPS[tier]
