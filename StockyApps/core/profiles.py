"""
Hardware Preset Profiles — controls which addons and models are active.

Profiles let users balance performance vs accuracy with one click:
- Max:       Everything on. Best accuracy, heaviest. Needs 16GB+ RAM, beefy CPU.
- Balanced:  Default. Core AI + lightweight addons. Good for most laptops.
- Light:     No transformer models, API-only addons. Fast on any hardware.
- Minimal:   Core engine only. No addons at all. Fastest possible.
- Custom:    User-defined. Save/load named profiles.

Each profile is a dict of {addon_module_name: bool} stored in settings.json.
"""

import os
import json
import copy

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "settings.json")


def get_optimal_workers():
    """Auto-detect optimal worker count from system CPU cores.

    Uses physical cores (not logical/hyperthreaded) since LightGBM
    training is CPU-bound. Reserves 2 cores for the UI + OS.
    Clamped to 2-8 range (more than 8 hits yfinance rate limits).
    """
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or 4
    except ImportError:
        physical = os.cpu_count() or 4
    workers = max(2, min(8, physical - 2))
    return workers

# ─── Built-in Presets ────────────────────────────────────────────────────────
# True = enabled, False = disabled.
# Addons not listed inherit their current state.

PRESETS = {
    "Max": {
        "description": "Everything on. Best accuracy, heaviest load. 16GB+ RAM recommended.",
        "scanner_workers": 3,
        "addons": {
            "fear_greed": True,
            "finbert_tone": True,       # ~420MB model, overlaps core FinBERT
            "finnhub_calendar": True,    # Needs API key
            "fred_macro": True,          # Needs API key
            "insider_trades": True,
            "reddit_wsb": True,
            "spy_correlation": True,
            "stocktwits": True,
            "timegpt": True,             # Needs API key
            "twitter_roberta": True,     # ~500MB model
        },
    },
    "Balanced": {
        "description": "Core AI + lightweight addons. Good for most laptops (8GB+ RAM).",
        "scanner_workers": 3,
        "addons": {
            "fear_greed": True,
            "finbert_tone": False,       # Overlaps with core FinBERT — skip
            "finnhub_calendar": True,    # If key is set
            "fred_macro": True,          # If key is set
            "insider_trades": True,
            "reddit_wsb": True,
            "spy_correlation": True,
            "stocktwits": True,
            "timegpt": False,            # API calls add latency
            "twitter_roberta": False,    # Heavy, overlaps StockTwits sentiment
        },
    },
    "Light": {
        "description": "No transformer models. API-only addons. Fast on any hardware.",
        "scanner_workers": 2,
        "addons": {
            "fear_greed": True,
            "finbert_tone": False,
            "finnhub_calendar": False,
            "fred_macro": False,
            "insider_trades": True,
            "reddit_wsb": True,
            "spy_correlation": True,
            "stocktwits": True,
            "timegpt": False,
            "twitter_roberta": False,
        },
    },
    "Minimal": {
        "description": "Core engine only. No addons. Fastest possible.",
        "scanner_workers": 2,
        "addons": {
            "fear_greed": False,
            "finbert_tone": False,
            "finnhub_calendar": False,
            "fred_macro": False,
            "insider_trades": False,
            "reddit_wsb": False,
            "spy_correlation": False,
            "stocktwits": False,
            "timegpt": False,
            "twitter_roberta": False,
        },
    },
}

# Default profile for first-time users
DEFAULT_PROFILE = "Balanced"


# ─── Profile Management ─────────────────────────────────────────────────────

def get_preset_names():
    """List all built-in preset names."""
    return list(PRESETS.keys())


def get_preset(name):
    """Get a built-in preset by name. Returns None if not found."""
    return PRESETS.get(name)


def get_preset_description(name):
    """Get human-readable description for a preset."""
    p = PRESETS.get(name)
    return p["description"] if p else ""


def get_all_profiles():
    """
    Get all profiles: built-in presets + user custom profiles.
    Returns dict of {name: profile_dict}.
    """
    profiles = copy.deepcopy(PRESETS)
    custom = _load_custom_profiles()
    profiles.update(custom)
    return profiles


def get_active_profile_name():
    """Get the name of the currently active profile."""
    settings = _load_settings()
    return settings.get("active_profile", DEFAULT_PROFILE)


def get_active_profile():
    """Get the currently active profile dict."""
    name = get_active_profile_name()
    all_profiles = get_all_profiles()
    return all_profiles.get(name, PRESETS[DEFAULT_PROFILE])


def apply_profile(name):
    """
    Apply a profile — updates addon states and saves to settings.

    Args:
        name: Profile name (built-in or custom)

    Returns:
        (success: bool, message: str)
    """
    all_profiles = get_all_profiles()
    profile = all_profiles.get(name)

    if not profile:
        return False, f"Profile '{name}' not found."

    try:
        from addons import set_addon_enabled

        # Apply addon states
        addon_states = profile.get("addons", {})
        for module_name, enabled in addon_states.items():
            set_addon_enabled(module_name, enabled)

        # Save to settings
        settings = _load_settings()
        settings["active_profile"] = name
        settings["addon_states"] = addon_states
        settings["scanner_workers"] = profile.get("scanner_workers", 3)
        _save_settings(settings)

        return True, f"Profile '{name}' applied."

    except Exception as e:
        return False, f"Failed to apply profile: {e}"


def save_custom_profile(name, description, addon_states, scanner_workers=3):
    """
    Save a user-defined custom profile.

    Args:
        name:           Profile name (must not conflict with built-in names)
        description:    Human-readable description
        addon_states:   Dict of {addon_module_name: bool}
        scanner_workers: Number of concurrent scanner threads
    """
    if name in PRESETS:
        return False, f"Cannot overwrite built-in preset '{name}'."

    profile = {
        "description": description,
        "scanner_workers": scanner_workers,
        "addons": addon_states,
        "custom": True,
    }

    settings = _load_settings()
    custom = settings.get("custom_profiles", {})
    custom[name] = profile
    settings["custom_profiles"] = custom
    _save_settings(settings)

    return True, f"Profile '{name}' saved."


def delete_custom_profile(name):
    """Delete a user-defined custom profile."""
    if name in PRESETS:
        return False, "Cannot delete built-in presets."

    settings = _load_settings()
    custom = settings.get("custom_profiles", {})
    if name in custom:
        del custom[name]
        settings["custom_profiles"] = custom
        _save_settings(settings)
        return True, f"Profile '{name}' deleted."
    return False, f"Profile '{name}' not found."


def get_current_addon_states():
    """Snapshot current addon enabled/disabled states for saving to a profile."""
    try:
        from addons import get_all_addons
        return {a.module_name: a.enabled for a in get_all_addons()}
    except ImportError:
        return {}


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)


def _load_custom_profiles():
    settings = _load_settings()
    return settings.get("custom_profiles", {})
