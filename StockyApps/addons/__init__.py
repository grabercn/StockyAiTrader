"""
Addon Registry — auto-discovers and manages trading signal addons.

Each addon is a standalone Python file in this directory that follows
a standard interface. The registry:

1. Scans for addon files on import
2. Checks if each addon's dependencies are installed
3. Provides a unified API to gather features from all active addons
4. Lets addons be enabled/disabled without touching core code

ADDON INTERFACE:
    Every addon module must define these attributes/functions:

    ADDON_NAME: str          — Human-readable name (e.g. "TimeGPT Forecast")
    ADDON_DESCRIPTION: str   — One-line description
    ADDON_FEATURES: list     — Feature column names this addon produces
    DEPENDENCIES: list       — pip package names required (e.g. ["nixtla"])
    REQUIRES_API_KEY: bool   — Whether the addon needs an API key
    API_KEY_NAME: str        — Settings key name (e.g. "timegpt_api_key"), or ""

    def check_available() -> (bool, str):
        Check if deps are installed. Returns (available, reason).

    def get_features(ticker: str, data: DataFrame) -> dict:
        Compute features and return {feature_name: value_or_series}.
        Called once per prediction cycle. Should handle errors gracefully.
"""

import os
import importlib
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


@dataclass
class AddonInfo:
    """Metadata about a registered addon."""
    name: str
    description: str
    module_name: str              # Python module name (e.g. "timegpt")
    features: List[str]           # Feature columns it produces
    dependencies: List[str]       # Required pip packages
    requires_api_key: bool
    api_key_name: str
    available: bool = False       # Whether deps are installed
    enabled: bool = True          # Whether user has enabled it
    status: str = "Not checked"   # Human-readable status
    _module: object = None        # Reference to loaded module


# Global addon registry
_addons: Dict[str, AddonInfo] = {}


def discover_addons():
    """
    Scan the addons directory and register all valid addon modules.
    Called once at startup.
    """
    _addons.clear()
    addons_dir = os.path.dirname(__file__)

    for filename in sorted(os.listdir(addons_dir)):
        # Skip __init__.py and non-Python files
        if filename.startswith("_") or not filename.endswith(".py"):
            continue

        module_name = filename[:-3]  # strip .py

        try:
            mod = importlib.import_module(f".{module_name}", package="addons")

            # Validate the addon has the required interface
            required_attrs = [
                "ADDON_NAME", "ADDON_DESCRIPTION", "ADDON_FEATURES",
                "DEPENDENCIES", "REQUIRES_API_KEY", "API_KEY_NAME",
                "check_available", "get_features",
            ]
            missing = [a for a in required_attrs if not hasattr(mod, a)]
            if missing:
                print(f"Addon {module_name}: skipped (missing: {', '.join(missing)})")
                continue

            # Check if dependencies are available
            available, reason = mod.check_available()

            info = AddonInfo(
                name=mod.ADDON_NAME,
                description=mod.ADDON_DESCRIPTION,
                module_name=module_name,
                features=mod.ADDON_FEATURES,
                dependencies=mod.DEPENDENCIES,
                requires_api_key=mod.REQUIRES_API_KEY,
                api_key_name=mod.API_KEY_NAME,
                available=available,
                enabled=available,  # Auto-enable if deps are met
                status=reason,
                _module=mod,
            )
            _addons[module_name] = info

        except Exception as e:
            print(f"Addon {module_name}: failed to load ({e})")


def get_all_addons() -> List[AddonInfo]:
    """Get info about all registered addons."""
    if not _addons:
        discover_addons()
    return list(_addons.values())


def get_active_addons() -> List[AddonInfo]:
    """Get only addons that are available and enabled."""
    return [a for a in get_all_addons() if a.available and a.enabled]


def get_addon_features() -> List[str]:
    """Get combined feature column names from all active addons."""
    features = []
    for addon in get_active_addons():
        features.extend(addon.features)
    return features


def set_addon_enabled(module_name: str, enabled: bool):
    """Enable or disable an addon by module name."""
    if module_name in _addons:
        _addons[module_name].enabled = enabled


def gather_features(ticker: str, data) -> dict:
    """
    Call get_features() on every active addon and merge results.

    Returns a dict of {feature_name: value_or_series}.
    Each addon runs independently — if one fails, others still work.
    """
    all_features = {}

    for addon in get_active_addons():
        try:
            features = addon._module.get_features(ticker, data)
            if isinstance(features, dict):
                all_features.update(features)
        except Exception as e:
            # Addon failure should never crash the main app
            print(f"Addon {addon.name} error: {e}")
            # Fill its features with 0 so the model still works
            for feat in addon.features:
                all_features[feat] = 0.0

    return all_features


# Auto-discover on first import
discover_addons()
