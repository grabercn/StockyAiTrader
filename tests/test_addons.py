"""Tests for the addon system — discovery, registry, and feature gathering."""

import numpy as np
import pandas as pd
from addons import (
    get_all_addons, get_active_addons, get_addon_features,
    gather_features, set_addon_enabled, discover_addons, AddonInfo,
)


class TestAddonDiscovery:
    def test_discovers_addons(self):
        addons = get_all_addons()
        assert len(addons) >= 5, "Should discover at least 5 addons"

    def test_addon_has_required_fields(self):
        for addon in get_all_addons():
            assert addon.name, f"Missing name on {addon.module_name}"
            assert addon.description, f"Missing desc on {addon.module_name}"
            assert isinstance(addon.features, list), f"Bad features on {addon.module_name}"
            assert isinstance(addon.dependencies, list)
            assert isinstance(addon.requires_api_key, bool)

    def test_free_addons_available(self):
        """Addons with no dependencies should be auto-available."""
        free = [a for a in get_all_addons() if not a.dependencies]
        for a in free:
            assert a.available, f"{a.name} should be available (no deps)"


class TestAddonToggle:
    def test_enable_disable(self):
        addons = get_all_addons()
        if not addons:
            return
        a = addons[0]
        original = a.enabled
        set_addon_enabled(a.module_name, False)
        assert not [x for x in get_all_addons() if x.module_name == a.module_name][0].enabled
        set_addon_enabled(a.module_name, original)

    def test_active_subset_of_all(self):
        active = get_active_addons()
        all_ = get_all_addons()
        active_names = {a.module_name for a in active}
        all_names = {a.module_name for a in all_}
        assert active_names.issubset(all_names)


class TestFeatureGathering:
    def test_gather_returns_dict(self, dummy_ohlcv):
        result = gather_features("AAPL", dummy_ohlcv)
        assert isinstance(result, dict)

    def test_gather_has_expected_keys(self, dummy_ohlcv):
        result = gather_features("AAPL", dummy_ohlcv)
        expected = get_addon_features()
        for key in expected:
            assert key in result, f"Missing feature: {key}"

    def test_gather_values_numeric(self, dummy_ohlcv):
        result = gather_features("AAPL", dummy_ohlcv)
        for k, v in result.items():
            assert isinstance(v, (int, float, np.floating)), f"{k} is {type(v)}, expected numeric"

    def test_addon_features_extend_core(self):
        from core.data import get_all_features
        from core.features import INTRADAY_FEATURES
        all_f = get_all_features("intraday")
        assert len(all_f) >= len(INTRADAY_FEATURES)
