"""Tests for core.profiles — hardware preset profiles."""

from core.profiles import (
    get_preset_names, get_preset, get_all_profiles,
    apply_profile, save_custom_profile, delete_custom_profile,
    get_current_addon_states, get_active_profile_name, PRESETS,
)


class TestPresets:
    def test_four_presets_exist(self):
        names = get_preset_names()
        assert "Max" in names
        assert "Balanced" in names
        assert "Light" in names
        assert "Minimal" in names

    def test_max_enables_all(self):
        p = get_preset("Max")
        assert all(p["addons"].values()), "Max should enable all addons"

    def test_minimal_disables_all(self):
        p = get_preset("Minimal")
        assert not any(p["addons"].values()), "Minimal should disable all addons"

    def test_balanced_is_subset_of_max(self):
        bal = get_preset("Balanced")["addons"]
        mx = get_preset("Max")["addons"]
        for k, v in bal.items():
            if v:
                assert mx.get(k, False), f"{k} is in Balanced but not Max"

    def test_presets_have_descriptions(self):
        for name in get_preset_names():
            p = get_preset(name)
            assert p["description"], f"{name} missing description"


class TestCustomProfiles:
    def test_save_and_load(self):
        states = {"stocktwits": True, "fear_greed": False}
        ok, _ = save_custom_profile("_test_profile_", "Test", states, 2)
        assert ok

        all_p = get_all_profiles()
        assert "_test_profile_" in all_p
        assert all_p["_test_profile_"]["addons"]["stocktwits"] is True
        assert all_p["_test_profile_"]["scanner_workers"] == 2

        # Cleanup
        delete_custom_profile("_test_profile_")

    def test_cannot_overwrite_builtin(self):
        ok, msg = save_custom_profile("Max", "Evil", {})
        assert not ok
        assert "built-in" in msg.lower()

    def test_delete_nonexistent(self):
        ok, _ = delete_custom_profile("_nonexistent_profile_99_")
        assert not ok

    def test_delete_builtin_blocked(self):
        ok, _ = delete_custom_profile("Balanced")
        assert not ok


class TestApply:
    def test_apply_balanced(self):
        ok, msg = apply_profile("Balanced")
        assert ok
        assert get_active_profile_name() == "Balanced"

    def test_apply_nonexistent(self):
        ok, _ = apply_profile("_does_not_exist_")
        assert not ok
