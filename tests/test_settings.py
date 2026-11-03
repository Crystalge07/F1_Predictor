"""Tests for the tuned-settings file helpers."""

import json

from f1predictor import settings


def test_real_settings_file_has_every_section():
    s = settings.load()
    assert {"weight_2026", "dnf", "probabilities", "frozen"} <= set(s)
    assert set(s["probabilities"]) == {"pre_weekend", "practice", "post_quali"}


def test_save_changes_only_the_given_key(tmp_path):
    # tmp_path is a fresh temporary folder from pytest, so the real file is untouched.
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"a": 1, "nested": {"x": 1, "y": 2}}))
    settings.save(["nested", "x"], 10, path)
    settings.save(["new", "deep"], "v", path)
    assert settings.load(path) == {"a": 1, "nested": {"x": 10, "y": 2}, "new": {"deep": "v"}}


def test_frozen_settings_refuse_changes(tmp_path):
    import pytest
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"frozen": {"date": "2026-11-03"}, "a": 1}))
    settings.save(["a"], 1, path)                     # same value: allowed, no-op
    with pytest.raises(settings.SettingsFrozenError, match="frozen"):
        settings.save(["a"], 2, path)
    assert settings.load(path)["a"] == 1              # unchanged
