"""Tests for the tuned-settings file helpers."""

import json

from f1predictor import settings


def test_real_settings_file_has_every_section():
    s = settings.load()
    assert {"season_decay", "dnf", "probabilities"} <= set(s)
    assert set(s["probabilities"]) == {"pre_weekend", "practice", "post_quali"}


def test_save_changes_only_the_given_key(tmp_path):
    # tmp_path is a fresh temporary folder from pytest, so the real file is untouched.
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"a": 1, "nested": {"x": 1, "y": 2}}))
    settings.save(["nested", "x"], 10, path)
    settings.save(["new", "deep"], "v", path)
    assert settings.load(path) == {"a": 1, "nested": {"x": 10, "y": 2}, "new": {"deep": "v"}}
