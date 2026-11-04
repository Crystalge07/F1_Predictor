"""Tests for the library lock (f1predictor/environment.py, requirements.lock).

The model is retrained on every prediction, so the installed XGBoost /
pandas / NumPy versions are part of the model. These check that the lock
file is read correctly, that a mismatch blocks a live run, and that the
versions end up in model_version.
"""

from datetime import datetime, timezone

from f1predictor import environment
from f1predictor.predict import stage


def test_the_lock_pins_every_model_library():
    pins = environment.locked_versions()
    for name in environment.MODEL_LIBRARIES:
        assert name in pins, f"{name} missing from requirements.lock"


def test_the_installed_model_libraries_are_the_locked_ones():
    # If this fails, the environment the tests run in isn't the locked one:
    # either reinstall with `pip install -e ".[dev]" -c requirements.lock`,
    # or (deliberate upgrade) re-run scripts/lock_requirements.py and commit.
    assert environment.lock_problems() == []


def test_a_mismatch_is_reported(tmp_path):
    lock = tmp_path / "requirements.lock"
    pins = {**environment.library_versions(), "xgboost": "0.0.1"}
    lock.write_text("# a comment\n" + "".join(f"{n}=={v}\n" for n, v in pins.items()))
    problems = environment.lock_problems(lock)
    assert len(problems) == 1 and problems[0].startswith("xgboost ") and "0.0.1 locked" in problems[0]


def test_a_missing_lock_is_a_problem(tmp_path):
    assert environment.lock_problems(tmp_path / "nope.lock") != []


def test_model_version_names_the_library_versions():
    version = stage.model_version()
    for name, installed in environment.library_versions().items():
        assert f"+{name}-{installed}" in version


def test_a_live_run_is_refused_on_unlocked_libraries(monkeypatch):
    # Every other live check passes on this stub; only the libraries differ.
    monkeypatch.setattr(stage, "check_prediction", lambda conn, prediction: [])
    monkeypatch.setattr(stage, "model_version", lambda: "abc1234+settings.00000000")
    monkeypatch.setattr(environment, "lock_problems", lambda path=None: ["xgboost 9.9 installed, 3.2.0 locked"])

    class Prediction:
        as_of_utc = "2099-01-01T00:00:00Z"
        event_id, stage = 1, "post_quali"

    class Conn:
        def execute(self, *args):
            class Rows:
                def fetchone(self):
                    return None
            return Rows()

    problems = stage.live_problems(Conn(), Prediction(), datetime(2027, 1, 1, tzinfo=timezone.utc))
    assert problems == ["library not as locked in requirements.lock: xgboost 9.9 installed, 3.2.0 locked"]
