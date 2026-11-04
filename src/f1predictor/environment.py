"""Which library versions a prediction ran on, and whether they're the locked ones.

The model is retrained for every prediction, so the libraries are part of
the model: a new XGBoost release can change the trees it grows, a new
pandas the way a rolling average treats a gap. Code (git commit) and
settings (hash) were already recorded with every run; this adds the third
ingredient (architecture review, 2026-11-03):

    * library_versions(): the installed versions of the MODEL_LIBRARIES,
      appended to model_version (predict/stage.py), so every logged run
      says exactly what it was computed with;
    * lock_problems(): differences between those and requirements.lock
      (written by scripts/lock_requirements.py). A LIVE run is refused while
      there are any (predict/stage.live_problems), the same way it is
      refused from uncommitted code.

The GitHub Actions workflow installs with `-c requirements.lock`, so live
runs get exactly the locked versions and this check passes. It catches
the cases where that didn't happen (the lock was regenerated but the
install wasn't, someone runs a live prediction from an upgraded laptop...).
"""

from importlib import metadata
from pathlib import Path

from f1predictor import config

LOCK_PATH: Path = config.PROJECT_ROOT / "requirements.lock"

# The libraries whose version can change what the model predicts. (requests,
# pytest & co. can't, so a difference there doesn't block a live run; the
# workflow installs them from the lock anyway.)
MODEL_LIBRARIES = ("xgboost", "pandas", "numpy", "scikit-learn", "scipy")


def library_versions() -> dict[str, str]:
    """{library: installed version} for MODEL_LIBRARIES ('missing' if not installed)."""
    versions = {}
    for name in MODEL_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def locked_versions(path: Path = LOCK_PATH) -> dict[str, str]:
    """{package: version} from requirements.lock (lines like 'pandas==2.3.3')."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()
    return pins


def lock_problems(path: Path = LOCK_PATH) -> list[str]:
    """Model libraries whose installed version differs from the lock (empty = all match)."""
    if not path.exists():
        return [f"no {path.name}: can't tell whether the model libraries are the tested ones"]
    locked = locked_versions(path)
    return [f"{name} {installed} installed, {locked.get(name, 'not')} locked"
            for name, installed in library_versions().items()
            if locked.get(name) != installed]
