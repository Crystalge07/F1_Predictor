"""Write requirements.lock: the exact version of every package the project runs on.

Why this exists (architecture review, 2026-11-03):
    pyproject.toml only gives LOWER bounds (xgboost>=2.0, pandas>=2.0), and
    the hourly pipeline installs the project from scratch on every run. The
    model is retrained on every prediction, so the day XGBoost or pandas
    publishes a new release, the "frozen" 2027 model could quietly start
    behaving differently, while model_version (git commit + settings hash)
    would still claim nothing changed.

    requirements.lock pins what the tests passed on. GitHub Actions installs
    with it as a CONSTRAINTS file:

        pip install -e . -c requirements.lock

    A constraints file only fixes the version of a package IF it gets
    installed; it never adds packages. So a Linux-only dependency that a Mac
    doesn't have (e.g. xgboost's nvidia-nccl on Linux) is simply resolved by
    pip as usual, and the lock written on a Mac still works on the runner.

    predict/stage.live_problems also refuses a LIVE run when the installed
    model libraries differ from the lock (f1predictor/environment.py).

How to use it (from the repo root, virtual environment active):

    python scripts/lock_requirements.py          # rewrite requirements.lock
    pytest                                       # the tests must pass on it
    git diff requirements.lock                   # review, then commit both

What goes in: the project's dependencies and the dev tools (pytest...),
and everything THEY depend on, recursively, at the version installed right
now. Not the optional Streamlit page: it isn't part of the pipeline.
"""

import re
import sys
from importlib import metadata
from pathlib import Path

try:
    import tomllib            # Python 3.11+
except ModuleNotFoundError:   # Python 3.10: the same parser, as a package (pytest needs it anyway)
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "requirements.lock"
EXTRAS = ["dev"]   # optional-dependency groups to include


def canonical(name: str) -> str:
    """Package names compare case-insensitively, with -, _ and . treated alike."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    """'pandas>=2.0 ; python_version >= "3.9"' -> 'pandas'."""
    return canonical(re.split(r"[\s;\[<>=!~(]", requirement.strip(), maxsplit=1)[0])


def applies_here(requirement: str, extras: set[str]) -> bool:
    """Does this requirement apply on THIS machine (its environment marker)?

    A requirement can carry a marker, e.g. 'nvidia-nccl-cu12 ; platform_system == "Linux"'
    or 'pytest ; extra == "dev"'. We evaluate the extra part ourselves and
    leave platform markers to the fact that the package is (or isn't)
    installed: if it isn't installed here, it isn't locked.
    """
    if ";" not in requirement:
        return True
    marker = requirement.split(";", 1)[1]
    extra = re.search(r"extra\s*==\s*['\"]([^'\"]+)['\"]", marker)
    return extra is None or extra.group(1) in extras


def project_requirements() -> list[str]:
    """The dependencies listed in pyproject.toml (plus the EXTRAS groups).

    Read from the file itself, not from the installed package's metadata:
    an editable install keeps the metadata from when it was installed, so a
    dependency added to pyproject.toml since then would be missed.
    """
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = list(project.get("dependencies", []))
    for extra in EXTRAS:
        requirements += project.get("optional-dependencies", {}).get(extra, [])
    return requirements


def closure() -> dict[str, str]:
    """{package name: installed version} for the project and everything it pulls in."""
    todo = project_requirements()
    found: dict[str, str] = {}
    while todo:
        name = requirement_name(todo.pop())
        if name in found:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue   # a platform-specific dependency not installed on this machine
        found[name] = dist.version
        # A dependency's own optional extras aren't installed, so only its
        # unconditional (or platform) requirements count.
        todo += [r for r in (dist.requires or []) if applies_here(r, set())]
    return found


def main() -> None:
    pins = closure()
    lines = [
        "# Exact versions the project is tested on. Written by scripts/lock_requirements.py;",
        "# don't edit by hand. Used as a pip CONSTRAINTS file:",
        "#     pip install -e . -c requirements.lock",
        f"# Python {sys.version_info.major}.{sys.version_info.minor} (the workflow's version).",
    ]
    lines += [f"{name}=={version}" for name, version in sorted(pins.items())]
    LOCK_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(pins)} pins to {LOCK_PATH.name}")


if __name__ == "__main__":
    main()
