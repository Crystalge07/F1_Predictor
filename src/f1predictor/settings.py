"""Tuned settings: every value an evaluation script chose, in one file.

Before this, chosen values (the 2026 weight, DNF shrinkage, simulator noise...)
were typed into the code by hand, sometimes in two places (the dashboard had
its own copy of the 2026 weight). Now:

    * the evaluation scripts WRITE what they choose into tuned_settings.json
      (via save()), and
    * the code READS it from there (via load()).

The file is in git, so `git diff` shows exactly which values a re-tune changed,
and every part of the project always uses the same numbers.

Why JSON rather than a Python file? Scripts can rewrite it safely with the
standard library, and it can't contain logic, only values.
"""

import json
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path(__file__).resolve().parent / "tuned_settings.json"


def load(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    """All tuned settings as a dict, e.g. load()["weight_2026"] -> 3.0."""
    return json.loads(path.read_text(encoding="utf-8"))


class SettingsFrozenError(RuntimeError):
    """Raised when something tries to change frozen settings."""


def save(keys: list[str], value: Any, path: Path = SETTINGS_PATH) -> None:
    """Set one (possibly nested) setting and write the file back.

    Refuses (SettingsFrozenError) while the file has a "frozen" entry, unless
    the value is unchanged. That stops a re-run tuning script from quietly
    changing a frozen model (docs/evaluation-protocol.md). Re-running a script
    that picks the same value is fine.

    keys is the path into the JSON, e.g.
        save(["weight_2026"], 3.0)
        save(["probabilities", "practice"], {"anchor_column": ..., ...})
    Everything else in the file is left as it was.
    """
    data = load(path)
    if "frozen" in data:
        current = data
        for key in keys:
            current = current.get(key) if isinstance(current, dict) else None
        if current != value:
            raise SettingsFrozenError(
                f"tuned_settings.json is frozen ({data['frozen'].get('date')}); not changing "
                f"{'.'.join(keys)} from {current!r} to {value!r}. See docs/evaluation-protocol.md.")
        return
    # Walk down to the dict that holds the last key, creating levels if needed.
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    # indent=2 keeps the file readable and the git diffs small.
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
