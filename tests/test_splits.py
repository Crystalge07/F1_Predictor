"""The calibration / held-out split (evaluate/splits.py), as the protocol states it."""

import pandas as pd

from f1predictor.evaluate.splits import describe, is_calibration, is_held_out, split_races


def test_the_protocols_split():
    # docs/evaluation-protocol.md: 2024 + 2026 rounds 1-9 calibrate,
    # 2025 + 2026 rounds 10+ are held out, everything else is neither.
    assert is_calibration(2024, 20) and is_calibration(2026, 9)
    assert is_held_out(2025, 1) and is_held_out(2026, 10)
    for season, rnd in [(2023, 5), (2027, 1), (2018, 1)]:
        assert not is_calibration(season, rnd) and not is_held_out(season, rnd)
    assert not is_held_out(2026, 9) and not is_calibration(2026, 10)
    assert describe() == ("2024 + 2026 rounds 1-9", "2025 + 2026 rounds 10+")


def test_split_races_matches_the_scripts_old_inline_rule():
    # The rule the evaluation scripts used to type out themselves.
    races = pd.DataFrame([(i, season, rnd) for i, (season, rnd) in enumerate(
        [(s, r) for s in (2023, 2024, 2025, 2026, 2027) for r in range(1, 25)])],
        columns=["session_id", "season", "round"])
    early_2026 = (races.season == 2026) & (races["round"] <= 9)
    old_calibration = races[(races.season == 2024) | early_2026].session_id.tolist()
    old_held_out = races[(races.season == 2025) | ((races.season == 2026) & ~early_2026)].session_id.tolist()
    # Rows repeat once per driver in the feature table: duplicates are dropped.
    assert split_races(pd.concat([races, races])) == (old_calibration, old_held_out)
