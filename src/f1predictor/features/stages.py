"""Which features each prediction stage may use.

The PRD predicts at several moments in a race weekend. Each stage only gets
what is known at that moment, so each has its own feature list (and its own
model: decision 2026-11-03, "one model per stage", recorded in docs/PRD.md).

    pre-weekend   before FP1: history only (form, team pace, track history,
                  championship standings)
    practice      after practice, before qualifying: + practice pace. On 2024+
                  sprint weekends the sprint runs before qualifying, so its laps
                  count too (practice_pace.py reads that from the calendar).
    post-quali    after qualifying: grid + qualifying gap + history     (Phase 2)

All columns come from the one feature table, build_race_features().
"""

from f1predictor.features.practice_pace import PRACTICE_PACE_COLUMNS
from f1predictor.features.race_features import POST_QUALI_FEATURES

# History features from the post-quali table that are known before the
# weekend starts (everything except grid / pit lane / qualifying gap).
HISTORY_FEATURES = [
    "driver_form_finish",
    "driver_form_grid",
    "team_season_finish",
    "team_season_quali",
    "driver_track_finish",
    "team_track_finish",
]

# Championship features for the models: the per-race RATES and last season's
# total. The running totals (driver_season_points, team_season_points) are
# left out: within a season they carry nearly the same information as the
# rates, and two features saying the same thing made the model split credit
# between them in odd ways (SHAP showed "team points 607: 1.0 places back" for
# a leading team). Dropping them cost ~0.5% on calibration races and tied on
# held-out races (2026-11-03). The totals stay in the table: the
# championship-order baseline sorts by them.
CHAMPIONSHIP_MODEL_FEATURES = [
    "driver_points_per_race",
    "team_points_per_race",
    "prev_season_points",
]

PRE_WEEKEND_FEATURES = HISTORY_FEATURES + CHAMPIONSHIP_MODEL_FEATURES
# Practice pace exists from 2023 only; older rows have NaN there, and XGBoost
# learns to lean on history for those.
PRACTICE_FEATURES = PRE_WEEKEND_FEATURES + PRACTICE_PACE_COLUMNS

STAGE_FEATURES = {
    "pre_weekend": PRE_WEEKEND_FEATURES,
    "practice": PRACTICE_FEATURES,
    "post_quali": POST_QUALI_FEATURES,
}
