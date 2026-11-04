# Evaluation protocol

How the model is tested, which races were used for what, and what the numbers
can honestly claim. Frozen on 2026-11-03, before Phase 4.

## The rule

A number only counts as a test result if **nothing was chosen by looking at
it**. Every setting (weights, noise, shrinkage...) is picked on one set of
races and reported on another. Every prediction is walk-forward: the model
for a race is trained only on races before it.

## Which races were used for what (2018–2026)

| Races | Role |
| --- | --- |
| 2018–2023 | Training history only (DNF settings were chosen on 2019–2023) |
| 2024 + 2026 rounds 1–9 | **Calibration**: every tuned setting was chosen here |
| 2025 + 2026 rounds 10+ | **Held-out test**: reported, never used to choose |

## Honest caveat: the held-out races are worn out

The held-out races were never used to *pick* a setting, but they were *looked
at* many times: after each phase, and again for every experiment branch on
2026-11-03 (recency weighting, noise shape, wet races, fuel correction,
feature pruning). Each look was followed by a decision about what to keep, so
the held-out numbers are somewhat optimistic. Several branches were rejected
partly because the held-out races disagreed with the calibration races, which
is itself a use of the test set.

**Consequences:**

1. The 2026 backtest in Phase 4 is **not** an out-of-sample result: 2026
   rounds 1–9 chose the settings, and rounds 10+ were looked at repeatedly.
   It's useful for checking that the pipeline works end to end, and its
   numbers must be labelled *in-sample*.
2. **The 2027 season, predicted live, is the real test.** Predictions are
   frozen before each race (Phase 4/5 prediction log), and no setting changes
   in response to 2027 results until the season is over.

## Frozen settings

`src/f1predictor/tuned_settings.json` is marked `"frozen"`. While it is,
`settings.save()` refuses to overwrite any value, so re-running a tuning
script can't quietly change the model. To change settings deliberately,
remove the `frozen` entry in the same commit as the change, and say why.

Held-out results at freeze time (2025 + 2026 rounds 10+, 34 races; Brier
score improvement over each stage's baseline, positive = better):

| Stage | Baseline | Win | Podium |
| --- | --- | --- | --- |
| Pre-weekend | championship-order slot rates | +7.7% | +2.7% |
| Practice | practice-order slot rates | +5.7% | +16.1% |
| Post-quali | grid-slot rates | −1.6% | +2.6% |

Post-quali finishing order (average places off): 2026 rounds 10+ 2.95 vs grid
3.25 (+9.2%); 2025 3.37 vs grid 3.34 (−0.9%).

Settings at freeze: 2026 weight 8 (every value from 1 to 12 was within 0.08
places on calibration, so effectively arbitrary; it applies to every season
from 2026 on, so during the 2027 test 2026 and 2027 rows both count x8 and
2027 is not weighted above 2026: decision 2026-11-04 in docs/PRD.md), DNF prior strength 80 and
driver weight 0.75, post-quali grid weight 0.9 with noise 2.0, pre-weekend and
practice anchor weight 0.25 with noise 4.0.

## Phase 4: the 2026 replay (in-sample)

`scripts/replay_season.py`, 2026-11-03, rounds 1–19. Each stage predicted
strictly as if live (`features/as_of.py`), logged, then scored. The baseline
uses the same information as the model (the as-of grid, not the official
grid published with the result). Improvement over the baseline, positive =
better:

| Stage | Races | Places off | Spearman | Win Brier | Podium Brier |
| --- | --- | --- | --- | --- | --- |
| Pre-weekend | 19 | +2.8% | −0.1% | −1.9% | +5.6% |
| Practice | 19 | −1.3% | −2.0% | +2.6% | +5.1% |
| Post-quali | 19 | −0.6% | +3.3% | −6.7% | +2.8% |

Rounds 10+ only, post-quali: 2.95 places off vs 3.21 for the as-of grid. The
model number matches the frozen 2.95; the grid is 3.21 instead of 3.25
because it is the as-of grid. Driver lists: the pre-weekend stage missed
6 starters and listed 4 non-starters over the season (round 1 and driver
swaps); practice and post-quali lists were all correct.

Reading: podium probabilities beat the baseline at every stage. Win
probabilities and finishing order are mixed, and after qualifying the grid's
win rates are still better than the model's. None of this is a test result
(rounds 1–9 chose the settings), and no setting was changed in response to it.

## 2026 live rehearsal (rounds 20–23)

From 2026-11-03 the last four races of 2026 are predicted **live**, as a dress
rehearsal for 2027: real predictions, logged before each cutoff and never
edited, with the frozen settings.

They are kept apart from everything else:

* **Not the resume metric.** Four races is a tiny sample, and the settings
  were chosen on 2026 data a few days earlier. The real test stays the 2027
  live season.
* **Own scorecard.** Every summary is per season *and* mode
  (`evaluate/scorer.summary`), and is printed with its label
  (`scorecard_label`): "2026 live REHEARSAL", "2027 live: the real test",
  "2026 backtest (in-sample)". The dashboard shows each as its own table.
  These are never added together.
* **Rounds 1–19 have backtest runs only**, so rounds 20–23 have both a
  backtest replay (if re-run) and live runs. They stay separate by mode.

Live runs made by hand go through `scripts/predict_event.py`. A dry run is
saved first, and `--live` refuses unless the new prediction is identical to
the dry run, the cutoff hasn't passed, the stage has no live run yet and the
code is a clean commit. A bad input can't be logged permanently by accident.

## Experiments tried on 2026-11-03 (separate branches, not merged)

| Branch | Idea | Result |
| --- | --- | --- |
| `model/recency-weighting` | per-season decay instead of a 2026 multiplier | calibration +1%, held-out worse on 3 of 4 measures |
| `model/position-dependent-noise` | calmer noise at the front | calibration chose the old flat noise |
| `model/wet-race-noise` | wider noise when it rains | worse even with a perfect forecast |
| `model/fuel-corrected-long-runs` | fuel-correct practice long runs | no measurable effect |
| `model/drop-overlapping-features` | drop running points totals | tie; cleaner SHAP explanations |

## Bug fixes on 2026-11-04 (branch `fix/model-probe-bugs`)

Edge-case, monotonicity and stability tests (`tests/test_grid_slots.py`,
`test_practice_pace_edges.py`, `test_monotonicity.py`, `test_stability.py`)
found seven bugs. These are corrections, not tuning: `tuned_settings.json`
did not change and stays frozen. The fixes change the model's inputs and the
model itself, so the numbers above are superseded by the table below.

| Bug | Fix |
| --- | --- |
| OpenF1's grid sometimes leaves out pit-lane starters, who then vanished from the post-quali prediction | qualifiers whose car isn't on the grid are added as pit-lane starters (`features/as_of.add_missing_starters`) |
| pit-lane flag was 1 in training (official result) but 0 live (OpenF1 lists most pit-lane starters in slot 20) | no longer a model input; `grid` puts them at the back in both |
| one glitched lap (2026 Miami, 60.4 s) put a whole field ~52% off the pace | laps 7% faster than every other car and the driver's own next-best are dropped (1 of 100,074 laps) |
| a red-flag "lap" (2023 Belgium, 1,631 s) gave a 1,224% short-run gap | a best lap outside 107% of the weekend's typical best gives no value |
| a stint of mostly cool-down laps counted as a long run (52.7%) | clean laps within 107% of the stint's fastest lap, not 105% of its median |
| the practice model could predict a better finish for slower practice pace | XGBoost monotone constraints on grid, qualifying gap and both practice gaps |
| 0.01% of practice pace could move a chance by 16 points | same fix: now 1.9 points at most |

Choices made on the calibration races only: which features get monotone
constraints (grid and pace only; constraining the form and points features
too was worse in total), and keeping the pit-lane flag out (slightly better).
The long-run threshold (107%) was chosen on data quality alone (real long
runs kept, absurd ones removed), without prediction scores.

Held-out, before → after (improvement over each stage's baseline, positive =
better; frozen settings, walk-forward, 34 races). One look, after all choices
were made:

| Stage | Win Brier | Podium Brier | Places off |
| --- | --- | --- | --- |
| Pre-weekend | +7.7% → +7.7% | +2.7% → +2.7% | +6.4% → +6.4% |
| Practice | +5.7% → +9.2% | +16.1% → +17.9% | +2.9% → +4.3% |
| Post-quali | −1.6% → −1.4% | +2.6% → +2.8% | +2.0% → +1.6% |

Pre-weekend is unchanged (it uses none of the changed inputs). The Phase 4
replay runs in the prediction log were made by the old code (their
`model_version` says so) and were not re-run.

## Architecture-review fixes (2026-11-03, branch `fix/architecture-review`)

### The predicted order now follows the odds

`predicted_rank` used to be the position model's raw order, while the win /
podium / points chances come from the simulator, which blends that order
with the stage's anchor (90% grid after qualifying). The two could disagree:
in 5 of 19 post-quali replay runs (3 of 19 practice) the driver ranked 1st
was not the favourite to win. 2026 R2: Russell ranked 1st at 31% to win,
Antonelli 5th at 40% (he was on pole).

From this date `predicted_rank` is the simulator's order: drivers sorted by
the blended score the Monte Carlo starts from (`predict/stage.py`). The
raw model order is still recoverable from the stored `expected_position`
(`predict/stage.model_order`). On the 58 runs logged up to then (in-sample):

| Stage | Ranked 1st = favourite (old → new) | Podium-chance contradictions > 5 pts | Places off (old → new) |
| --- | --- | --- | --- |
| Pre-weekend | 20/20 → 20/20 | 4 → 0 | 3.719 → 3.720 |
| Practice | 16/19 → 19/19 | 2 → 0 | 3.651 → 3.651 |
| Post-quali | 14/19 → 19/19 | 65 → 0 | 3.469 → 3.440 |

**What this changes in scoring.** Places off and Spearman are scored on
`predicted_rank`, so from now on they measure the order the dashboard
shows. Runs already in the log keep the old meaning (the log can't be
edited): the 2026 backtest and the first rehearsal run (run 58, 2026 R20
pre-weekend; at that stage the two orders almost always agree). Every 2027
live run uses the new definition. The held-out "places off" numbers above
were measured on the model's raw order and are not re-stated. The win and
podium probabilities, and so every Brier number, are unchanged.

### Library versions are part of the frozen model

The model is retrained for every prediction, but the pipeline used to
install the newest XGBoost / pandas / NumPy on every hourly run (only lower
bounds in `pyproject.toml`). A library release could have changed the
"frozen" model mid-season while `model_version` still claimed the same
commit and settings. Now:

* `requirements.lock` pins every package to the version the tests passed on
  (`scripts/lock_requirements.py`); the workflow installs with
  `pip install -e . -c requirements.lock`;
* `model_version` ends with the model libraries' versions
  (`+xgboost-3.2.0+pandas-2.3.3+...`, `f1predictor/environment.py`);
* a live run is refused if those differ from the lock, like a run from
  uncommitted code.

Upgrading a library is a deliberate change, made between seasons with the
same care as a settings change.
