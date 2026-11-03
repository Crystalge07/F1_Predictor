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
places on calibration, so effectively arbitrary), DNF prior strength 80 and
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
