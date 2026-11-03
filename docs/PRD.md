# F1 Race Prediction Platform — PRD

Nov 2, 2026 · @crystal

## Overview

A self-updating F1 prediction site: one ML model that works for any Grand Prix, refreshes its predictions after every session, and keeps a public scorecard of how accurate it has been.

Most F1 prediction projects are one script per race, rebuilt by hand each weekend. This one is a system. After each session (FP1, FP2, FP3, qualifying, race) a scheduled job pulls new data, updates predictions, and after the race it scores itself, retrains, and moves on to the next GP with no manual work.

Outputs per race: a predicted finishing order and win, podium and points probabilities for every driver, shown on a web dashboard.

## Goals and success metrics

The headline goal: beat the qualifying-order baseline across a full season, with every prediction logged before the race so the score is honest.

**Goals**

- One model that generalizes to every circuit, no per-race code
- Fully automated weekend pipeline: fetch, predict, score, retrain
- Calibrated probabilities (a driver given 30% should win about 30% of the time)
- A dashboard that explains why each prediction moved

**Non-goals (for now)**

- Pit strategy recommendations (possible phase 2 feature)
- True real-time lap-by-lap predictions during a race (needs paid live data)
- Betting tips or odds comparisons

**Success metrics**

| Metric | What it measures | Target |
| --- | --- | --- |
| Improvement over grid-order baseline | Mean absolute position error vs. predicting the grid as the result | Beat baseline by a clear % over the season |
| Winner hit rate | % of races where the top pick won | Above the pole-sitter's win rate |
| Podium hit rate | % of actual podium finishers in predicted top 3 | Track per stage |
| Calibration (Brier score) | How honest the win/podium % are | Lower than baseline |
| Pipeline reliability | % of sessions processed with no manual fix | Close to 100% |

## Race weekend lifecycle

Every stage runs on its own: the pipeline watches the F1 calendar and fires after each session ends.

1. **Pre-weekend (Mon–Thu):** predict using only history: season form, team pace, driver track history, weather forecast. Lowest confidence.
2. **After FP1, FP2, FP3:** add practice data, especially FP2 long-run pace (teams simulate race stints there). Re-predict and log the update.
3. **Sprint weekends:** sprint qualifying and sprint race results feed in as extra signals; the session schedule differs, so the pipeline reads it from the calendar.
4. **After qualifying:** add grid positions, the strongest single signal. This is the final pre-race prediction.
5. **After the race:** fetch official results, score every logged prediction from this weekend, append the race to the training set, retrain, then start pre-weekend predictions for the next GP.

Each run saves a timestamped snapshot, so the dashboard can show how each driver's odds moved through the weekend.

## Dashboard

Four pages, starting as a simple web dashboard; a bot or other channels can come later.

| Page | What it shows |
| --- | --- |
| Next race | Predicted finishing order; win, podium and points % per driver; a timeline of how odds moved after each session |
| Why | The top factors behind each driver's prediction (e.g. grid position and FP2 pace pushing a driver up), from SHAP values |
| Scorecard | Season accuracy vs. the grid-order baseline, winner and podium hit rates, calibration, accuracy by session stage |
| Past races | Every race's logged predictions at each stage vs. the actual result |

## Data sources

All sources used are free; the pipeline copies what it needs into its own database so nothing is re-downloaded each run.

| Source | Used for | Notes |
| --- | --- | --- |
| Jolpica-F1 | Calendar, race and sprint results, qualifying, drivers, constructors (2018+) | Ergast-compatible replacement; can lag a few hours after a race. Limits: ~4 requests/s, 200/hour |
| [OpenF1](https://github.com/br-g/openf1) | Lap-level data (2023+): laps, stints, pit stops, weather, race control, who drove each session | Main lap-data source. Historical data is free; data during the live window is paid, so we fetch after sessions end. Limits: 3 requests/s, 30/minute. Returns HTTP 404 for "no results" |
| [FastF1](https://github.com/theOehrly/Fast-F1) (Python) | Not used | Since Oct 2026 (v3.9.0), Formula 1's timing APIs require an F1TV Access subscription, so lap data via FastF1 is no longer free. Could backfill 2018–2022 lap data later if that becomes worthwhile |
| OpenWeatherMap | Race-day forecast for pre-weekend predictions | Free tier is enough |

**Coverage consequence:** results and qualifying exist for every season from 2018; lap-level data (and therefore practice-pace features) only from 2023.

**Storage:** SQLite to start, written in standard SQL so it can move to Postgres. Phase 1 tables: circuits, drivers, constructors, events, sessions, results, qualifying, session_entries, laps, stints, pit_stops, weather, race_control, ingestion_log. Later phases add features (one row per driver per race per stage), predictions (logged with a timestamp and model version), and scores.

## Features and model

One gradient-boosted model (XGBoost or LightGBM) trained on every past race, with a simulation step on top that turns predictions into probabilities.

**Features** (one row per driver per race per stage; only data available at that stage is used)

| Feature | Available from |
| --- | --- |
| Grid position | After qualifying |
| Quali gap to pole (%) | After qualifying |
| FP long-run pace, fuel-corrected | After FP2 (history from 2023 only) |
| FP short-run pace | After FP1 (history from 2023 only) |
| Recent form: avg finish, last 3–5 races | Pre-weekend |
| Team pace this season | Pre-weekend |
| Driver and team history at this track | Pre-weekend |
| Track type (street, high-speed, mixed) | Pre-weekend |
| Weather forecast / actual | Pre-weekend |
| DNF rate (driver and team) | Pre-weekend |

**Model options**

- Start: regressor predicting finishing position or gap to winner
- Upgrade: learning-to-rank model (e.g. LightGBM LambdaRank), which predicts an order directly
- Train one model per stage, or one model with a stage feature; compare both

**Probabilities:** run a Monte Carlo simulation (e.g. 10,000 races) that adds noise to each driver's predicted pace and samples DNFs, then count how often each driver wins, podiums or scores points.

**Explainability:** SHAP values per prediction feed the dashboard's Why page.

**Validation:** always split by time (train on earlier races, test on later ones), never randomly, so the model never sees the future.

## Automation and architecture

A scheduled job runs after every session; the whole thing can run for $0 on free tiers.

&#91;embedded content: pipeline architecture · 9 components\]

The prediction log is the key piece: predictions are frozen before the race, then the scorer compares them to results and triggers retraining.

- **Scheduler:** GitHub Actions cron jobs keyed off the F1 calendar; a job fires after each session's scheduled end
- **Fetch + validate:** retry with backoff until data appears; check row counts and drivers before saving
- **Hosting:** dashboard on Vercel or Streamlit Cloud, reading predictions from the database
- **Language:** Python for the pipeline and model; the dashboard can be Python (Streamlit) to start or React later

## Evaluation and scorecard

Every prediction is written to the database with a timestamp before the race starts and is never edited, so the scorecard can't be fudged.

**Baselines to beat**

- Grid order: predict the finishing order equals the starting grid
- Championship order: predict by current standings (for pre-weekend stage)

**Metrics, per stage and season to date:** mean absolute position error, Spearman rank correlation, winner and podium hit rate, Brier score for win and podium probabilities, and % improvement over each baseline.

**Backtest:** before going live, replay the previous season race by race as if in real time, to get a first honest score and tune the model.

## Edge cases

These are what make the project technically hard; each needs explicit handling and a test.

- **Rookies and new teams:** no history, so fall back to team pace and junior-series priors
- **Practice-only drivers (FP1 rookies, debut weekends):** lap data identifies cars by number, and drivers who haven't started a race aren't in Jolpica. Each session has a `session_entries` row per car with a nullable driver id, matched to Jolpica by car number at the same event, then by acronym; teams come from matched teammates
- **Driver or team swaps mid-season:** features keyed by driver and car separately
- **Sprint weekends:** different session schedule; read it from the calendar, never hard-code
- **Wet races, red flags, safety cars:** wider uncertainty in the simulation; weather as a feature
- **Grid penalties:** use the official starting grid, not raw qualifying order
- **Late or broken data:** retry with backoff, validate row counts, alert and skip instead of crashing
- **Cancelled sessions:** predict with whatever stages exist
- **Regulation resets (like 2026):** weight recent races more heavily than older seasons

## Build phases

Build it during the off-season, backtest on the 2026 season, and go live for the first race of 2027.

1. **Data foundation:** pull Jolpica (2018+) and OpenF1 (2023+) data into the database; backfill 2018–2026; write validation checks.
2. **Baseline + first model:** grid-order baseline, then a post-qualifying XGBoost model with a time-based split. Original gate: it beats the baseline. **Outcome (2026-11-02): not met as written; gate redefined** (see Decisions made).
3. **All stages + probabilities:** add pre-weekend and practice-stage models, the Monte Carlo step, and SHAP explanations. **Done 2026-11-03:** probabilities at every stage beat their baselines on held-out races (see Decisions made).
4. **Backtest:** replay the 2026 season race by race as if live, logging and scoring every prediction. This checks the whole pipeline end to end; because 2026 was used for calibration and looked at repeatedly, its numbers are labelled in-sample, and the 2027 live season is the out-of-sample resume metric (see `docs/evaluation-protocol.md`). **Done 2026-11-03:** `scripts/replay_season.py` predicts, logs and scores all 19 races so far at every stage (57 runs); results in `docs/evaluation-protocol.md`.
5. **Automation:** scheduler, fetch retries, scorer, and retraining, all hands-off. Gate: a full simulated weekend runs with zero manual steps. **Gate met 2026-11-03** (`tests/test_simulated_weekend.py`): hourly ticks over a fake 2027 season start (calendar fetch, two race weekends, the next race's pre-weekend) log every stage live before its cutoff, score both races and raise no alerts, using the real planner, loaders, validation, checks and scorer; with FP3 and the starting grid never published, every stage is still logged (fallbacks) and exactly one alert is raised. Scheduler: `.github/workflows/tick.yml`, hourly at :07 UTC, one run at a time; the database is downloaded from and uploaded to the `database` release, and an upload guard refuses any file that is damaged or has lost or changed a live run. Off until the repository variable `PIPELINE_ENABLED` is `true`; switched on 2026-11-03 after two dry runs on GitHub (clean code version, guard passed). **Phase 5 done 2026-11-03.**
6. **Dashboard + launch:** the four pages, deployed, live for the 2027 season.

**Stretch goals**

- Pit strategy simulator as a phase 2 feature that feeds the model
- Discord or X bot that posts updated odds after each session
- LLM-written race previews from the model's top factors
- Live in-race updates using paid real-time data

**Decisions made**

- **Seasons stored:** backfill 2018–2026 (2018 is the earliest season with lap-level timing).
- **Seasons trained on:** all of them, with 2026 (the new regulations) weighted more heavily via per-row sample weights. The exact 2026 multiplier is a hyperparameter, tuned with the time-based validation split in Phase 2. (A per-season recency decay was tried instead on 2026-11-03 and rejected; see `docs/evaluation-protocol.md`.)
- **Lap data source:** OpenF1 (free, 2023+) instead of FastF1 (now needs a paid F1TV subscription).
- **Database access:** raw standard SQL with `sqlite3`, no ORM, kept Postgres-compatible.
- **Telemetry:** not stored; no planned feature needs it.
- **DNFs in the position model:** the model predicts finishing position *given the driver finishes*; retirements are left out of the training target. DNF risk is handled in the Phase 3 Monte Carlo step. Scoring still covers the full grid (retirements included), the same as the baselines.
- **Phase 2 gate, redefined (2026-11-02):** the original gate (post-quali model beats the grid on position MAE) was not met. Walk-forward results: 2026 rounds 10–19 MAE 2.88 vs grid 3.25 (+11%), 2025 season 3.35 vs 3.34 (tie), and worse than the grid on the validation races (2024, 2026 rounds 1–9). Fixes tried on validation only: predicting places gained, regularisation, feature subsets, LambdaRank, and practice long/short-run pace; none clearly beat the grid. Conclusion: right after qualifying, the grid is near the ceiling for ranking finishers with free data, and differences of ~0.1 place over 20–30 races are mostly noise. Phase 2 closes with a model that matches the grid; the bar the project is judged on moves to Phase 3: win/podium probabilities (Brier score) and the pre-weekend stage, both scored against their baselines.
- **2026 sample weight:** 3 in Phase 2, 8 at the 2026-11-03 freeze (tuned on 2026 rounds 1–9; every value from 1 to 12 scores within ~0.08 places, so the exact value is effectively arbitrary).
- **Practice pace:** long-run and short-run pace features exist (2023+) but are not used by the post-quali model, where they didn't help on validation. Kept for the pre-weekend and practice stages in Phase 3.
- **Probabilities (Phase 3, 2026-11-03):** Monte Carlo (10,000 races) from a 75% grid / 25% model-order blend, Gaussian noise of 2.0 places, and per-driver DNF chances (driver + team history, heavily shrunk to the field rate; DNFs are mostly luck). Simulating straight from the model was 1.4% worse than the grid-slot baseline on the held-out races; the blend (chosen on calibration races only) is 0.6% better on win Brier and 4.9% better on podium Brier. Known weakness: still underconfident about clear favourites. *(Phase 3 numbers; superseded at the freeze, where re-tuning chose a 90% grid blend and post-quali is −1.6% / +2.6% vs the grid. See Evaluation protocol below.)*
- **One model per stage (2026-11-03):** each stage (pre-weekend, practice, post-quali) has its own model and feature list (`features/stages.py`), all drawn from one feature table. Simpler to explain and test than one model with a stage feature.
- **Pre-weekend stage (2026-11-03):** history + championship-standings features. Walk-forward over 2024–2026: 3.76 places MAE vs 3.83 for championship order (about 2% better; it wins in 2025 and 2026 and loses in 2024).
- **Practice stage (2026-11-03):** pre-weekend features + practice long/short-run pace (sessions before qualifying, including 2024+ sprints). Walk-forward over 2024–2026: 3.71 places MAE vs 3.76 for the pre-weekend model and 3.83 for championship order. In 2026 alone, plain practice order (3.60) beats every model: after a regulation reset, practice is a better guide than history.
- **Stage probabilities (2026-11-03):** pre-weekend and practice stages go through the same simulator, anchored on their own baseline order (championship order, practice order) instead of the grid, with more noise (4.0 places) because there's more uncertainty before qualifying. Held-out (2025 + 2026 rounds 10+) vs the slot baseline (historical rate for that place in the baseline order): pre-weekend win Brier +9.9%, podium +4.4%; practice win +7.5%, podium +15.6% *(Phase 3 numbers; at the freeze: pre-weekend +7.7% / +2.7%, practice +5.7% / +16.1%)*. Settings in `tuned_settings.json`.
- **Explanations (2026-11-03):** SHAP values from XGBoost's built-in `pred_contribs` (no `shap` dependency). They explain the position model; post-quali probabilities also include the grid blend, noise and DNF chances.
- **Tuned settings in one file (2026-11-03):** every value a tuning script chooses lives in `src/f1predictor/tuned_settings.json`; scripts write it, the code and dashboard read it. No hand-copied constants.
- **Team rebrands (2026-11-03):** Toro Rosso/AlphaTauri/RB, Renault/Alpine, Force India/Racing Point/Aston Martin and Sauber/Alfa/Audi share one lineage (`features/teams.py`) for team history that spans seasons. Accuracy effect within noise; kept as a correctness fix.
- **Experiments not adopted (2026-11-03):** recency decay, position-dependent noise, wet-race noise (worse even with a perfect forecast), fuel-corrected long runs (no effect), dropping overlapping features (tie). Each is on its own branch; results in `docs/evaluation-protocol.md`.
- **Evaluation protocol (2026-11-03):** tuned settings are frozen before Phase 4 (`docs/evaluation-protocol.md`). The held-out races (2025 + 2026 rounds 10+) were looked at many times, so the 2026 backtest is labelled in-sample; the 2027 live season is the real test. Results at freeze, Brier vs each stage's baseline: pre-weekend win +7.7% / podium +2.7%, practice +5.7% / +16.1%, post-quali −1.6% / +2.6%.
- **Features (Phase 2):** computed on the fly into a DataFrame for now; the stored `features` table comes in Phase 4 with the prediction log.
- **Strict as-if-live backtest (Phase 4, 2026-11-03):** the backtest builds each stage's prediction rows only from what existed at that moment: the driver list from the previous race / practice entries / qualifying, and the grid from qualifying plus known penalties, never from the race result. Reason: the feature table starts from race results, and `results.grid` (the official grid after penalties) is only published with the result; in 2023–2026, 22–28% of drivers started somewhere other than their qualifying position. A strict backtest runs the same code the 2027 live season will. Sources (`features/as_of.py`): pre-weekend = starters of the previous race (wrong at every round 1 and ~1 driver in ~25% of other races, the honest cost of predicting before the weekend); practice = entries of the last session before qualifying (equal to the race starters in 88 of 89 races since 2023); post-quali = OpenF1's `/starting_grid`, now stored in a `starting_grid` table (equal to the official grid for ~98% of drivers; misses are late penalty reshuffles at the back), falling back to Jolpica qualifying order when it isn't published. Open for Phase 5: whether OpenF1 publishes the grid before the race starts.
- **Prediction log (Phase 4, 2026-11-03):** tables `prediction_runs` (stage, `backtest`/`live` mode, information cutoff, model version, settings copy), `predictions` (order + probabilities per driver), `prediction_features` (each input value and its SHAP contribution, in long format so every stage fits one table; this is the PRD's features table) and `scores` (each metric, model vs baseline). SQLite triggers make predictions impossible to edit and live runs impossible to delete; backtest runs can be deleted so the replay can be re-run. Scores can be rewritten if the FIA changes a result.
- **Predicting into the log (Phase 4, 2026-11-03):** `predict/stage.predict_stage(conn, event_id, stage, mode)` is the one entry point the backtest and the live pipeline share: as-of rows, the stage's model trained on earlier races only, DNF chances, Monte Carlo with the frozen settings, SHAP, then one transaction into the log (a failed write leaves nothing behind). `predicted_rank` is the position model's own order (what the finishing-order numbers were measured on); the simulator's blended view is kept in `anchor_position` and `mean_position`, so after qualifying the top-ranked driver isn't always the favourite to win. Each run records `model_version` (git commit, `-dirty` if uncommitted, plus a hash of the settings) and `lineup_source` (column added 2026-11-03; `init_db` adds it to older databases).
- **Scorer (Phase 4, 2026-11-03):** `evaluate/scorer.py` writes, per logged run, position MAE, Spearman, winner/podium hits and win/podium Brier, each next to the stage's baseline on the same race. The baseline uses the same information as the model: its order is the run's logged `anchor_position` (as-of grid, championship or practice order), its probabilities are historical slot rates from earlier races. When the predicted drivers differ from the starters, non-starters are dropped and unpredicted starters go to the back with 0% chances (for both model and baseline), so a missed replacement costs the prediction; `lineup_missing` / `lineup_extra` record how often. Re-scoring replaces a run's scores.
- **Database location for automation (Phase 5, 2026-11-03):** the SQLite file lives as a GitHub Release asset. Each scheduled run downloads it, works, and uploads it back; a concurrency lock stops two runs overlapping. Chosen over hosted Postgres (more work, outside service), a local launchd job (only runs while the laptop is on) and committing the file (34 MB binary commits by a bot in the history). The dashboard will read the same file.
- **One idempotent tick (Phase 5, 2026-11-03):** the scheduler runs one job every hour; `pipeline/plan.py` reads the calendar, data and log and returns what's due (fetch a session, fetch Jolpica, predict a stage, score a race). Anything already done isn't planned again, so a failed or skipped tick is caught up by the next one. Stage rules: pre-weekend once the previous race result is in (round 1: within 14 days of FP1); practice once every pre-qualifying session is fetched, or within 2 hours of qualifying with whatever exists; post-quali once qualifying is in and either the starting grid is in or it's within 2 hours of the race. (The last call was 45 minutes until the simulated-weekend test showed an on-the-hour tick could never land in it; 2 hours gives two hourly ticks a chance even if GitHub Actions starts late.) A season with no calendar in the database gets its schedule fetched. A stage whose cutoff passed with no live run is reported as missed and never predicted late. Fetching (`pipeline/fetch.py`) loads only the sessions the planner found ready, through the same loaders as the backfill (retries with backoff per request, validation before commit, failures logged and retried next tick). The starting grid is its own fetch, retried until it appears or the race starts; a session still missing 24 hours after it was ready is reported as `gave_up` and no longer retried. One practice run per weekend for now (the PRD lifecycle's re-run after each practice session can be added later: the log allows repeated runs).
- **Late data vs broken data (Phase 5 test suite, 2026-11-03):** data that isn't there yet (a 404, an empty lap list, a server error) is retried every tick and only alerted if it lasts (24h `gave_up` for OpenF1, 48h `late` for Jolpica). Data that fails validation is never saved (the batch is rolled back) and raises an `invalid` alert in the tick it arrives, once; it's still retried in case the source corrects it. Two checks added for it, calibrated on 2018–2026 so they flag nothing real: a race result missing 2+ drivers who qualified is an error (exactly one is a warning, since a driver can withdraw after qualifying: Stroll, Spain 2025, the only case), and a race or sprint driver with official laps but no OpenF1 laps at all is an error. A practice session missing a driver is normal and saved. The tick now fetches Jolpica before OpenF1, so race laps are validated against the official result on race night. The frozen log also refuses `INSERT OR REPLACE` (recursive triggers) and a second live run for the same race and stage (unique index). A failed prediction write or scoring crash is alerted and retried, never crashes the tick.
- **Matching debutants and nameless cars (Phase 5 test suite, 2026-11-03):** replaying the real 2026 Mexico City weekend (`tests/test_real_weekend_replay.py`, recorded API responses, no network) showed two live-prediction gaps in `ingest/openf1.match_entries`: a debut driver (Hirakawa) wasn't matched until his race result existed, so the post-quali prediction left him out; and a car OpenF1's driver list omitted (Lawson, FP3) had no acronym to match on, so the practice prediction left him out. Two rules added: a car with no acronym takes the same car number's match from another session of the same weekend, and an acronym is matched against this event's Jolpica qualifying before the most-recent-result rule (team from that qualifying row). Re-running the matcher over all 443 stored sessions changes nothing (exact car-number matches win once results exist); only the hours before a result exists are affected. Still by design: a debutant can't be in the pre-weekend prediction (that stage copies the last race's lineup). The practice-stage gap is closed by the next entry.
- **Dashboard in React (2026-11-03):** Phase 6 builds the dashboard in React rather than Streamlit.
- **Dashboard data: static JSON on GitHub Pages (Phase 6, 2026-11-03):** the React dashboard never reads the database. After each tick the workflow exports small JSON files (`scripts/export_dashboard.py`, `export/dashboard.py`: a race index, one file per race with every logged run, factors, result and scores, and the scorecard with its labels) from a read-only connection, and deploys the site (page + data together) to GitHub Pages only when the data's hash or the code's commit differs from the live site. Why not read the database: browsers can't fetch release assets (no CORS), it's 35 MB vs ~70 KB per race, a server would be another always-on thing to sync and maintain, and this way every number is computed once in tested Python and the site can't touch the frozen log. Files are deterministic (same database, same bytes). Chosen over Vercel: no extra account or token, same workflow.
- **Debutants at the practice stage (2026-11-03):** a car that can't be matched to a driver but is still driving in the last session before qualifying is a race driver who has no Jolpica id yet (FP1-only drivers are gone by then). It's predicted under a temporary id, `provisional_<season>_r<round>_car<number>` (`features/provisional.py`): his practice laps count, his team is his teammate's (same OpenF1 team name), and a placeholder `drivers` row (no 3-letter code, so never matchable) is written with the run. After the race the scorer turns it into the real id (the driver who started in that car, else the car's match in another session that weekend). If that can't be done (nobody started in that car, two candidates, or it would duplicate a driver already predicted), the run is left unscored and the tick alerts once (`unresolved_driver:...`), retrying every tick for the planner's 10-day window. A debutant with no known team is left out rather than blocking the whole prediction. Checked on 2023–2026: no unmatched car ever ran in that last session, so history is unchanged (practice pace and every practice field identical). Pre-weekend lineups stay "copy the last race" (decision: no hand-edited entry list).
- **Guardrail: alert when the live model falls behind (Phase 5 test suite, 2026-11-03):** monitoring only, no automatic model swap. After scoring, each tick checks every stage over its last 8 scored live races (all seasons, rehearsal included, backtests never): if the model is more than 15% worse than the stage's baseline on places off or podium Brier, it alerts, once per newest race (`pipeline/drift.py`). Chosen on the 2026 backtest: over 3–5 race stretches ordinary luck already put some stage 13–29% behind; over 8 the worst stretch was +10%, so 15% leaves headroom for live noise while a 40%-worse model still trips it within 3 races. Winner hit, podium hits and win Brier are not watched (too jumpy, or a tiny baseline). Earliest possible alert: around 2027 round 4. Re-check the 15% after ~12 live 2027 races.
- **Times without a timezone are refused (Phase 5 test suite, 2026-11-03):** every time the pipeline handles must carry a timezone; any explicit one is converted to UTC (`pipeline/plan.as_utc` / `parse_utc`), and one with none is an error instead of being read as the machine's local time (e.g. `scripts/tick.py --dry-run --now 2026-11-07T17:20:00` on a Toronto laptop used to mean 22:20 UTC). The planner's "current season" is the UTC year. Neither ever affected the live pipeline (it always passes UTC); `tests/test_scheduler_utc.py` also pins the workflow's hourly :07 cron, the Run workflow button with its dry-run option, the no-overlap queue and the 30-minute limit.
- **2026 live rehearsal (2026-11-03):** rounds 20–23 of 2026 are predicted live as a dress rehearsal before 2027. They're reported as their own scorecard, labelled a rehearsal (small sample, settings chosen on 2026), and never mixed with the 2027 live scorecard or the backtests. Manual live runs need a matching dry run first (`scripts/predict_event.py`). See `docs/evaluation-protocol.md`.
- **The tick (Phase 5, 2026-11-03):** `scripts/tick.py` plans, fetches, predicts, scores and reports. Automated live runs go through the same checks as manual ones (`predict/stage.live_problems`: 18–26 drivers, probabilities add up, at most 4 driver changes vs the last race, cutoff not passed, no live run yet, clean commit); a prediction failing one is not logged, it's reported and retried next tick. Retraining has no separate step: every prediction trains on all earlier races. Each tick is recorded (`pipeline_ticks`, for the reliability metric); alerts (blocked prediction, missed stage, session given up on after 24h, Jolpica data late after 48h) are raised once each; a single failed fetch is not an alert, since data a few minutes late is normal (`pipeline_alerts`), and a new alert makes the tick exit with an error so GitHub Actions emails about it.
- **Drivers without a Jolpica id:** `session_entries` table with nullable driver/team ids (see Edge cases).

**Open questions**

- None right now.
