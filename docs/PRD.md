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

All sources are free; the pipeline copies what it needs into its own database so nothing is re-downloaded each run.

| Source | Used for | Notes |
| --- | --- | --- |
| [FastF1](https://github.com/theOehrly/Fast-F1) (Python) | Lap times, tire compound and age, stints, pit stops, track status, weather, telemetry | Main source; data appears after each session, so fetches need retries |
| [OpenF1](https://github.com/br-g/openf1) | Backup for laps, stints, pit, race control | Historical data free; real-time access is paid, so not used in v1 |
| Jolpica-F1 | Results, standings, long-term history | Ergast-compatible replacement; can lag a few hours after a race |
| OpenWeatherMap | Race-day forecast for pre-weekend predictions | Free tier is enough |

**Storage:** start with SQLite or Postgres. Core tables: sessions, laps, stints, results, features (one row per driver per race per stage), predictions (logged with a timestamp and model version), and scores.

## Features and model

One gradient-boosted model (XGBoost or LightGBM) trained on every past race, with a simulation step on top that turns predictions into probabilities.

**Features** (one row per driver per race per stage; only data available at that stage is used)

| Feature | Available from |
| --- | --- |
| Grid position | After qualifying |
| Quali gap to pole (%) | After qualifying |
| FP long-run pace, fuel-corrected | After FP2 |
| FP short-run pace | After FP1 |
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
- **Driver or team swaps mid-season:** features keyed by driver and car separately
- **Sprint weekends:** different session schedule; read it from the calendar, never hard-code
- **Wet races, red flags, safety cars:** wider uncertainty in the simulation; weather as a feature
- **Grid penalties:** use the official starting grid, not raw qualifying order
- **Late or broken data:** retry with backoff, validate row counts, alert and skip instead of crashing
- **Cancelled sessions:** predict with whatever stages exist
- **Regulation resets (like 2026):** weight recent races more heavily than older seasons

## Build phases

Build it during the off-season, backtest on the 2026 season, and go live for the first race of 2027.

1. **Data foundation:** pull FastF1 and Jolpica data into the database; backfill several past seasons; write validation checks.
2. **Baseline + first model:** grid-order baseline, then a post-qualifying XGBoost model with a time-based split. Gate: it beats the baseline.
3. **All stages + probabilities:** add pre-weekend and practice-stage models, the Monte Carlo step, and SHAP explanations.
4. **Backtest:** replay the 2026 season race by race as if live, logging and scoring every prediction. This produces the first resume metric.
5. **Automation:** scheduler, fetch retries, scorer, and retraining, all hands-off. Gate: a full simulated weekend runs with zero manual steps.
6. **Dashboard + launch:** the four pages, deployed, live for the 2027 season.

**Stretch goals**

- Pit strategy simulator as a phase 2 feature that feeds the model
- Discord or X bot that posts updated odds after each session
- LLM-written race previews from the model's top factors
- Live in-race updates using paid real-time data

**Open questions**

- How many past seasons to train on, given the 2026 regulation reset?
- One model per stage, or one model with a stage feature?
- Streamlit to start, or straight to a React dashboard?
