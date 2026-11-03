# F1 Race Prediction Platform

A self-updating F1 prediction system: one model for every Grand Prix, refreshed after each session, with a public scorecard of its own accuracy. See [docs/PRD.md](docs/PRD.md) for the full spec.

**Status:** Phases 1–4 done. Phase 4 replayed the 2026 season race by race as if live: every stage's prediction is built only from what was known at that moment (including the official starting grid from OpenF1 instead of the race result), written to an append-only prediction log with its inputs and SHAP contributions, and scored after the race against that stage's baseline. Tuned settings are frozen ([docs/evaluation-protocol.md](docs/evaluation-protocol.md)); the 2027 season, predicted live, is the real test. Phase 5 (automation) is in progress: the planner that works out what's due at any moment (fetch a session, predict a stage, score a race) is done; fetching, the tick itself, a simulated-weekend test and the GitHub Actions schedule are next.

## How it works

The model predicts at three moments in a race weekend, each using only what is known at that moment:

| Stage | When | Uses |
| --- | --- | --- |
| Pre-weekend | before FP1 | recent form, team pace, track history, championship standings |
| Practice | after practice, before qualifying | the above + practice long-run and short-run pace |
| Post-quali | after qualifying | starting grid, gap to pole, form and team pace |

Each prediction goes through three pieces:

1. **Position model** (XGBoost, one per stage): where each driver finishes *if they finish*.
2. **DNF model**: each driver's chance of not finishing, from driver and team reliability, strongly pulled towards the field average (DNFs are mostly luck).
3. **Monte Carlo simulation**: 10,000 simulated races from (1) blended with a baseline order (the grid after qualifying), plus random swings and DNF draws, giving each driver's chance to win, podium and score points.

SHAP values explain each prediction ("starting position = 2: 3.8 places forward").

Every result below is **walk-forward**: the model for each race is trained only on earlier races.

## Results

Held-out races (2025 + 2026 rounds 10+, 34 races). Improvement in Brier score over each stage's baseline; positive = better.

| Stage | Baseline | Win | Podium |
| --- | --- | --- | --- |
| Pre-weekend | championship order | +7.7% | +2.7% |
| Practice | practice order | +5.7% | +16.1% |
| Post-quali | starting grid | −1.6% | +2.6% |

After qualifying, the starting grid is very hard to beat. Before qualifying, the models add clear value.

**2026 replay (Phase 4, in-sample).** All 19 races so far, each stage predicted as if live and logged before scoring. Better-than-baseline on the same information:

| Stage | Places off | Win Brier | Podium Brier |
| --- | --- | --- | --- |
| Pre-weekend | +2.8% | −1.9% | +5.6% |
| Practice | −1.3% | +2.6% | +5.1% |
| Post-quali | −0.6% | −6.7% | +2.8% |

Podium chances beat the baseline at every stage; win chances and finishing order are mixed. 2026 chose the settings and was looked at during development, so these numbers aren't a real test.

**Caveat:** these races were looked at many times during development, so the numbers are somewhat optimistic. The 2027 season, predicted live, is the real test. Details in [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

## Data

[Jolpica-F1](https://github.com/jolpica/jolpica-f1) for calendar, results and qualifying (2018+); [OpenF1](https://openf1.org) for laps, stints, pit stops, weather, race control and the official starting grid (2023+). Stored in SQLite, in standard SQL so it can move to Postgres.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv            # create an isolated environment for this project
source .venv/bin/activate        # activate it (run this in every new terminal)
pip install -e ".[dev]"          # install the project + dev tools in editable mode
pip install -e ".[dashboard]"    # optional: the Streamlit results page
brew install libomp              # macOS only: the OpenMP runtime XGBoost needs
```

## Usage

All commands run from the repo root with the virtual environment active.

**1. Build the database**

```bash
python scripts/init_db.py        # create the tables
python scripts/backfill.py       # download 2018-2026 (Jolpica + OpenF1; slow, resumable)
python scripts/validate.py       # audit the whole database and print a report (read-only)
```

**2. Evaluate**

| Script | What it shows |
| --- | --- |
| `evaluate_baseline.py` | the grid-order baseline, every season |
| `train_post_quali.py` | post-quali model vs the grid; saves the model to `data/models/` |
| `evaluate_pre_weekend.py` | pre-weekend model vs championship order |
| `evaluate_practice.py` | practice model vs practice order and the pre-weekend model |
| `evaluate_dnf.py` | DNF probabilities vs a flat rate |
| `evaluate_probabilities.py` | post-quali win/podium probabilities vs the grid |
| `evaluate_stage_probabilities.py` | pre-weekend and practice probabilities vs their baselines |

**3. Replay a season into the prediction log**

```bash
python scripts/replay_season.py                  # 2026, every race and stage (~2 minutes)
python scripts/replay_season.py --summary-only   # print the totals again
```

Deletes the season's earlier backtest runs first, so it can be re-run (live runs can never be deleted).

**4. Explain a prediction**

```bash
python scripts/explain_race.py 2026 19                   # post-quali stage
python scripts/explain_race.py 2026 19 --stage practice
```

**5. Look at the results in a browser**

```bash
streamlit run app/dashboard.py
```

## Tuned settings

Every value a script tunes (2026 weight, DNF shrinkage, simulator blend and noise) lives in [`src/f1predictor/tuned_settings.json`](src/f1predictor/tuned_settings.json). The scripts write it; the code and the dashboard read it.

The file is currently **frozen**: re-running a tuning script can't change a value. To re-tune on purpose, remove the `frozen` entry in the same commit and explain why.

## Tests

```bash
pytest
```

## Layout

```
src/f1predictor/
  config.py, settings.py     paths and constants; tuned settings (+ tuned_settings.json)
  db/                        schema (raw data + the append-only prediction log) and connection
  ingest/                    Jolpica and OpenF1 loaders, backfill
  validate/                  data checks
  features/                  the race feature table: form, practice pace, championship,
                             team lineages, which features each stage may use, and
                             as_of.py: prediction rows from only what was known at each stage
  models/                    baselines, the position model, DNF probabilities
  simulate/                  Monte Carlo simulator and race probabilities
  predict/                   predict one race at one stage and write it to the prediction log
  explain/                   SHAP contributions
  pipeline/                  the hands-off weekend pipeline: plan.py decides what's due now
  evaluate/                  metrics, walk-forward evaluation, and the scorer for logged runs
scripts/                     command-line entry points (see Usage)
app/                         Streamlit results dashboard
tests/                       pytest tests
docs/                        PRD and evaluation protocol
data/                        local SQLite DB and saved models (gitignored, rebuildable)
```
