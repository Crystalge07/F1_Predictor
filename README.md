# F1 Race Prediction Platform

A self-updating F1 prediction system: one model for every Grand Prix, refreshed after each session, with a public scorecard of its own accuracy. See [docs/PRD.md](docs/PRD.md) for the full spec.

**Status:** Phases 1–3 done (data, models for every stage, win/podium probabilities, explanations). Tuned settings are frozen ([docs/evaluation-protocol.md](docs/evaluation-protocol.md)). Next: Phase 4, replaying the 2026 season race by race with every prediction logged.

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

**Caveat:** these races were looked at many times during development, so the numbers are somewhat optimistic. The 2027 season, predicted live, is the real test. Details in [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

## Data

[Jolpica-F1](https://github.com/jolpica/jolpica-f1) for calendar, results and qualifying (2018+); [OpenF1](https://openf1.org) for laps, stints, pit stops, weather and race control (2023+). Stored in SQLite, in standard SQL so it can move to Postgres.

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

**3. Explain a prediction**

```bash
python scripts/explain_race.py 2026 19                   # post-quali stage
python scripts/explain_race.py 2026 19 --stage practice
```

**4. Look at the results in a browser**

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
  db/                        schema and connection
  ingest/                    Jolpica and OpenF1 loaders, backfill
  validate/                  data checks
  features/                  the race feature table: form, practice pace, championship,
                             team lineages, and which features each stage may use
  models/                    baselines, the position model, DNF probabilities
  simulate/                  Monte Carlo simulator and race probabilities
  explain/                   SHAP contributions
  evaluate/                  metrics and walk-forward evaluation
scripts/                     command-line entry points (see Usage)
app/                         Streamlit results dashboard
tests/                       pytest tests
docs/                        PRD and evaluation protocol
data/                        local SQLite DB and saved models (gitignored, rebuildable)
```
