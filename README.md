# F1 Race Prediction Platform

A self-updating F1 prediction system: one model for every Grand Prix, refreshed after each session, with a public scorecard of its own accuracy. See [docs/PRD.md](docs/PRD.md) for the full spec.

**Status:** Phase 1 (data foundation) in progress.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv          # create an isolated environment for this project
source .venv/bin/activate      # activate it (run this in every new terminal)
pip install -e ".[dev]"        # install the project + dev tools in editable mode
```

## Layout

```
src/f1predictor/   the package: config, db, ingest, validate
scripts/           command-line entry points (init_db, backfill)
tests/             pytest tests
docs/              PRD and design notes
data/              local SQLite DB + FastF1 cache (gitignored, rebuildable)
```
