# ⚾ Homer Tracker

**MLB Home Run & Game Prediction System**

Predicts which batters will hit home runs tonight and which teams will win,
using machine learning trained on historical MLB data.

Built with the same architecture as [Snipe Tracker](../snipe-tracker) (NHL goals).

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Collect training data (takes ~30 min first time)
python -m src.data.collect_bulk

# Train models
python -m src.train

# Get tonight's picks
python -m src.predictions.daily

# Set up daily automation (email + Telegram + launchd)
python -m src.automation.setup
```

## Architecture

```
src/
├── data/           # MLB Stats API client + data collection
├── features/       # Feature engineering (batter, pitcher, team)
├── models/         # ML models (HR predictor + game winner)
├── predictions/    # Daily picks, HTML reports, grading
├── notifications/  # Email + Telegram delivery
├── automation/     # Daily runner + launchd scheduler
└── train.py        # Model training pipeline
```

## Data Source

Uses the free [MLB Stats API](https://statsapi.mlb.com) — no API key needed.

## Key Commands

| Command | What it does |
|---------|-------------|
| `python -m src.predictions.daily` | Tonight's HR picks + game predictions |
| `python -m src.predictions.grade 2025-06-15` | Grade a past day's predictions |
| `python -m src.predictions.grade --lifetime` | Show running accuracy stats |
| `python -m src.automation.runner` | Full daily pipeline (predict + grade + notify) |
| `python -m src.automation.runner --status` | Check system config status |
| `python -m src.data.collect_bulk` | Collect historical boxscore data |
| `python -m src.train` | Retrain models on latest data |

## Disclaimer

⚠️ For entertainment purposes only. Model probabilities are relative rankings,
not absolute odds. Never bet more than you can afford to lose.
