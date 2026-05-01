# BetSki

BetSki is an NBA analytics and betting-risk system that turns historical game results into leak-safe pre-game features. The project is being built as a production-style Python data pipeline for future win-probability modeling and risk scoring.

## Current Status

Feature engineering is complete for the 2023-24 NBA regular season. The model, API, frontend, cloud deployment, and betting-risk layer have not been built yet.

## Features

Current:

- Fetches historical NBA game results with `nba_api`
- Stores raw, cleaned, final, and feature-ready datasets locally
- Validates game schema, dates, teams, duplicates, and scoring fields
- Builds rolling team features using only games played before each matchup
- Adds rest-day and back-to-back indicators for home and away teams

Planned:

- Train baseline and advanced game outcome models
- Estimate win probability and betting risk scores
- Add explainability for model predictions
- Integrate player-level, injury, odds, and schedule data
- Expose predictions through an API and lightweight app
- Add AWS-based storage and scheduled pipelines

## Tech Stack

- Python
- pandas
- scikit-learn
- nba_api
- FastAPI
- Streamlit
- joblib
- python-dotenv

## Project Structure

```text
BetSki_App/
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── risk/
│   ├── api/
│   ├── services/
│   └── explain/
├── artifacts/
├── notebooks/
├── tests/
├── config.py
└── requirements.txt
```

## Run Locally

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the data pipeline:

```bash
python src/data/fetch_games.py
python src/data/inspect_raw_games.py
python src/data/clean_games.py
python src/data/build_dataset.py
python src/features/rolling_features.py
python src/features/build_features.py
```

The feature dataset is generated at:

```text
data/processed/features_2023_24.csv
```

## Data Note

Generated datasets and model artifacts are intentionally ignored by git. The repository preserves folder structure with `.gitkeep` files while keeping reproducible outputs local.

## Future Improvements

- Expand data collection across multiple NBA seasons
- Add time-aware model training and evaluation
- Build risk scoring for betting decisions
- Add player stats, injuries, odds, and market movement
- Create FastAPI endpoints for predictions
- Add a Streamlit dashboard
- Move storage and scheduled jobs to AWS
