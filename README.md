# BetSki

BetSki is an end-to-end NBA prediction and betting-risk application that estimates game win probabilities, assigns risk scores, and explains model outputs in plain language. It combines a complete data pipeline, feature engineering workflow, trained model, FastAPI backend, and Streamlit interface into a portfolio-ready analytics product.

## Key Features

- Win probability prediction for NBA matchups
- Risk scoring on a 1-10 scale
- Human-readable explanations for predictions and risk levels
- FastAPI backend for serving prediction requests
- Streamlit UI for interactive game analysis

## Demo

Demo link coming soon. To run the project locally, start the FastAPI backend and Streamlit frontend, then open the Streamlit app in your browser.

## Screenshots

Add screenshots after running the Streamlit app locally. The main screenshot should show a completed prediction flow, including matchup input, win probability output, risk score, and human-readable explanations.

### Streamlit UI

![Streamlit UI](screenshots/streamlit_ui.png)

### API Response

![API Response](screenshots/api_response.png)

## Tech Stack

- Python
- pandas
- scikit-learn
- FastAPI
- Streamlit
- joblib
- nba_api

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn src.api.main:app --reload
```

Start the frontend:

```bash
streamlit run src/app.py
```

## Project Structure

```text
BetSki_App/
├── data/                 # Raw, processed, and prediction datasets
├── artifacts/            # Trained models and reusable outputs
├── screenshots/          # Demo screenshots for README
├── src/
│   ├── api/              # FastAPI backend
│   ├── data/             # Data ingestion and cleaning
│   ├── explain/          # Human-readable explanation engine
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and inference
│   ├── risk/             # Risk scoring engine
│   └── services/         # Application service layer
├── tests/                # Project tests
└── requirements.txt
```

## Future Improvements

- Add player-level data and injury context
- Train across multiple NBA seasons
- Deploy the backend and frontend on AWS
