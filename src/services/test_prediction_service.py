from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR  # noqa: E402
from src.services.prediction_service import PredictionService  # noqa: E402


SEASON_SLUG = "2023_24"
FEATURES_FILE = PROCESSED_DATA_DIR / f"features_{SEASON_SLUG}.csv"
PROBABILITY_TOLERANCE = 1e-9


def load_sample_game_id(features_path: Path = FEATURES_FILE) -> str:
    features = pd.read_csv(features_path, dtype={"game_id": str}, usecols=["game_id"])

    if features.empty:
        raise ValueError(f"Feature dataset is empty: {features_path}")

    return str(features.iloc[0]["game_id"])


def validate_prediction_response(response: dict[str, Any]) -> None:
    if not isinstance(response, dict):
        raise TypeError("PredictionService response must be a dictionary.")

    required_keys = {
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_win_probability",
        "away_win_probability",
        "predicted_winner",
        "risk_score",
        "risk_level",
        "reasons",
    }
    missing_keys = required_keys.difference(response)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"PredictionService response missing keys: {missing}")

    home_probability = response["home_win_probability"]
    away_probability = response["away_win_probability"]

    if not 0 <= home_probability <= 1:
        raise ValueError(f"home_win_probability is outside [0, 1]: {home_probability}")

    if not 0 <= away_probability <= 1:
        raise ValueError(f"away_win_probability is outside [0, 1]: {away_probability}")

    probability_sum = home_probability + away_probability
    if abs(probability_sum - 1) > PROBABILITY_TOLERANCE:
        raise ValueError(
            "Home and away probabilities do not sum to approximately 1: "
            f"{probability_sum}"
        )

    valid_winners = {response["home_team"], response["away_team"]}
    if response["predicted_winner"] not in valid_winners:
        raise ValueError(
            "predicted_winner must equal either home_team or away_team: "
            f"{response['predicted_winner']}"
        )

    risk_score = response["risk_score"]
    if not 1 <= risk_score <= 10:
        raise ValueError(f"risk_score must be between 1 and 10: {risk_score}")

    if response["risk_level"] not in {"Low", "Moderate", "High"}:
        raise ValueError(f"Unexpected risk_level: {response['risk_level']}")

    reasons = response["reasons"]
    if not isinstance(reasons, list):
        raise TypeError("PredictionService reasons must be a list.")

    if not reasons:
        raise ValueError("PredictionService reasons must contain at least one item.")


def main() -> None:
    game_id = load_sample_game_id()
    service = PredictionService()
    response = service.predict_by_game_id(game_id)

    validate_prediction_response(response)

    print("PredictionService response:")
    print(f"game_id: {response['game_id']}")
    print(f"game_date: {response['game_date']}")
    print(f"home_team: {response['home_team']}")
    print(f"away_team: {response['away_team']}")
    print(f"home_win_probability: {response['home_win_probability']:.4f}")
    print(f"away_win_probability: {response['away_win_probability']:.4f}")
    print(f"predicted_winner: {response['predicted_winner']}")
    print(f"risk_score: {response['risk_score']}")
    print(f"risk_level: {response['risk_level']}")
    print(f"reasons: {response['reasons']}")
    print("\nPredictionService verification passed.")


if __name__ == "__main__":
    main()
