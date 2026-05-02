from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ARTIFACTS_DIR, FEATURE_COLUMNS_PATH, MODEL_PATH, PROCESSED_DATA_DIR  # noqa: E402
from src.models.predict import (  # noqa: E402
    load_feature_columns,
    load_model,
    predict_home_win_probability,
)
from src.models.train_model import FEATURE_COLUMNS  # noqa: E402


FEATURES_FILE = PROCESSED_DATA_DIR / "features_all_seasons.csv"
MODEL_METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"
PROBABILITY_TOLERANCE = 1e-9


def _validate_artifacts_exist() -> None:
    missing_paths = [
        path
        for path in [MODEL_PATH, FEATURE_COLUMNS_PATH, MODEL_METRICS_PATH]
        if not path.exists()
    ]

    if missing_paths:
        formatted_paths = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required model artifacts:\n{formatted_paths}")


def _validate_feature_columns(feature_columns: list[str], features: pd.DataFrame) -> None:
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError("Saved feature columns do not match the expected training order.")

    missing_columns = set(feature_columns).difference(features.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Feature dataset is missing saved feature columns: {missing}")


def _validate_probabilities(home_probability: float, away_probability: float) -> None:
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


def test_prediction() -> None:
    _validate_artifacts_exist()

    features = pd.read_csv(FEATURES_FILE, dtype={"game_id": str})
    model = load_model()
    feature_columns = load_feature_columns()

    _validate_feature_columns(feature_columns, features)

    sample_row = features.iloc[0]
    home_probability = predict_home_win_probability(sample_row)
    away_probability = 1 - home_probability

    _validate_probabilities(home_probability, away_probability)

    print("Model artifacts verified.")
    print(f"Loaded model type: {type(model).__name__}")
    print(f"Feature columns verified: {len(feature_columns)}")
    print("\nSample prediction:")
    print(f"game_id: {sample_row['game_id']}")
    print(f"home_team: {sample_row['home_team']}")
    print(f"away_team: {sample_row['away_team']}")
    print(f"actual home_win: {int(sample_row['home_win'])}")
    print(f"predicted home_win_probability: {home_probability:.4f}")
    print(f"predicted away_win_probability: {away_probability:.4f}")


if __name__ == "__main__":
    test_prediction()
