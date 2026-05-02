from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR  # noqa: E402


FEATURES_FILE = PROCESSED_DATA_DIR / "features_all_seasons.csv"
REQUIRED_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_points",
    "away_points",
    "season",
    "home_win",
    "home_last_5_win_pct",
    "home_last_10_point_diff",
    "home_avg_points_scored",
    "home_avg_points_allowed",
    "away_last_5_win_pct",
    "away_last_10_point_diff",
    "away_avg_points_scored",
    "away_avg_points_allowed",
    "home_rest_days",
    "away_rest_days",
    "home_back_to_back",
    "away_back_to_back",
]


def _load_features(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Feature dataset does not exist: {input_path}")

    features = pd.read_csv(input_path, dtype={"game_id": str})
    missing_columns = set(REQUIRED_COLUMNS).difference(features.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Feature dataset is missing required columns: {missing}")

    return features.loc[:, REQUIRED_COLUMNS].copy()


def _convert_types(features: pd.DataFrame) -> pd.DataFrame:
    features["game_id"] = features["game_id"].astype(str)
    features["game_date"] = pd.to_datetime(features["game_date"], errors="coerce")
    features["home_points"] = pd.to_numeric(features["home_points"], errors="coerce")
    features["away_points"] = pd.to_numeric(features["away_points"], errors="coerce")
    features["home_win"] = pd.to_numeric(features["home_win"], errors="coerce")
    features["home_rest_days"] = pd.to_numeric(features["home_rest_days"], errors="coerce")
    features["away_rest_days"] = pd.to_numeric(features["away_rest_days"], errors="coerce")
    features["home_back_to_back"] = pd.to_numeric(
        features["home_back_to_back"],
        errors="coerce",
    )
    features["away_back_to_back"] = pd.to_numeric(
        features["away_back_to_back"],
        errors="coerce",
    )

    for column in REQUIRED_COLUMNS:
        if column.startswith(("home_", "away_")) and column not in {
            "home_team",
            "away_team",
        }:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    return features


def validate_features(input_path: Path = FEATURES_FILE) -> None:
    print(f"Validating feature dataset: {input_path}")
    features = _convert_types(_load_features(input_path))

    if features.isna().any().any():
        missing_counts = features.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        raise ValueError(
            "Feature dataset contains missing values:\n"
            f"{missing_counts.to_string()}"
        )

    same_team_games = features[features["home_team"] == features["away_team"]]
    if not same_team_games.empty:
        raise ValueError(
            f"Found {len(same_team_games)} rows where home_team equals away_team."
        )

    invalid_home_win = ~features["home_win"].isin([0, 1])
    if invalid_home_win.any():
        raise ValueError(f"Found {invalid_home_win.sum()} rows with invalid home_win.")

    negative_rest = (features["home_rest_days"] < 0) | (features["away_rest_days"] < 0)
    if negative_rest.any():
        raise ValueError(f"Found {negative_rest.sum()} rows with negative rest days.")

    for column in ["home_back_to_back", "away_back_to_back"]:
        invalid_values = ~features[column].isin([0, 1])
        if invalid_values.any():
            raise ValueError(f"Found {invalid_values.sum()} rows with invalid {column}.")

    duplicate_game_ids = features["game_id"].duplicated().sum()
    if duplicate_game_ids:
        raise ValueError(f"Found {duplicate_game_ids} duplicate game_id rows.")

    sorted_features = features.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if not features.reset_index(drop=True).equals(sorted_features):
        raise ValueError("Feature dataset is not sorted by game_date and game_id.")

    print(f"Shape: {features.shape}")
    print(f"Seasons included: {', '.join(sorted(features['season'].unique()))}")
    print(
        "Date range: "
        f"{features['game_date'].min().date()} to {features['game_date'].max().date()}"
    )
    print("Success: features_all_seasons.csv is valid.")


if __name__ == "__main__":
    validate_features()
