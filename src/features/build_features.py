from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR  # noqa: E402
from src.features.rolling_features import (  # noqa: E402
    FINAL_GAMES_FILE,
    build_rolling_features,
    build_team_game_history,
    load_final_games,
)


SEASON_SLUG = "2023_24"
FEATURES_FILE = PROCESSED_DATA_DIR / f"features_{SEASON_SLUG}.csv"
BASE_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_points",
    "away_points",
    "home_win",
]
REST_FEATURE_COLUMNS = ["rest_days", "back_to_back"]
FEATURE_COLUMNS = [
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


def build_rest_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build pre-game rest features indexed by game_id and team."""
    team_games = build_team_game_history(games)
    team_games = team_games.sort_values(["team", "game_date", "game_id"]).copy()

    previous_game_date = team_games.groupby("team")["game_date"].shift(1)
    team_games["rest_days"] = (
        team_games["game_date"] - previous_game_date
    ).dt.days
    team_games["back_to_back"] = (team_games["rest_days"] == 1).astype(int)

    return (
        team_games.loc[:, ["game_id", "team", *REST_FEATURE_COLUMNS]]
        .set_index(["game_id", "team"])
        .sort_index()
    )


def _prefix_feature_columns(features: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return features.rename(columns={column: f"{prefix}_{column}" for column in features})


def _join_team_features(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
    team_column: str,
    prefix: str,
) -> pd.DataFrame:
    lookup = _prefix_feature_columns(team_features, prefix).reset_index()

    return games.merge(
        lookup,
        left_on=["game_id", team_column],
        right_on=["game_id", "team"],
        how="left",
    ).drop(columns=["team"])


def build_feature_dataset(games: pd.DataFrame) -> pd.DataFrame:
    games = games.loc[:, BASE_COLUMNS].copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")

    rolling_features = build_rolling_features(games)
    rest_features = build_rest_features(games)
    team_features = rolling_features.join(rest_features, how="left")

    features = _join_team_features(
        games=games,
        team_features=team_features,
        team_column="home_team",
        prefix="home",
    )
    features = _join_team_features(
        games=features,
        team_features=team_features,
        team_column="away_team",
        prefix="away",
    )

    features = features.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    rows_before_drop = len(features)
    features = features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    rows_dropped = rows_before_drop - len(features)

    features["home_points"] = features["home_points"].astype("int64")
    features["away_points"] = features["away_points"].astype("int64")
    features["home_win"] = features["home_win"].astype("int64")
    features["home_back_to_back"] = features["home_back_to_back"].astype("int64")
    features["away_back_to_back"] = features["away_back_to_back"].astype("int64")

    features.attrs["rows_dropped"] = rows_dropped
    return features


def save_feature_dataset(
    input_path: Path = FINAL_GAMES_FILE,
    output_path: Path = FEATURES_FILE,
) -> pd.DataFrame:
    print(f"Reading final games from {input_path}")
    games = load_final_games(input_path)
    print(f"Input shape: {games.shape}")

    features = build_feature_dataset(games)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    print(f"Output shape: {features.shape}")
    print(f"Rows dropped: {features.attrs['rows_dropped']}")
    print(f"Saved feature dataset to {output_path}")
    print("\nSample output:")
    print(features.head(10).to_string(index=False))

    return features


if __name__ == "__main__":
    save_feature_dataset()
