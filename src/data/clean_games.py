from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402


SEASON_SLUG = "2023_24"
RAW_GAMES_FILE = RAW_DATA_DIR / f"games_{SEASON_SLUG}.csv"
CLEAN_GAMES_FILE = PROCESSED_DATA_DIR / f"clean_games_{SEASON_SLUG}.csv"
EXPECTED_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_points",
    "away_points",
    "season",
]


def _enforce_schema(games: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(EXPECTED_COLUMNS).difference(games.columns)
    extra_columns = set(games.columns).difference(EXPECTED_COLUMNS)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Raw games file is missing required columns: {missing}")

    if extra_columns:
        extra = ", ".join(sorted(extra_columns))
        raise ValueError(f"Raw games file has unexpected columns: {extra}")

    return games.loc[:, EXPECTED_COLUMNS].copy()


def _convert_types(games: pd.DataFrame) -> pd.DataFrame:
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["home_points"] = pd.to_numeric(games["home_points"], errors="coerce")
    games["away_points"] = pd.to_numeric(games["away_points"], errors="coerce")

    return games


def _validate_clean_games(games: pd.DataFrame) -> None:
    required_columns = EXPECTED_COLUMNS + ["home_win"]

    missing_counts = games[required_columns].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(
            "Cleaned games contain missing values:\n"
            f"{missing_counts.to_string()}"
        )

    unique_home_teams = games["home_team"].nunique()
    if unique_home_teams != 30:
        raise ValueError(f"Expected 30 unique home teams, found {unique_home_teams}.")

    unique_away_teams = games["away_team"].nunique()
    if unique_away_teams != 30:
        raise ValueError(f"Expected 30 unique away teams, found {unique_away_teams}.")

    same_team_games = games[games["home_team"] == games["away_team"]]
    if not same_team_games.empty:
        raise ValueError(
            f"Found {len(same_team_games)} games where home_team equals away_team."
        )

    non_positive_points = games[
        (games["home_points"] <= 0) | (games["away_points"] <= 0)
    ]
    if not non_positive_points.empty:
        raise ValueError(f"Found {len(non_positive_points)} games with non-positive points.")

    non_integer_points = games[
        (games["home_points"] % 1 != 0) | (games["away_points"] % 1 != 0)
    ]
    if not non_integer_points.empty:
        raise ValueError(f"Found {len(non_integer_points)} games with non-integer points.")


def clean_games(
    input_path: Path = RAW_GAMES_FILE,
    output_path: Path = CLEAN_GAMES_FILE,
) -> pd.DataFrame:
    print(f"Reading raw games from {input_path}")
    games = pd.read_csv(input_path, dtype={"game_id": str})
    input_shape = games.shape
    print(f"Input shape: {input_shape}")

    games = _enforce_schema(games)
    games = _convert_types(games)

    duplicate_count = games["game_id"].duplicated().sum()
    games = games.drop_duplicates(subset=["game_id"], keep="first")

    games["home_points"] = games["home_points"].astype("Int64")
    games["away_points"] = games["away_points"].astype("Int64")
    games["home_win"] = (games["home_points"] > games["away_points"]).astype(int)

    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    _validate_clean_games(games)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    print(f"Duplicates removed: {duplicate_count}")
    print(f"Output shape: {games.shape}")
    print(f"Saved cleaned games to {output_path}")

    return games


if __name__ == "__main__":
    clean_games()
