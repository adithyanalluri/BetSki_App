from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR  # noqa: E402


ALL_GAMES_FILE = PROCESSED_DATA_DIR / "all_games.csv"
EXPECTED_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_points",
    "away_points",
    "season",
    "home_win",
]


def _enforce_schema(games: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(EXPECTED_COLUMNS).difference(games.columns)
    extra_columns = set(games.columns).difference(EXPECTED_COLUMNS)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"all_games.csv is missing required columns: {missing}")

    if extra_columns:
        extra = ", ".join(sorted(extra_columns))
        raise ValueError(f"all_games.csv has unexpected columns: {extra}")

    return games.loc[:, EXPECTED_COLUMNS].copy()


def _convert_types(games: pd.DataFrame) -> pd.DataFrame:
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["home_points"] = pd.to_numeric(games["home_points"], errors="coerce")
    games["away_points"] = pd.to_numeric(games["away_points"], errors="coerce")
    games["season"] = games["season"].astype(str)
    games["home_win"] = pd.to_numeric(games["home_win"], errors="coerce")
    return games


def _validate_values(games: pd.DataFrame) -> None:
    if games.isna().any().any():
        missing_counts = games.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        raise ValueError(
            "all_games.csv contains missing values:\n"
            f"{missing_counts.to_string()}"
        )

    duplicate_game_ids = games["game_id"].duplicated().sum()
    if duplicate_game_ids:
        raise ValueError(f"Found {duplicate_game_ids} duplicate game_id rows.")

    sorted_games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if not games.reset_index(drop=True).equals(sorted_games):
        raise ValueError("all_games.csv is not sorted by game_date, game_id.")

    same_team_games = games[games["home_team"] == games["away_team"]]
    if not same_team_games.empty:
        raise ValueError(
            f"Found {len(same_team_games)} rows where home_team equals away_team."
        )

    teams = pd.concat([games["home_team"], games["away_team"]]).dropna()
    blank_teams = teams.astype(str).str.strip().eq("").sum()
    if blank_teams:
        raise ValueError(f"Found {blank_teams} blank team values.")

    unique_teams = teams.nunique()
    if unique_teams != 30:
        raise ValueError(f"Expected 30 unique teams, found {unique_teams}.")

    non_positive_points = games[
        (games["home_points"] <= 0) | (games["away_points"] <= 0)
    ]
    if not non_positive_points.empty:
        raise ValueError(f"Found {len(non_positive_points)} rows with non-positive points.")

    invalid_home_win = ~games["home_win"].isin([0, 1])
    if invalid_home_win.any():
        raise ValueError(f"Found {invalid_home_win.sum()} rows with invalid home_win.")


def validate_all_games(input_path: Path = ALL_GAMES_FILE) -> None:
    print(f"Validating combined dataset: {input_path}")
    games = pd.read_csv(input_path, dtype={"game_id": str})
    games = _enforce_schema(games)
    games = _convert_types(games)

    print("\nShape")
    print(games.shape)

    print("\nMissing Values")
    print(games.isna().sum().to_string())

    print("\nDuplicate game_id Count")
    print(games["game_id"].duplicated().sum())

    game_dates = games["game_date"].dropna()
    min_date = game_dates.min().date() if not game_dates.empty else None
    max_date = game_dates.max().date() if not game_dates.empty else None
    print("\nDate Range")
    print(f"{min_date} to {max_date}")

    teams = pd.concat([games["home_team"], games["away_team"]]).dropna()
    print("\nUnique Teams")
    print(teams.nunique())

    print("\nSeason Distribution")
    print(games["season"].value_counts().sort_index().to_string())

    print("\nhome_win Distribution")
    print(games["home_win"].value_counts().sort_index().to_string())

    print("\nPoints Summary")
    points_summary = games[["home_points", "away_points"]].agg(["min", "max", "mean"])
    print(points_summary.to_string())

    _validate_values(games)
    print("\nSuccess: all_games.csv is valid.")


if __name__ == "__main__":
    validate_all_games()
