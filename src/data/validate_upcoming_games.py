from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR  # noqa: E402


UPCOMING_GAMES_FILE = RAW_DATA_DIR / "upcoming_games.csv"
REQUIRED_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "season",
    "status",
]
REQUIRED_NON_NULL_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
]


def validate_upcoming_games(input_path: Path = UPCOMING_GAMES_FILE) -> None:
    print(f"Validating upcoming games file: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"upcoming_games.csv does not exist: {input_path}")

    games = pd.read_csv(input_path, dtype={"game_id": str})

    missing_columns = set(REQUIRED_COLUMNS).difference(games.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"upcoming_games.csv is missing required columns: {missing}")

    games = games.loc[:, REQUIRED_COLUMNS].copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")

    missing_values = games[REQUIRED_NON_NULL_COLUMNS].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(
            "upcoming_games.csv contains missing required values:\n"
            f"{missing_values.to_string()}"
        )

    invalid_dates = games["game_date"].isna().sum()
    if invalid_dates:
        raise ValueError(f"Found {invalid_dates} rows with invalid game_date.")

    same_team_games = games[games["home_team"] == games["away_team"]]
    if not same_team_games.empty:
        raise ValueError(
            f"Found {len(same_team_games)} rows where home_team equals away_team."
        )

    duplicate_game_ids = games["game_id"].duplicated().sum()
    if duplicate_game_ids:
        raise ValueError(f"Found {duplicate_game_ids} duplicate game_id rows.")

    print(f"Rows: {len(games)}")
    if not games.empty:
        print(
            "Date range: "
            f"{games['game_date'].min().date()} to {games['game_date'].max().date()}"
        )
        print(f"Teams: {pd.concat([games['home_team'], games['away_team']]).nunique()}")
    print("Success: upcoming_games.csv is valid.")


if __name__ == "__main__":
    validate_upcoming_games()
