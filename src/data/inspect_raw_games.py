from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR  # noqa: E402


RAW_GAMES_FILE = RAW_DATA_DIR / "games_2023_24.csv"


def inspect_raw_games(input_path: Path = RAW_GAMES_FILE) -> None:
    """Print read-only quality checks for the raw NBA games CSV."""
    print(f"Inspecting raw games file: {input_path}")

    games = pd.read_csv(input_path, dtype={"game_id": str})

    print("\n1. Shape")
    print(games.shape)

    print("\n2. Column Names")
    print(list(games.columns))

    print("\n3. First 10 Rows")
    print(games.head(10).to_string(index=False))

    print("\n4. Missing Values Per Column")
    print(games.isna().sum().to_string())

    print("\n5. Duplicate game_id Count")
    print(games["game_id"].duplicated().sum())

    print("\n6. Number of Unique Home Teams")
    print(games["home_team"].nunique(dropna=True))

    print("\n7. Number of Unique Away Teams")
    print(games["away_team"].nunique(dropna=True))

    game_dates = pd.to_datetime(games["game_date"], errors="coerce")
    print("\n8. Date Range of Games")
    print(f"{game_dates.min().date()} to {game_dates.max().date()}")

    print("\n9. Basic Points Summary")
    points_summary = games[["home_points", "away_points"]].agg(["min", "max", "mean"])
    print(points_summary.to_string())


if __name__ == "__main__":
    inspect_raw_games()
