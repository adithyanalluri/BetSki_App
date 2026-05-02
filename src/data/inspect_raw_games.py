from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR  # noqa: E402


RAW_GAMES_PATTERN = "games_*.csv"


def _season_from_path(path: Path) -> str:
    return path.stem.removeprefix("games_").replace("_", "-")


def _points_summary(games: pd.DataFrame) -> pd.DataFrame:
    points = games[["home_points", "away_points"]].apply(pd.to_numeric, errors="coerce")
    return points.agg(["min", "max", "mean"])


def inspect_raw_file(input_path: Path) -> dict[str, object]:
    """Print read-only quality checks for the raw NBA games CSV."""
    season = _season_from_path(input_path)
    print(f"\n=== Raw season {season} ===")
    print(f"File: {input_path}")

    games = pd.read_csv(input_path, dtype={"game_id": str})

    print("\nShape")
    print(games.shape)

    print("\nColumns")
    print(list(games.columns))

    print("\nMissing Values Per Column")
    print(games.isna().sum().to_string())

    duplicate_game_ids = games["game_id"].duplicated().sum()
    print("\nDuplicate game_id Count")
    print(duplicate_game_ids)

    teams = pd.concat([games["home_team"], games["away_team"]]).dropna()
    unique_teams = teams.nunique()
    print("\nUnique Teams")
    print(unique_teams)

    game_dates = pd.to_datetime(games["game_date"], errors="coerce").dropna()
    min_date = game_dates.min().date() if not game_dates.empty else None
    max_date = game_dates.max().date() if not game_dates.empty else None
    print("\nDate Range")
    print(f"{min_date} to {max_date}")

    print("\nPoints Summary")
    print(_points_summary(games).to_string())

    return {
        "season": season,
        "rows": len(games),
        "columns": len(games.columns),
        "missing_values": int(games.isna().sum().sum()),
        "duplicate_game_ids": int(duplicate_game_ids),
        "unique_teams": int(unique_teams),
        "min_date": min_date,
        "max_date": max_date,
    }


def inspect_raw_games(raw_data_dir: Path = RAW_DATA_DIR) -> None:
    """Inspect every raw NBA games CSV without modifying data."""
    input_paths = sorted(raw_data_dir.glob(RAW_GAMES_PATTERN))
    if not input_paths:
        raise FileNotFoundError(f"No raw game files found in {raw_data_dir}")

    summaries = [inspect_raw_file(path) for path in input_paths]

    print("\n=== Raw seasons summary ===")
    summary = pd.DataFrame(summaries)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    inspect_raw_games()
