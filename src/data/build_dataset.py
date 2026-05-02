from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR  # noqa: E402


CLEAN_GAMES_PATTERN = "clean_games_*.csv"
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
        raise ValueError(f"Cleaned dataset is missing required columns: {missing}")

    if extra_columns:
        extra = ", ".join(sorted(extra_columns))
        raise ValueError(f"Cleaned dataset has unexpected columns: {extra}")

    return games.loc[:, EXPECTED_COLUMNS].copy()


def _enforce_types(games: pd.DataFrame) -> pd.DataFrame:
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["home_points"] = pd.to_numeric(games["home_points"], errors="coerce")
    games["away_points"] = pd.to_numeric(games["away_points"], errors="coerce")
    games["season"] = games["season"].astype(str)
    games["home_win"] = pd.to_numeric(games["home_win"], errors="coerce")

    return games


def _validate_final_dataset(games: pd.DataFrame) -> None:
    if games.isna().any().any():
        missing_counts = games.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        raise ValueError(
            "Final dataset contains missing values:\n"
            f"{missing_counts.to_string()}"
        )

    duplicate_game_ids = games["game_id"].duplicated().sum()
    if duplicate_game_ids:
        raise ValueError(f"Found {duplicate_game_ids} duplicate game_id rows.")

    if not games["game_date"].is_monotonic_increasing:
        raise ValueError("game_date is not sorted in ascending order.")

    same_team_games = games[games["home_team"] == games["away_team"]]
    if not same_team_games.empty:
        raise ValueError(
            f"Found {len(same_team_games)} rows where home_team equals away_team."
        )

    negative_points = games[
        (games["home_points"] < 0) | (games["away_points"] < 0)
    ]
    if not negative_points.empty:
        raise ValueError(f"Found {len(negative_points)} rows with negative points.")

    invalid_home_win = ~games["home_win"].isin([0, 1])
    if invalid_home_win.any():
        raise ValueError(f"Found {invalid_home_win.sum()} rows with invalid home_win.")


def _read_clean_games(input_path: Path) -> pd.DataFrame:
    print(f"Reading cleaned games from {input_path}")
    games = pd.read_csv(input_path, dtype={"game_id": str})
    print(f"Input shape: {games.shape}")

    games = _enforce_schema(games)
    return _enforce_types(games)


def build_dataset(
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    output_path: Path = ALL_GAMES_FILE,
) -> pd.DataFrame:
    input_paths = sorted(processed_data_dir.glob(CLEAN_GAMES_PATTERN))
    if not input_paths:
        raise FileNotFoundError(f"No cleaned game files found in {processed_data_dir}")

    games = pd.concat(
        [_read_clean_games(input_path) for input_path in input_paths],
        ignore_index=True,
    )
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    games["home_points"] = games["home_points"].astype("int64")
    games["away_points"] = games["away_points"].astype("int64")
    games["home_win"] = games["home_win"].astype("int64")

    _validate_final_dataset(games)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    games = games.loc[:, EXPECTED_COLUMNS]
    games.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    rows_per_season = games.groupby("season").size().sort_index()
    seasons = list(rows_per_season.index)
    min_date = games["game_date"].min().date()
    max_date = games["game_date"].max().date()

    print("\nBuild summary")
    print(f"Seasons included: {', '.join(seasons)}")
    print("Rows per season:")
    print(rows_per_season.to_string())
    print(f"Total rows: {len(games)}")
    print(f"Date range: {min_date} to {max_date}")
    print(f"Output shape: {games.shape}")
    print(f"Saved combined dataset to {output_path}")

    return games


if __name__ == "__main__":
    build_dataset()
