from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR  # noqa: E402


SEASON_SLUG = "2023_24"
FINAL_GAMES_FILE = PROCESSED_DATA_DIR / f"final_games_{SEASON_SLUG}.csv"
REQUIRED_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_points",
    "away_points",
]
ROLLING_FEATURE_COLUMNS = [
    "last_5_win_pct",
    "last_10_point_diff",
    "avg_points_scored",
    "avg_points_allowed",
]


def load_final_games(input_path: Path = FINAL_GAMES_FILE) -> pd.DataFrame:
    """Load the final game-level dataset for one NBA season."""
    games = pd.read_csv(input_path, dtype={"game_id": str})
    return _prepare_games(games)


def _prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(REQUIRED_COLUMNS).difference(games.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Final games dataset is missing required columns: {missing}")

    games = games.copy()
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["home_points"] = pd.to_numeric(games["home_points"], errors="coerce")
    games["away_points"] = pd.to_numeric(games["away_points"], errors="coerce")

    if games[REQUIRED_COLUMNS].isna().any().any():
        missing_counts = games[REQUIRED_COLUMNS].isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        raise ValueError(
            "Final games dataset contains missing required values:\n"
            f"{missing_counts.to_string()}"
        )

    return games


def build_team_game_history(games: pd.DataFrame) -> pd.DataFrame:
    """Convert game-level rows into one row per team per game."""
    games = _prepare_games(games)

    home_games = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "game_date": games["game_date"],
            "team": games["home_team"],
            "opponent": games["away_team"],
            "points_scored": games["home_points"],
            "points_allowed": games["away_points"],
        }
    )

    away_games = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "game_date": games["game_date"],
            "team": games["away_team"],
            "opponent": games["home_team"],
            "points_scored": games["away_points"],
            "points_allowed": games["home_points"],
        }
    )

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games["win"] = (
        team_games["points_scored"] > team_games["points_allowed"]
    ).astype(int)
    team_games["point_diff"] = (
        team_games["points_scored"] - team_games["points_allowed"]
    )

    return team_games.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)


def add_rolling_team_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Calculate pre-game rolling features for each team."""
    team_games = team_games.sort_values(["team", "game_date", "game_id"]).copy()
    grouped = team_games.groupby("team", group_keys=False)

    shifted_wins = grouped["win"].shift(1)
    shifted_point_diff = grouped["point_diff"].shift(1)
    shifted_points_scored = grouped["points_scored"].shift(1)
    shifted_points_allowed = grouped["points_allowed"].shift(1)

    team_games["last_5_win_pct"] = shifted_wins.groupby(team_games["team"]).rolling(
        window=5,
        min_periods=1,
    ).mean().reset_index(level=0, drop=True)

    team_games["last_10_point_diff"] = shifted_point_diff.groupby(
        team_games["team"]
    ).rolling(
        window=10,
        min_periods=1,
    ).mean().reset_index(level=0, drop=True)

    team_games["avg_points_scored"] = shifted_points_scored.groupby(
        team_games["team"]
    ).expanding(
        min_periods=1,
    ).mean().reset_index(level=0, drop=True)

    team_games["avg_points_allowed"] = shifted_points_allowed.groupby(
        team_games["team"]
    ).expanding(
        min_periods=1,
    ).mean().reset_index(level=0, drop=True)

    return team_games


def build_rolling_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build a reusable rolling feature table indexed by game_id and team."""
    team_games = build_team_game_history(games)
    team_features = add_rolling_team_features(team_games)

    return (
        team_features.loc[:, ["game_id", "team", *ROLLING_FEATURE_COLUMNS]]
        .set_index(["game_id", "team"])
        .sort_index()
    )


def main() -> None:
    games = load_final_games()
    rolling_features = build_rolling_features(games)

    team_count = rolling_features.index.get_level_values("team").nunique()
    print(f"Teams processed: {team_count}")
    print(f"Rows generated: {len(rolling_features)}")
    print("\nSample rolling features:")
    print(rolling_features.head(10).to_string())


if __name__ == "__main__":
    main()
