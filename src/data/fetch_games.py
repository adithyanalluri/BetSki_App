from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder


SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]
SEASON_TYPE = "Regular Season"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_points",
    "away_points",
    "season",
]


def fetch_team_game_logs(season: str) -> pd.DataFrame:
    """Fetch team-level NBA game logs for a single season."""
    print(f"Fetching NBA {SEASON_TYPE.lower()} games for season {season}...")

    finder = LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=season,
        season_type_nullable=SEASON_TYPE,
        league_id_nullable="00",
    )

    games = finder.get_data_frames()[0]
    print(f"Fetched {len(games)} team-game rows from nba_api.")
    return games


def _validate_source_columns(games: pd.DataFrame) -> None:
    required_columns = {
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ABBREVIATION",
        "MATCHUP",
        "PTS",
    }
    missing_columns = required_columns.difference(games.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"nba_api response missing required columns: {missing}")


def structure_game_rows(team_games: pd.DataFrame, season: str) -> pd.DataFrame:
    """Convert team-level game logs into one row per game."""
    _validate_source_columns(team_games)

    source_columns = [
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ABBREVIATION",
        "MATCHUP",
        "PTS",
    ]
    games = team_games.loc[:, source_columns].copy()

    games = games.dropna(subset=source_columns)
    games = games.drop_duplicates(
        subset=["GAME_ID", "TEAM_ABBREVIATION", "MATCHUP"],
        keep="first",
    )

    structured_games: list[dict[str, object]] = []
    skipped_games = 0

    for game_id, group in games.groupby("GAME_ID", sort=False):
        home_rows = group[group["MATCHUP"].str.contains("vs.", regex=False)]
        away_rows = group[group["MATCHUP"].str.contains("@", regex=False)]

        if len(home_rows) != 1 or len(away_rows) != 1:
            skipped_games += 1
            continue

        home = home_rows.iloc[0]
        away = away_rows.iloc[0]

        structured_games.append(
            {
                "game_id": str(game_id),
                "game_date": pd.to_datetime(home["GAME_DATE"]).date().isoformat(),
                "home_team": home["TEAM_ABBREVIATION"],
                "away_team": away["TEAM_ABBREVIATION"],
                "home_points": int(home["PTS"]),
                "away_points": int(away["PTS"]),
                "season": season,
            }
        )

    result = pd.DataFrame(structured_games, columns=EXPECTED_COLUMNS)

    result = result.dropna()
    result = result.drop_duplicates(subset=["game_id"], keep="first")
    result = result.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    if skipped_games:
        print(f"Skipped {skipped_games} games with incomplete home/away rows.")

    return result


def save_raw_games(games: pd.DataFrame, season: str) -> Path:
    """Save raw game-level data to data/raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    season_slug = season.replace("-", "_")
    output_path = RAW_DATA_DIR / f"games_{season_slug}.csv"

    games.to_csv(output_path, index=False)
    print(f"Saved {len(games)} games to {output_path}")
    return output_path


def fetch_and_save_games(season: str) -> Path:
    team_games = fetch_team_game_logs(season)
    games = structure_game_rows(team_games, season)

    if games.empty:
        raise ValueError(f"No complete game rows were structured for season {season}.")

    print(f"Structured {len(games)} unique games.")
    return save_raw_games(games, season)


def fetch_and_save_all_games(seasons: list[str] | None = None) -> list[Path]:
    """Fetch configured NBA seasons, allowing individual seasons to fail."""
    seasons = seasons or SEASONS
    saved_paths: list[Path] = []
    failures: dict[str, str] = {}

    print(f"Fetching {len(seasons)} NBA seasons: {', '.join(seasons)}")
    for season in seasons:
        print(f"\n=== Season {season} ===")
        try:
            saved_paths.append(fetch_and_save_games(season))
        except Exception as exc:  # noqa: BLE001
            failures[season] = str(exc)
            print(f"Failed to fetch season {season}: {exc}")

    print("\nFetch summary")
    print(f"Successful seasons: {len(saved_paths)}")
    print(f"Failed seasons: {len(failures)}")
    for season, message in failures.items():
        print(f"- {season}: {message}")

    if not saved_paths:
        raise RuntimeError("No seasons were fetched successfully.")

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch configured NBA seasons of raw historical game results."
    )
    parser.add_argument(
        "--season",
        action="append",
        dest="seasons",
        help=(
            "NBA season in YYYY-YY format. Can be passed more than once. "
            "Defaults to the configured SEASONS list."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fetch_and_save_all_games(args.seasons)
