from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import ScheduleLeagueV2
from nba_api.stats.static import teams


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR  # noqa: E402


UPCOMING_GAMES_FILE = RAW_DATA_DIR / "upcoming_games.csv"
DEFAULT_LOOKAHEAD_DAYS = 14
LEAGUE_ID = "00"
REQUIRED_SOURCE_COLUMNS = [
    "gameId",
    "gameDate",
    "gameStatus",
    "gameStatusText",
    "homeTeam_teamId",
    "homeTeam_teamTricode",
    "awayTeam_teamId",
    "awayTeam_teamTricode",
]
OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "season",
    "status",
]
TEAM_ID_TO_ABBREVIATION = {
    team["id"]: team["abbreviation"] for team in teams.get_teams()
}


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def infer_nba_season(search_date: date) -> str:
    """Return NBA season slug for a date, e.g. 2025-26."""
    start_year = search_date.year if search_date.month >= 9 else search_date.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _validate_date_range(start_date: date, end_date: date) -> None:
    if end_date < start_date:
        raise ValueError("end_date must be greater than or equal to start_date.")


def fetch_season_schedule(season: str, timeout: int = 30) -> pd.DataFrame:
    """Fetch the NBA schedule response for one season."""
    try:
        endpoint = ScheduleLeagueV2(
            league_id=LEAGUE_ID,
            season=season,
            timeout=timeout,
        )
        frames = endpoint.get_data_frames()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"nba_api schedule fetch failed for season {season}: {exc}") from exc

    if not frames:
        raise ValueError(f"Malformed schedule response for season {season}: no tables.")

    schedule = frames[0]
    if schedule.empty:
        return schedule

    missing_columns = set(REQUIRED_SOURCE_COLUMNS).difference(schedule.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Malformed schedule response for season {season}; missing fields: {missing}"
        )

    return schedule


def _team_abbreviation(row: pd.Series, tricode_column: str, team_id_column: str) -> str:
    tricode = row.get(tricode_column)
    if pd.notna(tricode) and str(tricode).strip():
        return str(tricode).strip().upper()

    team_id = row.get(team_id_column)
    if pd.isna(team_id):
        return ""

    try:
        return TEAM_ID_TO_ABBREVIATION.get(int(team_id), "")
    except ValueError:
        return ""


def structure_upcoming_games(
    schedule: pd.DataFrame,
    start_date: date,
    end_date: date,
    season: str,
) -> pd.DataFrame:
    """Filter and normalize schedule rows into BetSki's upcoming game schema."""
    if schedule.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    games = schedule.loc[:, REQUIRED_SOURCE_COLUMNS].copy()
    games["game_date"] = pd.to_datetime(games["gameDate"], errors="coerce").dt.date
    games["gameStatus"] = pd.to_numeric(games["gameStatus"], errors="coerce")

    malformed_dates = games["game_date"].isna().sum()
    if malformed_dates:
        raise ValueError(f"Found {malformed_dates} schedule rows with malformed gameDate.")

    in_range = games["game_date"].between(start_date, end_date)
    not_final = games["gameStatus"].ne(3)
    games = games.loc[in_range & not_final].copy()

    if games.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    games["home_team"] = games.apply(
        _team_abbreviation,
        axis=1,
        tricode_column="homeTeam_teamTricode",
        team_id_column="homeTeam_teamId",
    )
    games["away_team"] = games.apply(
        _team_abbreviation,
        axis=1,
        tricode_column="awayTeam_teamTricode",
        team_id_column="awayTeam_teamId",
    )

    upcoming = pd.DataFrame(
        {
            "game_id": games["gameId"].astype(str),
            "game_date": games["game_date"].astype(str),
            "home_team": games["home_team"],
            "away_team": games["away_team"],
            "season": season,
            "status": games["gameStatusText"].astype(str),
        },
        columns=OUTPUT_COLUMNS,
    )

    incomplete_team_rows = (
        upcoming["home_team"].astype(str).str.strip().eq("")
        | upcoming["away_team"].astype(str).str.strip().eq("")
    )
    skipped_incomplete_teams = int(incomplete_team_rows.sum())
    upcoming = upcoming.loc[~incomplete_team_rows].copy()

    upcoming = upcoming.drop_duplicates(subset=["game_id"], keep="first")
    upcoming = upcoming.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    upcoming.attrs["skipped_incomplete_teams"] = skipped_incomplete_teams
    return upcoming


def save_upcoming_games(games: pd.DataFrame, output_path: Path = UPCOMING_GAMES_FILE) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(output_path, index=False)
    return output_path


def fetch_and_save_upcoming_games(
    start_date: date,
    end_date: date,
    season: str | None = None,
    output_path: Path = UPCOMING_GAMES_FILE,
    timeout: int = 30,
) -> pd.DataFrame:
    _validate_date_range(start_date, end_date)
    season = season or infer_nba_season(start_date)

    print(f"Searching upcoming NBA games from {start_date} to {end_date}")
    print(f"Season: {season}")

    schedule = fetch_season_schedule(season=season, timeout=timeout)
    upcoming = structure_upcoming_games(
        schedule=schedule,
        start_date=start_date,
        end_date=end_date,
        season=season,
    )
    saved_path = save_upcoming_games(upcoming, output_path)

    skipped_incomplete_teams = upcoming.attrs.get("skipped_incomplete_teams", 0)
    print(f"Upcoming games found: {len(upcoming)}")
    if skipped_incomplete_teams:
        print(f"Skipped TBD games with incomplete teams: {skipped_incomplete_teams}")
    if upcoming.empty:
        print("No upcoming games found for the requested date range.")
    print(f"Output path: {saved_path}")
    print("\nSample rows:")
    print(upcoming.head(10).to_string(index=False))

    return upcoming


def parse_args() -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(
        description="Fetch upcoming NBA games into data/raw/upcoming_games.csv."
    )
    parser.add_argument(
        "--date",
        type=_parse_date,
        help="Single date to fetch in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=today,
        help=f"Start date in YYYY-MM-DD format. Defaults to today ({today}).",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_LOOKAHEAD_DAYS,
        help=f"Lookahead days when --end-date is omitted. Defaults to {DEFAULT_LOOKAHEAD_DAYS}.",
    )
    parser.add_argument(
        "--season",
        help="NBA season in YYYY-YY format. Defaults to inferred season from start date.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="nba_api request timeout in seconds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.date:
        search_start = args.date
        search_end = args.date
    else:
        search_start = args.start_date
        search_end = args.end_date or search_start + timedelta(days=args.days)

    fetch_and_save_upcoming_games(
        start_date=search_start,
        end_date=search_end,
        season=args.season,
        timeout=args.timeout,
    )
