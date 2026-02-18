"""
nathanbao_nba.py

Collects NBA Player Points + Rebounds + Assists (PRA) game-by-game data
for the starting 5 players on every team over the last 2 seasons.

Strategy:
  - Pulls all player game logs per season via LeagueGameLog (2 API calls total).
  - Identifies the top 5 starters per team by average minutes played.
  - Filters to those players' game rows and adds a combined PRA column.

Output: sports_data/raw/nathanbao_nba_raw.csv
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# ── Configuration ─────────────────────────────────────────────────────────────

SEASONS: list[str] = ["2023-24", "2024-25"]
STARTERS_PER_TEAM: int = 5
REQUEST_DELAY: float = 1.0  # seconds between API calls to avoid rate limiting

OUTPUT_DIR: Path = Path(__file__).parent.parent / "sports_data" / "raw"
OUTPUT_FILE: Path = OUTPUT_DIR / "nathanbao_nba_raw.csv"

OUTPUT_COLS: list[str] = [
    "SEASON",
    "TEAM",
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_DATE",
    "MATCHUP",
    "PTS",
    "REB",
    "AST",
    "PRA",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def fetch_season_game_log(season: str) -> pd.DataFrame:
    """Fetch all player game logs for a given regular season."""
    print(f"Fetching game logs for {season} ...")
    time.sleep(REQUEST_DELAY)
    endpoint = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",
    )
    df: pd.DataFrame = endpoint.get_data_frames()[0]
    print(f"  -> {len(df):,} game records retrieved.")
    return df


def identify_top_starters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of (TEAM_ABBREVIATION, PLAYER_ID) pairs for the
    top STARTERS_PER_TEAM players per team ranked by average minutes played.

    LeagueGameLog does not expose START_POSITION, so average minutes per game
    is used as a reliable proxy for identifying the starting five.
    """
    df = df.copy()
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)

    avg_min: pd.DataFrame = (
        df.groupby(["TEAM_ABBREVIATION", "PLAYER_ID"])["MIN"]
        .mean()
        .reset_index(name="AVG_MIN")
        .sort_values("AVG_MIN", ascending=False)
    )

    top_per_team: pd.DataFrame = (
        avg_min.groupby("TEAM_ABBREVIATION")
        .head(STARTERS_PER_TEAM)
        .reset_index(drop=True)
    )

    return top_per_team[["TEAM_ABBREVIATION", "PLAYER_ID"]]


def filter_to_starters(df: pd.DataFrame, top_starters: pd.DataFrame) -> pd.DataFrame:
    """Keep only game rows belonging to the identified top starters."""
    return df.merge(top_starters, on=["TEAM_ABBREVIATION", "PLAYER_ID"])


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    season_frames: list[pd.DataFrame] = []

    for season in SEASONS:
        raw = fetch_season_game_log(season)
        top_starters = identify_top_starters(raw)

        starters_df = filter_to_starters(raw, top_starters).copy()
        starters_df["SEASON"] = season
        starters_df["PRA"] = (
            starters_df["PTS"] + starters_df["REB"] + starters_df["AST"]
        )
        starters_df.rename(columns={"TEAM_ABBREVIATION": "TEAM"}, inplace=True)

        season_frames.append(starters_df[OUTPUT_COLS])
        print(f"  -> {len(starters_df):,} rows kept for top starters in {season}.")

    combined: pd.DataFrame = pd.concat(season_frames, ignore_index=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Saved {len(combined):,} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
