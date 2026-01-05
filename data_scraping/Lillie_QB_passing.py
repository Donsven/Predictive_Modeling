"""
NFL QB Passing Yards + Interceptions data collection.

Outputs RAW CSV to:
  sports_data/raw/qb_passing_2023_2024_raw.csv

Notes:
- Uses weekly player stats from nflreadpy
- "Starting QB" is approximated as the QB with the most pass attempts
  for each (season, week, team)
"""

from pathlib import Path

import nflreadpy as nfl
import polars as pl

SEASONS = [2023, 2024]


def pick_starting_qb_per_team_week(qb_stats: pl.DataFrame) -> pl.DataFrame:
    """
    Select one QB per (season, week, team) based on highest passing attempts.
    """
    required = {
        "season",
        "week",
        "team",
        "attempts",
        "passing_yards",
        "passing_interceptions",
    }
    missing = required - set(qb_stats.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    qb_stats = qb_stats.sort(
        ["season", "week", "team", "attempts"],
        descending=[False, False, False, True],
    )

    return qb_stats.unique(subset=["season", "week", "team"], keep="first")


def main() -> None:
    stats = nfl.load_player_stats(SEASONS)

    qbs = stats.filter(
        (pl.col("position") == "QB")
        & (pl.col("attempts").is_not_null())
        & (pl.col("attempts") > 0)
        & (pl.col("season_type") == "REG")
    )

    starters = pick_starting_qb_per_team_week(qbs)

    raw = starters.select(
        [
            "season",
            "week",
            "team",
            "opponent_team",
            "player_id",
            "player_display_name",
            "attempts",
            "passing_yards",
            "passing_interceptions",
        ]
    )

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "sports_data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "qb_passing_2023_2024_raw.csv"
    raw.write_csv(out_path)

    print(f"✅ Wrote raw data -> {out_path}")
    print(f"Rows: {raw.height}, Columns: {len(raw.columns)}")


if __name__ == "__main__":
    main()
