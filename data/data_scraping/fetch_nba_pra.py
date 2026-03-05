"""
Fetch current NBA season player PRA (Points + Rebounds + Assists) per game.
Uses the nba_api library to pull from NBA.com stats endpoints.
"""

import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import teams


def fetch_player_pra(season: str = "2025-26") -> pd.DataFrame:
    """
    Fetch per-game stats for all NBA players in a given season
    and compute PRA (Points + Rebounds + Assists) per game.

    Args:
        season: NBA season string, e.g. "2025-26"

    Returns:
        DataFrame with player name, team, GP, PPG, RPG, APG, and PRA per game.
    """
    print(f"Fetching player stats for {season} season...")
    time.sleep(1)  # rate-limit courtesy

    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
    )
    df = stats.get_data_frames()[0]

    # Build a team abbreviation lookup
    nba_teams = {t["id"]: t["abbreviation"] for t in teams.get_teams()}

    result = pd.DataFrame(
        {
            "PLAYER_ID": df["PLAYER_ID"],
            "PLAYER": df["PLAYER_NAME"],
            "TEAM": df["TEAM_ID"].map(nba_teams),
            "GP": df["GP"],
            "MIN": df["MIN"],
            "PPG": df["PTS"],
            "RPG": df["REB"],
            "APG": df["AST"],
            "PRA": df["PTS"] + df["REB"] + df["AST"],
        }
    )

    # Filter to players with meaningful minutes (at least 10 games, 10 min/game)
    result = result[(result["GP"] >= 10) & (result["MIN"] >= 10.0)]
    result = result.sort_values("PRA", ascending=False).reset_index(drop=True)

    return result


def save_data(df: pd.DataFrame, filename: str = "nba_pra_current_season.csv") -> str:
    """Save the DataFrame to the sports_data/raw directory."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "sports_data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} players to {filepath}")
    return filepath


if __name__ == "__main__":
    df = fetch_player_pra()
    print("\nTop 20 NBA Players by PRA per game:\n")
    print(
        df.head(20).to_string(
            index=False,
            columns=["PLAYER", "TEAM", "GP", "PPG", "RPG", "APG", "PRA"],
        )
    )
    save_data(df)
