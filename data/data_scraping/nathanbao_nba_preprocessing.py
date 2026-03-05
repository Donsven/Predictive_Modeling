"""
nathanbao_nba_preprocessing.py

Cleans and preprocesses raw NBA PRA data produced by nathanbao_nba_pra.py.

Steps:
  1. Handle missing values  -- drop rows where core stats are null
  2. Remove duplicates      -- deduplicate on (PLAYER_ID, GAME_DATE)
  3. Validate data types    -- parse dates; coerce stats to numeric
  4. Cap outliers           -- clip values beyond +/-3 standard deviations
  5. Consistency checks     -- standardize team/player names; recompute PRA

Input:  sports_data/raw/nathanbao_nba_pra_raw.csv
Output: sports_data/processed/nathanbao_nba_pra_processed.csv
"""

from pathlib import Path

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_FILE: Path = (
    Path(__file__).parent.parent / "sports_data" / "raw" / "nathanbao_nba_raw.csv"
)
OUTPUT_DIR: Path = Path(__file__).parent.parent / "sports_data" / "processed"
OUTPUT_FILE: Path = OUTPUT_DIR / "nathanbao_nba_processed.csv"

STAT_COLS: list[str] = ["PTS", "REB", "AST", "PRA"]
OUTLIER_STD: float = 3.0

# ── Steps ─────────────────────────────────────────────────────────────────────


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with nulls in core stat columns and report counts."""
    null_counts = df[STAT_COLS].isnull().sum()
    affected = null_counts[null_counts > 0]
    if not affected.empty:
        print(f"  Null values detected:\n{affected.to_string()}")

    before = len(df)
    df = df.dropna(subset=STAT_COLS)
    print(f"  Dropped {before - len(df)} rows with missing stats.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate game entries based on player + date."""
    before = len(df)
    df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_DATE"])
    print(f"  Removed {before - len(df)} duplicate rows.")
    return df


def validate_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to correct dtypes; drop rows that fail conversion."""
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["GAME_DATE", *STAT_COLS])
    print(f"  {before - len(df)} rows dropped after type coercion.")

    date_min = df["GAME_DATE"].min()
    date_max = df["GAME_DATE"].max()
    print(f"  GAME_DATE range: {date_min} -> {date_max}")
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap values beyond +/-OUTLIER_STD standard deviations to the boundary."""
    for col in STAT_COLS:
        mean = df[col].mean()
        std = df[col].std()
        lo = mean - OUTLIER_STD * std
        hi = mean + OUTLIER_STD * std
        n_outliers = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_outliers:
            print(f"  {col}: {n_outliers} outlier(s) capped to [{lo:.2f}, {hi:.2f}]")
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize team/player name formatting and validate stat ranges."""
    df["TEAM"] = df["TEAM"].str.strip().str.upper()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].str.strip().str.title()

    for col in ["PTS", "REB", "AST"]:
        neg_mask = df[col] < 0
        if neg_mask.any():
            print(
                f"  WARNING: {neg_mask.sum()} row(s) with negative {col} -- dropping."
            )
            df = df[~neg_mask]

    # Recompute PRA to guarantee consistency after capping/dropping
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_FILE)

    print("\n[1] Missing values")
    df = handle_missing_values(df)

    print("\n[2] Duplicates")
    df = remove_duplicates(df)

    print("\n[3] Type validation")
    df = validate_types(df)

    print("\n[4] Outlier detection")
    df = cap_outliers(df)

    print("\n[5] Consistency checks")
    df = consistency_checks(df)

    df = df.sort_values(["TEAM", "PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Saved {len(df):,} cleaned rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
