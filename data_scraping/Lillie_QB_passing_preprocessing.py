# data_scraping/Lillie_QB_passing_preprocessing.py
"""
Preprocessing script for NFL QB passing data.

Input:
  sports_data/raw/qb_passing_2023_2024_raw.csv

Output:
  sports_data/processed/qb_passing_2023_2024_processed.csv

Final dataset columns:
  - passing_yards
  - passing_interceptions
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def cap_outliers_3sigma(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Caps values outside mean ± 3 * std (winsorization).
    """
    stats = df.select(
        pl.col(column).mean().alias("mean"),
        pl.col(column).std().alias("std"),
    ).row(0)

    mean, std = stats
    if mean is None or std is None or std == 0:
        return df

    lower = mean - 3 * std
    upper = mean + 3 * std

    return df.with_columns(
        pl.when(pl.col(column) < lower).then(lower)
        .when(pl.col(column) > upper).then(upper)
        .otherwise(pl.col(column))
        .alias(column)
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    raw_path = repo_root / "sports_data" / "raw" / "qb_passing_2023_2024_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = pl.read_csv(raw_path)

    # -------------------------
    # a) Handle Missing Values
    # -------------------------
    df = df.drop_nulls(subset=["passing_yards", "passing_interceptions"])

    # -------------------------
    # b) Remove Duplicates
    # -------------------------
    # If duplicates exist for a team/week, keep the one with most attempts
    df = df.sort(["season", "week", "team", "attempts"], descending=[False, False, False, True])
    df = df.unique(subset=["season", "week", "team"], keep="first")

    # -------------------------
    # c) Data Type Validation
    # -------------------------
    df = df.with_columns(
        [
            pl.col("season").cast(pl.Int32, strict=False),
            pl.col("week").cast(pl.Int32, strict=False),
            pl.col("attempts").cast(pl.Int32, strict=False),
            pl.col("passing_yards").cast(pl.Int32, strict=False),
            pl.col("passing_interceptions").cast(pl.Int32, strict=False),
        ]
    )

    # -------------------------
    # d) Outlier Detection
    # -------------------------
    df = cap_outliers_3sigma(df, "passing_yards")
    df = cap_outliers_3sigma(df, "passing_interceptions")

    # -------------------------
    # e) Consistency Checks
    # -------------------------
    df = df.with_columns(
        [
            pl.col("team").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            pl.col("opponent_team").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            # sanity: no negative values
            pl.when(pl.col("passing_yards") < 0).then(0).otherwise(pl.col("passing_yards")).alias("passing_yards"),
            pl.when(pl.col("passing_interceptions") < 0)
            .then(0)
            .otherwise(pl.col("passing_interceptions"))
            .alias("passing_interceptions"),
        ]
    )

    # Final dataset: ONLY 2 columns (per assignment)
    processed = df.select(["passing_yards", "passing_interceptions"])

    out_dir = repo_root / "sports_data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "qb_passing_2023_2024_processed.csv"
    processed.write_csv(out_path)

    print(f"✅ Processed data written to {out_path}")
    print(f"Rows: {processed.height}, Columns: {len(processed.columns)}")


if __name__ == "__main__":
    main()
