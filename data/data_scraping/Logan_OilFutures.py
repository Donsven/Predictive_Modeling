import datetime as dt
import os
from pathlib import Path

import databento as db

# -------------------
# 1. Config
# -------------------
API_Key = "db-58uvdCKemq77SsUNGXEhs3jQd4Sv5"
DATASET = "GLBX.MDP3"  # CME Globex futures dataset
SYMBOL = "CL.n.0"  # Front-month continuous WTI Crude futures
STYPE_IN = "continuous"
SCHEMA = "ohlcv-1d"

today = dt.date.today()
start_date = (today - dt.timedelta(days=365 * 2)).isoformat()
end_date = today.isoformat()

client = db.Historical(API_Key)  # uses DATABENTO_API_KEY env var

# -------------------
#  Daily OHLCV data
# -------------------
bars = client.timeseries.get_range(
    dataset=DATASET,
    symbols=[SYMBOL],
    stype_in=STYPE_IN,
    schema=SCHEMA,
    start=start_date,
    end=end_date,
)

bars_df = bars.to_df()
bars_df["date"] = bars_df.index.date


# ---------------
# Get Expiration Dates
# -------------------
defs = client.timeseries.get_range(
    dataset=DATASET,
    symbols="ALL_SYMBOLS",
    schema="definition",
    start=start_date,
)
defs_df = defs.to_df()


# Filter to CL futures only
fut_df = defs_df[defs_df["instrument_class"] == db.InstrumentClass.FUTURE]
cl_defs = fut_df[fut_df["asset"] == "CL"].copy()

# Keep only what we need + drop duplicate instrument_ids
cl_defs = cl_defs[["instrument_id", "raw_symbol", "expiration"]].drop_duplicates(
    "instrument_id"
)

# -------------------
#  Join expirations onto OHLCV
# -------------------
merged = bars_df.merge(
    cl_defs,
    on="instrument_id",
    how="left",
)


# -------------------
#  Keep only your requested fields
# -------------------
final_df = merged[
    [
        "date",  # day of the bar
        "open",  # daily open price
        "high",  # daily high price
        "low",  # daily low price
        "close",  # daily closing price
        "volume",  # daily trading volume
        "expiration",  # contract expiration date
    ]
]


# Choose where you want to save it
BASE_DIR = Path(r"C:\Users\mammo\Documents\AISC\Predictive_Modeling")
output_path = BASE_DIR / "data" / "finance_data" / "raw" / "Logan_OilFuturesRaw.csv"
# Make sure the folder exists (optional but nice)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save to CSV (no index column in the file)
final_df.to_csv(output_path, index=False)

print(f"Saved CSV to: {output_path}")
