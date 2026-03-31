from pathlib import Path

import pandas as pd

csv_path = r"C:\Users\mammo\Documents\AISC\OilFutures-API\Logan_OilFuturesRaw.csv"

# Load and convert Date strings to date-time objects
df = pd.read_csv(csv_path, parse_dates=["date", "expiration"])

# Convert columns from string → datetime64[ns]
df["date"] = pd.to_datetime(df["date"])
df["expiration"] = pd.to_datetime(df["expiration"])


# Outlier detection beyond 3 standard deviations from the mean
num_df = df.select_dtypes(include=["float64", "int64"])
means = num_df.mean()
sstds = num_df.std(ddof=0)

# Z-score calculation
z_scores = (num_df - means) / sstds

# Filter rows with any z-score > 3 or < -3
outlier_mask = (z_scores.abs() > 3).any(axis=1)
outliers = df[outlier_mask]

# Drop the outliers from the original dataframe
df_no_outliers = df[~outlier_mask].copy()
df = df_no_outliers

BASE_DIR = Path(r"C:\Users\mammo\Documents\AISC\Predictive_Modeling")
out_path = (
    BASE_DIR / "data" / "finance_data" / "processed" / "Logan_OilFuturesProcessed.csv"
)
df.to_csv(out_path, index=False)
