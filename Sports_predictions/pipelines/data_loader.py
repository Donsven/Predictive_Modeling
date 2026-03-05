"""
Shared data loading and preprocessing for NBA PRA prediction pipelines.

All 6 ML pipelines import from this module so data prep is consistent.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "sports_data",
    "raw",
    "nba_pra_current_season.csv",
)


def load_pra_data() -> pd.DataFrame:
    """Load the NBA PRA dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data not found at {DATA_PATH}. "
            "Run data/data_scraping/fetch_nba_pra.py first."
        )
    return pd.read_csv(DATA_PATH)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target vector y from the PRA dataset.

    Current features: GP, MIN, PPG, RPG, APG
    Target: PRA (Points + Rebounds + Assists per game)

    NOTE: With the current dataset, PRA = PPG + RPG + APG exactly.
    This is intentionally simple as a starting point. The real value
    comes when you extend the features (see TODOs in each pipeline).
    """
    feature_cols = ["GP", "MIN", "PPG", "RPG", "APG"]
    X = df[feature_cols].copy()
    y = df["PRA"].copy()
    return X, y


def prepare_data(
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> dict:
    """
    Full data prep pipeline: load, feature engineer, split, optionally scale.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test, feature_names, scaler (or None),
        df (original DataFrame for reference)
    """
    df = load_pra_data()
    X, y = build_features(df)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=feature_names, index=X_test.index
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "df": df,
    }


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> dict:
    """Print and return standard regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'=' * 50}")
    print(f" {model_name} — Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²:   {r2:.4f}")
    print(f"{'=' * 50}\n")

    return {"mae": mae, "rmse": rmse, "r2": r2}
