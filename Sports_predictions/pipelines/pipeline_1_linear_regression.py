"""
Pipeline 1: Linear Regression — NBA PRA Prediction
====================================================
Owner: Abby Humphrey

Baseline model using ordinary least squares regression.
This is the simplest approach and serves as the benchmark
that all other pipelines should beat.
"""

from data_loader import evaluate_model, prepare_data
from sklearn.linear_model import LinearRegression


def run_pipeline():
    # --- Load & prep data ---
    data = prepare_data(scale=False)  # Linear regression doesn't need scaling
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # --- Train ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, model_name="Linear Regression")

    # --- Feature importance (coefficients) ---
    print("Feature Coefficients:")
    for name, coef in zip(data["feature_names"], model.coef_, strict=False):
        print(f"  {name}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    return model, metrics


# =============================================================================
# TODOs for Abby
# =============================================================================
#
# TODO 1: Add more features to improve predictions
#   - Pull additional stats from nba_api (FG%, 3P%, FT%, STL, BLK, TOV)
#   - Update data_loader.build_features() or create your own feature builder
#   - Hint: from nba_api.stats.endpoints import playergamelog
#
# TODO 2: Implement cross-validation instead of a single train/test split
#   - Use sklearn.model_selection.cross_val_score with cv=5 or cv=10
#   - Report mean and std of each metric across folds
#
# TODO 3: Add residual analysis and visualization
#   - Plot residuals (y_test - y_pred) vs predicted values
#   - Check for heteroscedasticity (residuals should be randomly scattered)
#   - Use matplotlib to create the plots, save to Sports_predictions/plots/
#
# =============================================================================


if __name__ == "__main__":
    run_pipeline()
