"""
Pipeline 2: Random Forest — NBA PRA Prediction
================================================
Owner: Lillie Vehling

Ensemble tree-based model. Should capture non-linear relationships
better than linear regression.
"""

from data_loader import evaluate_model, prepare_data
from sklearn.ensemble import RandomForestRegressor


def run_pipeline():
    # --- Load & prep data ---
    data = prepare_data(scale=False)  # Tree-based models don't need scaling
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # --- Train ---
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, model_name="Random Forest")

    # --- Feature importance ---
    print("Feature Importances:")
    for name, imp in sorted(
        zip(data["feature_names"], model.feature_importances_, strict=False),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {name}: {imp:.4f}")

    return model, metrics


# =============================================================================
# TODOs for Lillie
# =============================================================================
#
# TODO 1: Hyperparameter tuning
#   - Use sklearn.model_selection.GridSearchCV or RandomizedSearchCV
#   - Key params to tune: n_estimators, max_depth, min_samples_split,
#     min_samples_leaf, max_features
#   - Print the best params found
#
# TODO 2: Add advanced features from game logs
#   - Use nba_api.stats.endpoints.playergamelog to get per-game data
#   - Engineer features like: rolling averages (last 5/10 games),
#     home vs away splits, back-to-back game performance
#   - These temporal features are where Random Forest can really shine
#
# TODO 3: Feature importance visualization
#   - Create a horizontal bar chart of feature importances
#   - Use matplotlib, save to Sports_predictions/plots/rf_feature_importance.png
#   - Compare with the linear regression coefficients from Pipeline 1
#
# =============================================================================


if __name__ == "__main__":
    run_pipeline()
