"""
Pipeline 3: Gradient Boosting — NBA PRA Prediction
====================================================
Owner: Logan Tadano

Gradient boosting builds trees sequentially, where each tree
corrects errors from the previous one. Often the best performer
among classical ML models.
"""

from data_loader import evaluate_model, prepare_data
from sklearn.ensemble import GradientBoostingRegressor


def run_pipeline():
    # --- Load & prep data ---
    data = prepare_data(scale=False)  # Tree-based, no scaling needed
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # --- Train ---
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, model_name="Gradient Boosting")

    # --- Feature importance ---
    print("Feature Importances:")
    for name, imp in sorted(
        zip(data["feature_names"], model.feature_importances_, strict=False),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {name}: {imp:.4f}")

    # --- Training loss curve ---
    print(f"\nTraining stages: {model.n_estimators_}")
    print(f"Final training loss: {model.train_score_[-1]:.6f}")

    return model, metrics


# =============================================================================
# TODOs for Logan
# =============================================================================
#
# TODO 1: Plot the learning curve (training loss vs n_estimators)
#   - model.train_score_ gives you the training loss at each stage
#   - Use staged_predict() on test data to get test loss at each stage
#   - Plot both curves — where they diverge shows overfitting
#   - Save to Sports_predictions/plots/gb_learning_curve.png
#
# TODO 2: Try XGBoost as an alternative
#   - pip install xgboost
#   - from xgboost import XGBRegressor
#   - Compare XGBoost vs sklearn's GradientBoostingRegressor
#   - XGBoost often trains faster and has built-in regularization
#
# TODO 3: SHAP values for model interpretability
#   - pip install shap
#   - SHAP values show WHY the model made each prediction
#   - Create a SHAP summary plot and a SHAP force plot for a specific player
#   - Save to Sports_predictions/plots/
#
# =============================================================================


if __name__ == "__main__":
    run_pipeline()
