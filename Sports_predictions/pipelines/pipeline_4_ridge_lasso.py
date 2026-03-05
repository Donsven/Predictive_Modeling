"""
Pipeline 4: Ridge & Lasso Regression — NBA PRA Prediction
==========================================================
Owner: Nathan Bao

Regularized linear models. Ridge (L2) shrinks coefficients toward zero.
Lasso (L1) can zero out coefficients entirely, performing feature selection.
Great for understanding which features actually matter.
"""

from data_loader import evaluate_model, prepare_data
from sklearn.linear_model import Lasso, Ridge


def run_pipeline():
    # --- Load & prep data ---
    data = prepare_data(scale=True)  # Regularized models benefit from scaling
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # --- Train Ridge ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_metrics = evaluate_model(y_test, ridge_pred, model_name="Ridge Regression")

    print("Ridge Coefficients:")
    for name, coef in zip(data["feature_names"], ridge.coef_, strict=False):
        print(f"  {name}: {coef:.4f}")

    # --- Train Lasso ---
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_metrics = evaluate_model(y_test, lasso_pred, model_name="Lasso Regression")

    print("Lasso Coefficients:")
    for name, coef in zip(data["feature_names"], lasso.coef_, strict=False):
        marker = " (zeroed out!)" if abs(coef) < 1e-6 else ""
        print(f"  {name}: {coef:.4f}{marker}")

    # --- Compare ---
    print("\n--- Ridge vs Lasso Comparison ---")
    print(
        f"  Ridge R²: {ridge_metrics['r2']:.4f}  |  Lasso R²: {lasso_metrics['r2']:.4f}"
    )
    ridge_mae = ridge_metrics["mae"]
    lasso_mae = lasso_metrics["mae"]
    print(f"  Ridge MAE: {ridge_mae:.3f} |  Lasso MAE: {lasso_mae:.3f}")

    return {"ridge": ridge, "lasso": lasso}, {
        "ridge": ridge_metrics,
        "lasso": lasso_metrics,
    }


# =============================================================================
# TODOs for Nathan
# =============================================================================
#
# TODO 1: Alpha tuning with RidgeCV and LassoCV
#   - from sklearn.linear_model import RidgeCV, LassoCV
#   - These automatically find the best alpha using cross-validation
#   - Try alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
#   - Print the best alpha found for each
#
# TODO 2: Regularization path visualization
#   - Plot how coefficients change as alpha increases
#   - For Ridge: coefficients shrink smoothly toward zero
#   - For Lasso: coefficients drop to exactly zero at different alphas
#   - This is one of the most instructive ML visualizations
#   - Save to Sports_predictions/plots/regularization_path.png
#
# TODO 3: ElasticNet — best of both worlds
#   - from sklearn.linear_model import ElasticNet, ElasticNetCV
#   - ElasticNet combines L1 and L2 regularization
#   - Tune both alpha and l1_ratio (0 = Ridge, 1 = Lasso)
#   - Compare all three: Ridge vs Lasso vs ElasticNet
#
# =============================================================================


if __name__ == "__main__":
    run_pipeline()
