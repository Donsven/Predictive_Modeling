"""
Pipeline 6: Support Vector Regression (SVR) — NBA PRA Prediction
==================================================================
Owner: Sri Rama

SVR uses kernel functions to find non-linear patterns. The RBF kernel
maps data into a higher-dimensional space where linear separation is
possible. Most mathematically complex model in our lineup.
"""

from data_loader import evaluate_model, prepare_data
from sklearn.svm import SVR


def run_pipeline():
    # --- Load & prep data ---
    data = prepare_data(scale=True)  # SVR is very sensitive to scale
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # --- Train with different kernels ---
    kernels = {
        "linear": SVR(kernel="linear", C=1.0),
        "rbf": SVR(kernel="rbf", C=10.0, gamma="scale"),
        "poly": SVR(kernel="poly", degree=2, C=1.0),
    }

    best_kernel, best_r2 = None, float("-inf")
    all_metrics = {}

    for name, model in kernels.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, model_name=f"SVR ({name})")
        all_metrics[name] = metrics
        if metrics["r2"] > best_r2:
            best_kernel, best_r2 = name, metrics["r2"]

    print(f"\nBest kernel: {best_kernel} with R² = {best_r2:.4f}")

    # --- Support vector analysis ---
    best_model = kernels[best_kernel]
    n_sv = sum(best_model.n_support_)
    print(f"Number of support vectors: {n_sv}")
    print(f"Support vector ratio: {n_sv / len(X_train) * 100:.1f}% of training data")

    return kernels[best_kernel], all_metrics


# =============================================================================
# TODOs for Sri
# =============================================================================
#
# TODO 1: Hyperparameter tuning for C and gamma
#   - C controls the trade-off between margin width and training error
#   - gamma controls how much influence each training point has
#   - Use GridSearchCV with:
#     C = [0.1, 1, 10, 100]
#     gamma = ['scale', 'auto', 0.01, 0.1, 1.0]
#   - Plot a heatmap of C vs gamma performance
#
# TODO 2: Kernel comparison visualization
#   - For 2 features at a time, plot the decision boundary for each kernel
#   - Shows how linear, RBF, and polynomial kernels see the data differently
#   - Save to Sports_predictions/plots/svr_kernels.png
#
# TODO 3: Compare SVR with other pipelines
#   - Write a comparison script that loads all 6 trained models
#   - Run each on the same test set
#   - Create a bar chart comparing MAE, RMSE, and R² across all models
#   - Save to Sports_predictions/plots/model_comparison.png
#
# =============================================================================


if __name__ == "__main__":
    run_pipeline()
