"""
Pipeline 4: Ridge & Lasso Regression — NBA PRA Prediction
==========================================================
Owner: Nathan Bao

Regularized linear models. Ridge (L2) shrinks coefficients toward zero.
Lasso (L1) can zero out coefficients entirely, performing feature selection.
Great for understanding which features actually matter.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from data_loader import evaluate_model, prepare_data
from sklearn.linear_model import (
    ElasticNetCV,
    Lasso,
    LassoCV,
    Ridge,
    RidgeCV,
    lasso_path,
)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def run_pipeline():
    # --- Load & prep data ---
    data = prepare_data(scale=True)  # Regularized models benefit from scaling
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    feature_names = data["feature_names"]

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- Find best alphas with CV instead of hardcoding ---
    ridge_cv = RidgeCV(alphas=ALPHAS, cv=5)
    ridge_cv.fit(X_train, y_train)
    best_ridge_alpha = ridge_cv.alpha_
    print(f"Best Ridge alpha: {best_ridge_alpha}")

    lasso_cv = LassoCV(alphas=ALPHAS, cv=5, random_state=42)
    lasso_cv.fit(X_train, y_train)
    best_lasso_alpha = lasso_cv.alpha_
    print(f"Best Lasso alpha: {best_lasso_alpha}")

    # --- Train Ridge ---
    ridge = Ridge(alpha=best_ridge_alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_metrics = evaluate_model(y_test, ridge_pred, model_name="Ridge Regression")

    print("Ridge Coefficients:")
    for name, coef in zip(feature_names, ridge.coef_, strict=False):
        print(f"  {name}: {coef:.4f}")

    # --- Train Lasso ---
    lasso = Lasso(alpha=best_lasso_alpha)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_metrics = evaluate_model(y_test, lasso_pred, model_name="Lasso Regression")

    print("Lasso Coefficients:")
    for name, coef in zip(feature_names, lasso.coef_, strict=False):
        marker = " (zeroed out!)" if abs(coef) < 1e-6 else ""
        print(f"  {name}: {coef:.4f}{marker}")

    # --- Regularization path — see which features survive as alpha grows ---
    _plot_regularization_paths(X_train, y_train, feature_names)

    # --- ElasticNet — mix of L1 + L2, see if it beats either alone ---
    enet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        alphas=ALPHAS,
        cv=5,
        random_state=42,
    )
    enet_cv.fit(X_train, y_train)
    enet_pred = enet_cv.predict(X_test)
    enet_metrics = evaluate_model(y_test, enet_pred, model_name="ElasticNet")

    print(f"Best ElasticNet alpha: {enet_cv.alpha_}, l1_ratio: {enet_cv.l1_ratio_}")
    print("ElasticNet Coefficients:")
    for name, coef in zip(feature_names, enet_cv.coef_, strict=False):
        marker = " (zeroed out!)" if abs(coef) < 1e-6 else ""
        print(f"  {name}: {coef:.4f}{marker}")

    # --- Compare all three ---
    print("\n--- Ridge vs Lasso vs ElasticNet ---")
    for label, m in [
        ("Ridge", ridge_metrics),
        ("Lasso", lasso_metrics),
        ("ElasticNet", enet_metrics),
    ]:
        print(f"  {label:<10} R²: {m['r2']:.4f}  |  MAE: {m['mae']:.3f}")

    return {"ridge": ridge, "lasso": lasso, "elasticnet": enet_cv}, {
        "ridge": ridge_metrics,
        "lasso": lasso_metrics,
        "elasticnet": enet_metrics,
    }


def _plot_regularization_paths(X_train, y_train, feature_names):
    """Show how coefficients change as regularization gets stronger."""
    alphas_path = np.logspace(-3, 2, 100)

    # Ridge: no built-in path function so just loop over alphas
    ridge_coefs = []
    for a in alphas_path:
        r = Ridge(alpha=a)
        r.fit(X_train, y_train)
        ridge_coefs.append(r.coef_)
    ridge_coefs = np.array(ridge_coefs)

    # Lasso: use the built-in path — coefs shape is (n_features, n_alphas)
    # lasso_path returns alphas in decreasing order, sort ascending to match ridge
    alphas_lasso, lasso_coefs, _ = lasso_path(X_train, y_train, alphas=alphas_path)
    order = np.argsort(alphas_lasso)
    alphas_lasso = alphas_lasso[order]
    lasso_coefs = lasso_coefs[:, order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, name in enumerate(feature_names):
        ax1.plot(alphas_path, ridge_coefs[:, i], label=name)
    ax1.set_xscale("log")
    ax1.set_xlabel("Alpha (stronger regularization →)")
    ax1.set_ylabel("Coefficient value")
    ax1.set_title("Ridge: Coefficients shrink smoothly")
    ax1.legend()

    for i, name in enumerate(feature_names):
        ax2.plot(alphas_lasso, lasso_coefs[i], label=name)
    ax2.set_xscale("log")
    ax2.set_xlabel("Alpha (stronger regularization →)")
    ax2.set_ylabel("Coefficient value")
    ax2.set_title("Lasso: Coefficients drop to exactly zero")
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "regularization_path.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved regularization path plot → {save_path}")


if __name__ == "__main__":
    run_pipeline()
