"""
Pipeline 5: K-Nearest Neighbors — NBA PRA Prediction
======================================================
Owner: Samantha Bai

KNN predicts by finding the K most similar players (based on stats)
and averaging their PRA. Intuitive: "players with similar stats
produce similar PRA."
"""

import numpy as np
from data_loader import evaluate_model, prepare_data
from sklearn.neighbors import KNeighborsRegressor


def run_pipeline():
    # --- Load & prep data ---
    data = prepare_data(scale=True)  # KNN is distance-based, MUST scale
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # --- Try multiple K values ---
    best_k, best_r2 = 1, -np.inf
    results = []

    for k in [3, 5, 7, 10, 15, 20]:
        model = KNeighborsRegressor(n_neighbors=k, weights="distance")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, model_name=f"KNN (k={k})")
        results.append((k, metrics))
        if metrics["r2"] > best_r2:
            best_k, best_r2 = k, metrics["r2"]

    print(f"\nBest K: {best_k} with R² = {best_r2:.4f}")

    # --- Retrain with best K ---
    final_model = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
    final_model.fit(X_train, y_train)

    # --- Show nearest neighbors for a top player ---
    if len(X_test) > 0:
        idx = 0
        distances, indices = final_model.kneighbors(X_test.iloc[[idx]])
        print(f"\nNearest neighbors for test player index {idx}:")
        for dist, train_idx in zip(distances[0], indices[0], strict=False):
            player_row = data["df"].iloc[X_train.index[train_idx]]
            name = player_row["PLAYER"]
            pra = player_row["PRA"]
            print(f"  {name} (distance: {dist:.3f}, PRA: {pra:.1f})")

    return final_model, results


# =============================================================================
# TODOs for Samantha
# =============================================================================
#
# TODO 1: Optimal K selection with cross-validation
#   - Use sklearn.model_selection.cross_val_score for each K value
#   - Plot K vs cross-validated R² (the "elbow" plot)
#   - Find the optimal K more rigorously than the loop above
#   - Save plot to Sports_predictions/plots/knn_k_selection.png
#
# TODO 2: Experiment with different distance metrics
#   - Try metric="manhattan", metric="minkowski" with different p values
#   - Try weights="uniform" vs weights="distance"
#   - Which distance metric works best for basketball stats?
#
# TODO 3: Build a "player similarity" tool
#   - Given a player name, find their K nearest neighbors
#   - Display: similar players, their stats, and PRA comparison
#   - This is actually useful for scouting and player comparison!
#   - Example output: "Players most similar to Luka Doncic: ..."
#
# =============================================================================


if __name__ == "__main__":
    run_pipeline()
