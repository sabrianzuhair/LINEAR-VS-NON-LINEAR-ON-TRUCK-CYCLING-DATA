# model_02_decision_tree.py — Decision Tree regression for truck cycle_time.

# SECTION 0 — IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    t_script = time.time()
    print(f"[START] Decision Tree Analysis")

    SEED = 42
    np.random.seed(SEED)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    COLORS = {
        "primary": "#2E6DB4",
        "secondary": "#1A7A6E",
        "accent": "#E07B39",
        "danger": "#C0392B",
        "light": "#D6E8F7",
    }

    # SECTION 1 — LOAD PREPROCESSED DATA
    try:
        X_train = pd.read_csv("data_processed/X_train.csv")
        X_val = pd.read_csv("data_processed/X_val.csv")
        X_test = pd.read_csv("data_processed/X_test.csv")
        y_train = pd.read_csv("data_processed/y_train.csv").squeeze()
        y_val = pd.read_csv("data_processed/y_val.csv").squeeze()
        y_test = pd.read_csv("data_processed/y_test.csv").squeeze()
        with open("data_processed/feature_cols.json") as f:
            FEATURE_COLS = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError("ERROR: Run preprocessing.py first") from e

    if FEATURE_COLS:
        X_train = X_train[FEATURE_COLS]
        X_val = X_val[FEATURE_COLS]
        X_test = X_test[FEATURE_COLS]

    print(f"[INFO] Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"[INFO] Features: {FEATURE_COLS}")
    print(
        f"[INFO] Target mean: {y_train.mean():.2f} min, std: {y_train.std():.2f} min"
    )

    # SECTION 2 — BASELINE (UNPRUNED) DECISION TREE
    dt_baseline = DecisionTreeRegressor(random_state=SEED)
    dt_baseline.fit(X_train, y_train)

    r2_train_base = r2_score(y_train, dt_baseline.predict(X_train))
    r2_test_base = r2_score(y_test, dt_baseline.predict(X_test))

    print(
        f"""
[BASELINE — Unpruned Tree]
Train R²  : {r2_train_base:.4f}
Test  R²  : {r2_test_base:.4f}
Tree depth: {dt_baseline.get_depth()}
Leaf nodes: {dt_baseline.get_n_leaves()}
→ Likely shows overfitting (train R² >> test R²)
"""
    )

    # SECTION 3 — HYPERPARAMETER TUNING WITH GridSearchCV
    param_grid = {
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
    }

    dt_cv = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=SEED),
        param_grid=param_grid,
        cv=10,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    start_time = time.time()
    dt_cv.fit(X_train, y_train)
    tune_time = time.time() - start_time

    print(f"[INFO] GridSearchCV completed in {tune_time:.2f} seconds")
    print(f"[INFO] Best parameters: {dt_cv.best_params_}")
    print(f"[INFO] Best CV R² (10-fold): {dt_cv.best_score_:.4f}")

    cv_results = pd.DataFrame(dt_cv.cv_results_)
    top10 = cv_results.nlargest(10, "mean_test_score")[
        ["params", "mean_test_score", "std_test_score", "mean_train_score"]
    ]
    print(f"Top 10 Parameter Combinations:")
    print(top10.to_string())

    # SECTION 4 — FINAL MODEL EVALUATION
    dt_best = dt_cv.best_estimator_

    y_pred_train = dt_best.predict(X_train)
    y_pred_val = dt_best.predict(X_val)
    y_pred_test = dt_best.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    best_params = dt_cv.best_params_
    best_depth = best_params.get("max_depth")
    best_leaf = best_params.get("min_samples_leaf")
    gap = r2_train - r2_test

    print(
        f"""
┌──────────────────────────────────────────────────┐
│  DECISION TREE — EVALUATION RESULTS              │
├──────────────────────────────────────────────────┤
│  Best max_depth     : {best_depth}               │
│  Best leaf size     : {best_leaf}                │
│  Tree depth (actual): {dt_best.get_depth()}      │
│  Number of leaves   : {dt_best.get_n_leaves()}   │
├──────────────────────────────────────────────────┤
│  Train R²  : {r2_train:.4f}                      │
│  Val   R²  : {r2_val:.4f}                        │
│  Test  R²  : {r2_test:.4f}                       │
│  Test  MAE : {mae_test:.4f} minutes              │
│  Test  RMSE: {rmse_test:.4f} minutes             │
├──────────────────────────────────────────────────┤
│  Overfitting gap (Train-Test R²): {gap:.4f}      │
└──────────────────────────────────────────────────┘
"""
    )
    print(
        f"Test R² improvement from tuning: +{r2_test - r2_test_base:.4f}"
    )

    cv_scores = cross_val_score(
        dt_best, X_train, y_train, cv=10, scoring="r2", n_jobs=-1
    )
    print(
        f"10-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )
    print(f"CV scores: {[round(s, 4) for s in cv_scores]}")

    results = {
        "Model": "Decision Tree",
        "R2_Train": r2_train,
        "R2_Val": r2_val,
        "R2_Test": r2_test,
        "MAE": mae_test,
        "RMSE": rmse_test,
        "CV_R2_Mean": cv_scores.mean(),
        "CV_R2_Std": cv_scores.std(),
        "Best_Params": str(dt_cv.best_params_),
        "Tree_Depth": dt_best.get_depth(),
        "N_Leaves": dt_best.get_n_leaves(),
        "Tuning_Time": tune_time,
    }
    pd.DataFrame([results]).to_csv("results/dt_results.csv", index=False)

    # SECTION 5 — FEATURE IMPORTANCE ANALYSIS
    importances = dt_best.feature_importances_
    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLS,
            "Importance": importances,
            "Rank": range(1, len(FEATURE_COLS) + 1),
        }
    ).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance_df["Rank"] = range(1, len(importance_df) + 1)
    importance_df["Cumulative"] = importance_df["Importance"].cumsum()

    print(f"Feature Importance Ranking (Gini Impurity Reduction):")
    print(
        importance_df[["Rank", "Feature", "Importance", "Cumulative"]].to_string(
            index=False
        )
    )

    top_feature = importance_df.iloc[0]["Feature"]
    top_pct = importance_df.iloc[0]["Importance"] * 100
    print(
        f"Most important feature: {top_feature} ({top_pct:.1f}% of importance)"
    )

    features_80 = importance_df[importance_df["Cumulative"] <= 0.80]
    print(
        f"Features capturing 80% of importance: {len(features_80)}"
    )

    # SECTION 6 — VISUALIZATIONS
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("default")

    # PLOT 1 — Tree structure (depth=4 for viz)
    fig, ax = plt.subplots(figsize=(28, 12))
    viz_params = {
        k: v for k, v in dt_cv.best_params_.items() if k != "max_depth"
    }
    dt_viz = DecisionTreeRegressor(
        max_depth=4, random_state=SEED, **viz_params
    )
    dt_viz.fit(X_train, y_train)

    plot_tree(
        dt_viz,
        feature_names=FEATURE_COLS,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
        impurity=True,
        precision=2,
    )
    plt.title(
        f"Decision Tree Structure (visualized at depth=4, actual depth={dt_best.get_depth()})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "figures/dt_01_tree_structure.png", dpi=120, bbox_inches="tight"
    )
    plt.close()
    print(f"[SAVED] figures/dt_01_tree_structure.png")

    # PLOT 2 — Feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    n_feat = len(importance_df)
    bar_colors = [
        COLORS["primary"] if i < 3 else COLORS["light"]
        for i in range(n_feat - 1, -1, -1)
    ]
    bars = ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val in zip(bars, importance_df["Importance"][::-1]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Gini Importance Score", fontsize=11)
    ax.set_title(
        "Decision Tree: Feature Importance (Gini Impurity Reduction)",
        fontsize=13,
        fontweight="bold",
    )
    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(
        "figures/dt_02_feature_importance.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"[SAVED] figures/dt_02_feature_importance.png")

    # PLOT 3 — Actual vs Predicted
    fig, ax = plt.subplots(figsize=(9, 8))
    hb = ax.hexbin(
        y_test, y_pred_test, gridsize=50, cmap="Blues", mincnt=1
    )
    plt.colorbar(hb, ax=ax, label="Count")
    min_val = min(float(y_test.min()), float(np.min(y_pred_test)))
    max_val = max(float(y_test.max()), float(np.max(y_pred_test)))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    ax.set_xlabel("Actual cycle_time (minutes)", fontsize=11)
    ax.set_ylabel("Predicted cycle_time (minutes)", fontsize=11)
    ax.set_title(
        f"Decision Tree: Actual vs Predicted\nR²={r2_test:.4f} | MAE={mae_test:.4f} | RMSE={rmse_test:.4f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        "figures/dt_03_actual_vs_predicted.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"[SAVED] figures/dt_03_actual_vs_predicted.png")

    # PLOT 4 — Before vs After tuning
    fig, ax = plt.subplots(figsize=(9, 6))
    categories = ["Baseline\n(Unpruned)", "Tuned\n(GridSearchCV)"]
    train_r2_vals = [r2_train_base, r2_train]
    test_r2_vals = [r2_test_base, r2_test]
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        train_r2_vals,
        width,
        label="Train R²",
        color=COLORS["primary"],
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2,
        test_r2_vals,
        width,
        label="Test R²",
        color=COLORS["accent"],
        alpha=0.85,
    )
    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("R² Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(
        y=0.8,
        color="green",
        linestyle="--",
        alpha=0.6,
        label="R²=0.8 threshold",
    )
    ax.set_title(
        "Decision Tree: Before vs After Hyperparameter Tuning",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        "figures/dt_04_tuning_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"[SAVED] figures/dt_04_tuning_comparison.png")

    # PLOT 5 — Residuals
    residuals = np.asarray(y_test).ravel() - np.asarray(y_pred_test).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(
        residuals,
        bins=50,
        color=COLORS["primary"],
        alpha=0.75,
        edgecolor="white",
    )
    axes[0].axvline(
        0,
        color=COLORS["danger"],
        linestyle="--",
        lw=2,
        label="Zero error",
    )
    axes[0].axvline(
        residuals.mean(),
        color=COLORS["accent"],
        linestyle="--",
        lw=2,
        label=f"Mean={residuals.mean():.3f}",
    )
    axes[0].set_xlabel("Prediction Error (minutes)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residuals Distribution")
    axes[0].legend()
    axes[1].scatter(
        y_pred_test,
        residuals,
        alpha=0.2,
        s=5,
        color=COLORS["primary"],
    )
    axes[1].axhline(
        0, color=COLORS["danger"], linestyle="--", lw=2
    )
    axes[1].set_xlabel("Predicted Values (minutes)")
    axes[1].set_ylabel("Residuals (minutes)")
    axes[1].set_title("Residuals vs Fitted")
    plt.suptitle(
        "Decision Tree: Residual Analysis", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        "figures/dt_05_residuals.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    plt.close("all")

    # SECTION 7 — SAVE MODEL
    joblib.dump(dt_best, "models/dt_model.pkl")
    joblib.dump(dt_cv, "models/dt_gridsearch.pkl")
    print(f"[SAVED] models/dt_model.pkl")
    print(f"[SAVED] models/dt_gridsearch.pkl")

    tree_rules = export_text(
        dt_best, feature_names=FEATURE_COLS, max_depth=5
    )
    with open("results/dt_tree_rules.txt", "w") as f:
        f.write(tree_rules)
    print(f"[SAVED] results/dt_tree_rules.txt")

    # SECTION 8 — FINAL SUMMARY
    print(
        f"""
=======================================================
        DECISION TREE ANALYSIS COMPLETE
=======================================================
Best Parameters  : {dt_cv.best_params_}
Tree Depth       : {dt_best.get_depth()}
Number of Leaves : {dt_best.get_n_leaves()}
Tuning Time      : {tune_time:.2f} seconds

PERFORMANCE (Test Set):
  R²   : {r2_test:.4f}
  MAE  : {mae_test:.4f} minutes
  RMSE : {rmse_test:.4f} minutes

10-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

Top 3 Most Important Features:
  1. {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']*100:.1f}%)
  2. {importance_df.iloc[1]['Feature']} ({importance_df.iloc[1]['Importance']*100:.1f}%)
  3. {importance_df.iloc[2]['Feature']} ({importance_df.iloc[2]['Importance']*100:.1f}%)

FILES SAVED:
  models/dt_model.pkl
  models/dt_gridsearch.pkl
  results/dt_results.csv
  results/dt_tree_rules.txt
  figures/dt_01_tree_structure.png
  figures/dt_02_feature_importance.png
  figures/dt_03_actual_vs_predicted.png
  figures/dt_04_tuning_comparison.png
  figures/dt_05_residuals.png
=======================================================
"""
    )

    elapsed = time.time() - t_script
    print(f"Total elapsed time: {elapsed:.2f} seconds")
