# model_03_random_forest.py — Random Forest regression + SHAP for truck cycle_time.

# SECTION 0 — IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    t_script = time.time()
    print(f"[START] Random Forest Analysis")

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
        raise FileNotFoundError(
            "Run preprocessing.py first to generate data_processed/ files."
        ) from e

    if FEATURE_COLS:
        X_train = X_train[FEATURE_COLS]
        X_val = X_val[FEATURE_COLS]
        X_test = X_test[FEATURE_COLS]

    print(f"[INFO] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(
        f"[INFO] y_train — mean={y_train.mean():.4f}, std={y_train.std():.4f}, "
        f"min={y_train.min():.4f}, max={y_train.max():.4f}"
    )
    print(
        f"[INFO] Training on {len(X_train):,} rows | Testing on {len(X_test):,} rows"
    )

    # SECTION 2 — HYPERPARAMETER TUNING WITH RandomizedSearchCV
    param_distributions = {
        "n_estimators": [100, 150, 200, 300, 500],
        "max_depth": [10, 15, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "bootstrap": [True],
        "oob_score": [True],
    }

    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    rf_cv = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=30,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=SEED,
        verbose=2,
        return_train_score=True,
    )

    start_time = time.time()
    rf_cv.fit(X_train, y_train)
    tune_time = time.time() - start_time

    print(f"[INFO] RandomizedSearchCV completed in {tune_time:.2f} seconds")
    print(f"[INFO] Best parameters:")
    for key, value in rf_cv.best_params_.items():
        print(f"       {key:25s}: {value}")
    print(f"[INFO] Best CV R² (5-fold): {rf_cv.best_score_:.4f}")

    cv_results = pd.DataFrame(rf_cv.cv_results_)
    top5 = cv_results.nlargest(5, "mean_test_score")[
        ["params", "mean_test_score", "std_test_score"]
    ]
    print(f"Top 5 Parameter Combinations:")
    print(top5.to_string())

    # SECTION 3 — FINAL MODEL EVALUATION
    rf_best = rf_cv.best_estimator_

    y_pred_train = rf_best.predict(X_train)
    y_pred_val = rf_best.predict(X_val)
    y_pred_test = rf_best.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    if hasattr(rf_best, "oob_score_") and rf_best.oob_score_ is not None:
        oob = float(rf_best.oob_score_)
    else:
        oob = None

    oob_line = f"{oob:.4f}" if oob is not None else "N/A (not fitted)"

    print(
        f"""
┌──────────────────────────────────────────────────┐
│  RANDOM FOREST — EVALUATION RESULTS              │
├──────────────────────────────────────────────────┤
│  n_estimators   : {rf_best.n_estimators}         │
│  max_depth      : {rf_best.max_depth}            │
│  max_features   : {rf_best.max_features}         │
├──────────────────────────────────────────────────┤
│  Train R²       : {r2_train:.4f}                 │
│  Val   R²       : {r2_val:.4f}                   │
│  Test  R²       : {r2_test:.4f}                  │
│  Test  MAE      : {mae_test:.4f} minutes         │
│  Test  RMSE     : {rmse_test:.4f} minutes        │
│  OOB Score      : {oob_line} (if available)       │
│  Tuning Time    : {tune_time:.2f} seconds        │
└──────────────────────────────────────────────────┘
"""
    )

    cv_scores = cross_val_score(
        rf_best, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
    )
    print(f"5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results = {
        "Model": "Random Forest",
        "R2_Train": r2_train,
        "R2_Val": r2_val,
        "R2_Test": r2_test,
        "MAE": mae_test,
        "RMSE": rmse_test,
        "OOB_Score": oob if oob is not None else np.nan,
        "CV_R2_Mean": cv_scores.mean(),
        "CV_R2_Std": cv_scores.std(),
        "Best_Params": str(rf_cv.best_params_),
        "N_Estimators": rf_best.n_estimators,
        "Tuning_Time": tune_time,
    }
    pd.DataFrame([results]).to_csv("results/rf_results.csv", index=False)

    # SECTION 4 — FEATURE IMPORTANCE ANALYSIS

    # 4a. Gini importance
    gini_importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLS,
            "Gini_Imp": rf_best.feature_importances_,
        }
    ).sort_values("Gini_Imp", ascending=False).reset_index(drop=True)
    gini_importance["Rank"] = range(1, len(gini_importance) + 1)
    gini_importance["Cumulative"] = gini_importance["Gini_Imp"].cumsum()

    print(f"Feature Importance (Gini):")
    print(gini_importance.to_string(index=False))

    # 4b. Permutation importance (more reliable)
    from sklearn.inspection import permutation_importance

    perm_result = permutation_importance(
        rf_best,
        X_test,
        y_test,
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
        scoring="r2",
    )
    perm_importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLS,
            "Perm_Mean": perm_result.importances_mean,
            "Perm_Std": perm_result.importances_std,
        }
    ).sort_values("Perm_Mean", ascending=False).reset_index(drop=True)

    print(f"Permutation Importance (on test set):")
    print(perm_importance.to_string(index=False))

    # SECTION 5 — SHAP ANALYSIS
    print(f"[INFO] Computing SHAP values (this may take 1-2 minutes)...")
    explainer = shap.TreeExplainer(rf_best)
    n_shap = min(2000, len(X_test))
    X_shap = X_test.sample(n=n_shap, random_state=SEED)
    shap_values = explainer.shap_values(X_shap)

    shap_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLS,
            "Mean_SHAP": np.abs(np.asarray(shap_values)).mean(axis=0),
        }
    ).sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)

    print(f"Mean |SHAP| Values (Feature Importance via SHAP):")
    print(shap_df.to_string(index=False))

    top_feat = shap_df.iloc[0]["Feature"]
    top_shap = float(shap_df.iloc[0]["Mean_SHAP"])
    print(
        f"Most influential feature (SHAP): {top_feat} (mean |SHAP| = {top_shap:.4f})"
    )

    # SECTION 6 — VISUALIZATIONS
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("default")

    n_g = len(gini_importance)
    bar_colors_gini = [
        COLORS["primary"] if i >= n_g - 3 else COLORS["light"]
        for i in range(n_g)
    ]

    # PLOT 1 — Gini feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        gini_importance["Feature"][::-1],
        gini_importance["Gini_Imp"][::-1],
        color=bar_colors_gini,
        edgecolor="white",
    )
    for bar, val in zip(bars, gini_importance["Gini_Imp"][::-1]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Gini Importance", fontsize=11)
    ax.set_title(
        "Random Forest: Feature Importance (Gini)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "figures/rf_01_feature_importance_gini.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")
    print(f"[SAVED] figures/rf_01_feature_importance_gini.png")

    # PLOT 2 — SHAP beeswarm
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=FEATURE_COLS,
        plot_type="dot",
        show=False,
        max_display=12,
    )
    plt.title(
        "Random Forest: SHAP Summary Plot (Beeswarm)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("figures/rf_02_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[SAVED] figures/rf_02_shap_beeswarm.png")

    # PLOT 3 — SHAP bar
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=FEATURE_COLS,
        plot_type="bar",
        show=False,
    )
    plt.title(
        "Random Forest: SHAP Feature Importance (Bar)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("figures/rf_03_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[SAVED] figures/rf_03_shap_bar.png")

    # PLOT 4 — SHAP dependence (top 2 features)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for i, feat in enumerate(shap_df["Feature"].values[:2]):
        plt.sca(axes[i])
        shap.dependence_plot(
            feat,
            shap_values,
            X_shap,
            feature_names=FEATURE_COLS,
            ax=axes[i],
            show=False,
            alpha=0.4,
        )
        axes[i].set_title(f"SHAP Dependence: {feat}", fontsize=12)
    plt.suptitle(
        "Random Forest: SHAP Dependence Plots", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        "figures/rf_04_shap_dependence.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    print(f"[SAVED] figures/rf_04_shap_dependence.png")

    # PLOT 5 — Actual vs predicted
    fig, ax = plt.subplots(figsize=(9, 8))
    hb = ax.hexbin(
        y_test,
        y_pred_test,
        gridsize=60,
        cmap="Blues",
        mincnt=1,
    )
    plt.colorbar(hb, ax=ax, label="Count")
    lims = [
        min(float(np.min(y_test)), float(np.min(y_pred_test))),
        max(float(np.max(y_test)), float(np.max(y_pred_test))),
    ]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual cycle_time (minutes)", fontsize=11)
    ax.set_ylabel("Predicted cycle_time (minutes)", fontsize=11)
    ax.set_title(
        f"Random Forest: Actual vs Predicted\nR²={r2_test:.4f} | MAE={mae_test:.4f} | RMSE={rmse_test:.4f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(
        "figures/rf_05_actual_vs_predicted.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    print(f"[SAVED] figures/rf_05_actual_vs_predicted.png")

    # PLOT 6 — Gini vs permutation
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].barh(
        gini_importance["Feature"][::-1],
        gini_importance["Gini_Imp"][::-1],
        color=COLORS["primary"],
        alpha=0.8,
    )
    axes[0].set_title("Gini Importance", fontsize=12)
    axes[0].set_xlabel("Importance Score")
    perm_sorted = perm_importance.sort_values("Perm_Mean")
    axes[1].barh(
        perm_sorted["Feature"],
        perm_sorted["Perm_Mean"],
        xerr=perm_sorted["Perm_Std"],
        color=COLORS["secondary"],
        alpha=0.8,
        capsize=4,
    )
    axes[1].set_title("Permutation Importance (±std)", fontsize=12)
    axes[1].set_xlabel("Mean Decrease in R²")
    plt.suptitle(
        "Random Forest: Gini vs Permutation Importance Comparison",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "figures/rf_06_importance_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    print(f"[SAVED] figures/rf_06_importance_comparison.png")

    # SECTION 7 — SAVE MODEL
    joblib.dump(rf_best, "models/rf_model.pkl")
    joblib.dump(rf_cv, "models/rf_randomsearch.pkl")
    print(f"[SAVED] models/rf_model.pkl")
    print(f"[SAVED] models/rf_randomsearch.pkl")

    np.save("results/rf_shap_values.npy", shap_values)
    X_shap.to_csv("results/rf_shap_samples.csv", index=False)
    shap_df.to_csv("results/rf_shap_importance.csv", index=False)
    print(f"[SAVED] SHAP values and importance tables")

    # SECTION 8 — FINAL SUMMARY
    oob_summary = f"{oob:.4f}" if oob is not None else "N/A"

    print(
        f"""
=======================================================
        RANDOM FOREST ANALYSIS COMPLETE
=======================================================
Best Parameters:
  n_estimators : {rf_best.n_estimators}
  max_depth    : {rf_best.max_depth}
  max_features : {rf_best.max_features}
  min_leaf     : {rf_best.min_samples_leaf}
Tuning Time  : {tune_time:.2f} seconds

PERFORMANCE (Test Set):
  R²   : {r2_test:.4f}
  MAE  : {mae_test:.4f} minutes
  RMSE : {rmse_test:.4f} minutes
  OOB  : {oob_summary}

5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

Top 3 Features (SHAP):
  1. {shap_df.iloc[0]['Feature']}: {shap_df.iloc[0]['Mean_SHAP']:.4f}
  2. {shap_df.iloc[1]['Feature']}: {shap_df.iloc[1]['Mean_SHAP']:.4f}
  3. {shap_df.iloc[2]['Feature']}: {shap_df.iloc[2]['Mean_SHAP']:.4f}

FILES SAVED:
  models/rf_model.pkl
  models/rf_randomsearch.pkl
  results/rf_results.csv
  results/rf_shap_values.npy
  results/rf_shap_samples.csv
  results/rf_shap_importance.csv
  figures/rf_01 through rf_06 (6 PNG files)
=======================================================
"""
    )

    elapsed = time.time() - t_script
    print(f"Total elapsed time: {elapsed:.2f} seconds")
