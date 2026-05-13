# model_04_gradient_boosting.py — Gradient Boosting regression + SHAP for cycle_time.

# SECTION 0 — IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import shap
from sklearn.ensemble import GradientBoostingRegressor
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
    print(f"[START] Gradient Boosting Analysis")

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

    print(
        f"[INFO] Shapes — X_train: {X_train.shape}, X_val: {X_val.shape}, "
        f"X_test: {X_test.shape}"
    )
    print(f"[INFO] Feature count: {len(FEATURE_COLS)}")
    print(
        f"[INFO] y_train — mean={y_train.mean():.4f}, std={y_train.std():.4f}, "
        f"min={y_train.min():.4f}, max={y_train.max():.4f}"
    )
    print(
        f"[INFO] Gradient Boosting training on {len(X_train):,} rows"
    )

    # SECTION 2 — HYPERPARAMETER TUNING WITH RandomizedSearchCV
    param_distributions = {
        "n_estimators": [100, 150, 200, 300, 500],
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.15, 0.2],
        "max_depth": [3, 4, 5, 6, 7, 9],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2", None],
        "loss": ["squared_error"],
    }

    gb = GradientBoostingRegressor(random_state=SEED)
    gb_cv = RandomizedSearchCV(
        estimator=gb,
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
    gb_cv.fit(X_train, y_train)
    tune_time = time.time() - start_time

    print(f"[INFO] Tuning completed in {tune_time:.2f} seconds")
    print(f"[INFO] Best CV R² (5-fold): {gb_cv.best_score_:.4f}")
    print(f"Best Parameters:")
    for key, value in sorted(gb_cv.best_params_.items()):
        print(f"  {key:25s} : {value}")

    cv_df = pd.DataFrame(gb_cv.cv_results_)
    top5 = cv_df.nlargest(5, "mean_test_score")[
        ["params", "mean_test_score", "std_test_score"]
    ]
    print(f"Top 5 Parameter Combinations:")
    print(top5.to_string())

    # SECTION 3 — TRAINING LOSS TRAJECTORY (Staged Predict)
    staged_params = dict(gb_cv.best_params_)
    staged_params["random_state"] = SEED
    gb_staged = GradientBoostingRegressor(**staged_params)
    gb_staged.fit(X_train, y_train)

    staged_train_mse = []
    staged_val_mse = []
    staged_train_r2 = []
    staged_val_r2 = []

    for y_pred_stage_train, y_pred_stage_val in zip(
        gb_staged.staged_predict(X_train),
        gb_staged.staged_predict(X_val),
    ):
        staged_train_mse.append(
            mean_squared_error(y_train, y_pred_stage_train)
        )
        staged_val_mse.append(mean_squared_error(y_val, y_pred_stage_val))
        staged_train_r2.append(r2_score(y_train, y_pred_stage_train))
        staged_val_r2.append(r2_score(y_val, y_pred_stage_val))

    best_iter = int(np.argmin(staged_val_mse) + 1)
    best_val_r2 = staged_val_r2[best_iter - 1]
    n_trees_staged = getattr(gb_staged, "n_estimators_", gb_staged.n_estimators)
    print(
        f"[INFO] Best iteration (min val MSE): {best_iter} / {n_trees_staged}"
    )
    print(f"[INFO] Val R² at best iteration: {best_val_r2:.4f}")

    # SECTION 4 — FINAL MODEL EVALUATION
    gb_best = gb_cv.best_estimator_

    y_pred_train = gb_best.predict(X_train)
    y_pred_val = gb_best.predict(X_val)
    y_pred_test = gb_best.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    print(
        f"""
┌──────────────────────────────────────────────────┐
│  GRADIENT BOOSTING — EVALUATION RESULTS          │
├──────────────────────────────────────────────────┤
│  n_estimators  : {gb_best.n_estimators}          │
│  learning_rate : {gb_best.learning_rate}         │
│  max_depth     : {gb_best.max_depth}             │
│  subsample     : {gb_best.subsample}             │
├──────────────────────────────────────────────────┤
│  Train R²      : {r2_train:.4f}                  │
│  Val   R²      : {r2_val:.4f}                    │
│  Test  R²      : {r2_test:.4f}                   │
│  Test  MAE     : {mae_test:.4f} minutes          │
│  Test  RMSE    : {rmse_test:.4f} minutes         │
│  Tuning Time   : {tune_time:.2f} seconds         │
└──────────────────────────────────────────────────┘
"""
    )

    cv_scores = cross_val_score(
        gb_best, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
    )
    print(f"5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results = {
        "Model": "Gradient Boosting",
        "R2_Train": r2_train,
        "R2_Val": r2_val,
        "R2_Test": r2_test,
        "MAE": mae_test,
        "RMSE": rmse_test,
        "CV_R2_Mean": cv_scores.mean(),
        "CV_R2_Std": cv_scores.std(),
        "Best_Params": str(gb_cv.best_params_),
        "Best_Iteration": best_iter,
        "Tuning_Time": tune_time,
    }
    pd.DataFrame([results]).to_csv("results/gb_results.csv", index=False)

    # SECTION 5 — FEATURE IMPORTANCE ANALYSIS

    # 5a. Built-in importance
    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLS,
            "Importance": gb_best.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance_df["Rank"] = range(1, len(importance_df) + 1)
    importance_df["Cumulative"] = importance_df["Importance"].cumsum()
    print(importance_df.to_string(index=False))

    # 5b. SHAP
    print(f"[INFO] Computing SHAP values...")
    explainer = shap.TreeExplainer(gb_best)
    n_shap = min(2000, len(X_test))
    X_shap = X_test.sample(n=n_shap, random_state=SEED)
    shap_values = explainer.shap_values(X_shap)

    shap_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLS,
            "Mean_SHAP": np.abs(np.asarray(shap_values)).mean(axis=0),
        }
    ).sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)

    print(f"SHAP Feature Importance:")
    print(shap_df.to_string(index=False))

    # 5c. SHAP vs Gini
    merged = importance_df[["Feature", "Importance"]].merge(
        shap_df[["Feature", "Mean_SHAP"]], on="Feature"
    )
    merged["Gini_Rank"] = merged["Importance"].rank(ascending=False).astype(int)
    merged["SHAP_Rank"] = merged["Mean_SHAP"].rank(ascending=False).astype(int)
    print(f"SHAP vs Gini Ranking Comparison:")
    print(
        merged[["Feature", "Gini_Rank", "SHAP_Rank"]]
        .sort_values("SHAP_Rank")
        .to_string(index=False)
    )

    # SECTION 6 — VISUALIZATIONS
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("default")

    # PLOT 1 — Training trajectory
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    iterations = range(1, len(staged_train_mse) + 1)
    axes[0].plot(
        iterations,
        staged_train_mse,
        color=COLORS["primary"],
        lw=1.5,
        label="Train MSE",
        alpha=0.8,
    )
    axes[0].plot(
        iterations,
        staged_val_mse,
        color=COLORS["accent"],
        lw=1.5,
        label="Validation MSE",
        alpha=0.8,
    )
    axes[0].axvline(
        x=best_iter,
        color=COLORS["danger"],
        linestyle="--",
        lw=2,
        label=f"Best iter={best_iter}",
    )
    axes[0].set_xlabel("Number of Trees (Iterations)", fontsize=11)
    axes[0].set_ylabel("Mean Squared Error", fontsize=11)
    axes[0].set_title("Training & Validation MSE", fontsize=12, fontweight="bold")
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(10))
    axes[0].legend()

    axes[1].plot(
        iterations,
        staged_train_r2,
        color=COLORS["primary"],
        lw=1.5,
        label="Train R²",
        alpha=0.8,
    )
    axes[1].plot(
        iterations,
        staged_val_r2,
        color=COLORS["accent"],
        lw=1.5,
        label="Validation R²",
        alpha=0.8,
    )
    axes[1].axvline(
        x=best_iter,
        color=COLORS["danger"],
        linestyle="--",
        lw=2,
        label=f"Best iter={best_iter}",
    )
    axes[1].set_xlabel("Number of Trees (Iterations)", fontsize=11)
    axes[1].set_ylabel("R² Score", fontsize=11)
    axes[1].set_title("Training & Validation R²", fontsize=12, fontweight="bold")
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(10))
    axes[1].legend()

    plt.suptitle(
        "Gradient Boosting: Training Loss Trajectory",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "figures/gb_01_training_trajectory.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")
    print(f"[SAVED] figures/gb_01_training_trajectory.png")

    # PLOT 2 — Feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_bars = [
        COLORS["primary"] if i < 3 else COLORS["light"]
        for i in range(len(importance_df))
    ]
    bars = ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
        color=list(reversed(colors_bars)),
        edgecolor="white",
    )
    for bar, val in zip(bars, importance_df["Importance"][::-1]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Feature Importance (Reduction in Loss)", fontsize=11)
    ax.set_title(
        "Gradient Boosting: Feature Importance",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "figures/gb_02_feature_importance.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    print(f"[SAVED] figures/gb_02_feature_importance.png")

    # PLOT 3 — SHAP beeswarm
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
        "Gradient Boosting: SHAP Summary Plot",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("figures/gb_03_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[SAVED] figures/gb_03_shap_beeswarm.png")

    # PLOT 4 — SHAP dependence (top 2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for i, feat in enumerate(shap_df["Feature"].values[:2]):
        try:
            shap.dependence_plot(
                feat,
                shap_values,
                X_shap,
                feature_names=FEATURE_COLS,
                ax=axes[i],
                show=False,
                alpha=0.4,
                dot_size=5,
            )
        except TypeError:
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
        "Gradient Boosting: SHAP Dependence Plots",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "figures/gb_04_shap_dependence.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    print(f"[SAVED] figures/gb_04_shap_dependence.png")

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
        f"Gradient Boosting: Actual vs Predicted\nR²={r2_test:.4f} | MAE={mae_test:.4f} | RMSE={rmse_test:.4f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(
        "figures/gb_05_actual_vs_predicted.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    print(f"[SAVED] figures/gb_05_actual_vs_predicted.png")

    # PLOT 6 — Residuals
    residuals = np.asarray(y_test).ravel() - np.asarray(y_pred_test).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(
        residuals,
        bins=60,
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
        lw=1.5,
        label=f"Mean={residuals.mean():.3f}",
    )
    axes[0].set_xlabel("Prediction Error (minutes)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].legend(fontsize=9)
    axes[1].scatter(
        y_pred_test,
        residuals,
        alpha=0.15,
        s=4,
        color=COLORS["primary"],
    )
    axes[1].axhline(0, color=COLORS["danger"], linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted Values (minutes)")
    axes[1].set_ylabel("Residuals (minutes)")
    axes[1].set_title("Residuals vs Fitted")
    plt.suptitle(
        f"Gradient Boosting: Residual Analysis | RMSE={rmse_test:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("figures/gb_06_residuals.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[SAVED] figures/gb_06_residuals.png")

    # SECTION 7 — SAVE MODEL
    joblib.dump(gb_best, "models/gb_model.pkl")
    joblib.dump(gb_cv, "models/gb_randomsearch.pkl")
    np.save("results/gb_shap_values.npy", shap_values)
    X_shap.to_csv("results/gb_shap_samples.csv", index=False)
    shap_df.to_csv("results/gb_shap_importance.csv", index=False)
    importance_df.to_csv("results/gb_feature_importance.csv", index=False)
    print(f"[SAVED] All GB model files")

    # SECTION 8 — FINAL SUMMARY
    print(
        f"""
=======================================================
        GRADIENT BOOSTING ANALYSIS COMPLETE
=======================================================
Best Parameters:
  n_estimators  : {gb_best.n_estimators}
  learning_rate : {gb_best.learning_rate}
  max_depth     : {gb_best.max_depth}
  subsample     : {gb_best.subsample}
Best Iteration  : {best_iter}
Tuning Time     : {tune_time:.2f} seconds

PERFORMANCE (Test Set):
  R²   : {r2_test:.4f}
  MAE  : {mae_test:.4f} minutes
  RMSE : {rmse_test:.4f} minutes

5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

Top 3 Features (SHAP):
  1. {shap_df.iloc[0]['Feature']}: {shap_df.iloc[0]['Mean_SHAP']:.4f}
  2. {shap_df.iloc[1]['Feature']}: {shap_df.iloc[1]['Mean_SHAP']:.4f}
  3. {shap_df.iloc[2]['Feature']}: {shap_df.iloc[2]['Mean_SHAP']:.4f}

FILES SAVED:
  models/gb_model.pkl
  results/gb_results.csv
  results/gb_shap_importance.csv
  figures/gb_01 through gb_06 (6 PNG files)
=======================================================
"""
    )

    elapsed = time.time() - t_script
    print(f"Total elapsed time: {elapsed:.2f} seconds")
