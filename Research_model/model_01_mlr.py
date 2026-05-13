# model_01_mlr.py — Multiple Linear Regression for truck cycle_time prediction.

# SECTION 0 — IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import joblib
import json
import os
import warnings
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)


def _dtype_inspect_log(label, X):
    print(f"[DTYPE_INSPECT] {label}")
    print(X.dtypes.to_string())
    print(f"[DTYPE_INSPECT] np.asarray(X).dtype = {np.asarray(X).dtype}")


def _series_or_col_as_frame(y):
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0] if y.shape[1] == 1 else y.squeeze()
    return y


def _dtype_is_object_or_bool(dt):
    return dt == object or dt == bool or str(dt) in ("object", "bool", "boolean")


def _safe_to_float64_frame(X):
    try:
        return X.astype("float64", errors="ignore")
    except TypeError:
        return X.apply(pd.to_numeric, errors="coerce")


def _safe_to_float64_series(y):
    s = _series_or_col_as_frame(y)
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    try:
        return s.astype("float64", errors="ignore")
    except TypeError:
        return pd.to_numeric(s, errors="coerce")


def _log_converted_xy(X_before_dtypes, y_before_dtype, X_train_after, y_train_after):
    print(
        "[NUMERIC_COERCION] Columns / target that were object or bool "
        "(original dtype → current dtype):"
    )
    logged = False
    for col, dt in X_before_dtypes.items():
        if _dtype_is_object_or_bool(dt):
            print(f"  X[{col!r}]: {dt} → {X_train_after[col].dtype}")
            logged = True
    if _dtype_is_object_or_bool(y_before_dtype):
        print(f"  y: {y_before_dtype} → {y_train_after.dtype}")
        logged = True
    if not logged:
        print("  (none at this step — no object/bool source dtypes in X or y)")


def coerce_numeric_splits_for_ols(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Inspect dtypes; coerce object/bool to float64; on persistent OLS failure,
    use pd.to_numeric + train-only dropna. Logs original dtypes for converted columns.
    """
    y_train = _series_or_col_as_frame(y_train)
    y_val = _series_or_col_as_frame(y_val)
    y_test = _series_or_col_as_frame(y_test)

    X_dtypes_before = X_train.dtypes.copy()
    y_dtype_before = y_train.dtype

    _dtype_inspect_log("X_train (pre-coercion)", X_train)
    print(f"[DTYPE_INSPECT] y_train.dtype = {y_dtype_before}")

    needs_step2 = any(_dtype_is_object_or_bool(dt) for dt in X_dtypes_before) or _dtype_is_object_or_bool(
        y_dtype_before
    )

    if needs_step2:
        print(
            "[NUMERIC_COERCION] Step 2: object/bool detected → "
            "astype('float64', errors='ignore') on X splits and y splits."
        )
        X_train = _safe_to_float64_frame(X_train)
        X_val = _safe_to_float64_frame(X_val)
        X_test = _safe_to_float64_frame(X_test)
        y_train = _safe_to_float64_series(y_train)
        y_val = _safe_to_float64_series(y_val)
        y_test = _safe_to_float64_series(y_test)
        _log_converted_xy(X_dtypes_before, y_dtype_before, X_train, y_train)
    else:
        print(
            "[NUMERIC_COERCION] Step 2 skipped: no object/bool dtypes in X_train or y_train."
        )

    def _fit_ols():
        X_sm = sm.add_constant(X_train)
        return sm.OLS(y_train, X_sm).fit()

    try:
        ols_model = _fit_ols()
    except ValueError as exc:
        err = str(exc).lower()
        if "object" not in err and "dtype" not in err and "numpy" not in err:
            raise
        print(
            f"[NUMERIC_COERCION] Step 3: OLS still failing ({exc!r}); "
            "pd.to_numeric(errors='coerce') + train dropna + align y."
        )
        X_train = X_train.apply(pd.to_numeric, errors="coerce")
        X_val = X_val.apply(pd.to_numeric, errors="coerce")
        X_test = X_test.apply(pd.to_numeric, errors="coerce")
        y_train = pd.to_numeric(y_train, errors="coerce")
        y_val = pd.to_numeric(y_val, errors="coerce")
        y_test = pd.to_numeric(y_test, errors="coerce")
        train_idx = X_train.index
        X_train = X_train.dropna()
        y_train = y_train.reindex(X_train.index)
        bad_y = y_train.isna()
        if bad_y.any():
            n_bad = int(bad_y.sum())
            X_train = X_train.loc[~bad_y]
            y_train = y_train.loc[~bad_y]
            print(f"[NUMERIC_COERCION] Dropped {n_bad} train row(s) with NaN in y after X dropna.")
        dropped = len(train_idx) - len(X_train)
        if dropped:
            print(f"[NUMERIC_COERCION] Total train rows removed vs pre-step-3: {dropped}")
        med = X_train.median(numeric_only=True)
        X_val = X_val.fillna(med)
        X_test = X_test.fillna(med)
        y_val = y_val.fillna(y_train.median())
        y_test = y_test.fillna(y_train.median())
        _log_converted_xy(X_dtypes_before, y_dtype_before, X_train, y_train)
        ols_model = _fit_ols()

    return X_train, y_train, X_val, y_val, X_test, y_test, ols_model


if __name__ == "__main__":
    t_script = time.time()
    print(f"[START] MLR Analysis")

    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

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

    # Align columns to saved feature order when present
    if FEATURE_COLS:
        X_train = X_train[FEATURE_COLS]
        X_val = X_val[FEATURE_COLS]
        X_test = X_test[FEATURE_COLS]

    print(f"[INFO] Data loaded successfully. Features: {FEATURE_COLS}")

    # Dtype inspect + numeric coercion + OLS (handles object/bool → float64)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        ols_model,
    ) = coerce_numeric_splits_for_ols(X_train, y_train, X_val, y_val, X_test, y_test)

    print(f"[INFO] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(
        f"[INFO] y_train stats — mean={y_train.mean():.4f}, std={y_train.std():.4f}, "
        f"min={y_train.min():.4f}, max={y_train.max():.4f}"
    )

    # SECTION 2 — STATSMODELS OLS (FULL STATISTICAL ANALYSIS)

    # 2a. OLS model (fitted inside coerce_numeric_splits_for_ols)

    # 2b. Print full summary
    print(ols_model.summary())

    # 2c. Key statistics (box)
    f_p = float(ols_model.f_pvalue)
    f_p_str = "p < 0.001" if f_p < 0.001 else f"p = {f_p:.4f}"
    print(
        f"""
┌─────────────────────────────────────────┐
│  OLS REGRESSION SUMMARY                 │
├─────────────────────────────────────────┤
│  R²              : {ols_model.rsquared:.4f}              │
│  Adjusted R²     : {ols_model.rsquared_adj:.4f}              │
│  F-statistic     : {ols_model.fvalue:.2f} ({f_p_str})  │
│  AIC             : {ols_model.aic:.2f}           │
│  BIC             : {ols_model.bic:.2f}           │
│  No. Observations: {int(ols_model.nobs)}                │
└─────────────────────────────────────────┘
"""
    )

    # 2d. Coefficient table with significance flags
    print(
        f"{'Feature':<22} | {'Coefficient':>12} | {'Std Error':>10} | "
        f"{'t-value':>8} | {'p-value':>10} | {'Sig':>4}"
    )
    print("-" * 85)
    for name in ols_model.params.index:
        coef = float(ols_model.params[name])
        se = float(ols_model.bse[name])
        tval = float(ols_model.tvalues[name])
        pval = float(ols_model.pvalues[name])
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(
            f"{str(name):<22} | {coef:12.6f} | {se:10.6f} | "
            f"{tval:8.3f} | {pval:10.6f} | {sig:>4}"
        )

    # 2e. Interpretation (exclude const for influence / nonsig features)
    params_no_const = ols_model.params.drop("const", errors="ignore")
    pvals_no_const = ols_model.pvalues.drop("const", errors="ignore")
    top3 = params_no_const.reindex(
        params_no_const.abs().sort_values(ascending=False).index
    ).head(3)
    print(f"\n[INFO] Top 3 features by |coefficient| (excluding const):")
    for fname, val in top3.items():
        print(f"  - {fname}: {val:.6f}")
    nonsig = pvals_no_const[pvals_no_const >= 0.05].index.tolist()
    if nonsig:
        print(f"\n[INFO] NOT statistically significant (p >= 0.05): {nonsig}")
    else:
        print(f"\n[INFO] All features (excl. const) significant at p < 0.05.")

    # SECTION 3 — SKLEARN LinearRegression (FOR PREDICTION)

    # 3a. Fit sklearn model
    mlr = LinearRegression()
    start_time = time.time()
    mlr.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"[INFO] MLR training completed in {train_time:.2f} seconds")

    # 3b. Predict
    y_pred_test = mlr.predict(X_test)
    y_pred_train = mlr.predict(X_train)

    # 3c. Metrics
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2_train = r2_score(y_train, y_pred_train)

    print(
        f"""
┌──────────────────────────────────────────┐
│  MLR EVALUATION RESULTS                  │
├──────────────────────────────────────────┤
│  Train R²  : {r2_train:.4f}                      │
│  Test  R²  : {r2_test:.4f}                      │
│  Test  MAE : {mae_test:.4f} minutes              │
│  Test  RMSE: {rmse_test:.4f} minutes              │
│  Training time: {train_time:.2f} seconds             │
└──────────────────────────────────────────┘
"""
    )

    # 3d. Overfitting check
    diff = r2_train - r2_test
    if diff > 0.05:
        print(
            f"[WARNING] Possible overfitting (R² gap = {diff:.4f})"
        )
    else:
        print(f"[OK] No significant overfitting detected")

    # 3e. Save results CSV
    results = {
        "Model": "MLR",
        "R2": r2_test,
        "MAE": mae_test,
        "RMSE": rmse_test,
        "Train_R2": r2_train,
        "Training_Time": train_time,
    }
    pd.DataFrame([results]).to_csv("results/mlr_results.csv", index=False)

    # SECTION 4 — REGRESSION ASSUMPTION DIAGNOSTICS

    # 4a. Residuals (test set)
    residuals = pd.Series(y_test).values.ravel() - np.asarray(y_pred_test).ravel()
    standardized_res = (residuals - residuals.mean()) / residuals.std()

    # 4b. Shapiro-Wilk (first 5000)
    sw_n = min(5000, len(residuals))
    stat_sw, p_norm = stats.shapiro(residuals[:sw_n])
    print(f"Shapiro-Wilk: stat={stat_sw:.4f}, p={p_norm:.6f}")
    print(
        f"→ Residuals are NORMAL"
        if p_norm > 0.05
        else f"→ Residuals are NOT NORMAL"
    )

    # 4c. Breusch-Pagan
    X_test_sm = sm.add_constant(X_test)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_test_sm)
    print(f"Breusch-Pagan: stat={bp_stat:.4f}, p={bp_p:.6f}")
    print(
        f"→ Homoscedastic (constant variance)"
        if bp_p > 0.05
        else f"→ Heteroscedastic (non-constant variance) — MLR assumption VIOLATED"
    )

    # 4d. VIF
    X_with_const = sm.add_constant(X_train)
    vif_data = pd.DataFrame(
        {
            "Feature": X_with_const.columns,
            "VIF": [
                variance_inflation_factor(X_with_const.values.astype(float), i)
                for i in range(X_with_const.shape[1])
            ],
        }
    )
    vif_data = vif_data[vif_data["Feature"] != "const"]
    vif_data = vif_data.sort_values("VIF", ascending=False)
    print(vif_data.to_string(index=False))
    for _, row in vif_data.iterrows():
        feat = row["Feature"]
        vif = float(row["VIF"])
        if vif > 10:
            print(f"[WARNING] HIGH MULTICOLLINEARITY: {feat} (VIF={vif:.2f})")
        elif vif > 5:
            print(f"[CAUTION] Moderate multicollinearity: {feat} (VIF={vif:.2f})")

    # SECTION 5 — VISUALIZATIONS
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("default")

    COLORS = {
        "blue": "#2E6DB4",
        "red": "#C0392B",
        "green": "#1A7A6E",
        "grey": "#8A9BB0",
    }

    # PLOT 1 — Residuals vs Fitted
    fig, ax = plt.subplots(figsize=(10, 6))
    x_fit = np.asarray(y_pred_test).ravel()
    ax.scatter(
        x_fit,
        residuals,
        alpha=0.3,
        s=5,
        color=COLORS["blue"],
    )
    ax.axhline(0, color=COLORS["red"], linestyle="--", linewidth=1.5)
    order = np.argsort(x_fit)
    smoothed = lowess(
        residuals[order], x_fit[order], frac=0.1, return_sorted=True
    )
    ax.plot(smoothed[:, 0], smoothed[:, 1], color=COLORS["red"], linewidth=2)
    ax.set_xlabel("Fitted Values (minutes)")
    ax.set_ylabel("Residuals (minutes)")
    ax.set_title("MLR: Residuals vs Fitted Values")
    ax.text(
        0.02,
        0.98,
        f"Breusch-Pagan p = {bp_p:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    plt.tight_layout()
    fig.savefig(
        "figures/mlr_01_residuals_vs_fitted.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"[SAVED] figures/mlr_01_residuals_vs_fitted.png")
    plt.close(fig)

    # PLOT 2 — Q-Q
    fig, ax = plt.subplots(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=ax)
    lines = ax.get_lines()
    if len(lines) >= 1:
        lines[0].set_markerfacecolor(COLORS["blue"])
        lines[0].set_markeredgecolor(COLORS["blue"])
        lines[0].set_markersize(4)
    if len(lines) >= 2:
        lines[1].set_color(COLORS["red"])
        lines[1].set_linewidth(2)
    ax.set_title("MLR: Q-Q Plot of Residuals")
    ax.text(
        0.02,
        0.98,
        f"Shapiro-Wilk p = {p_norm:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    plt.tight_layout()
    fig.savefig("figures/mlr_02_qq_plot.png", dpi=150, bbox_inches="tight")
    print(f"[SAVED] figures/mlr_02_qq_plot.png")
    plt.close(fig)

    # PLOT 3 — Actual vs Predicted (hexbin)
    fig, ax = plt.subplots(figsize=(10, 8))
    hb = ax.hexbin(
        np.asarray(y_test).ravel(),
        np.asarray(y_pred_test).ravel(),
        gridsize=50,
        cmap="Blues",
        mincnt=1,
    )
    lims = [
        min(np.asarray(y_test).min(), np.asarray(y_pred_test).min()),
        max(np.asarray(y_test).max(), np.asarray(y_pred_test).max()),
    ]
    ax.plot(lims, lims, color=COLORS["red"], linestyle="-", linewidth=2, label="Ideal")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count")
    ax.set_xlabel("Actual cycle_time (min)")
    ax.set_ylabel("Predicted cycle_time (min)")
    ax.set_title(
        f"MLR: Actual vs Predicted\nR²={r2_test:.4f} | MAE={mae_test:.4f} | RMSE={rmse_test:.4f}"
    )
    plt.tight_layout()
    fig.savefig(
        "figures/mlr_03_actual_vs_predicted.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"[SAVED] figures/mlr_03_actual_vs_predicted.png")
    plt.close(fig)

    # PLOT 4 — Coefficients + 95% CI (OLS, no const)
    params_plot = ols_model.params.drop("const", errors="ignore")
    ci = ols_model.conf_int(alpha=0.05).drop("const", errors="ignore")
    ci_low = ci[0]
    ci_high = ci[1]
    err_low = params_plot - ci_low
    err_high = ci_high - params_plot
    order_idx = np.argsort(np.abs(params_plot.values))
    names_ord = params_plot.index[order_idx]
    vals_ord = params_plot.values[order_idx]
    el_ord = err_low.values[order_idx]
    eh_ord = err_high.values[order_idx]
    colors_bar = [COLORS["green"] if v >= 0 else COLORS["red"] for v in vals_ord]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names_ord))
    ax.barh(y_pos, vals_ord, xerr=[el_ord, eh_ord], color=colors_bar, alpha=0.85, capsize=3)
    ax.axvline(0, color=COLORS["grey"], linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_ord)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("MLR: Regression Coefficients with 95% CI")
    plt.tight_layout()
    fig.savefig("figures/mlr_04_coefficients.png", dpi=150, bbox_inches="tight")
    print(f"[SAVED] figures/mlr_04_coefficients.png")
    plt.close(fig)

    # PLOT 5 — Error distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        residuals,
        kde=True,
        ax=ax,
        color=COLORS["blue"],
        alpha=0.7,
        edgecolor="white",
    )
    ax.axvline(0, color=COLORS["red"], linestyle="--", linewidth=1.5)
    ax.axvline(rmse_test, color="#E67E22", linestyle="--", linewidth=1.2, label="+1 RMSE")
    ax.axvline(-rmse_test, color="#E67E22", linestyle="--", linewidth=1.2, label="-1 RMSE")
    ax.set_xlabel("Prediction Error (minutes)")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"MLR: Distribution of Prediction Errors\nMean={residuals.mean():.4f}, Std={residuals.std():.4f}"
    )
    plt.tight_layout()
    fig.savefig(
        "figures/mlr_05_error_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"[SAVED] figures/mlr_05_error_distribution.png")
    plt.close(fig)

    plt.close("all")

    # SECTION 6 — SAVE MODEL
    joblib.dump(mlr, "models/mlr_model.pkl")
    print(f"[SAVED] models/mlr_model.pkl")

    # SECTION 7 — FINAL SUMMARY
    norm_pass = p_norm > 0.05
    homo_pass = bp_p > 0.05
    max_vif = float(vif_data["VIF"].max())
    vif_pass = max_vif < 10

    print(
        f"""
=======================================================
        MLR ANALYSIS COMPLETE
=======================================================
Model          : Multiple Linear Regression (OLS)
Training rows  : {len(X_train):,}
Test rows      : {len(X_test):,}
Features used  : {len(FEATURE_COLS)}

PERFORMANCE (Test Set):
  R²   : {r2_test:.4f}
  MAE  : {mae_test:.4f} minutes
  RMSE : {rmse_test:.4f} minutes

ASSUMPTION TESTS:
  Normality (Shapiro-Wilk) : p = {p_norm:.4f} → {'PASS' if norm_pass else 'FAIL'}
  Homoscedasticity (BP)    : p = {bp_p:.4f} → {'PASS' if homo_pass else 'FAIL'}
  Multicollinearity (VIF)  : Max VIF = {max_vif:.2f} → {'PASS' if vif_pass else 'FAIL'}

FILES SAVED:
  models/mlr_model.pkl
  results/mlr_results.csv
  figures/mlr_01_residuals_vs_fitted.png
  figures/mlr_02_qq_plot.png
  figures/mlr_03_actual_vs_predicted.png
  figures/mlr_04_coefficients.png
  figures/mlr_05_error_distribution.png
=======================================================
"""
    )

    elapsed = time.time() - t_script
    print(f"[DONE] MLR Analysis completed in {elapsed:.2f} seconds")
