# model_05_ann.py — Feedforward ANN (MLP) for truck cycle_time (scaled features).

# SECTION 0 — IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
import joblib
import json
import os
import random
import time
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    t_script = time.time()
    print(f"[START] ANN Analysis")

    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    COLORS = {
        "primary": "#2E6DB4",
        "secondary": "#1A7A6E",
        "accent": "#E07B39",
        "danger": "#C0392B",
        "purple": "#7B3FA0",
        "light": "#D6E8F7",
    }

    # SECTION 1 — LOAD PREPROCESSED DATA
    try:
        X_train_scaled = pd.read_csv("data_processed/X_train_scaled.csv")
        X_val_scaled = pd.read_csv("data_processed/X_val_scaled.csv")
        X_test_scaled = pd.read_csv("data_processed/X_test_scaled.csv")
        y_train = pd.read_csv("data_processed/y_train.csv").squeeze()
        y_val = pd.read_csv("data_processed/y_val.csv").squeeze()
        y_test = pd.read_csv("data_processed/y_test.csv").squeeze()
        with open("data_processed/feature_cols.json") as f:
            FEATURE_COLS = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError("ERROR: Run preprocessing.py first") from e

    if FEATURE_COLS:
        X_train_scaled = X_train_scaled[FEATURE_COLS]
        X_val_scaled = X_val_scaled[FEATURE_COLS]
        X_test_scaled = X_test_scaled[FEATURE_COLS]

    INPUT_DIM = X_train_scaled.shape[1]
    arch_label = f"{INPUT_DIM}→128→64→32→1"

    print(f"[INFO] Input dimension: {INPUT_DIM} features")
    print(f"[INFO] Training: {X_train_scaled.shape[0]:,} rows")
    print(f"[INFO] Validation: {X_val_scaled.shape[0]:,} rows")
    print(f"[INFO] Test: {X_test_scaled.shape[0]:,} rows")
    print(
        f"[INFO] Target mean: {y_train.mean():.2f} ± {y_train.std():.2f} minutes"
    )

    X_train_np = X_train_scaled.values.astype(np.float32)
    X_val_np = X_val_scaled.values.astype(np.float32)
    X_test_np = X_test_scaled.values.astype(np.float32)
    y_train_np = np.asarray(y_train).astype(np.float32).ravel()
    y_val_np = np.asarray(y_val).astype(np.float32).ravel()
    y_test_np = np.asarray(y_test).astype(np.float32).ravel()

    # SECTION 2 — BUILD MODEL ARCHITECTURE

    def build_ann_model(input_dim, seed=42):
        """
        Build ANN (MLP) for regression.
        Architecture: Input → Dense(128) → Dense(64) → Dense(32) → Output(1)
        Regularization: BatchNorm + Dropout after each hidden layer
        """
        tf.random.set_seed(seed)

        try:
            d1 = layers.Dropout(0.3, seed=seed, name="dropout_1")
            d2 = layers.Dropout(0.2, seed=seed, name="dropout_2")
        except TypeError:
            d1 = layers.Dropout(0.3, name="dropout_1")
            d2 = layers.Dropout(0.2, name="dropout_2")

        model = keras.Sequential(
            [
                layers.Input(shape=(input_dim,), name="input_layer"),
                layers.Dense(128, name="dense_1"),
                layers.BatchNormalization(name="bn_1"),
                layers.Activation("relu", name="relu_1"),
                d1,
                layers.Dense(64, name="dense_2"),
                layers.BatchNormalization(name="bn_2"),
                layers.Activation("relu", name="relu_2"),
                d2,
                layers.Dense(32, activation="relu", name="dense_3"),
                layers.Dense(1, activation="linear", name="output_layer"),
            ],
            name="ANN_Truck_Cycle",
        )
        return model

    # SECTION 2 (continued) — COMPILE
    model = build_ann_model(INPUT_DIM, seed=SEED)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()
    total_params = model.count_params()
    print(f"[INFO] Total trainable parameters: {total_params:,}")

    # SECTION 3 — CALLBACKS
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
            mode="min",
        ),
        callbacks.ModelCheckpoint(
            filepath="models/ann_best.keras",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode="min",
        ),
        callbacks.CSVLogger(
            filename="results/ann_training_log.csv",
            separator=",",
            append=False,
        ),
    ]
    print(
        f"[INFO] Callbacks defined: EarlyStopping(p=20), ReduceLROnPlateau(p=10),"
    )
    print(
        f"       ModelCheckpoint → models/ann_best.keras, CSVLogger"
    )

    # SECTION 4 — TRAIN
    start_time = time.time()
    history = model.fit(
        X_train_np,
        y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=200,
        batch_size=256,
        callbacks=callbacks_list,
        verbose=1,
    )
    train_time = time.time() - start_time

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_val_loss = float(min(history.history["val_loss"]))
    best_val_mae = float(history.history["val_mae"][best_epoch - 1])
    lr_hist = history.history.get("lr", [0.001])
    final_lr = float(lr_hist[-1])
    total_epochs = len(history.history["loss"])

    print(
        f"""
[TRAINING COMPLETE]
Total epochs run    : {total_epochs} / 200
Best epoch          : {best_epoch}
Best val loss (MSE) : {best_val_loss:.4f}
Best val MAE        : {best_val_mae:.4f} minutes
Final learning rate : {final_lr:.6f}
Training time       : {train_time:.2f} seconds
"""
    )

    # SECTION 5 — EVALUATION
    ckpt_path = "models/ann_best.keras"
    if os.path.isfile(ckpt_path):
        best_model = keras.models.load_model(ckpt_path)
        print(f"[INFO] Best model loaded from checkpoint")
    else:
        best_model = model
        print(
            f"[INFO] Checkpoint not found; using in-memory model (with best weights if ES ran)."
        )

    y_pred_train = best_model.predict(X_train_np, verbose=0).flatten()
    y_pred_val = best_model.predict(X_val_np, verbose=0).flatten()
    y_pred_test = best_model.predict(X_test_np, verbose=0).flatten()

    r2_train = r2_score(y_train_np, y_pred_train)
    r2_val = r2_score(y_val_np, y_pred_val)
    r2_test = r2_score(y_test_np, y_pred_test)
    mae_test = mean_absolute_error(y_test_np, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test_np, y_pred_test)))
    mape_test = float(
        np.mean(np.abs((y_test_np - y_pred_test) / y_test_np)) * 100
    )

    total_params_eval = int(best_model.count_params())

    print(
        f"""
┌──────────────────────────────────────────────────┐
│  ANN (MLP) — EVALUATION RESULTS                  │
├──────────────────────────────────────────────────┤
│  Architecture  : {arch_label}                   │
│  Total Params  : {total_params_eval:,}                 │
│  Best Epoch    : {best_epoch} / 200               │
│  Total Epochs  : {total_epochs}                   │
├──────────────────────────────────────────────────┤
│  Train R²      : {r2_train:.4f}                  │
│  Val   R²      : {r2_val:.4f}                    │
│  Test  R²      : {r2_test:.4f}                   │
│  Test  MAE     : {mae_test:.4f} minutes          │
│  Test  RMSE    : {rmse_test:.4f} minutes         │
│  Test  MAPE    : {mape_test:.2f} %               │
│  Training Time : {train_time:.2f} seconds        │
└──────────────────────────────────────────────────┘
"""
    )

    results = {
        "Model": "ANN (MLP)",
        "R2_Train": r2_train,
        "R2_Val": r2_val,
        "R2_Test": r2_test,
        "MAE": mae_test,
        "RMSE": rmse_test,
        "MAPE": mape_test,
        "Best_Epoch": best_epoch,
        "Total_Epochs": total_epochs,
        "Total_Params": total_params_eval,
        "Architecture": arch_label,
        "Training_Time": train_time,
    }
    pd.DataFrame([results]).to_csv("results/ann_results.csv", index=False)

    # SECTION 6 — SHAP
    print(f"[INFO] Computing SHAP values using GradientExplainer...")
    background = X_train_np[:500]
    n_shap = min(1000, len(X_test_np))
    X_shap_np = X_test_np[:n_shap]
    X_shap_df = X_test_scaled.iloc[:n_shap].copy()

    shap_values = None
    try:
        explainer = shap.GradientExplainer(best_model, background)
        shap_values = explainer.shap_values(X_shap_np)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]
    except Exception:
        try:
            explainer = shap.KernelExplainer(
                lambda x: best_model.predict(x, verbose=0).flatten(),
                X_train_np[:100],
            )
            shap_values = explainer.shap_values(X_shap_np[:200])
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 0]
            X_shap_df = X_shap_df.iloc[:200].copy()
            X_shap_np = X_shap_np[:200]
            print(
                f"[INFO] GradientExplainer unavailable; used KernelExplainer on 200 rows."
            )
        except Exception:
            print(f"WARNING: SHAP computation failed. Skipping.")
            shap_values = None

    if shap_values is not None:
        shap_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLS,
                "Mean_SHAP": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
        shap_df["Rank"] = range(1, len(shap_df) + 1)
        print(f"ANN SHAP Feature Importance:")
        print(shap_df.to_string(index=False))
    else:
        shap_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLS,
                "Mean_SHAP": 0.0,
                "Rank": list(range(1, len(FEATURE_COLS) + 1)),
            }
        )

    # SECTION 7 — VISUALIZATIONS
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("default")

    epochs_range = range(1, len(history.history["loss"]) + 1)

    # PLOT 1 — Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        epochs_range,
        history.history["loss"],
        color=COLORS["primary"],
        lw=2,
        label="Training Loss (MSE)",
        alpha=0.9,
    )
    ax.plot(
        epochs_range,
        history.history["val_loss"],
        color=COLORS["accent"],
        lw=2,
        label="Validation Loss (MSE)",
        alpha=0.9,
    )
    ax.axvline(
        x=best_epoch,
        color=COLORS["danger"],
        linestyle="--",
        lw=2,
        label=f"Best Epoch = {best_epoch}",
    )
    ax.scatter(
        [best_epoch],
        [best_val_loss],
        color=COLORS["danger"],
        s=100,
        zorder=5,
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (MSE)", fontsize=11)
    ax.set_title(
        "ANN: Training & Validation Loss (MSE)\nEarlyStopping prevents overfitting",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        "figures/ann_01_training_loss.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    print(f"[SAVED] figures/ann_01_training_loss.png")

    # PLOT 2 — MAE
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        epochs_range,
        history.history["mae"],
        color=COLORS["primary"],
        lw=2,
        label="Training MAE",
        alpha=0.9,
    )
    ax.plot(
        epochs_range,
        history.history["val_mae"],
        color=COLORS["accent"],
        lw=2,
        label="Validation MAE",
        alpha=0.9,
    )
    ax.axvline(
        x=best_epoch,
        color=COLORS["danger"],
        linestyle="--",
        lw=2,
        label=f"Best Epoch = {best_epoch}",
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Absolute Error (minutes)", fontsize=11)
    ax.set_title(
        f"ANN: Training & Validation MAE\nBest Val MAE = {best_val_mae:.4f} minutes",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("figures/ann_02_training_mae.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[SAVED] figures/ann_02_training_mae.png")

    # PLOT 3 — Actual vs predicted
    fig, ax = plt.subplots(figsize=(9, 8))
    hb = ax.hexbin(
        y_test_np,
        y_pred_test,
        gridsize=60,
        cmap="Purples",
        mincnt=1,
    )
    plt.colorbar(hb, ax=ax, label="Count")
    lims = [
        min(float(np.min(y_test_np)), float(np.min(y_pred_test))),
        max(float(np.max(y_test_np)), float(np.max(y_pred_test))),
    ]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual cycle_time (minutes)", fontsize=11)
    ax.set_ylabel("Predicted cycle_time (minutes)", fontsize=11)
    ax.set_title(
        f"ANN: Actual vs Predicted\nR²={r2_test:.4f} | MAE={mae_test:.4f} | RMSE={rmse_test:.4f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(
        "figures/ann_03_actual_vs_predicted.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    print(f"[SAVED] figures/ann_03_actual_vs_predicted.png")

    # PLOT 4–5 — SHAP (only if values computed)
    if shap_values is not None:
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_shap_df,
            feature_names=FEATURE_COLS,
            plot_type="dot",
            show=False,
            max_display=12,
        )
        plt.title(
            "ANN: SHAP Summary Plot (Beeswarm)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            "figures/ann_04_shap_beeswarm.png", dpi=150, bbox_inches="tight"
        )
        plt.close("all")
        print(f"[SAVED] figures/ann_04_shap_beeswarm.png")

        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_shap_df,
            feature_names=FEATURE_COLS,
            plot_type="bar",
            show=False,
        )
        plt.title(
            "ANN: SHAP Feature Importance (Bar)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig("figures/ann_05_shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"[SAVED] figures/ann_05_shap_bar.png")
    else:
        print(f"[INFO] Skipping SHAP plots (no SHAP values).")

    # PLOT 6 — Residuals
    residuals = y_test_np - y_pred_test
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(
        residuals,
        bins=60,
        color=COLORS["purple"],
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
        color=COLORS["purple"],
    )
    axes[1].axhline(0, color=COLORS["danger"], linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted Values (minutes)")
    axes[1].set_ylabel("Residuals (minutes)")
    axes[1].set_title("Residuals vs Fitted")
    plt.suptitle(
        f"ANN: Residual Analysis | RMSE={rmse_test:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("figures/ann_06_residuals.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[SAVED] figures/ann_06_residuals.png")

    # PLOT 7 — Architecture diagram
    fig = plt.figure(figsize=(8, 10))
    gs_arch = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs_arch[0, 0])
    ax.axis("off")

    layers_info = [
        ("INPUT LAYER", f"{INPUT_DIM} neurons", "#1B2A4A", "white"),
        (
            "HIDDEN LAYER 1",
            "128 neurons\nReLU + BatchNorm\nDropout 0.3",
            "#2E6DB4",
            "white",
        ),
        (
            "HIDDEN LAYER 2",
            "64 neurons\nReLU + BatchNorm\nDropout 0.2",
            "#1A7A6E",
            "white",
        ),
        ("HIDDEN LAYER 3", "32 neurons\nReLU", "#E07B39", "white"),
        (
            "OUTPUT LAYER",
            "1 neuron\nLinear activation\n→ cycle_time (minutes)",
            "#7B3FA0",
            "white",
        ),
    ]

    y_positions = [0.90, 0.72, 0.54, 0.36, 0.16]
    box_height = 0.14
    box_width = 0.65

    for idx, ((name, desc, fc, tc), ypos) in enumerate(
        zip(layers_info, y_positions)
    ):
        rect = plt.Rectangle(
            (0.175, ypos - box_height / 2),
            box_width,
            box_height,
            facecolor=fc,
            edgecolor="white",
            linewidth=2,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            0.5,
            ypos + 0.01,
            name,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=tc,
        )
        ax.text(
            0.5,
            ypos - 0.025,
            desc,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8.5,
            color=tc,
            multialignment="center",
        )
        if idx < len(y_positions) - 1:
            y_next = y_positions[idx + 1]
            y_bottom = ypos - box_height / 2
            y_top_next = y_next + box_height / 2
            ax.annotate(
                "",
                xy=(0.5, y_top_next + 0.02),
                xytext=(0.5, y_bottom - 0.02),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
            )

    ax.set_title(
        f"ANN Architecture\n({total_params_eval:,} trainable parameters)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(
        "figures/ann_07_architecture.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    print(f"[SAVED] figures/ann_07_architecture.png")

    # SECTION 8 — SAVE MODEL & ARTIFACTS
    best_model.save("models/ann_model.keras")
    try:
        best_model.save("models/ann_model.h5", save_format="h5")
    except Exception:
        try:
            best_model.save("models/ann_model.h5")
        except Exception as e:
            print(f"[WARNING] Could not save .h5: {e}")

    if shap_values is not None:
        np.save("results/ann_shap_values.npy", shap_values)
    shap_df.to_csv("results/ann_shap_importance.csv", index=False)
    pd.DataFrame(history.history).to_csv(
        "results/ann_training_history.csv", index=False
    )

    print(f"[SAVED] models/ann_model.keras")
    print(f"[SAVED] models/ann_model.h5")
    print(f"[SAVED] results/ann_shap_importance.csv")
    print(f"[SAVED] results/ann_training_history.csv")

    # SECTION 9 — FINAL SUMMARY
    top3_lines = ""
    if shap_values is not None and len(shap_df) >= 3:
        top3_lines = f"""Top 3 Features (SHAP):
  1. {shap_df.iloc[0]['Feature']} : {shap_df.iloc[0]['Mean_SHAP']:.4f}
  2. {shap_df.iloc[1]['Feature']} : {shap_df.iloc[1]['Mean_SHAP']:.4f}
  3. {shap_df.iloc[2]['Feature']} : {shap_df.iloc[2]['Mean_SHAP']:.4f}
"""
    else:
        top3_lines = "Top 3 Features (SHAP): N/A (SHAP failed)\n"

    print(
        f"""
=======================================================
        ANN (MLP) ANALYSIS COMPLETE
=======================================================
Architecture    : Input({INPUT_DIM}) → Dense(128) → Dense(64)
                  → Dense(32) → Dense(1)
Regularization  : BatchNorm + Dropout per hidden layer
Total Params    : {total_params_eval:,}

Training Config:
  Optimizer   : Adam (lr=0.001)
  Loss        : MSE
  Batch size  : 256
  Max epochs  : 200
  Best epoch  : {best_epoch}
  Training    : {train_time:.2f} seconds

PERFORMANCE (Test Set):
  R²   : {r2_test:.4f}
  MAE  : {mae_test:.4f} minutes
  RMSE : {rmse_test:.4f} minutes
  MAPE : {mape_test:.2f} %

{top3_lines}
FILES SAVED:
  models/ann_model.keras + ann_model.h5 + ann_best.keras
  results/ann_results.csv
  results/ann_training_history.csv
  results/ann_training_log.csv
  results/ann_shap_importance.csv
  figures/ann_01 through ann_07 (7 PNG files)
=======================================================
"""
    )

    elapsed = time.time() - t_script
    print(f"Total elapsed time: {elapsed:.2f} seconds")
