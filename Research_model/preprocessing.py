# Truck cycle dataset: load, clean, encode, scale, split, and save for downstream models.

# STEP 0 — IMPORTS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


def main():
    # STEP 1 — CREATE OUTPUT DIRECTORIES
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("data_processed", exist_ok=True)
    print(f"[INFO] Output directories created.")

    # STEP 2 — LOAD RAW DATA
    csv_path = "final_truck_data.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        alt_path = os.path.join("Dataset", "final_truck_data.csv")
        if os.path.isfile(alt_path):
            print(
                f"[INFO] '{csv_path}' not in cwd; loading from '{alt_path}' instead."
            )
            csv_path = alt_path
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"Could not find '{csv_path}' in the current working directory "
                f"or '{alt_path}'. Place final_truck_data.csv in the project root "
                f"or under Dataset/, then run this script again."
            ) from None

    print(f"[INFO] Loaded CSV from: {csv_path}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] dtypes:\n{df.dtypes}")
    print(f"[INFO] First 3 rows:\n{df.head(3)}")
    ct = df["cycle_time"]
    print(
        f"[INFO] cycle_time stats — mean={ct.mean():.4f}, std={ct.std():.4f}, "
        f"min={ct.min():.4f}, max={ct.max():.4f}"
    )

    # STEP 3 — SAMPLE DATA
    original_size = len(df)
    df = df.sample(n=100_000, random_state=SEED).reset_index(drop=True)
    print(
        f"[INFO] Sampled {len(df):,} rows from {original_size:,} total rows"
    )
    print(
        f"[INFO] Location_ID value counts (after sampling):\n{df['Location_ID'].value_counts()}"
    )
    print(
        f"[INFO] shift value counts (after sampling):\n{df['shift'].value_counts()}"
    )

    # STEP 4 — DROP UNNECESSARY COLUMNS
    drop_cols = [
        "cycle_id",
        "timestamp",
        "payload_weight",
        "loading_time",
        "hauling_time",
        "dumping_time",
        "returning_time",
    ]
    df = df.drop(columns=drop_cols)
    print(
        f"[INFO] Dropped 7 columns. Remaining columns: {df.columns.tolist()}"
    )

    # STEP 5 — ENCODE CATEGORICAL FEATURES

    # 5a. Encode 'shift'
    le_shift = LabelEncoder()
    df["shift"] = le_shift.fit_transform(df["shift"])
    print(f"shift encoding: Day=0, Night=1")

    # 5b. Encode 'truck_id'
    le_truck = LabelEncoder()
    df["truck_id"] = le_truck.fit_transform(df["truck_id"])
    print(f"truck_id encoded: {le_truck.classes_}")

    # 5c. One-Hot Encode 'Location_ID'
    location_dummies = pd.get_dummies(
        df["Location_ID"], prefix="loc", drop_first=True
    )
    df = pd.concat([df.drop("Location_ID", axis=1), location_dummies], axis=1)
    print(
        f"Location_ID encoded into columns: {location_dummies.columns.tolist()}"
    )

    # 5d. Save encoders
    joblib.dump(le_shift, "models/le_shift.pkl")
    joblib.dump(le_truck, "models/le_truck.pkl")
    print(f"[INFO] Encoders saved.")

    # STEP 6 — DEFINE FEATURES AND TARGET
    loc_feature_cols = sorted([c for c in df.columns if c.startswith("loc_")])
    FEATURE_COLS = [
        "road_condition_score",
        "traffic_score",
        "weather_score",
        "road_gradient",
        "hauling_distance",
        "truck_speed",
        "shift",
        "truck_id",
    ] + loc_feature_cols

    TARGET_COL = "cycle_time"
    print(f"[INFO] Final FEATURE_COLS: {FEATURE_COLS}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Target shape: {y.shape}")
    print(f"[INFO] Features: {FEATURE_COLS}")

    # STEP 7 — TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=SEED
    )

    n_total = len(X)
    print(
        f"[INFO] Train set : {len(X_train):,} rows ({len(X_train) / n_total * 100:.1f}%)"
    )
    print(
        f"[INFO] Val set   : {len(X_val):,} rows ({len(X_val) / n_total * 100:.1f}%)"
    )
    print(
        f"[INFO] Test set  : {len(X_test):,} rows ({len(X_test) / n_total * 100:.1f}%)"
    )

    # STEP 8 — FEATURE SCALING (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=FEATURE_COLS)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS)

    joblib.dump(scaler, "models/scaler.pkl")
    print(f"[INFO] StandardScaler fitted and saved.")

    # STEP 9 — SAVE PROCESSED DATA FILES
    X_train.to_csv("data_processed/X_train.csv", index=False)
    X_val.to_csv("data_processed/X_val.csv", index=False)
    X_test.to_csv("data_processed/X_test.csv", index=False)

    X_train_scaled.to_csv("data_processed/X_train_scaled.csv", index=False)
    X_val_scaled.to_csv("data_processed/X_val_scaled.csv", index=False)
    X_test_scaled.to_csv("data_processed/X_test_scaled.csv", index=False)

    y_train.to_csv("data_processed/y_train.csv", index=False)
    y_val.to_csv("data_processed/y_val.csv", index=False)
    y_test.to_csv("data_processed/y_test.csv", index=False)

    import json

    with open("data_processed/feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    print(f"[INFO] All processed data files saved to data_processed/")

    # STEP 10 — VERIFICATION SUMMARY
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y_min = float(y_train.min())
    y_max = float(y_train.max())

    print(
        f"""
=====================================================
     PREPROCESSING COMPLETE — SUMMARY
=====================================================
Original dataset    : 1,048,575 rows × 17 cols
After sampling      :   100,000 rows × 17 cols
Features used       : {len(FEATURE_COLS)} features
Target variable     : cycle_time (minutes)
Train set           : {len(X_train):,} rows
Validation set      : {len(X_val):,} rows
Test set            : {len(X_test):,} rows

Target statistics (y_train):
  Mean : {y_mean:.2f} minutes
  Std  : {y_std:.2f} minutes
  Min  : {y_min:.2f} minutes
  Max  : {y_max:.2f} minutes

Files saved:
  models/scaler.pkl
  models/le_shift.pkl
  models/le_truck.pkl
  data_processed/X_train.csv   (unscaled)
  data_processed/X_val.csv
  data_processed/X_test.csv
  data_processed/X_train_scaled.csv  (scaled)
  data_processed/X_val_scaled.csv
  data_processed/X_test_scaled.csv
  data_processed/y_train.csv
  data_processed/y_val.csv
  data_processed/y_test.csv
  data_processed/feature_cols.json
=====================================================
Preprocessing script is READY.
Run model scripts next.
=====================================================
"""
    )


if __name__ == "__main__":
    main()
