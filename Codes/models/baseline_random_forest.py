# baseline_random_forest.py
import os
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
import argparse
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Configuration (adjust if needed)
# -----------------------------
# This assumes this file lives in: deepVision_crowd_monitor/Codes/models/
BASE_DIR = Path(__file__).resolve().parents[2]    # project root
DATASET_ROOT = BASE_DIR / "Dataset" / "ShanghaiTech"   # expected dataset location
OUT_DIR = Path(__file__).resolve().parent            # Codes/models/
FEATURE_CSV = OUT_DIR / "rf_features.csv"
MODEL_PATH = OUT_DIR / "rf_baseline.joblib"
RANDOM_STATE = 42

# -----------------------------
# Feature extraction utilities
# -----------------------------
def image_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-7)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return float(entropy)

def extract_features_from_image(img_path):
    """Return a dict of features for a single image. Returns None on failure."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_b = float(np.mean(gray))
    std_b = float(np.std(gray))
    ent = image_entropy(gray)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # blur/sharpness
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges) / 255.0)  # normalized [0,1]

    # Optionally: small histogram summary (5 bins)
    hist = cv2.calcHist([gray], [0], None, [5], [0, 256]).ravel()
    hist = (hist / (hist.sum() + 1e-7)).tolist()  # normalized

    features = {
        "mean_brightness": mean_b,
        "std_brightness": std_b,
        "entropy": ent,
        "laplacian_var": lap_var,
        "edge_density": edge_density,
        "hist_b0": hist[0],
        "hist_b1": hist[1],
        "hist_b2": hist[2],
        "hist_b3": hist[3],
        "hist_b4": hist[4],
    }
    return features

# -----------------------------
# Build dataset (features + label)
# -----------------------------
def build_feature_dataset(dataset_root: Path, save_csv: Path):
    assert dataset_root.exists(), f"Dataset not found at: {dataset_root}"

    rows = []
    parts = ["part_A", "part_B"]
    modes = ["train_data", "test_data"]  # include both for more data (you can change)

    for part in parts:
        for mode in modes:
            img_dir = dataset_root / part / mode / "images"
            gt_dir = dataset_root / part / mode / "ground-truth"
            if not img_dir.exists():
                print(f"[WARN] Missing: {img_dir} — skipping")
                continue

            img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
            for img_file in tqdm(img_files, desc=f"{part}/{mode}", unit="img"):
                img_path = img_dir / img_file
                gt_path = gt_dir / f"GT_{img_file.split('.')[0]}.mat"
                if not gt_path.exists():
                    # possible mismatch or missing ground-truth
                    # skip gracefully
                    continue

                feats = extract_features_from_image(img_path)
                if feats is None:
                    continue

                # Load ground-truth points and set people_count
                try:
                    mat = loadmat(str(gt_path))
                    pts = mat["image_info"][0, 0][0, 0][0]
                    people_count = int(len(pts))
                except Exception:
                    # if annotation loading fails, skip
                    continue

                row = {
                    "part": part,
                    "mode": mode,
                    "image": img_file,
                    "people_count": people_count,
                    **feats
                }
                rows.append(row)

    if not rows:
        raise RuntimeError("No feature rows generated. Check dataset paths and ground-truth files.")

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)
    print(f"[INFO] Saved feature CSV to: {save_csv}  (rows={len(df)})")
    return df

# -----------------------------
# Train & evaluate RF
# -----------------------------
def train_and_evaluate(df: pd.DataFrame, save_model_path: Path):
    # drop non-feature columns
    X = df.drop(columns=["part", "mode", "image", "people_count"])
    y = df["people_count"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    print("[INFO] Training Random Forest...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # Compute MSE then RMSE without using 'squared' kwarg to remain compatible
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))

    r2 = r2_score(y_test, preds)

    print(f"[RESULT] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

    # Save model
    joblib.dump(model, save_model_path)
    print(f"[INFO] Saved model to: {save_model_path}")

    # Optionally: return trained model and metrics
    return model, {"mae": mae, "rmse": rmse, "r2": r2}

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Baseline Random Forest for crowd-count (feature-based)")
    p.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT),
                   help="Path to ShanghaiTech dataset root (overrides default)")
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                   help="Output directory for csv/model")
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip feature extraction if rf_features.csv already exists")
    return p.parse_args()

def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "rf_features.csv"
    model_path = out_dir / "rf_baseline.joblib"

    if args.skip_extract and csv_path.exists():
        print(f"[INFO] Loading existing features CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[INFO] Building features from dataset: {dataset_root}")
        df = build_feature_dataset(dataset_root, csv_path)

    # Basic sanity: drop rows with NaNs if any
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"[WARN] Dropped {before-after} rows with NaN values")

    # Train
    model, metrics = train_and_evaluate(df, model_path)

    print("[DONE] Baseline pipeline finished.")

if __name__ == "__main__":
    main()
