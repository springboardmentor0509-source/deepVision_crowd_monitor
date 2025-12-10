# baseline_random_forest.py
import os
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET_ROOT = Path("/path/to/ShanghaiTech")     # ← update this
OUT_DIR       = Path("/path/to/output")          # ← update this
FEATURE_CSV   = OUT_DIR / "rf_features.csv"
MODEL_PATH    = OUT_DIR / "rf_baseline.joblib"
RANDOM_STATE  = 42

def image_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-7)
    return float(-np.sum(hist * np.log2(hist + 1e-7)))

def extract_features_from_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = {
        "mean_brightness": float(np.mean(gray)),
        "std_brightness": float(np.std(gray)),
        "entropy": image_entropy(gray),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float(np.mean(cv2.Canny(gray, 100, 200)) / 255.0),
    }

    # histogram (5-bin summary)
    hist = cv2.calcHist([gray], [0], None, [5], [0, 256]).ravel()
    hist = (hist / (hist.sum() + 1e-7)).tolist()
    for i in range(5):
        features[f"hist_b{i}"] = hist[i]

    return features

def build_feature_dataset(dataset_root, save_csv):
    rows = []
    parts = ["part_A", "part_B"]
    modes = ["train_data", "test_data"]

    for part in parts:
        for mode in modes:
            img_dir = dataset_root / part / mode / "images"
            gt_dir = dataset_root / part / mode / "ground-truth"

            if not img_dir.exists():
                print(f"[WARN] Missing directory: {img_dir}")
                continue

            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

            for img_file in tqdm(img_files, desc=f"{part}/{mode}", unit="img"):
                img_path = img_dir / img_file
                gt_path = gt_dir / f"GT_{img_file.split('.')[0]}.mat"

                if not gt_path.exists():
                    continue

                feats = extract_features_from_image(img_path)
                if feats is None:
                    continue

                # load people count from MAT file
                try:
                    mat = loadmat(str(gt_path))
                    pts = mat["image_info"][0, 0][0, 0][0]
                    people_count = len(pts)
                except:
                    continue

                rows.append({
                    "part": part,
                    "mode": mode,
                    "image": img_file,
                    "people_count": people_count,
                    **feats
                })

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)
    print(f"[INFO] Saved feature CSV → {save_csv} (rows={len(df)})")
    return df

def train_and_evaluate(df, model_path):
    X = df.drop(columns=["part", "mode", "image", "people_count"])
    y = df["people_count"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=200,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    print("[INFO] Training Random Forest...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"[RESULT] MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}")

    joblib.dump(model, model_path)
    print(f"[INFO] Saved model → {model_path}")

    return model
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if FEATURE_CSV.exists():
        print(f"[INFO] Loading existing CSV: {FEATURE_CSV}")
        df = pd.read_csv(FEATURE_CSV)
    else:
        print("[INFO] Extracting features from dataset...")
        df = build_feature_dataset(DATASET_ROOT, FEATURE_CSV)

    df = df.dropna().reset_index(drop=True)

    train_and_evaluate(df, MODEL_PATH)
    print("[DONE] Pipeline completed successfully.")

if __name__ == "__main__":
    main()