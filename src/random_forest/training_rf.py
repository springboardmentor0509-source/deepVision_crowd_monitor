# training_rf.py

import os
import joblib
from pathlib import Path
from random_forest.preprocessing_rf import get_counts, extract_features
from sklearn.ensemble import RandomForestRegressor

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models and code"
RESULT_DIR = PROJECT_ROOT / "results" / "random_forest"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = PROJECT_ROOT / "Dataset" / "ShanghaiTech"

TRAIN_DIR = DATA_ROOT / "part_A" / "train_data"
TEST_DIR  = DATA_ROOT / "part_A" / "test_data"

def train_random_forest():

    print("Loading train & test data...")
    train_paths, train_counts = get_counts(str(TRAIN_DIR))
    test_paths,  test_counts  = get_counts(str(TEST_DIR))

    print("Extracting features...")
    train_X = extract_features(train_paths)
    test_X  = extract_features(test_paths)

    train_y = train_counts
    test_y  = test_counts

    print("Training Random Forest Model...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )

    rf.fit(train_X, train_y)
    print("Training completed!")

    # Save model + test dataset
    joblib.dump(rf, str(MODEL_DIR / "random_forest_model.pkl"))
    joblib.dump((test_X, test_y, test_paths), str(MODEL_DIR / "rf_test_data.pkl"))

    print(f"Model saved → {MODEL_DIR / 'random_forest_model.pkl'}")
    print(f"Test dataset saved → {MODEL_DIR / 'rf_test_data.pkl'}")
