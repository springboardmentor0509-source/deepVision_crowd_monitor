# training_rf.py

import os
import joblib
from preprocessing_rf import get_counts, extract_features
from sklearn.ensemble import RandomForestRegressor

MODEL_DIR = "../../models/random_forest"
RESULT_DIR = "../../results/random_forest"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DATA_ROOT = "../../dataset/ShanghaiTech"

TRAIN_DIR = f"{DATA_ROOT}/part_A/train_data"
TEST_DIR  = f"{DATA_ROOT}/part_A/test_data"

print("Loading train & test data...")
train_paths, train_counts = get_counts(TRAIN_DIR)
test_paths,  test_counts  = get_counts(TEST_DIR)

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
joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")
joblib.dump((test_X, test_y, test_paths), f"{MODEL_DIR}/test_data.pkl")

print(f"Model saved → {MODEL_DIR}/rf_model.pkl")
print(f"Test dataset saved → {MODEL_DIR}/test_data.pkl")
