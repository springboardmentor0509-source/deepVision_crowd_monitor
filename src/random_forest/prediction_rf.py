# prediction_rf.py

import joblib
import numpy as np
from preprocessing_rf import extract_features
import cv2
import os

MODEL_PATH = "../../models/random_forest/rf_model.pkl"

rf = joblib.load(MODEL_PATH)
print("Model loaded!")

def predict_count(image_path):
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return None

    features = extract_features([image_path])
    pred = rf.predict(features)[0]

    print(f"Image: {image_path} → Predicted Count: {pred:.2f}")
    return pred


# Example usage:
predict_count("../../dataset/ShanghaiTech/part_A/test_data/images/IMG_1.jpg")
