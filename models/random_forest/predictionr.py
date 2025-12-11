import cv2
import numpy as np
import joblib
from skimage.feature import hog

# ---------------------------
# MODEL PATH
# ---------------------------
MODEL_PATH = r"C:\Users\payan\Desktop\infosys_git\final\models\random_forest\random_forest_model.pkl"


# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
def extract_features(image):
    """
    Extract HOG + intensity features for Random Forest.
    """
    # Resize image
    image_resized = cv2.resize(image, (128, 128))

    # Convert to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # HOG features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )

    # Flatten raw pixel intensities
    pixels = gray.flatten() / 255.0

    # Combine features
    features = np.hstack((hog_features, pixels))

    return features


# ---------------------------
# LOAD MODEL
# ---------------------------
def load_model():
    print("Loading Random Forest model...")
    model = joblib.load(MODEL_PATH)
    return model


# ---------------------------
# MAKE PREDICTION
# ---------------------------
def predict(image_path):
    # Load image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Extract features
    features = extract_features(img)
    features = features.reshape(1, -1)

    # Load model
    model = load_model()

    # Predict count
    count = model.predict(features)[0]

    return max(0, round(count))  # avoid negative predictions


# ---------------------------
# MAIN TEST
# ---------------------------
if __name__ == "__main__":
    test_image = r"C:\Users\payan\Desktop\infosys_git\final\test_imagess\IMG_13.jpg"

    try:
        result = predict(test_image)
        print(f"Predicted Crowd Count: {result}")
    except Exception as e:
        print("Error:", e)
