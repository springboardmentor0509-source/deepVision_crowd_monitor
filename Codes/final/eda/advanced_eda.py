# eda/advanced_eda.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

def compute_brightness_entropy(dataset_root, df):
    brights, ents = [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Computing brightness/entropy"):
        img_path = os.path.join(dataset_root, row["part"], row["mode"], "images", row["image"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            brights.append(np.nan)
            ents.append(np.nan)
            continue

        # Brightness (mean pixel value)
        brights.append(float(img.mean()))

        # Entropy (image complexity measure)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
        hist /= hist.sum() + 1e-7
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        ents.append(float(entropy))

    df["brightness"] = brights
    df["entropy"] = ents
    return df


def plot_advanced_correlations(df):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="brightness", y="people_count", hue="part", data=df)
    plt.title("Brightness vs People Count")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="entropy", y="people_count", hue="part", data=df)
    plt.title("Image Entropy vs People Count")
    plt.show()
