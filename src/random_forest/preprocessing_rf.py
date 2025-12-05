import os
import glob
import cv2
import numpy as np
from scipy.io import loadmat
from skimage.feature import local_binary_pattern, hog


def get_counts(folder):
    """
    Reads image paths and crowd counts from .mat ground truth files.
    """
    img_dir = os.path.join(folder, "images")
    gt_dir = os.path.join(folder, "ground-truth")

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    counts = []

    for img_path in img_paths:
        base = os.path.basename(img_path).replace(".jpg", "")
        mat_path = os.path.join(gt_dir, f"GT_{base}.mat")

        mat = loadmat(mat_path)
        points = mat["image_info"][0][0][0][0][0]   # coordinates array
        counts.append(len(points))

    return img_paths, np.array(counts)


def extract_features(img_paths, size=128):
    """
    Extracts feature vector for each image:
    - raw grayscale pixels
    - HOG
    - LBP histogram
    - edge density
    - brightness mean + variance
    - ORB keypoint count
    """
    features = []

    radius = 2
    n_points = 8 * radius

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            print("Could not read:", p)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (size, size))

        raw_pix = gray_small.flatten()

        hog_feat = hog(
            gray_small,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=False,
            channel_axis=None
        )

        lbp = local_binary_pattern(gray_small, n_points, radius, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=int(lbp.max() + 1), density=True)

        edges = cv2.Canny(gray_small, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        mean_intensity = np.mean(gray_small)
        var_intensity = np.var(gray_small)

        orb = cv2.ORB_create()
        kpt_count = len(orb.detect(gray_small, None))

        feat_vector = np.concatenate([
            raw_pix,
            hog_feat,
            lbp_hist,
            [edge_density],
            [mean_intensity, var_intensity],
            [kpt_count]
        ])

        features.append(feat_vector)

    return np.array(features)
