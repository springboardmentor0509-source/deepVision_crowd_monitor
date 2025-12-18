# eda/visualization.py
import os, random, cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

def display_random_samples(dataset_root, part, mode, n=4):
    img_dir = os.path.join(dataset_root, part, mode, "images")
    gt_dir = os.path.join(dataset_root, part, mode, "ground-truth")
    files = random.sample(os.listdir(img_dir), n)
    plt.figure(figsize=(12,8))
    for i,f in enumerate(files):
        img_path = os.path.join(img_dir, f)
        gt_path = os.path.join(gt_dir, "GT_" + f.split('.')[0] + ".mat")

        # Read image as BGR (force 3 channels) and validate
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

        # Load ground-truth points (MAT structure may vary)
        mat = loadmat(gt_path)
        try:
            pts = mat["image_info"][0, 0][0, 0][0]
        except Exception:
            if "annPoints" in mat:
                pts = mat["annPoints"]
            elif "points" in mat:
                pts = mat["points"]
            else:
                raise

        # Draw circles on BGR image (cv2 uses BGR colors)
        for p in pts:
            cv2.circle(img_bgr, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

        # Convert to RGB for matplotlib
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.subplot(2,n//2,i+1)
        plt.imshow(img); plt.axis("off")
        plt.title(f"{f} | {len(pts)}")
    plt.tight_layout(); plt.show()

def plot_distributions(df):
    plt.figure(figsize=(10,6))
    sns.histplot(df[df.part=="part_A"].people_count, bins=30, kde=True, color='r', label='Part A')
    sns.histplot(df[df.part=="part_B"].people_count, bins=30, kde=True, color='b', label='Part B')
    plt.legend(); plt.title("Crowd Count Distribution"); plt.show()

    plt.figure(figsize=(6,5))
    sns.boxplot(x='part', y='people_count', data=df)
    plt.title("Crowd Count Comparison"); plt.show()
