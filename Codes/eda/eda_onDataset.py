import sys
import os

# Add parent directory to Python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports from eda package
from eda.basic_stats import summarize_dataset, check_corruption
from eda.visualization import display_random_samples, plot_distributions
from eda.density_map_demo import show_density_example
from eda.advanced_eda import compute_brightness_entropy, plot_advanced_correlations

# ============================
# CONFIG
# ============================

DATASET_ROOT = r"D:\DeepVision\Dataset\ShanghaiTech"

# Create /charts directory (sibling to this file)
CHARTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)
print(f"[INFO] Charts will be saved to: {CHARTS_DIR}")

# Check dataset path
if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(f"[ERROR] Dataset NOT found at: {DATASET_ROOT}")
else:
    print(f"[INFO] Dataset found at: {DATASET_ROOT}")

# ============================
# MAIN SCRIPT
# ============================

if __name__ == "__main__":

    # STEP 1 ───────────────────────────────────────────────
    print("\n=== STEP 1: BASIC STATS ===")
    df = summarize_dataset(DATASET_ROOT)
    print(df.head())
    print(df.groupby(["part", "mode"])["people_count"].describe())

    # STEP 6 ───────────────────────────────────────────────
    print("\n=== STEP 6: CORRUPTION CHECK ===")
    bad_files = check_corruption(DATASET_ROOT)
    print(f"Corrupted Images Found: {len(bad_files)}")

    # STEP 2 ───────────────────────────────────────────────
    print("\n=== STEP 2: SAMPLE VISUALS ===")
    display_random_samples(
        DATASET_ROOT, "part_A", "train_data",
        n=4,
        save_path=os.path.join(CHARTS_DIR, "samples_partA.png")
    )
    display_random_samples(
        DATASET_ROOT, "part_B", "train_data",
        n=4,
        save_path=os.path.join(CHARTS_DIR, "samples_partB.png")
    )

    # STEP 3 ───────────────────────────────────────────────
    print("\n=== STEP 3: DISTRIBUTION PLOTS ===")
    plot_distributions(
        df,
        save_path=os.path.join(CHARTS_DIR, "distributions.png")
    )

    # STEP 4 ───────────────────────────────────────────────
    print("\n=== STEP 4: DENSITY MAP DEMO ===")
    sample_img = os.path.join(DATASET_ROOT, "part_A/train_data/images/IMG_1.jpg")
    sample_gt  = os.path.join(DATASET_ROOT, "part_A/train_data/ground-truth/GT_IMG_1.mat")

    show_density_example(
        sample_img,
        sample_gt,
        save_path=os.path.join(CHARTS_DIR, "density_demo.png")
    )

    # STEP 5 ───────────────────────────────────────────────
    print("\n=== STEP 5: ADVANCED ANALYSIS ===")
    df = compute_brightness_entropy(DATASET_ROOT, df)

    plot_advanced_correlations(
        df,
        save_path=os.path.join(CHARTS_DIR, "advanced_corr.png")
    )

    print("\n[INFO] EDA Completed Successfully!")
    print(f"[INFO] All charts saved inside: {CHARTS_DIR}")
