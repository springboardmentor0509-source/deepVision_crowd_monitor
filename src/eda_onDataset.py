import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now imports work
from eda.basic_stats import summarize_dataset, check_corruption
from eda.visualization import display_random_samples, plot_distributions
from eda.density_map_demo import show_density_example
from eda.advanced_eda import compute_brightness_entropy, plot_advanced_correlations

DATASET_ROOT = r"E:\mertor\P4_AI-DeepVision\DeepVision_Crowd_Monitor\dataset\ShanghaiTech"
if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(f" Dataset not found at: {DATASET_ROOT}")
else:
    print(f" Dataset found at: {DATASET_ROOT}")

if __name__ == "__main__":
    print("=== STEP 1: BASIC STATS ===")
    df = summarize_dataset(DATASET_ROOT)
    print(df.head())
    new_df= df.groupby(["part","mode"])["people_count"].describe()
    print(new_df)
    # new_df.to_csv("reults/dataset_summary_detailed.csv", index=False)
    print("\n=== STEP 6: CORRUPTION CHECK ===")
    bad_files = check_corruption(DATASET_ROOT)
    print(f"Corrupted Images Found: {len(bad_files)}")

    print("\n=== STEP 2: SAMPLE VISUALS ===")
    display_random_samples(DATASET_ROOT, "part_A", "train_data", n=4)
    display_random_samples(DATASET_ROOT, "part_B", "train_data", n=4)

    print("\n=== STEP 3: DISTRIBUTION PLOTS ===")
    plot_distributions(df)

    print("\n=== STEP 4: DENSITY MAP DEMO ===")
    sample_img = os.path.join(DATASET_ROOT, "part_A/train_data/images/IMG_1.jpg")
    sample_gt  = os.path.join(DATASET_ROOT, "part_A/train_data/ground-truth/GT_IMG_1.mat")
    show_density_example(sample_img, sample_gt)

    print("\n=== STEP 5: ADVANCED ANALYSIS ===")
    df = compute_brightness_entropy(DATASET_ROOT, df)
    plot_advanced_correlations(df)

    
