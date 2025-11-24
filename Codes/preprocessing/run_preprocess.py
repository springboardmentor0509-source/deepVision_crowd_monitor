
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.split_processor import process_split
import preprocessing.config as config


splits = [
    ("part_A", "train_data"),
    ("part_A", "test_data"),
    ("part_B", "train_data"),
    ("part_B", "test_data")
]

print("      Running Preprocessing for ALL Splits")


for part, mode in splits:
    print(f"\n\n>>> Processing: {part}/{mode}")
    df = process_split(
        dataset_root=config.DATASET_ROOT,
        out_root=config.OUT_ROOT,
        part=part,
        mode=mode,
        resize_short_side=config.RESIZE_SHORT_SIDE,
        extract_patches_flag=config.EXTRACT_PATCHES,
        patch_size=config.PATCH_SIZE,
        overlap=config.OVERLAP,
        save_resized=config.SAVE_RESIZED_IMAGES,
        k=config.K_NEIGHBORS
    )
    print(df.head())
print("      ALL PREPROCESSING COMPLETED SUCCESSFULLY!")
