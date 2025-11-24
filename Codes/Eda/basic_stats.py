# eda/basic_stats.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from scipy.io import loadmat

def analyze_subset(dataset_root, part, mode):
    img_dir = os.path.join(dataset_root, part, mode, "images")
    gt_dir = os.path.join(dataset_root, part, mode, "ground-truth")
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".mat")])
    data = []

    for img_file, gt_file in zip(img_files, gt_files):
        img = Image.open(os.path.join(img_dir, img_file))
        w, h = img.size
        mat = loadmat(os.path.join(gt_dir, gt_file))
        pts = mat["image_info"][0,0][0,0][0]
        data.append({"part": part, "mode": mode, "image": img_file,
                     "width": w, "height": h, "people_count": len(pts)})
    return pd.DataFrame(data)

def summarize_dataset(dataset_root):
    all_dfs = []
    for part in ["part_A", "part_B"]:
        for mode in ["train_data", "test_data"]:
            df = analyze_subset(dataset_root, part, mode)
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

def check_corruption(dataset_root):
    corrupted = []
    for root, _, files in os.walk(dataset_root):
        for f in files:
            if f.endswith(".jpg"):
                try: Image.open(os.path.join(root, f)).verify()
                except: corrupted.append(f)
    return corrupted
