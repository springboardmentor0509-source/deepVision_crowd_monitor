
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

from preprocessing.mat_parser import load_points_from_mat
from preprocessing.density_map import geometry_adaptive_density
from preprocessing.resize_utils import resize_image_density
from preprocessing.patch_extractor import extract_patches

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def list_images(dir):
    exts = ["*.jpg", "*.png", "*.jpeg"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(dir, e)))
    return sorted(files)

def process_split(dataset_root, out_root, part, mode,
                  resize_short_side, extract_patches_flag,
                  patch_size, overlap, save_resized, k):
    
    img_dir = os.path.join(dataset_root, part, mode, "images")
    gt_dir  = os.path.join(dataset_root, part, mode, "ground-truth")
    
    out_density = os.path.join(out_root, part, mode, "density")
    out_resized = os.path.join(out_root, part, mode, "images_resized")
    out_patch_i = os.path.join(out_root, part, mode, "patches/images")
    out_patch_d = os.path.join(out_root, part, mode, "patches/density")

    mkdir(out_density)
    if save_resized:
        mkdir(out_resized)
    if extract_patches_flag:
        mkdir(out_patch_i)
        mkdir(out_patch_d)

    images = list_images(img_dir)
    meta = []

    for img_path in tqdm(images, desc=f"Processing {part}/{mode}"):
        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]

        mat_path = os.path.join(gt_dir, f"GT_{base}.mat")
        pts = load_points_from_mat(mat_path)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        den = geometry_adaptive_density(pts, h, w, k)

        if resize_short_side:
            img, den = resize_image_density(img, den, resize_short_side)

        np.save(os.path.join(out_density, f"{base}.npy"), den)

        if save_resized:
            img.save(os.path.join(out_resized, img_name))

        patch_count = 0
        if extract_patches_flag:
            patch_count = extract_patches(img, den, patch_size, overlap,
                                          out_patch_i, out_patch_d, base)

        meta.append({
            "part": part,
            "mode": mode,
            "image": img_name,
            "width": img.size[0],
            "height": img.size[1],
            "people_count": float(den.sum()),
            "density_path": os.path.abspath(os.path.join(out_density, base + ".npy")),
            "resized_image_path": os.path.abspath(os.path.join(out_resized, img_name)),
            "patch_count": patch_count
        })

    df = pd.DataFrame(meta)
    df.to_csv(os.path.join(out_root, part, mode, "metadata.csv"), index=False)
    return df
