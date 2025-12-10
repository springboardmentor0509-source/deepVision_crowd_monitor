# patch_extractor.py
import numpy as np

def extract_patches(img_pil, density, patch_size, overlap, out_img, out_den, base_name):
    w, h = img_pil.size
    step = int(patch_size * (1 - overlap))

    count = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = min(y, h - patch_size)
            x1 = min(x, w - patch_size)

            box = (x1, y1, x1 + patch_size, y1 + patch_size)
            img_patch = img_pil.crop(box)
            den_patch = density[y1:y1+patch_size, x1:x1+patch_size]

            if den_patch.sum() < 1:
                continue

            img_patch.save(f"{out_img}/{base_name}_{y1}_{x1}.jpg")
            np.save(f"{out_den}/{base_name}_{y1}_{x1}.npy", den_patch)

            count += 1

    return count