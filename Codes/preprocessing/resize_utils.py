# resize_utils.py
from PIL import Image
import cv2
import numpy as np

def resize_image_density(img_pil, density, target_short_side=None):
    w0, h0 = img_pil.size

    if h0 <= w0:
        new_h = target_short_side
        new_w = int(w0 * new_h / h0)
    else:
        new_w = target_short_side
        new_h = int(h0 * new_w / w0)

    img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
    den_resized = cv2.resize(density, (new_w, new_h))

    scale = density.sum() / (den_resized.sum() + 1e-12)
    den_resized *= scale

    return img_resized, den_resized.astype(np.float32)
