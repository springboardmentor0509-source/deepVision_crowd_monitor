# resize_utils.py
from PIL import Image
import cv2
import numpy as np


def resize_image_density(img_pil: Image.Image, density: np.ndarray, target_short_side: int | None = None):
    """Resize PIL image and corresponding density map while preserving total mass.

    If `target_short_side` is None, the original image and density are returned.
    """
    if target_short_side is None:
        return img_pil, density.astype(np.float32)

    if not isinstance(target_short_side, int) or target_short_side <= 0:
        raise ValueError("target_short_side must be a positive integer or None")

    w0, h0 = img_pil.size

    if h0 <= w0:
        new_h = target_short_side
        new_w = max(1, int(w0 * new_h / h0))
    else:
        new_w = target_short_side
        new_h = max(1, int(h0 * new_w / w0))

    # Support both old and new Pillow APIs for resampling constants
    # Determine resample constant without triggering static attribute lookup errors
    resample = getattr(Image, 'BILINEAR', None)
    if resample is None:
        resampling_cls = getattr(Image, 'Resampling', None)
        if resampling_cls is not None:
            resample = getattr(resampling_cls, 'BILINEAR', 1)
        else:
            # Fallback numeric value (should be a valid interpolation flag)
            resample = 1
    img_resized = img_pil.resize((new_w, new_h), resample)
    # cv2.resize expects (width, height) as (cols, rows) when given as tuple
    den_resized = cv2.resize(density, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Preserve total density sum (avoid divide-by-zero)
    orig_sum = float(density.sum())
    resized_sum = float(den_resized.sum())
    if resized_sum > 0:
        den_resized = den_resized * (orig_sum / (resized_sum + 1e-12))

    return img_resized, den_resized.astype(np.float32)
