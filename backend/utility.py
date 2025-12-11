import os
import tempfile
import shutil
import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image

def save_upload_tmpfile(upload_file: UploadFile, suffix: str = "") -> str:
    """
    Save an UploadFile to a temporary file and return the path.
    """
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name


def load_image_safe(path: str):
    """Safely load image using cv2 with PIL fallback."""
    img = cv2.imread(path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # fallback for HEIC / JPEG errors
    try:
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img)
    except Exception:
        raise ValueError(f"Unable to load image: {path}")


def generate_heatmap_overlay(image_path: str, density: np.ndarray):
    """
    Generate a heatmap overlay (PNG bytes) for UI display.
    """
    if density is None:
        return None

    # safe load
    img = load_image_safe(image_path)
    h, w = img.shape[:2]

    # normalize density map
    d = density.astype(np.float32)
    d_min, d_max = d.min(), d.max()
    d_norm = (d - d_min) / (d_max - d_min + 1e-6)

    # resize density to image size
    d_resized = cv2.resize(d_norm, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap = cv2.applyColorMap((d_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img.astype(np.uint8), 0.6, heatmap, 0.4, 0)

    # encode as PNG (no quality loss)
    success, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Could not encode heatmap image")

    return buf.tobytes()
