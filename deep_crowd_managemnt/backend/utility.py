import os
import tempfile
import shutil
import cv2
import numpy as np
from fastapi import UploadFile

def save_upload_tmpfile(upload_file: UploadFile, suffix: str = "") -> str:
    """
    Save an UploadFile to a temporary file and return the path.
    Caller is responsible for removing the file after use.
    """
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, tmp)
        tmp_path = tmp.name

    return tmp_path


def generate_heatmap_overlay(image_path: str, density: np.ndarray):
    """
    Read image from image_path, resize density to image shape, apply colormap and return JPEG bytes.
    If density is None, returns None.
    """
    if density is None:
        return None

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    d = np.array(density, dtype=np.float32)
    maxv = d.max() if d.size > 0 else 0.0
    if maxv > 0:
        d = d / maxv
    d_resized = cv2.resize(d, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((d_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb.astype(np.uint8), 0.6, heatmap.astype(np.uint8), 0.4, 0)
    success, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode heatmap image to JPG")

    return buf.tobytes()
