import numpy as np
import cv2
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def generate_density_map(img_shape, points, sigma=15):
    density = np.zeros(img_shape, dtype=np.float32)

    for p in points:
        x = min(int(p[0]), img_shape[1] - 1)
        y = min(int(p[1]), img_shape[0] - 1)
        density[y, x] += 1

    density = gaussian_filter(density, sigma=sigma)
    return density


def show_density_example(img_path, gt_path, save_path=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mat = loadmat(gt_path)
    pts = mat["image_info"][0, 0][0, 0][0]

    density = generate_density_map(img.shape[:2], pts)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original Image ({len(pts)} people)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(density, cmap="jet")
    plt.title(f"Density Map (Sum={density.sum():.0f})")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Saved: {save_path}")
        plt.close()
    else:
        plt.show()
