# density_map.py

import numpy as np
from scipy.spatial import KDTree
import cv2

def geometry_adaptive_density(points, h, w, k=3, sigma_scale=0.3):
    density = np.zeros((h, w), np.float32)
    N = points.shape[0]
    if N == 0:
        return density

    pts = points.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    try:
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=k + 1)
        avg_neigh = np.mean(dists[:, 1:], axis=1)
    except:
        avg_neigh = np.ones(N) * 15

    for i, p in enumerate(pts):
        x, y = int(p[0]), int(p[1])
        sigma = max(1.0, avg_neigh[i] * sigma_scale)

        rad = int(3 * sigma)
        x1, x2 = max(0, x - rad), min(w - 1, x + rad)
        y1, y2 = max(0, y - rad), min(h - 1, y + rad)

        xs = np.arange(x1, x2 + 1)
        ys = np.arange(y1, y2 + 1)
        gx, gy = np.meshgrid(xs, ys)

        kernel = np.exp(-((gx - x)**2 + (gy - y)**2)/(2 * sigma**2))
        kernel /= kernel.sum() + 1e-7

        density[y1:y2 + 1, x1:x2 + 1] += kernel.astype(np.float32)

    return density
