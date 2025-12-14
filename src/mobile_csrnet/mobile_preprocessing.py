import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


# Density Map Generator
def generate_density_map(shape, points, sigma=2.0):
    H, W = shape
    density = np.zeros((H, W), dtype=np.float32)

    for p in points:
        x, y = int(p[0]), int(p[1])
        if 0 <= x < W and 0 <= y < H:
            density[y, x] += 1.0

    density = gaussian_filter(density, sigma=sigma, mode="constant")
    return density


# Dataset Loader for MobileNet CSRNet
class MobileCSRNetDataset(Dataset):
    def __init__(self, root_dir, part="A", mode="train",
                 target_size=(512, 512), downsample_ratio=8):

        self.img_dir = os.path.join(root_dir, f"part_{part}", f"{mode}_data", "images")
        self.gt_dir  = os.path.join(root_dir, f"part_{part}", f"{mode}_data", "ground-truth")

        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        self.target_size = target_size
        self.downsample_ratio = downsample_ratio

        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size
        img_resized = img.resize(self.target_size[::-1], Image.BILINEAR)

        name = os.path.basename(img_path).replace(".jpg", "")
        gt_path = os.path.join(self.gt_dir, f"GT_{name}.mat")

        if os.path.exists(gt_path):
            mat = loadmat(gt_path)
            try:
                points = mat["image_info"][0][0][0][0][0]
            except:
                points = np.array([])
        else:
            points = np.array([])

        # scale to resized image
        sx = self.target_size[1] / float(orig_w)
        sy = self.target_size[0] / float(orig_h)

        if len(points) > 0:
            points = points.astype(np.float32)
            points[:, 0] *= sx
            points[:, 1] *= sy
            points = points / self.downsample_ratio

        # density map shape
        out_h = self.target_size[0] // self.downsample_ratio
        out_w = self.target_size[1] // self.downsample_ratio

        density = generate_density_map((out_h, out_w), points, sigma=2.0)

        return (
            self.transform(img_resized),
            torch.from_numpy(density).unsqueeze(0),
            img_path
        )
