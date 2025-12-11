import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# Fast Dataset Loader for Mobile CSRNet using preprocessed data
class MobileCSRNetDatasetPreprocessed(Dataset):
    """
    Uses preprocessed data from processed_data/ folder.
    Much faster than generating density maps on-the-fly.
    """
    
    def __init__(self, preprocessed_root, part="A", mode="train",
                 target_size=(512, 512), downsample_ratio=8):

        self.preprocessed_root = preprocessed_root
        self.part = part
        self.mode = mode
        self.target_size = target_size
        self.downsample_ratio = downsample_ratio

        # Read metadata CSV
        metadata_path = os.path.join(preprocessed_root, f"part_{part}", 
                                      f"{mode}_data", "metadata.csv")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. "
                f"Please run preprocessing/run_preprocess.py first!"
            )
        
        self.metadata = pd.read_csv(metadata_path)
        
        self.img_dir = os.path.join(preprocessed_root, f"part_{part}", 
                                     f"{mode}_data", "images_resized")
        self.density_dir = os.path.join(preprocessed_root, f"part_{part}", 
                                        f"{mode}_data", "density")

        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load preprocessed image
        img_path = os.path.join(self.img_dir, row['image'])
        img = Image.open(img_path).convert("RGB")
        
        # Load preprocessed density map
        density_filename = os.path.splitext(row['image'])[0] + ".npy"
        density_path = os.path.join(self.density_dir, density_filename)
        density = np.load(density_path).astype(np.float32)
        
        # Resize density to match downsample ratio
        out_h = self.target_size[0] // self.downsample_ratio
        out_w = self.target_size[1] // self.downsample_ratio
        
        if density.shape != (out_h, out_w):
            from scipy.ndimage import zoom
            scale_h = out_h / density.shape[0]
            scale_w = out_w / density.shape[1]
            density = zoom(density, (scale_h, scale_w), order=1)
            # Preserve count after resize (handle empty density maps)
            current_sum = density.sum()
            if current_sum > 0 and row['people_count'] > 0:
                density = density * (row['people_count'] / current_sum)
        
        # Ensure float32 dtype
        density = density.astype(np.float32)
        
        return (
            self.transform(img),
            torch.from_numpy(density).unsqueeze(0).float(),
            img_path
        )
