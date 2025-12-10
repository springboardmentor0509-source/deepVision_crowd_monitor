# dataset_torch.py

import torch
import numpy as np
from PIL import Image

class ShanghaiTechDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["resized_image_path"]).convert("RGB")
        den = np.load(row["density_path"]).astype(np.float32)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(np.array(img).transpose(2,0,1) / 255.0).float()

        den = torch.tensor(den).unsqueeze(0).float()
        return img, den
