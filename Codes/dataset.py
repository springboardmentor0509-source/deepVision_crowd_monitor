import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class CrowdDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        data_path: Path to the folder containing images
        transform: Optional image transformations (like resizing/normalizing)
        """
        self.data_path = data_path
        # List all .jpg images in the folder
        self.img_list = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.data_path, img_name)
        
        # 1. Load the Image
        # Load in BGR format (OpenCV standard)
        image = cv2.imread(img_path)
        # Convert to RGB (standard for AI models)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Pre-process (Make it compatible with PyTorch)
        # Normalize pixel values to be between 0 and 1
        image = image.astype(np.float32) / 255.0
        # Move channels to the front: (Height, Width, 3) -> (3, Height, Width)
        image = image.transpose((2, 0, 1)) 
        
        # Convert to Tensor (PyTorch format)
        image_tensor = torch.from_numpy(image)

        # Note: In a real training loop, we would also load the 'Ground Truth' here.
        # For now, we are just returning the image to test the pipeline.
        
        return image_tensor, img_name

# Test the Dataset
if __name__ == "__main__":
    # Point this to the 'frames' folder you created earlier
    path = r"E:\DeepVision Crowd Monitor\Codes\dataCollections\frames"
    
    # Check if folder exists before running
    if os.path.exists(path):
        dataset = CrowdDataset(path)
        print(f"✅ Found {len(dataset)} images in the dataset.")
        
        # Try loading one image
        if len(dataset) > 0:
            data, name = dataset[0]
            print(f"Successfully loaded image: {name} with shape {data.shape}")
    else:
        print(f"❌ Error: The folder '{path}' does not exist. Check the path.")