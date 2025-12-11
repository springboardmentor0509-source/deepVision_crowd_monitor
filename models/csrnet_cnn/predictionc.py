import torch
import cv2
import numpy as np
from csr_model import CSRNet
from torchvision import transforms

MODEL_PATH = "best_csrnet_model.pth"

# Load model
def load_model():
    model = CSRNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# Image transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    model = load_model()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        density_map = model(img_tensor)

    count = density_map.sum().item()
    return round(count)
