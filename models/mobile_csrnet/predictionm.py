import torch
import cv2
import numpy as np
from mobile_csr_model import MobileNetCSRNet
import torchvision.transforms as transforms
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_mobile_csrnet_model.pth")
DEVICE = "cpu"

def load_model():
    model = MobileNetCSRNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"❌ Unable to read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    max_side = 720
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img

def predict(img_path):
    model = load_model()

    img_tensor, img_display = preprocess_image(img_path)
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)

    # Sum density map
    count = float(output.sum().item())
    count = round(count, 2)

    return count
if __name__ == "__main__":
    image_path = r"C:\Users\payan\Desktop\infosys_git\final\test_imagess\IMG_69.jpg"

    count, density = predict(image_path)
    print(f"Predicted Count: {count:.2f}")