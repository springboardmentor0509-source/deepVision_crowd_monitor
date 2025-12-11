import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from model_simplecnn import SimpleCNN  # Import your CNN model

# ------------------------------
# Load Trained Model
# ------------------------------
model_path = r"C:\Users\payan\Desktop\infosys_git\final\models\simple_cnn\best_simplecnn.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------------------
# Image Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(img):
    """Convert BGR → RGB → Tensor → Normalized"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Resize to nearest multiple of 16 to avoid scaling distortion
    new_h, new_w = (h // 16) * 16, (w // 16) * 16
    img = cv2.resize(img, (new_w, new_h))

    tensor_img = transform(img).unsqueeze(0).to(device)
    return tensor_img

# ------------------------------
# Prediction Function
# ------------------------------
def predict(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image:", image_path)
        return None

    input_tensor = preprocess_image(img)

    with torch.no_grad():
        density_map = model(input_tensor)

    # Convert to numpy
    density_np = density_map.squeeze().detach().cpu().numpy()

    # Sum density map → person count
    count = float(np.sum(density_np))

    return count, density_np


# ------------------------------
# Run Prediction
# ------------------------------
if __name__ == "__main__":
    image_path = r"C:\Users\payan\Desktop\infosys_git\final\test_imagess\IMG_11.jpg"

    count, density = predict(image_path)
    print(f"Predicted Count: {count:.2f}")
