# src/simple_cnn/prediction.py

import torch
import torchvision.transforms as transforms
from PIL import Image
from simple_cnn.model_simplecnn import SimpleCNN


MODEL_PATH = "models/simple_cnn/best_simplecnn.pth"


def predict_single_image(img_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        pred_map = model(img_tensor)
        count = pred_map.sum().item()

    print(f"Predicted Count: {count:.2f}")
    return count


# Example:
predict_single_image("test.jpg")
