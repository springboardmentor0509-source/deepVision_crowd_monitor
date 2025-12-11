import os
import sys
import torch
import joblib
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

from simple_cnn.model_simplecnn import SimpleCNN
from csrNet_Model.csr_model import CSRNet
from mobile_csrnet.mobile_csr_model import MobileNetCSRNet
from random_forest.preprocessing_rf import extract_features


class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}

        # model weight paths
        self.models_dir = os.path.join(project_root, "models")
        self.model_paths = {
            "SimpleCNN": os.path.join(self.models_dir, "simple_cnn", "best_simplecnn.pth"),
            "CSRNet": os.path.join(self.models_dir, "csrnet_cnn", "best_csrnet_model.pth"),
            "MobileNetCSRNet": os.path.join(self.models_dir, "mobile_csrnet", "best_mobile_csrnet_model.pth"),
            "RandomForest": os.path.join(self.models_dir, "random_forest", "random_forest_model.pkl"),
        }

        # match training transform EXACTLY (modify if needed)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])

        torch.backends.cudnn.benchmark = True


    def load_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]

        print(f"[ModelManager] Loading {model_name}...")

        if model_name == "SimpleCNN":
            model = SimpleCNN().to(self.device)
            model.load_state_dict(torch.load(self.model_paths[model_name], map_location=self.device))

        elif model_name == "CSRNet":
            model = CSRNet().to(self.device)
            model.load_state_dict(torch.load(self.model_paths[model_name], map_location=self.device))

        elif model_name == "MobileNetCSRNet":
            model = MobileNetCSRNet().to(self.device)
            model.load_state_dict(torch.load(self.model_paths[model_name], map_location=self.device))

        elif model_name == "RandomForest":
            model = joblib.load(self.model_paths[model_name])
            self.models[model_name] = model
            return model

        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.eval()
        self.models[model_name] = model
        return model


    def predict(self, model_name, image_path):
        model = self.load_model(model_name)

        # RandomForest special case
        if model_name == "RandomForest":
            features = extract_features([image_path])
            count = float(model.predict(features)[0])
            return None, count

        # Load & transform image
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(tensor)

            # output shapes:
            # CSRNet: (1,1,H,W)
            # SimpleCNN: scalar or (1,)
            # MobileNetCSRNet: similar to CSRNet

            if not isinstance(output, torch.Tensor):
                raise RuntimeError("Model returned non-tensor output")

            out = output.squeeze().cpu().numpy()

            # scalar output (count regression)
            if np.isscalar(out):
                return None, float(out)

            # density map output (2D)
            if out.ndim == 2:
                density = out
                count = float(density.sum())
                return density, count

            # fallback — unexpected shape
            raise RuntimeError(f"Unexpected model output shape: {output.shape}")


manager = ModelManager()
