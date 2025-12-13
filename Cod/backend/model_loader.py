import os
import pickle
import sys
from xml.parsers.expat import model
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
        self.models_dir = os.path.join(project_root, "models")
        self.model_paths = {
            "SimpleCNN": os.path.join(self.models_dir, "simple_cnn", "best_simplecnn.pth"),
            "CSRNet": os.path.join(self.models_dir, "csrnet_cnn", "best_csrnet_model.pth"),
            "MobileNetCSRNet": os.path.join(self.models_dir, "mobile_csrnet", "best_mobile_csrnet_model.pth"),
            "RandomForest": os.path.join(self.models_dir, "random_forest", "random_forest_model.pkl"),
        }
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, model_name):
        """Load and cache model by name. Returns loaded model object."""
        if model_name in self.models:
            return self.models[model_name]
        print(f"Loading {model_name}...")

        if model_name == "SimpleCNN":
            model = SimpleCNN().to(self.device)
            model.load_state_dict(torch.load(self.model_paths[model_name], map_location=self.device))

        elif model_name == "CSRNet":
            model = CSRNet(load_weights=False).to(self.device)
            model.load_state_dict(torch.load(self.model_paths[model_name], map_location=self.device))

        elif model_name == "MobileNetCSRNet":
            model = MobileNetCSRNet(load_weights=False).to(self.device)
            model.load_state_dict(torch.load(self.model_paths[model_name], map_location=self.device))

        elif model_name == "RandomForest":
            model_path = self.model_paths["RandomForest"]

            try:
                # Try normal loading first
                model = joblib.load(model_path)
            except Exception as e:
                print("⚠ Joblib failed. Trying fallback pickle loader...")
                print("Error:", e)
                with open(model_path, "rb") as f:
                    model = pickle.load(f)   # built-in pickle works in Python 3.13

            self.models[model_name] = model
            return model


        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.eval()
        self.models[model_name] = model
        return model

    def predict(self, model_name, image_path):
        """
        Run prediction for given model and image path.
        Returns (density_map_or_None, count_float).
        """
        model = self.load_model(model_name)

        if model_name == "RandomForest":
            features = extract_features([image_path])
            count = model.predict(features)[0]
            return None, float(count)
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(tensor)

            if isinstance(output, torch.Tensor) and output.ndim >= 3:
                out = output.squeeze().cpu().numpy()
                if out.ndim == 0:
                    return None, float(out.item())
                if out.ndim == 2:
                    density = out
                    count = float(density.sum())
                    return density, count
                return None, float(out.sum())

            else:
                try:
                    val = float(output.sum().item())
                    return None, val
                except Exception:
                    raise RuntimeError("Model output has unexpected shape/type")

manager = ModelManager()
