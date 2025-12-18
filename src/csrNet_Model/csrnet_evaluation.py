import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Use relative imports to load modules from the same package
from .csrnet_preprocessing import CSRNetDataset
from .csr_model import CSRNet


MODEL_DIR = "../../models/csrnet_cnn"
RESULT_DIR = "../../results/csrnet_cnn"
DATASET_ROOT = "../../Dataset/ShanghaiTech"

# Resolve absolute paths (relative to this file) to avoid path issues
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(HERE, MODEL_DIR))
RESULT_DIR = os.path.abspath(os.path.join(HERE, RESULT_DIR))
DATASET_ROOT = os.path.abspath(os.path.join(HERE, DATASET_ROOT))

# Ensure output directories exist when running as a script
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def evaluate_csrnet():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = CSRNet(load_weights=False).to(DEVICE)
    model_path = os.path.join(MODEL_DIR, "best_csrnet_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    val_ds = CSRNetDataset(DATASET_ROOT, part="A", mode="test")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    preds, gts = [], []

    with torch.no_grad():
        for img, target, _ in val_loader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            out = model(img)

            p = float(out.sum().item())
            g = float(target.sum().item())

            preds.append(p)
            gts.append(g)

    preds = np.array(preds)
    gts = np.array(gts)

    if preds.size == 0 or gts.size == 0:
        print("No predictions or ground-truths were produced. Check dataset and model.")
        return

    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))

    pd.DataFrame([{"MAE": mae, "RMSE": rmse}]).to_csv(
        os.path.join(RESULT_DIR, "csrnet_final_metrics.csv"), index=False
    )

    # Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(gts, preds, alpha=0.6)
    # Safe plotting: compute mins/maxs
    xmin, xmax = float(gts.min()), float(gts.max())
    ymin, ymax = float(preds.min()), float(preds.max())
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title("CSRNet GT vs Pred")
    plt.grid()
    plt.savefig(os.path.join(RESULT_DIR, "csrnet_gt_vs_pred.png"))
    plt.close()

    print("Evaluation completed. MAE:", mae, "RMSE:", rmse)


if __name__ == "__main__":
    evaluate_csrnet()
