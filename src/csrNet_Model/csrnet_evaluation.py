import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from csrnet_Model.csrnet_preprocessing import CSRNetDataset
from csrnet_Model.csr_model import CSRNet


MODEL_DIR = "../../models/csrnet_cnn"
RESULT_DIR = "../../results/csrnet_cnn"
DATASET_ROOT = "../../dataset/ShanghaiTech"


def evaluate_csrnet():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = CSRNet(load_weights=False).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_csrnet_model", map_location=DEVICE))
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

    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))

    pd.DataFrame([{"MAE": mae, "RMSE": rmse}]).to_csv(
        f"{RESULT_DIR}/csrnet_final_metrics.csv", index=False
    )

    # Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(gts, preds, alpha=0.6)
    plt.plot([gts.min(), gts.max()], [gts.min(), gts.max()], 'r--')
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title("CSRNet GT vs Pred")
    plt.grid()
    plt.savefig(f"{RESULT_DIR}/csrnet_gt_vs_pred.png")
    plt.close()

    print("Evaluation completed. MAE:", mae, "RMSE:", rmse)
