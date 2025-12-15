import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from mobile_csrnet.mobile_preprocessing import MobileCSRNetDataset
from mobile_csrnet.mobile_csr_model import MobileNetCSRNet


MODEL_DIR = "../../models/mobile_csrnet"
RESULT_DIR = "../../results/mobile_csrnet"
DATASET_ROOT = "../../dataset/ShanghaiTech"


def evaluate_mobile_csrnet():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = MobileNetCSRNet(load_weights=False).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_mobile_csrnet.pth", map_location=DEVICE))
    model.eval()

    val_ds = MobileCSRNetDataset(DATASET_ROOT, part="A", mode="test")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    preds, gts = [], []

    with torch.no_grad():
        for img, target, _ in val_loader:
            img, target = img.to(DEVICE), target.to(DEVICE)

            p = model(img)
            preds.append(float(p.sum().item()))
            gts.append(float(target.sum().item()))

    preds = np.array(preds)
    gts = np.array(gts)

    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))

    pd.DataFrame([{"MAE": mae, "RMSE": rmse}]).to_csv(
        f"{RESULT_DIR}/final_error_matrix_best_model.csv", index=False
    )

    # scatter
    plt.figure(figsize=(6,6))
    plt.scatter(gts, preds, alpha=0.6)
    plt.plot([gts.min(), gts.max()], [gts.min(), gts.max()], 'r--')
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title("Mobile CSRNet: GT vs Pred")
    plt.grid()
    plt.savefig(f"{RESULT_DIR}/mobile_csr_gt_vs_pred.png")
    plt.close()

    print("Evaluation complete! MAE:", mae, "RMSE:", rmse)
