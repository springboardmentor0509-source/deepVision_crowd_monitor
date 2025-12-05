import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

from mobile_csrnet.mobile_preprocessing import MobileCSRNetDataset
from mobile_csrnet.mobile_csr_model import MobileNetCSRNet, save_model_architecture


MODEL_DIR = "../../models/mobile_csrnet"
RESULT_DIR = "../../results/mobile_csrnet"
DATASET_ROOT = "../../dataset/ShanghaiTech"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def train_mobile_csrnet():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LR = 1e-5
    BATCH = 4
    EPOCHS = 30

    train_ds = MobileCSRNetDataset(DATASET_ROOT, part="A", mode="train")
    val_ds   = MobileCSRNetDataset(DATASET_ROOT, part="A", mode="test")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = MobileNetCSRNet(load_weights=True).to(DEVICE)

    save_model_architecture(model, f"{MODEL_DIR}/model_architecture.csv")

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_mae = float("inf")
    history = []

    for epoch in range(1, EPOCHS+1):

        # ---------------- TRAIN ----------------
        model.train()
        total_loss = 0

        for img, target, _ in train_loader:
            img, target = img.to(DEVICE), target.to(DEVICE)

            pred = model(img)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---------------- VAL ----------------
        model.eval()
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

        history.append({"epoch": epoch, "train_loss": avg_loss, "MAE": mae, "RMSE": rmse})

        print(f"Epoch {epoch}/{EPOCHS} | Loss={avg_loss:.3f} | MAE={mae:.2f} | RMSE={rmse:.2f}")

        # Save best model
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_mobile_csrnet.pth")
            print("✔ Saved best model")

    # Save metrics
    pd.DataFrame(history).to_csv(f"{RESULT_DIR}/mobile_csr_training_metrics.csv", index=False)

    # Plot training loss
    plt.figure()
    plt.plot([h["train_loss"] for h in history], label="Train Loss")
    plt.plot([h["MAE"] for h in history], label="Validation MAE")
    plt.legend()
    plt.title("Mobile CSRNet Training Curve")
    plt.grid()
    plt.savefig(f"{RESULT_DIR}/mobile_csr_training_plot.png")
    plt.close()
