import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from csrnet_Model.csrnet_preprocessing import CSRNetDataset
from csrnet_Model.csr_model import CSRNet, save_model_architecture


# Paths
MODEL_DIR = "../../models/csrnet_cnn"
RESULT_DIR = "../../results/csrnet_cnn"
DATASET_ROOT = "../../dataset/ShanghaiTech"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# Training Function
def train_csrnet():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LR = 1e-5
    BATCH = 4
    EPOCHS = 20

    train_ds = CSRNetDataset(DATASET_ROOT, part="A", mode="train")
    val_ds   = CSRNetDataset(DATASET_ROOT, part="A", mode="test")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = CSRNet(load_weights=True).to(DEVICE)

    # Save model architecture
    save_model_architecture(model, f"{MODEL_DIR}/csrnet_architecture.csv")

    criterion = nn.MSELoss(reduction='sum')
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
        mae = 0

        with torch.no_grad():
            for img, target, _ in val_loader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                pmap = model(img)

                p = float(pmap.sum().item())
                g = float(target.sum().item())

                preds.append(p)
                gts.append(g)
                mae += abs(p - g)

        mae /= len(val_loader)
        rmse = np.sqrt(np.mean((np.array(preds) - np.array(gts)) ** 2))

        history.append({"epoch": epoch, "train_loss": avg_loss, "MAE": mae, "RMSE": rmse})

        print(f"Epoch {epoch}/{EPOCHS} | Loss={avg_loss:.3f} | MAE={mae:.2f} | RMSE={rmse:.2f}")

        # Save best model
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_csrnet.pth")
            print("✔ Saved best model")

    # Save training history
    pd.DataFrame(history).to_csv(f"{RESULT_DIR}/csrnet_training_metrics.csv", index=False)

    # Plot training loss
    plt.figure()
    plt.plot([h["train_loss"] for h in history])
    plt.title("CSRNet Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(f"{RESULT_DIR}/csrnet_training_plot.png")
    plt.close()
