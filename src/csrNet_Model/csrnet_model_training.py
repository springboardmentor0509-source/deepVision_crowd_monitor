import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from csrNet_Model.csrnet_preprocessing import CSRNetDataset
from csrNet_Model.csr_model import CSRNet, save_model_architecture
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models and code"
RESULT_DIR = PROJECT_ROOT / "results" / "csrnet_cnn"
DATASET_ROOT = PROJECT_ROOT / "Dataset" / "ShanghaiTech"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# Training Function
def train_csrnet():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LR = 1e-5
    BATCH = 4
    EPOCHS = 20

    train_ds = CSRNetDataset(str(DATASET_ROOT), part="A", mode="train")
    val_ds   = CSRNetDataset(str(DATASET_ROOT), part="A", mode="test")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = CSRNet(load_weights=True).to(DEVICE)

    # Save model architecture
    save_model_architecture(model, str(MODEL_DIR / "csrnet_architecture.csv"))

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
            torch.save(model.state_dict(), str(MODEL_DIR / "best_csrnet_model.pth"))
            print("✔ Saved best model")

    # Save training history
    pd.DataFrame(history).to_csv(str(RESULT_DIR / "csr_training_metrics.csv"), index=False)

    # Plot training loss
    plt.figure()
    plt.plot([h["train_loss"] for h in history])
    plt.title("CSRNet Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(str(RESULT_DIR / "csrnet_training_plot.png"))
    plt.close()
