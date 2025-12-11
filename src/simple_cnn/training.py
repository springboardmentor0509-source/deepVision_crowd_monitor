# src/simple_cnn/training.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from simple_cnn.preprocessing_fast import SimpleCNNDatasetPreprocessed
from simple_cnn.model_simplecnn import SimpleCNN, save_model_architecture
from pathlib import Path

# Get project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models and code"
RESULT_DIR = PROJECT_ROOT / "results" / "simple_cnn"
PREPROCESSED_ROOT = PROJECT_ROOT / "processed_data"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def train_simplecnn():

    # Use preprocessed data for faster training

    LR = 1e-4
    BATCH = 4
    EPOCHS = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using preprocessed data from: {PREPROCESSED_ROOT}")
    train_ds = SimpleCNNDatasetPreprocessed(str(PREPROCESSED_ROOT), mode="train", part="A")
    val_ds   = SimpleCNNDatasetPreprocessed(str(PREPROCESSED_ROOT), mode="test",  part="A")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = SimpleCNN().to(DEVICE)

    # save architecture once
    save_model_architecture(model, str(MODEL_DIR / "simplecnn_architecture.txt"))

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_mae_list = [], []
    best_mae = float("inf")
    best_preds, best_gts = [], []

    for epoch in range(EPOCHS):

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
        train_losses.append(avg_loss)

        # ---------------- EVAL ----------------
        model.eval()
        mae = 0
        mse = 0
        preds, gts = [], []

        with torch.no_grad():
            for img, target, _ in val_loader:
                img, target = img.to(DEVICE), target.to(DEVICE)

                p = model(img)
                pred_c = p.sum().item()
                gt_c = target.sum().item()

                preds.append(pred_c)
                gts.append(gt_c)

                mae += abs(pred_c - gt_c)
                mse += (pred_c - gt_c) ** 2

        mae /= len(val_loader)
        rmse = np.sqrt(mse / len(val_loader))
        val_mae_list.append(mae)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss={avg_loss:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}")

        if mae < best_mae:
            best_mae = mae
            best_preds, best_gts = preds, gts
            torch.save(model.state_dict(), str(MODEL_DIR / "best_simplecnn.pth"))
            print("✔ Saved Best Model")

    # save metrics
    metrics = pd.DataFrame({
        "Epoch": list(range(1, EPOCHS + 1)),
        "Train Loss": train_losses,
        "Val MAE": val_mae_list
    })

    metrics.to_csv(str(RESULT_DIR / "cnn_training_metrics.csv"), index=False)

    # save best prediction data
    pd.DataFrame({"GT": best_gts, "Pred": best_preds}).to_csv(
        str(RESULT_DIR / "cnn_predictions.csv"), index=False
    )

    # Plot training curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_mae_list, label="Validation MAE")
    plt.legend()
    plt.grid()
    plt.title("Training Progress")
    plt.savefig(str(RESULT_DIR / "cnn_training_plot.png"))
    plt.close()
