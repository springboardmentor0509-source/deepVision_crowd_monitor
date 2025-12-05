import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

RESULT_DIR = "results/simple_cnn"

df = pd.read_csv(f"{RESULT_DIR}/cnn_predictions.csv")

preds = df["Pred"].values
gts   = df["GT"].values
errors = preds - gts

# Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(gts, preds, alpha=0.6)
plt.plot([gts.min(), gts.max()], [gts.min(), gts.max()], "r--")
plt.xlabel("Ground Truth")
plt.ylabel("Predicted")
plt.title("GT vs Pred")
plt.grid()
plt.savefig(f"{RESULT_DIR}/cnn_gt_vs_pred.png")
plt.close()

# Error Histogram
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=40)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.savefig(f"{RESULT_DIR}/cnn_error_histogram.png")
plt.close()

# First 50 samples plot
N = min(50, len(gts))
plt.figure(figsize=(12, 5))
plt.plot(gts[:N], label="GT")
plt.plot(preds[:N], label="Pred")
plt.legend()
plt.savefig(f"{RESULT_DIR}/cnn_gt_vs_pred_first50.png")
plt.close()

# Residual Plot
plt.figure(figsize=(7, 5))
plt.scatter(gts, errors)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("GT")
plt.ylabel("Residual (Pred-GT)")
plt.savefig(f"{RESULT_DIR}/cnn_residual_plot.png")
plt.close()
