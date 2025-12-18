# evaluating_rf.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_DIR = "../../models/random_forest"
SAVE_DIR = "../../results/random_forest"

os.makedirs(SAVE_DIR, exist_ok=True)

# Load model & test data
rf = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
test_X, test_y, test_paths = joblib.load(f"{MODEL_DIR}/test_data.pkl")

pred = rf.predict(test_X)
errors = pred - test_y

# ----- Metrics -----
mae  = mean_absolute_error(test_y, pred)
rmse = np.sqrt(mean_squared_error(test_y, pred))
r2   = r2_score(test_y, pred)

print("==== Evaluation Report ====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
print("===========================")

# Save metrics
with open(f"{SAVE_DIR}/metrics.csv", "w") as f:
    f.write("Metric,Value\n")
    f.write(f"MAE  : {mae:.4f}\n")
    f.write(f"RMSE : {rmse:.4f}\n")
    f.write(f"R²   : {r2:.4f}\n")

print(f"Metrics saved → {SAVE_DIR}/metrics.csv")


# ----- Plots -----

# Scatter GT vs Pred
plt.figure(figsize=(6,6))
plt.scatter(test_y, pred, alpha=0.6)
plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], '--')
plt.xlabel("Ground Truth")
plt.ylabel("Prediction")
plt.title("GT vs Prediction")
plt.grid(True)
plt.savefig(f"{SAVE_DIR}/scatter_gt_vs_pred.png")
plt.close()

# Error histogram
plt.figure(figsize=(8,5))
plt.hist(errors, bins=40)
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.grid(True)
plt.savefig(f"{SAVE_DIR}/error_histogram.png")
plt.close()

# Line plot for first 50
N = min(50, len(test_y))
plt.figure(figsize=(12,5))
plt.plot(test_y[:N], label="Ground Truth")
plt.plot(pred[:N], label="Prediction")
plt.legend()
plt.grid(True)
plt.title("GT vs Pred (first 50)")
plt.savefig(f"{SAVE_DIR}/lineplot_gt_pred.png")
plt.close()

# Residual plot
plt.figure(figsize=(6,5))
plt.scatter(test_y, errors, alpha=0.6)
plt.axhline(0, linestyle='--')
plt.xlabel("Ground Truth")
plt.ylabel("Residual (Pred - GT)")
plt.grid(True)
plt.title("Residual Plot")
plt.savefig(f"{SAVE_DIR}/residual_plot.png")
plt.close()

print(f"All plots saved → {SAVE_DIR}/")
