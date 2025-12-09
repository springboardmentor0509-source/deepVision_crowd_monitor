# evaluating_rf.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models and code"
SAVE_DIR = PROJECT_ROOT / "results" / "random_forest"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_random_forest():
    # Load model & test data
    rf = joblib.load(str(MODEL_DIR / "random_forest_model.pkl"))
    test_X, test_y, test_paths = joblib.load(str(MODEL_DIR / "rf_test_data.pkl"))

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

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Value": [mae, rmse, r2]
    })
    metrics_df.to_csv(str(SAVE_DIR / "random_forest_metrics.csv"), index=False)

    # Save predictions
    pred_df = pd.DataFrame({
        "GT": test_y,
        "Pred": pred
    })
    pred_df.to_csv(str(SAVE_DIR / "random_forest_predictions.csv"), index=False)

    print(f"Metrics saved → {SAVE_DIR / 'random_forest_metrics.csv'}")


    # ----- Plots -----

    # Scatter GT vs Pred
    plt.figure(figsize=(6,6))
    plt.scatter(test_y, pred, alpha=0.6)
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], '--')
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title("GT vs Prediction")
    plt.grid(True)
    plt.savefig(str(SAVE_DIR / "rf_scatter_gt_vs_pred.png"))
    plt.close()

    # Error histogram
    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=40)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")
    plt.grid(True)
    plt.savefig(str(SAVE_DIR / "rf_error_histogram.png"))
    plt.close()

    # Line plot for first 50
    N = min(50, len(test_y))
    plt.figure(figsize=(12,5))
    plt.plot(test_y[:N], label="Ground Truth")
    plt.plot(pred[:N], label="Prediction")
    plt.legend()
    plt.grid(True)
    plt.title("GT vs Pred (first 50)")
    plt.savefig(str(SAVE_DIR / "rf_lineplot_gt_pred.png"))
    plt.close()

    # Residual plot
    plt.figure(figsize=(6,5))
    plt.scatter(test_y, errors, alpha=0.6)
    plt.axhline(0, linestyle='--')
    plt.xlabel("Ground Truth")
    plt.ylabel("Residual (Pred - GT)")
    plt.grid(True)
    plt.title("Residual Plot")
    plt.savefig(str(SAVE_DIR / "rf_residual_plot.png"))
    plt.close()

    print(f"All plots saved → {SAVE_DIR}/")
