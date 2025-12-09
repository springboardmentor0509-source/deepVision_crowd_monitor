"""
Run CSRNet Training and Evaluation

This script trains the CSRNet model and generates evaluation results.
Run from the project root: python run_csrnet.py
"""

import os
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Import the training module
from csrNet_Model.csrnet_model_training import train_csrnet

if __name__ == "__main__":
    print("=" * 60)
    print("CSRNet Training Pipeline")
    print("=" * 60)
    
    print("\n[1/1] Training CSRNet model...")
    train_csrnet()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {PROJECT_ROOT / 'results' / 'csrnet_cnn'}")
    print(f"Model saved to: {PROJECT_ROOT / 'models and code'}")
