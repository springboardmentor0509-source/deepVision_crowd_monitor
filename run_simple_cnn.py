"""
Run SimpleCNN Training and Evaluation

This script trains the SimpleCNN model and generates evaluation results.
Run from the project root: python run_simple_cnn.py
"""

import os
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Import the training module
from simple_cnn.training import train_simplecnn

if __name__ == "__main__":
    print("=" * 60)
    print("SimpleCNN Training Pipeline")
    print("=" * 60)
    
    # Change imports in training.py to use absolute paths
    print("\n[1/1] Training SimpleCNN model...")
    train_simplecnn()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {PROJECT_ROOT / 'results' / 'simple_cnn'}")
    print(f"Model saved to: {PROJECT_ROOT / 'models and code'}")
