"""
Run Random Forest Training and Evaluation

This script trains the Random Forest model and generates evaluation results.
Run from the project root: python run_random_forest.py
"""

import os
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Import modules
from random_forest.preprocessing_rf import get_counts, extract_features
from random_forest.training_rf import train_random_forest
from random_forest.evaluating_rf import evaluate_random_forest

if __name__ == "__main__":
    print("=" * 60)
    print("Random Forest Training Pipeline")
    print("=" * 60)
    
    print("\n[1/2] Training Random Forest model...")
    train_random_forest()
    
    print("\n[2/2] Evaluating model and generating plots...")
    evaluate_random_forest()
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {PROJECT_ROOT / 'results' / 'random_forest'}")
    print(f"Model saved to: {PROJECT_ROOT / 'models and code'}")
