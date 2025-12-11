# Training and Evaluation Guide

This guide explains how to run training for all models in the DeepVision Crowd Monitor project.

## Prerequisites

1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Ensure Dataset is available:**
   - Dataset should be in: `Dataset/ShanghaiTech/`
   - Structure:
     ```
     Dataset/ShanghaiTech/
       part_A/
         train_data/
           images/
           ground-truth/
         test_data/
           images/
           ground-truth/
       part_B/
         ...
     ```

3. **⚠️ IMPORTANT: Run Preprocessing First**
   ```powershell
   cd preprocessing
   python run_preprocess.py
   cd ..
   ```
   
   This creates optimized preprocessed data in `processed_data/` folder with:
   - Pre-computed geometry-adaptive density maps
   - Resized images (1024px)
   - Metadata CSV files
   
   **Why?** Training will be **2-3x faster** and use **better quality density maps**!

## Running Training Scripts

All scripts should be run from the **project root directory** (`deepVision_crowd_monitor/`).

### 1. Train SimpleCNN Model

```powershell
python run_simple_cnn.py
```

**Output:**
- Model saved to: `models and code/best_simplecnn.pth`
- Results saved to: `results/simple_cnn/`
  - `cnn_training_metrics.csv`
  - `cnn_predictions.csv`
  - `cnn_training_plot.png`

**Then run evaluation plots:**
```powershell
python -c "import sys; sys.path.insert(0, 'src'); from simple_cnn.evaluating import *"
```

---

### 2. Train CSRNet Model

```powershell
python run_csrnet.py
```

**Output:**
- Model saved to: `models and code/best_csrnet_model.pth`
- Results saved to: `results/csrnet_cnn/`
  - `csr_training_metrics.csv`
  - `csrnet_training_plot.png`

---

### 3. Train Mobile CSRNet Model

```powershell
python run_mobile_csrnet.py
```

**Output:**
- Model saved to: `models and code/best_mobilenet_csrnet_model.pth`
- Results saved to: `results/mobile_csrnet/`
  - `mobile_csr_training_metrics.csv`
  - `mobile_csr_training_plot.png`

---

### 4. Train Random Forest Model

```powershell
python run_random_forest.py
```

**Output:**
- Model saved to: `models and code/random_forest_model.pkl`
- Results saved to: `results/random_forest/`
  - `random_forest_metrics.csv`
  - `random_forest_predictions.csv`
  - Various plots (scatter, histogram, residual, etc.)

---

## Training Parameters

### SimpleCNN
- Learning Rate: 1e-4
- Batch Size: 4
- Epochs: 20
- Dataset: ShanghaiTech Part A

### CSRNet
- Learning Rate: 1e-5
- Batch Size: 4
- Epochs: 20
- Dataset: ShanghaiTech Part A
- Pretrained: VGG16 (frontend)

### Mobile CSRNet
- Learning Rate: 1e-5
- Batch Size: 4
- Epochs: 30
- Dataset: ShanghaiTech Part A
- Pretrained: MobileNetV2 (frontend)

### Random Forest
- n_estimators: 200
- max_depth: 20
- Dataset: ShanghaiTech Part A

---

## Viewing Results

After training, use the Streamlit dashboard to view all results:

```powershell
streamlit run output.py
```

Navigate to:
- **Data Visualization** - View training plots
- **Model Evaluation Results** - View metrics CSVs
- **Live Demo** - Test models with custom images (requires backend running)

---

## Running the Backend API

To enable live inference in the dashboard:

```powershell
python start_backend.py
```

This starts the FastAPI backend on `http://localhost:8000`.

---

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in training scripts
- Use CPU by setting `DEVICE = "cpu"`

### Import Errors
- Ensure you're running from project root
- Check that `src/` folder is in the correct location
- Verify all dependencies are installed

### Dataset Not Found
- Check dataset path is correct: `Dataset/ShanghaiTech/`
- Verify folder structure matches expected format

---

## File Structure After Training

```
deepVision_crowd_monitor/
├── models and code/
│   ├── best_simplecnn.pth
│   ├── best_csrnet_model.pth
│   ├── best_mobilenet_csrnet_model.pth
│   └── random_forest_model.pkl
├── results/
│   ├── simple_cnn/
│   │   ├── cnn_training_metrics.csv
│   │   ├── cnn_predictions.csv
│   │   └── *.png
│   ├── csrnet_cnn/
│   │   ├── csr_training_metrics.csv
│   │   └── *.png
│   ├── mobile_csrnet/
│   │   ├── mobile_csr_training_metrics.csv
│   │   └── *.png
│   └── random_forest/
│       ├── random_forest_metrics.csv
│       ├── random_forest_predictions.csv
│       └── *.png
```

---

## Notes

- All models train on **ShanghaiTech Part A** by default
- Training times vary: SimpleCNN (~30min), CSRNet (~1-2 hours), Mobile CSRNet (~2-3 hours)
- GPU recommended for deep learning models
- Results are automatically saved to `results/` folder
- Models are automatically saved to `models and code/` folder
