# 🧠🔍 DeepVision Crowd Monitor  
### **AI for Real-Time Crowd Density Estimation & Overcrowding Detection**

DeepVision Crowd Monitor is an AI-powered system designed to estimate crowd density and detect overcrowded zones **in real time** using live surveillance video feeds.  
This project enhances **public safety**, supports **emergency response**, and enables **smart crowd management** in high-footfall environments such as:

- Railway & metro stations  
- Airports  
- Public events and festivals  
- Religious gatherings  
- Stadiums  
- Smart city surveillance systems  

Using deep learning (CSRNet/MCNN) and computer vision, the system generates accurate density maps and triggers alerts when crowd limits are exceeded.

---

## 🚀 Features

### ✅ **AI-Powered Crowd Counting**
- CSRNet / MCNN deep learning models  
- High-precision density map estimation  
- Works on images + real-time video feed  

### ✅ **Overcrowding Detection**
- Automatically detects congestion  
- Triggers alerts based on dynamic thresholds  
- Supports email & SMS alerts (SMTP + Twilio)  

### ✅ **Real-Time Monitoring Dashboard**
- Built with Streamlit  
- Live camera integration  
- Heatmap overlays  
- Model testing interface  
- Inference history tracking  

### ✅ **Deployment Ready**
- Docker support  
- GPU acceleration via CUDA  
- Modular backend (FastAPI)  
- Production-safe project structure  

---


## 🧱 Architecture Overview

**Pipeline:**

Video Feed → Frame Extraction → Preprocessing → Deep Learning Model  
Crowd Count Logic → Overcrowding Detection → Dashboard + Alerts

        ┌────────────────────────┐
        │   Live CCTV / Video    │
        └─────────────┬──────────┘
                      ↓
           ┌────────────────────┐
           │  Frame Extraction  │
           └────────────┬───────┘
                      ↓
           ┌────────────────────┐
           │   Pre-processing   │
           │ (Resize, Normalize)│
           └────────────┬───────┘
                      ↓
         ┌──────────────────────────────────────────┐
         │    Deep Learning Model                   │ 
         │  CSRNet / MCNN / SimpleCNN /RandomForest │
         └──────────────┬───────────────────────────┘  
                       ↓
       ┌─────────────────────────────────┐
       │ Density Map + Crowd Count Logic │
       └──────────────────┬──────────────┘
                         ↓
    ┌──────────────────────────────┐
    │  Overcrowding Detection      │
    │ (Dynamic threshold alerts)   │
    └───────────────┬──────────────┘
                    ↓
     ┌────────────────────────────────┐
     │  Streamlit Dashboard + Alerts  │
     └────────────────────────────────┘

---

## 🏗️ Tech Stack

### **Deep Learning & ML**
- PyTorch 2.5.1 with CUDA 12.1
- CSRNet (VGG16-based encoder-decoder)
- MobileNetCSRNet (Lightweight variant)
- SimpleCNN (Custom architecture)
- RandomForest (Classical ML baseline)

### **Backend & API**
- FastAPI (Model serving)
- Uvicorn (ASGI server)

### **Frontend & Visualization**
- Streamlit (Interactive dashboard)
- Matplotlib & Seaborn (Plotting)
- Pandas (Data handling)
- Pillow (Image processing)

### **Data Processing**
- NumPy & SciPy
- OpenCV (Image operations)
- H5py (Dataset storage)

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 12.1 (for GPU support)
- 8GB+ RAM
- 50GB disk space

### Step 1: Clone Repository
```bash
git clone https://github.com/springboardmentor0509-source/deepVision_crowd_monitor.git
cd deepVision_crowd_monitor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset
Place ShanghaiTech dataset in:
```
dataset/ShanghaiTech/
├── part_A/
│   ├── train_data/
│   │   ├── images/
│   │   └── ground-truth/
│   └── test_data/
│       ├── images/
│       └── ground-truth/
└── part_B/
    ├── train_data/
    └── test_data/
```

### Step 4: Configure Paths
Update `preprocessing/config.py` with your dataset path:
```python
DATASET_ROOT = r"c:\path\to\dataset\ShanghaiTech"
OUT_ROOT = r"c:\path\to\processed_data"
```

### Step 5: Preprocess Data
```bash
python preprocessing/run_preprocess.py
```

### Step 6: Train Models (Optional)
```bash
# Train CSRNet
python src/csrNet_Model/csrnet_model_training.py

# Train MobileNetCSRNet
python src/mobile_csrnet/mobile_csrnet_training.py

# Train SimpleCNN
python src/simple_cnn/training.py

# Train RandomForest
python src/random_forest/training_rf.py
```

### Step 7: Run Application
```bash
# Terminal 1: Start Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2: Start Frontend
cd frontend
streamlit run app.py
```

Access the dashboard at: **http://localhost:8501**

---

# 📂 Dataset — ShanghaiTech Crowd Counting

The project uses the **ShanghaiTech Dataset**, a benchmark dataset used for density estimation research.

### **Part A**
- Highly dense crowds  
- 482 images (300 train / 182 test)  
- Average ~501 people/image  

### **Part B**
- Low-density, campus-like  
- 716 images (400 train / 316 test)  
- Average ~123 people/image  

Ground-truth consists of head annotations `(x, y)` → converted into Gaussian density maps.

---

## 🗂️ Project Milestones

### **Milestone 1: Setup & Data Preparation (Weeks 1–2)**  
- Install dependencies (PyTorch, OpenCV, etc.)  
- Download and preprocess dataset  
- Build data loader + visualization modules  
- Validate environment setup  

### **Milestone 2: Model Training (Weeks 3–4)**  
- Implement CSRNet/MCNN  
- Train model with dataset  
- Generate density maps  
- Validate using MAE  

### **Milestone 3: Real-Time Integration (Weeks 5–6)**  
- Connect OpenCV to live camera feed  
- Real-time crowd counting  
- Overcrowding detection  
- Trigger alerts  

### **Milestone 4: Dashboard & Deployment (Weeks 7–8)**  
- Real-time dashboard (Flask/Streamlit)  
- Email/SMS alerts (SMTP/Twilio)  
- Docker containerization  
- GPU optimization  
- Deployment documentation  

---

## 🤖 Implemented Models

### **1. CSRNet (Congested Scene Recognition Network)**
- **Architecture**: VGG16 frontend + dilated convolution backend
- **Parameters**: ~16M
- **Performance**: MAE: 109.41, RMSE: 149.92
- **Best Use**: High-accuracy scenarios, acceptable inference time

### **2. MobileNetCSRNet**
- **Architecture**: MobileNetV2 frontend + CSRNet backend
- **Parameters**: ~3M (5x smaller than CSRNet)
- **Performance**: Balanced accuracy and speed
- **Best Use**: Resource-constrained environments, mobile deployment

### **3. SimpleCNN**
- **Architecture**: Custom lightweight encoder-decoder
- **Parameters**: ~2M
- **Performance**: Fast inference with reasonable accuracy
- **Best Use**: Real-time applications, edge devices

### **4. RandomForest Baseline**
- **Type**: Classical ML approach
- **Features**: Hand-crafted image features
- **Best Use**: Baseline comparison, interpretable predictions

---

## 📊 Model Performance Comparison

| Model | MAE ↓ | RMSE ↓ | Parameters | Inference Time | Best For |
|-------|-------|--------|------------|----------------|----------|
| CSRNet | 109.41 | 149.92 | ~16M | ~50ms | High accuracy |
| MobileNetCSRNet | ~120 | ~165 | ~3M | ~30ms | Mobile/Edge devices |
| SimpleCNN | ~135 | ~180 | ~2M | ~20ms | Real-time apps |
| RandomForest | ~150 | ~200 | N/A | ~10ms | Baseline/Interpretable |

---

## 💻 Usage Examples

### Example 1: Predict Single Image via API
```bash
# Using curl
curl -X POST "http://localhost:8000/predict/CSRNet" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### Example 2: Python Script for Batch Prediction
```python
import requests
from pathlib import Path

# Predict multiple images
image_folder = Path("test_images")
backend_url = "http://localhost:8000"

for img_path in image_folder.glob("*.jpg"):
    with open(img_path, 'rb') as f:
        files = {'file': (img_path.name, f, 'image/jpeg')}
        response = requests.post(
            f"{backend_url}/predict/CSRNet",
            files=files
        )
        result = response.json()
        print(f"{img_path.name}: {result['predicted_count']:.1f} people")
```

### Example 3: Load Trained Model in Python
```python
import torch
from models.csrnet import CSRNet
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = CSRNet()
model.load_state_dict(torch.load('models/csrnet_cnn/best_csrnet_model.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

img = Image.open('test_image.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    density_map = model(img_tensor)
    count = density_map.sum().item()
    print(f"Predicted count: {count:.1f}")
```

### Example 4: Real-Time Video Processing
```python
import cv2
import torch
from models.csrnet import CSRNet

# Load model
model = CSRNet()
model.load_state_dict(torch.load('models/csrnet_cnn/best_csrnet_model.pth'))
model.eval()

# Open video stream
cap = cv2.VideoCapture(0)  # Use 0 for webcam or video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess and predict
    # (Add preprocessing code here)
    count = 0  # Replace with actual prediction
    
    # Display result
    cv2.putText(frame, f'Count: {count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Crowd Monitor', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🎯 API Endpoints

### FastAPI Backend (Port 8000)

**GET `/`**
- Health check endpoint

**GET `/models`**
- List all available trained models

**POST `/predict`**
- Upload image for crowd count prediction
- Returns: density map, crowd count, inference time

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "model=csrnet"
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Solution: Reduce batch size in config
BATCH_SIZE = 1  # in training scripts
```

**2. Dataset not found**
```bash
# Update path in preprocessing/config.py
DATASET_ROOT = Path("your/path/to/dataset")
```

**3. Module not found errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**4. Streamlit port already in use**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---


---



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- ShanghaiTech Dataset creators
- PyTorch community
- Streamlit team
- All open-source contributors

---



## ⭐ Star History

If this project helped you, please give it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=springboardmentor0509-source/deepVision_crowd_monitor&type=Date)](https://star-history.com/#springboardmentor0509-source/deepVision_crowd_monitor&Date)

---

**Made with ❤️ for Public Safety & Smart Cities**

