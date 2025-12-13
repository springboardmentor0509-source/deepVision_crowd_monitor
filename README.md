# 🧠🔍 DeepVision Crowd Monitor  
### Real-Time Crowd Density Estimation & Overcrowding Detection System

DeepVision Crowd Monitor is an AI-powered computer vision system designed to analyze crowd density and detect overcrowding conditions in real time using surveillance imagery. By leveraging deep learning–based density estimation models, the system produces accurate crowd counts, visual heatmaps, and safety alerts to support crowd management and public safety.

This project is suitable for deployment in:
- Railway and metro stations  
- Airports and terminals  
- Public events and festivals  
- Religious gatherings  
- Stadiums  
- Smart city surveillance systems  

---

## 🚀 Project Capabilities

### 🔹 AI-Based Crowd Counting
- Deep learning models such as CSRNet, MobileNetCSRNet, and SimpleCNN  
- Pixel-wise density map estimation  
- Works with both static images and real-time video frames  

### 🔹 Overcrowding Detection & Safety Alerts
- Automatic detection of congestion  
- Configurable crowd thresholds  
- Four safety levels: Safe, Moderate, High Risk, Critical  
- Alert-ready design (Email/SMS integration supported)  

### 🔹 Interactive Monitoring Dashboard
- Built using Streamlit  
- Image upload and live monitoring  
- Heatmap overlays and density visualization  
- Inference history and model comparison  

### 🔹 Scalable & Deployment Ready
- Modular backend using FastAPI  
- GPU acceleration using CUDA  
- Docker-ready architecture  

---

## 🧱 System Architecture

### Processing Pipeline

Video / Image Input  
→ Frame Extraction  
→ Preprocessing (Resize, Normalize)  
→ Deep Learning Model  
→ Density Map Generation  
→ Crowd Count Estimation  
→ Overcrowding Detection  
→ Dashboard Visualization & Alerts  

---

## 🏗️ Technology Stack

### Deep Learning & ML
- PyTorch (CUDA-enabled)  
- CSRNet (VGG16-based)  
- MobileNetCSRNet  
- SimpleCNN  
- RandomForest (baseline)  

### Backend
- FastAPI  
- Uvicorn  

### Frontend & Visualization
- Streamlit  
- Matplotlib  
- Seaborn  
- Pandas  
- Pillow  

### Data Processing
- NumPy  
- SciPy  
- OpenCV  
- H5py  

---

## 📂 Dataset: ShanghaiTech Crowd Counting Dataset

- Total Images: 1,198  
- Total Annotations: 330,000+ people  

### Dataset Parts
- Part A: Highly dense crowd scenes (Internet images)  
- Part B: Medium-density street and campus scenes  

### Ground Truth
- Head annotations stored in .mat files  
- Density maps generated using Gaussian kernels  

---

## 🤖 Implemented Models

### CSRNet
- VGG16 frontend with dilated convolution backend  
- High accuracy in dense crowds  

### MobileNetCSRNet
- Lightweight and faster inference  
- Suitable for edge devices  

### SimpleCNN
- Custom lightweight architecture  
- Optimized for real-time performance  

### RandomForest
- Classical ML baseline  
- Useful for comparison and interpretability  

---

## 🧪 Running the Project

### Installation

```bash
git clone https://github.com/springboardmentor0509-source/deepVision_crowd_monitor.git
cd deepVision_crowd_monitor
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac
pip install -r requirements.txt
```

### Dataset Setup
Place the dataset in:
Dataset/ShanghaiTech/

### Start Backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### Launch Dashboard
```bash
cd frontend
streamlit run app.py
```

---

## 🛡️ Use Cases
- Crowd safety monitoring  
- Event and festival management  
- Transportation hub surveillance  
- Emergency evacuation planning  
- Smart city analytics  

---

## 🔮 Future Scope
- Live CCTV stream processing  
- Multi-camera crowd fusion  
- Predictive crowd analytics  
- Edge AI deployment  
- Cloud-based scalability  

---

## 📄 License
This project is intended for educational and research purposes.
