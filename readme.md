#  DeepVision Crowd Monitor

### AI for Density Estimation and Overcrowding Detection

---

##  Overview

**DeepVision Crowd Monitor** is a real-time deep learning-based system designed to estimate crowd density and detect overcrowded zones using surveillance video feeds.  
The system aims to enhance public safety, support emergency response, and optimize crowd management in high-footfall areas such as:

- Transit hubs  
- Public events  
- Religious gatherings  
- Smart city infrastructures 
- Stadiums
- Concerts 

By leveraging **Convolutional Neural Networks (CNNs)** and advanced **crowd estimation algorithms**, the system provides **timely insights** and **automated alerts** to authorities for proactive crowd control.

---

##  Project Workflow

1. **Video Feed** → Source of crowd data (CCTV/live stream/video file).
2. **Frame Extraction** → Converts video into frames for model processing.
3. **Preprocessing** → Standardizes frames for robust model inference.
4. **Deep Learning Model** → Predicts density maps representing people distribution.
5. **Crowd Count Logic** → Integrates density map to estimate total crowd count.
6. **Overcrowding Detection** → Flags risk zones using predefined thresholds.
7. **Dashboard + Alerts** → Displays real-time analytics and safety warnings

---

##  Architecture Diagram

Project Pipeline
```bash
┌──────────────┐
│ Streamlit UI │
└──────┬───────┘
       │ HTTP Request (frame)
       ↓
┌────────────────┐
│ FastAPI Server │
│ (Uvicorn)      │
└──────┬─────────┘
       ↓
┌────────────────┐
│ DL Model       │
│ (PyTorch)      │
└──────┬─────────┘
       ↓
┌────────────────┐
│ JSON Response  │
│ count + map    │
└──────┬─────────┘
       ↓
┌──────────────┐
│ Streamlit UI │
└──────────────┘
```

##  Tech Stack

### Deep Learning & Model
- **PyTorch** - Used to build and train deep neural networks efficiently on GPU.
- **CSRNet** - VGG16-based encoder-decoder used for accurate crowd counting via density-map regression.
- **MobileNetCSRNet** - Lightweight version used for faster crowd counting on low-resource devices.
- **SimpleCNN** - Used as a baseline to compare performance with complex models.
- **RandomForest** - Used to benchmark deep learning models against traditional ML. 

### Computer Vision & Processing
- **NumPy & SciPy** - Used for numerical computation and scientific processing.
- **OpenCV** - Used for image transformations and preprocessing.
- **Pillow (PIL)** - Image loading and format handling

### Visualization & Alerts
- **Matplotlib** / **Plotly** - heatmaps and overlays  
- **Flask** / **Streamlit** - real-time web dashboard  
- **SMTP / Twilio API** - alert system integration  
- **Pandas** -  Structures data for tables, charts, and logs.
- **Heatmaps** - Density Maps which visually show crowd concentration.

### Deployment & Integration
- **Docker** — containerization  
- **Nginx (optional)** — reverse proxy for dashboard  
- **GPU Support (NVIDIA CUDA)** — optimized real-time performance  
- **FastAPI** - Deploys trained models as REST APIs for inference.
- **Uvicorn** - Runs the FastAPI application using an ASGI server.
- **PyTorch** - Loads trained models for production use.
- **Virtual Environment(venv / conda)** - Isolates dependencies for stable deployment.

---

##  Dataset

- **ShanghaiTech Crowd Counting Dataset**  
  This dataset contains total of 1198 images and density maps used for training and validation.
- **Total Images** : 1,198 images
- **Part A** : 482 images (300 train + 182 test)
- **Part B** : 716 images (400 train + 316 test)
---

##  Project Milestones

### **Milestone 1: Setup and Data Preparation (Weeks 1–2)**

**Tasks**
- Setup Python environment (PyTorch, OpenCV, etc.)
- Download & preprocess ShanghaiTech dataset (resize, normalize)
- Implement data loading and visualization scripts

**Evaluation**
- Successful environment setup  
- Dataset loaded and visualized without errors  
- Documentation of setup and preprocessing steps  

---

### **Milestone 2: Model Development and Training (Weeks 3–4)**

**Tasks**
- Implement CSRNet/MCNN architecture in PyTorch  
- Train model on dataset subset  
- Generate and visualize initial density maps  

**Evaluation**
- Model implemented correctly  
- Reasonable loss convergence  
- Visual accuracy in density maps  
- Initial MAE performance metrics  

---

### **Milestone 3: Real-time Integration and Core Functionality (Weeks 5–6)**

**Tasks**
- Integrate with OpenCV for live feed processing  
- Implement real-time crowd counting and overcrowding detection  
- Add basic alert mechanism for threshold breaches  

**Evaluation**
- Stable video input processing  
- Accurate real-time estimation  
- Functional alert trigger system  

---

### **Milestone 4: Dashboard, Alerts, and Deployment (Weeks 7–8)**

**Tasks**
- Build web dashboard (Flask/Streamlit) showing live density maps and alerts  
- Enhance alerts via SMTP/Twilio API  
- Dockerize entire system for deployment  
- Optimize for GPU performance  

**Evaluation**
- Responsive, user-friendly dashboard  
- Alerts sent successfully  
- Containerized and deployable system  
- Efficient GPU-enabled performance  
- Complete documentation and deployment guide  

---

## How to Run the Project

**1. Clone the Repository**
```bash
git clone https://github.com/springboardmentor0509-source/deepVision_crowd_monitor.git
cd deepVision_crowd_monitor
```

**2. Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venvdvc
# Activate virtual environment
# Windows (Command Prompt / PowerShell)
venvdvc\Scripts\activate
# Linux / macOS
source venvdvc/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```
```bash
Or install these following:
yt-dlp
opencv-python
numpy
matplotlib
pillow
torch
torchvision

#req for eda_onDataset.py
#numpy
pandas
#matplotlib
seaborn
#opencv-python
#Pillow
scipy
tqdm
#torch
#torchvision
torchaudio

# Backend & Frontend requirements
fastapi
uvicorn
python-multipart
streamlit
scikit-learn
joblib
scikit-image
```

**4. Download Dataset**
```bash
Download the ShanghaiTech dataset and place it in Dataset/ShanghaiTech/
```

**5. Preprocess Data (Optional - if training from scratch)**
```bash
python preprocessing/run_preprocess.py
```

**6. Train Models (Optional - pre-trained models available)**
```bash
# CSRNet
python run_csrnet.py

# MobileNetCSRNet
python run_mobile_csrnet.py

# SimpleCNN
python run_simple_cnn.py

# RandomForest
python run_random_forest.py
```

**7. Start Backend**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#Backend runs on http://localhost:8000
```

**8. Launch Streamlit Dashboard(Frontend)**
```bash
streamlit run app.py
# Dashboard opens at http://localhost:8501
```

---

##  Future Enhancements

- 1. Integration with Drone-Based Crowd Monitoring

    Enable aerial video input from drones for wide-area and dynamic crowd surveillance.

    Improve accuracy in open grounds, festivals, and disaster-response scenarios.

2. Multi-Camera Synchronization

    Fuse data from multiple overlapping cameras to avoid double counting.

    Perform cross-camera identity and density correlation for large venues.

3. Predictive Analytics for Crowd Flow Trends

    Use time-series models to forecast crowd growth, movement, and congestion.

    Enable early warnings before overcrowding actually occurs.

4. Real-Time Crowd Movement Tracking

    Extend from static counting to directional flow analysis.

    Identify entry/exit bottlenecks and evacuation risks.

5. Edge Deployment on Embedded Devices

    Deploy lightweight models on edge devices (Jetson / edge GPUs).

    Reduce latency and dependence on centralized servers.

6. Automated Alert Escalation System

    Trigger tiered alerts (visual → audio → authority notification).

    Integrate SMS/email alerts for emergency response teams.

7. Weather & Event-Aware Crowd Prediction

    Combine crowd data with weather and event schedules.

    Improve prediction accuracy during peak or abnormal conditions.

---

##  License

This project is intended for **research and educational purposes**.  
Refer to the LICENSE file (if applicable) for usage terms.

---
