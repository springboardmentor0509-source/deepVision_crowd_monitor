# **📷🧠 DeepVision Crowd Monitor**

## AI for Real Time Crowd Density Estimation and Overcrowding Detection

---

## 📌 **Overview**

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

## **✨ Key Features**

### 1. Real-Time Crowd Detection & Analysis
Processes live video streams to detect and analyze crowd presence, density, and movement patterns with low latency.

### 2. Modular End-to-End Architecture
Designed with a modular pipeline that separates video ingestion, preprocessing, inference, analytics, and alerting for scalability and maintainability.

### 3. Scalable Video Processing Pipeline
Supports multiple camera feeds simultaneously with efficient frame sampling and resource-aware processing.

### 4. Crowd Density Estimation
Accurately estimates crowd density in defined regions of interest using computer vision and deep learning models.

### 5. Anomaly & Abnormal Behavior Detection
Identifies irregular crowd behavior such as sudden congestion, rapid dispersal, or unusual movement patterns.

### 6. Region of Interest (ROI) Based Monitoring
Allows configurable monitoring zones for targeted analysis of critical or sensitive areas.

### 7. Alert & Notification System
Triggers real-time alerts when predefined thresholds or abnormal conditions are detected.

### 8. Dataset Logging & Analytics Support
Stores processed frames, metadata, and analytics results for offline analysis, reporting, and model improvement.

### 9. Model-Agnostic Design
Easily integrates different object detection, tracking, or density estimation models without architectural changes.

### 10. Configurable & Extensible Framework
Supports easy configuration of thresholds, models, pipelines, and future feature extensions.

### 11. Visualization & Monitoring Support
Provides visual overlays such as bounding boxes, heatmaps, and crowd density indicators for better interpretability.

### 12. Research & Production Friendly
Designed to support both experimental research workflows and real-world deployment scenarios.

---

## **🎯 Use Cases**

### 1. Real-Time Crowd Density Monitoring
The system continuously analyzes live video streams to estimate crowd density in public spaces.  
This enables early detection of overcrowding and helps authorities take preventive actions to maintain safety.

### 2. Anomaly & Unusual Behavior Detection
By learning normal crowd movement patterns, the system identifies anomalies such as sudden dispersions, stampede-like motion, or restricted-area breaches.

### 3. Public Event Management
Event organizers can monitor crowd flow during concerts, festivals, and sports events to optimize entry/exit points and prevent congestion.

### 4. Smart Surveillance in Transportation Hubs
Airports, railway stations, and metro terminals can use the system to track crowd buildup, queue lengths, and movement trends for operational efficiency.

### 5. Emergency Response & Early Warning
The system can trigger real-time alerts during emergencies such as fire outbreaks, panic situations, or abnormal crowd accumulation, supporting rapid response.

### 6. Crowd Flow & Movement Analysis
Analyzes directional movement of people over time to identify bottlenecks and high-traffic zones, aiding infrastructure planning and optimization.

### 7. Restricted Zone Intrusion Detection
Detects unauthorized crowd presence in restricted or sensitive areas such as construction zones, secure facilities, or off-limit regions.

### 8. Urban Planning & Smart City Analytics
Aggregated crowd data can be used by urban planners to study pedestrian behavior and improve city layout, walkways, and public space utilization.

### 9. Retail & Commercial Space Analytics
Shopping malls and commercial complexes can leverage crowd insights to understand footfall trends and optimize store placement and layouts.

### 10. Dataset Generation for Research & Model Training
The system can generate structured datasets from real-world video feeds, supporting further research, benchmarking, and model improvement.

---

## 🏗️ **Project Workflow**

1. **Video Feed** → Source of crowd data (CCTV/live stream/video file).
2. **Frame Extraction** → Converts video into frames for model processing.
3. **Preprocessing** → Standardizes frames for robust model inference.
4. **Deep Learning Model** → Predicts density maps representing people distribution.
5. **Crowd Count Logic** → Integrates density map to estimate total crowd count.
6. **Overcrowding Detection** → Flags risk zones using predefined thresholds.
7. **Dashboard + Alerts** → Displays real-time analytics and safety warnings

---

## 🛠️ **Architecture Diagram**

**Project Architecture**

```bash
┌─────────────┐
│  Video Feed │  (CCTV / Video File / Live Stream)
└──────┬──────┘
       ↓
┌──────────────────┐
│ Frame Extraction │  (FPS control, frame sampling)
└──────┬───────────┘
       ↓
┌──────────────────┐
│ Preprocessing    │
│ • Resize         │
│ • Normalize      │
│ • Denoise        │
│ • ROI selection  │
└──────┬───────────┘
       ↓
┌──────────────────────────┐
│ Deep Learning Model      │
│ • CSRNet                 │
│ • MobileNetCSRNet        │
│ • SimpleCNN              │
└──────┬───────────────────┘
       ↓
┌──────────────────┐
│ Density Map      │
│ Generation       │
└──────┬───────────┘
       ↓
┌──────────────────┐
│ Crowd Count      │
│ Logic (Σ pixels) │
└──────┬───────────┘
       ↓
┌──────────────────────────┐
│ Overcrowding Detection   │
│ • Threshold comparison   │
│ • Temporal smoothing     │
└──────┬───────────────────┘
       ↓
┌──────────────────────────┐
│ Dashboard & Alerts       │
│ • Streamlit UI           │
│ • Heatmaps               │
│ • Live count             │
│ • Warning notifications  │
└──────────────────────────┘

```

---

##  📂 **Project File Structure**

```bash
deepVision_crowd_monitor/
├── backend/            # FastAPI application logic
├── frontend/           # Streamlit user interface (app.py)
├── models/             # PyTorch model weights (.pth)
├── preprocessing/      # Data handling scripts
├── results/            # Prediction outputs and EDA reports
│   ├── eda_results/
│   └── ... (model specific folders)
└── src/                # Shared source code / utils
```
---

## ⚙️ **Tech Stack**

### Deep Learning & Model

| Model Name                              | Description |
|-----------------------------------------|-------------|
|  **PyTorch**                            | Used to build and train deep neural networks efficiently on GPU.|
| **CSRNet**                              | VGG16-based encoder-decoder used for accurate crowd counting via density-map regression.|
| **MobileNetCSRNet**                     | Lightweight version used for faster crowd counting on low-resource devices.|
| **SimpleCNN**                           | Used as a baseline to compare performance with complex models.|
| **RandomForest**                        | Used to benchmark deep learning models against traditional ML. |

### Computer Vision & Processing

| Model Name                              | Description |
|-----------------------------------------|-------------|
| **NumPy & SciPy**                       | Used for numerical computation and scientific processing. |
| **OpenCV**                              | Used for image transformations and preprocessing. |
| **Pillow (PIL)**                        | Image loading and format handling |

### Visualization & Alerts

| Model Name                              | Description |
|-----------------------------------------|-------------|
| **Matplotlib** / **Plotly**             | Heatmaps and overlays  
| **Flask** / **Streamlit**               | Real-time web dashboard  
| **SMTP / Twilio API**                   | Alert system integration  
| **Pandas**                              | Structures data for tables, charts, and logs.
| **Heatmaps**                            | Density Maps which visually show crowd concentration. |

### Deployment & Integration

| Model Name                              | Description |
|-----------------------------------------|-------------|
| **Docker**                              | Containerization  
| **Nginx (optional)**                    | Reverse proxy for dashboard  
| **GPU Support (NVIDIA CUDA)**           | Optimized real-time performance  
| **FastAPI**                             | Deploys trained models as REST APIs for inference.
| **Uvicorn**                             | Runs the FastAPI application using an ASGI server.
| **PyTorch**                             | Loads trained models for production use.
| **Virtual Environment (venv / conda)**  | Isolates dependencies for stable deployment.

---

## 📊 **Dataset**

- **ShanghaiTech Crowd Counting Dataset**  
  This dataset contains total of 1198 images and density maps used for training and validation.
- **Total Images** : 1,198 images
- **Part A** : 482 images (300 train + 182 test)
- **Part B** : 716 images (400 train + 316 test)
---

## ▶️ **How to Run the Project**

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
# Backend runs on http://localhost:8000
```

**8. Launch Streamlit Dashboard(Frontend)**
```bash
streamlit run app.py
# Dashboard opens at http://localhost:8501
```

---
## 📧 Alerts Integration

Supports:
- Email alerts using SMTP  
- SMS alerts using Twilio API  

Triggered when crowd count crosses a predefined threshold.

---

## 🏁 **Project Milestones**

#### **Milestone 1: Setup and Data Preparation (Weeks 1–2)**

**Tasks**
- Setup Python environment (PyTorch, OpenCV, etc.)
- Download & preprocess ShanghaiTech dataset (resize, normalize)
- Implement data loading and visualization scripts

**Evaluation**
- Successful environment setup  
- Dataset loaded and visualized without errors  
- Documentation of setup and preprocessing steps  

---

#### **Milestone 2: Model Development and Training (Weeks 3–4)**

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

#### **Milestone 3: Real-time Integration and Core Functionality (Weeks 5–6)**

**Tasks**
- Integrate with OpenCV for live feed processing  
- Implement real-time crowd counting and overcrowding detection  
- Add basic alert mechanism for threshold breaches  

**Evaluation**
- Stable video input processing  
- Accurate real-time estimation  
- Functional alert trigger system  

---

#### **Milestone 4: Dashboard, Alerts, and Deployment (Weeks 7–8)**

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

## 🔮 **Future Enhancements**

### 1. Integration with Drone-Based Crowd Monitoring

  - Enable aerial video input from drones for wide-area and dynamic crowd surveillance.
  - Improve accuracy in open grounds, festivals, and disaster-response scenarios.

### 2. Multi-Camera Synchronization

  - Fuse data from multiple overlapping cameras to avoid double counting.
  - Perform cross-camera identity and density correlation for large venues.

### 3. Predictive Analytics for Crowd Flow Trends

  - Use time-series models to forecast crowd growth, movement, and congestion.
  - Enable early warnings before overcrowding actually occurs.

### 4. Real-Time Crowd Movement Tracking

  - Extend from static counting to directional flow analysis.
  - Identify entry/exit bottlenecks and evacuation risks.

### 5. Edge Deployment on Embedded Devices

  - Deploy lightweight models on edge devices (Jetson / edge GPUs).
  - Reduce latency and dependence on centralized servers.

### 6. Automated Alert Escalation System

  - Trigger tiered alerts (visual → audio → authority notification).
  - Integrate SMS/email alerts for emergency response teams.

### 7. Weather & Event-Aware Crowd Prediction

  - Combine crowd data with weather and event schedules.
  - Improve prediction accuracy during peak or abnormal conditions.

---

## 📜 License

This project is intended for **research and educational purposes** and is licensed under the **MIT License** &copy;.

Refer to the LICENSE file for usage terms.

---
