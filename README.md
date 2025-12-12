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
           │  Frame Extraction   │
           └────────────┬───────┘
                      ↓
           ┌────────────────────┐
           │   Pre-processing   │
           │ (Resize, Normalize)│
           └────────────┬───────┘
                      ↓
         ┌───────────────────────────┐
         │    Deep Learning Model    │
         │  CSRNet / MCNN / SimpleCNN│
         └──────────────┬────────────┘
                       ↓
       ┌─────────────────────────────────┐
       │ Density Map + Crowd Count Logic │
       └──────────────────┬──────────────┘
                         ↓
    ┌──────────────────────────────┐
    │  Overcrowding Detection      │
    │ (Dynamic threshold alerts)   │
    └───────────────┬─────────────┘
                    ↓
     ┌────────────────────────────────┐
     │  Streamlit Dashboard + Alerts  │
     └────────────────────────────────┘

---

## 🏗️ Tech Stack

### **Deep Learning**
- CSRNet or MCNN  
- PyTorch  

### **Computer Vision**
- OpenCV  
- NumPy  
- Pillow  

### **Visualization & Alerts**
- Matplotlib / Plotly  
- Flask or Streamlit  
- SMTP / Twilio API  

### **Deployment**
- Docker  
- Nginx (optional)  
- GPU acceleration (CUDA)  

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

## 🧪 How to Run the Project

### **1. Clone the Repository**
```
git clone https://github.com/sehaj_kaur/DeepVision-Crowd-Monitor.git
cd DeepVision-Crowd-Monitor
```

### **2. Install Dependencies**
```
pip install -r requirements.txt
```


### **3.Setup Dataset**
```
dataset/
 └── ShanghaiTech/
      ├── part_A/
      └── part_B/
```


### **4. Run Backend**
```
uvicorn backend.main:app --reload --port 8000
```

### **5. Launch Dashboard**
```
streamlit run app.py
```
---

## 🖥️ Dashboard Preview
Features:

Upload images → get instant crowd estimate

Heatmap & density map visualization

Inference history

Model comparison

Clean UI/UX with modern design

## 🛡️ Use Cases

- Crowd safety monitoring  
- Smart city surveillance  
- Event management  
- Railway/Metro stations  
- Emergency evacuation assistance  

---

## 🔮 Future Enhancements

- Multi-camera fusion  
- Predictive crowd analytics  
- IoT/Edge deployment  
- Model compression  

---

