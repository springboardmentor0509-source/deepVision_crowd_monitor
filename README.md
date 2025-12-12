🧠🔍 DeepVision Crowd Monitor
AI System for Real-Time Crowd Density Estimation, Overcrowding Detection & Visual Analytics

DeepVision Crowd Monitor is an end-to-end AI platform designed to estimate crowd density, detect overcrowded regions, and visualize density maps using deep learning models and live video analysis.

Built for real-world safety applications such as:

🚉 Railway & Metro Stations

✈️ Airports

🕌 Religious Gatherings

🎉 Festivals & Public Events

🏟 Stadiums

🏙 Smart City Surveillance

The system uses deep learning, computer vision, statistical analysis, and an interactive dashboard to enable intelligent crowd monitoring.

🚀 Key Features
🔹 Real-Time Processing

Crowd density estimation on image/video frames

Fast inference using FastAPI backend

Live visualization in Streamlit dashboard

🔹 Multiple ML/DL Models Supported
Model	Description
CSRNet	High-accuracy crowd counting using dilated CNN
MobileCSRNet	Faster, lightweight variant optimized for realtime
SimpleCNN	Beginner-friendly baseline CNN
Random Forest	Classical ML model used as non-DL baseline
🔹 Interactive Dashboard (Streamlit)

📊 EDA Viewer

🧪 Model Evaluation Viewer

🖼 Prediction Samples

🎛 Live Demo

📚 About Page

🔹 Automated EDA

Histograms

Heatmaps

Summary stats

Auto-generated visualization outputs

🔹 Model Evaluation

Loss curves

Validation predictions

MAE/MSE/RMSE statistics

CSV-based metric outputs

## 🧱 Architecture Overview

**Pipeline:**

Video Feed → Frame Extraction → Preprocessing → Deep Learning Model  
Crowd Count Logic → Overcrowding Detection → Dashboard + Alerts

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

## 📂 Dataset

**ShanghaiTech Crowd Counting Dataset**  
- High-density crowd images  
- Ground-truth density maps  
- Benchmark dataset for CSRNet  

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
git clone https://github.com/your-username/AI-DeepVision.git
cd AI-DeepVision
```

### **2. Install Dependencies**
```
pip install -r requirements.txt
```

### **3. Run Real-Time Monitoring**
```
python run_realtime.py
```

### **4. Launch Dashboard**
```
streamlit run app.py
```

---

## 📦 Docker Deployment

```
docker build -t deepvision .
docker run -p 8080:8080 deepvision
```

---

## 📧 Alerts Integration

Supports:
- Email alerts using SMTP  
- SMS alerts using Twilio API  

Triggered when crowd count crosses a predefined threshold.

---

## 📸 Suggested Output Screenshots  
(Add in repo)  
- Density map  
- Heatmap overlay  
- Dashboard view  
- Alert screenshot  

---

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

