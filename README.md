# 🧠🔍 DeepVision Crowd Monitor

### AI System for Real-Time Crowd Density Estimation, Overcrowding Detection & Visual Analytics

DeepVision Crowd Monitor is an **end-to-end AI platform** designed to estimate crowd density, detect overcrowded regions, and visualize density maps using **deep learning models and video analysis**.

Built for real-world **public safety and smart surveillance** applications such as:

- 🚉 Railway & Metro Stations  
- ✈️ Airports  
- 🕌 Religious Gatherings  
- 🎉 Festivals & Public Events  
- 🏟 Stadiums  
- 🏙 Smart City Surveillance  

The system combines **deep learning, computer vision, statistical analysis, and an interactive dashboard** to enable intelligent crowd monitoring.

---

## 🚀 Key Features

### 🔹 Real-Time Processing
- Crowd density estimation on image/video frames  
- Fast inference using **FastAPI backend**  
- Live visualization using **Streamlit dashboard**

---

### 🔹 Multiple ML/DL Models Supported

| Model Name       | Description |
|------------------|------------|
| **CSRNet**       | High-accuracy crowd counting using dilated CNN |
| **MobileCSRNet** | Lightweight and fast model optimized for real-time |
| **SimpleCNN**    | Baseline CNN model for demonstration |
| **Random Forest**| Classical ML baseline for comparison |

---

### 🔹 Interactive Dashboard (Streamlit)

Includes:
- 📊 **EDA Viewer**
- 🧪 **Model Evaluation Viewer**
- 🖼 **Prediction Samples**
- 🎛 **Live Demo Tab**
- 📚 **About Page**

---

### 🔹 Automated EDA (Exploratory Data Analysis)
- Distribution plots  
- Heatmaps  
- Correlation matrices  
- Summary statistics  
- Auto-generated CSV reports  

---

### 🔹 Model Evaluation Tools
- MAE, MSE, RMSE metrics  
- Training & validation curves  
- Per-model prediction samples  
- CSV-based evaluation reports  

---

## 🧱 Architecture Overview

**Pipeline:**

Video Feed → Frame Extraction → Preprocessing → Deep Learning Model  
Crowd Count Logic → Overcrowding Detection → Dashboard + Alerts

---

## 🧬 Tech Stack

### 🔹 Deep Learning & Machine Learning
- **PyTorch** – Deep learning framework  
- **CSRNet** – Crowd density estimation model  
- **MobileCSRNet** – Lightweight real-time crowd estimation variant  
- **SimpleCNN** – Baseline convolutional neural network  
- **Random Forest** – Classical machine learning baseline  

### 🔹 Computer Vision & Data Processing
- **OpenCV** – Video frame extraction & image processing  
- **NumPy** – Numerical computations  
- **Pandas** – Data analysis & evaluation  

### 🔹 Backend
- **FastAPI** – High-performance inference API  
- **Uvicorn** – ASGI server for FastAPI  

### 🔹 Frontend & Visualization
- **Streamlit** – Interactive web dashboard  
- **Matplotlib / Seaborn** – Data visualization & plots  

### 🔹 Tools & Utilities
- **Python 3.9+** – Core programming language  
- **Git & GitHub** – Version control & collaboration  
- **CUDA (optional)** – GPU acceleration for deep learning  


---

## 📂 Dataset

**ShanghaiTech Crowd Counting Dataset**  
- High-density crowd images  
- Ground-truth density maps  
- Benchmark dataset for CSRNet  
 
---

## ⚙️ Installation & Setup

### **1. Clone the Repository**
```
git clone https://github.com/springboardmentor0509-source/deepVision_crowd_monitor.git
cd deepVision_crowd_monitor
```

### **2. Create virtual environment**
```
python -m venv venv
venv\Scripts\activate
```

### **3. Install Dependencies**
```
pip install -r requirements.txt
```

## 🖥️ Running the Application
## 🔹 Start FastAPI Backend

```bash
cd backend
uvicorn main:app --reload

Backend runs at:  
👉 [http://localhost:8000]

API docs:  
👉 [http://localhost:8000/docs]

```

### ** Launch Dashboard**
```
streamlit run app.py
```

## 📧 Alerts Integration

Supports:
- Email alerts using SMTP  
- SMS alerts using Twilio API  

Triggered when crowd count crosses a predefined threshold.

---

## 🎯 Use Cases

- Smart city surveillance  
- Public safety monitoring  
- Stadium & event crowd control  
- Metro & railway station monitoring  
- Emergency response systems  

## 🔮 Future Enhancements

- Multi-camera fusion  
- Edge deployment (Jetson Nano)  
- ONNX / TensorRT optimization  
- Predictive crowd analytics  
- Automated SMS / Email alerting  

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues, submit pull requests, or suggest improvements.

## 📜 License

This project is licensed under the **MIT License**.
