# 🧠🔍 DeepVision Crowd Monitor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)

**DeepVision Crowd Monitor** is a state-of-the-art AI system designed to estimate crowd density and detect potential overcrowding in real-time. By leveraging deep learning models like **CSRNet**, it provides actionable insights for public safety and event management.

---

## 🚀 Key Features

*   **Real-Time Estimation**: Instant crowd counting from image uploads or live camera feeds.
*   **Visual Analytics**: 
    *   **Density Maps**: Understand exactly where the crowd is concentrated.
    *   **Heatmap Overlays**: Intuitive color-coded visualization on top of the original image.
*   **Multi-Model Architecture**: Switch between high-accuracy models (CSRNet) and lightweight ones (SimpleCNN, RandomForest) on the fly.
*   **Interactive Dashboard**: A modern, user-friendly interface built with **Streamlit** for easy interaction and comparison.
*   **Model Comparison**: Side-by-side analysis of how different models interpret the same scene.

---

## 🧱 How It Works

1.  **Input**: The system accepts an image (via upload or camera).
2.  **Processing**: The backend (FastAPI) routes the image to the selected PyTorch model.
3.  **Inference**: 
    *   The model generates a density map (predicted count per pixel).
    *   Total crowd count is derived by summing the density map.
4.  **Visualization**: The density map is colorized into a heatmap and overlaid on the original image for the user.

---

## 🏗️ Tech Stack

*   **Frontend**: Streamlit (Python)
*   **Backend**: FastAPI
*   **Deep Learning**: PyTorch, TorchLegacy
*   **Computer Vision**: OpenCV, Pillow, NumPy
*   **Deployment**: Docker

---

## 📂 Project Structure

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

## 🧪 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/springboardmentor0509-source/deepVision_crowd_monitor.git
cd deepVision_crowd_monitor
pip install -r requirements.txt
```

### 3. Run the Application
You need to run both the backend and frontend. It's recommended to use two terminal windows.

**Terminal 1: Backend**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2: Frontend**
```bash
streamlit run frontend/app.py
```

Visit `http://localhost:8501` in your browser to start using DeepVision!

---

## 📦 Docker Support

Build and run the entire stack in a container:

```bash
docker build -t deepvision .
docker run -p 8501:8501 -p 8000:8000 deepvision
```

---

## 🔮 Future Roadmap

- [ ] **Video Stream Support**: Real-time RTSP/CCTV feed processing.
- [ ] **Alert System**: SMS/Email notifications when density exceeds a threshold.
- [ ] **Edge Deployment**: Optimization for Jetson Nano / Raspberry Pi.

---
*Created for the DeepVision Project.*

